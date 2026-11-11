"""A group run killed in finalisation finishes when started again (Bitcrush ISSUE-1690).

A group run backs up after every epoch, and started again with the same folders
it resumes from its newest backup. After the last epoch it finalises: SWA and
the greedy soup may rewrite ``candidate.pt``, the candidate is judged against
the current model, the winner is put in place, and priors, calibration and the
strictness data are written into it. Only then are the backups deleted. A kill
in that stretch used to leave backups pointing at a candidate already moved
over the current model, and a "current model" that was really the half-finished
candidate: the next start could not finish (on the Azure VM, a group that fails
three times and loses a run of hours).

Each test kills a run at one point of its finalisation, starts it again with the
same folders, and holds it to the outcome of the same run left alone: nothing
trained the second time, the same promotion decision and reason (judged against
the ORIGINAL current model), the same result, the same final model file and the
same files left behind. A kill is a ``BaseException`` raised at the point: no
``except Exception`` handler sees it, so nothing tidies up or backs up, as with
SIGKILL. The pins at the top hold a run that is left alone to what it has always
done.

The head-only task shares the finalisation but has no backups: started again it
probes afresh (seconds on cached features), and must land where the run left
alone did, judged against and warm-started from the original model.

CPU-only, on the micro stand-in trunk (``group_stub``); each run takes seconds.
"""

from __future__ import annotations

import json
import math
import random
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

import bittrainer.finalize as finalize
import bittrainer.generic.tasks.group_task as group_task
import bittrainer.group_trainer as gt
from bittrainer.generic.generic_trainer import GenericTrainer
from bittrainer.generic.tasks.group_task import GroupTask
from bittrainer.generic.tasks.head_only_task import HeadOnlyTask
from bittrainer.promotion import PromotionReason
from bittrainer.tests.group_stub import (
    CLASSES,
    group_config,
    make_group,
    micro_convnext,
    use_micro_trunk,
)
from bittrainer.training_state import BackupCoordinator, TrainingStateManager

# Every key a single-label group result carries today, sample stats on.
_RESULT_KEYS = {
    "epochs_completed", "best_epoch", "best_val_macro_f1", "validation_metric",
    "selected_validation_score", "final_val_macro_f1", "final_val_weighted_f1",
    "final_val_micro_f1", "selection_metric", "final_val_macro_f1_supported",
    "final_val_macro_f1_excl_none", "final_val_macro_f1_supported_excl_none",
    "final_val_macro_precision", "final_val_macro_recall", "final_val_macro_f1_original",
    "final_val_macro_f1_normalized", "final_val_macro_f1_dual", "final_val_loss",
    "per_class_f1", "per_class_precision", "per_class_recall", "per_class_support",
    "checkpoint_path", "class_counts", "val_class_counts", "total_images",
    "promotion_reason", "selected_softness_kind", "selected_softness_value",
    "soft_label_tuning_metric", "soft_label_tuning_results", "soft_label_tuning_elapsed_ms",
    "selected_oversample_none", "oversample_tuning_metric", "oversample_tuning_results",
    "oversample_tuning_elapsed_ms", "data_quality_warnings", "final_val_none_precision",
    "final_val_none_recall", "final_val_none_f1", "final_val_none_false_positive_rate",
    "calibration_temperature", "none_logit_bias", "ordinal_sigma", "ordinal_cut_points",
    "ordinal_decode", "selection_decode", "label_smoothing", "confusion_matrix",
    "balanced_accuracy", "sample_stats_path",
}  # fmt: skip

# What finalisation fits into a promoted single-label group with a __none__ class.
_FITTED_KEYS = {"temperature", "class_logit_bias", "none_metrics", "val_probs", "val_labels"}


class _Killed(BaseException):
    """The process dies here: no ``except Exception`` sees it."""


# --------------------------------------------------------------------------- #
# Runs                                                                         #
# --------------------------------------------------------------------------- #


def _seed(n: int) -> None:
    torch.manual_seed(n)
    np.random.seed(n)
    random.seed(n)


@pytest.fixture(scope="module")
def workdir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("finalize_kill")
    make_group(root / "group", per_class=6, val_per_class=3)
    return root


def _place_incumbent(checkpoints: Path, kind: str) -> bytes | None:
    """The current model a run starts from and is judged against.

    ``none``: there is none. ``other_classes``: one for another class list, so
    the candidate wins on ``class_mismatch`` whatever it scores; a candidate
    judged against anything else wins on another reason. ``kept``: one for these
    classes, which :func:`_stand_ins` makes win.
    """
    checkpoints.mkdir(parents=True, exist_ok=True)
    if kind == "none":
        return None
    torch.manual_seed(11)
    names = ["up", "down", "__none__"] if kind == "other_classes" else list(CLASSES)
    torch.save(
        {
            "state_dict": micro_convnext().state_dict(),
            "num_classes": len(names),
            "model_size": "atto",
            "class_names": names,
        },
        checkpoints / "best.pt",
    )
    return (checkpoints / "best.pt").read_bytes()


def _stand_ins(monkeypatch, kind: str) -> None:
    """The micro trunk; a greedy soup that is always adopted, so ``candidate.pt``
    is rewritten in finalisation; for ``kept``, an incumbent that wins whenever
    it could be scored."""
    use_micro_trunk(monkeypatch)
    real_soup = group_task.greedy_soup

    def _adopted_soup(candidates, eval_fn):
        state, score, accepted = real_soup(candidates, eval_fn)
        return state, score + 1.0, accepted

    monkeypatch.setattr(group_task, "greedy_soup", _adopted_soup)
    if kind == "kept":
        real_decide = finalize.decide_promotion

        def _incumbent_wins(**kw):
            if kw.get("eval_ok") and kw.get("incumbent_score") is not None:
                return False, PromotionReason.incumbent_wins
            return real_decide(**kw)

        monkeypatch.setattr(finalize, "decide_promotion", _incumbent_wins)


def _config(workdir: Path, tag: str, *, features: str = "features", **kw):
    run = workdir / tag
    base = dict(
        max_epochs=2,
        batch_size=4,
        use_greedy_soup=True,
        use_swa=True,
        swa_start_frac=0.0,
        record_sample_stats=True,
        backup_dir=str(run / "backups"),
        resume_from=str(run / "backups"),
        embedding_cache_dir=str(run / features),
    )
    base.update(kw)
    return group_config(workdir / "group", run / "checkpoints", **base)


def _attempt(config, *, seed: int = 0, on_frame=None, task_cls=GroupTask, **events) -> dict:
    _seed(seed)
    return GenericTrainer().run(task_cls(config), progress_callback=on_frame, **events)


def _head_config(workdir: Path, tag: str, features: Path):
    return group_config(
        workdir / "group", workdir / tag / "checkpoints", embedding_cache_dir=str(features)
    )


# --------------------------------------------------------------------------- #
# Outcomes                                                                     #
# --------------------------------------------------------------------------- #


def _portable(value, root: Path):
    """``value`` with the run's own folder named ``<run>``, so two runs compare."""
    if isinstance(value, dict):
        return {k: _portable(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_portable(v, root) for v in value]
    if isinstance(value, str):
        return value.replace(str(root), "<run>")
    return value


def _outcome(result: dict, checkpoints: Path) -> dict:
    files = sorted(
        p.relative_to(checkpoints).as_posix() for p in checkpoints.rglob("*") if p.is_file()
    )
    stats = checkpoints / "sample_stats.json"
    return {
        "result": _portable(result, checkpoints),
        "files": files,
        "best": torch.load(checkpoints / "best.pt", map_location="cpu", weights_only=False),
        "sample_stats": json.loads(stats.read_text()) if stats.exists() else None,
    }


def _assert_same(a, b, where: str = "outcome") -> None:
    """Exactly equal, tensors and NaNs included."""
    if isinstance(a, dict):
        assert isinstance(b, dict), where
        assert set(a) == set(b), (where, sorted(set(a) ^ set(b), key=str))
        for key in a:
            _assert_same(a[key], b[key], f"{where}.{key}")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), (where, a, b)
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{where}[{i}]")
    elif isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape, where
        if a.is_floating_point():
            assert torch.allclose(a, b, rtol=0.0, atol=0.0, equal_nan=True), where
        else:
            assert torch.equal(a, b), where
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype, where
        assert np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"), where
    elif isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        pass
    else:
        assert a == b, (where, a, b)


_CONTROLS: dict = {}


def _control(workdir: Path, kind: str, scenario: str = "plain", *, events=None, **kw) -> dict:
    """The run left alone, once per (incumbent, scenario)."""
    key = (kind, scenario, tuple(sorted(kw.items())))
    if key not in _CONTROLS:
        tag = f"control-{kind}-{scenario}"
        checkpoints = workdir / tag / "checkpoints"
        _place_incumbent(checkpoints, kind)
        hooks = events() if events else {}
        result = _attempt(_config(workdir, tag, **kw), **hooks)
        _CONTROLS[key] = _outcome(result, checkpoints)
    return _CONTROLS[key]


# --------------------------------------------------------------------------- #
# Kills                                                                        #
# --------------------------------------------------------------------------- #

# point -> (when, owner, name): the process dies on entering ``owner.name``, or
# just after it returns.
_SEAMS = {
    # The greedy soup is under way; candidate.pt still holds the best epoch.
    "soup": ("entry", group_task, "greedy_soup"),
    # The soup has rewritten candidate.pt; nothing has been judged.
    "after_soup": ("entry", gt, "_compare_promote_finalize"),
    # The candidate is being judged against the current model.
    "comparison": ("entry", finalize, "decide_promotion"),
    # The current model is kept aside; the candidate has not landed over it.
    "promotion_copy": ("entry", finalize, "_copy_into_place"),
    # The candidate is the current model; nothing is fitted into it yet.
    "after_copy": ("entry", finalize, "_apply_and_persist_priors"),
    # Priors and calibration are in the promoted model; the rest is not.
    "calibration_write": ("return", finalize, "_persist_softmax_calibration"),
    # The strictness data, finalisation's last write, is in.
    "last_write": ("return", finalize, "_persist_strictness_val_data"),
    # Finalisation has returned; every backup is still there.
    "before_backup_deletion": ("entry", BackupCoordinator, "delete_backups"),
    # The backups are gone; what finalisation kept for a restart is not.
    "before_cleanup": ("entry", GroupTask, "on_backups_deleted"),
    "head_before_cleanup": ("entry", HeadOnlyTask, "on_backups_deleted"),
}


def _kill_at(mp: pytest.MonkeyPatch, point: str) -> None:
    if point == "during_backup_deletion":
        # The first backup file is unlinked, then the process dies.
        deleting = {"on": False}
        real_unlink = Path.unlink
        real_delete_all = TrainingStateManager.delete_all

        def _unlink(self, *a, **k):
            real_unlink(self, *a, **k)
            if deleting["on"]:
                raise _Killed(point)

        def _delete_all(self):
            deleting["on"] = True
            try:
                real_delete_all(self)
            finally:
                deleting["on"] = False

        mp.setattr(Path, "unlink", _unlink)
        mp.setattr(TrainingStateManager, "delete_all", _delete_all)
        return
    when, owner, name = _SEAMS[point]
    real = getattr(owner, name)

    def _die(*_a, **_k):
        raise _Killed(point)

    def _die_after(*a, **k):
        real(*a, **k)
        raise _Killed(point)

    mp.setattr(owner, name, _die if when == "entry" else _die_after)


def _kill(config, point: str, **run_kw) -> None:
    kill = pytest.MonkeyPatch()
    _kill_at(kill, point)
    try:
        with pytest.raises(_Killed):
            _attempt(config, **run_kw)
    finally:
        kill.undo()


def _watch_training(monkeypatch) -> dict:
    """Counts the epochs trained (not the preflight's dry step) and warm-ups."""
    seen = {"epochs": 0, "warmup": 0}
    real_epoch, real_warmup = gt._train_one_epoch, gt._warmup_head_probe

    def _epoch(*a, **k):
        if not k.get("dry_run"):
            seen["epochs"] += 1
        return real_epoch(*a, **k)

    def _warmup(*a, **k):
        seen["warmup"] += 1
        return real_warmup(*a, **k)

    monkeypatch.setattr(gt, "_train_one_epoch", _epoch)
    monkeypatch.setattr(gt, "_warmup_head_probe", _warmup)
    return seen


def _killed_and_started_again(
    workdir: Path,
    monkeypatch,
    kind: str,
    point: str,
    *,
    scenario: str = "plain",
    events=None,
    seed: int = 123,
    **kw,
):
    """Kill a run at ``point``, start it again with the same folders (a fresh
    RNG: finalisation must not depend on it). Returns its outcome, what it
    trained, and its frames."""
    tag = f"{kind}-{scenario}-{point}"
    checkpoints = workdir / tag / "checkpoints"
    _place_incumbent(checkpoints, kind)
    _kill(_config(workdir, tag, **kw), point, **(events() if events else {}))
    backups = workdir / tag / "backups"
    if point not in ("during_backup_deletion", "before_cleanup"):
        assert list(backups.glob("backup_*.pt")), "the kill came after the backups went"

    seen = _watch_training(monkeypatch)
    frames: list[dict] = []
    result = _attempt(_config(workdir, tag, **kw), seed=seed, on_frame=frames.append)
    assert not backups.exists()
    return _outcome(result, checkpoints), seen, frames


# --------------------------------------------------------------------------- #
# Point 3: left alone, a run finalises as it always did                        #
# --------------------------------------------------------------------------- #


def test_left_alone_a_promoted_run_ends_as_it_always_did(workdir, monkeypatch):
    """The candidate as it was judged is the shipped model, plus what
    finalisation fitted into it; the result carries the keys it always did; the
    checkpoint folder holds the shipped model and the sample stats, nothing
    else; the backups are gone."""
    _stand_ins(monkeypatch, "other_classes")
    judged: dict = {}
    real = gt._compare_promote_finalize

    def _spy(config, **kw):
        judged.update(torch.load(kw["candidate_path"], map_location="cpu", weights_only=False))
        return real(config, **kw)

    monkeypatch.setattr(gt, "_compare_promote_finalize", _spy)
    tag = "pin-promoted"
    checkpoints = workdir / tag / "checkpoints"
    incumbent = _place_incumbent(checkpoints, "other_classes")

    result = _attempt(_config(workdir, tag))

    assert result["promotion_reason"] == "class_mismatch"
    assert result["checkpoint_path"] == str(checkpoints / "best.pt")
    assert set(result) == _RESULT_KEYS
    assert result["epochs_completed"] == 2 and result["val_class_counts"]
    assert sorted(p.relative_to(checkpoints).as_posix() for p in checkpoints.rglob("*")) == [
        "best.pt",
        "sample_stats.json",
    ]
    assert not (workdir / tag / "backups").exists()
    shipped = torch.load(checkpoints / "best.pt", map_location="cpu", weights_only=False)
    assert (checkpoints / "best.pt").read_bytes() != incumbent
    for key, value in judged.items():
        _assert_same(value, shipped[key], f"shipped.{key}")
    assert _FITTED_KEYS <= set(shipped) - set(judged)
    assert shipped["temperature"] == result["calibration_temperature"]
    assert result["sample_stats_path"] == str(checkpoints / "sample_stats.json")


def test_left_alone_a_kept_incumbent_ends_as_it_always_did(workdir, monkeypatch):
    """Kept: the current model's file is not touched, the candidate goes."""
    _stand_ins(monkeypatch, "kept")
    tag = "pin-kept"
    checkpoints = workdir / tag / "checkpoints"
    incumbent = _place_incumbent(checkpoints, "kept")

    result = _attempt(_config(workdir, tag))

    assert result["promotion_reason"] == "incumbent_wins"
    assert set(result) == _RESULT_KEYS
    assert (checkpoints / "best.pt").read_bytes() == incumbent
    assert sorted(p.relative_to(checkpoints).as_posix() for p in checkpoints.rglob("*")) == [
        "best.pt",
        "sample_stats.json",
    ]
    assert not (workdir / tag / "backups").exists()


def test_left_alone_a_head_only_run_ends_as_it_always_did(workdir, monkeypatch):
    use_micro_trunk(monkeypatch)
    judged: dict = {}
    real = gt._compare_promote_finalize

    def _spy(config, **kw):
        judged.update(torch.load(kw["candidate_path"], map_location="cpu", weights_only=False))
        return real(config, **kw)

    monkeypatch.setattr(gt, "_compare_promote_finalize", _spy)
    tag = "pin-head"
    checkpoints = workdir / tag / "checkpoints"
    _place_incumbent(checkpoints, "none")

    result = _attempt(
        _head_config(workdir, tag, workdir / tag / "features"), task_cls=HeadOnlyTask
    )

    assert result["promotion_reason"] == "no_incumbent" and result["mode"] == "head_only"
    assert sorted(p.relative_to(checkpoints).as_posix() for p in checkpoints.rglob("*")) == [
        "best.pt"
    ]
    shipped = torch.load(checkpoints / "best.pt", map_location="cpu", weights_only=False)
    for key, value in judged.items():
        _assert_same(value, shipped[key], f"shipped.{key}")
    assert _FITTED_KEYS <= set(shipped)


# --------------------------------------------------------------------------- #
# Points 1 and 2: killed anywhere after the last epoch, a run finishes          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "point",
    [
        "soup",
        "after_soup",
        "comparison",
        "promotion_copy",
        "after_copy",
        "calibration_write",
        "last_write",
        "before_backup_deletion",
        "during_backup_deletion",
    ],
)
def test_killed_in_finalisation_it_finishes_as_if_left_alone(workdir, monkeypatch, point):
    """Judged against the original current model: a candidate judged against
    the copy of itself an earlier attempt put in place would not win on
    ``class_mismatch``."""
    _stand_ins(monkeypatch, "other_classes")
    control = _control(workdir, "other_classes")

    outcome, seen, frames = _killed_and_started_again(workdir, monkeypatch, "other_classes", point)

    assert seen == {"epochs": 0, "warmup": 0}
    assert not any(f.get("type") == "epoch_complete" for f in frames)
    assert outcome["result"]["promotion_reason"] == "class_mismatch"
    _assert_same(control, outcome)


@pytest.mark.parametrize("point", ["promotion_copy", "after_copy", "last_write"])
def test_with_no_current_model_a_killed_run_still_has_none_to_beat(workdir, monkeypatch, point):
    _stand_ins(monkeypatch, "none")
    control = _control(workdir, "none")

    outcome, seen, _frames = _killed_and_started_again(workdir, monkeypatch, "none", point)

    assert seen == {"epochs": 0, "warmup": 0}
    assert outcome["result"]["promotion_reason"] == "no_incumbent"
    _assert_same(control, outcome)


@pytest.mark.parametrize("point", ["after_soup", "comparison", "before_backup_deletion"])
def test_a_kept_incumbent_killed_in_finalisation_is_kept_untouched(workdir, monkeypatch, point):
    _stand_ins(monkeypatch, "kept")
    control = _control(workdir, "kept")

    outcome, seen, _frames = _killed_and_started_again(workdir, monkeypatch, "kept", point)

    assert seen == {"epochs": 0, "warmup": 0}
    assert outcome["result"]["promotion_reason"] == "incumbent_wins"
    _assert_same(control, outcome)


def _stop_after_first_epoch() -> dict:
    stop = threading.Event()

    def _on_frame(frame: dict) -> None:
        if frame.get("type") == "epoch_complete":
            stop.set()

    return {"stop_event": stop, "on_frame": _on_frame}


def test_a_stopped_run_killed_in_finalisation_does_not_train_on(workdir, monkeypatch):
    """Stopped after the first of three epochs: the restart finalises what the
    stop left, it does not train the two epochs the stop skipped."""
    _stand_ins(monkeypatch, "other_classes")
    control = _control(
        workdir, "other_classes", "stopped", events=_stop_after_first_epoch, max_epochs=3
    )

    outcome, seen, _frames = _killed_and_started_again(
        workdir, monkeypatch, "other_classes", "after_copy",
        scenario="stopped", events=_stop_after_first_epoch, max_epochs=3,
    )  # fmt: skip

    assert seen == {"epochs": 0, "warmup": 0}
    _assert_same(control, outcome)


def test_an_early_stopped_run_killed_in_finalisation_does_not_train_on(workdir, monkeypatch):
    """Patience ran out after the first of three epochs."""
    _stand_ins(monkeypatch, "other_classes")
    control = _control(workdir, "other_classes", "patience", max_epochs=3, patience=0)
    assert control["result"]["epochs_completed"] == 1

    outcome, seen, _frames = _killed_and_started_again(
        workdir, monkeypatch, "other_classes", "after_copy",
        scenario="patience", max_epochs=3, patience=0,
    )  # fmt: skip

    assert seen == {"epochs": 0, "warmup": 0}
    _assert_same(control, outcome)


class _Scaler:
    """Plays torch's GradScaler on a CPU: step 1 is skipped, halving the scale."""

    def __init__(self) -> None:
        self.value, self.steps, self._skip = 65536.0, 0, False

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer) -> None:
        pass

    def step(self, optimizer) -> None:
        self.steps += 1
        self._skip = self.steps == 1
        if not self._skip:
            optimizer.step()

    def update(self) -> None:
        if self._skip:
            self.value *= 0.5

    def get_scale(self) -> float:
        return self.value

    def state_dict(self) -> dict:
        return {"scale": self.value}

    def load_state_dict(self, state: dict) -> None:
        self.value = state["scale"]


def test_an_fp16_run_killed_in_finalisation_reports_its_scaler(workdir, monkeypatch):
    """The VM trains in fp16: the restart's result carries the scale and the
    skipped steps the run trained with."""
    _stand_ins(monkeypatch, "other_classes")
    monkeypatch.setattr(group_task, "make_grad_scaler", lambda *_a: _Scaler())
    control = _control(workdir, "other_classes", "fp16")
    assert control["result"]["grad_scaler"] is True
    assert control["result"]["grad_scale"] == 32768.0 and control["result"]["skipped_steps"] == 1

    outcome, seen, _frames = _killed_and_started_again(
        workdir, monkeypatch, "other_classes", "calibration_write", scenario="fp16"
    )

    assert seen == {"epochs": 0, "warmup": 0}
    _assert_same(control, outcome)


def test_a_pause_after_the_last_epoch_stops_before_finalisation(workdir, monkeypatch):
    """A pause (the VM's SIGTERM) that comes in after the last epoch trained
    stops the run before finalisation; started again it only finalises."""
    _stand_ins(monkeypatch, "other_classes")
    control = _control(workdir, "other_classes")
    tag = "paused-after-last-epoch"
    checkpoints = workdir / tag / "checkpoints"
    incumbent = _place_incumbent(checkpoints, "other_classes")
    pause = threading.Event()
    epochs = []

    def _on_frame(frame: dict) -> None:
        if frame.get("type") == "epoch_complete":
            epochs.append(frame)
            if len(epochs) == 2:
                pause.set()

    first = _attempt(_config(workdir, tag), on_frame=_on_frame, pause_event=pause)

    assert first.get("paused") is True and first["backup_path"]
    assert (checkpoints / "best.pt").read_bytes() == incumbent
    seen = _watch_training(monkeypatch)
    result = _attempt(_config(workdir, tag), seed=123)
    assert seen == {"epochs": 0, "warmup": 0}
    _assert_same(control, _outcome(result, checkpoints))


def test_killed_after_the_backups_went_it_runs_again_from_the_original(workdir, monkeypatch):
    """The backups are gone, so there is nothing to resume: the next start puts
    the original current model back and the run trains again from it, ending
    where the run left alone did. (Its result was never handed back.)"""
    _stand_ins(monkeypatch, "other_classes")
    control = _control(workdir, "other_classes")
    tag = "killed-before-cleanup"
    checkpoints = workdir / tag / "checkpoints"
    _place_incumbent(checkpoints, "other_classes")
    _kill(_config(workdir, tag), "before_cleanup")
    assert not (workdir / tag / "backups").exists()

    seen = _watch_training(monkeypatch)
    # A fresh feature cache, built as the run left alone built its own.
    result = _attempt(_config(workdir, tag, features="features-again"))

    assert seen["epochs"] == 2
    assert result["promotion_reason"] == "class_mismatch"
    _assert_same(control, _outcome(result, checkpoints))


def test_the_newest_backup_is_deleted_last(tmp_path, monkeypatch):
    """Whatever order the file system lists them in: a kill part-way through
    the deletion leaves the newest backup, never only an older one."""
    manager = TrainingStateManager(tmp_path / "backups", keep=2)
    manager.save({"global_step": 1}, reason="periodic")
    newest = manager.save({"global_step": 2}, reason="periodic")
    (tmp_path / "backups" / f"{newest.name}.tmp").write_bytes(b"")
    real_glob, real_unlink = Path.glob, Path.unlink
    unlinked: list[str] = []

    def _newest_first(self, pattern, **kw):
        return iter(sorted(real_glob(self, pattern, **kw), reverse=True))

    def _unlink(self, *a, **k):
        if len(unlinked) == 2:
            raise _Killed("during the deletion")
        unlinked.append(self.name)
        real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "glob", _newest_first)
    monkeypatch.setattr(Path, "unlink", _unlink)
    with pytest.raises(_Killed):
        manager.delete_all()
    monkeypatch.undo()

    assert [p.name for p in TrainingStateManager(tmp_path / "backups").list_backups()] == [
        newest.name
    ]


# --------------------------------------------------------------------------- #
# Head-only: no backups; started again it probes afresh from the original      #
# --------------------------------------------------------------------------- #


def _head_features(workdir: Path, kind: str) -> Path:
    """A feature cache every head-only run of ``kind`` shares, built once, so
    each run reuses it the same way."""
    features = workdir / f"head-features-{kind}"
    if not features.exists():
        tag = f"head-warm-{kind}"
        _place_incumbent(workdir / tag / "checkpoints", kind)
        _attempt(_head_config(workdir, tag, features), task_cls=HeadOnlyTask)
    return features


def _head_control(workdir: Path, kind: str) -> dict:
    key = ("head", kind)
    if key not in _CONTROLS:
        features = _head_features(workdir, kind)
        tag = f"head-control-{kind}"
        checkpoints = workdir / tag / "checkpoints"
        _place_incumbent(checkpoints, kind)
        result = _attempt(_head_config(workdir, tag, features), task_cls=HeadOnlyTask)
        _CONTROLS[key] = _outcome(result, checkpoints)
    return _CONTROLS[key]


@pytest.mark.parametrize(
    ("kind", "point"),
    [
        ("none", "after_copy"),
        ("none", "last_write"),
        ("none", "head_before_cleanup"),
        ("other_classes", "after_copy"),
    ],
)
def test_a_head_only_run_killed_in_finalisation_lands_where_it_would_have(
    workdir, monkeypatch, kind, point
):
    use_micro_trunk(monkeypatch)
    control = _head_control(workdir, kind)
    features = _head_features(workdir, kind)
    tag = f"head-{kind}-{point}"
    checkpoints = workdir / tag / "checkpoints"
    _place_incumbent(checkpoints, kind)
    _kill(_head_config(workdir, tag, features), point, task_cls=HeadOnlyTask)

    result = _attempt(_head_config(workdir, tag, features), task_cls=HeadOnlyTask)

    assert result["promotion_reason"] == control["result"]["promotion_reason"]
    _assert_same(control, _outcome(result, checkpoints))
