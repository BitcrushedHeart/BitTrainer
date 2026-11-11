"""A resumed group run trains and reports what its warm-up sweeps chose (Bitcrush ISSUE-1690).

Before the first epoch a fresh group run sweeps the label softness and the
``__none__`` oversampling on cached features and writes its choices onto the
config. A resume skips the sweeps and re-applies the chosen values from the
backup (``_apply_resolved``), before the datasets are built. Two things did not
come back:

* what the sweeps found (the chosen values and their tables), which the result
  reports: a resumed run reported them unset;
* whether the loss takes soft targets. A fresh run decides that from the label
  smoothing it starts with (0.1) before the sweep runs; a resumed run decided it
  from the smoothing the sweep chose. A sweep that chose no smoothing left the
  resumed epochs on the plain cross-entropy branch where the unbroken run was on
  the soft-target branch.

And a run resumed in mid-training rewrote ``sample_stats.json`` with only the
epochs it trained itself.

The sweeps here are made to choose values the run does not start with: no label
smoothing, ``__none__`` oversampled. CPU-only, on the micro stand-in trunk
(``group_stub``); each run takes seconds.
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

import bittrainer.group_trainer as gt
import bittrainer.probes as probes
from bittrainer.generic.generic_trainer import GenericTrainer
from bittrainer.generic.tasks.group_task import GroupTask
from bittrainer.tests.group_stub import group_config, make_group, use_micro_trunk
from bittrainer.training_state import TrainingStateManager

_SWEEP_REPORTS = (
    "selected_softness_kind",
    "selected_softness_value",
    "soft_label_tuning_metric",
    "soft_label_tuning_results",
    "soft_label_tuning_elapsed_ms",
    "selected_oversample_none",
    "oversample_tuning_metric",
    "oversample_tuning_results",
    "oversample_tuning_elapsed_ms",
)
_TIMINGS = ("soft_label_tuning_elapsed_ms", "oversample_tuning_elapsed_ms")
_RESOLVED_BEFORE = ("oversample_none", "label_smoothing", "ordinal_sigma", "class_balance_mode")


def _seed(n: int = 0) -> None:
    torch.manual_seed(n)
    np.random.seed(n)
    random.seed(n)


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    use_micro_trunk(monkeypatch)
    # Choices the run does not start with (label smoothing 0.1, oversampling off).
    monkeypatch.setattr(probes, "_auto_softness_candidates", lambda _kind: [0.0])
    monkeypatch.setattr(probes, "_OVERSAMPLE_NONE_CANDIDATES", [("1.5x", True)])
    make_group(tmp_path / "group", per_class=6, val_per_class=3)
    return tmp_path


def _config(root: Path, tag: str):
    run = root / tag
    return group_config(
        root / "group",
        run / "checkpoints",
        max_epochs=2,
        auto_label_softness=True,
        auto_oversample_none=True,
        record_sample_stats=True,
        backup_dir=str(run / "backups"),
        resume_from=str(run / "backups"),
        embedding_cache_dir=str(run / "features"),
    )


def _record_epochs(monkeypatch) -> list[dict]:
    """What every trained epoch trains with."""
    epochs: list[dict] = []
    real = gt._train_one_epoch

    def _spy(model, loader, optimizer, config, device, dtype, **kw):
        if not kw.get("dry_run"):
            weights = kw.get("class_weights")
            epochs.append(
                {
                    "use_soft_targets": kw.get("use_soft_targets"),
                    "label_smoothing": config.label_smoothing,
                    "ordinal_sigma": config.ordinal_sigma,
                    "oversample_none": config.oversample_none,
                    "train_set_oversampled": loader.dataset.oversample_none,
                    "batches": len(loader),
                    "class_balance_mode": config.class_balance_mode,
                    "mixup_enabled": kw.get("mixup_enabled"),
                    "class_weights": None if weights is None else weights.tolist(),
                }
            )
        return real(model, loader, optimizer, config, device, dtype, **kw)

    monkeypatch.setattr(gt, "_train_one_epoch", _spy)
    return epochs


def _paused_after_first_epoch(config) -> dict:
    pause = threading.Event()

    def _on_frame(frame: dict) -> None:
        if frame.get("type") == "epoch_complete":
            pause.set()

    _seed(0)
    result = GenericTrainer().run(GroupTask(config), progress_callback=_on_frame, pause_event=pause)
    assert result.get("paused") is True
    return result


def _untimed(value):
    """``value`` without the sweep tables' per-row timings."""
    if isinstance(value, dict):
        return {k: _untimed(v) for k, v in value.items() if k != "elapsed_ms"}
    if isinstance(value, list):
        return [_untimed(v) for v in value]
    return value


def _same(a, b) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def test_a_resumed_run_trains_and_reports_what_the_unbroken_run_does(root, monkeypatch):
    epochs = _record_epochs(monkeypatch)
    _seed(0)
    unbroken = GenericTrainer().run(GroupTask(_config(root, "unbroken")))
    unbroken_epochs = list(epochs)
    # What the sweeps chose, not what the run started with.
    assert unbroken_epochs[1]["label_smoothing"] == 0.0
    assert unbroken_epochs[1]["oversample_none"] is True
    assert unbroken_epochs[1]["train_set_oversampled"] is True
    assert unbroken["selected_softness_value"] == 0.0
    assert unbroken["selected_oversample_none"] is True

    epochs.clear()
    first = _config(root, "resumed")
    _paused_after_first_epoch(first)
    assert epochs == unbroken_epochs[:1]

    epochs.clear()
    _seed(123)
    resumed = GenericTrainer().run(GroupTask(_config(root, "resumed")))

    assert "paused" not in resumed
    # The epoch the pause left is trained as the unbroken run trained it.
    assert epochs == unbroken_epochs[1:]
    # It reports what its first attempt's sweeps found ...
    for name in _SWEEP_REPORTS:
        assert _same(resumed[name], getattr(first, name)), name
    # ... which is what the unbroken run found, bar how long the sweeps took.
    for name in set(_SWEEP_REPORTS) - set(_TIMINGS):
        assert _same(_untimed(resumed[name]), _untimed(unbroken[name])), name


def test_a_backup_from_before_still_resumes_as_it_did(root, monkeypatch):
    """A backup without the new fields resumes as before: the chosen values
    train, the loss branch follows them, the sweep reports stay unset."""
    epochs = _record_epochs(monkeypatch)
    _paused_after_first_epoch(_config(root, "old"))
    newest = TrainingStateManager(root / "old" / "backups").list_backups()[-1]
    state = torch.load(newest, map_location="cpu", weights_only=False)
    state.pop("use_soft", None)
    state["resolved"] = {k: state["resolved"][k] for k in _RESOLVED_BEFORE}
    torch.save(state, newest)

    epochs.clear()
    resumed = GenericTrainer().run(GroupTask(_config(root, "old")))

    assert "paused" not in resumed and resumed["epochs_completed"] == 2
    assert epochs[0]["label_smoothing"] == 0.0 and epochs[0]["train_set_oversampled"] is True
    assert epochs[0]["use_soft_targets"] is False
    assert resumed["selected_softness_value"] is None
    assert resumed["soft_label_tuning_results"] == []
    assert resumed["selected_oversample_none"] is None


def test_a_resumed_run_keeps_the_sample_stats_of_its_earlier_epochs(root):
    stats = root / "stats" / "checkpoints" / "sample_stats.json"
    _paused_after_first_epoch(_config(root, "stats"))
    before = json.loads(stats.read_text(encoding="utf-8"))
    assert before["epochs"] == 1

    GenericTrainer().run(GroupTask(_config(root, "stats")))
    after = json.loads(stats.read_text(encoding="utf-8"))

    assert after["epochs"] == 2
    assert [s["path"] for s in after["samples"]][: len(before["samples"])] == [
        s["path"] for s in before["samples"]
    ]
    for old, new in zip(before["samples"], after["samples"]):
        if old["split"] == "train":
            assert new["train_loss"][0] == old["train_loss"][0]
            assert len(new["train_loss"]) == 2
        else:
            assert new["val_probs"][0] == old["val_probs"][0]
            assert new["val_loss"][0] == old["val_loss"][0]
            assert new["val_probs"][1] is not None
