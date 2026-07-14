"""End-to-end backup / pause / resume for the group trainer (Bitcrush ISSUE-0405).

CPU-only, tiny synthetic groups, ``dataloader_workers=0`` for bit-exactness, the
smallest ``atto`` backbone, and the warmup sweeps + compile disabled to keep each
run fast. These drive ``run_group_training`` for real (not a seam) because the
whole point is that a resumed run continues the *actual* training trajectory.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image

import bittrainer.group_trainer as gt
from bittrainer.group_trainer import GroupTrainConfig, run_group_training

_CLASSES = ["a", "b", "c"]


def _make_group(root, *, per_class=6, seed=0):
    """Write a tiny deterministic group folder: root/<class>/<split>/img.png."""
    _make_labelled_group(root, _CLASSES, per_class=per_class, seed=seed)


def _cfg(group_folder, checkpoint_dir, **kw) -> GroupTrainConfig:
    base = dict(
        group_folder=str(group_folder),
        num_classes=len(_CLASSES),
        class_names=list(_CLASSES),
        checkpoint_dir=str(checkpoint_dir),
        max_epochs=3,
        patience=99,  # never early-stop in these short runs
        backbone_variant="atto",
        device="cpu",
        dtype="float32",
        batch_size=2,           # skip autobatch; ~9 train imgs -> a few batches
        use_cache=False,
        use_compile=False,
        channels_last=False,
        auto_label_softness=False,
        auto_oversample_none=False,
        use_greedy_soup=False,
        dataloader_workers=0,
        head_max_epochs=2,   # keep the warmup head probe fast
        head_patience=1,
        # Random-init backbone: offline + fully seed-deterministic (no timm download).
        backbone_init={"source": "random_init", "checkpoint_path": None},
    )
    base.update(kw)
    return GroupTrainConfig(**base)


def _seed(n: int = 0) -> None:
    torch.manual_seed(n)
    np.random.seed(n)
    random.seed(n)


def _final_state(result):
    ckpt = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=False)
    return ckpt["state_dict"]


class _FlagEvent:
    """Duck-typed stop/pause event (``.is_set()``) that a callback can flip."""

    def __init__(self) -> None:
        self._set = False

    def set(self) -> None:
        self._set = True

    def is_set(self) -> bool:
        return self._set


def _pause_after_first_periodic():
    """Returns (pause_event, callback) that pauses right after the first
    mid-epoch periodic backup — i.e. at the next gradient-accumulation boundary."""
    pause = _FlagEvent()

    def _cb(msg):
        if msg.get("type") == "backup_complete" and msg.get("reason") == "periodic":
            pause.set()

    return pause, _cb


def _pause_at_epoch(n: int):
    """Returns (pause_event, callback) that pauses at the top of epoch index
    ``n`` — i.e. right after epoch ``n``'s ``epoch_complete``."""
    pause = _FlagEvent()

    def _cb(msg):
        if msg.get("type") == "epoch_complete" and msg.get("epoch") == n:
            pause.set()

    return pause, _cb


def test_backward_compat_no_backup_when_disabled(tmp_path):
    """Regression guard: called without backup_dir/resume_from/pause_event, the
    trainer writes ZERO backup files and returns the normal result shape."""
    _make_group(tmp_path / "grp", per_class=4, seed=1)
    _seed(0)
    result = run_group_training(_cfg(tmp_path / "grp", tmp_path / "ckpt", max_epochs=2))
    assert "paused" not in result
    assert result["checkpoint_path"].endswith("best.pt")
    assert result["epochs_completed"] == 2
    # No backup directory materialised anywhere.
    assert not list((tmp_path).glob("**/backup_*.pt"))


def _resume_dir(ckpt_dir):
    return str(ckpt_dir / "backups")


def _assert_states_close(a, b, *, atol=1e-5, rtol=1e-4):
    assert set(a) == set(b)
    for k in a:
        torch.testing.assert_close(a[k].float(), b[k].float(), atol=atol, rtol=rtol)


def test_group_resume_epoch_boundary_equivalence(tmp_path):
    """Regression guard: a 3-epoch control run and a (1-epoch + pause + resume
    for the remaining 2) run land on allclose final weights and equal best epoch.
    """
    _make_group(tmp_path / "grp", per_class=4, seed=3)

    # Control: 3 uninterrupted epochs, no backups.
    _seed(0)
    control = run_group_training(_cfg(tmp_path / "grp", tmp_path / "ck_ctrl", max_epochs=3))

    # Interrupted: pause at the epoch-1 boundary, then resume for the rest.
    pause, _cb = _pause_at_epoch(1)

    _seed(0)
    first = run_group_training(
        _cfg(tmp_path / "grp", tmp_path / "ck_res", max_epochs=3, backup_dir=_resume_dir(tmp_path / "ck_res")),
        progress_callback=_cb, pause_event=pause,
    )
    assert first.get("paused") is True

    _seed(0)
    resumed = run_group_training(
        _cfg(
            tmp_path / "grp", tmp_path / "ck_res", max_epochs=3,
            backup_dir=_resume_dir(tmp_path / "ck_res"),
            resume_from=_resume_dir(tmp_path / "ck_res"),
        ),
    )
    assert "paused" not in resumed
    _assert_states_close(_final_state(control), _final_state(resumed))
    assert control["best_epoch"] == resumed["best_epoch"]
    assert abs(control["best_val_macro_f1"] - resumed["best_val_macro_f1"]) < 1e-4


def test_group_resume_mid_epoch_bit_exact(tmp_path):
    """Regression guard (bit-exact, workers=0): pausing mid-epoch and resuming
    reproduces the uninterrupted control's final state_dict EXACTLY."""
    _make_group(tmp_path / "grp", per_class=6, seed=5)

    _seed(0)
    control = run_group_training(_cfg(tmp_path / "grp", tmp_path / "ck_ctrl", max_epochs=2))

    pause, cb = _pause_after_first_periodic()
    _seed(0)
    first = run_group_training(
        _cfg(
            tmp_path / "grp", tmp_path / "ck_res", max_epochs=2,
            backup_dir=_resume_dir(tmp_path / "ck_res"), backup_every_steps=2,
        ),
        progress_callback=cb, pause_event=pause,
    )
    assert first.get("paused") is True
    assert first["backup_path"]  # a mid-epoch backup exists

    _seed(0)
    resumed = run_group_training(
        _cfg(
            tmp_path / "grp", tmp_path / "ck_res", max_epochs=2,
            backup_dir=_resume_dir(tmp_path / "ck_res"),
            resume_from=_resume_dir(tmp_path / "ck_res"),
        ),
    )
    assert "paused" not in resumed
    _assert_states_close(_final_state(control), _final_state(resumed), atol=0.0, rtol=0.0)


def test_pause_returns_paused_and_never_promotes(tmp_path, monkeypatch):
    """Regression guard: a pause returns {"paused": True} with a backup on disk,
    does NOT create best.pt, and never runs the promotion/finalisation path."""
    _make_group(tmp_path / "grp", per_class=6, seed=7)

    called = {"finalize": False}
    real_finalize = gt._compare_promote_finalize

    def _spy_finalize(*a, **k):
        called["finalize"] = True
        return real_finalize(*a, **k)

    monkeypatch.setattr(gt, "_compare_promote_finalize", _spy_finalize)

    pause, cb = _pause_after_first_periodic()
    _seed(0)
    result = run_group_training(
        _cfg(
            tmp_path / "grp", tmp_path / "ck", max_epochs=2,
            backup_dir=_resume_dir(tmp_path / "ck"), backup_every_steps=2,
        ),
        progress_callback=cb, pause_event=pause,
    )
    assert result["paused"] is True
    assert result["backup_path"] and "backup_" in result["backup_path"]
    assert called["finalize"] is False  # promotion never ran
    assert not (tmp_path / "ck" / "best.pt").exists()  # nothing promoted


def test_backup_on_exception_writes_backup_and_reraises(tmp_path, monkeypatch):
    """Regression guard: a crash inside the training loop leaves a
    reason='exception' backup on disk and the original exception propagates."""
    _make_group(tmp_path / "grp", per_class=6, seed=8)

    real_train = gt._train_one_epoch
    state = {"calls": 0}

    def _boom(*a, **k):
        state["calls"] += 1
        if state["calls"] == 2:
            raise RuntimeError("synthetic training crash")
        return real_train(*a, **k)

    monkeypatch.setattr(gt, "_train_one_epoch", _boom)

    _seed(0)
    raised = False
    try:
        run_group_training(
            _cfg(
                tmp_path / "grp", tmp_path / "ck", max_epochs=3,
                backup_dir=_resume_dir(tmp_path / "ck"),
            ),
        )
    except RuntimeError as e:
        raised = str(e) == "synthetic training crash"
    assert raised, "original exception did not propagate"

    from bittrainer.training_state import TrainingStateManager

    mgr = TrainingStateManager(_resume_dir(tmp_path / "ck"))
    latest = mgr.load_latest()
    assert latest is not None and latest["reason"] == "exception"


def test_resume_skips_warmup_and_autobatch(tmp_path, monkeypatch):
    """Regression guard: a resume rebuilds the model from the backup and skips
    the warmup head probe + model warm-start; autobatch is reported as resumed."""
    _make_group(tmp_path / "grp", per_class=6, seed=9)

    pause, cb = _pause_after_first_periodic()
    _seed(0)
    first = run_group_training(
        _cfg(
            tmp_path / "grp", tmp_path / "ck", max_epochs=2,
            backup_dir=_resume_dir(tmp_path / "ck"), backup_every_steps=2,
        ),
        progress_callback=cb, pause_event=pause,
    )
    assert first["paused"] is True

    spied = {"warmup": 0, "warmstart": 0}
    real_warmup = gt._warmup_head_probe
    real_ws = gt._create_or_warmstart_model
    monkeypatch.setattr(gt, "_warmup_head_probe", lambda *a, **k: spied.__setitem__("warmup", spied["warmup"] + 1) or real_warmup(*a, **k))
    monkeypatch.setattr(gt, "_create_or_warmstart_model", lambda *a, **k: spied.__setitem__("warmstart", spied["warmstart"] + 1) or real_ws(*a, **k))

    msgs = []
    _seed(0)
    resumed = run_group_training(
        _cfg(
            tmp_path / "grp", tmp_path / "ck", max_epochs=2,
            backup_dir=_resume_dir(tmp_path / "ck"),
            resume_from=_resume_dir(tmp_path / "ck"),
        ),
        progress_callback=msgs.append,
    )
    assert "paused" not in resumed
    assert spied["warmup"] == 0, "warmup head probe ran on resume"
    assert spied["warmstart"] == 0, "warm-start model builder ran on resume"
    assert any(m.get("type") == "training_resumed" for m in msgs)
    assert any(m.get("type") == "autobatch" and m.get("resumed") for m in msgs)


def test_fingerprint_mismatch_starts_fresh_with_warning(tmp_path):
    """Regression guard: a backup whose fingerprint no longer matches the config
    is ignored — the run emits resume_skipped and trains fresh (no crash)."""
    _make_group(tmp_path / "grp", per_class=6, seed=10)

    pause, cb = _pause_after_first_periodic()
    _seed(0)
    run_group_training(
        _cfg(
            tmp_path / "grp", tmp_path / "ck", max_epochs=2,
            backup_dir=_resume_dir(tmp_path / "ck"), backup_every_steps=2,
        ),
        progress_callback=cb, pause_event=pause,
    )
    # Tamper the newest backup's fingerprint so it no longer matches.
    from bittrainer.training_state import TrainingStateManager

    mgr = TrainingStateManager(_resume_dir(tmp_path / "ck"))
    newest = mgr.list_backups()[-1]
    data = torch.load(newest, map_location="cpu", weights_only=False)
    data["fingerprint"]["class_names"] = ["x", "y", "z"]
    torch.save(data, newest)

    msgs = []
    _seed(0)
    result = run_group_training(
        _cfg(
            tmp_path / "grp", tmp_path / "ck", max_epochs=2,
            backup_dir=_resume_dir(tmp_path / "ck"),
            resume_from=_resume_dir(tmp_path / "ck"),
        ),
        progress_callback=msgs.append,
    )
    assert any(m.get("type") == "resume_skipped" for m in msgs)
    assert "paused" not in result
    assert result["checkpoint_path"].endswith("best.pt")


def test_soup_pool_survives_pause_and_resume(tmp_path):
    """Regression guard: greedy-soup candidate files survive a pause+resume and
    the finalisation still consumes them (soup_cands cleaned up only at the end)."""
    _make_group(tmp_path / "grp", per_class=6, seed=11)

    pause, _cb = _pause_at_epoch(2)

    _seed(0)
    first = run_group_training(
        _cfg(
            tmp_path / "grp", tmp_path / "ck", max_epochs=4,
            backup_dir=_resume_dir(tmp_path / "ck"), use_greedy_soup=True,
        ),
        progress_callback=_cb, pause_event=pause,
    )
    assert first["paused"] is True
    soup_dir = tmp_path / "ck" / "soup_cands"
    assert list(soup_dir.glob("cand_*.pt")), "soup candidates were deleted by the pause"

    _seed(0)
    resumed = run_group_training(
        _cfg(
            tmp_path / "grp", tmp_path / "ck", max_epochs=4,
            backup_dir=_resume_dir(tmp_path / "ck"),
            resume_from=_resume_dir(tmp_path / "ck"), use_greedy_soup=True,
        ),
    )
    assert "paused" not in resumed
    # Normal end-of-run cleanup removed the candidates.
    assert not list(soup_dir.glob("cand_*.pt"))


def test_batch_size_change_on_resume_epoch_restarts(tmp_path):
    """Regression guard: if the batch size changes on resume the stored schedule
    is discarded and the epoch restarts — no crash, run completes."""
    _make_group(tmp_path / "grp", per_class=6, seed=12)

    pause, cb = _pause_after_first_periodic()
    _seed(0)
    first = run_group_training(
        _cfg(
            tmp_path / "grp", tmp_path / "ck", max_epochs=2,
            backup_dir=_resume_dir(tmp_path / "ck"), backup_every_steps=2, batch_size=2,
        ),
        progress_callback=cb, pause_event=pause,
    )
    assert first["paused"] is True

    _seed(0)
    resumed = run_group_training(
        _cfg(
            tmp_path / "grp", tmp_path / "ck", max_epochs=2,
            backup_dir=_resume_dir(tmp_path / "ck"),
            resume_from=_resume_dir(tmp_path / "ck"), batch_size=3,  # changed
        ),
    )
    assert "paused" not in resumed
    assert resumed["checkpoint_path"].endswith("best.pt")


# ---------------------------------------------------------------------------
# Binary trainer (trainer.py): epoch-restart resume across the unfreeze boundary
# ---------------------------------------------------------------------------


def _make_concept(root, *, n_pos=55, n_neg=55, n_val=10, seed=0):
    """Binary concept folder: root/<positive|negative>/<split>/img.png."""
    rng = np.random.default_rng(seed)
    for label, n_tr in (("positive", n_pos), ("negative", n_neg)):
        for split, n in (("train", n_tr), ("val", n_val)):
            d = root / label / split
            d.mkdir(parents=True, exist_ok=True)
            for j in range(n):
                Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
                    d / f"{label}_{j}.png"
                )


def _bin_cfg(concept, ckpt, **kw):
    from bittrainer.trainer import TrainConfig

    base = dict(
        concept_folder=str(concept), checkpoint_dir=str(ckpt), max_epochs=3, patience=99,
        model_size="atto", device="cpu", dtype="float32", use_cache=False, from_scratch=True,
        dataloader_workers=0, backbone_init={"source": "random_init", "checkpoint_path": None},
    )
    base.update(kw)
    return TrainConfig(**base)


def test_binary_backward_compat_no_backup(tmp_path):
    """Regression guard: binary run without backup args writes no backups and
    returns the normal result shape."""
    from bittrainer.trainer import run_training

    _make_concept(tmp_path / "c", n_pos=8, n_neg=8, n_val=4)
    _seed(0)
    r = run_training(_bin_cfg(tmp_path / "c", tmp_path / "ck", max_epochs=2))
    assert "paused" not in r
    assert r["checkpoint_path"].endswith("best.pt")
    assert not list(tmp_path.glob("**/backup_*.pt"))


def test_binary_pause_resume_across_unfreeze_boundary(tmp_path):
    """Regression guard: pausing at the epoch-0→1 boundary and resuming replays
    the non-gradual unfreeze reconstruction (fresh optimizer/scheduler) so the
    param_groups match — the run resumes into epoch 1 and completes, no crash."""
    from bittrainer.trainer import run_training

    _make_concept(tmp_path / "c", n_pos=55, n_neg=55, n_val=10)  # >=50 -> non-gradual

    # Pause at the top of epoch index 1 (the unfreeze boundary).
    pause, _cb = _pause_at_epoch(1)

    _seed(0)
    first = run_training(
        _bin_cfg(tmp_path / "c", tmp_path / "ck", max_epochs=3, backup_dir=str(tmp_path / "ck" / "backups")),
        progress_callback=_cb, pause_event=pause,
    )
    assert first["paused"] is True
    assert first["epoch"] == 1  # paused entering the unfreeze epoch

    _seed(0)
    resumed = run_training(
        _bin_cfg(
            tmp_path / "c", tmp_path / "ck", max_epochs=3,
            backup_dir=str(tmp_path / "ck" / "backups"),
            resume_from=str(tmp_path / "ck" / "backups"),
        ),
    )
    assert "paused" not in resumed
    assert resumed["checkpoint_path"].endswith("best.pt")
    assert resumed["epochs_completed"] == 3


# ---------------------------------------------------------------------------
# Multi-head + dual-branch mirror trainers: pause + epoch-restart resume
# ---------------------------------------------------------------------------


def _make_labelled_group(root, classes, *, per_class=6, seed=0):
    rng = np.random.default_rng(seed)
    for split, n in (("train", per_class), ("val", max(2, per_class // 2))):
        for cname in classes:
            d = root / cname / split
            d.mkdir(parents=True, exist_ok=True)
            for j in range(n):
                Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
                    d / f"{cname}_{j}.png"
                )


def test_multihead_pause_and_resume_completes(tmp_path):
    """Regression guard: multi-head trainer pauses at an epoch boundary, backs up,
    and a resume rebuilds + restarts the epoch, completing without a crash."""
    from bittrainer.multihead_trainer import MultiHeadTrainConfig, run_multihead_training

    sizes = ["__none__", "34A", "34B", "36A", "36B"]
    _make_labelled_group(tmp_path / "grp", sizes, per_class=6, seed=0)

    def _cfg(**kw):
        return MultiHeadTrainConfig(
            group_folder=str(tmp_path / "grp"), size_classes=sizes,
            checkpoint_dir=str(tmp_path / "ck"), max_epochs=3, patience=99,
            backbone_variant="atto", device="cpu", dtype="float32", batch_size=4,
            use_compile=False, channels_last=False, from_scratch=True, dataloader_workers=0,
            backbone_init={"source": "random_init", "checkpoint_path": None}, **kw,
        )

    pause, _cb = _pause_at_epoch(1)

    _seed(0)
    first = run_multihead_training(
        _cfg(backup_dir=str(tmp_path / "ck" / "backups")),
        progress_callback=_cb, pause_event=pause,
    )
    assert first["paused"] is True

    _seed(0)
    resumed = run_multihead_training(
        _cfg(backup_dir=str(tmp_path / "ck" / "backups"), resume_from=str(tmp_path / "ck" / "backups")),
    )
    assert "paused" not in resumed
    assert resumed["checkpoint_path"] is not None
    assert resumed["epochs_completed"] == 3


def test_dual_branch_pause_and_resume_completes(tmp_path):
    """Regression guard: dual-branch trainer pauses at an epoch boundary, backs
    up, and a resume rebuilds + restarts the epoch, completing without a crash."""
    from bittrainer.dual_branch_trainer import DualBranchTrainConfig, run_dual_branch_training

    classes = ["a", "b", "c"]
    _make_labelled_group(tmp_path / "crops", classes, per_class=6, seed=0)
    _make_labelled_group(tmp_path / "context", classes, per_class=6, seed=0)

    def _cfg(**kw):
        return DualBranchTrainConfig(
            group_folder=str(tmp_path / "crops"), context_folder=str(tmp_path / "context"),
            num_classes=3, class_names=classes, checkpoint_dir=str(tmp_path / "ck"),
            max_epochs=3, patience=99, backbone_variant="atto", device="cpu", dtype="float32",
            batch_size=4, use_compile=False, channels_last=False, from_scratch=True,
            dataloader_workers=0, backbone_init={"source": "random_init", "checkpoint_path": None}, **kw,
        )

    pause, _cb = _pause_at_epoch(1)

    _seed(0)
    first = run_dual_branch_training(
        _cfg(backup_dir=str(tmp_path / "ck" / "backups")),
        progress_callback=_cb, pause_event=pause,
    )
    assert first["paused"] is True

    _seed(0)
    resumed = run_dual_branch_training(
        _cfg(backup_dir=str(tmp_path / "ck" / "backups"), resume_from=str(tmp_path / "ck" / "backups")),
    )
    assert "paused" not in resumed
    assert resumed["checkpoint_path"] is not None
    assert resumed["epochs_completed"] == 3
