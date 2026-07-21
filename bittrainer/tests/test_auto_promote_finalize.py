"""Auto-Promote short-circuits the incumbent comparison in _compare_promote_finalize.

Companion to the pure ``test_auto_promote`` (which pins ``decide_promotion``):
this drives the trainer seam and proves that with ``auto_promote=True`` the
incumbent is NEVER loaded or scored, yet the candidate is still promoted to
best.pt. Imports torch (checkpoint IO), so it lives with the other trainer tests.
"""

from __future__ import annotations

import torch

import bittrainer.finalize as finalize  # ISSUE-0542: seams read here now
from bittrainer.group_trainer import GroupTrainConfig, _compare_promote_finalize

_CLASSES = ["0", "1", "2", "__none__"]


def test_auto_promote_skips_incumbent_load_and_promotes(tmp_path, monkeypatch):
    ckpt_dir = tmp_path
    # A perfectly loadable, class-compatible incumbent that WOULD win a fair
    # comparison — auto_promote must ignore it without even loading it.
    torch.save(
        {"class_names": _CLASSES, "num_classes": 4, "model_size": "nano"},
        ckpt_dir / "best.pt",
    )
    torch.save({"state_dict": {}}, ckpt_dir / "candidate.pt")

    def _boom_load(*_a, **_k):
        raise AssertionError("incumbent must NOT be loaded when auto_promote is on")

    # If the short-circuit works, neither the incumbent load nor its scoring runs.
    monkeypatch.setattr(finalize, "load_checkpoint", _boom_load)
    # The later finalisation-calibration pass has no real val_loader; let it fail
    # exactly like val_loader=None would, so the promoted candidate survives.
    monkeypatch.setattr(
        finalize,
        "_collect_val_logits",
        lambda *_a, **_k: (_ for _ in ()).throw(
            RuntimeError("No validation logits available for calibration")
        ),
    )

    cfg = GroupTrainConfig(
        group_folder=str(tmp_path), num_classes=4, class_names=_CLASSES,
        ordinal=True, best_model_name="best.pt", auto_promote=True,
    )

    result = _compare_promote_finalize(
        cfg,
        candidate_path=str(ckpt_dir / "candidate.pt"),
        best_metrics={"macro_f1": 0.10, "qwk": 0.10, "val_loss": 2.0},  # weak candidate
        candidate_macro_f1=0.10,
        candidate_qwk=0.10,
        best_epoch_display=4,
        epochs_completed=6,
        val_loader=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        checkpoint_dir=ckpt_dir,
        class_counts={0: 10},
        total_raw=10,
    )

    assert result["promotion_reason"] == "auto_promote"
    # The weak candidate shipped as best.pt; candidate.pt was moved into place.
    assert result["checkpoint_path"].endswith("best.pt")
    assert not (ckpt_dir / "candidate.pt").exists()
