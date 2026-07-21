"""Regression: when the incumbent is kept, every reported summary scalar must
reflect the kept model — not just the ordinal/non-ordinal selection metric.

Previously an ordinal keep synced best_val_qwk to the incumbent but left
best_val_macro_f1 at the losing candidate's value, so the group summary card
showed a different Macro F1 (candidate's) than the training-history row
(incumbent's). Pinned here.
"""

from __future__ import annotations

import torch

import bittrainer.finalize as finalize  # ISSUE-0542: seams read here now
from bittrainer.group_trainer import GroupTrainConfig, _compare_promote_finalize

_CLASSES = ["0", "1", "2", "__none__"]
_INCUMBENT = {
    "macro_f1": 0.441, "qwk": 0.547, "val_loss": 0.798,
    "adjacent_accuracy": 0.955, "ordinal_mae": 0.3,
    "macro_precision": 0.5, "macro_recall": 0.42,
    "per_class_f1": {"0": 0.6}, "per_class_precision": {}, "per_class_recall": {},
}


def test_ordinal_keep_syncs_all_summary_metrics(tmp_path, monkeypatch):
    ckpt_dir = tmp_path
    # A real (metadata-only) incumbent so the class-name/count read succeeds.
    torch.save(
        {"class_names": _CLASSES, "num_classes": 4, "model_size": "nano"},
        ckpt_dir / "best.pt",
    )
    torch.save({"state_dict": {}}, ckpt_dir / "candidate.pt")

    # Incumbent evaluates higher on QWK than the candidate → incumbent kept.
    # Single-label incumbents are scored via _collect_val_logits +
    # _incumbent_decode_metrics (the shipped-decode fair comparison); mock that
    # seam — this test pins summary-metric syncing, not the decode itself.
    monkeypatch.setattr(finalize, "load_checkpoint", lambda *a, **k: torch.nn.Identity())

    calls = {"n": 0}

    def _fake_collect(*_a, **_k):
        # First call = incumbent fair-comparison pass (succeeds); the later
        # finalisation-calibration pass fails like the real val_loader=None
        # would, so the kept incumbent's metrics survive to the result dict.
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("No validation logits available for calibration")
        return torch.zeros(1, 4), torch.zeros(1, dtype=torch.long)

    monkeypatch.setattr(finalize, "_collect_val_logits", _fake_collect)
    monkeypatch.setattr(finalize, "_incumbent_decode_metrics", lambda *a, **k: dict(_INCUMBENT))

    cfg = GroupTrainConfig(
        group_folder=str(tmp_path), num_classes=4, class_names=_CLASSES,
        ordinal=True, best_model_name="best.pt",
    )

    result = _compare_promote_finalize(
        cfg,
        candidate_path=str(ckpt_dir / "candidate.pt"),
        best_metrics={"macro_f1": 0.361, "qwk": 0.50, "val_loss": 1.1},  # losing candidate
        candidate_macro_f1=0.361,
        candidate_qwk=0.50,
        best_epoch_display=4,
        epochs_completed=6,
        val_loader=None,  # _evaluate is mocked
        device=torch.device("cpu"),
        dtype=torch.float32,
        checkpoint_dir=ckpt_dir,
        class_counts={0: 10},
        total_raw=10,
    )

    assert result["promotion_reason"] == "incumbent_wins"
    # the kept model's F1 — NOT the candidate's 0.361 — in BOTH fields
    assert result["best_val_macro_f1"] == 0.441
    assert result["final_val_macro_f1"] == 0.441
    assert result["best_val_qwk"] == 0.547
    assert result["final_val_loss"] == 0.798
    # candidate.pt is discarded; best.pt stays the deployed model
    assert result["checkpoint_path"].endswith("best.pt")
    assert not (ckpt_dir / "candidate.pt").exists()
