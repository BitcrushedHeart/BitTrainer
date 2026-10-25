"""A kept incumbent leaves the run byte-identical (Bitcrush ISSUE-1501).

When the candidate loses, finalisation used to re-calibrate the incumbent and
rewrite its checkpoint (priors, temperature, none-bias, cut points, val probs,
multi-label thresholds, region views). The file hash is the suite's model
version, so every kept run threw away the suite's cached scores for a model
that had not changed. Now nothing writes to a kept checkpoint, and the result
reports the incumbent's own decode on today's split plus its stored calibration.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch

import bittrainer.finalize as finalize
from bittrainer.generic.tasks import group_task, head_only_task
from bittrainer.group_trainer import GroupTrainConfig, _compare_promote_finalize

_CLASSES = ["__none__", "0", "1", "2"]
_INCUMBENT_METRICS = {
    "macro_f1": 0.70, "qwk": 0.90, "val_loss": 0.4, "none_f1": 0.6,
    "adjacent_accuracy": 0.98, "ordinal_decode": "cut_points",
}
_CANDIDATE_METRICS = {"macro_f1": 0.50, "qwk": 0.60, "val_loss": 1.1, "none_f1": 0.3}


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finalize(cfg, ckpt_dir, **kw):
    return _compare_promote_finalize(
        cfg,
        candidate_path=str(ckpt_dir / "candidate.pt"),
        best_metrics=dict(_CANDIDATE_METRICS),
        candidate_macro_f1=0.50,
        candidate_qwk=0.60,
        best_epoch_display=4,
        epochs_completed=6,
        val_loader=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        checkpoint_dir=ckpt_dir,
        class_counts={0: 5, 1: 10, 2: 10, 3: 10},
        total_raw=35,
        **kw,
    )


def test_a_kept_single_label_incumbent_is_never_rewritten(tmp_path, monkeypatch):
    torch.save(
        {
            "class_names": _CLASSES, "num_classes": 4, "model_size": "nano",
            "temperature": 1.3, "class_logit_bias": [0.1, 0.0, 0.0, 0.0],
            "ordinal_cut_points": [0.5, 1.5, 2.5],
        },
        tmp_path / "best.pt",
    )
    torch.save({"state_dict": {}}, tmp_path / "candidate.pt")
    before = _digest(tmp_path / "best.pt")

    monkeypatch.setattr(finalize, "load_checkpoint", lambda *a, **k: torch.nn.Identity())
    passes = {"n": 0}

    def _collect(*_a, **_k):
        passes["n"] += 1
        return torch.zeros(4, 4), torch.zeros(4, dtype=torch.long)

    monkeypatch.setattr(finalize, "_collect_val_logits", _collect)
    monkeypatch.setattr(
        finalize, "_incumbent_decode_metrics", lambda *a, **k: dict(_INCUMBENT_METRICS)
    )

    def _no_write(*_a, **_k):
        raise AssertionError("a kept incumbent's checkpoint must not be written")

    monkeypatch.setattr(finalize, "_apply_and_persist_priors", _no_write)
    monkeypatch.setattr(finalize, "_persist_strictness_val_data", _no_write)

    cfg = GroupTrainConfig(
        group_folder=str(tmp_path), num_classes=4, class_names=_CLASSES,
        ordinal=True, best_model_name="best.pt",
    )
    result = _finalize(cfg, tmp_path)

    assert result["promotion_reason"] == "incumbent_wins"
    assert _digest(tmp_path / "best.pt") == before
    assert not (tmp_path / "candidate.pt").exists()
    assert passes["n"] == 1  # the comparison pass only; no calibration pass
    # The incumbent's own decode on today's split, under its stored calibration.
    assert result["best_val_qwk"] == pytest.approx(0.90)
    assert result["final_val_macro_f1"] == pytest.approx(0.70)
    assert result["calibration_temperature"] == pytest.approx(1.3)
    assert result["none_logit_bias"] == pytest.approx(0.1)
    assert result["ordinal_cut_points"] == [0.5, 1.5, 2.5]
    assert "ordinal_head" not in result  # a softmax incumbent


def test_a_kept_multi_label_incumbent_keeps_its_thresholds(tmp_path, monkeypatch):
    names = ["a", "b"]
    torch.save(
        {
            "class_names": names, "num_classes": 2, "model_size": "nano",
            "multi_label": True, "per_class_thresholds": [0.3, 0.7],
        },
        tmp_path / "best.pt",
    )
    torch.save({"state_dict": {}}, tmp_path / "candidate.pt")
    before = _digest(tmp_path / "best.pt")

    monkeypatch.setattr(finalize, "load_checkpoint", lambda *a, **k: torch.nn.Identity())
    probs = torch.tensor([[0.9, 0.1], [0.2, 0.8]]).numpy()
    labels = torch.tensor([[1, 0], [0, 1]]).numpy()
    monkeypatch.setattr(
        finalize,
        "_evaluate",
        lambda *a, **k: {"macro_f1": 0.95, "val_loss": 0.2, "_probs": probs, "_labels": labels},
    )

    def _no_tuning(*_a, **_k):
        raise AssertionError("a kept incumbent's thresholds must not be re-tuned")

    monkeypatch.setattr(finalize, "find_per_class_thresholds", _no_tuning)
    monkeypatch.setattr(
        finalize,
        "_persist_strictness_val_data",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no strictness write")),
    )

    cfg = GroupTrainConfig(
        group_folder=str(tmp_path), num_classes=2, class_names=names,
        multi_label=True, best_model_name="best.pt",
    )
    result = _finalize(cfg, tmp_path)

    assert result["promotion_reason"] == "incumbent_wins"
    assert _digest(tmp_path / "best.pt") == before
    assert result["per_class_thresholds"] == [0.3, 0.7]
    # Scored under its own 0.3 / 0.7 thresholds (perfect on these rows), not
    # the 0.95 the default-threshold pass reported.
    assert result["best_val_macro_f1"] == pytest.approx(1.0)
    assert "_probs" not in result


@pytest.mark.parametrize("task_cls", [head_only_task.HeadOnlyTask, group_task.GroupTask])
def test_a_kept_incumbent_skips_region_view_finalisation(tmp_path, monkeypatch, task_cls):
    module = head_only_task if task_cls is head_only_task.HeadOnlyTask else group_task

    def _no_regions(*_a, **_k):
        raise AssertionError("a kept incumbent's region views must not be re-fit")

    monkeypatch.setattr(module, "finalize_region_views", _no_regions)
    cfg = GroupTrainConfig(group_folder=str(tmp_path), num_classes=4, class_names=_CLASSES)
    task = task_cls(cfg)
    task.region_manifest = SimpleNamespace(type_names=["face"])
    result = {"promotion_reason": "incumbent_wins", "checkpoint_path": str(tmp_path / "best.pt")}

    task._finalize_regions(SimpleNamespace(device="cpu", dtype=torch.float32, cb=None), result)

    assert result["region_training"] == {"status": "incumbent_kept"}
