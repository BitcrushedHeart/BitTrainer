"""Backbone selection metrics: per-head F1/AP instead of mean raw accuracy.

Trainer-parity round, workstream A1. ``bb._evaluate`` historically returned
``{binary/<c>: accuracy@logit>0, group/<g>: argmax accuracy}`` and the task
selected on the blind mean — saturating on imbalanced heads and dominated by
easy ones. The new contract is a pure metrics helper computing per-binary-head
F1 + average precision from collected val logits, per-group macro-F1, and a
defensible ``selection`` aggregate; zero-support heads are excluded from the
aggregate (never frozen zeros) and reported via their support key. All legacy
accuracy keys remain (additive wire contract).

Pure tests over crafted logits — no images, no training run.
"""

from __future__ import annotations

import numpy as np
import pytest

import bittrainer.backbone_trainer as bb
from bittrainer.generic.tasks.backbone_task import BackboneTask


def _metrics_fn():
    fn = getattr(bb, "_backbone_metrics", None)
    assert fn is not None, (
        "bb._backbone_metrics is missing (workstream A1 not implemented)"
    )
    return fn


def _binary(logits, targets):
    return (np.asarray(logits, dtype=np.float64), np.asarray(targets, dtype=np.float64))


def _group(logits, targets):
    return (np.asarray(logits, dtype=np.float64), np.asarray(targets, dtype=np.int64))


def test_perfect_binary_head_scores_f1_and_ap_one():
    metrics = _metrics_fn()(
        {"watermark": _binary([3.0, 2.0, -2.0, -3.0], [1, 1, 0, 0])}, {}
    )
    assert metrics["binary_f1/watermark"] == pytest.approx(1.0)
    assert metrics["binary_ap/watermark"] == pytest.approx(1.0)
    assert metrics["binary_support/watermark"] == 2
    assert metrics["binary_macro_f1"] == pytest.approx(1.0)
    assert metrics["selection"] == pytest.approx(1.0)
    # Legacy accuracy key preserved byte-for-byte (additive contract).
    assert metrics["binary/watermark"] == pytest.approx(1.0)


def test_binary_f1_matches_sklearn_at_logit_zero_threshold():
    from sklearn.metrics import average_precision_score, f1_score

    logits = np.array([1.5, 0.5, -0.5, 2.0, -1.0, -2.0])
    targets = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    metrics = _metrics_fn()({"c": _binary(logits, targets)}, {})
    preds = (logits > 0).astype(int)
    assert metrics["binary_f1/c"] == pytest.approx(
        f1_score(targets.astype(int), preds, zero_division=0)
    )
    probs = 1.0 / (1.0 + np.exp(-logits))
    assert metrics["binary_ap/c"] == pytest.approx(
        average_precision_score(targets.astype(int), probs)
    )


def test_accuracy_saturates_but_f1_does_not():
    """The motivating failure: 90% majority-negative head where the model
    predicts all-negative. Accuracy says 0.9; F1 says 0.0 — selection must see
    the 0.0, not the flattering accuracy."""
    logits = np.full(10, -2.0)
    targets = np.array([1.0] + [0.0] * 9)
    metrics = _metrics_fn()({"c": _binary(logits, targets)}, {})
    assert metrics["binary/c"] == pytest.approx(0.9)  # legacy accuracy retained
    assert metrics["binary_f1/c"] == pytest.approx(0.0)
    assert metrics["binary_macro_f1"] == pytest.approx(0.0)


def test_zero_support_head_is_excluded_from_aggregate():
    """A head with no val positives cannot have an F1; it must be excluded
    from the aggregate (support reported), not counted as a frozen zero."""
    metrics = _metrics_fn()(
        {
            "good": _binary([3.0, -3.0], [1, 0]),
            "empty": _binary([-1.0, -2.0, -3.0], [0, 0, 0]),
        },
        {},
    )
    assert metrics["binary_support/empty"] == 0
    assert "binary_f1/empty" not in metrics
    assert "binary_ap/empty" not in metrics
    # Aggregate over supported heads only: just "good" -> 1.0, not 0.5.
    assert metrics["binary_macro_f1"] == pytest.approx(1.0)


def test_group_macro_f1_and_aggregate():
    # 3-class group, one class never predicted. preds = [0, 1, 0, 0, 1] vs
    # targets [0, 1, 2, 0, 1]: the class-2 miss is a class-0 FALSE POSITIVE,
    # so per-class F1 = {0: 0.8, 1: 1.0, 2: 0.0} -> macro-F1 = 0.6 (NOT the
    # mean recall 2/3 — macro-F1 charges the miss to class 0's precision too).
    logits = np.array(
        [
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [4.0, 0.0, 0.0],  # true class 2 predicted as 0
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
        ]
    )
    targets = np.array([0, 1, 2, 0, 1])
    metrics = _metrics_fn()({}, {"shot_type": _group(logits, targets)})
    assert metrics["group_f1/shot_type"] == pytest.approx(0.6, abs=1e-6)
    assert metrics["group_val_support/shot_type"] == 5
    assert metrics["group_macro_f1"] == pytest.approx(0.6, abs=1e-6)
    # Legacy argmax accuracy key preserved: 4/5 correct.
    assert metrics["group/shot_type"] == pytest.approx(0.8)


def test_selection_is_mean_of_binary_and_group_panels():
    metrics = _metrics_fn()(
        {"c": _binary([3.0, -3.0], [1, 0])},  # binary panel = 1.0
        {"g": _group([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 0.0]], [0, 1, 1, 0])},
    )
    # group: class0 F1=0.8 (p=2/3, r=1), class1 F1=2/3 (p=1, r=1/2) -> macro ~0.7333
    expected_group = (0.8 + 2.0 / 3.0) / 2.0
    assert metrics["group_macro_f1"] == pytest.approx(expected_group, abs=1e-6)
    assert metrics["selection"] == pytest.approx((1.0 + expected_group) / 2.0, abs=1e-6)


def test_selection_with_only_one_panel_present():
    only_binary = _metrics_fn()({"c": _binary([3.0, -3.0], [1, 0])}, {})
    assert only_binary["selection"] == pytest.approx(1.0)
    assert "group_macro_f1" not in only_binary
    empty = _metrics_fn()({}, {})
    assert empty["selection"] == pytest.approx(0.0)


def _tiny_request() -> dict:
    return {
        "run_id": "run_sel",
        "convnextv2_size": "atto",
        "candidate_checkpoint_path": "unused/candidate.safetensors",
        "records": [
            {"binary": {"watermark": "positive"}, "file_paths": [], "groups": {}},
            {"binary": {"watermark": "negative"}, "file_paths": [], "groups": {}},
        ],
        "training_config": {"device": "cpu"},
        "heads": {},
    }


def test_selection_score_prefers_selection_key():
    task = BackboneTask(_tiny_request())
    assert task.selection_score(
        {"selection": 0.42, "binary_f1/x": 0.9}
    ) == pytest.approx(0.42)


def test_selection_score_falls_back_to_legacy_mean():
    """Fakes in existing tests monkeypatch bb._evaluate to return bare
    accuracy dicts — the fallback must keep scoring them as the plain mean."""
    task = BackboneTask(_tiny_request())
    assert task.selection_score({"binary/watermark": 0.9}) == pytest.approx(0.9)
    assert task.selection_score({}) == pytest.approx(0.0)


def test_evaluate_accepts_score_sink_kwarg():
    """bb._evaluate must grow a score_sink kwarg (calibration reads the raw
    logits of the winning epoch through it)."""
    import inspect

    assert "score_sink" in inspect.signature(bb._evaluate).parameters
