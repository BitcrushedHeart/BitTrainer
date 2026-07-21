"""weighted_f1 / micro_f1 additions to compute_multiclass_metrics (ISSUE-0490 B).

micro-F1 for single-label exclusive predictions equals accuracy; weighted-F1 is
the support-weighted mean of per-class F1 and diverges from macro-F1 under class
imbalance. Both are exact against hand-computed values on a small imbalanced set.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from bittrainer.group_validation import compute_multiclass_metrics


def test_micro_f1_equals_accuracy_single_label() -> None:
    # 3 classes, imbalanced. Single-label predictions => micro-F1 == accuracy.
    labels = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2]
    preds = [0, 0, 0, 0, 0, 1, 1, 2, 2, 0]
    m = compute_multiclass_metrics(labels, preds, num_classes=3)
    accuracy = sum(int(a == b) for a, b in zip(labels, preds)) / len(labels)
    assert m["micro_f1"] == accuracy


def test_weighted_f1_matches_sklearn_and_differs_from_macro() -> None:
    labels = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2]
    preds = [0, 0, 0, 0, 0, 1, 1, 2, 2, 0]
    m = compute_multiclass_metrics(labels, preds, num_classes=3)

    expected_weighted = float(
        f1_score(
            np.array(labels), np.array(preds),
            average="weighted", zero_division=0, labels=[0, 1, 2],
        )
    )
    expected_micro = float(
        f1_score(
            np.array(labels), np.array(preds),
            average="micro", zero_division=0, labels=[0, 1, 2],
        )
    )
    assert m["weighted_f1"] == expected_weighted
    assert m["micro_f1"] == expected_micro
    # Under this imbalance the two aggregations must genuinely disagree.
    assert abs(m["weighted_f1"] - m["macro_f1"]) > 1e-6


def test_empty_input_defines_new_keys() -> None:
    m = compute_multiclass_metrics([], [], num_classes=3)
    assert m["weighted_f1"] == 0.0
    assert m["micro_f1"] == 0.0
