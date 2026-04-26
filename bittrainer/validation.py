"""Validation metrics for binary classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def compute_metrics(
    labels: list[int],
    probs: list[float],
    *,
    threshold: float = 0.5,
) -> dict:
    """Compute F1, precision, recall, AUPRC, and confusion matrix."""
    if len(labels) == 0:
        return {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "auprc": 0.0,
            "confusion_matrix": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        }

    y_true = np.array(labels)
    y_prob = np.array(probs)
    y_pred = (y_prob >= threshold).astype(int)

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    # AUPRC — needs at least one positive
    if y_true.sum() > 0:
        auprc = float(average_precision_score(y_true, y_prob))
    else:
        auprc = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "auprc": auprc,
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }


def find_optimal_threshold(
    labels: list[int],
    probs: list[float],
) -> float:
    """Find the threshold that maximises F1 on the given labels/probs."""
    if len(labels) == 0 or sum(labels) == 0:
        return 0.5

    y_true = np.array(labels)
    y_prob = np.array(probs)

    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_prob)

    # F1 = 2 * (precision * recall) / (precision + recall)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = 2 * (precision_arr[:-1] * recall_arr[:-1]) / (precision_arr[:-1] + recall_arr[:-1])
        f1_scores = np.nan_to_num(f1_scores, 0.0)

    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])
