"""Validation metrics for multi-class and multi-label classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_multiclass_metrics(
    labels: list[int],
    predictions: list[int],
    num_classes: int,
) -> dict:
    """Compute per-class and macro metrics for multi-class classification.

    Returns dict with macro_f1, per_class_f1, per_class_precision,
    per_class_recall, confusion_matrix, balanced_accuracy.
    """
    if len(labels) == 0:
        return {
            "macro_f1": 0.0,
            "per_class_f1": {},
            "per_class_precision": {},
            "per_class_recall": {},
            "confusion_matrix": [],
            "balanced_accuracy": 0.0,
        }

    y_true = np.array(labels)
    y_pred = np.array(predictions)
    class_labels = list(range(num_classes))

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=class_labels))

    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=class_labels)
    per_prec = precision_score(y_true, y_pred, average=None, zero_division=0, labels=class_labels)
    per_rec = recall_score(y_true, y_pred, average=None, zero_division=0, labels=class_labels)

    per_class_f1 = {str(i): float(per_f1[i]) for i in range(num_classes)}
    per_class_precision = {str(i): float(per_prec[i]) for i in range(num_classes)}
    per_class_recall = {str(i): float(per_rec[i]) for i in range(num_classes)}

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    # Balanced accuracy = mean of per-class recall
    balanced_acc = float(np.mean(per_rec))

    return {
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm.tolist(),
        "balanced_accuracy": balanced_acc,
    }


def compute_ordinal_metrics(
    labels: list[int],
    predictions: list[int],
    num_classes: int,
    *,
    none_index: int = -1,
) -> dict:
    """Compute QWK, MAE and adjacent-accuracy over the ordinal scale only.

    Samples whose true OR predicted class is ``none_index`` (the ``__none__``
    class) are excluded — ``__none__`` is a separate semantic category, not a
    position on the ordinal scale, so treating its index distance as ordinal
    error pollutes the metrics. ``__none__`` recall/precision is still
    captured by ``compute_multiclass_metrics``.
    """
    y_true = np.array(labels)
    y_pred = np.array(predictions)

    if 0 <= none_index < num_classes:
        mask = (y_true != none_index) & (y_pred != none_index)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        ordinal_labels = [i for i in range(num_classes) if i != none_index]
    else:
        ordinal_labels = list(range(num_classes))

    if len(y_true) == 0:
        return {"qwk": 0.0, "ordinal_mae": 0.0, "adjacent_accuracy": 0.0}

    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=ordinal_labels))
    ordinal_mae = float(np.mean(np.abs(y_true - y_pred)))
    adjacent_accuracy = float(np.mean(np.abs(y_true - y_pred) <= 1))

    return {
        "qwk": qwk,
        "ordinal_mae": ordinal_mae,
        "adjacent_accuracy": adjacent_accuracy,
    }


def compute_multilabel_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> dict:
    if labels.shape[0] == 0:
        return {
            "macro_f1": 0.0,
            "per_class_f1": {},
            "per_class_precision": {},
            "per_class_recall": {},
            "hamming_loss": 0.0,
            "exact_match_ratio": 0.0,
        }

    per_f1 = f1_score(labels, predictions, average=None, zero_division=0)
    per_prec = precision_score(labels, predictions, average=None, zero_division=0)
    per_rec = recall_score(labels, predictions, average=None, zero_division=0)

    # Pad if sklearn dropped trailing all-zero classes
    def _pad(arr: np.ndarray) -> np.ndarray:
        if len(arr) < num_classes:
            return np.pad(arr, (0, num_classes - len(arr)), constant_values=0.0)
        return arr

    per_f1 = _pad(per_f1)
    per_prec = _pad(per_prec)
    per_rec = _pad(per_rec)

    macro_f1 = float(np.mean(per_f1))

    per_class_f1 = {str(i): float(per_f1[i]) for i in range(num_classes)}
    per_class_precision = {str(i): float(per_prec[i]) for i in range(num_classes)}
    per_class_recall = {str(i): float(per_rec[i]) for i in range(num_classes)}

    # Hamming loss: fraction of wrong individual labels
    hamming = float(np.mean(labels != predictions))

    # Exact match ratio: fraction of samples where all labels match exactly
    exact_match = float(np.mean(np.all(labels == predictions, axis=1)))

    return {
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "hamming_loss": hamming,
        "exact_match_ratio": exact_match,
    }
