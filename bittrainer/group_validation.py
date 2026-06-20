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
            "macro_precision": 0.0,
            "macro_recall": 0.0,
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
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0, labels=class_labels))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0, labels=class_labels))

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
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
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


def compute_none_metrics(
    labels: list[int],
    predictions: list[int],
    num_classes: int,
    *,
    none_index: int = -1,
) -> dict:
    """Metrics for the absence/``__none__`` class.

    ``none_false_positive_rate`` is the fraction of true-``__none__`` samples
    predicted as any real class, which is the hallucination shape we care about
    for open-world group models.
    """
    if not (0 <= none_index < num_classes) or len(labels) == 0:
        return {
            "none_precision": None,
            "none_recall": None,
            "none_f1": None,
            "none_false_positive_rate": None,
            "none_support": 0,
        }

    y_true = np.array(labels)
    y_pred = np.array(predictions)
    true_none = y_true == none_index
    pred_none = y_pred == none_index
    tp = int(np.sum(true_none & pred_none))
    fp = int(np.sum(~true_none & pred_none))
    fn = int(np.sum(true_none & ~pred_none))
    support = int(np.sum(true_none))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if support > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    false_positive_rate = fn / support if support > 0 else 0.0

    return {
        "none_precision": float(precision),
        "none_recall": float(recall),
        "none_f1": float(f1),
        "none_false_positive_rate": float(false_positive_rate),
        "none_support": support,
    }


def find_per_class_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    grid: np.ndarray | None = None,
    min_positive: int = 1,
) -> np.ndarray:
    """Pick per-class binarisation thresholds that maximise per-class F1 on val.

    ``probs`` and ``labels`` are both ``[N, num_classes]`` arrays. Sweeps a
    coarse threshold grid for each class independently, returning a
    ``[num_classes]`` vector of thresholds.

    Classes with fewer than ``min_positive`` positives in ``labels`` keep the
    default 0.5 — the F1 surface is unstable without positive support.
    """
    if grid is None:
        grid = np.arange(0.05, 0.95, 0.025)

    num_classes = probs.shape[1]
    thresholds = np.full(num_classes, 0.5, dtype=np.float64)

    for c in range(num_classes):
        y_true = labels[:, c]
        y_prob = probs[:, c]
        if int(y_true.sum()) < min_positive:
            continue

        best_f1 = -1.0
        best_t = 0.5
        for t in grid:
            y_pred = (y_prob >= t).astype(np.int64)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = float(f1)
                best_t = float(t)
        thresholds[c] = best_t

    return thresholds


def compute_multilabel_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
    *,
    thresholds: np.ndarray | None = None,
    probs: np.ndarray | None = None,
) -> dict:
    """Compute multi-label metrics.

    Either pass binarised ``predictions`` (legacy 0.5-threshold call site) or
    pass raw ``probs`` plus a ``thresholds`` vector; in the latter case the
    binarisation happens here using the per-class thresholds.
    """
    if probs is not None and thresholds is not None:
        predictions = (probs >= thresholds[None, :]).astype(np.int64)

    if labels.shape[0] == 0:
        return {
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "per_class_f1": {},
            "per_class_precision": {},
            "per_class_recall": {},
            "hamming_loss": 0.0,
            "exact_match_ratio": 0.0,
            "thresholds": (thresholds.tolist() if thresholds is not None else [0.5] * num_classes),
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
    macro_precision = float(np.mean(per_prec))
    macro_recall = float(np.mean(per_rec))

    per_class_f1 = {str(i): float(per_f1[i]) for i in range(num_classes)}
    per_class_precision = {str(i): float(per_prec[i]) for i in range(num_classes)}
    per_class_recall = {str(i): float(per_rec[i]) for i in range(num_classes)}

    # Hamming loss: fraction of wrong individual labels
    hamming = float(np.mean(labels != predictions))

    # Exact match ratio: fraction of samples where all labels match exactly
    exact_match = float(np.mean(np.all(labels == predictions, axis=1)))

    return {
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "hamming_loss": hamming,
        "exact_match_ratio": exact_match,
        "thresholds": (thresholds.tolist() if thresholds is not None else [0.5] * num_classes),
    }
