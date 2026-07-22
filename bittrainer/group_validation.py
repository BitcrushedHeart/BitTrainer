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
            "weighted_f1": 0.0,
            "micro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "per_class_f1": {},
            "per_class_precision": {},
            "per_class_recall": {},
            "per_class_support": {},
            "confusion_matrix": [],
            "balanced_accuracy": 0.0,
        }

    y_true = np.array(labels)
    y_pred = np.array(predictions)
    class_labels = list(range(num_classes))

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=class_labels))
    # weighted_f1: support-weighted mean of per-class F1 (diverges from macro
    # under imbalance); micro_f1: global TP/FP/FN pool (== accuracy for
    # single-label exclusive predictions). Both surface the "actual spread"
    # number balanced-training macro selection hides (ISSUE-0490 B).
    weighted_f1 = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=class_labels)
    )
    micro_f1 = float(
        f1_score(y_true, y_pred, average="micro", zero_division=0, labels=class_labels)
    )
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0, labels=class_labels))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0, labels=class_labels))

    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=class_labels)
    per_prec = precision_score(y_true, y_pred, average=None, zero_division=0, labels=class_labels)
    per_rec = recall_score(y_true, y_pred, average=None, zero_division=0, labels=class_labels)

    per_class_f1 = {str(i): float(per_f1[i]) for i in range(num_classes)}
    per_class_precision = {str(i): float(per_prec[i]) for i in range(num_classes)}
    per_class_recall = {str(i): float(per_rec[i]) for i in range(num_classes)}

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    # True-label counts per class (CM row sums) — the denominator context the
    # per-class F1 numbers are meaningless without.
    per_class_support = {str(i): int(cm[i].sum()) for i in range(num_classes)}

    # Balanced accuracy = mean of per-class recall
    balanced_acc = float(np.mean(per_rec))

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_support": per_class_support,
        "confusion_matrix": cm.tolist(),
        "balanced_accuracy": balanced_acc,
    }


def macro_f1_variants(
    per_class_f1: dict,
    per_class_support: dict,
    num_classes: int,
    none_index: int = -1,
) -> dict:
    """Report-only macro-F1 variants over filtered class subsets.

    The raw macro-F1 averages every defined class, so classes with zero
    validation support each contribute a permanent 0 and cap the metric below
    1.0 regardless of model quality. These variants make the honest number
    visible without changing selection:

    - ``macro_f1_supported``: mean over classes with val support > 0.
    - ``macro_f1_excl_none``: mean over real (non-``__none__``) classes.
    - ``macro_f1_supported_excl_none``: both filters combined.

    A filter that removes every class falls back to the unfiltered mean so the
    value stays defined (mirrors ``_real_macro_f1``'s fallback).
    """

    def _mean_over(indices: list[int]) -> float:
        if not indices:
            indices = list(range(num_classes))
        vals = [float(per_class_f1.get(str(i), 0.0)) for i in indices]
        return float(np.mean(vals)) if vals else 0.0

    supported = [
        i for i in range(num_classes) if int(per_class_support.get(str(i), 0) or 0) > 0
    ]
    real = [i for i in range(num_classes) if i != none_index]
    supported_real = [i for i in supported if i != none_index]

    return {
        "macro_f1_supported": _mean_over(supported),
        "macro_f1_excl_none": _mean_over(real),
        "macro_f1_supported_excl_none": _mean_over(supported_real),
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


def _real_ordinal_indices(num_classes: int, none_index: int) -> list[int]:
    """Class indices that lie on the ordinal scale (everything but ``__none__``).

    ``__none__`` is a separate semantic category, never a position on the
    ordinal scale (see :func:`compute_ordinal_metrics`). The remaining indices
    keep their raw values as ordinal positions, matching the distance space QWK
    is computed over.
    """
    return [i for i in range(num_classes) if i != none_index]


# Bimodal fallback (Bitcrush ISSUE-0562): when the two heaviest real classes
# sit this far apart on the ordinal scale, E[j] lands between the modes where
# almost no probability mass lives, so argmax is authoritative.
_BIMODAL_GAP = 2.0


def ordinal_decode(
    probs: np.ndarray,
    *,
    none_index: int = -1,
    cut_points: list[float] | np.ndarray | None = None,
    confidence_gate: float = 0.5,
) -> list[int]:
    """Decode softmax probabilities to class labels under the QWK-optimal rule.

    Under quadratic-weighted kappa the cost of predicting ``k`` for true class
    ``j`` is ``(k - j)**2``, so the Bayes-optimal prediction is the class that
    minimises expected quadratic cost::

        argmin_k  sum_j p_j (k - j)**2  =  round(sum_j j * p_j)  =  round(E[j])

    i.e. the **rounded expected ordinal index**, not ``argmax``. When the
    posterior is even slightly diffuse the mean and the mode diverge and
    rounding the mean scores higher QWK — but only ambiguity justifies that
    (ISSUE-0540): when one real class holds ``confidence_gate`` or more of the
    renormalised ordinal mass, argmax is authoritative and no fitted boundary
    may override it. Likewise (Bitcrush ISSUE-0562), when the two heaviest real
    classes are ``_BIMODAL_GAP`` or more apart on the ordinal scale the
    posterior is bimodal — E[j] falls between the modes where almost no mass
    lives, so those rows also decode argmax. The EV + cut-point decode only
    decides the remaining genuinely ambiguous rows. Because
    :func:`find_ordinal_cut_points` scores candidates through this function,
    boundaries are fitted under the same gates.

    ``__none__`` is off the ordinal scale: samples whose overall ``argmax`` is
    ``none_index`` are decoded as ``__none__`` (preserving the tuned none-bias
    gate), everything else is decoded over the real classes.

    ``cut_points`` are the ``len(real) - 1`` ascending boundaries on ``E[j]``
    (see :func:`find_ordinal_cut_points`). ``None`` (or malformed) means the
    checkpoint shipped no fitted decode — plain argmax, matching what selection
    scored (`_finalise_ordinal_decode` only persists cut-points that beat
    argmax, so callers need no branch of their own).

    Returns predictions in the original full class-index space, so every
    downstream consumer is unchanged.
    """
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    num_classes = arr.shape[1]

    real_indices = _real_ordinal_indices(num_classes, none_index)
    argmax_all = arr.argmax(axis=1)
    # Fewer than two real classes => no ordinal scale to decode over.
    if len(real_indices) < 2:
        return argmax_all.astype(int).tolist()

    real_arr = np.asarray(real_indices, dtype=np.float64)

    boundaries = None
    if cut_points is not None:
        cand = np.asarray(cut_points, dtype=np.float64)
        if cand.ndim == 1 and cand.shape[0] == len(real_indices) - 1:
            boundaries = cand
    if boundaries is None:
        # No shipped decode => argmax (the contract every caller assumes).
        decoded = argmax_all.astype(np.int64)
    else:
        real_probs = arr[:, real_indices]
        real_sum = real_probs.sum(axis=1, keepdims=True)
        norm = real_probs / np.clip(real_sum, 1e-12, None)
        ev = (norm * real_arr[None, :]).sum(axis=1)  # E[j] over real classes

        # Number of boundaries each E[j] exceeds => position within real_indices.
        # side="right" rounds the half-integer boundary up, matching round-half-up.
        pos = np.searchsorted(boundaries, ev, side="right")
        pos = np.clip(pos, 0, len(real_indices) - 1)
        decoded = real_arr[pos].astype(np.int64)

        # Confidence gate: a majority-of-ordinal-mass class is decoded as
        # itself, whatever the boundaries say. Bimodal gate: top-2 real
        # classes >= _BIMODAL_GAP apart means E[j] sits in the trough between
        # modes, so the mode wins there too.
        real_argmax = real_arr[norm.argmax(axis=1)].astype(np.int64)
        confident = norm.max(axis=1) >= confidence_gate
        order = np.argsort(norm, axis=1)
        bimodal = np.abs(real_arr[order[:, -1]] - real_arr[order[:, -2]]) >= _BIMODAL_GAP
        decoded = np.where(confident | bimodal, real_argmax, decoded)

    if 0 <= none_index < num_classes:
        decoded = np.where(argmax_all == none_index, none_index, decoded)
    return decoded.astype(int).tolist()


def find_ordinal_cut_points(
    probs: np.ndarray,
    labels: np.ndarray | list[int],
    num_classes: int,
    *,
    none_index: int = -1,
    grid_steps: int = 20,
    passes: int = 3,
    clamp: float = 0.25,
) -> list[float] | None:
    """Fit ``E[j]`` decision boundaries that maximise validation QWK.

    The ordinal analogue of :func:`find_per_class_thresholds`: instead of
    rounding ``E[j]`` at the half-integers, search the ``len(real) - 1``
    boundaries (an OptimizedRounder, Kaggle-style) so the decode is tuned to the
    metric we actually ship on. Coordinate ascent over a per-boundary grid,
    keeping boundaries monotonic; deterministic and dependency-free.

    Each boundary is confined to its neutral half-integer ``+/- clamp``
    (Bitcrush ISSUE-0562): the unconstrained fit could drift a boundary far
    enough to squeeze a class band to a fraction of its natural width —
    aggregate-QWK-optimal on the val split, per-image indefensible. With the
    default 0.25 no band can shrink below half its natural width.

    Returns the boundary list, or ``None`` when there is no ordinal scale to fit
    (fewer than two real classes) so callers fall back to round-to-nearest.
    """
    real_indices = _real_ordinal_indices(num_classes, none_index)
    if len(real_indices) < 2:
        return None

    arr = np.asarray(probs, dtype=np.float64)
    labels = list(np.asarray(labels).astype(int).tolist())
    real_arr = np.asarray(real_indices, dtype=np.float64)

    def _qwk(boundaries: np.ndarray) -> float:
        preds = ordinal_decode(arr, none_index=none_index, cut_points=boundaries)
        return compute_ordinal_metrics(
            labels, preds, num_classes, none_index=none_index,
        )["qwk"]

    neutral = (real_arr[:-1] + real_arr[1:]) / 2.0
    boundaries = neutral.copy()
    best_score = _qwk(boundaries)

    for _ in range(max(1, passes)):
        improved = False
        for b in range(len(boundaries)):
            lo = real_arr[b] if b == 0 else boundaries[b - 1]
            hi = real_arr[b + 1] if b == len(boundaries) - 1 else boundaries[b + 1]
            # Clamp to the neutral half-integer's neighbourhood (ISSUE-0562).
            lo = max(lo, neutral[b] - clamp)
            hi = min(hi, neutral[b] + clamp)
            if hi - lo <= 1e-9:
                continue
            # Open interval so boundaries stay strictly monotonic.
            candidates = np.linspace(lo, hi, grid_steps + 2)[1:-1]
            for c in candidates:
                trial = boundaries.copy()
                trial[b] = c
                score = _qwk(trial)
                if score > best_score + 1e-9:
                    best_score = score
                    boundaries = trial
                    improved = True
        if not improved:
            break

    return [float(x) for x in boundaries]


def compute_multihead_metrics(
    *,
    band_labels: list[int],
    band_preds: list[int],
    num_bands: int,
    size_volume_labels: list[int],
    size_volume_preds: list[int],
    num_size_ranks: int,
    none_index: int = -1,
) -> dict:
    """Per-head + combined metrics for the multi-head size model.

    Band and size are scored independently as ordinal scales: band over its band vocabulary,
    size over its **volume-rank** scale (so sister-size confusion is zero ordinal error).
    The ``multi_head`` figures are the mean of the two heads' F1 / QWK — a single summary
    number for "how good is the model overall" that the issue asks for.

    Returns ``{band: {f1, qwk}, size: {f1, qwk}, multi_head: {f1, qwk}}``.
    """
    band_mc = compute_multiclass_metrics(band_labels, band_preds, num_bands)
    band_ord = compute_ordinal_metrics(band_labels, band_preds, num_bands, none_index=none_index)
    size_mc = compute_multiclass_metrics(size_volume_labels, size_volume_preds, num_size_ranks)
    size_ord = compute_ordinal_metrics(
        size_volume_labels, size_volume_preds, num_size_ranks, none_index=none_index
    )

    band = {"f1": band_mc["macro_f1"], "qwk": band_ord["qwk"]}
    size = {"f1": size_mc["macro_f1"], "qwk": size_ord["qwk"]}
    multi_head = {
        "f1": (band["f1"] + size["f1"]) / 2.0,
        "qwk": (band["qwk"] + size["qwk"]) / 2.0,
    }
    return {"band": band, "size": size, "multi_head": multi_head}


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
            "per_class_support": {},
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
    per_class_support = {
        str(i): int(labels[:, i].sum()) if i < labels.shape[1] else 0
        for i in range(num_classes)
    }

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
        "per_class_support": per_class_support,
        "hamming_loss": hamming,
        "exact_match_ratio": exact_match,
        "thresholds": (thresholds.tolist() if thresholds is not None else [0.5] * num_classes),
    }
