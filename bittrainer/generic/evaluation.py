"""Validation evaluation, metrics, calibration and ordinal decode (ISSUE-0542).

Extracted verbatim from ``bittrainer.group_trainer``: the val-set evaluator, the
logit->metrics decode, the softmax temperature / ``__none__`` bias calibration,
and the shipped / incumbent ordinal decode scoring. Kept together because the
"decode a batch of val logits under the shipped decode" family is mutually
recursive with calibration and cut-point fitting. ``group_trainer`` (and
``bittrainer.finalize``) re-import these names, so the objects stay identical.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bittrainer.group_validation import (
    compute_multiclass_metrics,
    compute_multilabel_metrics,
    compute_none_metrics,
    compute_ordinal_metrics,
    find_ordinal_cut_points,
    macro_f1_variants,
    ordinal_decode,
)
from bittrainer.selection import _has_none_class, _metric_score

if TYPE_CHECKING:
    from bittrainer.group_trainer import GroupTrainConfig

logger = logging.getLogger(__name__)

_REAL_MACRO_F1_REGRESSION_TOLERANCE = 0.01
_TEMPERATURE_GRID = [0.75, 0.85, 1.0, 1.15, 1.3, 1.5]
_NONE_LOGIT_BIAS_GRID = [round(i * 0.025, 3) for i in range(21)]
# Per-epoch ordinal cut-point fit budget. The finalisation fit keeps the full
# find_ordinal_cut_points defaults (20 steps x 3 passes); per-epoch selection
# only needs the boundaries roughly right for a fair inter-epoch comparison,
# and the full-budget cost is quadratic-ish in num_classes (Age: 101
# boundaries). Bump after GPU profiling if the reduced fit proves unstable.
_EPOCH_CUT_GRID_STEPS = 8
_EPOCH_CUT_PASSES = 1


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    multi_label: bool = False,
    ordinal: bool = False,
    none_index: int = -1,
    thresholds: np.ndarray | None = None,
    channels_last: bool = False,
) -> dict:
    """Evaluate ``model`` on the validation set.

    For multi-label, sigmoid probs and labels are accumulated and stored on
    the returned dict under ``_probs`` and ``_labels`` so the caller can run
    per-class threshold tuning. ``thresholds`` may be passed to binarise at
    custom thresholds (otherwise 0.5 is used).
    """
    model.eval()
    all_probs_ml = []
    all_labels_ml = []
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    if multi_label:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    from bittrainer.gpu_augment import apply_val_transform

    memory_format = torch.channels_last if channels_last else None
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        images = apply_val_transform(images, dtype=dtype, memory_format=memory_format)
        labels = labels.to(device)

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            logits = model(images)
            if multi_label:
                loss = criterion(logits.float(), labels.float())
            else:
                loss = criterion(logits, labels)

        if multi_label:
            probs = torch.sigmoid(logits.float())
            all_probs_ml.append(probs.cpu().numpy())
            all_labels_ml.append(labels.cpu().int().numpy())
        else:
            # Per-epoch selection decodes on argmax (the unbiased mode estimate).
            # Raw round(E[j]) is biased inward at the scale edges for symmetric
            # posteriors, so the EV decode is only adopted at finalisation, and
            # only with fitted cut-points that beat argmax on val (see
            # _finalise_ordinal_decode). This keeps selection stable.
            preds = logits.float().argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        total_loss += loss.item()
        num_batches += 1

    if multi_label:
        all_labels_arr = np.concatenate(all_labels_ml, axis=0)
        all_probs_arr = np.concatenate(all_probs_ml, axis=0)
        if thresholds is None:
            thresholds_arr = np.full(num_classes, 0.5, dtype=np.float64)
        else:
            thresholds_arr = np.asarray(thresholds, dtype=np.float64)
        preds_arr = (all_probs_arr >= thresholds_arr[None, :]).astype(np.int64)
        metrics = compute_multilabel_metrics(
            all_labels_arr, preds_arr, num_classes, thresholds=thresholds_arr,
        )
        metrics["_probs"] = all_probs_arr
        metrics["_labels"] = all_labels_arr
    else:
        metrics = compute_multiclass_metrics(all_labels, all_preds, num_classes)
        if none_index >= 0:
            metrics.update(compute_none_metrics(
                all_labels, all_preds, num_classes, none_index=none_index,
            ))
        if ordinal:
            metrics.update(compute_ordinal_metrics(
                all_labels, all_preds, num_classes, none_index=none_index,
            ))
        _augment_metric_variants(metrics, num_classes, none_index)

    metrics["val_loss"] = total_loss / max(num_batches, 1)
    return metrics


def _per_class_val_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    """Mean (unweighted, unsmoothed) cross-entropy per TRUE class.

    Keyed by ``str(class_index)`` to match ``per_class_f1`` et al. Classes with
    no samples in ``labels`` are omitted (their loss is undefined). By
    construction the support-weighted mean of the returned values equals the
    aggregate ``val_loss`` — this is the per-class overtraining signal the
    dynamic-class-weight controller (and the diagnostics) read.
    """
    if labels.numel() == 0:
        return {}
    per_example = nn.functional.cross_entropy(
        logits.float(), labels.long(), reduction="none"
    )
    out: dict[str, float] = {}
    for c in range(num_classes):
        mask = labels == c
        if bool(mask.any()):
            out[str(c)] = float(per_example[mask].mean().item())
    return out


def _metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    config: GroupTrainConfig,
    none_index: int,
    cut_points: list[float] | None = None,
) -> dict:
    probs = torch.softmax(logits.float(), dim=1)
    if config.ordinal:
        # Shipped ordinal decode: confidence-gated E[j] at the fitted
        # cut-points; cut_points=None decodes as argmax inside (ISSUE-0540).
        preds = ordinal_decode(
            probs.cpu().numpy(), none_index=none_index, cut_points=cut_points,
        )
    else:
        preds = probs.argmax(dim=1).cpu().tolist()
    label_list = labels.cpu().tolist()
    metrics = compute_multiclass_metrics(label_list, preds, config.num_classes)
    if none_index >= 0:
        metrics.update(compute_none_metrics(
            label_list, preds, config.num_classes, none_index=none_index,
        ))
    if config.ordinal:
        metrics.update(compute_ordinal_metrics(
            label_list, preds, config.num_classes, none_index=none_index,
        ))
    _augment_metric_variants(metrics, config.num_classes, none_index)
    metrics["val_loss"] = float(nn.CrossEntropyLoss()(logits.float(), labels.long()).item())
    metrics["per_class_val_loss"] = _per_class_val_loss(logits, labels, config.num_classes)
    return metrics


def _augment_metric_variants(metrics: dict, num_classes: int, none_index: int) -> dict:
    """Attach the report-only macro-F1 variants (supported / __none__-excluded).

    Selection stays on the raw metrics; the variants exist so consumers can see
    the honest number for groups whose class list outruns their val support.
    """
    metrics.update(macro_f1_variants(
        metrics.get("per_class_f1") or {},
        metrics.get("per_class_support") or {},
        num_classes,
        none_index=none_index,
    ))
    return metrics


def _real_macro_f1(metrics: dict, config: GroupTrainConfig, none_index: int) -> float:
    per_class = metrics.get("per_class_f1") or {}
    if config.num_classes <= (1 if 0 <= none_index < config.num_classes else 0):
        return float(metrics.get("macro_f1") or 0.0)
    variants = macro_f1_variants(
        per_class, {}, config.num_classes, none_index=none_index,
    )
    return variants["macro_f1_excl_none"]


@torch.no_grad()
def _collect_val_logits(
    model: nn.Module,
    val_loader: DataLoader,
    config: GroupTrainConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    from bittrainer.gpu_augment import apply_val_transform

    model.eval()
    memory_format = torch.channels_last if config.channels_last else None
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        images = apply_val_transform(images, dtype=dtype, memory_format=memory_format)
        labels = labels.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            logits = model(images)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.long().cpu())
    if not all_logits:
        raise RuntimeError("No validation logits available for calibration")
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def _apply_calibration(
    logits: torch.Tensor,
    *,
    temperature: float,
    none_bias: float,
    none_index: int,
) -> torch.Tensor:
    calibrated = logits.float() / max(float(temperature), 1e-6)
    if none_bias and 0 <= none_index < calibrated.shape[1]:
        calibrated = calibrated.clone()
        calibrated[:, none_index] += float(none_bias)
    return calibrated


def _tune_softmax_calibration(
    logits: torch.Tensor,
    labels: torch.Tensor,
    config: GroupTrainConfig,
    none_index: int,
) -> tuple[float, list[float], dict]:
    if config.multi_label or none_index < 0:
        return 1.0, [0.0] * config.num_classes, _metrics_from_logits(logits, labels, config, none_index)

    base_logits = logits.float()
    base_metrics = _metrics_from_logits(base_logits, labels, config, none_index)
    base_score = _metric_score(base_metrics, config)
    base_loss = float(base_metrics.get("val_loss") or 0.0)

    best_temp = 1.0
    best_temp_logits = base_logits
    best_temp_metrics = base_metrics
    best_temp_loss = base_loss
    for temp in _TEMPERATURE_GRID:
        cand_logits = _apply_calibration(base_logits, temperature=temp, none_bias=0.0, none_index=none_index)
        cand_metrics = _metrics_from_logits(cand_logits, labels, config, none_index)
        cand_loss = float(cand_metrics.get("val_loss") or 0.0)
        cand_score = _metric_score(cand_metrics, config)
        if cand_loss < best_temp_loss and cand_score + 1e-9 >= base_score:
            best_temp = float(temp)
            best_temp_logits = cand_logits
            best_temp_metrics = cand_metrics
            best_temp_loss = cand_loss

    base_real_f1 = _real_macro_f1(best_temp_metrics, config, none_index)
    best_bias = 0.0
    best_metrics = best_temp_metrics
    best_score = _metric_score(best_metrics, config)
    for bias in _NONE_LOGIT_BIAS_GRID:
        cand_logits = _apply_calibration(
            base_logits, temperature=best_temp, none_bias=float(bias), none_index=none_index,
        )
        cand_metrics = _metrics_from_logits(cand_logits, labels, config, none_index)
        cand_score = _metric_score(cand_metrics, config)
        cand_real_f1 = _real_macro_f1(cand_metrics, config, none_index)
        if (
            cand_score > best_score + 1e-9
            and cand_real_f1 + _REAL_MACRO_F1_REGRESSION_TOLERANCE >= base_real_f1
        ):
            best_bias = float(bias)
            best_metrics = cand_metrics
            best_score = cand_score
            break

    bias_vec = [0.0] * config.num_classes
    if 0 <= none_index < config.num_classes:
        bias_vec[none_index] = best_bias
    best_metrics["selected_validation_score"] = _metric_score(best_metrics, config)
    best_metrics["calibration_temperature"] = best_temp
    best_metrics["none_logit_bias"] = best_bias
    return best_temp, bias_vec, best_metrics


def _finalise_ordinal_decode(
    logits: torch.Tensor,
    labels: torch.Tensor,
    config: GroupTrainConfig,
    none_index: int,
    *,
    grid_steps: int = 20,
    passes: int = 3,
) -> tuple[list[float] | None, dict]:
    """Fit E[j] cut-points on (calibrated) val logits, adopting the EV decode
    only when it beats argmax on the selection score.

    Returns ``(cut_points or None, metrics under the chosen decode)``. ``None``
    cut-points mean inference keeps argmax (the safe default) — so the shipped
    ordinal decode can never score below argmax on validation. ``argmax`` is the
    unbiased mode estimate; raw ``round(E[j])`` is biased inward at the scale
    edges, and only the fitted cut-points (OptimizedRounder) reliably correct it.

    ``grid_steps``/``passes`` bound the coordinate-ascent budget: finalisation
    keeps the full defaults, per-epoch selection uses the reduced
    ``_EPOCH_CUT_*`` budget.
    """
    argmax_metrics = _metrics_from_logits(logits, labels, config, none_index, cut_points=None)
    argmax_metrics["ordinal_decode"] = "argmax"
    if not config.ordinal:
        return None, argmax_metrics

    probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
    label_list = labels.cpu().tolist()
    cuts = find_ordinal_cut_points(
        probs, label_list, config.num_classes, none_index=none_index,
        grid_steps=grid_steps, passes=passes,
    )
    if not cuts:
        return None, argmax_metrics

    ev_metrics = _metrics_from_logits(logits, labels, config, none_index, cut_points=cuts)
    if _metric_score(ev_metrics, config) > _metric_score(argmax_metrics, config) + 1e-9:
        ev_metrics["ordinal_decode"] = "expected_value"
        ev_metrics["ordinal_cut_points"] = cuts
        return cuts, ev_metrics
    return None, argmax_metrics


def _shipped_decode_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    config: GroupTrainConfig,
    none_index: int,
    *,
    cut_grid_steps: int = _EPOCH_CUT_GRID_STEPS,
    cut_passes: int = _EPOCH_CUT_PASSES,
) -> dict:
    """Score val logits under the decode the model actually ships with.

    Mirrors finalisation: temperature + ``__none__`` logit bias (when a none
    class exists) then the ordinal EV cut-point decode (adopted only when it
    beats argmax), so per-epoch selection and the shipped model agree on what
    "best" means. Plain single-label groups (no none, not ordinal) reduce to
    argmax — identical to the previous behaviour.
    """
    if _has_none_class(config):
        temperature, bias_vec, metrics = _tune_softmax_calibration(
            logits, labels, config, none_index,
        )
        none_bias = (
            float(bias_vec[none_index]) if 0 <= none_index < len(bias_vec) else 0.0
        )
        calibrated = _apply_calibration(
            logits, temperature=temperature, none_bias=none_bias, none_index=none_index,
        )
    else:
        calibrated = logits.float()
        metrics = _metrics_from_logits(calibrated, labels, config, none_index)
    if config.ordinal:
        _, metrics = _finalise_ordinal_decode(
            calibrated, labels, config, none_index,
            grid_steps=cut_grid_steps, passes=cut_passes,
        )
    metrics["selection_decode"] = "shipped"
    return metrics


def _incumbent_decode_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    config: GroupTrainConfig,
    none_index: int,
    ckpt: object,
) -> dict:
    """Score the incumbent under its OWN persisted calibration.

    The fair comparison must judge the incumbent by what it ships with —
    re-fitting calibration on it would credit it with tuning it never had,
    and scoring it on raw argmax would penalise a well-calibrated incumbent.
    Checkpoints from before calibration persistence (or non-dict payloads)
    have no keys and fall back to plain argmax.
    """
    temperature = 1.0
    none_bias = 0.0
    cuts: list[float] | None = None
    if isinstance(ckpt, dict):
        temperature = float(ckpt.get("temperature") or 1.0)
        bias_list = ckpt.get("class_logit_bias")
        if bias_list is not None and 0 <= none_index < len(bias_list):
            none_bias = float(bias_list[none_index])
        raw_cuts = ckpt.get("ordinal_cut_points")
        if config.ordinal and raw_cuts:
            cuts = [float(x) for x in raw_cuts]
    calibrated = _apply_calibration(
        logits, temperature=temperature, none_bias=none_bias, none_index=none_index,
    )
    metrics = _metrics_from_logits(calibrated, labels, config, none_index, cut_points=cuts)
    metrics["selection_decode"] = "shipped"
    return metrics
