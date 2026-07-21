"""Promotion + finalisation of a trained group checkpoint (Bitcrush ISSUE-0542).

Extracted verbatim from ``bittrainer.group_trainer``: the head-to-head
promotion-vs-incumbent gate, calibration / cut-point / prior / strictness-data
persistence, and the finalisation result dict. Shared by ``run_group_training``
and ``run_head_only_training``. ``_finalise_ordinal_decode`` is re-exported from
``bittrainer.generic.evaluation`` (its definition site) so it stays importable
here too. ``group_trainer`` re-imports every name below, keeping objects identical.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from bittrainer.generic.evaluation import (
    _collect_val_logits,
    _evaluate,
    _finalise_ordinal_decode,
    _incumbent_decode_metrics,
    _metrics_from_logits,
    _tune_softmax_calibration,
)
from bittrainer.group_validation import (
    compute_multilabel_metrics,
    find_per_class_thresholds,
)
from bittrainer.model import load_checkpoint
from bittrainer.priors import _apply_and_persist_priors
from bittrainer.promotion import PromotionReason, decide_promotion
from bittrainer.selection import (
    _has_none_class,
    _metric_score,
    _primary_validation_metric,
    _resolve_none_index,
)

if TYPE_CHECKING:
    from bittrainer.group_trainer import GroupTrainConfig

logger = logging.getLogger(__name__)

# ``_finalise_ordinal_decode`` lives in generic.evaluation (mutually recursive
# with calibration there); re-exported so ``bittrainer.finalize`` keeps the name.
__all__ = [
    "_compare_promote_finalize",
    "_finalise_ordinal_decode",
    "_persist_ordinal_cut_points",
    "_persist_softmax_calibration",
    "_persist_strictness_val_data",
]

# Cap stored validation rows so the checkpoint stays small (a few hundred KB at
# most). The suite only needs enough resolution to draw a selective-metric curve;
# evenly subsampling large val sets preserves the curve shape.
_STRICTNESS_MAX_VAL_ROWS = 5000


def _persist_softmax_calibration(
    checkpoint_path: str | None,
    *,
    config: GroupTrainConfig,
    metrics: dict,
    temperature: float,
    class_logit_bias: list[float],
) -> None:
    if not checkpoint_path or config.multi_label:
        return
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict):
            ckpt["validation_metric"] = _primary_validation_metric(config)
            ckpt["temperature"] = float(temperature)
            ckpt["class_logit_bias"] = [float(v) for v in class_logit_bias]
            ckpt["none_metrics"] = {
                "none_precision": metrics.get("none_precision"),
                "none_recall": metrics.get("none_recall"),
                "none_f1": metrics.get("none_f1"),
                "none_false_positive_rate": metrics.get("none_false_positive_rate"),
                "none_support": metrics.get("none_support"),
            }
            torch.save(ckpt, checkpoint_path)
    except Exception:
        logger.warning("Failed to persist softmax calibration to checkpoint", exc_info=True)


def _persist_strictness_val_data(
    checkpoint_path: str | None,
    *,
    config: GroupTrainConfig,
    val_loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    temperature: float,
    class_logit_bias: list[float],
    ml_probs: np.ndarray | None,
    ml_labels: np.ndarray | None,
    cached_logits: torch.Tensor | None,
    cached_labels: torch.Tensor | None,
) -> None:
    """Stash *calibrated* validation probabilities + labels in the checkpoint.

    Generic, gating-agnostic data: the suite reconstructs the auto-correct
    "strictness curve" (selective metric vs confidence gate) from these without
    re-running inference. The probabilities match what inference ships — softmax
    over temperature/bias-calibrated logits (single-label) or sigmoid scores
    (multi-label) — so the suite's gate confidences line up exactly.
    """
    if not checkpoint_path:
        return
    try:
        if config.multi_label:
            if ml_probs is None or ml_labels is None:
                return
            probs = np.asarray(ml_probs, dtype=np.float32)
            labels = np.asarray(ml_labels).astype(np.int64)  # [N, C] indicator
        else:
            if cached_logits is not None and cached_labels is not None:
                logits, labels_t = cached_logits, cached_labels
            else:
                model = load_checkpoint(
                    checkpoint_path, device=str(device), dtype=dtype,
                    model_size=config.backbone_variant, num_classes=config.num_classes,
                ).to(device)
                logits, labels_t = _collect_val_logits(model, val_loader, config, device, dtype)
                del model
            bias_t = torch.tensor(class_logit_bias, dtype=torch.float32)
            calibrated = logits.float() / max(float(temperature), 1e-6) + bias_t
            probs = torch.softmax(calibrated, dim=1).numpy().astype(np.float32)
            labels = labels_t.numpy().astype(np.int64)  # [N]

        if probs.shape[0] == 0:
            return
        if probs.shape[0] > _STRICTNESS_MAX_VAL_ROWS:
            idx = np.linspace(0, probs.shape[0] - 1, _STRICTNESS_MAX_VAL_ROWS).astype(np.int64)
            probs = probs[idx]
            labels = labels[idx]

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict):
            ckpt["val_probs"] = torch.from_numpy(np.ascontiguousarray(probs))
            ckpt["val_labels"] = torch.from_numpy(np.ascontiguousarray(labels))
            torch.save(ckpt, checkpoint_path)
    except Exception:
        logger.warning("Failed to persist strictness val data to checkpoint", exc_info=True)


def _persist_ordinal_cut_points(checkpoint_path: str | None, cut_points: list[float]) -> None:
    if not checkpoint_path:
        return
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict):
            ckpt["ordinal_cut_points"] = [float(x) for x in cut_points]
            torch.save(ckpt, checkpoint_path)
    except Exception:
        logger.warning("Failed to persist ordinal_cut_points to checkpoint", exc_info=True)


def _compare_promote_finalize(
    config: GroupTrainConfig,
    *,
    candidate_path: str | None,
    best_metrics: dict,
    candidate_macro_f1: float,
    candidate_qwk: float,
    best_epoch_display: int,
    epochs_completed: int,
    val_loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    checkpoint_dir: Path,
    class_counts: dict[int, int],
    total_raw: int,
    effective_class_counts: dict[int, int] | None = None,
    cb: Callable[[dict], None] | None = None,
) -> dict:
    """Promote-if-better vs the incumbent, tune thresholds, build the result dict.

    Shared by ``run_group_training`` (after the FT loop) and
    ``run_head_only_training`` (after the probe) so both resolve a candidate
    identically: a worse candidate never replaces a better incumbent, and the
    winner's path is returned for the group to adopt.
    """
    def _emit(stage: str, status_text: str) -> None:
        if cb is not None:
            cb({"type": "training_progress", "stage": stage, "status_text": status_text})

    promotion_reason: PromotionReason | None = None
    existing_best = checkpoint_dir / config.best_model_name
    best_val_macro_f1 = candidate_macro_f1
    best_val_qwk = candidate_qwk
    best_checkpoint_path = candidate_path
    # Which decode produced the selection metrics ("shipped" from the FT epoch
    # loop; None for legacy/head-only paths). Captured before finalisation
    # overwrites best_metrics with the calibrated dicts.
    selection_decode = best_metrics.get("selection_decode")

    if best_checkpoint_path:
        candidate_score = _metric_score(best_metrics, config)
        incumbent_class_names: list[str] | None = None
        incumbent_num_classes: int | None = None
        incumbent_score: float | None = None
        old_metrics: dict | None = None
        eval_ok = False

        if config.auto_promote:
            # Auto-Promote: ship the candidate without loading or scoring the
            # incumbent at all (no head-to-head, no guard). The caller has
            # asserted the incumbent should not be the thing to beat — typically
            # because it is known-leaky on the current validation split.
            _emit("comparing", "Auto-Promote on — shipping new model without comparison")
            promote, promotion_reason = decide_promotion(
                incumbent_exists=existing_best.exists(),
                incumbent_class_names=None,
                candidate_class_names=list(config.class_names),
                incumbent_score=None,
                candidate_score=candidate_score,
                eval_ok=False,
                auto_promote=True,
            )
        elif not existing_best.exists():
            _emit("comparing", "Comparing against current model")
            promote, promotion_reason = decide_promotion(
                incumbent_exists=False,
                incumbent_class_names=None,
                candidate_class_names=list(config.class_names),
                incumbent_score=None,
                candidate_score=candidate_score,
                eval_ok=False,
            )
        else:
            _emit("comparing", "Comparing against current model")
            try:
                old_data = torch.load(str(existing_best), map_location=device, weights_only=True)
                if isinstance(old_data, dict):
                    incumbent_class_names = old_data.get("class_names")
                    incumbent_num_classes = old_data.get("num_classes")
                    old_size = old_data.get("model_size", config.backbone_variant)
                else:
                    old_size = config.backbone_variant

                names_match = (
                    incumbent_class_names is None
                    or list(incumbent_class_names) == list(config.class_names)
                )
                counts_match = (
                    incumbent_num_classes is None
                    or incumbent_num_classes == config.num_classes
                )
                if names_match and counts_match:
                    # load_checkpoint infers head_hidden_size from the weights, so
                    # an MLP-head incumbent reconstructs correctly.
                    old_model = load_checkpoint(
                        str(existing_best), device=str(device), dtype=dtype,
                        model_size=old_size, num_classes=config.num_classes,
                    ).to(device)
                    if config.multi_label:
                        old_metrics = _evaluate(
                            old_model, val_loader, config.num_classes, device, dtype,
                            multi_label=True,
                            ordinal=config.ordinal,
                            none_index=_resolve_none_index(config.class_names),
                        )
                    else:
                        # Fair comparison: the incumbent is judged under its own
                        # persisted calibration/decode (argmax for pre-calibration
                        # checkpoints), matching the shipped-decode candidate score.
                        inc_logits, inc_labels = _collect_val_logits(
                            old_model, val_loader, config, device, dtype,
                        )
                        old_metrics = _incumbent_decode_metrics(
                            inc_logits, inc_labels, config,
                            _resolve_none_index(config.class_names), old_data,
                        )
                    del old_model
                    incumbent_score = _metric_score(old_metrics, config)
                    eval_ok = True
            except Exception:
                logger.warning("Failed to load/evaluate incumbent checkpoint", exc_info=True)
                eval_ok = False

            promote, promotion_reason = decide_promotion(
                incumbent_exists=True,
                incumbent_class_names=incumbent_class_names,
                candidate_class_names=list(config.class_names),
                incumbent_score=incumbent_score,
                candidate_score=candidate_score,
                eval_ok=eval_ok,
                incumbent_num_classes=incumbent_num_classes,
                candidate_num_classes=config.num_classes,
            )

        if promote:
            logger.info("Promoting new checkpoint (reason=%s)", promotion_reason.value)
            _emit("promoting", "Promoting new model")
            Path(best_checkpoint_path).replace(existing_best)
            best_checkpoint_path = str(existing_best)
        else:
            logger.info(
                "Keeping incumbent (reason=%s, incumbent=%.4f vs candidate=%.4f)",
                promotion_reason.value,
                incumbent_score if incumbent_score is not None else -1.0,
                candidate_score,
            )
            _emit("promoting", "Keeping current model (scored higher)")
            Path(best_checkpoint_path).unlink(missing_ok=True)
            best_checkpoint_path = str(existing_best)
            # The kept incumbent's metrics become the reported metrics. Sync ALL
            # summary scalars to it (not just the ordinal/non-ordinal selection
            # one) â€” otherwise group.best_val_macro_f1 keeps showing the losing
            # candidate's F1 while every other field shows the kept model.
            if old_metrics is not None:
                best_metrics = old_metrics
                best_val_macro_f1 = old_metrics.get("macro_f1", best_val_macro_f1)
                best_val_qwk = old_metrics.get("qwk", best_val_qwk)

    # Per-class threshold tuning for multi-label â€” replaces the hardcoded 0.5
    # with F1-optimal thresholds picked on the validation set used by the best
    # (or post-compare) model. Thresholds are baked into the checkpoint.
    calibration_temperature = 1.0
    class_logit_bias = [0.0] * config.num_classes
    ordinal_cut_points: list[float] | None = None
    none_idx = _resolve_none_index(config.class_names)
    # Reused below to persist the strictness val data without a second val pass.
    val_logits_cache: torch.Tensor | None = None
    val_labels_cache: torch.Tensor | None = None
    # Single-label groups always reach finalisation so prior-correction vectors
    # are persisted (ISSUE-0490 A) — even plain groups with no __none__ / ordinal
    # calibration, which is the motivating class-imbalance case. Calibration
    # (temperature / none-bias / cut-points) is layered on only when applicable.
    if best_checkpoint_path and not config.multi_label:
        needs_calibration = _has_none_class(config) or config.ordinal
        try:
            _emit(
                "calibrating",
                "Calibrating decision boundaries" if needs_calibration else "Finalising priors",
            )
            calib_model = load_checkpoint(
                best_checkpoint_path, device=str(device), dtype=dtype,
                model_size=config.backbone_variant, num_classes=config.num_classes,
            ).to(device)
            logits, labels = _collect_val_logits(calib_model, val_loader, config, device, dtype)
            del calib_model

            # Prior correction (ISSUE-0490 A): adjust val logits by
            # tau*(log natural - log effective train prior) and persist both
            # vectors, BEFORE fitting temperature / none-bias so calibration
            # (and the strictness val data below) sees the shipped decode input.
            logits = _apply_and_persist_priors(
                logits,
                class_counts,
                effective_class_counts if effective_class_counts is not None else class_counts,
                config,
                best_checkpoint_path,
            )
            val_logits_cache, val_labels_cache = logits, labels

            # __none__ gate: temperature + none-bias (only when a none class exists).
            if _has_none_class(config):
                calibration_temperature, class_logit_bias, calibrated_metrics = _tune_softmax_calibration(
                    logits, labels, config, none_idx,
                )
                best_metrics = calibrated_metrics
                _persist_softmax_calibration(
                    best_checkpoint_path,
                    config=config,
                    metrics=best_metrics,
                    temperature=calibration_temperature,
                    class_logit_bias=class_logit_bias,
                )

            # Ordinal EV cut-points, fit on the *calibrated* logits (temperature
            # and bias shift E[j]) and adopted only when they beat argmax on the
            # selection score. Absent cut-points => inference keeps argmax.
            if config.ordinal:
                bias_t = torch.tensor(class_logit_bias, dtype=torch.float32)
                calibrated_logits = logits.float() / max(calibration_temperature, 1e-6) + bias_t
                ordinal_cut_points, decoded_metrics = _finalise_ordinal_decode(
                    calibrated_logits, labels, config, none_idx,
                )
                best_metrics = decoded_metrics
                if ordinal_cut_points is not None:
                    _persist_ordinal_cut_points(best_checkpoint_path, ordinal_cut_points)

            # Plain single-label groups (no __none__ gate, not ordinal) skipped
            # both re-scoring branches above, so their best_metrics still reflect
            # the pre-prior epoch-loop decode. Re-score on the prior-adjusted
            # logits (already in memory — no extra forward pass) so reported
            # final_val_* match what the shipped model actually decodes.
            elif not _has_none_class(config):
                best_metrics = _metrics_from_logits(logits, labels, config, none_idx)

            best_val_macro_f1 = best_metrics.get("macro_f1", best_val_macro_f1)
            best_val_qwk = best_metrics.get("qwk", best_val_qwk)
        except Exception:
            logger.warning("Calibration / decode tuning failed; keeping uncalibrated checkpoint", exc_info=True)

    final_thresholds: list[float] | None = None
    if (
        config.multi_label
        and config.per_class_thresholds_enabled
        and best_metrics.get("_probs") is not None
        and best_metrics.get("_labels") is not None
    ):
        probs_arr = best_metrics["_probs"]
        labels_arr = best_metrics["_labels"]
        thresholds_arr = find_per_class_thresholds(probs_arr, labels_arr)
        tuned = compute_multilabel_metrics(
            labels_arr, predictions=None,
            num_classes=config.num_classes,
            thresholds=thresholds_arr, probs=probs_arr,
        )
        best_metrics.update(tuned)
        best_val_macro_f1 = tuned["macro_f1"]
        final_thresholds = thresholds_arr.tolist()
        if best_checkpoint_path:
            try:
                ckpt = torch.load(best_checkpoint_path, map_location="cpu", weights_only=True)
                if isinstance(ckpt, dict):
                    ckpt["per_class_thresholds"] = final_thresholds
                    torch.save(ckpt, best_checkpoint_path)
            except Exception:
                logger.warning("Failed to persist per_class_thresholds to checkpoint", exc_info=True)

    # Stash calibrated val probs + labels in the checkpoint so the suite can draw
    # the auto-correct strictness curve later without re-running inference. Uses
    # the logits already collected during calibration when available; for plain
    # single-label groups (no __none__, not ordinal) it does one extra val pass.
    _persist_strictness_val_data(
        best_checkpoint_path,
        config=config,
        val_loader=val_loader,
        device=device,
        dtype=dtype,
        temperature=calibration_temperature,
        class_logit_bias=class_logit_bias,
        ml_probs=best_metrics.get("_probs"),
        ml_labels=best_metrics.get("_labels"),
        cached_logits=val_logits_cache,
        cached_labels=val_labels_cache,
    )

    # Strip internal numpy arrays before constructing the result dict â€”
    # downstream consumers serialise this to JSON.
    best_metrics.pop("_probs", None)
    best_metrics.pop("_labels", None)

    # Val-side per-class counts: the support context that separates "class F1
    # is 0" from "class had no validation samples to score".
    try:
        val_class_counts = dict(val_loader.dataset.get_class_counts())
    except Exception:
        val_class_counts = {}

    result = {
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch_display,
        "best_val_macro_f1": best_val_macro_f1,
        "validation_metric": _primary_validation_metric(config),
        "selected_validation_score": _metric_score(best_metrics, config),
        "final_val_macro_f1": best_metrics.get("macro_f1"),
        "final_val_weighted_f1": best_metrics.get("weighted_f1"),
        "final_val_micro_f1": best_metrics.get("micro_f1"),
        # What drove checkpoint selection this run (ISSUE-0490 B). Ordinal groups
        # ignore selection_metric, so record the composite label instead.
        "selection_metric": (
            "ordinal_composite" if config.ordinal else getattr(config, "selection_metric", "macro_f1")
        ),
        "final_val_macro_f1_supported": best_metrics.get("macro_f1_supported"),
        "final_val_macro_f1_excl_none": best_metrics.get("macro_f1_excl_none"),
        "final_val_macro_f1_supported_excl_none": best_metrics.get(
            "macro_f1_supported_excl_none"
        ),
        "final_val_macro_precision": best_metrics.get("macro_precision"),
        "final_val_macro_recall": best_metrics.get("macro_recall"),
        # Skin Tone V2 dual-view tracks (None for non-dual-view runs).
        "final_val_macro_f1_original": best_metrics.get("macro_f1_original"),
        "final_val_macro_f1_normalized": best_metrics.get("macro_f1_normalized"),
        "final_val_macro_f1_dual": best_metrics.get("macro_f1_dual"),
        "final_val_loss": best_metrics.get("val_loss"),
        "per_class_f1": best_metrics.get("per_class_f1", {}),
        "per_class_precision": best_metrics.get("per_class_precision", {}),
        "per_class_recall": best_metrics.get("per_class_recall", {}),
        "per_class_support": best_metrics.get("per_class_support", {}),
        "checkpoint_path": best_checkpoint_path,
        "class_counts": class_counts,
        "val_class_counts": val_class_counts,
        "total_images": total_raw,
        "promotion_reason": promotion_reason.value if promotion_reason else None,
        "selected_softness_kind": config.selected_softness_kind,
        "selected_softness_value": config.selected_softness_value,
        "soft_label_tuning_metric": config.soft_label_tuning_metric,
        "soft_label_tuning_results": config.soft_label_tuning_results,
        "soft_label_tuning_elapsed_ms": config.soft_label_tuning_elapsed_ms,
        "selected_oversample_none": config.selected_oversample_none,
        "oversample_tuning_metric": config.oversample_tuning_metric,
        "oversample_tuning_results": config.oversample_tuning_results,
        "oversample_tuning_elapsed_ms": config.oversample_tuning_elapsed_ms,
        "data_quality_warnings": config.data_quality_warnings,
        "final_val_none_precision": best_metrics.get("none_precision"),
        "final_val_none_recall": best_metrics.get("none_recall"),
        "final_val_none_f1": best_metrics.get("none_f1"),
        "final_val_none_false_positive_rate": best_metrics.get("none_false_positive_rate"),
        "calibration_temperature": calibration_temperature,
        "none_logit_bias": (
            class_logit_bias[_resolve_none_index(config.class_names)]
            if _resolve_none_index(config.class_names) >= 0 and class_logit_bias
            else 0.0
        ),
        "ordinal_sigma": config.ordinal_sigma,
        "ordinal_cut_points": ordinal_cut_points,
        "ordinal_decode": best_metrics.get("ordinal_decode"),
        "selection_decode": selection_decode,
        "label_smoothing": config.label_smoothing,
    }
    if final_thresholds is not None:
        result["per_class_thresholds"] = final_thresholds
    if config.ordinal:
        result["best_val_qwk"] = best_val_qwk
        result["qwk"] = best_metrics.get("qwk")
        result["ordinal_mae"] = best_metrics.get("ordinal_mae")
        result["adjacent_accuracy"] = best_metrics.get("adjacent_accuracy")
    if config.multi_label:
        result["hamming_loss"] = best_metrics.get("hamming_loss")
        result["exact_match_ratio"] = best_metrics.get("exact_match_ratio")
    else:
        result["confusion_matrix"] = best_metrics.get("confusion_matrix", [])
        result["balanced_accuracy"] = best_metrics.get("balanced_accuracy")
    return result
