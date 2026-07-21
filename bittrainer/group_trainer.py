"""Training loop for ConvNeXt V2 multi-class group classifiers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from adv_optm import Prodigy_adv
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from bittrainer.ema import ModelEMA
from bittrainer.group_dataset import (
    GroupDataset,
    build_group_bucket_sampler,
)
from bittrainer.dynamic_class_weights import DynamicClassWeightController
from bittrainer.losses import AsymmetricLoss, FocalLoss
from bittrainer.model_soup import greedy_soup
from bittrainer.embedding_cache import EmbeddingCache
from bittrainer.backbone_init import apply_backbone_init, wants_timm_pretrained
from bittrainer.model import (
    backbone_feature_hash,
    create_model,
    unfreeze_backbone,
)

# ---------------------------------------------------------------------------
# Extracted helpers (Bitcrush ISSUE-0542): the clusters below were moved to
# dedicated modules; re-imported here so the old ``bittrainer.group_trainer``
# import paths keep resolving to the very same objects (head_only_trainer,
# Engine shims, older pickles). See selection / soft_labels / collate / probes /
# priors / finalize / generic.optimizer / generic.evaluation.
# ---------------------------------------------------------------------------
# Redundant ``X as X`` aliases mark these as explicit re-exports (PEP 484), so
# they survive dead-import pruning even where only the moved module uses them.
from bittrainer.collate import (
    _collate_bucket_batch as _collate_bucket_batch,
    _collate_multilabel_batch as _collate_multilabel_batch,
)
from bittrainer.finalize import (
    _compare_promote_finalize as _compare_promote_finalize,
    _persist_ordinal_cut_points as _persist_ordinal_cut_points,
    _persist_softmax_calibration as _persist_softmax_calibration,
    _persist_strictness_val_data as _persist_strictness_val_data,
)
from bittrainer.generic.evaluation import (
    _apply_calibration as _apply_calibration,
    _augment_metric_variants as _augment_metric_variants,
    _collect_val_logits as _collect_val_logits,
    _evaluate as _evaluate,
    _finalise_ordinal_decode as _finalise_ordinal_decode,
    _incumbent_decode_metrics as _incumbent_decode_metrics,
    _metrics_from_logits as _metrics_from_logits,
    _per_class_val_loss as _per_class_val_loss,
    _real_macro_f1 as _real_macro_f1,
    _shipped_decode_metrics as _shipped_decode_metrics,
    _tune_softmax_calibration as _tune_softmax_calibration,
)
from bittrainer.generic.optimizer import make_optimizer as make_optimizer
from bittrainer.priors import (
    _apply_and_persist_priors as _apply_and_persist_priors,
    _compute_prior_vectors as _compute_prior_vectors,
    _compute_prior_vectors_from_counts as _compute_prior_vectors_from_counts,
    _persist_class_priors as _persist_class_priors,
    _prior_logit_delta as _prior_logit_delta,
)
from bittrainer.probes import (
    _apply_resolved as _apply_resolved,
    _apply_softness as _apply_softness,
    _auto_oversample_enabled as _auto_oversample_enabled,
    _auto_softness_candidates as _auto_softness_candidates,
    _auto_softness_kind as _auto_softness_kind,
    _build_oversampled_tensors as _build_oversampled_tensors,
    _capture_rng_state as _capture_rng_state,
    _oversample_candidate_better as _oversample_candidate_better,
    _resolved_snapshot as _resolved_snapshot,
    _restore_rng_state as _restore_rng_state,
    _run_auto_oversample_probe as _run_auto_oversample_probe,
    _run_auto_softness_probe as _run_auto_softness_probe,
    _softness_candidate_better as _softness_candidate_better,
    _softness_status_label as _softness_status_label,
)
from bittrainer.selection import (
    _NONE_CLASS_NAME as _NONE_CLASS_NAME,
    _NONE_F1_WEIGHT as _NONE_F1_WEIGHT,
    _SELECTION_SECONDARY_WEIGHT as _SELECTION_SECONDARY_WEIGHT,
    _composite_selection_score as _composite_selection_score,
    _guarded_metric_enabled as _guarded_metric_enabled,
    _guarded_score as _guarded_score,
    _has_none_class as _has_none_class,
    _metric_score as _metric_score,
    _ordinal_primary_score as _ordinal_primary_score,
    _primary_validation_metric as _primary_validation_metric,
    _resolve_none_index as _resolve_none_index,
    _score_metric_label as _score_metric_label,
    _selection_base_f1 as _selection_base_f1,
)
from bittrainer.soft_labels import (
    _build_gaussian_kernel as _build_gaussian_kernel,
    _build_perceptual_kernel as _build_perceptual_kernel,
    _build_soft_targets as _build_soft_targets,
    _soft_ce_loss as _soft_ce_loss,
)

logger = logging.getLogger(__name__)


def _make_optimizer(model: nn.Module, config: GroupTrainConfig) -> Prodigy_adv:
    """Delegate to the shared factory (Bitcrush ISSUE-0542); signature kept so
    ``run_group_training`` / ``head_only_trainer`` call sites are unchanged."""
    return make_optimizer(model, llrd=config.llrd, llrd_decay=config.llrd_decay)


# --- Checkpoint selection (epoch "best" criterion) -------------------------
# QWK alone is gameable: a model that collapses toward the centre of the
# ordinal scale keeps QWK high (adjacent errors are cheap) while exact-match
# macro-F1 craters. Selecting argmax(QWK) then latches onto a noise-level QWK
# spike that happens to coincide with an F1 collapse (e.g. QWK 0.89/F1 0.45
# "beating" QWK 0.88/F1 0.70). We instead select ordinal models on a weighted
# harmonic mean of the ordinal metric and macro-F1 (the harmonic mean, as in
# F1 itself, punishes any single weak component far more than an arithmetic
# mean), and require a minimum improvement so epoch-to-epoch noise cannot flip
# the choice (one-standard-error-rule spirit: Breiman/CART, ESL 7.10;
# min_delta: Prechelt 1998, Keras EarlyStopping).
_SELECTION_MIN_DELTA = 0.002        # min composite gain required to replace the incumbent best


@dataclass
class GroupTrainConfig:
    group_folder: str
    num_classes: int
    class_names: list[str]
    max_epochs: int = 50
    patience: int = 3
    backbone_variant: str = "nano"
    label_smoothing: float = 0.1
    ordinal: bool = False
    ordinal_sigma: float = 1.0
    validation_metric: str = "qwk"
    # __none__ "guard": when True, fold the __none__-class F1 into epoch/candidate
    # selection (macro_f1 + 0.1*none_f1, and qwk + 0.1*none_f1 for guarded_qwk
    # ordinal groups) and veto promotion on __none__-recall regression. Default
    # OFF — selection is plain macro_f1 (non-ordinal) / the raw qwk+macro_f1
    # composite (ordinal). Opt in per-group when __none__ recall matters more
    # than raw macro-F1.
    none_guard: bool = False
    # Non-ordinal checkpoint-selection metric (ISSUE-0490 B): "macro_f1"
    # (default, unchanged), "weighted_f1" (support-weighted per-class F1), or
    # "balanced" (harmonic mean of macro and weighted). IGNORED for ordinal
    # groups, which always select on the QWK+macro composite.
    selection_metric: str = "macro_f1"
    # Inference-time prior correction (ISSUE-0490 A): tau scales the Bayes logit
    # adjustment log(natural_prior) - log(effective_train_prior). 1.0 = full
    # correction; stored in the checkpoint so it can be tuned later without a
    # retrain. Not exposed in UI for v1.
    prior_tau: float = 1.0
    multi_label: bool = False
    # Spatial-grid groups (e.g. Subject Location): per-class grid cell masks in
    # class-index order (``__none__`` = empty list). When set, the trainer
    # (a) swaps the classifier fc for the cell-structured SpatialCellFC head,
    # (b) flips labels together with images instead of the label-blind hflip,
    # and (c) restricts RandAugment to photometric ops. None = ordinary group.
    cell_masks: list[list[int]] | None = None
    grid_rows: int = 3
    grid_cols: int = 3
    oversample_none: bool = False
    extra_paths_train: dict[str, list[str]] = field(default_factory=dict)
    extra_paths_val: dict[str, list[str]] = field(default_factory=dict)
    soft_aliases: dict = field(default_factory=dict)
    device: str = "cuda"
    dtype: str = "bfloat16"
    from_scratch: bool = False
    # Bitcrush Engine backbone spec (see bittrainer.backbone_init) — governs
    # where fresh-model backbone weights come from. None = timm pretrained.
    backbone_init: dict | None = None
    # Auto-Promote: skip the incumbent comparison entirely and ship the freshly
    # trained candidate as best.pt unconditionally (no incumbent load, no score
    # compare, no guard). The escape hatch for a known-leaky incumbent — e.g. a
    # re-split group whose incumbent trained on images now in the current
    # validation split, which would otherwise be scored unfairly high. Off by
    # default: the head-to-head promotion gate governs.
    auto_promote: bool = False
    best_model_name: str = "best.pt"
    checkpoint_dir: str | None = None
    skin_normalise: bool = False
    # Skin Tone V2 dual-view (Bitcrush Engine ISSUE-0217, spec §8): path to
    # the engine-written per-image normalisation manifest
    # (skin_tone_normalisation.json), the frozen calibration snapshot
    # (informational, rides along for provenance), and the probability of
    # swapping a TRAIN sample for its colour-normalised view. Validation
    # always scores both views separately when the manifest is present.
    skin_tone_views_manifest: str = ""
    skin_tone_calibration: dict | None = None
    skin_tone_dual_view_prob: float = 0.5
    # ΔE-perceptual soft labels (Skin Tone V2 spec §8.1): Oklab [L, a, b]
    # centroid per class NAME. When >=2 classes carry centroids, soft targets
    # use a Gaussian kernel over PERCEPTUAL centroid distance (Euclidean in
    # Oklab) instead of ordinal rank or uniform label smoothing — probability
    # bleeds into genuinely-near classes (undertone-aware) and none into
    # display-adjacent-but-perceptually-distant ones. perceptual_sigma is in
    # Oklab ΔE units (~0.02 = one JND).
    class_similarity_centroids: dict = field(default_factory=dict)
    perceptual_sigma: float = 0.035
    face_model_path: str = ""
    # Region-crop training: a YOLO detector localises the group's concept and
    # the crop centres on it (fine-grained groups lose their discriminative
    # region to the ~512px bucket resolution otherwise). Takes precedence over
    # face_model_path. region_classes filters detector classes (empty = all);
    # region_selection is "highest_conf" or "union"; region_fallback is
    # "full_frame" (centre crop for undetected images, the face behaviour) or
    # "drop" (remove undetected TRAIN images — val always keeps full coverage
    # so metrics stay comparable).
    region_model_path: str = ""
    region_classes: list[str] = field(default_factory=list)
    region_selection: str = "highest_conf"
    region_fallback: str = "full_frame"
    cache_dir: str | None = None
    use_cache: bool = True
    cache_workers: int = 10
    sourceless: bool = False
    group_name: str = ""
    modeltype: str = "convnext_v2"
    progress_callback: Callable[[dict], None] | None = None
    # Layer-wise learning rate decay
    llrd: bool = True
    llrd_decay: float = 0.8
    # Asymmetric loss (multi-label only â€” no effect on single-label paths)
    use_asl: bool = True
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 0.0
    asl_clip: float = 0.05
    # --- Phase-2 regularisers: OPT-IN, default OFF. An A/B on the ordinal
    # "Inner Labia" group (6 epochs) regressed QWK -0.19 vs the resample
    # baseline, so these ship off and are enabled per-run/per-group (frontend
    # plumbing TBD). The parameter values below are the recommended settings
    # *when* enabled. ---
    # Exponential moving average of weights. 0.999 decay (time constant ~1k
    # steps) engages at 1k-10k-image sizes; the old 0.9999 only neared target
    # after ~90k steps. When on, EMA weights become the primary state_dict.
    use_ema: bool = False
    ema_decay: float = 0.999
    # Stochastic Weight Averaging over the cosine tail (epoch >= swa_start_frac *
    # max_epochs). ConvNeXt is LayerNorm-only, so no BatchNorm recalibration is
    # needed. Needs >= 2 epochs in the tail to do anything.
    use_swa: bool = False
    swa_start_frac: float = 0.6
    # MixUp / CutMix (batch-level). Composes with ordinal soft targets: targets
    # are smoothed first, then interpolated. Gated off on tiny datasets.
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mixup_prob: float = 0.5
    mixup_min_images: int = 200
    # Class imbalance handling. "resample" (default) replicates every class up to
    # the largest; "reweight" samples at the natural distribution and applies
    # effective-number class weights (Cui et al., 2019) in the loss; "auto" picks
    # "reweight" only above class_balance_auto_ratio. resample + weights would
    # double-correct, so the two are mutually exclusive by construction.
    # NOTE: "reweight"/"auto" reduce gradient steps/epoch (natural < replicated),
    # which under-trained short runs in testing — prefer for long-horizon runs.
    class_balance_mode: str = "resample"
    class_balance_beta: float = 0.999
    class_balance_auto_ratio: float = 4.0
    use_focal: bool = False
    focal_gamma: float = 2.0
    # --- Dynamic per-class loss weighting (ISSUE-0392): OPT-IN, default OFF, a
    # soft per-class early-stop. When a class's smoothed val-F1 declines from its
    # per-class peak for ``dcw_patience`` epochs (for "both", corroborated by
    # rising per-class val loss), its loss-weight multiplier is shrunk by
    # ``dcw_decay``, clamped to [``dcw_floor``, ``dcw_ceiling``], with a
    # ``dcw_cooldown``-epoch refractory gap. Weights are renormalised to mean 1,
    # holding the effective LR constant (reallocation, not global scale-down).
    # Single-label only (rides the CE ``weight=`` path); ignored for multi-label.
    # Composes on top of ``class_balance_mode="reweight"`` (that becomes the base
    # weight vector; otherwise the base is all-ones = a numerical no-op at start).
    dynamic_class_weighting: bool = False
    dcw_metric: str = "val_f1"  # "val_f1" | "val_loss" | "both"
    dcw_patience: int = 2
    dcw_ema_decay: float = 0.5
    dcw_decay: float = 0.8
    dcw_floor: float = 0.25
    dcw_ceiling: float = 1.0
    dcw_cooldown: int = 1
    dcw_min_delta: float = 0.005
    # Per-epoch weight snapshots for snapshot-ensemble experiments (ISSUE-0392
    # follow-up). When set, the deployable state_dict is written to
    # {snapshot_dir}/epoch_NNN.pt after every epoch. Off by default.
    snapshot_dir: str | None = None
    # --- Greedy weight soup (ISSUE-0392): ON by default. After training, average
    # the weights of the strongest epochs into ONE model, greedily accepting an
    # epoch only when it does not lower the val selection score, and adopt the
    # soup as the shipped checkpoint only when it strictly beats the best single
    # epoch. Zero extra inference/storage cost (one model). By construction it
    # can only match or beat best-single on val -> safe as a default for all
    # trainable group types. Toggle off to ship the single best epoch. ---
    use_greedy_soup: bool = True
    soup_max_candidates: int = 8
    # RandAugment + RandomErasing (DeiT/ConvNeXt official fine-tune recipe)
    randaugment_n: int = 2
    randaugment_m: int = 9
    random_erasing_p: float = 0.25
    # Per-class threshold tuning for multi-label
    per_class_thresholds_enabled: bool = True
    # Manual batch size override â€” skips the auto-batch probe/heuristic when set
    batch_size: int | None = None
    # VRAM probe target fraction for the autobatch heuristic. 0.85 of *free*
    # VRAM (optimizer/EMA state budgeted separately); cloud runs tighten this
    # per OOM retry.
    vram_fraction: float = 0.85
    # Cached-feature head probe ("train heads" scouting + full-FT warmup).
    # probe_head: "linear" trains head.fc only (canonical linear probe);
    # "mlp" adds a Linear->GELU->Linear pre_logits MLP as the intermediate
    # escalation rung before a full fine-tune. probe_mlp_hidden sizes the MLP.
    probe_head: str = "linear"
    probe_mlp_hidden: int = 512
    head_max_epochs: int = 50
    head_patience: int = 5
    head_weight_decay: float = 0.02
    embedding_cache_dir: str | None = None
    auto_label_softness: bool = True
    selected_softness_kind: str | None = None
    selected_softness_value: float | None = None
    soft_label_tuning_metric: str | None = None
    soft_label_tuning_results: list[dict] = field(default_factory=list)
    soft_label_tuning_elapsed_ms: int | None = None
    # Auto ``__none__`` oversample sweep: a second pre-training probe (after the
    # soft-label sweep) that trains a head with no rare-group oversampling and
    # again with 1.5x ``__none__`` oversampling, then selects the better by
    # validation score and applies it to the full fine-tune dataset.
    # ``oversample_none`` (above) becomes the *resolved* value the sweep writes.
    auto_oversample_none: bool = True
    selected_oversample_none: bool | None = None
    oversample_tuning_metric: str | None = None
    oversample_tuning_results: list[dict] = field(default_factory=list)
    oversample_tuning_elapsed_ms: int | None = None
    data_quality_warnings: list[dict] = field(default_factory=list)
    # torch.compile for the full fine-tune forward/backward. Falls back to
    # eager (with a status message) when triton is unavailable.
    use_compile: bool = True
    # NHWC layout â€” ConvNeXt stem/downsample/dwconv save permute traffic.
    channels_last: bool = True
    # Gradient accumulation escape hatch: optimizer steps every N batches.
    grad_accum_steps: int = 1
    # --- Backup / Pause / Resume (Bitcrush ISSUE-0405) ---
    # backup_dir=None => NO backups written and NO resume attempted (exact
    # legacy behaviour). backup_every_steps=0 => epoch-boundary backups only.
    # resume_from points at a backup dir (newest-compatible is loaded) or a
    # specific backup file. dataloader_workers replaces the previously-hardcoded
    # DataLoader num_workers=6; 0 makes a mid-epoch resume bit-exact.
    backup_dir: str | None = None
    backup_every_steps: int = 500
    resume_from: str | None = None
    dataloader_workers: int = 6


def _spatial_ckpt_meta(config: GroupTrainConfig) -> dict:
    """Checkpoint metadata for spatial groups — load_checkpoint reconstructs
    the SpatialCellFC head from these keys. Empty for ordinary groups."""
    if not config.cell_masks:
        return {}
    return {
        "cell_masks": [list(m) for m in config.cell_masks],
        "grid_rows": int(config.grid_rows),
        "grid_cols": int(config.grid_cols),
    }


def _build_epoch_message(
    *,
    epoch: int,
    config: GroupTrainConfig,
    train_loss: float,
    val_metrics: dict,
    best_val_macro_f1: float,
    best_val_qwk: float,
    selected_score: float,
    best_validation_score: float,
    best_epoch: int,
    per_class_train_loss: dict,
    elapsed_seconds: float | None = None,
    dcw_multipliers: dict | None = None,
) -> dict:
    """Assemble the per-epoch ``epoch_complete`` progress payload.

    Extracted so the Engine-facing contract (which drives per-epoch run-history
    persistence, ISSUE-0491) is unit-testable without a training loop. Carries
    ``val_weighted_f1`` / ``val_micro_f1`` / ``elapsed_seconds`` alongside the
    existing loss / macro-F1 / per-class fields the history graphs need.
    """
    msg = {
        "type": "epoch_complete",
        "stage": "training",
        "status_text": (
            f"Epoch {epoch + 1}/{config.max_epochs} complete "
            f"(val macro F1 {val_metrics.get('macro_f1', 0.0):.3f})"
        ),
        "epoch": epoch + 1,
        "max_epochs": config.max_epochs,
        "train_loss": train_loss,
        "val_loss": val_metrics.get("val_loss"),
        "val_macro_f1": val_metrics.get("macro_f1"),
        "val_weighted_f1": val_metrics.get("weighted_f1"),
        "val_micro_f1": val_metrics.get("micro_f1"),
        "val_macro_f1_supported": val_metrics.get("macro_f1_supported"),
        "val_macro_precision": val_metrics.get("macro_precision", 0.0),
        "val_macro_recall": val_metrics.get("macro_recall", 0.0),
        "per_class_f1": val_metrics.get("per_class_f1", {}),
        "per_class_precision": val_metrics.get("per_class_precision", {}),
        "per_class_recall": val_metrics.get("per_class_recall", {}),
        "per_class_support": val_metrics.get("per_class_support", {}),
        "per_class_val_loss": val_metrics.get("per_class_val_loss", {}),
        "per_class_train_loss": per_class_train_loss,
        "val_none_precision": val_metrics.get("none_precision"),
        "val_none_recall": val_metrics.get("none_recall"),
        "val_none_f1": val_metrics.get("none_f1"),
        "val_none_false_positive_rate": val_metrics.get("none_false_positive_rate"),
        "best_val_macro_f1": best_val_macro_f1,
        "selected_validation_score": selected_score,
        "best_validation_score": best_validation_score,
        "validation_metric": _primary_validation_metric(config),
        "best_epoch": best_epoch + 1,
        "elapsed_seconds": elapsed_seconds,
    }
    if dcw_multipliers is not None:
        msg["per_class_weight_multiplier"] = dcw_multipliers
    if config.ordinal:
        msg["val_qwk"] = val_metrics.get("qwk", 0.0)
        msg["val_ordinal_mae"] = val_metrics.get("ordinal_mae")
        msg["val_adjacent_accuracy"] = val_metrics.get("adjacent_accuracy")
        msg["best_val_qwk"] = best_val_qwk
    return msg


def _build_data_quality_warnings(
    train_ds: GroupDataset,
    val_ds: GroupDataset,
    config: GroupTrainConfig,
) -> list[dict]:
    warnings: list[dict] = []
    none_index = _resolve_none_index(config.class_names)
    train_counts = train_ds.get_class_counts()
    val_counts = val_ds.get_class_counts()
    total_train = sum(int(v or 0) for v in train_counts.values())
    total_val = sum(int(v or 0) for v in val_counts.values())
    if none_index >= 0:
        none_train = int(train_counts.get(none_index, 0) or 0)
        none_val = int(val_counts.get(none_index, 0) or 0)
        none_train_ratio = none_train / total_train if total_train else 0.0
        none_val_ratio = none_val / total_val if total_val else 0.0
        if none_train_ratio < 0.10:
            warnings.append({
                "code": "low_none_train_ratio",
                "severity": "warning",
                "message": "__none__ training coverage is below 10%",
                "none_train": none_train,
                "total_train": total_train,
                "ratio": none_train_ratio,
            })
        if none_val < 25:
            warnings.append({
                "code": "low_none_val_support",
                "severity": "warning",
                "message": "__none__ validation support is below 25 images",
                "none_val": none_val,
                "total_val": total_val,
            })
        if none_val_ratio < 0.05:
            warnings.append({
                "code": "low_none_val_ratio",
                "severity": "warning",
                "message": "__none__ validation coverage is below 5%",
                "none_val": none_val,
                "total_val": total_val,
                "ratio": none_val_ratio,
            })
    elif not config.multi_label:
        warnings.append({
            "code": "missing_none_class",
            "severity": "high",
            "message": "Open-world group has no absence class",
            "total_train": total_train,
            "total_val": total_val,
        })
    return warnings


# Step-callback throttle for hot training loops (~4 Hz keeps the UI live
# without flooding the multiprocessing queue).
_STEP_REPORT_INTERVAL = 0.25


def _get_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


# ---------------------------------------------------------------------------
# Soft target construction (ordinal + soft aliases)
# ---------------------------------------------------------------------------


def build_group_loss_fn(
    config: GroupTrainConfig,
    *,
    use_soft_targets: bool,
    none_index: int,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return ``loss_fn(logits, labels)`` for the group head/loss zoo.

    Single source of truth for the three branches (multi-label ASL/BCE,
    soft-CE with ordinal Gaussian smoothing + soft aliases, plain/label-smoothed
    CE) shared by the full-FT loop and the cached head probe â€” so neither can
    drift from the other. ``class_weights`` (the "reweight" balance mode) and
    ``config.use_focal`` apply to the single-label paths only.
    """
    focal_gamma = config.focal_gamma if config.use_focal else 0.0

    if config.multi_label:
        if config.use_asl:
            ml_criterion: nn.Module = AsymmetricLoss(
                gamma_neg=config.asl_gamma_neg,
                gamma_pos=config.asl_gamma_pos,
                clip=config.asl_clip,
            )
        else:
            ml_criterion = nn.BCEWithLogitsLoss()

        def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return ml_criterion(logits.float(), labels.float())

    elif use_soft_targets:
        perceptual_kernel = _build_perceptual_kernel(
            list(config.class_names),
            config.class_similarity_centroids or {},
            config.perceptual_sigma,
            none_index=none_index,
        )

        def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            soft = _build_soft_targets(
                labels, config.num_classes,
                ordinal=config.ordinal,
                ordinal_sigma=config.ordinal_sigma,
                label_smoothing=config.label_smoothing,
                soft_aliases=config.soft_aliases or None,
                none_index=none_index,
                device=device,
                perceptual_kernel=perceptual_kernel,
            )
            log_probs = torch.log_softmax(logits.float(), dim=1)
            return _soft_ce_loss(
                log_probs, soft, class_weights=class_weights, focal_gamma=focal_gamma,
            )

    elif focal_gamma > 0:
        focal = FocalLoss(gamma=focal_gamma, label_smoothing=config.label_smoothing)

        def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return focal(logits, labels)

    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.0, weight=class_weights)

        def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return criterion(logits, labels)

    return loss_fn


def _resolve_class_balance(config: GroupTrainConfig, class_counts: dict[int, int]) -> str:
    """Resolve ``class_balance_mode`` to a concrete "resample"/"reweight".

    ``auto`` picks "reweight" only when the max/min class-count ratio (over
    classes with at least one image) exceeds ``class_balance_auto_ratio`` — mild
    imbalance is served fine by the existing replication-equalised sampling, and
    switching to weights there adds variance for no gain.
    """
    mode = (config.class_balance_mode or "resample").lower()
    if mode != "auto":
        return mode if mode in ("resample", "reweight") else "resample"
    counts = [c for c in class_counts.values() if c > 0]
    if len(counts) < 2:
        return "resample"
    ratio = max(counts) / max(1, min(counts))
    return "reweight" if ratio >= config.class_balance_auto_ratio else "resample"


def _effective_number_weights(
    class_counts: dict[int, int],
    num_classes: int,
    beta: float,
    device: torch.device,
) -> torch.Tensor:
    """Class weights by the effective number of samples (Cui et al., CVPR 2019).

    ``w_c ∝ (1 - beta) / (1 - beta^{n_c})``, normalised so the mean weight is 1
    (keeps the loss scale comparable to the unweighted baseline). Empty classes
    keep weight 1.
    """
    weights = torch.ones(num_classes, dtype=torch.float32)
    for i in range(num_classes):
        n = int(class_counts.get(i, 0))
        if n > 0:
            eff = (1.0 - beta**n) / (1.0 - beta) if beta < 1.0 else float(n)
            weights[i] = 1.0 / max(eff, 1e-8)
    weights = weights * (num_classes / weights.sum().clamp(min=1e-8))
    return weights.to(device)


def _build_dcw_controller(
    config: GroupTrainConfig,
    class_weights: torch.Tensor | None,
    device: torch.device,
) -> DynamicClassWeightController | None:
    """Construct the dynamic per-class loss-weight controller, or None.

    Gated to single-label groups only (the multi-label ASL/BCE path has no
    per-class weight parameter). The base weight vector is the static
    ``class_weights`` when reweight balancing is active, else all-ones — and an
    all-ones base with all-ones multipliers renormalises to all-ones, i.e. a
    numerical no-op versus unweighted CE, so enabling the controller does not
    perturb epoch 1.
    """
    if not config.dynamic_class_weighting or config.multi_label:
        return None
    base = (
        class_weights
        if class_weights is not None
        else torch.ones(config.num_classes, device=device)
    )
    return DynamicClassWeightController(
        config.num_classes, base,
        metric=config.dcw_metric,
        patience=config.dcw_patience,
        ema_decay=config.dcw_ema_decay,
        decay=config.dcw_decay,
        floor=config.dcw_floor,
        ceiling=config.dcw_ceiling,
        cooldown=config.dcw_cooldown,
        min_delta=config.dcw_min_delta,
    )


def _update_soup_pool(
    pool: list[tuple[float, int, str]],
    soup_dir: Path,
    score: float,
    epoch: int,
    state_dict: dict,
    max_candidates: int,
) -> None:
    """Keep the top-``max_candidates`` epochs (by val selection score) on disk as
    greedy-soup ingredients. Saves this epoch's (CPU) weights when it qualifies and
    evicts the lowest-scoring candidate so at most ``max_candidates`` files exist."""
    if len(pool) >= max_candidates and score <= min(s for s, _, _ in pool):
        return
    soup_dir.mkdir(parents=True, exist_ok=True)
    path = str(soup_dir / f"cand_{epoch + 1:03d}.pt")
    torch.save({k: v.detach().cpu() for k, v in state_dict.items()}, path)
    pool.append((score, epoch, path))
    pool.sort(key=lambda t: t[0], reverse=True)
    while len(pool) > max_candidates:
        _, _, drop = pool.pop()
        try:
            Path(drop).unlink()
        except OSError:
            pass


class _SWA:
    """Running average of model weights over the cosine tail (LayerNorm backbone
    => no BatchNorm recalibration needed). Captures a uniform average of the
    state_dicts handed to ``update`` and materialises them into a model on
    request."""

    def __init__(self) -> None:
        self._avg: dict[str, torch.Tensor] | None = None
        self.n = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        sd = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}
        if self._avg is None:
            self._avg = sd
        else:
            for k, v in sd.items():
                self._avg[k].mul_(self.n / (self.n + 1)).add_(v / (self.n + 1))
        self.n += 1

    def state_dict(self) -> dict[str, torch.Tensor] | None:
        return self._avg

    def load_state_dict(self, avg: dict[str, torch.Tensor] | None, n: int) -> None:
        """Restore the running average and its sample count (for resume).

        Cloned onto CPU float tensors so a subsequent ``update`` keeps compositing
        into the same accumulator the interrupted run held."""
        if avg is None:
            self._avg = None
        else:
            self._avg = {k: v.detach().float().cpu().clone() for k, v in avg.items()}
        self.n = int(n)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: GroupTrainConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    use_soft_targets: bool = False,
    step_callback: Callable[[int, int, float], None] | None = None,
    stop_now_event: object | None = None,
    ema: ModelEMA | None = None,
    class_weights: torch.Tensor | None = None,
    mixup_enabled: bool = False,
    pause_event: object | None = None,
    boundary_hook: Callable[[int], str | None] | None = None,
    start_batch: int = 0,
) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    # ``num_batches`` is the ABSOLUTE position within the epoch: it starts at
    # ``start_batch`` on a mid-epoch resume so the accumulation modulo and the
    # ``== total_steps`` final-flush check stay aligned with the full epoch, while
    # the dataloader only yields the remaining ``schedule[start_batch:]`` batches.
    # ``ran`` counts locally-run batches for loss averaging.
    num_batches = start_batch
    ran = 0
    total_steps = start_batch + len(dataloader)
    # Per-class train-loss telemetry (diagnostic): mean hard-label CE per true
    # class, mirroring _per_class_val_loss so train-vs-val divergence is visible
    # per class. MixUp batches are skipped (hard labels no longer match the
    # optimised soft target). Kept on-device; reduced to a dict at return.
    per_class_loss_sum = torch.zeros(config.num_classes, device=device)
    per_class_loss_count = torch.zeros(config.num_classes, device=device)
    accum = max(1, int(config.grad_accum_steps))
    _last_report = time.monotonic()
    none_index = _resolve_none_index(config.class_names)

    loss_fn = build_group_loss_fn(
        config, use_soft_targets=use_soft_targets,
        none_index=none_index, device=device, class_weights=class_weights,
    )
    focal_gamma = config.focal_gamma if config.use_focal else 0.0

    from bittrainer.gpu_augment import apply_train_augment
    from bittrainer.mixing import apply_mixing

    # Spatial groups: horizontal flip must remap the label to the mirrored
    # composition (or skip samples with no mirror class) — the label-blind
    # flip inside apply_train_augment would teach left/right classes to
    # collapse into each other.
    spatial_flip_map: torch.Tensor | None = None
    if config.cell_masks:
        from bittrainer.spatial import build_hflip_class_map

        spatial_flip_map = torch.tensor(
            build_hflip_class_map(config.cell_masks, config.grid_rows, config.grid_cols),
            dtype=torch.long, device=device,
        )

    memory_format = torch.channels_last if config.channels_last else None
    optimizer.zero_grad()
    for images, labels in dataloader:
        if stop_now_event is not None and stop_now_event.is_set():
            break
        # Pause between clean boundaries (accum==1 only, where every top-of-loop
        # is a boundary): the boundary_hook below owns the pause save at true
        # gradient-accumulation boundaries, so this is a secondary early-out when
        # no boundary_hook is wired.
        if boundary_hook is None and pause_event is not None and pause_event.is_set():
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device)
        if spatial_flip_map is not None:
            from bittrainer.spatial import spatial_hflip_batch

            images, labels = spatial_hflip_batch(images, labels, spatial_flip_map)
        images = apply_train_augment(
            images, dtype=dtype,
            randaugment_n=config.randaugment_n,
            randaugment_m=config.randaugment_m,
            random_erasing_p=config.random_erasing_p,
            memory_format=memory_format,
            hflip=spatial_flip_map is None,
            photometric_only=spatial_flip_map is not None,
        )

        # MixUp/CutMix: smooth targets first (preserving ordinal/label smoothing),
        # then interpolate, so mixing composes with the soft-target loss.
        mix_soft = None
        if mixup_enabled and torch.rand(1).item() < config.mixup_prob:
            mix_soft = _build_soft_targets(
                labels, config.num_classes,
                ordinal=config.ordinal, ordinal_sigma=config.ordinal_sigma,
                label_smoothing=config.label_smoothing,
                soft_aliases=config.soft_aliases or None,
                none_index=none_index, device=device,
                perceptual_kernel=_build_perceptual_kernel(
                    list(config.class_names),
                    config.class_similarity_centroids or {},
                    config.perceptual_sigma,
                    none_index=none_index,
                ),
            )
            images, mix_soft = apply_mixing(
                images, mix_soft, config.num_classes,
                mixup_alpha=config.mixup_alpha, cutmix_alpha=config.cutmix_alpha,
                label_smoothing=0.0,
            )

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            logits = model(images)
            if mix_soft is not None:
                loss = _soft_ce_loss(
                    torch.log_softmax(logits.float(), dim=1), mix_soft,
                    class_weights=class_weights, focal_gamma=focal_gamma,
                )
            else:
                loss = loss_fn(logits, labels)

        scaled = loss / accum if accum > 1 else loss
        scaled.backward()
        num_batches += 1
        ran += 1
        boundary_signal = None
        if num_batches % accum == 0 or num_batches == total_steps:
            optimizer.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)
            # Backups fire ONLY here — at a real gradient-accumulation boundary,
            # with no in-flight grads — so a restored optimizer state is coherent.
            if boundary_hook is not None:
                boundary_signal = boundary_hook(num_batches)

        total_loss += loss.item()

        # Per-class train-loss telemetry — hard-label CE, diagnostic only, no
        # grad. Skipped for MixUp batches where `labels` no longer matches the
        # optimised (interpolated soft) target.
        if mix_soft is None:
            with torch.no_grad():
                per_ex = nn.functional.cross_entropy(
                    logits.float(), labels.long(), reduction="none"
                )
            per_class_loss_sum.index_add_(0, labels.long(), per_ex)
            per_class_loss_count.index_add_(0, labels.long(), torch.ones_like(per_ex))

        if step_callback is not None:
            now = time.monotonic()
            if now - _last_report >= _STEP_REPORT_INTERVAL or num_batches == total_steps:
                _last_report = now
                step_callback(num_batches, total_steps, total_loss / max(ran, 1))

        # Pause requested at this boundary: the boundary_hook already saved the
        # backup; stop consuming batches.
        if boundary_signal == "stop":
            break

    counts = per_class_loss_count.cpu()
    sums = per_class_loss_sum.cpu()
    per_class_train_loss = {
        str(c): float(sums[c] / counts[c])
        for c in range(config.num_classes)
        if counts[c] > 0
    }
    return total_loss / max(ran, 1), per_class_train_loss


# Cap stored validation rows so the checkpoint stays small (a few hundred KB at
# most). The suite only needs enough resolution to draw a selective-metric curve;
# evenly subsampling large val sets preserves the curve shape.


def _prepare_datasets_and_cache(
    config: GroupTrainConfig,
    *,
    cb: Callable[[dict], None],
    stop_event: object | None,
) -> tuple[GroupDataset, GroupDataset, object | None, dict[tuple[int, int], int]]:
    """Build train/val datasets, warm the SmartCache, and count buckets.

    Shared verbatim by ``run_group_training`` and ``run_head_only_training`` so
    both consume identical datasets and the same cached input tensors.
    """
    from bittrainer.smart_cache import _noop_callback, _never_stop
    from bittrainer.trainer import _stop_event_is_set

    group_folder = Path(config.group_folder)
    group_name = config.group_name or group_folder.name

    # --- SmartCache setup ---
    smart_cache = None
    if config.use_cache:
        from bittrainer.smart_cache import SmartCache, region_signature
        cache_root = Path(config.cache_dir) if config.cache_dir else (group_folder / ".smart_cache")
        # region_signature reduces to the historical face_model_signature for
        # face-style args, so existing face-crop caches stay valid.
        smart_cache = SmartCache(
            cache_root,
            modeltype=config.modeltype,
            progress_callback=cb,
            stop_check=partial(_stop_event_is_set, stop_event),
            face_model_sig=region_signature(
                (config.region_model_path or config.face_model_path) or None,
                config.region_classes if config.region_model_path else None,
                config.region_selection if config.region_model_path else "union",
            ),
        )

    # When the auto __none__ oversample sweep is enabled it decides off-vs-1.5x
    # during warmup, so build the train set un-oversampled here and let the
    # caller rebuild it once the sweep has chosen. A manual oversample_none is
    # only honoured up-front when the auto sweep is off.
    initial_oversample_none = config.oversample_none and not _auto_oversample_enabled(
        config, _resolve_none_index(config.class_names),
    )

    if config.sourceless:
        if smart_cache is None:
            raise RuntimeError("sourceless=True requires use_cache=True and a cache_dir")
        cb({
            "type": "training_progress", "stage": "validating",
            "status_text": "Loading sourceless samples from cache",
            "step": 0, "total_steps": 0,
        })
        train_ds = GroupDataset(
            group_folder, config.class_names, split="train",
            multi_label=config.multi_label,
            cache=smart_cache, sourceless=True, group_name=group_name,
            oversample_none=initial_oversample_none,
            extra_paths=config.extra_paths_train,
        )
        val_ds = GroupDataset(
            group_folder, config.class_names, split="val",
            multi_label=config.multi_label,
            cache=smart_cache, sourceless=True, group_name=group_name,
            extra_paths=config.extra_paths_val,
        )
    else:
        train_ds = GroupDataset(
            group_folder, config.class_names, split="train",
            multi_label=config.multi_label,
            skin_normalise=config.skin_normalise, group_name=group_name,
            oversample_none=initial_oversample_none,
            extra_paths=config.extra_paths_train,
        )
        val_ds = GroupDataset(
            group_folder, config.class_names, split="val",
            multi_label=config.multi_label,
            skin_normalise=config.skin_normalise, group_name=group_name,
            extra_paths=config.extra_paths_val,
        )

        # --- Face/region-aware cropping pre-computation ---
        face_bboxes: dict[str, list[int]] = {}
        crop_model = config.region_model_path or config.face_model_path
        if crop_model:
            from bittrainer.face_crop import (
                FaceBBoxCache,
                precompute_region_bboxes,
                region_bbox_cache_name,
            )
            if config.region_model_path:
                cache_name = region_bbox_cache_name(
                    config.region_model_path, config.region_classes, config.region_selection,
                )
                target_classes = config.region_classes or None
                selection = config.region_selection
                stage, verb = "region_detection", "Detecting crop regions"
            else:
                cache_name = "face_bboxes.json"
                target_classes = None
                selection = "union"
                stage, verb = "face_detection", "Detecting faces"
            bbox_cache = FaceBBoxCache(group_folder / ".resize_cache" / cache_name)
            all_image_paths = [s["path"] for s in train_ds.samples] + [s["path"] for s in val_ds.samples]

            def _crop_progress(done: int, total: int) -> None:
                cb({
                    "type": "training_progress", "stage": stage,
                    "status_text": f"{verb} ({done}/{total})",
                    "step": done, "total_steps": total,
                })

            precompute_region_bboxes(
                all_image_paths, bbox_cache, crop_model,
                target_classes=target_classes, selection=selection,
                device=config.device,
                progress_fn=_crop_progress,
            )
            for p in all_image_paths:
                bbox = bbox_cache.get(p)
                if bbox:
                    face_bboxes[p] = bbox
            train_ds.refresh_face_bboxes(face_bboxes)
            val_ds.refresh_face_bboxes(face_bboxes)

            # Undetected-region policy: "drop" removes train images the
            # detector found nothing in (a centre crop of a region-less image
            # is mostly label noise for a fine-grained group). Val keeps full
            # coverage so metrics stay comparable across fallback modes.
            if config.region_model_path and config.region_fallback == "drop":
                dropped = train_ds.drop_paths_without_bbox(face_bboxes)
                if dropped:
                    cb({
                        "type": "training_progress", "stage": stage,
                        "status_text": f"Dropped {dropped} train images with no detected region",
                    })

        # --- Warm SmartCache ---
        if smart_cache is not None:
            from bittrainer.cache_builders import build_image_tensor
            from bittrainer.smart_cache import CachingStoppedException
            all_cache_samples = train_ds.samples + val_ds.samples
            try:
                smart_cache.prepare(
                    all_cache_samples, build_image_tensor,
                    num_workers=config.cache_workers, stage_label="caching",
                )
            except CachingStoppedException:
                logger.info("Caching interrupted by stop_event")
                cb({"type": "training_cancelled", "stage": "caching",
                    "status_text": "Cancelled during cache build"})
                raise
            # Callbacks are only needed during prepare(). Replace with picklable
            # no-ops so the cache (now attached to datasets) survives pickling
            # when DataLoader workers spawn on Windows â€” mp.Event and local
            # closures aren't picklable.
            smart_cache._progress_cb = _noop_callback
            smart_cache._stop_check = _never_stop
            train_ds.set_cache(smart_cache)
            val_ds.set_cache(smart_cache)

    # --- Skin Tone V2 dual-view bank (ISSUE-0217) ---
    if config.skin_tone_views_manifest:
        from bittrainer.skin_tone_views import load_view_bank

        view_bank = load_view_bank(config.skin_tone_views_manifest)
        if view_bank is not None:
            train_ds.skin_tone_views = view_bank
            train_ds.skin_tone_view_prob = float(config.skin_tone_dual_view_prob)
            # Val default stays on the ORIGINAL view (prob 0); the epoch loop
            # flips skin_tone_force_view for the "normalized" scoring pass.
            val_ds.skin_tone_views = view_bank
            cb({
                "type": "training_progress", "stage": "preparing",
                "status_text": f"Skin Tone dual-view: {len(view_bank)} image transforms loaded",
            })

    total_samples = len(train_ds)
    if total_samples == 0:
        raise RuntimeError("No training images found")

    config.data_quality_warnings = _build_data_quality_warnings(train_ds, val_ds, config)
    if config.data_quality_warnings:
        cb({
            "type": "training_progress",
            "stage": "data_quality",
            "status_text": f"{len(config.data_quality_warnings)} data quality warning(s)",
            "data_quality_warnings": config.data_quality_warnings,
        })

    # --- Count samples per bucket ---
    bucket_counts: dict[tuple[int, int], int] = {}
    for s in train_ds.samples:
        b = s["bucket"]
        bucket_counts[b] = bucket_counts.get(b, 0) + 1

    return train_ds, val_ds, smart_cache, bucket_counts


def _emit_model_load_stage(em, config: GroupTrainConfig, checkpoint_dir: Path) -> None:
    """Emit downloading_model vs loading_model so a cold timm cache never looks hung."""
    from bittrainer.progress import Stage

    existing_best = checkpoint_dir / config.best_model_name
    if not config.from_scratch and existing_best.exists():
        em.stage(Stage.loading_model, f"Loading model ({config.backbone_variant}, warm start)")
        return
    if not wants_timm_pretrained(config.backbone_init):
        em.stage(Stage.loading_model, f"Loading model ({config.backbone_variant}, local backbone)")
        return
    try:
        from huggingface_hub import try_to_load_from_cache

        from bittrainer.model import _MODEL_REGISTRY

        model_name = _MODEL_REGISTRY.get(config.backbone_variant, "")
        cached = try_to_load_from_cache(f"timm/{model_name}", "model.safetensors")
        downloading = not isinstance(cached, str)
    except (ImportError, OSError, ValueError):
        downloading = False
    if downloading:
        em.stage(
            Stage.downloading_model,
            f"Downloading pretrained weights ({config.backbone_variant}, first run)",
        )
    else:
        em.stage(Stage.loading_model, f"Loading model ({config.backbone_variant})")


def _create_or_warmstart_model(
    config: GroupTrainConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    head_hidden_size: int | None,
    checkpoint_dir: Path,
) -> nn.Module:
    """Create the target model, warm-starting backbone weights from best.pt.

    Builds the requested head architecture (linear when *head_hidden_size* is
    None, MLP otherwise), then loads every checkpoint tensor whose shape matches
    the target â€” so the backbone and ``head.norm`` carry over while a reshaped or
    newly-added head tail (different class count, or a fresh MLP) starts clean.

    Master weights stay float32 regardless of the training dtype: optimizer
    updates are far smaller than bf16 mantissa resolution, so accumulating them
    into bf16 weights silently loses late-training progress. The training dtype
    applies through autocast only.
    """
    del dtype  # training dtype applies via autocast; master weights are fp32

    def _finalise_head(model: nn.Module) -> nn.Module:
        # Spatial groups swap the classifier fc for the cell-structured head.
        # Done before warm-start matching so a spatial incumbent's cell_fc
        # carries over while a pre-spatial (linear) head simply starts clean.
        if config.cell_masks:
            from bittrainer.spatial import install_spatial_head

            install_spatial_head(
                model, config.cell_masks, config.grid_rows * config.grid_cols,
            )
        return model

    existing_best = checkpoint_dir / config.best_model_name
    if not config.from_scratch and existing_best.exists():
        try:
            data = torch.load(str(existing_best), map_location=device, weights_only=True)
            if isinstance(data, dict) and "state_dict" in data:
                state = data["state_dict"]
                size = data.get("model_size", config.backbone_variant)
            else:
                state = data
                size = config.backbone_variant
            model = _finalise_head(create_model(
                model_size=size, pretrained=False,
                num_classes=config.num_classes, head_hidden_size=head_hidden_size,
            )).to(device)
            target = model.state_dict()
            matched = {
                k: v.to(target[k].dtype) for k, v in state.items()
                if k in target and target[k].shape == v.shape
            }
            model.load_state_dict(matched, strict=False)
            logger.info(
                "Warm-starting from %s (%d/%d tensors matched)",
                existing_best, len(matched), len(target),
            )
            return model
        except (RuntimeError, OSError, KeyError, EOFError):
            logger.warning("Warm-start failed, falling back to pretrained", exc_info=True)
    model = _finalise_head(create_model(
        model_size=config.backbone_variant,
        pretrained=wants_timm_pretrained(config.backbone_init),
        num_classes=config.num_classes, head_hidden_size=head_hidden_size,
    ))
    apply_backbone_init(model, config.backbone_init)
    return model.to(device)


def _warmup_head_probe(
    model: nn.Module,
    config: GroupTrainConfig,
    train_ds: GroupDataset,
    val_ds: GroupDataset,
    smart_cache: object | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    cb: Callable[[dict], None],
    stop_event: object | None,
    stop_now_event: object | None,
) -> None:
    """Train the head to convergence on cached features before the full fine-tune.

    Replaces the old fixed 1-epoch frozen warmup. Builds (or reuses) the
    embedding cache for the current backbone era, verifies it, then runs the
    shared probe â€” leaving *model* with a converged head and a frozen backbone
    (the caller unfreezes for the fine-tune).
    """
    backbone_hash = backbone_feature_hash(model)
    group_folder = Path(config.group_folder)
    embed_root = config.embedding_cache_dir or str(group_folder / ".embedding_cache")
    embed_cache = EmbeddingCache(embed_root, backbone_hash, int(getattr(model, "num_features", 0)))
    all_samples = train_ds.samples + val_ds.samples

    def _stop() -> bool:
        return bool(
            (stop_event is not None and stop_event.is_set())
            or (stop_now_event is not None and stop_now_event.is_set())
        )

    def _build_progress(done: int, total: int) -> None:
        cb({
            "type": "training_progress", "stage": "embedding_build",
            "status_text": f"Warmup: caching features ({done}/{total})",
            "step": done, "total_steps": total,
        })

    cb({
        "type": "training_progress", "stage": "embedding_build",
        "status_text": f"Warmup: caching backbone features (era {backbone_hash})",
    })
    embed_cache.ensure(
        all_samples, model, smart_cache, device=device, dtype=dtype,
        batch_size=config.batch_size or 64,
        progress_cb=_build_progress, stop_check=_stop,
    )
    if _stop():
        return
    embed_cache.verify(all_samples, model, smart_cache, device=device, dtype=dtype)
    cb({
        "type": "training_progress", "stage": "training",
        "status_text": f"Warmup: training head probe ({config.probe_head}) to convergence",
    })
    none_index = _resolve_none_index(config.class_names)
    _run_auto_softness_probe(
        model, config, embed_cache, smart_cache,
        train_ds.samples, val_ds.samples,
        device=device, none_index=none_index,
        cb=cb, stop_event=stop_event,
    )
    if _stop():
        return
    # Second pre-training sweep: pick __none__ oversample off vs 1.5x on the
    # soft-label-selected head. Writes config.oversample_none; the caller
    # rebuilds train_ds to honour it before the full fine-tune.
    _run_auto_oversample_probe(
        model, config, embed_cache, smart_cache,
        train_ds.samples, val_ds.samples,
        device=device, none_index=none_index,
        cb=cb, stop_event=stop_event,
    )


def run_group_training(
    config: GroupTrainConfig,
    *,
    progress_callback: Callable[[dict], None] | None = None,
    stop_event: object | None = None,
    stop_now_event: object | None = None,
    pause_event: object | None = None,
) -> dict:
    """Run the full multi-class training loop.

    stop_event signals a graceful stop at the next epoch boundary.
    stop_now_event additionally interrupts the current epoch's training loop
    mid-batch; validation and the fair-comparison block still run.

    pause_event (Bitcrush ISSUE-0405) requests a *resumable* pause: at the next
    gradient-accumulation boundary the full training state is backed up and the
    loop returns ``{"paused": True, "backup_path", "epoch", "global_step"}``
    WITHOUT running SWA finalisation / greedy soup / promotion. Combined with
    ``config.backup_dir`` (periodic/exception backups) and ``config.resume_from``
    (load the newest compatible backup and continue) it gives exact mid-epoch
    continuation when ``config.dataloader_workers == 0``.
    """
    from functools import partial

    from bittrainer.progress import ProgressEmitter, Stage
    from bittrainer.runtime import configure_cuda_backend, maybe_compile, prewarm_compile
    from bittrainer.smart_cache import _noop_callback
    from bittrainer.training_state import (
        _FixedBatchSampler,
        capture_rng_states,
        collect_epoch_state,
        init_backup,
        loader_kwargs,
        paused_result,
        restore_optimizer_state,
        restore_rng_states,
    )
    em = ProgressEmitter(progress_callback or config.progress_callback or _noop_callback)
    cb = em.raw
    device = torch.device(config.device)
    dtype = _get_dtype(config.dtype)
    configure_cuda_backend()
    group_folder = Path(config.group_folder)
    checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else group_folder / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    coordinator, fingerprint, resume_state = init_backup(
        config, pause_event, cb,
        class_names=config.class_names, num_classes=config.num_classes,
        max_epochs=config.max_epochs, multi_label=config.multi_label,
        ordinal=config.ordinal, best_model_name=config.best_model_name,
        model_size=config.backbone_variant,
    )
    if resume_state is not None:
        # Re-apply the sweep outcomes the interrupted run resolved (the sweeps
        # themselves are skipped below) before anything reads label_smoothing /
        # ordinal_sigma / oversample_none / class_balance_mode.
        _apply_resolved(config, resume_state.get("resolved") or {})
        em.stage(Stage.resuming, f"Resuming from backup (epoch {resume_state.get('epoch')})")

    _paused_result = partial(
        paused_result, cb,
        stage="backing_up", status_text="Training paused — state backed up",
    )

    use_soft = (
        config.ordinal
        or bool(config.soft_aliases)
        or bool(config.class_similarity_centroids)
        or (not config.multi_label and config.label_smoothing > 0)
    )

    em.stage(Stage.scanning, "Scanning dataset")
    train_ds, val_ds, smart_cache, bucket_counts = _prepare_datasets_and_cache(
        config, cb=cb, stop_event=stop_event,
    )
    if coordinator.paused:  # pause requested during scan/cache — nothing to finalise
        return _paused_result(resume_state["epoch"] if resume_state else 0, 0, None)

    # Create model â€” warm-start from best.pt unless from_scratch is set.
    head_hidden_size = config.probe_mlp_hidden if config.probe_head == "mlp" else None
    memory_format = torch.channels_last if config.channels_last else None
    if resume_state is None:
        _emit_model_load_stage(em, config, checkpoint_dir)
        model = _create_or_warmstart_model(
            config, device=device, dtype=dtype,
            head_hidden_size=head_hidden_size, checkpoint_dir=checkpoint_dir,
        )
        if memory_format is not None:
            model = model.to(memory_format=memory_format)

        # Head warmup on cached features (replaces the fixed 1-epoch frozen
        # warmup), then fine-tune fully unfrozen. A converged head removes the
        # feature-distortion risk a random head poses, so there is no
        # gradual-unfreeze ramp.
        _warmup_head_probe(
            model, config, train_ds, val_ds, smart_cache,
            device=device, dtype=dtype, cb=cb,
            stop_event=stop_event, stop_now_event=stop_now_event,
        )
        unfreeze_backbone(model)  # the probe froze the backbone â€” restore full grad
        if coordinator.paused:  # pause requested during warmup/sweeps
            return _paused_result(0, 0, None)
    else:
        # Resume: rebuild the architecture directly and load the backed-up
        # weights (skip warm-start, warmup probe and the sweeps entirely).
        model = create_model(
            model_size=config.backbone_variant, pretrained=False,
            num_classes=config.num_classes, head_hidden_size=head_hidden_size,
        )
        if config.cell_masks:
            from bittrainer.spatial import install_spatial_head

            install_spatial_head(model, config.cell_masks, config.grid_rows * config.grid_cols)
        model.load_state_dict(resume_state["model"])
        model = model.to(device)
        if memory_format is not None:
            model = model.to(memory_format=memory_format)
        unfreeze_backbone(model)

    # The warmup oversample sweep may have flipped config.oversample_none; rebuild
    # the train set (and bucket histogram) so the full fine-tune trains on the
    # chosen __none__ composition. No-op when the sweep was off/undecided.
    if config.oversample_none != train_ds.oversample_none:
        train_ds.set_oversample_none(config.oversample_none)
        bucket_counts = {}
        for s in train_ds.samples:
            b = s["bucket"]
            bucket_counts[b] = bucket_counts.get(b, 0) + 1

    # --- Auto batch sizing (probe unfrozen = worst-case VRAM) ---
    # Targets config.vram_fraction of free VRAM. Prodigy_adv state (~2.2x param
    # bytes, allocated lazily on first .step()) is budgeted explicitly inside
    # determine_batch_size via param_overhead_bytes, so the fraction only needs
    # to absorb allocator fragmentation and activation variance across buckets.
    resume_bs_changed = False
    if resume_state is not None:
        # Resume: reuse the backed-up batch size (skip the probe). If the caller
        # now forces a different batch_size (Engine's OOM degrade halves it) the
        # backed-up batch_schedule no longer maps, so honour the new size and
        # fall back to epoch-restart resume (schedule discarded below).
        backup_bs = int(resume_state["eff_bs"])
        if config.batch_size and int(config.batch_size) > 0 and int(config.batch_size) != backup_bs:
            eff_bs = int(config.batch_size)
            resume_bs_changed = True
            logger.info("Resume batch size changed %d -> %d; epoch-restart resume", backup_bs, eff_bs)
        else:
            eff_bs = backup_bs
        cb({"type": "autobatch", "batch_size": eff_bs, "resumed": True})
    elif config.batch_size is not None and config.batch_size > 0:
        eff_bs = int(config.batch_size)
        cb({
            "type": "autobatch",
            "batch_size": eff_bs,
            "manual_override": True,
        })
    else:
        from bittrainer.autobatch import determine_batch_size

        def _probe_progress(attempt: int, candidate: int, cap: int, status: str) -> None:
            cb({
                "type": "training_progress", "stage": "autobatch",
                "status_text": f"Probing batch size (try {attempt}: {candidate}/{cap} â€” {status})",
            })

        em.stage(Stage.autobatch, "Probing optimal batch size")
        auto_result = determine_batch_size(
            model, bucket_counts, device, dtype=dtype, vram_fraction=config.vram_fraction,
            use_ema=config.use_ema, memory_format=memory_format,
            progress_callback=_probe_progress,
        )
        eff_bs = auto_result["batch_size"]
        cb({"type": "autobatch", **auto_result})

    class_counts = train_ds.get_class_counts()
    total_raw = sum(class_counts.values())

    # --- Class imbalance strategy: resample (replicate) vs reweight (natural
    # sampling + effective-number weights). Mutually exclusive, so no double
    # correction. ---
    balance_mode = _resolve_class_balance(config, class_counts)
    class_weights: torch.Tensor | None = None
    if not config.multi_label and balance_mode == "reweight":
        train_ds.set_natural_sampling(True)
        class_weights = _effective_number_weights(
            class_counts, config.num_classes, config.class_balance_beta, device,
        )
        cb({
            "type": "training_progress", "stage": "preparing",
            "status_text": "Class balance: reweight (natural sampling + effective-number weights)",
        })

    # --- Dynamic per-class loss weighting (ISSUE-0392): soft per-class
    # early-stop. Composes on top of whatever class_weights resolved above
    # (reweight vector, or all-ones = no-op at start). Single-label only. ---
    dcw_controller = _build_dcw_controller(config, class_weights, device)
    if dcw_controller is not None:
        if resume_state is not None and resume_state.get("dcw") is not None:
            # Restore the controller's full mutable history so its next update
            # continues exactly where the interrupted run left off.
            base = class_weights if class_weights is not None else torch.ones(config.num_classes, device=device)
            dcw_controller = DynamicClassWeightController.from_dict(resume_state["dcw"], base)
        class_weights = dcw_controller.current_weights()
        cb({
            "type": "training_progress", "stage": "preparing",
            "status_text": (
                f"Dynamic per-class loss weighting ON "
                f"(trigger={config.dcw_metric}, patience={config.dcw_patience})"
            ),
        })
    if resume_state is not None and resume_state.get("class_weights") is not None and dcw_controller is None:
        # Reweight (non-dcw) vector: recomputed deterministically above, but adopt
        # the backed-up tensor verbatim so a resume is bit-identical.
        class_weights = resume_state["class_weights"].to(device)

    # --- MixUp/CutMix gate: skip on tiny datasets where the full aug stack
    # over-regularises, and for multi-label (single-label soft-target path only). ---
    mixup_enabled = (
        config.use_mixup and not config.multi_label and total_raw >= config.mixup_min_images
    )

    # --- SWA: average weights over the cosine tail (epoch >= swa_start_epoch). ---
    swa = _SWA() if (config.use_swa and not config.multi_label) else None
    swa_start_epoch = int(config.swa_start_frac * config.max_epochs)
    if swa is not None and resume_state is not None and resume_state.get("swa") is not None:
        swa.load_state_dict(resume_state["swa"]["avg"], resume_state["swa"]["n"])

    # Optimizer (LLRD param groups when config.llrd, else flat). Built once over
    # the fully-unfrozen model â€” the warm head means no epoch-1 rebuild.
    optimizer = _make_optimizer(model, config)
    # The group trainer never recreates the scheduler mid-run, so T_max is always
    # config.max_epochs; carried in the backup for schema symmetry with the
    # binary trainer's epoch-1 unfreeze rebuild.
    scheduler_t_max = int(resume_state["scheduler_t_max"]) if resume_state is not None else config.max_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_t_max)

    # EMA tracks all params from the start; freeze/unfreeze only affects which
    # ones receive gradient updates, but the EMA still mirrors the live tensor
    # values, which is what we want for inference-time smoothing.
    ema = ModelEMA(model, decay=config.ema_decay) if config.use_ema else None

    if resume_state is not None:
        restore_optimizer_state(resume_state, optimizer, scheduler, device)
        if ema is not None and resume_state.get("ema") is not None:
            ema.load_full_state_dict(resume_state["ema"])

    # fwd_model shares parameters with the eager model â€” optimizer, EMA and
    # checkpoint saves keep operating on `model`; only forward calls go
    # through the compiled wrapper.
    fwd_model, compiled = maybe_compile(model, enabled=config.use_compile, cb=cb)
    if compiled and not prewarm_compile(
        fwd_model, bucket_counts, eff_bs, device, dtype,
        memory_format=memory_format, cb=cb,
    ):
        fwd_model = model

    best_val_macro_f1 = -1.0
    best_val_qwk = -1.0
    best_validation_score = -1.0
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path = None
    best_metrics: dict = {}
    global_step = 0
    start_epoch = 0
    # Greedy-soup candidate pool (top-N epochs by selection score, kept on disk).
    soup_pool: list[tuple[float, int, str]] = []
    soup_dir = checkpoint_dir / "soup_cands"
    # Mid-epoch resume state (one-shot; cleared after the first resumed epoch).
    resume_batch_in_epoch = 0
    resume_schedule: list[list[int]] | None = None
    resume_rng_epoch_start: dict | None = None
    resume_rng_now: dict | None = None

    if resume_state is not None:
        best = resume_state["best"]
        best_val_macro_f1 = best["best_val_macro_f1"]
        best_val_qwk = best["best_val_qwk"]
        best_validation_score = best["best_validation_score"]
        best_epoch = best["best_epoch"]
        patience_counter = best["patience_counter"]
        best_checkpoint_path = best["best_checkpoint_path"]
        best_metrics = dict(best.get("best_metrics") or {})
        soup_pool = [tuple(t) for t in (resume_state.get("soup_pool") or [])]
        global_step = int(resume_state.get("global_step", 0))
        resume_batch_in_epoch = int(resume_state.get("batch_in_epoch", 0))
        resume_rng_epoch_start = resume_state.get("rng_epoch_start")
        resume_rng_now = resume_state.get("rng_now")
        # ``epoch`` in the envelope is always the epoch to resume INTO (a
        # boundary backup stores epoch+1, so it never re-runs a finished epoch).
        start_epoch = int(resume_state["epoch"])
        if resume_batch_in_epoch > 0 and not resume_bs_changed:
            # Mid-epoch backup: continue the SAME epoch from the stored schedule.
            resume_schedule = resume_state.get("batch_schedule")
        else:
            # Epoch boundary, or a batch-size change that invalidated the
            # schedule → start ``start_epoch`` fresh from batch 0.
            resume_batch_in_epoch = 0
        cb({
            "type": "training_resumed",
            "resumed_from": str(config.resume_from),
            "epoch": start_epoch,
            "global_step": global_step,
            "best_val_macro_f1": best_val_macro_f1,
            "best_validation_score": best_validation_score,
            "best_val_qwk": best_val_qwk if config.ordinal else None,
            "best_epoch": best_epoch + 1,
        })

    def _collect_state(cur_epoch: int, batch_in_epoch: int, schedule) -> dict:
        """Assemble the schema-v1 backup envelope from the live run state."""
        swa_payload = None
        if swa is not None and swa.state_dict() is not None:
            swa_payload = {"avg": swa.state_dict(), "n": swa.n}
        return collect_epoch_state(
            fingerprint=fingerprint, trainer="group", epoch=cur_epoch,
            batch_in_epoch=batch_in_epoch, global_step=global_step, eff_bs=eff_bs,
            scheduler_t_max=scheduler_t_max,
            model=model, optimizer=optimizer, scheduler=scheduler,
            best={
                "best_val_macro_f1": best_val_macro_f1,
                "best_val_qwk": best_val_qwk,
                "best_validation_score": best_validation_score,
                "best_epoch": best_epoch,
                "patience_counter": patience_counter,
                "best_checkpoint_path": best_checkpoint_path,
                "best_metrics": best_metrics,
            },
            ema=ema.full_state_dict() if ema is not None else None,
            swa=swa_payload,
            soup_pool=[list(t) for t in soup_pool],
            dcw=dcw_controller.to_dict() if dcw_controller is not None else None,
            class_weights=class_weights.detach().cpu() if class_weights is not None else None,
            resolved=_resolved_snapshot(config),
            rng_epoch_start=rng_epoch_start,
            rng_now=capture_rng_states(device),
            batch_schedule=(
                [list(b) for b in schedule]
                if (schedule is not None and batch_in_epoch > 0) else None
            ),
            head_hidden_size=head_hidden_size,
        )

    rng_epoch_start: dict | None = None
    epoch = start_epoch - 1  # so epochs_completed is defined if the loop is empty
    _exc_epoch = start_epoch

    with coordinator.backup_on_exception(lambda: _collect_state(_exc_epoch, 0, None)):
        for epoch in range(start_epoch, config.max_epochs):
            _exc_epoch = epoch
            if stop_now_event is not None and stop_now_event.is_set():
                logger.info("Stop-now requested before epoch %d â€” running final comparison", epoch)
                cb({"type": "stop_now", "epoch": epoch, "max_epochs": config.max_epochs})
                break
            if stop_event is not None and stop_event.is_set():
                logger.info("Graceful stop requested after epoch %d â€” running final comparison", epoch)
                cb({"type": "graceful_stop", "epoch": epoch, "max_epochs": config.max_epochs})
                break
            if coordinator.paused:
                # Pause at an epoch boundary (before any training this epoch):
                # back up a clean batch_in_epoch=0 snapshot and stop.
                path = coordinator.save(_collect_state(epoch, 0, None), reason="pause")
                return _paused_result(epoch, global_step, path)

            # Exact mid-epoch continuation only for the FIRST resumed epoch.
            mid_resume = (
                epoch == start_epoch and resume_schedule is not None and resume_batch_in_epoch > 0
            )

            # Capture the epoch-start RNG BEFORE reshuffle so the sample layout is
            # reproducible; on a mid-epoch resume restore the stored epoch-start
            # RNG instead so reshuffle rebuilds the identical sample list.
            if mid_resume:
                restore_rng_states(resume_rng_epoch_start, device)
                rng_epoch_start = resume_rng_epoch_start
            else:
                if epoch == start_epoch and resume_rng_now is not None:
                    # Epoch-boundary resume: continue the RNG stream from the
                    # backup point so this fresh epoch matches the control run.
                    restore_rng_states(resume_rng_now, device)
                rng_epoch_start = capture_rng_states(device)
            train_ds.reshuffle()

            if epoch == 0:
                cb({
                    "type": "training_progress", "stage": "preparing",
                    "status_text": f"Batch size {eff_bs} â€” spawning data workers",
                })

            # Build dataloaders. The batch order is materialised into a fixed
            # schedule so a backup can replay it exactly on resume.
            collate_fn = _collate_multilabel_batch if config.multi_label else _collate_bucket_batch
            train_sampler = build_group_bucket_sampler(train_ds, batch_size=eff_bs)
            if mid_resume:
                schedule = [list(b) for b in resume_schedule]
                loader_batches = schedule[resume_batch_in_epoch:]
                start_batch = resume_batch_in_epoch
                # Jump the augmentation/mixup RNG to the mid-epoch backup point.
                restore_rng_states(resume_rng_now, device)
            else:
                schedule = [list(b) for b in train_sampler]
                loader_batches = schedule
                start_batch = 0
            lk = loader_kwargs(config.dataloader_workers)
            if lk["num_workers"] == 0:
                # workers=0 (bit-exact resume mode): a DataLoader iterator draws a
                # base-seed from the GLOBAL torch RNG on creation, even with no
                # workers. A mid-epoch resume builds a fresh iterator, so that
                # extra draw would desync the augmentation stream from an
                # uninterrupted run. A private generator keeps the base-seed off
                # the global stream, leaving it purely augmentation-driven.
                lk["generator"] = torch.Generator().manual_seed(0)
            train_loader = DataLoader(
                train_ds, batch_sampler=_FixedBatchSampler(loader_batches),
                collate_fn=collate_fn, **lk,
            )
            val_sampler = build_group_bucket_sampler(val_ds, batch_size=eff_bs)
            val_loader = DataLoader(
                val_ds, batch_sampler=val_sampler, collate_fn=collate_fn, **lk,
            )

            # One-shot: subsequent epochs are ordinary.
            resume_schedule = None

            # Train
            epoch_start_mono = time.monotonic()

            def _on_step(step: int, total_steps: int, avg_loss: float) -> None:
                elapsed = time.monotonic() - epoch_start_mono
                throughput = step / elapsed if elapsed > 0 else None
                eta_seconds = (total_steps - step) / throughput if throughput and throughput > 0 else None
                cb({
                    "type": "training_progress",
                    "stage": "training",
                    "status_text": f"Training (epoch {epoch + 1}/{config.max_epochs}, step {step}/{total_steps})",
                    "epoch": epoch + 1,
                    "max_epochs": config.max_epochs,
                    "step": step,
                    "total_steps": total_steps,
                    "eta_seconds": eta_seconds,
                    "throughput": throughput,
                    "throughput_unit": "batch/s",
                    "images_per_s": round(throughput * eff_bs, 1) if throughput else None,
                    "batch_size": eff_bs,
                    "train_loss": round(avg_loss, 4),
                    "best_val_macro_f1": best_val_macro_f1 if best_val_macro_f1 >= 0 else None,
                    "best_validation_score": best_validation_score if best_validation_score >= 0 else None,
                    "validation_metric": _primary_validation_metric(config),
                    "best_val_qwk": (
                        best_val_qwk if config.ordinal and best_val_qwk > -1.0 else None
                    ),
                    "best_epoch": best_epoch + 1 if best_val_macro_f1 >= 0 else None,
                })

            def _boundary_hook(num_batches: int) -> str | None:
                # Fires at every gradient-accumulation boundary. Owns the global
                # optimizer-step counter and the pause/periodic backup cadence.
                nonlocal global_step
                global_step += 1
                return coordinator.on_boundary(
                    lambda: _collect_state(epoch, num_batches, schedule), global_step,
                )

            train_loss, per_class_train_loss = _train_one_epoch(
                fwd_model, train_loader, optimizer, config, device, dtype,
                use_soft_targets=use_soft,
                step_callback=_on_step,
                stop_now_event=stop_now_event,
                ema=ema,
                class_weights=class_weights,
                mixup_enabled=mixup_enabled,
                pause_event=pause_event,
                boundary_hook=_boundary_hook,
                start_batch=start_batch,
            )

            if coordinator.paused:
                # Pause fired mid-epoch — the boundary hook already wrote the
                # backup. Return WITHOUT SWA / greedy soup / promotion so a pause
                # can never ship or promote a model.
                return _paused_result(epoch, global_step, coordinator.last_backup_path)

            if stop_now_event is not None and stop_now_event.is_set():
                cb({
                    "type": "stop_now",
                    "epoch": epoch + 1,
                    "max_epochs": config.max_epochs,
                    "status_text": f"Stop-now triggered mid-epoch {epoch + 1} â€” finishing up",
                })
            scheduler.step()

            # Capture the post-epoch weights into the SWA running average over the
            # cosine tail (uniform average; LayerNorm backbone needs no BN update).
            if swa is not None and epoch >= swa_start_epoch:
                swa.update(model)

            # Validate (against EMA weights when enabled â€” they generalise better)
            em.stage(
                Stage.validating,
                f"Validating (epoch {epoch + 1}/{config.max_epochs})",
                epoch=epoch + 1, max_epochs=config.max_epochs,
            )
            eval_model = ema.module if ema is not None else fwd_model
            if config.multi_label:
                val_metrics = _evaluate(
                    eval_model, val_loader, config.num_classes, device, dtype,
                    multi_label=True,
                    ordinal=config.ordinal,
                    none_index=_resolve_none_index(config.class_names),
                    channels_last=config.channels_last,
                )
            else:
                # Score the epoch under the decode the model ships with
                # (temperature + __none__ bias + ordinal cut-points) so selection
                # and the shipped model agree on what "best" means.
                epoch_logits, epoch_labels = _collect_val_logits(
                    eval_model, val_loader, config, device, dtype,
                )
                val_metrics = _shipped_decode_metrics(
                    epoch_logits, epoch_labels, config,
                    _resolve_none_index(config.class_names),
                )
                # Skin Tone V2 dual-view (ISSUE-0217, spec §8): score the
                # colour-normalised view and the averaged-logit combination as
                # separate tracks. Selection stays on the ORIGINAL view — raw
                # inference is the deployment default (spec §9).
                if getattr(val_ds, "skin_tone_views", None) is not None:
                    val_ds.skin_tone_force_view = True
                    try:
                        # A FRESH in-process loader is load-bearing: val_loader's
                        # persistent workers hold a dataset copy pickled before
                        # the flag flip and would silently re-score the ORIGINAL
                        # view. num_workers=0 runs __getitem__ in this process,
                        # so the flag is guaranteed visible.
                        view_loader = DataLoader(
                            val_ds, batch_sampler=val_sampler, collate_fn=collate_fn,
                            num_workers=0, pin_memory=True,
                        )
                        view_logits, view_labels = _collect_val_logits(
                            eval_model, view_loader, config, device, dtype,
                        )
                    finally:
                        val_ds.skin_tone_force_view = False
                    view_metrics = _shipped_decode_metrics(
                        view_logits, view_labels, config,
                        _resolve_none_index(config.class_names),
                    )
                    val_metrics["macro_f1_original"] = val_metrics["macro_f1"]
                    val_metrics["macro_f1_normalized"] = view_metrics["macro_f1"]
                    # Averaged logits are only meaningful if both passes saw the
                    # samples in the same order (val sampler is deterministic;
                    # guard anyway).
                    if torch.equal(epoch_labels, view_labels):
                        dual_metrics = _shipped_decode_metrics(
                            (epoch_logits + view_logits) / 2.0, epoch_labels, config,
                            _resolve_none_index(config.class_names),
                        )
                        val_metrics["macro_f1_dual"] = dual_metrics["macro_f1"]
            val_metrics["train_loss"] = train_loss

            val_macro_f1 = val_metrics["macro_f1"]
            val_qwk = val_metrics.get("qwk", 0.0)

            selected_score = _metric_score(val_metrics, config)
            # Require a minimum gain so epoch-to-epoch noise can't flip the
            # selection onto a marginally-higher but less robust checkpoint. The
            # first epoch always clears this (incumbent starts at -1.0).
            improved = selected_score > best_validation_score + _SELECTION_MIN_DELTA
            if improved:
                best_val_macro_f1 = val_macro_f1
                best_val_qwk = val_qwk
                best_validation_score = selected_score
                best_epoch = epoch
                patience_counter = 0
                best_metrics = val_metrics.copy()

                ckpt_path = checkpoint_dir / "candidate.pt"
                # When EMA is active, persist the EMA weights as the primary
                # state_dict (downstream inference loads this key unchanged). Raw
                # weights survive under model_state_dict for diagnostic purposes.
                primary_state = ema.state_dict() if ema is not None else model.state_dict()
                ckpt_meta = {
                    "state_dict": primary_state,
                    "num_classes": config.num_classes,
                    "model_size": config.backbone_variant,
                    "class_names": list(config.class_names),
                    "validation_metric": _primary_validation_metric(config),
                    **_spatial_ckpt_meta(config),
                }
                if head_hidden_size is not None:
                    ckpt_meta["head_hidden_size"] = head_hidden_size
                if ema is not None:
                    ckpt_meta["model_state_dict"] = model.state_dict()
                    ckpt_meta["ema_decay"] = config.ema_decay
                if config.multi_label:
                    ckpt_meta["multi_label"] = True
                torch.save(ckpt_meta, ckpt_path)
                best_checkpoint_path = str(ckpt_path)
            else:
                patience_counter += 1

            # Dynamic per-class loss weighting: fold this epoch's per-class val
            # signal into the controller and reassign class_weights for the NEXT
            # epoch. The loss_fn is rebuilt fresh inside _train_one_epoch, so the
            # updated tensor takes effect immediately with no other plumbing.
            if dcw_controller is not None:
                class_weights = dcw_controller.update(
                    val_metrics.get("per_class_f1", {}),
                    val_metrics.get("per_class_val_loss", {}),
                )

            # Per-epoch snapshot dump for snapshot-ensemble experiments (ISSUE-0392
            # follow-up: bank each class at its own val-F1 peak). Writes the
            # deployable weights every epoch when snapshot_dir is set; off by default.
            if config.snapshot_dir:
                snap_dir = Path(config.snapshot_dir)
                snap_dir.mkdir(parents=True, exist_ok=True)
                snap_state = ema.state_dict() if ema is not None else model.state_dict()
                snap_meta = {
                    "state_dict": snap_state,
                    "num_classes": config.num_classes,
                    "model_size": config.backbone_variant,
                    "class_names": list(config.class_names),
                    **_spatial_ckpt_meta(config),
                }
                if head_hidden_size is not None:
                    snap_meta["head_hidden_size"] = head_hidden_size
                torch.save(snap_meta, snap_dir / f"epoch_{epoch + 1:03d}.pt")

            # Track this epoch as a greedy-soup candidate (top-N by selection score).
            if config.use_greedy_soup:
                _update_soup_pool(
                    soup_pool, soup_dir, selected_score, epoch,
                    ema.state_dict() if ema is not None else model.state_dict(),
                    config.soup_max_candidates,
                )

            epoch_msg = _build_epoch_message(
                epoch=epoch,
                config=config,
                train_loss=train_loss,
                val_metrics=val_metrics,
                best_val_macro_f1=best_val_macro_f1,
                best_val_qwk=best_val_qwk,
                selected_score=selected_score,
                best_validation_score=best_validation_score,
                best_epoch=best_epoch,
                per_class_train_loss=per_class_train_loss,
                elapsed_seconds=time.monotonic() - epoch_start_mono,
                dcw_multipliers=(
                    dcw_controller.multipliers() if dcw_controller is not None else None
                ),
            )
            cb(epoch_msg)

            # Epoch-boundary backup: the cleanest resume point (scheduler stepped,
            # best/dcw/soup updated). Stored as epoch+1/batch_in_epoch=0 so a
            # resume starts the NEXT epoch fresh, never re-running this one.
            if coordinator.enabled:
                coordinator.save(_collect_state(epoch + 1, 0, None), reason="periodic")

            if patience_counter >= config.patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, config.patience)
                break

        # --- SWA finalisation: materialise the averaged weights, evaluate them, and
        # adopt as the candidate only when they beat the best single-epoch checkpoint
        # on the selection score (so calibration/cut-points then fit on them). ---
        if swa is not None and swa.n >= 2 and best_checkpoint_path:
            try:
                swa_sd_cpu = swa.state_dict()
                model.load_state_dict({k: v.to(device) for k, v in swa_sd_cpu.items()})
                # SWA competes against per-epoch bests, so it must be scored under
                # the same shipped decode (SWA is single-label-only by the gate).
                swa_logits, swa_labels = _collect_val_logits(model, val_loader, config, device, dtype)
                swa_metrics = _shipped_decode_metrics(
                    swa_logits, swa_labels, config, _resolve_none_index(config.class_names),
                )
                swa_score = _metric_score(swa_metrics, config)
                cb({
                    "type": "training_progress", "stage": "validating",
                    "status_text": (
                        f"SWA ({swa.n} snapshots): score {swa_score:.4f} "
                        f"vs best {best_validation_score:.4f}"
                    ),
                })
                if swa_score > best_validation_score:
                    ckpt_meta = {
                        "state_dict": swa_sd_cpu,
                        "num_classes": config.num_classes,
                        "model_size": config.backbone_variant,
                        "class_names": list(config.class_names),
                        "validation_metric": _primary_validation_metric(config),
                        **_spatial_ckpt_meta(config),
                    }
                    if head_hidden_size is not None:
                        ckpt_meta["head_hidden_size"] = head_hidden_size
                    ckpt_path = checkpoint_dir / "candidate.pt"
                    torch.save(ckpt_meta, ckpt_path)
                    best_checkpoint_path = str(ckpt_path)
                    best_metrics = swa_metrics.copy()
                    best_val_macro_f1 = swa_metrics["macro_f1"]
                    best_val_qwk = swa_metrics.get("qwk", 0.0)
                    best_validation_score = swa_score
                    logger.info("SWA weights adopted (score %.4f)", swa_score)
            except Exception:
                logger.warning("SWA evaluation failed; keeping best single-epoch checkpoint", exc_info=True)

        # --- Greedy weight soup: average the strongest epochs into ONE model, keeping
        # only additions that don't lower the val selection score, and adopt the soup
        # only when it strictly beats the best single-epoch checkpoint. One deployable
        # model, no extra inference cost; safe for all group types (never worse on
        # val by construction). ---
        if config.use_greedy_soup and len(soup_pool) >= 2 and best_checkpoint_path:
            none_index = _resolve_none_index(config.class_names)

            def _soup_metrics(state: dict) -> dict:
                model.load_state_dict({k: v.to(device) for k, v in state.items()})
                if config.multi_label:
                    return _evaluate(
                        model, val_loader, config.num_classes, device, dtype,
                        multi_label=True, ordinal=config.ordinal,
                        none_index=none_index, channels_last=config.channels_last,
                    )
                logits, labels = _collect_val_logits(model, val_loader, config, device, dtype)
                return _shipped_decode_metrics(logits, labels, config, none_index)

            try:
                candidates = [
                    (score, torch.load(path, map_location="cpu")) for score, _ep, path in soup_pool
                ]
                soup_state, soup_score, accepted = greedy_soup(
                    candidates, lambda s: _metric_score(_soup_metrics(s), config),
                )
                cb({
                    "type": "training_progress", "stage": "validating",
                    "status_text": (
                        f"Greedy soup ({len(accepted)}/{len(candidates)} epochs): "
                        f"score {soup_score:.4f} vs best {best_validation_score:.4f}"
                    ),
                })
                if soup_score > best_validation_score:
                    soup_metrics = _soup_metrics(soup_state)
                    ckpt_meta = {
                        "state_dict": {k: v.detach().cpu() for k, v in soup_state.items()},
                        "num_classes": config.num_classes,
                        "model_size": config.backbone_variant,
                        "class_names": list(config.class_names),
                        "validation_metric": _primary_validation_metric(config),
                        **_spatial_ckpt_meta(config),
                    }
                    if head_hidden_size is not None:
                        ckpt_meta["head_hidden_size"] = head_hidden_size
                    if config.multi_label:
                        ckpt_meta["multi_label"] = True
                    ckpt_path = checkpoint_dir / "candidate.pt"
                    torch.save(ckpt_meta, ckpt_path)
                    best_checkpoint_path = str(ckpt_path)
                    best_metrics = soup_metrics.copy()
                    best_val_macro_f1 = soup_metrics.get("macro_f1", best_val_macro_f1)
                    best_val_qwk = soup_metrics.get("qwk", best_val_qwk)
                    best_validation_score = soup_score
                    logger.info("Greedy soup adopted (%d epochs, score %.4f)", len(accepted), soup_score)
            except Exception:
                logger.warning("Greedy soup failed; keeping best single-epoch checkpoint", exc_info=True)
            finally:
                for _s, _e, _p in soup_pool:
                    try:
                        Path(_p).unlink()
                    except OSError:
                        pass
                try:
                    soup_dir.rmdir()
                except OSError:
                    pass

        result = _compare_promote_finalize(
            config,
            candidate_path=best_checkpoint_path,
            best_metrics=best_metrics,
            candidate_macro_f1=best_val_macro_f1,
            candidate_qwk=best_val_qwk,
            best_epoch_display=best_epoch + 1,
            epochs_completed=epoch + 1,
            val_loader=val_loader,
            device=device, dtype=dtype,
            checkpoint_dir=checkpoint_dir,
            class_counts=train_ds.get_class_counts(),
            effective_class_counts=train_ds.get_effective_class_counts(),
            total_raw=total_raw,
            cb=cb,
        )

    # Training completed successfully (no pause, no exception): the backups are
    # obsolete — clear them so a later run doesn't resume a finished job.
    coordinator.delete_backups()
    return result
