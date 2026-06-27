"""Training loop for ConvNeXt V2 multi-class group classifiers."""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import numpy as np
from adv_optm import Prodigy_adv
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from bittrainer.class_balance import cap_weight_ratio
from bittrainer.ema import ModelEMA
from bittrainer.group_dataset import (
    GroupDataset,
    build_group_bucket_sampler,
)
from bittrainer.group_validation import (
    compute_multiclass_metrics,
    compute_multilabel_metrics,
    compute_none_metrics,
    compute_ordinal_metrics,
    find_ordinal_cut_points,
    find_per_class_thresholds,
    ordinal_decode,
)
from bittrainer.losses import AsymmetricLoss, FocalLoss
from bittrainer.embedding_cache import EmbeddingCache
from bittrainer.head_probe import (
    prepare_head_probe_tensors,
    train_head_probe,
    train_head_probe_from_tensors,
)
from bittrainer.model import (
    backbone_feature_hash,
    build_llrd_param_groups,
    create_model,
    load_checkpoint,
    unfreeze_backbone,
)
from bittrainer.promotion import (
    PromotionReason,
    decide_promotion,
)

logger = logging.getLogger(__name__)

_NONE_CLASS_NAME = "__none__"
_ORDINAL_SIGMA_CANDIDATES = [round(i / 10, 3) for i in range(11)]
_LABEL_SMOOTHING_CANDIDATES = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
_NONE_F1_WEIGHT = 0.10
_NONE_RECALL_REGRESSION_TOLERANCE = 0.02
_REAL_MACRO_F1_REGRESSION_TOLERANCE = 0.01
_NONE_LOGIT_BIAS_GRID = [round(i * 0.025, 3) for i in range(21)]

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
_SELECTION_SECONDARY_WEIGHT = 0.40  # weight on macro-F1 in the composite (0 = pure ordinal metric)
_SELECTION_MIN_DELTA = 0.002        # min composite gain required to replace the incumbent best
_TEMPERATURE_GRID = [0.75, 0.85, 1.0, 1.15, 1.3, 1.5]


def _resolve_none_index(class_names: list[str]) -> int:
    """Return the position of the ``__none__`` class, or -1 if absent.

    ``__none__`` is a valid output class the model learns to predict, but it
    must be excluded from any code path that assumes class indices are
    positions on an ordinal scale (Gaussian soft-target smoothing, ordinal
    validation metrics, etc.).
    """
    try:
        return class_names.index(_NONE_CLASS_NAME)
    except ValueError:
        return -1


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
    multi_label: bool = False
    # Oversample __none__ up to 1:1 with the combined positives when it is
    # under-populated (negatives are outside class balancing; never downsampled).
    oversample_none: bool = True
    extra_paths_train: dict[str, list[str]] = field(default_factory=dict)
    extra_paths_val: dict[str, list[str]] = field(default_factory=dict)
    soft_aliases: dict = field(default_factory=dict)
    device: str = "cuda"
    dtype: str = "bfloat16"
    from_scratch: bool = False
    best_model_name: str = "best.pt"
    checkpoint_dir: str | None = None
    skin_normalise: bool = False
    face_model_path: str = ""
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
    # Ceiling on how far class balancing may correct, applied to BOTH mechanisms:
    # in "resample" the minority replication factor is capped at this, in "reweight"
    # the max/min effective-number loss-weight ratio is. Bounds memorisation from
    # extreme over-replication and respects the genuine (deployment-matched) prior —
    # tuned for real-world / weighted F1 rather than balanced accuracy. 0 = uncapped.
    class_balance_max_ratio: float = 4.0
    use_focal: bool = False
    focal_gamma: float = 2.0
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
    data_quality_warnings: list[dict] = field(default_factory=list)
    # torch.compile for the full fine-tune forward/backward. Falls back to
    # eager (with a status message) when triton is unavailable.
    use_compile: bool = True
    # NHWC layout â€” ConvNeXt stem/downsample/dwconv save permute traffic.
    channels_last: bool = True
    # Gradient accumulation escape hatch: optimizer steps every N batches.
    grad_accum_steps: int = 1
    # --- OFTv2-style orthogonal fine-tuning (training_mode="oft"). Defaults are
    # the fast path for normal incremental updates; see bittrainer/oft.py. ---
    # Backend: "cayley_neumann" (default, fast, guarded by oft_clipped_norm),
    # "cayley" (exact solve), or "cans" (slower, lower orthogonalisation error).
    oft_backend: str = "cayley_neumann"
    # Clipped OFT norm — Neumann-divergence guard. Float threshold (~0.95) or
    # None to disable (only safe for backends that don't rely on Neumann).
    oft_clipped_norm: float | None = 0.95
    # Number of block-diagonal orthogonal blocks per wrapped Linear (auto-reduced
    # to the nearest divisor of each layer's output dim).
    oft_blocks: int = 8
    # Truncated Neumann series length (cayley_neumann / cans seed).
    oft_neumann_terms: int = 6
    # Newton-Schulz polar refinement iterations (cans only).
    oft_cans_iters: int = 3
    # Experimental DoRA-OFT (OFT rotation + learned magnitude). Not default; not
    # surfaced in the main UI.
    oft_dora: bool = False


def _primary_validation_metric(config: GroupTrainConfig) -> str:
    if config.ordinal:
        if config.validation_metric == "guarded_qwk":
            return "guarded_qwk"
        return "qwk" if config.validation_metric == "qwk" else "macro_f1"
    return "macro_f1"


def _has_none_class(config: GroupTrainConfig) -> bool:
    return _resolve_none_index(config.class_names) >= 0


def _guarded_metric_enabled(config: GroupTrainConfig) -> bool:
    return _has_none_class(config) and not config.multi_label


def _guarded_score(metrics: dict, config: GroupTrainConfig) -> float:
    none_f1 = float(metrics.get("none_f1") or 0.0)
    if config.ordinal and _primary_validation_metric(config) == "guarded_qwk":
        return float(metrics.get("qwk") or 0.0) + _NONE_F1_WEIGHT * none_f1
    return float(metrics.get("macro_f1") or 0.0) + _NONE_F1_WEIGHT * none_f1


def _ordinal_primary_score(metrics: dict, config: GroupTrainConfig) -> float:
    """The ordinal driver metric (QWK, plus the __none__ guard term when the
    group uses guarded_qwk) BEFORE it is composited with macro-F1."""
    qwk = float(metrics.get("qwk") or 0.0)
    if _primary_validation_metric(config) == "guarded_qwk" and _guarded_metric_enabled(config):
        return qwk + _NONE_F1_WEIGHT * float(metrics.get("none_f1") or 0.0)
    return qwk


def _composite_selection_score(primary: float, macro_f1: float) -> float:
    """Weighted harmonic mean of the ordinal primary metric and macro-F1.

    The harmonic mean (as in F1 itself) is dominated by whichever component is
    weakest, so a marginal QWK gain cannot buy back a large macro-F1 collapse.
    Degenerate inputs (<= 0, where the harmonic mean is undefined) fall back to
    the weighted arithmetic mean so the ordering stays well-defined.
    """
    p = max(0.0, primary)
    s = max(0.0, macro_f1)
    w_secondary = _SELECTION_SECONDARY_WEIGHT
    w_primary = 1.0 - w_secondary
    if p <= 0.0 or s <= 0.0:
        return w_primary * p + w_secondary * s
    return 1.0 / (w_primary / p + w_secondary / s)


def _metric_score(metrics: dict, config: GroupTrainConfig) -> float:
    # Ordinal groups: select on a composite of the ordinal metric (QWK or
    # guarded QWK) and macro-F1 so a marginal QWK gain cannot override an
    # exact-match (macro-F1) collapse. See the _SELECTION_* constants above.
    if config.ordinal and _primary_validation_metric(config) in ("qwk", "guarded_qwk"):
        primary = _ordinal_primary_score(metrics, config)
        macro_f1 = float(metrics.get("macro_f1") or 0.0)
        return _composite_selection_score(primary, macro_f1)
    # Non-ordinal groups: macro-F1 (with the __none__ guard term when present).
    if _guarded_metric_enabled(config) and not config.ordinal:
        return _guarded_score(metrics, config)
    # Ordinal-as-macro_f1 or any other fallthrough: the primary metric as-is.
    metric = _primary_validation_metric(config)
    value = metrics.get("qwk" if metric == "qwk" else "macro_f1")
    return float(value) if value is not None else 0.0


def _score_metric_label(config: GroupTrainConfig) -> str:
    metric = _primary_validation_metric(config)
    if metric == "guarded_qwk":
        return "guarded_qwk"
    if _guarded_metric_enabled(config) and not config.ordinal:
        return "guarded_macro_f1"
    return metric


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


def _make_optimizer(model: nn.Module, config: GroupTrainConfig) -> Prodigy_adv:
    if config.llrd:
        params = build_llrd_param_groups(model, config.llrd_decay)
    else:
        params = model.parameters()
    return Prodigy_adv(
        params, lr=1.0, d_coef=0.9,
        weight_decay=0.01, betas=(0.9, 0.999),
        kourkoutas_beta=True, k_warmup_steps=50,
        cautious_wd=True,
    )


# Step-callback throttle for hot training loops (~4 Hz keeps the UI live
# without flooding the multiprocessing queue).
_STEP_REPORT_INTERVAL = 0.25


def _get_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _collate_bucket_batch(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


def _collate_multilabel_batch(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return images, labels


# ---------------------------------------------------------------------------
# Soft target construction (ordinal + soft aliases)
# ---------------------------------------------------------------------------


def _build_gaussian_kernel(
    num_classes: int,
    sigma: float = 1.0,
    *,
    none_index: int = -1,
) -> torch.Tensor:
    """Build a Gaussian smoothing kernel for ordinal classes.

    kernel[i, j] = exp(-(i-j)^2 / (2*sigma^2)), then normalised per row.

    When ``none_index >= 0`` the corresponding class (``__none__``) is treated
    as a separate semantic category, not a position on the ordinal scale.
    Its row and column are zeroed and the diagonal entry is set to 1, so no
    probability bleeds between ``__none__`` and its numeric neighbours during
    soft-target smoothing â€” without this, the model learns that ``__none__``
    is adjacent to the lowest ordinal class (e.g. ``__none__`` â†” "Augmented
    Breasts" or ``__none__`` â†” "0-year-old"), which corrupts predictions on
    visually-empty inputs.
    """
    indices = torch.arange(num_classes, dtype=torch.float32)
    diffs = indices.unsqueeze(0) - indices.unsqueeze(1)
    kernel = torch.exp(-diffs ** 2 / (2 * sigma ** 2))
    if 0 <= none_index < num_classes:
        kernel[none_index, :] = 0.0
        kernel[:, none_index] = 0.0
        kernel[none_index, none_index] = 1.0
    kernel = kernel / kernel.sum(dim=1, keepdim=True)
    return kernel


def _build_soft_targets(
    labels: torch.Tensor,
    num_classes: int,
    *,
    ordinal: bool = False,
    ordinal_sigma: float = 1.0,
    label_smoothing: float = 0.0,
    soft_aliases: dict | None = None,
    none_index: int = -1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Convert integer labels to soft target vectors.

    1. Start with one-hot
    2. Apply ordinal Gaussian smoothing (if ordinal and sigma > 0),
       excluding ``none_index``
    3. Apply global label smoothing for non-ordinal softmax groups, excluding
       ``none_index`` from both directions
    4. Apply soft aliases
    """
    batch_size = labels.shape[0]
    targets = torch.zeros(batch_size, num_classes, device=device)
    targets.scatter_(1, labels.unsqueeze(1), 1.0)

    if ordinal and num_classes > 2 and ordinal_sigma > 0:
        kernel = _build_gaussian_kernel(num_classes, sigma=ordinal_sigma, none_index=none_index).to(device)
        targets = targets @ kernel
    elif not ordinal and label_smoothing > 0:
        real_indices = [i for i in range(num_classes) if i != none_index]
        if len(real_indices) > 1:
            smoothed = targets.clone()
            real = torch.tensor(real_indices, device=device, dtype=torch.long)
            for idx in real_indices:
                mask = labels == idx
                if not mask.any():
                    continue
                peer_count = len(real_indices) - 1
                smoothed[mask, :] = 0.0
                smoothed[mask, idx] = 1.0 - label_smoothing
                peer_mask = torch.zeros(num_classes, device=device, dtype=torch.bool)
                peer_mask[real[real != idx]] = True
                row_idx = mask.nonzero(as_tuple=True)[0]
                col_idx = peer_mask.nonzero(as_tuple=True)[0]
                smoothed[row_idx.unsqueeze(1), col_idx] = label_smoothing / peer_count
        targets = smoothed

    # Soft aliases: redistribute weight
    if soft_aliases:
        for src_str, alias_list in soft_aliases.items():
            src = int(src_str)
            for tgt, weight in alias_list:
                mask = labels == src
                if mask.any():
                    transfer = targets[mask, src] * weight
                    targets[mask, src] -= transfer
                    targets[mask, tgt] += transfer

    # Re-normalise to sum to 1
    targets = targets / targets.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return targets


def _soft_ce_loss(
    log_probs: torch.Tensor,
    soft_targets: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy against soft targets, with optional focal + class weights.

    Single source of truth for the soft-target loss, so the full-FT loop, the
    head probe, and the MixUp path all reduce identically. ``focal_gamma`` adds
    ``(1 - p_t)^gamma`` modulation (p_t = expected prob under the soft target);
    ``class_weights`` applies the per-class weight as the expected weight under
    the soft target, normalised so the batch loss scale is invariant.
    """
    ce = -(soft_targets * log_probs).sum(dim=1)  # [N]
    if focal_gamma > 0:
        p_t = (soft_targets * log_probs.exp()).sum(dim=1).clamp(0.0, 1.0)
        ce = (1.0 - p_t).pow(focal_gamma) * ce
    if class_weights is not None:
        w = (soft_targets * class_weights.unsqueeze(0)).sum(dim=1)
        return (ce * w).sum() / w.sum().clamp(min=1e-8)
    return ce.mean()


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

        def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            soft = _build_soft_targets(
                labels, config.num_classes,
                ordinal=config.ordinal,
                ordinal_sigma=config.ordinal_sigma,
                label_smoothing=config.label_smoothing,
                soft_aliases=config.soft_aliases or None,
                none_index=none_index,
                device=device,
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
    max_ratio: float = 0.0,
) -> torch.Tensor:
    """Class weights by the effective number of samples (Cui et al., CVPR 2019).

    ``w_c ∝ (1 - beta) / (1 - beta^{n_c})``, optionally clamped so the max/min weight
    over non-empty classes is at most ``max_ratio`` (bounds over-correction; 0 =
    uncapped), then normalised so the mean weight is 1 (keeps the loss scale
    comparable to the unweighted baseline). Empty classes keep weight 1.
    """
    raw = [1.0] * num_classes
    counts = [0] * num_classes
    for i in range(num_classes):
        n = int(class_counts.get(i, 0))
        counts[i] = n
        if n > 0:
            eff = (1.0 - beta**n) / (1.0 - beta) if beta < 1.0 else float(n)
            raw[i] = 1.0 / max(eff, 1e-8)
    raw = cap_weight_ratio(raw, counts, max_ratio)
    weights = torch.tensor(raw, dtype=torch.float32)
    weights = weights * (num_classes / weights.sum().clamp(min=1e-8))
    return weights.to(device)


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
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    total_steps = len(dataloader)
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

    memory_format = torch.channels_last if config.channels_last else None
    optimizer.zero_grad()
    for images, labels in dataloader:
        if stop_now_event is not None and stop_now_event.is_set():
            break
        images = images.to(device, non_blocking=True)
        images = apply_train_augment(
            images, dtype=dtype,
            randaugment_n=config.randaugment_n,
            randaugment_m=config.randaugment_m,
            random_erasing_p=config.random_erasing_p,
            memory_format=memory_format,
        )
        labels = labels.to(device)

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
        if num_batches % accum == 0 or num_batches == total_steps:
            optimizer.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)

        total_loss += loss.item()

        if step_callback is not None:
            now = time.monotonic()
            if now - _last_report >= _STEP_REPORT_INTERVAL or num_batches == total_steps:
                _last_report = now
                step_callback(num_batches, total_steps, total_loss / num_batches)

    return total_loss / max(num_batches, 1)


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

    metrics["val_loss"] = total_loss / max(num_batches, 1)
    return metrics


def _metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    config: GroupTrainConfig,
    none_index: int,
    cut_points: list[float] | None = None,
) -> dict:
    probs = torch.softmax(logits.float(), dim=1)
    if config.ordinal and cut_points is not None:
        # Shipped ordinal decode: round E[j] at the fitted cut-points. Without
        # cut-points we stay on argmax (the unbiased mode), matching per-epoch
        # selection — raw round-to-nearest E[j] is biased inward at the edges.
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
    metrics["val_loss"] = float(nn.CrossEntropyLoss()(logits.float(), labels.long()).item())
    return metrics


def _real_macro_f1(metrics: dict, config: GroupTrainConfig, none_index: int) -> float:
    per_class = metrics.get("per_class_f1") or {}
    vals = [
        float(per_class.get(str(i), 0.0))
        for i in range(config.num_classes)
        if i != none_index
    ]
    return float(np.mean(vals)) if vals else float(metrics.get("macro_f1") or 0.0)


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


# Cap stored validation rows so the checkpoint stays small (a few hundred KB at
# most). The suite only needs enough resolution to draw a selective-metric curve;
# evenly subsampling large val sets preserves the curve shape.
_STRICTNESS_MAX_VAL_ROWS = 5000


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


def _finalise_ordinal_decode(
    logits: torch.Tensor,
    labels: torch.Tensor,
    config: GroupTrainConfig,
    none_index: int,
) -> tuple[list[float] | None, dict]:
    """Fit E[j] cut-points on (calibrated) val logits, adopting the EV decode
    only when it beats argmax on the selection score.

    Returns ``(cut_points or None, metrics under the chosen decode)``. ``None``
    cut-points mean inference keeps argmax (the safe default) — so the shipped
    ordinal decode can never score below argmax on validation. ``argmax`` is the
    unbiased mode estimate; raw ``round(E[j])`` is biased inward at the scale
    edges, and only the fitted cut-points (OptimizedRounder) reliably correct it.
    """
    argmax_metrics = _metrics_from_logits(logits, labels, config, none_index, cut_points=None)
    argmax_metrics["ordinal_decode"] = "argmax"
    if not config.ordinal:
        return None, argmax_metrics

    probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
    label_list = labels.cpu().tolist()
    cuts = find_ordinal_cut_points(probs, label_list, config.num_classes, none_index=none_index)
    if not cuts:
        return None, argmax_metrics

    ev_metrics = _metrics_from_logits(logits, labels, config, none_index, cut_points=cuts)
    if _metric_score(ev_metrics, config) > _metric_score(argmax_metrics, config) + 1e-9:
        ev_metrics["ordinal_decode"] = "expected_value"
        ev_metrics["ordinal_cut_points"] = cuts
        return cuts, ev_metrics
    return None, argmax_metrics


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
        from bittrainer.smart_cache import SmartCache, face_model_signature
        cache_root = Path(config.cache_dir) if config.cache_dir else (group_folder / ".smart_cache")
        smart_cache = SmartCache(
            cache_root,
            modeltype=config.modeltype,
            progress_callback=cb,
            stop_check=partial(_stop_event_is_set, stop_event),
            face_model_sig=face_model_signature(config.face_model_path or None),
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
            oversample_none=config.oversample_none,
            extra_paths=config.extra_paths_train,
            oversample_max_ratio=config.class_balance_max_ratio,
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
            oversample_none=config.oversample_none,
            extra_paths=config.extra_paths_train,
            oversample_max_ratio=config.class_balance_max_ratio,
        )
        val_ds = GroupDataset(
            group_folder, config.class_names, split="val",
            multi_label=config.multi_label,
            skin_normalise=config.skin_normalise, group_name=group_name,
            extra_paths=config.extra_paths_val,
        )

        # --- Face-aware cropping pre-computation ---
        face_bboxes: dict[str, list[int]] = {}
        if config.face_model_path:
            from bittrainer.face_crop import FaceBBoxCache, precompute_face_bboxes
            face_cache = FaceBBoxCache(group_folder / ".resize_cache" / "face_bboxes.json")
            all_image_paths = [s["path"] for s in train_ds.samples] + [s["path"] for s in val_ds.samples]

            def _face_progress(done: int, total: int) -> None:
                cb({
                    "type": "training_progress", "stage": "face_detection",
                    "status_text": f"Detecting faces ({done}/{total})",
                    "step": done, "total_steps": total,
                })

            precompute_face_bboxes(
                all_image_paths, face_cache, config.face_model_path,
                device=config.device,
                progress_fn=_face_progress,
            )
            for p in all_image_paths:
                bbox = face_cache.get(p)
                if bbox:
                    face_bboxes[p] = bbox
            train_ds.refresh_face_bboxes(face_bboxes)
            val_ds.refresh_face_bboxes(face_bboxes)

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
            model = create_model(
                model_size=size, pretrained=False,
                num_classes=config.num_classes, head_hidden_size=head_hidden_size,
            ).to(device)
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
    return create_model(
        model_size=config.backbone_variant, pretrained=True,
        num_classes=config.num_classes, head_hidden_size=head_hidden_size,
    ).to(device)


def _auto_softness_kind(config: GroupTrainConfig) -> str | None:
    if not config.auto_label_softness or config.multi_label:
        return None
    return "ordinal_sigma" if config.ordinal else "label_smoothing"


def _auto_softness_candidates(kind: str) -> list[float]:
    return _ORDINAL_SIGMA_CANDIDATES if kind == "ordinal_sigma" else _LABEL_SMOOTHING_CANDIDATES


def _capture_rng_state(device: torch.device) -> dict:
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict, device: torch.device) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    if device.type == "cuda" and torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _apply_softness(config: GroupTrainConfig, kind: str, value: float) -> None:
    if kind == "ordinal_sigma":
        config.ordinal_sigma = float(value)
    else:
        config.label_smoothing = float(value)


def _softness_status_label(kind: str) -> str:
    return "ordinal softness" if kind == "ordinal_sigma" else "label smoothing"


def _softness_candidate_better(candidate: dict, incumbent: dict | None) -> bool:
    if incumbent is None:
        return True
    cand_score = float(candidate.get("score") or 0.0)
    inc_score = float(incumbent.get("score") or 0.0)
    if cand_score != inc_score:
        return cand_score > inc_score
    cand_loss = candidate.get("val_loss")
    inc_loss = incumbent.get("val_loss")
    if cand_loss is not None and inc_loss is not None and float(cand_loss) != float(inc_loss):
        return float(cand_loss) < float(inc_loss)
    return float(candidate["value"]) < float(incumbent["value"])


def _run_auto_softness_probe(
    model: nn.Module,
    config: GroupTrainConfig,
    embed_cache: EmbeddingCache,
    smart_cache: object | None,
    train_samples: list[dict],
    val_samples: list[dict],
    *,
    device: torch.device,
    none_index: int,
    cb: Callable[[dict], None],
    stop_event: object | None,
) -> dict:
    kind = _auto_softness_kind(config)
    if kind is None:
        return train_head_probe(
            model, embed_cache, smart_cache,
            train_samples, val_samples, config,
            device=device, none_index=none_index,
            cb=cb, stop_event=stop_event,
        )

    x_train, y_train, x_val, y_val = prepare_head_probe_tensors(
        embed_cache, smart_cache, train_samples, val_samples, config, cb=cb,
    )
    original_head_state = copy.deepcopy(model.head.state_dict())
    original_rng_state = _capture_rng_state(device)
    original_sigma = config.ordinal_sigma
    original_smoothing = config.label_smoothing
    candidates = _auto_softness_candidates(kind)
    label = _softness_status_label(kind)
    score_metric = _score_metric_label(config)

    best_row: dict | None = None
    best_probe: dict | None = None
    best_head_state: dict | None = None
    matrix: list[dict] = []
    sweep_start = time.monotonic()

    for idx, value in enumerate(candidates, start=1):
        if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
            break
        model.head.load_state_dict(original_head_state)
        _restore_rng_state(original_rng_state, device)
        _apply_softness(config, kind, value)
        cb({
            "type": "training_progress",
            "stage": "soft_label_tuning",
            "status_text": f"Testing {label} {value:g} ({idx}/{len(candidates)})",
            "step": idx,
            "total_steps": len(candidates),
            "softness_kind": kind,
            "softness_value": value,
            "soft_label_tuning_metric": score_metric,
        })
        candidate_start = time.monotonic()
        probe = train_head_probe_from_tensors(
            model, x_train, y_train, x_val, y_val, config,
            device=device, none_index=none_index, cb=cb,
            stop_event=stop_event,
            progress_stage="soft_label_tuning",
            progress_prefix=f"{label.capitalize()} {value:g}",
            score_metric=score_metric,
        )
        score = _metric_score(probe, config)
        row = {
            "kind": kind,
            "value": float(value),
            "score": score,
            "macro_f1": probe.get("macro_f1"),
            "qwk": probe.get("qwk"),
            "none_f1": probe.get("none_f1"),
            "none_recall": probe.get("none_recall"),
            "none_precision": probe.get("none_precision"),
            "none_false_positive_rate": probe.get("none_false_positive_rate"),
            "val_loss": probe.get("val_loss"),
            "best_epoch": probe.get("best_epoch"),
            "epochs_completed": probe.get("epochs_completed"),
            "elapsed_ms": int(round((time.monotonic() - candidate_start) * 1000)),
        }
        matrix.append(row)
        cb({
            "type": "training_progress",
            "stage": "soft_label_tuning",
            "status_text": (
                f"Tested {label} {value:g}: {score_metric} {score:.3f}, macro F1 {(row['macro_f1'] or 0.0):.3f}"
                + (f", QWK {row['qwk']:.3f}" if row.get("qwk") is not None else "")
            ),
            "step": idx,
            "total_steps": len(candidates),
            "softness_kind": kind,
            "softness_value": value,
            "val_macro_f1": row["macro_f1"],
            "val_qwk": row.get("qwk"),
            "val_none_f1": row.get("none_f1"),
            "val_none_recall": row.get("none_recall"),
            "selected_validation_score": score,
        })
        if _softness_candidate_better(row, best_row):
            best_row = row
            best_probe = probe
            best_head_state = copy.deepcopy(model.head.state_dict())

    config.ordinal_sigma = original_sigma
    config.label_smoothing = original_smoothing

    if best_row is None or best_probe is None or best_head_state is None:
        model.head.load_state_dict(original_head_state)
        return {"best_epoch": 0, "epochs_completed": 0}

    model.head.load_state_dict(best_head_state)
    _apply_softness(config, kind, float(best_row["value"]))
    config.selected_softness_kind = kind
    config.selected_softness_value = float(best_row["value"])
    config.soft_label_tuning_metric = score_metric
    config.soft_label_tuning_results = matrix
    config.soft_label_tuning_elapsed_ms = int(round((time.monotonic() - sweep_start) * 1000))
    cb({
        "type": "training_progress",
        "stage": "soft_label_tuning",
        "status_text": f"Selected {label} {best_row['value']:g} by {score_metric}",
        "step": len(candidates),
        "total_steps": len(candidates),
        "softness_kind": kind,
        "softness_value": best_row["value"],
        "best_val_macro_f1": best_row.get("macro_f1"),
        "best_val_qwk": best_row.get("qwk"),
        "best_val_none_f1": best_row.get("none_f1"),
        "best_val_none_recall": best_row.get("none_recall"),
        "selected_validation_score": best_row.get("score"),
        "soft_label_tuning_metric": score_metric,
    })
    return best_probe


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

    if best_checkpoint_path:
        _emit("comparing", "Comparing against current model")
        candidate_score = _metric_score(best_metrics, config)
        incumbent_class_names: list[str] | None = None
        incumbent_num_classes: int | None = None
        incumbent_score: float | None = None
        old_metrics: dict | None = None
        eval_ok = False

        if not existing_best.exists():
            promote, promotion_reason = decide_promotion(
                incumbent_exists=False,
                incumbent_class_names=None,
                candidate_class_names=list(config.class_names),
                incumbent_score=None,
                candidate_score=candidate_score,
                eval_ok=False,
            )
        else:
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
                    old_metrics = _evaluate(
                        old_model, val_loader, config.num_classes, device, dtype,
                        multi_label=config.multi_label,
                        ordinal=config.ordinal,
                        none_index=_resolve_none_index(config.class_names),
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
            if (
                promote
                and _guarded_metric_enabled(config)
                and old_metrics is not None
                and best_metrics.get("none_recall") is not None
                and old_metrics.get("none_recall") is not None
                and float(best_metrics.get("none_recall") or 0.0) + _NONE_RECALL_REGRESSION_TOLERANCE
                < float(old_metrics.get("none_recall") or 0.0)
            ):
                logger.info(
                    "Keeping incumbent because candidate regressed __none__ recall "
                    "(incumbent=%.4f candidate=%.4f)",
                    float(old_metrics.get("none_recall") or 0.0),
                    float(best_metrics.get("none_recall") or 0.0),
                )
                promote = False
                promotion_reason = PromotionReason.incumbent_wins

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
    if (
        best_checkpoint_path
        and not config.multi_label
        and (_has_none_class(config) or config.ordinal)
    ):
        try:
            _emit("calibrating", "Calibrating decision boundaries")
            calib_model = load_checkpoint(
                best_checkpoint_path, device=str(device), dtype=dtype,
                model_size=config.backbone_variant, num_classes=config.num_classes,
            ).to(device)
            logits, labels = _collect_val_logits(calib_model, val_loader, config, device, dtype)
            val_logits_cache, val_labels_cache = logits, labels
            del calib_model

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

    result = {
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch_display,
        "best_val_macro_f1": best_val_macro_f1,
        "validation_metric": _primary_validation_metric(config),
        "selected_validation_score": _metric_score(best_metrics, config),
        "final_val_macro_f1": best_metrics.get("macro_f1"),
        "final_val_macro_precision": best_metrics.get("macro_precision"),
        "final_val_macro_recall": best_metrics.get("macro_recall"),
        "final_val_loss": best_metrics.get("val_loss"),
        "per_class_f1": best_metrics.get("per_class_f1", {}),
        "per_class_precision": best_metrics.get("per_class_precision", {}),
        "per_class_recall": best_metrics.get("per_class_recall", {}),
        "checkpoint_path": best_checkpoint_path,
        "class_counts": class_counts,
        "total_images": total_raw,
        "promotion_reason": promotion_reason.value if promotion_reason else None,
        "selected_softness_kind": config.selected_softness_kind,
        "selected_softness_value": config.selected_softness_value,
        "soft_label_tuning_metric": config.soft_label_tuning_metric,
        "soft_label_tuning_results": config.soft_label_tuning_results,
        "soft_label_tuning_elapsed_ms": config.soft_label_tuning_elapsed_ms,
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


def run_group_training(
    config: GroupTrainConfig,
    *,
    progress_callback: Callable[[dict], None] | None = None,
    stop_event: object | None = None,
    stop_now_event: object | None = None,
) -> dict:
    """Run the full multi-class training loop.

    stop_event signals a graceful stop at the next epoch boundary.
    stop_now_event additionally interrupts the current epoch's training loop
    mid-batch; validation and the fair-comparison block still run.
    """
    from bittrainer.progress import ProgressEmitter, Stage
    from bittrainer.runtime import configure_cuda_backend, maybe_compile, prewarm_compile
    from bittrainer.smart_cache import _noop_callback
    em = ProgressEmitter(progress_callback or config.progress_callback or _noop_callback)
    cb = em.raw
    device = torch.device(config.device)
    dtype = _get_dtype(config.dtype)
    configure_cuda_backend()
    group_folder = Path(config.group_folder)
    checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else group_folder / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    use_soft = config.ordinal or bool(config.soft_aliases) or (
        not config.multi_label and config.label_smoothing > 0
    )

    em.stage(Stage.scanning, "Scanning dataset")
    train_ds, val_ds, smart_cache, bucket_counts = _prepare_datasets_and_cache(
        config, cb=cb, stop_event=stop_event,
    )

    # Create model â€” warm-start from best.pt unless from_scratch is set.
    head_hidden_size = config.probe_mlp_hidden if config.probe_head == "mlp" else None
    _emit_model_load_stage(em, config, checkpoint_dir)
    model = _create_or_warmstart_model(
        config, device=device, dtype=dtype,
        head_hidden_size=head_hidden_size, checkpoint_dir=checkpoint_dir,
    )
    memory_format = torch.channels_last if config.channels_last else None
    if memory_format is not None:
        model = model.to(memory_format=memory_format)

    # Head warmup on cached features (replaces the fixed 1-epoch frozen warmup),
    # then fine-tune fully unfrozen. A converged head removes the
    # feature-distortion risk a random head poses, so there is no
    # gradual-unfreeze ramp.
    _warmup_head_probe(
        model, config, train_ds, val_ds, smart_cache,
        device=device, dtype=dtype, cb=cb,
        stop_event=stop_event, stop_now_event=stop_now_event,
    )
    unfreeze_backbone(model)  # the probe froze the backbone â€” restore full grad

    # --- Auto batch sizing (probe unfrozen = worst-case VRAM) ---
    # Targets config.vram_fraction of free VRAM. Prodigy_adv state (~2.2x param
    # bytes, allocated lazily on first .step()) is budgeted explicitly inside
    # determine_batch_size via param_overhead_bytes, so the fraction only needs
    # to absorb allocator fragmentation and activation variance across buckets.
    if config.batch_size is not None and config.batch_size > 0:
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
            max_ratio=config.class_balance_max_ratio,
        )
        cb({
            "type": "training_progress", "stage": "preparing",
            "status_text": "Class balance: reweight (natural sampling + effective-number weights)",
        })

    # --- MixUp/CutMix gate: skip on tiny datasets where the full aug stack
    # over-regularises, and for multi-label (single-label soft-target path only). ---
    mixup_enabled = (
        config.use_mixup and not config.multi_label and total_raw >= config.mixup_min_images
    )

    # --- SWA: average weights over the cosine tail (epoch >= swa_start_epoch). ---
    swa = _SWA() if (config.use_swa and not config.multi_label) else None
    swa_start_epoch = int(config.swa_start_frac * config.max_epochs)

    # Optimizer (LLRD param groups when config.llrd, else flat). Built once over
    # the fully-unfrozen model â€” the warm head means no epoch-1 rebuild.
    optimizer = _make_optimizer(model, config)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    # EMA tracks all params from the start; freeze/unfreeze only affects which
    # ones receive gradient updates, but the EMA still mirrors the live tensor
    # values, which is what we want for inference-time smoothing.
    ema = ModelEMA(model, decay=config.ema_decay) if config.use_ema else None

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

    for epoch in range(config.max_epochs):
        if stop_now_event is not None and stop_now_event.is_set():
            logger.info("Stop-now requested before epoch %d â€” running final comparison", epoch)
            cb({"type": "stop_now", "epoch": epoch, "max_epochs": config.max_epochs})
            break
        if stop_event is not None and stop_event.is_set():
            logger.info("Graceful stop requested after epoch %d â€” running final comparison", epoch)
            cb({"type": "graceful_stop", "epoch": epoch, "max_epochs": config.max_epochs})
            break

        # Reshuffle for class-balanced sampling
        train_ds.reshuffle()

        if epoch == 0:
            cb({
                "type": "training_progress", "stage": "preparing",
                "status_text": f"Batch size {eff_bs} â€” spawning data workers",
            })

        # Build dataloaders
        collate_fn = _collate_multilabel_batch if config.multi_label else _collate_bucket_batch
        train_sampler = build_group_bucket_sampler(train_ds, batch_size=eff_bs)
        train_loader = DataLoader(
            train_ds, batch_sampler=train_sampler, collate_fn=collate_fn,
            num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4,
        )
        val_sampler = build_group_bucket_sampler(val_ds, batch_size=eff_bs)
        val_loader = DataLoader(
            val_ds, batch_sampler=val_sampler, collate_fn=collate_fn,
            num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4,
        )

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

        train_loss = _train_one_epoch(
            fwd_model, train_loader, optimizer, config, device, dtype,
            use_soft_targets=use_soft,
            step_callback=_on_step,
            stop_now_event=stop_now_event,
            ema=ema,
            class_weights=class_weights,
            mixup_enabled=mixup_enabled,
        )
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
        val_metrics = _evaluate(
            eval_model, val_loader, config.num_classes, device, dtype,
            multi_label=config.multi_label,
            ordinal=config.ordinal,
            none_index=_resolve_none_index(config.class_names),
            channels_last=config.channels_last,
        )
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

        epoch_msg = {
            "type": "epoch_complete",
            "stage": "training",
            "status_text": f"Epoch {epoch + 1}/{config.max_epochs} complete (val macro F1 {val_macro_f1:.3f})",
            "epoch": epoch + 1,
            "max_epochs": config.max_epochs,
            "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "val_macro_f1": val_macro_f1,
            "val_macro_precision": val_metrics.get("macro_precision", 0.0),
            "val_macro_recall": val_metrics.get("macro_recall", 0.0),
            "per_class_f1": val_metrics.get("per_class_f1", {}),
            "per_class_precision": val_metrics.get("per_class_precision", {}),
            "per_class_recall": val_metrics.get("per_class_recall", {}),
            "val_none_precision": val_metrics.get("none_precision"),
            "val_none_recall": val_metrics.get("none_recall"),
            "val_none_f1": val_metrics.get("none_f1"),
            "val_none_false_positive_rate": val_metrics.get("none_false_positive_rate"),
            "best_val_macro_f1": best_val_macro_f1,
            "selected_validation_score": selected_score,
            "best_validation_score": best_validation_score,
            "validation_metric": _primary_validation_metric(config),
            "best_epoch": best_epoch + 1,
        }
        if config.ordinal:
            epoch_msg["val_qwk"] = val_qwk
            epoch_msg["val_ordinal_mae"] = val_metrics.get("ordinal_mae")
            epoch_msg["val_adjacent_accuracy"] = val_metrics.get("adjacent_accuracy")
            epoch_msg["best_val_qwk"] = best_val_qwk
        cb(epoch_msg)

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
            swa_metrics = _evaluate(
                model, val_loader, config.num_classes, device, dtype,
                multi_label=config.multi_label, ordinal=config.ordinal,
                none_index=_resolve_none_index(config.class_names),
                channels_last=config.channels_last,
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

    return _compare_promote_finalize(
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
        total_raw=total_raw,
        cb=cb,
    )
