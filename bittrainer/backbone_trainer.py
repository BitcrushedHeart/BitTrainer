"""Backbone Builder training entry point for Bitcrush Engine.

Engine's BackboneTrainingManager calls ``run_backbone_training(request,
progress_callback=...)`` with a request assembled from the suite DB:

    {
        "run_id": str,
        "family_name" / "architecture" / "size_alias" / "display_size"
            / "convnextv2_size": str,
        "candidate_checkpoint_path": str,   # where to write the safetensors
        "records": [ {content_hash, file_paths, binary{concept: label},
                      groups{group: class}, splits{split: n}} ],
        "dataset_snapshot_id" / "content_hash_index_id": str,
        "heads": {...},                     # head status block (echoed back)
        "training_config": {image_size, batch_size, epochs, max_steps,
                            learning_rate, validation_split, device?,
                            # Sampling layer (Bitcrush ISSUE-0545/0546):
                            neg_pos_ratio,           # per-head explicit-negative cap,
                                                     # default 5.0; <=0 = uncapped;
                                                     # auto-tightens toward 1:1 for
                                                     # tiny heads (see _effective_neg_ratio)
                            label_policy,            # {"mode": "masked_unknown" |
                                                     #  "soft_implicit_negative",
                                                     #  "implicit_negative_value": 0.1}
                                                     # soft mode: unlabelled (image, head)
                                                     # pairs fill the cap's headroom AFTER
                                                     # explicit negatives, as soft targets.
                                                     # Training only — validation always
                                                     # stays masked-unknown.
                            use_pos_weight,          # BCE pos_weight from the residual
                                                     # post-cap ratio, clamped to [1, 10]
                            max_positives_per_class, # per-head positive cap, default 1000;
                                                     # 0 = uncapped; label-dense images
                                                     # survive the cut first
                            oversample_positives,    # replicate tiny heads' positives
                                                     # (bounded 4x); default true
                            min_positive_threshold,  # oversample + ratio-tightening
                                                     # trigger, default 30
                            sampling_seed,
                            # Resolution tail (train cheap, finish sharp):
                            finetune_image_size,     # optional high-res tail size;
                                                     # 0/absent = single-resolution run
                            finetune_epochs,         # last N epochs run (train + val)
                                                     # at finetune_image_size; the best
                                                     # tracker resets at the switch so
                                                     # the exported candidate is always
                                                     # selected at the tail resolution.
                                                     # Note: early stopping can still
                                                     # end a run before the tail.
                            ...},
        "backbone_init": {...},             # see bittrainer.backbone_init
        "license_provenance" / "external_pretrained_used"
            / "temporary_timm_fallback_used" / "release_blocking": provenance,
    }

Training: one shared ConvNeXt V2 backbone with a supervised head per binary
concept (BCE) and per group (CE), losses masked to whichever labels each
record actually carries. The candidate checkpoint holds ONLY the backbone
state dict (safetensors) plus Engine-readable string metadata, so any trainer
can later ``apply_backbone_init`` it regardless of head layout.

The entry point is async: the torch loop runs in a worker thread and progress
dicts are marshalled back to the caller's event loop, so Engine's FastAPI
process stays responsive while a backbone trains.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import random
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

from bittrainer.training_state import make_fingerprint

logger = logging.getLogger(__name__)

# AMP autocast dtype aliases (Bitcrush ISSUE-0476). float16 is accepted but,
# like the binary trainer, no GradScaler is wired — bfloat16 is the tested
# default and needs none; float16 on CUDA without a scaler can underflow.
_AMP_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _amp_settings(config: dict) -> tuple[bool, torch.dtype]:
    """Resolve (enabled, autocast dtype) from the training config.

    AMP defaults ON with bfloat16 (matches the binary trainer). Disabled when
    ``use_amp`` is false or the resolved dtype is float32.
    """
    name = str(config.get("amp_dtype") or "bfloat16").lower()
    dtype = _AMP_DTYPES.get(name, torch.bfloat16)
    enabled = bool(config.get("use_amp", True)) and dtype != torch.float32
    return enabled, dtype


def _backbone_fingerprint(
    vocab: "_Vocab",
    model_size: str,
    epochs: int,
    *,
    resolution: str | None = None,
    optimizer_identity: str = "Prodigy_adv",
) -> dict:
    """Resume-compatibility identity for a backbone run.

    Folds the label space (concepts + multi-class groups) into the fingerprint
    so a resume whose dataset changed head layout starts fresh rather than
    crashing on a shape mismatch.
    """
    names = [f"binary/{c}" for c in vocab.concepts] + [
        f"group/{g}" for g in vocab.groups
    ]
    fingerprint = make_fingerprint(
        class_names=names,
        num_classes=len(names),
        max_epochs=int(epochs),
        multi_label=True,
        ordinal=False,
        best_model_name="backbone_candidate",
        model_size=str(model_size),
    )
    # Fold the optimizer identity into the fingerprint so an in-flight backup from
    # an older AdamW / flat-Prodigy run (which lacks this key, or carries a
    # different layout string) cleanly mismatches and starts fresh (resume_skipped)
    # rather than loading incompatible optimizer state.
    fingerprint["optimizer"] = optimizer_identity
    # Resolution schedule identity ("384" or "256->384@2"): a backup from a run
    # with a different image size / finetune tail cleanly mismatches and starts
    # fresh (same accepted precedent as the optimizer key above).
    if resolution is not None:
        fingerprint["resolution"] = resolution
    return fingerprint


_POSITIVE_LABELS = frozenset({"positive", "explicit_positive", "1", "true"})
_NEGATIVE_LABELS = frozenset(
    {
        "negative",
        "known_negative",
        "explicit_known_negative",
        "implicit_negative",
        "0",
        "false",
    }
)


class BackboneTrainingCancelled(RuntimeError):
    """Raised inside the worker thread when the caller cancelled the task."""


# --------------------------------------------------------------------------- #
# Sample assembly                                                             #
# --------------------------------------------------------------------------- #


def _safe_module_key(name: str) -> str:
    """nn.Module/ModuleDict keys reject '.' (e.g. a person concept like
    "Carmen B. Sanchez") - normalise before using a label as one."""
    return name.replace(".", "_") if name else name


class _Vocab:
    """Label spaces derived from the record payload."""

    def __init__(self, records: list[dict]):
        concepts: set[str] = set()
        group_classes: dict[str, set[str]] = {}
        for record in records:
            for concept, label in (record.get("binary") or {}).items():
                if str(label).lower() in _POSITIVE_LABELS | _NEGATIVE_LABELS:
                    concepts.add(_safe_module_key(concept))
            for group, class_name in (record.get("groups") or {}).items():
                if class_name:
                    group_classes.setdefault(_safe_module_key(group), set()).add(
                        str(class_name)
                    )
        self.concepts = sorted(concepts)
        # Single-class groups carry no training signal for CE.
        self.groups = {
            name: sorted(classes)
            for name, classes in sorted(group_classes.items())
            if len(classes) >= 2
        }

    @property
    def has_targets(self) -> bool:
        return bool(self.concepts or self.groups)


class _Sample:
    __slots__ = ("path", "binary", "groups")

    def __init__(self, path: str, binary: dict[str, float], groups: dict[str, int]):
        self.path = path
        self.binary = binary
        self.groups = groups


def _build_samples(records: list[dict], vocab: _Vocab) -> tuple[list[_Sample], int]:
    samples: list[_Sample] = []
    missing = 0
    duplicates = 0
    # De-dup by content_hash (falling back to the resolved path) so the same
    # image entering the audit twice never trains twice or skews the counts.
    seen: set[str] = set()
    for record in records:
        path = next(
            (p for p in record.get("file_paths") or [] if Path(p).is_file()),
            None,
        )
        if path is None:
            missing += 1
            continue
        binary: dict[str, float] = {}
        for concept, label in (record.get("binary") or {}).items():
            lowered = str(label).lower()
            key = _safe_module_key(concept)
            if lowered in _POSITIVE_LABELS:
                binary[key] = 1.0
            elif lowered in _NEGATIVE_LABELS:
                binary[key] = 0.0
        groups: dict[str, int] = {}
        for group, class_name in (record.get("groups") or {}).items():
            key = _safe_module_key(group)
            classes = vocab.groups.get(key)
            if classes and str(class_name) in classes:
                groups[key] = classes.index(str(class_name))
        if not (binary or groups):
            continue
        dedup_key = str(record.get("content_hash") or path)
        if dedup_key in seen:
            duplicates += 1
            continue
        seen.add(dedup_key)
        samples.append(_Sample(path, binary, groups))
    if duplicates:
        logger.warning(
            "Backbone training: skipped %d duplicate record(s) sharing a content_hash",
            duplicates,
        )
    return samples, missing


def _split_samples(
    records: list[dict], samples: list[_Sample], validation_split: float
) -> tuple[list[_Sample], list[_Sample]]:
    """Deterministic content-hash split so re-runs see the same partition."""
    hash_by_path: dict[str, str] = {}
    for record in records:
        for path in record.get("file_paths") or []:
            hash_by_path[path] = record.get("content_hash") or path
    train: list[_Sample] = []
    val: list[_Sample] = []
    for sample in samples:
        digest = hash_by_path.get(sample.path, sample.path)
        try:
            bucket = int(str(digest)[:8], 16) % 10_000 / 10_000
        except ValueError:
            bucket = (hash(str(digest)) % 10_000) / 10_000
        (val if bucket < validation_split else train).append(sample)
    if not train:  # tiny datasets: never let validation starve training
        train, val = val, []
    return train, val


def _effective_neg_ratio(num_pos: int, base_ratio: float) -> float:
    """Per-head neg:pos cap, auto-tightened for tiny heads (ISSUE-0545/0546).

    ``<=0`` means uncapped. A head with ~10 positives cannot afford 5:1 —
    most batches would carry zero positives and the head collapses to
    always-negative — so the ratio scales with the positive count:
    10 pos -> 1:1, 20 -> 2:1, 30 -> 3:1, plateauing at ``base_ratio``.
    """
    if base_ratio <= 0:
        return 0.0
    return min(float(base_ratio), max(1.0, num_pos / 10.0))


def _plan_epoch_samples(
    samples: list["_Sample"],
    vocab: "_Vocab",
    epoch: int,
    *,
    seed: int = 0,
    neg_pos_ratio: float = 5.0,
    label_policy: dict | None = None,
    positive_cap: int = 1000,
    min_positive_threshold: int = 30,
    max_oversample_factor: float = 4.0,
    positive_cap_mode: str = "density",
) -> tuple[list["_Sample"], dict[str, dict]]:
    """Build one epoch's training view of ``samples`` (ISSUE-0545/0546).

    The cap operates at the (head, image) LABEL level: an image whose negative
    label for head A misses this epoch's draw still trains its other heads and
    groups. Per head: capped positives (label-dense images survive first) +
    all-or-sampled explicit negatives up to the effective ratio + — under the
    ``soft_implicit_negative`` policy — unlabelled images filling whatever
    headroom the explicit negatives leave, as soft targets. Tiny heads'
    positives are replicated (bounded), each replica carrying ONLY that head's
    positive label so oversampling never skews another head or group.

    Pure function of ``(samples, epoch, seed, config)``; an epoch-restart
    resume rebuilds the identical plan. Group labels are never capped.
    """
    policy = label_policy or {}
    soft = str(policy.get("mode") or "masked_unknown") == "soft_implicit_negative"
    implicit_value = float(policy.get("implicit_negative_value", 0.1))

    eff_binary: list[dict[str, float]] = [{} for _ in samples]
    replicas: list[_Sample] = []
    stats: dict[str, dict] = {}

    for concept in vocab.concepts:
        rng = random.Random(f"{seed}|{epoch}|{concept}")
        pos = [i for i, s in enumerate(samples) if s.binary.get(concept) == 1.0]
        neg = [i for i, s in enumerate(samples) if s.binary.get(concept) == 0.0]
        pos_total = len(pos)
        if 0 < positive_cap < len(pos):
            if positive_cap_mode == "random":
                # Plain seeded uniform draw: density-first selection systematically
                # over-represents multi-label images. Deterministic per
                # (seed, epoch, concept) via the same per-concept rng.
                pos = rng.sample(pos, positive_cap)
            else:
                order = list(pos)
                rng.shuffle(order)  # epoch-varying tiebreak among equal densities
                order.sort(
                    key=lambda i: -(len(samples[i].binary) + len(samples[i].groups))
                )
                pos = order[:positive_cap]

        ratio = _effective_neg_ratio(len(pos), neg_pos_ratio)
        unlabelled = (
            [i for i, s in enumerate(samples) if concept not in s.binary]
            if soft
            else []
        )
        cap = (
            len(neg) + len(unlabelled)
            if ratio <= 0
            else math.ceil(ratio * max(len(pos), 1))
        )
        neg_selected = neg if len(neg) <= cap else rng.sample(neg, cap)
        implicit_selected: list[int] = []
        if soft:
            headroom = cap - len(neg_selected)
            if headroom > 0 and unlabelled:
                implicit_selected = rng.sample(
                    unlabelled, min(headroom, len(unlabelled))
                )

        for i in pos:
            eff_binary[i][concept] = 1.0
        for i in neg_selected:
            eff_binary[i][concept] = 0.0
        for i in implicit_selected:
            eff_binary[i][concept] = implicit_value

        factor = 1
        if 0 < len(pos) < min_positive_threshold:
            factor = max(
                1,
                min(
                    int(max_oversample_factor),
                    math.ceil(min_positive_threshold / len(pos)),
                ),
            )
        for _ in range(factor - 1):
            replicas.extend(_Sample(samples[i].path, {concept: 1.0}, {}) for i in pos)

        stats[concept] = {
            "pos": len(pos),
            "pos_total": pos_total,
            "neg_explicit": len(neg_selected),
            "neg_implicit": len(implicit_selected),
            "effective_ratio": ratio,
            "oversample_factor": factor,
        }

    planned = [
        _Sample(sample.path, eff_binary[i], dict(sample.groups))
        for i, sample in enumerate(samples)
        if eff_binary[i] or sample.groups
    ]
    planned.extend(replicas)
    return planned, stats


def _head_pos_weights(
    stats: dict[str, dict], *, clamp: float = 10.0
) -> dict[str, float]:
    """Residual BCE ``pos_weight`` per head from the post-cap epoch plan.

    ``neg_selected / positive_occurrences`` (occurrences count oversample
    replicas), clamped to ``[1, clamp]`` — a complement to the sampling cap,
    never a substitute. Heads with no positives or no negatives get 1.0.
    """
    weights: dict[str, float] = {}
    for concept, head_stats in stats.items():
        occurrences = head_stats.get("pos", 0) * max(
            head_stats.get("oversample_factor", 1), 1
        )
        negatives = head_stats.get("neg_explicit", 0) + head_stats.get(
            "neg_implicit", 0
        )
        if occurrences <= 0 or negatives <= 0:
            weights[concept] = 1.0
        else:
            weights[concept] = min(clamp, max(1.0, negatives / occurrences))
    return weights


class _BackboneDataset(Dataset):
    def __init__(self, samples: list[_Sample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, sample.binary, sample.groups


_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _train_transform(image_size: int, aug: dict | None = None) -> transforms.Compose:
    """Backbone train transform, defaulting to the group trainer's recipe.

    ``aug=None`` == every knob at its parity default (RandomResizedCrop replacing
    the squash Resize, RandAugment(2, 9), RandomErasing(0.25)), so the backbone
    trains under the same augmentation the group trainer already runs. Each knob
    gates independently:

    * ``use_random_resized_crop`` (True) -> ``RandomResizedCrop(scale=(min, 1))``;
      False -> the legacy squash ``Resize((s, s))``.
    * ``randaugment_n`` / ``randaugment_m`` (2 / 9) -> ``RandAugment``; either 0
      disables it (applied on the PIL image before ToTensor).
    * ``random_erasing_p`` (0.25) -> ``RandomErasing`` after Normalize; 0 disables.
    """
    aug = aug or {}
    use_rrc = bool(aug.get("use_random_resized_crop", True))
    rrc_scale_min = float(aug.get("rrc_scale_min", 0.6))
    ra_n = int(aug.get("randaugment_n", 2))
    ra_m = int(aug.get("randaugment_m", 9))
    erasing_p = float(aug.get("random_erasing_p", 0.25))

    ops: list = []
    if use_rrc:
        ops.append(transforms.RandomResizedCrop(image_size, scale=(rrc_scale_min, 1.0)))
    else:
        ops.append(transforms.Resize((image_size, image_size)))
    ops.append(transforms.RandomHorizontalFlip(p=0.5))
    if ra_n > 0 and ra_m > 0:
        ops.append(transforms.RandAugment(num_ops=ra_n, magnitude=ra_m))
    ops.append(
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)
    )
    ops.append(transforms.ToTensor())
    ops.append(_NORMALIZE)
    if erasing_p > 0:
        ops.append(transforms.RandomErasing(p=erasing_p))
    return transforms.Compose(ops)


def _val_transform(image_size: int) -> transforms.Compose:
    """Deterministic eval transform: aspect-preserving Resize + CenterCrop.

    The ImageNet eval convention (resize the shorter side to ``image_size``, then
    centre-crop) matches what an RRC-trained model sees and removes the squash
    distortion the old ``Resize((s, s))`` introduced. Deterministic by
    construction, so the val score is reproducible.
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            _NORMALIZE,
        ]
    )


def _collate(batch):
    images = torch.stack([item[0] for item in batch])
    return images, [item[1] for item in batch], [item[2] for item in batch]


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #


class _MultiTaskHeads(nn.Module):
    def __init__(self, feature_dim: int, vocab: _Vocab):
        super().__init__()
        self.binary = nn.ModuleDict(
            {concept: nn.Linear(feature_dim, 1) for concept in vocab.concepts}
        )
        self.groups = nn.ModuleDict(
            {
                group: nn.Linear(feature_dim, len(classes))
                for group, classes in vocab.groups.items()
            }
        )


class _BackboneWithHeads(nn.Module):
    """Backbone + multi-task heads as one module.

    Bundling them lets a single :class:`~bittrainer.ema.ModelEMA` track every
    trainable parameter, and lets the backup envelope carry one ``state_dict``
    (Bitcrush ISSUE-0476). The backbone alone is what the candidate checkpoint
    finally serialises.
    """

    def __init__(self, backbone: nn.Module, heads: _MultiTaskHeads):
        super().__init__()
        self.backbone = backbone
        self.heads = heads


def _batch_loss(
    features: torch.Tensor,
    heads: _MultiTaskHeads,
    binary_labels: list[dict[str, float]],
    group_labels: list[dict[str, int]],
    device: torch.device,
    *,
    pos_weight: dict[str, float] | None = None,
    group_label_smoothing: float = 0.0,
    reduction: str = "sum",
) -> torch.Tensor | None:
    """Masked multi-task loss over whichever heads each row carries a label for.

    ``group_label_smoothing`` smooths the group-CE targets (parity with the group
    trainer's default). ``reduction`` combines the per-head losses: ``"sum"`` (the
    legacy default, so direct callers are unchanged) or ``"mean"`` — averaging over
    the number of contributing head losses so the gradient scale no longer grows
    with per-batch label density.
    """
    losses: list[torch.Tensor] = []
    bce = nn.functional.binary_cross_entropy_with_logits
    ce = nn.functional.cross_entropy
    for concept, head in heads.binary.items():
        rows = [i for i, labels in enumerate(binary_labels) if concept in labels]
        if not rows:
            continue
        logits = head(features[rows]).squeeze(-1)
        targets = torch.tensor(
            [binary_labels[i][concept] for i in rows], device=device, dtype=logits.dtype
        )
        weight = (pos_weight or {}).get(concept)
        if weight is not None and weight != 1.0:
            losses.append(
                bce(
                    logits,
                    targets,
                    pos_weight=torch.tensor(weight, device=device, dtype=logits.dtype),
                )
            )
        else:
            losses.append(bce(logits, targets))
    for group, head in heads.groups.items():
        rows = [i for i, labels in enumerate(group_labels) if group in labels]
        if not rows:
            continue
        logits = head(features[rows])
        targets = torch.tensor([group_labels[i][group] for i in rows], device=device)
        losses.append(ce(logits, targets, label_smoothing=group_label_smoothing))
    if not losses:
        return None
    stacked = torch.stack(losses)
    return stacked.mean() if reduction == "mean" else stacked.sum()


@torch.no_grad()
def _collect_val_scores(
    backbone: nn.Module,
    heads: _MultiTaskHeads,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]], dict[str, tuple[np.ndarray, np.ndarray]]
]:
    """One pass over the (masked-unknown) val loader collecting raw head logits.

    Returns ``(binary_scores, group_scores)`` where
    ``binary_scores[concept] = (logits[N] float64, targets[N] float64)`` and
    ``group_scores[group] = (logits[N, C] float64, targets[N] int64)``. Heads run
    in fp32 exactly like the legacy ``_evaluate`` (avoids the autocast dtype
    mismatch); the raw logits feed both the metric panel and calibration.
    """
    backbone.eval()
    heads.eval()
    bin_logits: dict[str, list[np.ndarray]] = {c: [] for c in heads.binary}
    bin_targets: dict[str, list[np.ndarray]] = {c: [] for c in heads.binary}
    grp_logits: dict[str, list[np.ndarray]] = {g: [] for g in heads.groups}
    grp_targets: dict[str, list[np.ndarray]] = {g: [] for g in heads.groups}
    for images, binary_labels, group_labels in loader:
        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            features = backbone(images.to(device))
        features = features.float()  # heads run in fp32; avoids autocast dtype mismatch
        for concept, head in heads.binary.items():
            rows = [i for i, labels in enumerate(binary_labels) if concept in labels]
            if not rows:
                continue
            logits = head(features[rows]).squeeze(-1)
            bin_logits[concept].append(
                logits.detach().cpu().numpy().astype(np.float64).reshape(-1)
            )
            bin_targets[concept].append(
                np.asarray([binary_labels[i][concept] for i in rows], dtype=np.float64)
            )
        for group, head in heads.groups.items():
            rows = [i for i, labels in enumerate(group_labels) if group in labels]
            if not rows:
                continue
            logits = head(features[rows])
            grp_logits[group].append(logits.detach().cpu().numpy().astype(np.float64))
            grp_targets[group].append(
                np.asarray([group_labels[i][group] for i in rows], dtype=np.int64)
            )
    backbone.train()
    heads.train()
    binary_scores = {
        c: (np.concatenate(bin_logits[c]), np.concatenate(bin_targets[c]))
        for c in heads.binary
        if bin_logits[c]
    }
    group_scores = {
        g: (np.concatenate(grp_logits[g]), np.concatenate(grp_targets[g]))
        for g in heads.groups
        if grp_logits[g]
    }
    return binary_scores, group_scores


def _backbone_metrics(
    binary_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    group_scores: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    """Per-head F1/AP + per-group macro-F1 + a defensible ``selection`` aggregate.

    Accuracy at ``logit > 0`` saturates on imbalanced heads and the blind mean of
    accuracies is dominated by easy heads, so selection moves to F1/AP. Legacy
    accuracy keys (``binary/<c>`` at ``logit > 0``, ``group/<g>`` argmax) are kept
    byte-for-byte (additive wire contract). Zero-positive binary heads get no
    F1/AP (never a frozen zero — only a ``binary_support`` key) and are excluded
    from ``binary_macro_f1``; a panel's aggregate is omitted (not zeroed) when it
    has no qualifying member. ``selection`` is the equal-weight mean of whichever
    of the two panel aggregates exist, else 0.0.
    """
    from sklearn.metrics import average_precision_score, f1_score

    from bittrainer.group_validation import compute_multiclass_metrics

    metrics: dict[str, float] = {}
    binary_f1s: list[float] = []
    for concept, (logits, targets) in binary_scores.items():
        logits = np.asarray(logits, dtype=np.float64)
        tgt_int = np.asarray(targets, dtype=np.float64).astype(np.int64)
        preds = (logits > 0).astype(np.int64)
        n = int(tgt_int.shape[0])
        metrics[f"binary/{concept}"] = float((preds == tgt_int).mean()) if n else 0.0
        support = int(tgt_int.sum())
        metrics[f"binary_support/{concept}"] = support
        if support >= 1:
            f1 = float(f1_score(tgt_int, preds, zero_division=0))
            probs = 1.0 / (1.0 + np.exp(-logits))
            metrics[f"binary_f1/{concept}"] = f1
            metrics[f"binary_ap/{concept}"] = float(
                average_precision_score(tgt_int, probs)
            )
            binary_f1s.append(f1)

    group_f1s: list[float] = []
    for group, (logits, targets) in group_scores.items():
        logits = np.asarray(logits, dtype=np.float64)
        tgt = np.asarray(targets, dtype=np.int64)
        n = int(tgt.shape[0])
        preds = logits.argmax(axis=1)
        metrics[f"group/{group}"] = float((preds == tgt).mean()) if n else 0.0
        metrics[f"group_val_support/{group}"] = n
        if n:
            gm = compute_multiclass_metrics(
                tgt.tolist(), preds.tolist(), int(logits.shape[1])
            )
            f1 = float(gm["macro_f1"])
            metrics[f"group_f1/{group}"] = f1
            group_f1s.append(f1)

    aggregates: list[float] = []
    if binary_f1s:
        metrics["binary_macro_f1"] = float(np.mean(binary_f1s))
        aggregates.append(metrics["binary_macro_f1"])
    if group_f1s:
        metrics["group_macro_f1"] = float(np.mean(group_f1s))
        aggregates.append(metrics["group_macro_f1"])
    metrics["selection"] = float(np.mean(aggregates)) if aggregates else 0.0
    return metrics


@torch.no_grad()
def _evaluate(
    backbone: nn.Module,
    heads: _MultiTaskHeads,
    loader: DataLoader,
    vocab: _Vocab,
    device: torch.device,
    *,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    score_sink: dict | None = None,
) -> dict[str, float]:
    """Collect raw val logits then compute the metric panel (see
    :func:`_backbone_metrics`). Kept as the live monkeypatch seam.

    When ``score_sink`` is provided the raw ``{"binary": ..., "groups": ...}``
    scores of THIS pass are stashed there so calibration can fit on the winning
    epoch's logits; fakes that ignore the kwarg simply leave the sink empty.
    """
    binary_scores, group_scores = _collect_val_scores(
        backbone, heads, loader, device, amp_enabled=amp_enabled, amp_dtype=amp_dtype
    )
    if score_sink is not None:
        score_sink["binary"] = binary_scores
        score_sink["groups"] = group_scores
    return _backbone_metrics(binary_scores, group_scores)


def _tune_binary_thresholds(
    binary_scores: dict[str, tuple[np.ndarray, np.ndarray]], *, min_positive: int = 1
) -> dict[str, float]:
    """F1-optimal per-head decision threshold on ``sigmoid(logit)``.

    Reuses :func:`group_validation.find_per_class_thresholds` one column at a time
    (its F1 grid sweep + min-positive 0.5 fallback), so a head below
    ``min_positive`` val positives keeps the naive 0.5.
    """
    from bittrainer.group_validation import find_per_class_thresholds

    thresholds: dict[str, float] = {}
    for concept, (logits, targets) in binary_scores.items():
        probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
        labels = np.asarray(targets, dtype=np.float64)
        column = find_per_class_thresholds(
            probs.reshape(-1, 1), labels.reshape(-1, 1), min_positive=min_positive
        )
        thresholds[concept] = float(column[0])
    return thresholds


def _fit_binary_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Deterministic grid search for the BCE-NLL-minimising temperature.

    Sweeps ``T in geomspace(0.25, 4.0, 33)`` and returns the ``T`` minimising the
    mean BCE negative-log-likelihood of ``sigmoid(logit / T)``. ``T > 1`` cools an
    overconfident head; ``T ~ 1`` means the logits are already calibrated.
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    best_t = 1.0
    best_nll = np.inf
    for t in np.geomspace(0.25, 4.0, 33):
        p = 1.0 / (1.0 + np.exp(-logits / t))
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        nll = float(-(targets * np.log(p) + (1.0 - targets) * np.log(1.0 - p)).mean())
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


# --------------------------------------------------------------------------- #
# Worker                                                                      #
# --------------------------------------------------------------------------- #


def _stringify(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _train_backbone(
    request: dict,
    emit: Callable[[dict], None],
    stop: threading.Event,
    pause_event: object | None = None,
    stop_event: object | None = None,
    stop_now_event: object | None = None,
) -> dict:
    """Worker-thread target: drive :class:`GenericTrainer` with a ``BackboneTask``.

    Kept as the same-signature thread body ``run_backbone_training`` dispatches to
    (Bitcrush ISSUE-0542 Step 6). ``stop`` is the cancellation event — set from the
    async wrapper on ``CancelledError`` and raised as ``BackboneTrainingCancelled``
    inside the epoch loop; ``pause_event`` rides the generic backup/pause machinery.
    The multi-task epoch body, best/heads tracking and safetensors export live in
    the task; the lifecycle (backup / resume / pause / best / patience / finalise)
    runs through the shared core.
    """
    from bittrainer.generic.generic_trainer import GenericTrainer
    from bittrainer.generic.tasks.backbone_task import BackboneTask

    task = BackboneTask(request, cancel_event=stop, stop_event=stop_event)
    return GenericTrainer().run(
        task,
        progress_callback=emit,
        pause_event=pause_event,
        stop_event=task.steps_stop_event,
        stop_now_event=stop_now_event,
    )


def _train_backbone_heads(
    request: dict,
    emit: Callable[[dict], None],
    stop: threading.Event,
    pause_event: object | None = None,
    stop_event: object | None = None,
    stop_now_event: object | None = None,
) -> dict:
    """Worker-thread target for head-only retraining (frozen trunk, cached
    features); same shape as :func:`_train_backbone`."""
    from bittrainer.generic.generic_trainer import GenericTrainer
    from bittrainer.generic.tasks.backbone_heads_task import BackboneHeadsTask

    task = BackboneHeadsTask(request, cancel_event=stop, stop_event=stop_event)
    return GenericTrainer().run(
        task,
        progress_callback=emit,
        pause_event=pause_event,
        stop_event=task.steps_stop_event,
        stop_now_event=stop_now_event,
    )


# --------------------------------------------------------------------------- #
# Async entry point                                                           #
# --------------------------------------------------------------------------- #

_DONE = object()


async def run_backbone_training(
    request: dict,
    progress_callback=None,
    *,
    pause_event=None,
    stop_event=None,
    stop_now_event=None,
) -> dict:
    """Train a backbone candidate; see module docstring for the contract.

    ``pause_event`` (Bitcrush ISSUE-0405/0476) is an optional threading/mp event.
    When set, the training loop backs up its state and returns
    ``{"paused": True, ...}``. Combined with ``training_config``'s ``backup_dir``
    / ``resume_from`` a later call rebuilds and resumes the run. All new config
    lives on ``training_config``, so the Engine wire contract stays additive.

    ``stop_event`` / ``stop_now_event`` (Bitcrush ISSUE-0554) are the finish-early
    signals: the first breaks at the next epoch boundary, the second also cuts the
    current epoch short. BOTH still run finalisation, so the best candidate is
    exported — unlike a pause, which deliberately ships nothing.
    """
    return await _run_worker_async(
        _train_backbone,
        request,
        progress_callback,
        pause_event,
        stop_event,
        stop_now_event,
    )


async def run_backbone_head_training(
    request: dict,
    progress_callback=None,
    *,
    pause_event=None,
    stop_event=None,
    stop_now_event=None,
) -> dict:
    """Retrain the multi-task heads against a FROZEN backbone.

    Same request contract as :func:`run_backbone_training`, except
    ``backbone_init`` must explicitly select either an active-local checkpoint
    or the exact size-matched approved timm-pretrained model, and
    ``training_config`` may carry ``embedding_cache_dir`` (default: an
    ``.embedding_cache`` dir next to the candidate) and ``head_batch_size``
    (default 256). One embedding pass per backbone era is memoised in the
    :class:`~bittrainer.embedding_cache.EmbeddingCache`; head epochs then run
    over cached pooled vectors, so retraining every head after a labelling
    round takes minutes. The exported candidate carries the frozen trunk plus
    fresh ``heads.*`` tensors (``head_only_retrain`` = "1" in the metadata).
    """
    return await _run_worker_async(
        _train_backbone_heads,
        request,
        progress_callback,
        pause_event,
        stop_event,
        stop_now_event,
    )


async def _run_worker_async(
    worker,
    request: dict,
    progress_callback,
    pause_event,
    stop_event=None,
    stop_now_event=None,
) -> dict:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()

    def emit(message: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, message)

    async def forward() -> None:
        while True:
            message = await queue.get()
            if message is _DONE:
                return
            if progress_callback is None:
                continue
            result = progress_callback(message)
            if inspect.isawaitable(result):
                await result

    forwarder = asyncio.create_task(forward())
    try:
        return await asyncio.to_thread(
            worker, request, emit, stop, pause_event, stop_event, stop_now_event
        )
    except asyncio.CancelledError:
        stop.set()
        raise
    finally:
        queue.put_nowait(_DONE)
        await forwarder
