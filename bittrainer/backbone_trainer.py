"""Backbone Builder training entry point for Bitcrush Engine.

Engine's BackboneTrainingManager calls ``run_backbone_training(request,
progress_callback=...)`` with a request assembled from the suite DB:

    {
        "run_id": str,
        "family_name" / "architecture" / "size_alias" / "display_size"
            / "convnextv2_size": str,
        "candidate_checkpoint_path": str,   # where to write the safetensors
        "records": [ {content_hash, file_paths, binary{concept: label},
                      groups{group: class}, splits{split: n},
                      held?: [concept or group, ...]} ],
                                            # held: answers of this record that
                                            # are validation answers at home;
                                            # they never train, and when any
                                            # record has one they are the
                                            # validation set (see _split_samples)
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
                                                     # Optional keys for those implied
                                                     # negatives, all off by default
                                                     # (Bitcrush ISSUE-1655):
                                                     #  "implicit_negative_loss": "bce"
                                                     #    | "hill" (see _batch_loss),
                                                     #  "reject_rate": 0.0 and
                                                     #  "reject_from_epoch": 2
                                                     #    (see _reject_share),
                                                     #  "implicit_negative_max_prior":
                                                     #    None (see _plan_epoch_samples).
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
                            hflip,                   # mirror half the training
                                                     # images; default false (see
                                                     # _train_transform)
                            ...},
        "backbone_init": {...},             # see bittrainer.backbone_init
        "geometry_sensitive": [...],        # optional: groups and yes/no heads
                                            # whose answer depends on framing,
                                            # position or tilt; see
                                            # _GeometryCarefulTransform
        "distillation": {...},              # optional: a teacher candidate to
                                            # distil from; see
                                            # generic.tasks.backbone_distill_task
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

# AMP autocast dtype aliases (Bitcrush ISSUE-0476). bfloat16 needs no GradScaler;
# float16 on CUDA does, and the backbone task steps through one whenever the
# resolved dtype is float16 (Bitcrush ISSUE-1655, see bittrainer.amp).
_AMP_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _amp_settings(
    config: dict, device: torch.device | None = None
) -> tuple[bool, torch.dtype]:
    """Resolve (enabled, autocast dtype) from the training config.

    AMP defaults ON with bfloat16 (matches the binary trainer). Disabled when
    ``use_amp`` is false or the resolved dtype is float32.

    With a ``device`` the dtype is fitted to the card (Bitcrush ISSUE-1655): an
    absent or ``"auto"`` ``amp_dtype`` picks bf16 where it is native and fp16
    where it is not, and an explicit bf16 on a card without it becomes fp16.
    Without a device the answer is the legacy one, untouched.
    """
    name = str(config.get("amp_dtype") or "").lower()
    if device is None:
        dtype = _AMP_DTYPES.get(name, torch.bfloat16)
    else:
        from bittrainer import amp

        dtype = amp.resolve_amp_dtype(_AMP_DTYPES.get(name), device)
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
    # The concept heads are one matrix (see _MultiTaskHeads). A backup taken when
    # each concept was its own Linear holds optimiser state for a tensor per
    # concept; without this key it would match and then fail to load.
    fingerprint["heads"] = "matrix"
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
        # Each concept's prior, written by _plan_epoch_samples from the training
        # images; validation bands its recall figures by it.
        self.priors: dict[str, float] = {}

    @property
    def has_targets(self) -> bool:
        return bool(self.concepts or self.groups)


class _ImpliedNegative(float):
    """The target of a label nobody gave: an unlabelled image drawn as a negative.

    It is still a float, so it travels in a sample's ``binary`` dict through the
    dataset, the loader workers and the collate like any other target. The type
    is what tells ``_batch_loss`` the pair is implied, never the value: a human
    no is a plain ``0.0``, and the implied target may be set to ``0.0`` too.
    """

    __slots__ = ()


class _Sample:
    """One image's labels. ``held_binary`` / ``held_groups`` are the answers
    held back for validation at home (Bitcrush ISSUE-1718): never in ``binary``
    or ``groups``, so they never train, and kept so the planner knows the image
    is not unlabelled for those heads."""

    __slots__ = ("path", "binary", "groups", "held_binary", "held_groups")

    def __init__(
        self,
        path: str,
        binary: dict[str, float],
        groups: dict[str, int],
        held_binary: dict[str, float] | None = None,
        held_groups: dict[str, int] | None = None,
    ):
        self.path = path
        self.binary = binary
        self.groups = groups
        self.held_binary = held_binary or {}
        self.held_groups = held_groups or {}


HOME_SPLIT = "home_split"
CONTENT_HASH = "content_hash"


def validation_source(records: list[dict]) -> str:
    """Where a run's validation comes from: ``"home_split"`` when any record
    holds an answer back (see :func:`_split_samples`), else ``"content_hash"``."""
    return HOME_SPLIT if any(record.get("held") for record in records) else CONTENT_HASH


def _build_samples(records: list[dict], vocab: _Vocab) -> tuple[list[_Sample], int]:
    """One sample per image with a file and a usable answer.

    A record's ``held`` answers go to the sample's ``held_binary`` /
    ``held_groups`` and never to its ``binary`` / ``groups``; a sample whose
    answers are all held is still built, for validation."""
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
        held = {str(name) for name in record.get("held") or ()}
        binary: dict[str, float] = {}
        held_binary: dict[str, float] = {}
        for concept, label in (record.get("binary") or {}).items():
            lowered = str(label).lower()
            key = _safe_module_key(concept)
            into = held_binary if concept in held else binary
            if lowered in _POSITIVE_LABELS:
                into[key] = 1.0
            elif lowered in _NEGATIVE_LABELS:
                into[key] = 0.0
        groups: dict[str, int] = {}
        held_groups: dict[str, int] = {}
        for group, class_name in (record.get("groups") or {}).items():
            key = _safe_module_key(group)
            classes = vocab.groups.get(key)
            if classes and str(class_name) in classes:
                into = held_groups if group in held else groups
                into[key] = classes.index(str(class_name))
        if not (binary or groups or held_binary or held_groups):
            continue
        dedup_key = str(record.get("content_hash") or path)
        if dedup_key in seen:
            duplicates += 1
            continue
        seen.add(dedup_key)
        samples.append(_Sample(path, binary, groups, held_binary, held_groups))
    if duplicates:
        logger.warning(
            "Backbone training: skipped %d duplicate record(s) sharing a content_hash",
            duplicates,
        )
    return samples, missing


def _split_samples(
    records: list[dict], samples: list[_Sample], validation_split: float
) -> tuple[list[_Sample], list[_Sample]]:
    """Deterministic content-hash split so re-runs see the same partition.

    When any record holds answers back (``validation_source`` is
    ``"home_split"``, Bitcrush ISSUE-1718) validation is those answers and
    nothing else, and no image is held out by hash: every sample with an answer
    left to train is a training sample (its held answers stay out of its
    labels), and every sample with a held answer gives a validation sample
    carrying only those. An image can be on both sides, with disjoint answers.
    Training is never swapped with validation here: that would train on them.
    """
    if validation_source(records) == HOME_SPLIT:
        train = [sample for sample in samples if sample.binary or sample.groups]
        val = [
            _Sample(sample.path, dict(sample.held_binary), dict(sample.held_groups))
            for sample in samples
            if sample.held_binary or sample.held_groups
        ]
        return train, val
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

    Implied negatives are written as :class:`_ImpliedNegative`, so the loss can
    tell them from a human no. A concept's prior is the share of the training
    images that carry a human yes for it: a lower bound of the true prior,
    since unlabelled images hold positives too. It is reported in the stats and
    kept on ``vocab.priors``. With ``implicit_negative_max_prior`` set, a
    concept whose prior is above it takes no implied negatives: unlabelled as
    negative helps rare classes and hurts common ones (Ben-Baruch et al. 2022,
    arXiv 2110.10955, who used 0.05 on OpenImages and 0.5 in a COCO example).
    """
    policy = label_policy or {}
    soft = str(policy.get("mode") or "masked_unknown") == "soft_implicit_negative"
    implicit_value = _ImpliedNegative(policy.get("implicit_negative_value", 0.1))
    max_prior = policy.get("implicit_negative_max_prior")
    priors: dict[str, float] = {}

    eff_binary: list[dict[str, float]] = [{} for _ in samples]
    replicas: list[_Sample] = []
    stats: dict[str, dict] = {}
    # Answers held back for validation (Bitcrush ISSUE-1718) are not in
    # ``binary``, so they cannot be drawn as a label or a replica. An image
    # holding ANY answer back is never drawn as an implied negative, for any
    # head: at home a concept's validation negatives include other concepts'
    # validation yes images, so an image held for one head may be a validation
    # no for another without carrying an answer for it here. Its other answers
    # train as usual. An image held for a head does not count as an image
    # without a yes in that head's prior.
    held_count: dict[str, int] = {}
    for sample in samples:
        for concept in sample.held_binary:
            held_count[concept] = held_count.get(concept, 0) + 1

    for concept in vocab.concepts:
        rng = random.Random(f"{seed}|{epoch}|{concept}")
        pos = [i for i, s in enumerate(samples) if s.binary.get(concept) == 1.0]
        neg = [i for i, s in enumerate(samples) if s.binary.get(concept) == 0.0]
        pos_total = len(pos)
        priors[concept] = pos_total / max(len(samples) - held_count.get(concept, 0), 1)
        common = max_prior is not None and priors[concept] > float(max_prior)
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
            [
                i
                for i, s in enumerate(samples)
                if concept not in s.binary and not (s.held_binary or s.held_groups)
            ]
            if soft and not common
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
            "prior": priors[concept],
        }
    vocab.priors = priors

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


class _GeometryCarefulTransform:
    """The training transform of a run that names geometry-sensitive heads.

    Some answers depend on framing, position or tilt: a camera-roll group, a
    subject-location head. RandomResizedCrop reframes an image, and
    RandAugment(2, 9) may rotate it by 9 degrees, shear it or shift it by 13%,
    so for those heads the usual pipeline trains wrong labels. An image with a
    human label for a listed head is therefore prepared like a validation image
    and gets the colour jitter alone, as the group trainer does for its spatial
    groups (``gpu_augment``: ``photometric_only``). Every other image gets
    ``usual``. The choice is per image, so it is made here with the labels in
    hand, by the dataset, and not inside one shared pipeline.

    An implied negative is not a label for this purpose: which images get the
    full augmentation must not change with the label policy being compared.
    """

    def __init__(self, sensitive, usual, colour_only):
        self.sensitive = frozenset(_safe_module_key(str(name)) for name in sensitive)
        self.usual = usual
        self.colour_only = colour_only

    def __call__(self, image, binary: dict, groups: dict) -> torch.Tensor:
        careful = not self.sensitive.isdisjoint(groups) or any(
            head in self.sensitive and not isinstance(value, _ImpliedNegative)
            for head, value in binary.items()
        )
        return (self.colour_only if careful else self.usual)(image)


class _BackboneDataset(Dataset):
    def __init__(self, samples: list[_Sample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as img:
            image = img.convert("RGB")
            if isinstance(self.transform, _GeometryCarefulTransform):
                tensor = self.transform(image, sample.binary, sample.groups)
            else:
                tensor = self.transform(image)
        return tensor, sample.binary, sample.groups


_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _train_transform(
    image_size: int, aug: dict | None = None
) -> transforms.Compose | _GeometryCarefulTransform:
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
    * ``hflip`` (False) -> ``RandomHorizontalFlip``. Off since Bitcrush
      ISSUE-1655: a mirror changes the answer for direction, roll and position
      groups (a gaze to the left becomes a gaze to the right), every image was
      flipped half the time whatever it was labelled, and nobody can vouch for
      every concept. The knob stays so the two can be compared.
    * ``geometry_sensitive`` (none) -> the names of heads whose images take the
      colour jitter only; the result is then a
      :class:`_GeometryCarefulTransform`, which is called with the labels.
    """
    aug = aug or {}
    use_rrc = bool(aug.get("use_random_resized_crop", True))
    rrc_scale_min = float(aug.get("rrc_scale_min", 0.6))
    ra_n = int(aug.get("randaugment_n", 2))
    ra_m = int(aug.get("randaugment_m", 9))
    erasing_p = float(aug.get("random_erasing_p", 0.25))
    sensitive = aug.get("geometry_sensitive")
    jitter = transforms.ColorJitter(
        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
    )

    ops: list = []
    if use_rrc:
        ops.append(transforms.RandomResizedCrop(image_size, scale=(rrc_scale_min, 1.0)))
    else:
        ops.append(transforms.Resize((image_size, image_size)))
    if aug.get("hflip", False):
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    if ra_n > 0 and ra_m > 0:
        ops.append(transforms.RandAugment(num_ops=ra_n, magnitude=ra_m))
    ops.append(jitter)
    ops.append(transforms.ToTensor())
    ops.append(_NORMALIZE)
    if erasing_p > 0:
        ops.append(transforms.RandomErasing(p=erasing_p))
    usual = transforms.Compose(ops)
    if not sensitive:
        return usual
    colour_only = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            jitter,
            transforms.ToTensor(),
            _NORMALIZE,
        ]
    )
    return _GeometryCarefulTransform(sensitive, usual, colour_only)


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
    """The training heads on the trunk's pooled feature.

    Every yes/no concept is a row of ``binary_weight`` and an entry of
    ``binary_bias`` (Bitcrush ISSUE-1655); each group keeps a ``Linear``. The
    concepts used to be a ``Linear`` each, which at 900 concepts gave the
    optimiser, the gradient scaler, the clipper and the EMA 1,800 small tensors
    to walk every step: on a T4 a Base step took 4.8 s instead of 1.1 s.

    What is saved keeps the older layout, a weight and a bias per concept
    (``binary.<concept>.weight`` of shape ``[1, features]``), so candidates,
    backups and everything that reads them are unaffected, and a file written
    with a ``Linear`` per concept loads.
    """

    def __init__(self, feature_dim: int, vocab: _Vocab):
        super().__init__()
        self.binary_names: list[str] = list(vocab.concepts)
        self.binary_index: dict[str, int] = {
            name: column for column, name in enumerate(self.binary_names)
        }
        weight = torch.empty(len(self.binary_names), feature_dim)
        bias = torch.empty(len(self.binary_names))
        if self.binary_names:
            # What a Linear(feature_dim, 1) per concept drew its values from.
            bound = 1.0 / math.sqrt(feature_dim)
            nn.init.uniform_(weight, -bound, bound)
            nn.init.uniform_(bias, -bound, bound)
        self.binary_weight = nn.Parameter(weight)
        self.binary_bias = nn.Parameter(bias)
        self.groups = nn.ModuleDict(
            {
                group: nn.Linear(feature_dim, len(classes))
                for group, classes in vocab.groups.items()
            }
        )

    def binary_rows(self, names) -> tuple[torch.Tensor, torch.Tensor]:
        """The weight rows and biases of the named concepts, in that order."""
        index = torch.tensor(
            [self.binary_index[name] for name in names],
            dtype=torch.long,
            device=self.binary_weight.device,
        )
        return self.binary_weight[index], self.binary_bias[index]

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        weight, bias = self.binary_weight, self.binary_bias
        if not keep_vars:
            weight, bias = weight.detach(), bias.detach()
        for column, name in enumerate(self.binary_names):
            destination[f"{prefix}binary.{name}.weight"] = weight[column : column + 1]
            destination[f"{prefix}binary.{name}.bias"] = bias[column : column + 1]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        known: set[str] = set()
        parts = (("weight", self.binary_weight), ("bias", self.binary_bias))
        with torch.no_grad():
            for column, name in enumerate(self.binary_names):
                for part, parameter in parts:
                    key = f"{prefix}binary.{name}.{part}"
                    known.add(key)
                    row = parameter[column : column + 1]
                    value = state_dict.get(key)
                    if value is None:
                        missing_keys.append(key)
                    elif value.shape != row.shape:
                        error_msgs.append(
                            f"size mismatch for {key}: copying a param with shape "
                            f"{tuple(value.shape)} from checkpoint, the shape in "
                            f"current model is {tuple(row.shape)}."
                        )
                    else:
                        row.copy_(value)
        if strict:
            for key in state_dict:
                if not key.startswith(prefix):
                    continue
                first = key[len(prefix) :].split(".", 1)[0]
                if (
                    (key not in known)
                    if first == "binary"
                    else (first not in self._modules)
                ):
                    unexpected_keys.append(key)


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


def _reject_share(policy: dict, epoch: int | None) -> float:
    """The share of a batch's implied negatives that large-loss rejection drops.

    Large-loss rejection (Kim et al. 2022, "Large Loss Matters", arXiv
    2206.03740): nothing in the first epoch, then a share that grows by
    ``reject_rate`` every epoch, because a network fits the clean labels first
    and memorises the wrong ones later. ``epoch`` counts from 0, as in the
    planner; ``reject_from_epoch`` (default 2, the paper's) counts from 1.
    Off without a rate, and for a caller that gives no epoch.
    """
    rate = float(policy.get("reject_rate") or 0.0)
    if rate <= 0 or epoch is None:
        return 0.0
    first = int(policy.get("reject_from_epoch") or 2)
    return min(1.0, rate * max(0, epoch + 2 - first))


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
    label_policy: dict | None = None,
    epoch: int | None = None,
) -> torch.Tensor | None:
    """Masked multi-task loss over whichever heads each row carries a label for.

    ``label_policy`` and ``epoch`` bring two options for implied negatives
    (targets of type :class:`_ImpliedNegative`); a human yes or no is plain BCE
    whatever they say, and with neither set the loss is what it always was.

    * ``implicit_negative_loss: "hill"`` swaps BCE towards the soft target for
      the negative part of the Hill loss, ``(1.5 - p) * p**2`` (Zhang et al.
      2021, arXiv 2112.07368). BCE pushes an implied negative harder the higher
      the model scores it; this term pushes hardest near 0.5 and lets go as the
      score nears 1, where the pair is more likely a positive nobody labelled.
    * ``reject_rate`` drops the highest-scoring share of the batch's implied
      negatives (see :func:`_reject_share`). A dropped pair is treated as
      unlabelled for the step; it is never trained towards yes.

    Both are worked out in float32 from the logits and stay finite for any
    logit.

    ``group_label_smoothing`` smooths the group-CE targets (parity with the group
    trainer's default). ``reduction`` combines the per-head losses: ``"sum"`` (the
    legacy default, so direct callers are unchanged) or ``"mean"`` — averaging over
    the number of contributing head losses so the gradient scale no longer grows
    with per-batch label density.

    The concept heads run as ONE matrix product (Bitcrush ISSUE-1655), instead
    of a row scan, an index, a Linear and a loss per head: with several hundred
    concepts that was thousands of tiny calls per batch. Each head's loss is
    still the mean over its own labelled rows. A head with no label in the batch
    takes no part in the loss; being a row of the one head matrix, it gets a
    zero gradient rather than none.
    """
    losses: list[torch.Tensor] = []
    bce = nn.functional.binary_cross_entropy_with_logits
    ce = nn.functional.cross_entropy
    policy = label_policy or {}
    hill = str(policy.get("implicit_negative_loss") or "bce") == "hill"
    reject = _reject_share(policy, epoch)

    # One pass over the labels: the (row, column, target) of every label, a
    # column per concept head, and whether the label is an implied negative.
    column_of = heads.binary_index
    label_rows: list[int] = []
    label_cols: list[int] = []
    label_values: list[float] = []
    label_implied: list[bool] = []
    for row, labels in enumerate(binary_labels):
        for concept, value in labels.items():
            column = column_of.get(concept)
            if column is None:
                continue
            label_rows.append(row)
            label_cols.append(column)
            label_values.append(value)
            label_implied.append(isinstance(value, _ImpliedNegative))
    if label_rows:
        logits = nn.functional.linear(
            features, heads.binary_weight, heads.binary_bias
        ).float()
        index = (
            torch.tensor(label_rows, device=device),
            torch.tensor(label_cols, device=device),
        )
        targets = torch.zeros(logits.shape, device=device, dtype=torch.float32)
        targets[index] = torch.tensor(label_values, device=device, dtype=torch.float32)
        labelled = torch.zeros_like(targets)
        labelled[index] = 1.0
        head_weight = None
        if pos_weight:
            weights = [
                1.0 if (value := pos_weight.get(concept)) is None else float(value)
                for concept in heads.binary_names
            ]
            if any(value != 1.0 for value in weights):
                head_weight = torch.tensor(weights, device=device, dtype=torch.float32)
        per_label = bce(logits, targets, pos_weight=head_weight, reduction="none")
        if (hill or reject > 0) and any(label_implied):
            implied = torch.zeros(logits.shape, device=device, dtype=torch.bool)
            implied[index] = torch.tensor(label_implied, device=device)
            if hill:
                prob = torch.sigmoid(logits)
                per_label = torch.where(implied, (1.5 - prob) * prob * prob, per_label)
            if reject > 0:
                # Ranked by score, which is the paper's ranking by loss towards
                # 0. Not by the loss towards the soft target: that one is also
                # large for a pair the model is sure is a no. The share is
                # rounded to whole pairs, so a small batch may drop none.
                scores = logits.detach()
                implied_scores = scores[implied]
                drop = min(int(reject * len(implied_scores) + 0.5), len(implied_scores))
                if drop:
                    cut = torch.topk(implied_scores, drop).values[-1]
                    labelled = labelled.masked_fill(implied & (scores >= cut), 0.0)
        # Mean over each head's own labelled rows, for the heads that have any.
        counts = labelled.sum(0)
        present = counts > 0
        losses.append((per_label * labelled).sum(0)[present] / counts[present])

    group_heads = heads.groups
    group_rows: dict[str, tuple[list[int], list[int]]] = {}
    for row, labels in enumerate(group_labels):
        for group, target in labels.items():
            if group not in group_heads:
                continue
            rows, targets_list = group_rows.setdefault(group, ([], []))
            rows.append(row)
            targets_list.append(target)
    for group, (rows, targets_list) in group_rows.items():
        logits = group_heads[group](features[rows])
        group_targets = torch.tensor(targets_list, device=device)
        losses.append(
            ce(logits, group_targets, label_smoothing=group_label_smoothing).reshape(1)
        )
    if not losses:
        return None
    stacked = torch.cat(losses)
    if stacked.numel() == 0:  # every label of the batch was a rejected one
        return None
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

    Every head's logits come from ONE matrix product and ONE device-to-CPU copy
    per batch (Bitcrush ISSUE-1655); the per-head split happens in numpy. The
    old loop made a call and a copy per head per batch. Output is unchanged:
    heads in their own order, rows in loader order, unlabelled heads left out.
    """
    backbone.eval()
    heads.eval()
    concepts = list(heads.binary_names)
    groups = list(heads.groups.keys())
    concept_column = {concept: column for column, concept in enumerate(concepts)}
    # Columns: one per concept, then each group's classes side by side.
    group_span: dict[str, tuple[int, int]] = {}
    offset = len(concepts)
    for group in groups:
        width = heads.groups[group].out_features
        group_span[group] = (offset, offset + width)
        offset += width
    group_heads = [heads.groups[g] for g in groups]
    weight = bias = None
    if concepts or groups:
        weight = torch.cat([heads.binary_weight, *[m.weight for m in group_heads]])
        bias = torch.cat([heads.binary_bias, *[m.bias for m in group_heads]])

    bin_columns: list[np.ndarray] = []
    bin_logits: list[np.ndarray] = []
    bin_targets: list[np.ndarray] = []
    grp_logits: dict[str, list[np.ndarray]] = {g: [] for g in groups}
    grp_targets: dict[str, list[np.ndarray]] = {g: [] for g in groups}
    for images, binary_labels, group_labels in loader:
        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            features = backbone(images.to(device))
        if weight is None:
            continue
        # Heads run in fp32; avoids the autocast dtype mismatch.
        logits = nn.functional.linear(features.float(), weight, bias).cpu().numpy()

        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for row, labels in enumerate(binary_labels):
            for concept, value in labels.items():
                column = concept_column.get(concept)
                if column is not None:
                    rows.append(row)
                    columns.append(column)
                    values.append(value)
        if rows:
            bin_columns.append(np.asarray(columns, dtype=np.int64))
            bin_logits.append(logits[rows, columns].astype(np.float64))
            bin_targets.append(np.asarray(values, dtype=np.float64))

        by_group: dict[str, tuple[list[int], list[int]]] = {}
        for row, labels in enumerate(group_labels):
            for group, target in labels.items():
                if group in group_span:
                    group_rows, group_targets = by_group.setdefault(group, ([], []))
                    group_rows.append(row)
                    group_targets.append(target)
        for group, (group_rows, group_targets) in by_group.items():
            start, stop = group_span[group]
            grp_logits[group].append(logits[group_rows, start:stop].astype(np.float64))
            grp_targets[group].append(np.asarray(group_targets, dtype=np.int64))
    backbone.train()
    heads.train()

    # Split the concept labels back out per head. The stable sort keeps each
    # head's rows in loader order; np.unique walks the heads in their own order.
    binary_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if bin_columns:
        all_columns = np.concatenate(bin_columns)
        order = np.argsort(all_columns, kind="stable")
        all_columns = all_columns[order]
        all_logits = np.concatenate(bin_logits)[order]
        all_targets = np.concatenate(bin_targets)[order]
        present, starts = np.unique(all_columns, return_index=True)
        stops = [*starts[1:], len(all_columns)]
        for column, start, stop in zip(present, starts, stops):
            binary_scores[concepts[column]] = (
                all_logits[start:stop],
                all_targets[start:stop],
            )
    group_scores = {
        g: (np.concatenate(grp_logits[g]), np.concatenate(grp_targets[g]))
        for g in groups
        if grp_logits[g]
    }
    return binary_scores, group_scores


# A head needs this many validation labels of each kind before its recall at a
# fixed precision means much: with fewer, one image moves it by a tenth or more.
_RAP_MIN_LABELS = 10
_RAP_TARGETS = ((90, 0.90), (95, 0.95))
# Concept frequency bands: a prior (see _plan_epoch_samples) below each bound.
# 0.05 is where Ben-Baruch et al. 2022 began masking a class on OpenImages;
# 0.01 has no source.
_PRIOR_BANDS = (("rare", 0.01), ("medium", 0.05), ("common", math.inf))


def _precision_recall(
    probs: np.ndarray, positive: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """A head's precision and recall at every score it gave, best score first.

    scikit-learn's ``precision_recall_curve`` without its cost per call: one
    point per distinct score, calling a yes on everything that scores at least
    that much. ``positive`` must hold at least one True.
    """
    order = np.argsort(-probs)  # how tied scores fall does not matter below
    ranked = probs[order]
    # The last row of each run of equal scores.
    ends = np.flatnonzero(np.r_[np.diff(ranked) != 0, True])
    hits = np.cumsum(positive[order])[ends]
    return hits / (ends + 1), hits / hits[-1]


def _recall_at_precision(precision: np.ndarray, recall: np.ndarray) -> dict[int, float]:
    """The best recall with precision at least 0.90 and 0.95, keyed 90 and 95;
    0.0 where no threshold reaches the precision."""
    return {
        tag: float(recall[precision >= target].max(initial=0.0))
        for tag, target in _RAP_TARGETS
    }


def _confusion(
    targets: np.ndarray, predictions: np.ndarray, classes: int
) -> np.ndarray:
    """Counts ``[true class, predicted class]`` for one group."""
    return np.bincount(
        targets * classes + predictions, minlength=classes * classes
    ).reshape(classes, classes)


def _per_class_scores(
    confusion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision, recall and F1 per class; 0 where there is nothing to divide by
    (scikit-learn's ``zero_division=0``)."""
    hits = np.diag(confusion).astype(np.float64)
    actual, called = confusion.sum(1), confusion.sum(0)
    return (
        hits / np.maximum(called, 1),
        hits / np.maximum(actual, 1),
        2.0 * hits / np.maximum(actual + called, 1),
    )


def _backbone_metrics(
    binary_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    group_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    priors: dict[str, float] | None = None,
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

    Recall at precision 0.90 and 0.95 (Bitcrush ISSUE-1655) is the figure that
    compares ways of treating unlabelled images: a tag is only written above a
    precision target, so what matters is how much is still found there. A head
    with at least ``_RAP_MIN_LABELS`` validation yes and as many no gets
    ``binary_r_at_p90/<c>`` and ``binary_r_at_p95/<c>``. Their macro averages are
    ``binary_macro_r_at_p90`` / ``..._p95`` and, for the concepts ``priors``
    covers, the same with ``_rare`` / ``_medium`` / ``_common`` appended;
    ``binary_r_at_p_heads`` (and its band forms) count the heads averaged.

    All of it is numpy. Two scikit-learn calls per concept head and a dozen per
    group cost 4.75 s for 900 heads on 582 validation images on a T4, and grow
    with the validation set.
    """
    metrics: dict[str, float] = {}
    binary_f1s: list[float] = []
    ranked_heads: list[str] = []
    for concept, (logits, targets) in binary_scores.items():
        logits = np.asarray(logits, dtype=np.float64)
        tgt_int = np.asarray(targets, dtype=np.float64).astype(np.int64)
        preds = (logits > 0).astype(np.int64)
        n = int(tgt_int.shape[0])
        metrics[f"binary/{concept}"] = float((preds == tgt_int).mean()) if n else 0.0
        support = int(tgt_int.sum())
        metrics[f"binary_support/{concept}"] = support
        if support >= 1:
            positive = tgt_int == 1
            # F1 = 2 TP / (called + actual positives).
            f1 = float(2.0 * preds[positive].sum() / (preds.sum() + support))
            precision, recall = _precision_recall(
                1.0 / (1.0 + np.exp(-logits)), positive
            )
            metrics[f"binary_f1/{concept}"] = f1
            # Average precision: each gain in recall, weighted by the precision
            # it was reached at.
            metrics[f"binary_ap/{concept}"] = float(
                np.sum(np.diff(recall, prepend=0.0) * precision)
            )
            binary_f1s.append(f1)
            if min(support, n - support) >= _RAP_MIN_LABELS:
                ranked_heads.append(concept)
                for tag, value in _recall_at_precision(precision, recall).items():
                    metrics[f"binary_r_at_p{tag}/{concept}"] = value

    if ranked_heads:
        bands: dict[str, list[str]] = {"": ranked_heads}
        for concept in ranked_heads:
            prior = (priors or {}).get(concept)
            if prior is not None:
                band = next(name for name, below in _PRIOR_BANDS if prior < below)
                bands.setdefault(f"_{band}", []).append(concept)
        for suffix, members in bands.items():
            metrics[f"binary_r_at_p_heads{suffix}"] = len(members)
            for tag, _target in _RAP_TARGETS:
                metrics[f"binary_macro_r_at_p{tag}{suffix}"] = float(
                    np.mean([metrics[f"binary_r_at_p{tag}/{c}"] for c in members])
                )

    group_f1s: list[float] = []
    for group, (logits, targets) in group_scores.items():
        logits = np.asarray(logits, dtype=np.float64)
        tgt = np.asarray(targets, dtype=np.int64)
        n = int(tgt.shape[0])
        preds = logits.argmax(axis=1)
        metrics[f"group/{group}"] = float((preds == tgt).mean()) if n else 0.0
        metrics[f"group_val_support/{group}"] = n
        if n:
            # Macro F1 over every class of the head, seen in validation or not.
            confusion = _confusion(tgt, preds, int(logits.shape[1]))
            f1 = float(_per_class_scores(confusion)[2].mean())
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


def _group_validation_report(
    group_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    classes: dict[str, list[str]] | None = None,
) -> dict[str, dict]:
    """Per single-label group, how validation went class by class.

    From the scores :func:`_collect_val_scores` returns::

        {group: {"classes": [name, ...],
                 "support": [validation images of each class],
                 "precision": [...], "recall": [...], "f1": [...],
                 "confusion": [[count, ...], ...]}}

    Every list has one entry per class in the head's own order;
    ``confusion[t][p]`` counts the images of class ``t`` that were called
    class ``p``. ``classes`` (``vocab.groups``) names the classes; without it
    they are numbered. Plain lists and numbers, ready for JSON. Separate from
    :func:`_backbone_metrics`, whose flat dict feeds epoch selection.
    """
    report: dict[str, dict] = {}
    for group, (logits, targets) in group_scores.items():
        logits = np.asarray(logits, dtype=np.float64)
        width = int(logits.shape[1])
        confusion = _confusion(
            np.asarray(targets, dtype=np.int64), logits.argmax(axis=1), width
        )
        precision, recall, f1 = _per_class_scores(confusion)
        report[group] = {
            "classes": list((classes or {}).get(group) or map(str, range(width))),
            "support": confusion.sum(1).tolist(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "confusion": confusion.tolist(),
        }
    return report


def _binary_validation_report(
    binary_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    thresholds: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Per yes/no head, how validation went at the threshold it will use.

    From the scores :func:`_collect_val_scores` returns::

        {concept: {"yes": validation yes labels, "no": validation no labels,
                   "threshold": on sigmoid(logit); 0.5 if ``thresholds`` has none,
                   "called": images at or above it, "hits": the yes among them,
                   "precision": hits / called, None when nothing was called,
                   "recall": hits / yes, None when there is no yes,
                   "recall_at_p90": ..., "recall_at_p95": ...}}

    The last two are the figures of :func:`_backbone_metrics`, None for a head
    with too few labels of either kind to have them.
    """
    report: dict[str, dict] = {}
    for concept, (logits, targets) in binary_scores.items():
        probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
        positive = np.asarray(targets, dtype=np.float64) == 1.0
        yes, no = int(positive.sum()), int((~positive).sum())
        threshold = float((thresholds or {}).get(concept, 0.5))
        called = probs >= threshold
        hits = int((called & positive).sum())
        entry = {
            "yes": yes,
            "no": no,
            "threshold": threshold,
            "called": int(called.sum()),
            "hits": hits,
            "precision": hits / int(called.sum()) if called.any() else None,
            "recall": hits / yes if yes else None,
        }
        at: dict[int, float | None] = dict.fromkeys((tag for tag, _ in _RAP_TARGETS))
        if min(yes, no) >= _RAP_MIN_LABELS:
            at = _recall_at_precision(*_precision_recall(probs, positive))
        entry.update({f"recall_at_p{tag}": value for tag, value in at.items()})
        report[concept] = entry
    return report


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

    The recall-at-precision figures are banded by the priors the epoch plan
    left on ``vocab``.
    """
    binary_scores, group_scores = _collect_val_scores(
        backbone, heads, loader, device, amp_enabled=amp_enabled, amp_dtype=amp_dtype
    )
    if score_sink is not None:
        score_sink["binary"] = binary_scores
        score_sink["groups"] = group_scores
    return _backbone_metrics(
        binary_scores, group_scores, priors=getattr(vocab, "priors", None)
    )


def _binary_threshold_tuning(
    binary_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    min_positive: int = 1,
    min_negative: int = 0,
) -> tuple[dict[str, float], dict[str, bool]]:
    """F1-optimal per-head decision threshold on ``sigmoid(logit)``, and which
    heads had the labels to be tuned.

    The search of :func:`group_validation.find_per_class_thresholds` (its grid,
    the first best value on a tie), done for a head as one numpy comparison
    against the whole grid. Calling the shared function per head cost 36
    scikit-learn calls each: 108 s for 902 heads on a T4 (Bitcrush ISSUE-1655).

    A head with fewer than ``min_positive`` validation yes or ``min_negative``
    validation no keeps 0.5 and is False in the second dict: a threshold tuned
    on a handful of labels is noise.
    """
    grid = np.arange(0.05, 0.95, 0.025)
    thresholds: dict[str, float] = {}
    tuned: dict[str, bool] = {}
    for concept, (logits, targets) in binary_scores.items():
        labels = np.asarray(targets, dtype=np.float64)
        positive = labels == 1.0
        tuned[concept] = (
            int(labels.sum()) >= min_positive and int((~positive).sum()) >= min_negative
        )
        if not tuned[concept]:
            thresholds[concept] = 0.5
            continue
        probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
        called = probs[None, :] >= grid[:, None]
        hits = (called & positive).sum(1)
        # F1 = 2 TP / (called + actual positives); 0 where that is empty.
        f1 = 2.0 * hits / np.maximum(called.sum(1) + positive.sum(), 1)
        thresholds[concept] = float(grid[int(np.argmax(f1))])
    return thresholds, tuned


def _tune_binary_thresholds(
    binary_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    min_positive: int = 1,
    min_negative: int = 0,
) -> dict[str, float]:
    """The thresholds of :func:`_binary_threshold_tuning`, without the record of
    which heads were tuned."""
    return _binary_threshold_tuning(
        binary_scores, min_positive=min_positive, min_negative=min_negative
    )[0]


def _fit_group_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """:func:`_fit_binary_temperature` for a softmax head.

    The ``T`` on the same grid that minimises the mean cross-entropy of
    ``softmax(logits / T)`` over one group's validation images (``logits``
    ``[N, classes]``, ``targets`` ``[N]`` class indices); 1.0 for a group with
    none.
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    rows = np.arange(targets.shape[0])
    best_t = 1.0
    best_nll = np.inf
    if not rows.size:
        return best_t
    for t in np.geomspace(0.25, 4.0, 33):
        scaled = logits / t
        scaled -= scaled.max(axis=1, keepdims=True)  # keeps exp() from overflowing
        nll = float((np.log(np.exp(scaled).sum(axis=1)) - scaled[rows, targets]).mean())
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


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


def backbone_task_class(request: dict):
    """The task a backbone request runs as: ``BackboneDistillTask`` when
    ``request["distillation"]`` names a teacher, else ``BackboneTask``.

    One place decides, because the choice also decides the resume fingerprint:
    anything that asks "do these backups belong to this run?" before training
    (the pack runner does) must ask the same task the run will use.
    """
    if (request.get("distillation") or {}).get("teacher_checkpoint_path"):
        from bittrainer.generic.tasks.backbone_distill_task import BackboneDistillTask

        return BackboneDistillTask
    from bittrainer.generic.tasks.backbone_task import BackboneTask

    return BackboneTask


def _train_backbone(
    request: dict,
    emit: Callable[[dict], None],
    stop: threading.Event,
    pause_event: object | None = None,
    stop_event: object | None = None,
    stop_now_event: object | None = None,
) -> dict:
    """Worker-thread target: drive :class:`GenericTrainer` with a ``BackboneTask``,
    or a ``BackboneDistillTask`` when ``request["distillation"]`` names a teacher.

    Kept as the same-signature thread body ``run_backbone_training`` dispatches to
    (Bitcrush ISSUE-0542 Step 6). ``stop`` is the cancellation event — set from the
    async wrapper on ``CancelledError`` and raised as ``BackboneTrainingCancelled``
    inside the epoch loop; ``pause_event`` rides the generic backup/pause machinery.
    The multi-task epoch body, best/heads tracking and safetensors export live in
    the task; the lifecycle (backup / resume / pause / best / patience / finalise)
    runs through the shared core.
    """
    from bittrainer.generic.generic_trainer import GenericTrainer

    task = backbone_task_class(request)(
        request, cancel_event=stop, stop_event=stop_event
    )
    # The task hands its augmentation settings to _train_transform as they are,
    # so these two reach the training dataset through them.
    task._aug["hflip"] = bool(task.config.get("hflip", False))
    task._aug["geometry_sensitive"] = request.get("geometry_sensitive")
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
