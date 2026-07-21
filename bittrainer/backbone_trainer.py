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
                            learning_rate, validation_split, device?, ...},
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
import threading
from pathlib import Path
from typing import Callable

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


def _backbone_fingerprint(vocab: "_Vocab", model_size: str, epochs: int) -> dict:
    """Resume-compatibility identity for a backbone run.

    Folds the label space (concepts + multi-class groups) into the fingerprint
    so a resume whose dataset changed head layout starts fresh rather than
    crashing on a shape mismatch.
    """
    names = [f"binary/{c}" for c in vocab.concepts] + [f"group/{g}" for g in vocab.groups]
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
    # an older AdamW run (which lacks this key) cleanly mismatches and starts fresh
    # (resume_skipped) rather than trying to load AdamW state into Prodigy_adv.
    fingerprint["optimizer"] = "Prodigy_adv"
    return fingerprint

_POSITIVE_LABELS = frozenset({"positive", "explicit_positive", "1", "true"})
_NEGATIVE_LABELS = frozenset(
    {"negative", "known_negative", "explicit_known_negative", "implicit_negative", "0", "false"}
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
                    group_classes.setdefault(_safe_module_key(group), set()).add(str(class_name))
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
            "Backbone training: skipped %d duplicate record(s) sharing a content_hash", duplicates
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


def _train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            _NORMALIZE,
        ]
    )


def _val_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(), _NORMALIZE]
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
            {group: nn.Linear(feature_dim, len(classes)) for group, classes in vocab.groups.items()}
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
) -> torch.Tensor | None:
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
        losses.append(bce(logits, targets))
    for group, head in heads.groups.items():
        rows = [i for i, labels in enumerate(group_labels) if group in labels]
        if not rows:
            continue
        logits = head(features[rows])
        targets = torch.tensor([group_labels[i][group] for i in rows], device=device)
        losses.append(ce(logits, targets))
    if not losses:
        return None
    return torch.stack(losses).sum()


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
) -> dict[str, float]:
    backbone.eval()
    heads.eval()
    correct: dict[str, int] = {}
    total: dict[str, int] = {}
    for images, binary_labels, group_labels in loader:
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            features = backbone(images.to(device))
        features = features.float()  # heads run in fp32; avoids autocast dtype mismatch
        for concept, head in heads.binary.items():
            rows = [i for i, labels in enumerate(binary_labels) if concept in labels]
            if not rows:
                continue
            predictions = (head(features[rows]).squeeze(-1) > 0).float()
            targets = torch.tensor(
                [binary_labels[i][concept] for i in rows], device=device
            )
            key = f"binary/{concept}"
            correct[key] = correct.get(key, 0) + int((predictions == targets).sum())
            total[key] = total.get(key, 0) + len(rows)
        for group, head in heads.groups.items():
            rows = [i for i, labels in enumerate(group_labels) if group in labels]
            if not rows:
                continue
            predictions = head(features[rows]).argmax(dim=-1)
            targets = torch.tensor([group_labels[i][group] for i in rows], device=device)
            key = f"group/{group}"
            correct[key] = correct.get(key, 0) + int((predictions == targets).sum())
            total[key] = total.get(key, 0) + len(rows)
    backbone.train()
    heads.train()
    return {key: correct[key] / total[key] for key in sorted(total) if total[key]}


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

    task = BackboneTask(request, cancel_event=stop)
    return GenericTrainer().run(
        task,
        progress_callback=emit,
        pause_event=pause_event,
        stop_event=task.steps_stop_event,
    )


# --------------------------------------------------------------------------- #
# Async entry point                                                           #
# --------------------------------------------------------------------------- #

_DONE = object()


async def run_backbone_training(request: dict, progress_callback=None, *, pause_event=None) -> dict:
    """Train a backbone candidate; see module docstring for the contract.

    ``pause_event`` (Bitcrush ISSUE-0405/0476) is an optional threading/mp event.
    When set, the training loop backs up its state and returns
    ``{"paused": True, ...}``. Combined with ``training_config``'s ``backup_dir``
    / ``resume_from`` a later call rebuilds and resumes the run. All new config
    lives on ``training_config``, so the Engine wire contract stays additive.
    """
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
        return await asyncio.to_thread(_train_backbone, request, emit, stop, pause_event)
    except asyncio.CancelledError:
        stop.set()
        raise
    finally:
        queue.put_nowait(_DONE)
        await forwarder
