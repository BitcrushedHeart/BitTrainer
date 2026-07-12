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
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

from bittrainer.backbone_init import apply_backbone_init, wants_timm_pretrained
from bittrainer.model import create_model

logger = logging.getLogger(__name__)

_POSITIVE_LABELS = frozenset({"positive", "explicit_positive", "1", "true"})
_NEGATIVE_LABELS = frozenset(
    {"negative", "known_negative", "explicit_known_negative", "implicit_negative", "0", "false"}
)


class BackboneTrainingCancelled(RuntimeError):
    """Raised inside the worker thread when the caller cancelled the task."""


# --------------------------------------------------------------------------- #
# Sample assembly                                                             #
# --------------------------------------------------------------------------- #


class _Vocab:
    """Label spaces derived from the record payload."""

    def __init__(self, records: list[dict]):
        concepts: set[str] = set()
        group_classes: dict[str, set[str]] = {}
        for record in records:
            for concept, label in (record.get("binary") or {}).items():
                if str(label).lower() in _POSITIVE_LABELS | _NEGATIVE_LABELS:
                    concepts.add(concept)
            for group, class_name in (record.get("groups") or {}).items():
                if class_name:
                    group_classes.setdefault(group, set()).add(str(class_name))
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
            if lowered in _POSITIVE_LABELS:
                binary[concept] = 1.0
            elif lowered in _NEGATIVE_LABELS:
                binary[concept] = 0.0
        groups: dict[str, int] = {}
        for group, class_name in (record.get("groups") or {}).items():
            classes = vocab.groups.get(group)
            if classes and str(class_name) in classes:
                groups[group] = classes.index(str(class_name))
        if binary or groups:
            samples.append(_Sample(path, binary, groups))
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
) -> dict[str, float]:
    backbone.eval()
    heads.eval()
    correct: dict[str, int] = {}
    total: dict[str, int] = {}
    for images, binary_labels, group_labels in loader:
        features = backbone(images.to(device))
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
) -> dict:
    config = dict(request.get("training_config") or {})
    image_size = int(config.get("image_size") or 384)
    batch_size = int(config.get("batch_size") or 8)
    epochs = int(config.get("epochs") or 10)
    max_steps = config.get("max_steps")
    learning_rate = float(config.get("learning_rate") or 1e-4)
    validation_split = float(config.get("validation_split") or 0.15)
    requested_device = config.get("device")
    device = torch.device(
        requested_device
        if requested_device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    records = list(request.get("records") or [])
    vocab = _Vocab(records)
    if not vocab.has_targets:
        raise RuntimeError(
            "Backbone training needs at least one labelled binary concept or "
            "group with 2+ classes; the dataset audit found none."
        )
    samples, missing = _build_samples(records, vocab)
    if not samples:
        raise RuntimeError("No labelled images with existing files to train on.")
    if missing:
        logger.warning("Backbone training: %d records had no existing file on disk", missing)
    train_samples, val_samples = _split_samples(records, samples, validation_split)

    seq = 1

    def progress(stage: str, status_text: str, **extra) -> None:
        nonlocal seq
        seq += 1
        emit(
            {
                "type": "training_progress",
                "stage": stage,
                "status_text": status_text,
                "run_id": request.get("run_id"),
                "seq": seq,
                **extra,
            }
        )

    progress(
        "preparing",
        f"Preparing backbone training ({len(train_samples)} train / {len(val_samples)} val)",
        train_samples=len(train_samples),
        val_samples=len(val_samples),
        concepts=len(vocab.concepts),
        groups=len(vocab.groups),
    )

    backbone_spec = request.get("backbone_init")
    backbone = create_model(
        model_size=request.get("convnextv2_size") or "nano",
        pretrained=wants_timm_pretrained(backbone_spec),
        num_classes=0,
    )
    apply_backbone_init(backbone, backbone_spec)
    backbone = backbone.to(device)
    heads = _MultiTaskHeads(backbone.num_features, vocab).to(device)

    loader_kwargs = {"batch_size": batch_size, "collate_fn": _collate, "num_workers": 0}
    train_loader = DataLoader(
        _BackboneDataset(train_samples, _train_transform(image_size)),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        _BackboneDataset(val_samples, _val_transform(image_size)),
        shuffle=False,
        **loader_kwargs,
    )

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(heads.parameters()), lr=learning_rate
    )
    backbone.train()
    heads.train()

    step = 0
    steps_exhausted = max_steps is not None and int(max_steps) <= 0
    for epoch in range(epochs):
        if steps_exhausted:
            break
        epoch_loss = 0.0
        epoch_batches = 0
        for images, binary_labels, group_labels in train_loader:
            if stop.is_set():
                raise BackboneTrainingCancelled
            if max_steps is not None and step >= int(max_steps):
                steps_exhausted = True
                break
            optimizer.zero_grad(set_to_none=True)
            features = backbone(images.to(device))
            loss = _batch_loss(features, heads, binary_labels, group_labels, device)
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            step += 1
            epoch_loss += float(loss.detach())
            epoch_batches += 1
        progress(
            "training",
            f"Epoch {epoch + 1}/{epochs}",
            epoch=epoch + 1,
            epochs=epochs,
            steps=step,
            loss=epoch_loss / max(epoch_batches, 1),
        )

    progress("validating", "Validating backbone candidate")
    validation_metrics = (
        _evaluate(backbone, heads, val_loader, vocab, device) if val_samples else {}
    )
    validation_score = (
        sum(validation_metrics.values()) / len(validation_metrics)
        if validation_metrics
        else 0.0
    )

    candidate_path = Path(request["candidate_checkpoint_path"])
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "family_name": request.get("family_name"),
        "architecture": request.get("architecture"),
        "size_alias": request.get("size_alias"),
        "display_size": request.get("display_size"),
        "convnextv2_size": request.get("convnextv2_size"),
        "version": "1",
        "status": "candidate",
        "created_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "training_run_id": request.get("run_id"),
        "dataset_snapshot_id": request.get("dataset_snapshot_id"),
        "content_hash_index_id": request.get("content_hash_index_id"),
        "license_provenance": request.get("license_provenance") or "locally_trained",
        "external_pretrained_used": bool(request.get("external_pretrained_used")),
        "temporary_timm_fallback_used": bool(request.get("temporary_timm_fallback_used")),
        "release_blocking": bool(request.get("release_blocking")),
        "validation_score": validation_score,
        "validation_metrics_json": validation_metrics,
        "heads_json": request.get("heads") or {},
        "training_config_json": config,
    }
    from safetensors.torch import save_file

    state = {key: value.detach().to("cpu") for key, value in backbone.state_dict().items()}
    save_file(
        state,
        str(candidate_path),
        metadata={key: _stringify(value) for key, value in metadata.items() if value is not None},
    )
    progress(
        "saving",
        "Backbone candidate checkpoint written",
        candidate_checkpoint_path=str(candidate_path),
        validation_score=validation_score,
    )

    return {
        "candidate_checkpoint_path": str(candidate_path),
        "validation_score": float(validation_score),
        "validation_metrics": validation_metrics,
        "heads": request.get("heads") or {},
        "release_blocking": bool(request.get("release_blocking")),
    }


# --------------------------------------------------------------------------- #
# Async entry point                                                           #
# --------------------------------------------------------------------------- #

_DONE = object()


async def run_backbone_training(request: dict, progress_callback=None) -> dict:
    """Train a backbone candidate; see module docstring for the contract."""
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
        return await asyncio.to_thread(_train_backbone, request, emit, stop)
    except asyncio.CancelledError:
        stop.set()
        raise
    finally:
        queue.put_nowait(_DONE)
        await forwarder
