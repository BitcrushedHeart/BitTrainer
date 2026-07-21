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
from functools import partial
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

from bittrainer.backbone_init import apply_backbone_init, wants_timm_pretrained
from bittrainer.ema import ModelEMA
from bittrainer.model import create_model
from bittrainer.training_state import (
    BackupCoordinator,
    capture_rng_states,
    make_fingerprint,
    paused_result,
    restore_rng_states,
    sanitize_for_backup,
)

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


def _build_scheduler(optimizer, config: dict, *, epochs: int):
    """Cosine LR schedule (default ON), stepped once per epoch. ``None`` when
    ``use_cosine`` is disabled (plain constant-LR AdamW, the legacy behaviour)."""
    if not config.get("use_cosine", True):
        return None
    return CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))


def _backbone_fingerprint(vocab: "_Vocab", model_size: str, epochs: int) -> dict:
    """Resume-compatibility identity for a backbone run.

    Folds the label space (concepts + multi-class groups) into the fingerprint
    so a resume whose dataset changed head layout starts fresh rather than
    crashing on a shape mismatch.
    """
    names = [f"binary/{c}" for c in vocab.concepts] + [f"group/{g}" for g in vocab.groups]
    return make_fingerprint(
        class_names=names,
        num_classes=len(names),
        max_epochs=int(epochs),
        multi_label=True,
        ordinal=False,
        best_model_name="backbone_candidate",
        model_size=str(model_size),
    )

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

    model = _BackboneWithHeads(backbone, heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = _build_scheduler(optimizer, config, epochs=epochs)
    amp_enabled, amp_dtype = _amp_settings(config)
    ema = (
        ModelEMA(model, decay=float(config.get("ema_decay") or 0.9999))
        if config.get("use_ema", True)
        else None
    )
    patience = int(config.get("patience") or config.get("early_stopping_patience") or 0)

    # --- Backup / Pause / Resume (Bitcrush ISSUE-0405 machinery) ---
    fingerprint = _backbone_fingerprint(
        vocab, request.get("convnextv2_size") or "nano", epochs
    )
    coordinator = BackupCoordinator(
        backup_dir=config.get("backup_dir"),
        backup_every_steps=int(config.get("backup_every_steps") or 0),
        pause_event=pause_event,
        cb=emit,
    )
    resume_from = config.get("resume_from")
    resume_state = (
        coordinator.load_resume(fingerprint, resume_from=resume_from) if resume_from else None
    )

    step = 0
    start_epoch = 0
    best_score = -1.0
    best_epoch = 0
    patience_counter = 0
    best_metrics: dict = {}
    best_backbone_state: dict | None = None

    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        if scheduler is not None and resume_state.get("scheduler") is not None:
            scheduler.load_state_dict(resume_state["scheduler"])
        if ema is not None and resume_state.get("ema") is not None:
            ema.load_full_state_dict(resume_state["ema"])
        restore_rng_states(resume_state.get("rng"), device)
        start_epoch = int(resume_state.get("epoch", 0))
        step = int(resume_state.get("global_step", 0))
        b = resume_state.get("best") or {}
        best_score = float(b.get("score", -1.0))
        best_epoch = int(b.get("epoch", 0))
        patience_counter = int(b.get("patience", 0))
        best_metrics = dict(b.get("metrics") or {})
        best_backbone_state = b.get("backbone_state")
        emit({
            "type": "training_resumed",
            "run_id": request.get("run_id"),
            "resumed_from": str(resume_from),
            "epoch": start_epoch,
            "global_step": step,
            "best_score": best_score,
        })

    model.train()

    def _eval_modules():
        src = ema.module if ema is not None else model
        return src.backbone, src.heads

    def _collect(cur_epoch: int) -> dict:
        return {
            "fingerprint": fingerprint,
            "trainer": "backbone",
            "epoch": cur_epoch,
            "batch_in_epoch": 0,
            "global_step": step,
            "eff_bs": batch_size,
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "ema": ema.full_state_dict() if ema is not None else None,
            "rng": capture_rng_states(device),
            "best": {
                "score": best_score,
                "epoch": best_epoch,
                "patience": patience_counter,
                "metrics": sanitize_for_backup(best_metrics),
                "backbone_state": best_backbone_state,
            },
        }

    _paused_result = partial(paused_result, emit)

    validation_metrics: dict = dict(best_metrics)
    validation_score = best_score if best_score >= 0 else 0.0
    epochs_completed = start_epoch
    steps_exhausted = max_steps is not None and int(max_steps) <= 0

    with coordinator.backup_on_exception(lambda: _collect(epochs_completed)):
        for epoch in range(start_epoch, epochs):
            if steps_exhausted:
                break
            if coordinator.paused:
                path = coordinator.save(_collect(epoch), reason="pause")
                return _paused_result(
                    epoch, step, path, run_id=request.get("run_id")
                )
            model.train()
            epoch_loss = 0.0
            epoch_batches = 0
            paused_mid = False
            for images, binary_labels, group_labels in train_loader:
                if stop.is_set():
                    raise BackboneTrainingCancelled
                if max_steps is not None and step >= int(max_steps):
                    steps_exhausted = True
                    break
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                ):
                    features = model.backbone(images.to(device))
                    loss = _batch_loss(
                        features.float(), model.heads, binary_labels, group_labels, device
                    )
                if loss is None:
                    continue
                loss.backward()
                optimizer.step()
                if ema is not None:
                    ema.update(model)
                step += 1
                epoch_loss += float(loss.detach())
                epoch_batches += 1
                if coordinator.on_boundary(lambda: _collect(epoch), step) == "stop":
                    paused_mid = True
                    break
            if paused_mid:
                return _paused_result(
                    epoch, step, coordinator.last_backup_path, run_id=request.get("run_id")
                )
            if scheduler is not None:
                scheduler.step()

            # Validate on EMA weights (they generalise better on small datasets).
            eval_backbone, eval_heads = _eval_modules()
            validation_metrics = (
                _evaluate(
                    eval_backbone, eval_heads, val_loader, vocab, device,
                    amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                )
                if val_samples
                else {}
            )
            validation_score = (
                sum(validation_metrics.values()) / len(validation_metrics)
                if validation_metrics
                else 0.0
            )
            epochs_completed = epoch + 1

            if val_samples:
                if validation_score > best_score:
                    best_score = validation_score
                    best_epoch = epoch
                    best_metrics = dict(validation_metrics)
                    best_backbone_state = {
                        k: v.detach().to("cpu") for k, v in eval_backbone.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1

            progress(
                "training",
                f"Epoch {epoch + 1}/{epochs}",
                epoch=epoch + 1,
                epochs=epochs,
                steps=step,
                loss=epoch_loss / max(epoch_batches, 1),
                validation_score=validation_score,
                best_score=best_score if best_score >= 0 else None,
                best_epoch=best_epoch + 1 if best_score >= 0 else None,
            )

            if coordinator.enabled:
                coordinator.save(_collect(epoch + 1), reason="periodic")

            if patience > 0 and val_samples and patience_counter >= patience:
                progress(
                    "training",
                    f"Early stopping at epoch {epoch + 1} (patience {patience})",
                    early_stop=True,
                    epoch=epoch + 1,
                )
                break

    # Successful completion: backups are obsolete.
    coordinator.delete_backups()

    progress("validating", "Validating backbone candidate")
    if val_samples and best_backbone_state is not None:
        validation_metrics = best_metrics
        validation_score = best_score
    else:
        # No validation signal: serialise the final (EMA) backbone weights.
        eval_backbone, _ = _eval_modules()
        best_backbone_state = {
            k: v.detach().to("cpu") for k, v in eval_backbone.state_dict().items()
        }

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

    state = {key: value.detach().to("cpu") for key, value in best_backbone_state.items()}
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
        "epochs_completed": int(epochs_completed),
        "best_epoch": int(best_epoch + 1) if val_samples else int(epochs_completed),
    }


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
