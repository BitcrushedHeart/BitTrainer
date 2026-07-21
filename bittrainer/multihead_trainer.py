"""Multi-head ordinal trainer for size prediction.

Single ConvNeXt V2 backbone -> shared trunk -> {band head, size head}. The band head trains
only on the band (cup ignored); the size head trains on the full US size with volume-distance
soft labels so sister sizes are near-equivalent. A band-consistency term keeps the two heads
agreeing. Validation reports per-head and combined F1/QWK; model selection is on combined QWK.

Structurally mirrors :mod:`bittrainer.dual_branch_trainer` (warm-start, autobatch, compile,
early stopping, best-checkpoint promotion) — adapted to one input and two heads. The size
classes come in at their suite class-index order; all ordinal bookkeeping (band vocab,
size->band, size->volume, volume ranks) is derived here from the class names.

The lifecycle (dataset prep, warm-start, autobatch, epoch loop, backup/pause/resume,
best-checkpoint promotion) runs on the shared :class:`~bittrainer.generic.generic_trainer.GenericTrainer`
via :class:`~bittrainer.generic.tasks.multihead_task.MultiHeadTask` (Bitcrush ISSUE-0542 Step 7).
``run_multihead_training`` is a thin wrapper; the module-level helpers below stay here so the
task (and the existing test/monkeypatch seams) keep reaching them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from bittrainer.group_validation import compute_multihead_metrics
from bittrainer.multihead_ordinal import (
    build_band_vocab,
    size_to_band_index,
    volume_ranks,
)

_NONE_CLASS_NAME = "__none__"


@dataclass
class MultiHeadTrainConfig:
    group_folder: str
    size_classes: list[str]  # full size class names incl __none__, in class-index order
    classifier_mode: str = "multihead"
    max_epochs: int = 50
    patience: int = 3
    backbone_variant: str = "nano"
    band_loss_weight: float = 1.0
    consistency_weight: float = 0.5
    size_temperature: float = 2.0
    band_temperature: float = 1.5
    device: str = "cuda"
    dtype: str = "bfloat16"
    from_scratch: bool = False
    # Bitcrush Engine backbone spec (see bittrainer.backbone_init) — governs
    # where fresh-model backbone weights come from. None = timm pretrained.
    backbone_init: dict | None = None
    best_model_name: str = "best.pt"
    checkpoint_dir: str | None = None
    batch_size: int | None = None
    use_compile: bool = True
    channels_last: bool = True
    progress_callback: Callable[[dict], None] | None = None
    band_classes: list[str] = field(default_factory=list)
    # --- Backup / Pause / Resume (Bitcrush ISSUE-0405) ---
    # backup_dir=None => NO backups written and NO resume attempted (legacy).
    # backup_every_steps=0 => epoch-boundary backups only. resume_from points at
    # a backup dir/file. dataloader_workers replaces the hardcoded num_workers=6.
    backup_dir: str | None = None
    backup_every_steps: int = 500
    resume_from: str | None = None
    dataloader_workers: int = 6


def _get_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _collate(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([item[0] for item in batch])
    size_labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, size_labels


@dataclass
class _OrdinalMaps:
    band_vocab: list[str]
    size_to_band: torch.Tensor  # [n_sizes] long, -1 = ignore
    size_to_rank: torch.Tensor  # [n_sizes] long, -1 = none
    none_index: int
    num_ranks: int


def _build_maps(size_classes: list[str]) -> _OrdinalMaps:
    band_vocab = build_band_vocab(size_classes, none_name=_NONE_CLASS_NAME)
    s2b = size_to_band_index(size_classes, band_vocab, none_name=_NONE_CLASS_NAME, ignore_index=-1)
    ranks = volume_ranks(size_classes, none_name=_NONE_CLASS_NAME, none_index=-1)
    none_index = (
        size_classes.index(_NONE_CLASS_NAME) if _NONE_CLASS_NAME in size_classes else -1
    )
    num_ranks = (max(ranks) + 1) if any(r >= 0 for r in ranks) else 0
    return _OrdinalMaps(
        band_vocab=band_vocab,
        size_to_band=torch.tensor(s2b, dtype=torch.long),
        size_to_rank=torch.tensor(ranks, dtype=torch.long),
        none_index=none_index,
        num_ranks=num_ranks,
    )


def _train_one_epoch(
    model,
    dataloader,
    optimizer,
    criteria,
    maps: _OrdinalMaps,
    device,
    dtype,
    *,
    step_callback=None,
    stop_now_event=None,
    memory_format=None,
    boundary_hook=None,
) -> float:
    model.train()
    size_loss_fn, band_loss_fn, consistency_fn = criteria
    total_loss = 0.0
    num_batches = 0
    total_steps = len(dataloader)
    _last = time.monotonic()
    s2b = maps.size_to_band.to(device)

    optimizer.zero_grad()
    for images, size_labels in dataloader:
        if stop_now_event is not None and stop_now_event.is_set():
            break
        images = images.to(device=device, dtype=dtype)
        if memory_format is not None:
            images = images.contiguous(memory_format=memory_format)
        size_labels = size_labels.to(device)
        band_labels = s2b[size_labels]  # -1 for __none__ -> ignored by band loss

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            out = model(images)
            loss = (
                size_loss_fn(out["size"], size_labels)
                + band_loss_fn(out["band"], band_labels)
                + consistency_fn(out["band"], out["size"])
            )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1
        boundary_signal = boundary_hook(num_batches) if boundary_hook is not None else None
        if step_callback is not None:
            now = time.monotonic()
            if now - _last >= 0.25 or num_batches == total_steps:
                _last = now
                step_callback(num_batches, total_steps, total_loss / num_batches)
        if boundary_signal == "stop":
            break

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def _evaluate(model, dataloader, criteria, maps: _OrdinalMaps, num_bands, device, dtype, memory_format=None) -> dict:
    model.eval()
    size_loss_fn, band_loss_fn, consistency_fn = criteria
    s2b = maps.size_to_band.to(device)
    s2r = maps.size_to_rank.to(device)

    band_labels_all: list[int] = []
    band_preds_all: list[int] = []
    rank_labels_all: list[int] = []
    rank_preds_all: list[int] = []
    total_loss = 0.0
    num_batches = 0

    for images, size_labels in dataloader:
        images = images.to(device=device, dtype=dtype)
        if memory_format is not None:
            images = images.contiguous(memory_format=memory_format)
        size_labels = size_labels.to(device)
        band_labels = s2b[size_labels]

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            out = model(images)
            loss = (
                size_loss_fn(out["size"], size_labels)
                + band_loss_fn(out["band"], band_labels)
                + consistency_fn(out["band"], out["size"])
            )

        size_pred = out["size"].float().argmax(dim=1)
        band_pred = out["band"].float().argmax(dim=1)

        band_labels_all.extend(band_labels.cpu().tolist())
        band_preds_all.extend(band_pred.cpu().tolist())
        rank_labels_all.extend(s2r[size_labels].cpu().tolist())
        rank_preds_all.extend(s2r[size_pred].cpu().tolist())
        total_loss += loss.item()
        num_batches += 1

    metrics = compute_multihead_metrics(
        band_labels=band_labels_all,
        band_preds=band_preds_all,
        num_bands=num_bands,
        size_volume_labels=rank_labels_all,
        size_volume_preds=rank_preds_all,
        num_size_ranks=maps.num_ranks,
        none_index=-1,
    )
    metrics["val_loss"] = total_loss / max(num_batches, 1)
    return metrics


class _Scaled(torch.nn.Module):
    """Wrap a loss module to scale its output by a constant weight."""

    def __init__(self, loss: torch.nn.Module, weight: float):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *args) -> torch.Tensor:
        return self.weight * self.loss(*args)


def run_multihead_training(
    config: MultiHeadTrainConfig,
    *,
    progress_callback: Callable[[dict], None] | None = None,
    stop_event: object | None = None,
    stop_now_event: object | None = None,
    pause_event: object | None = None,
) -> dict:
    """Run the multi-head (band + size) training loop.

    Thin wrapper over :class:`~bittrainer.generic.generic_trainer.GenericTrainer`
    driving :class:`~bittrainer.generic.tasks.multihead_task.MultiHeadTask`.
    ``stop_event`` / ``stop_now_event`` / ``pause_event`` behave exactly as for the
    other trainers (graceful stop, mid-epoch interrupt, and resumable backup-pause;
    see the config's ``backup_dir`` / ``resume_from``). Resume is epoch-restart.
    """
    from bittrainer.generic.generic_trainer import GenericTrainer
    from bittrainer.generic.tasks.multihead_task import MultiHeadTask

    return GenericTrainer().run(
        MultiHeadTask(config),
        progress_callback=progress_callback,
        stop_event=stop_event,
        stop_now_event=stop_now_event,
        pause_event=pause_event,
    )
