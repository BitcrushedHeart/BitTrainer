"""Dual-branch (crop + context) trainer for skin-tone classification.

Two ConvNeXt V2 branches (a tight crop and its surrounding context) are fused into one
head and trained with label-smoothed cross-entropy; model selection is on macro-F1 and the
finalisation compares against the incumbent before promoting.

The lifecycle (dataset prep, warm-start, autobatch, compile, epoch loop, backup/pause/resume)
runs on the shared :class:`~bittrainer.generic.generic_trainer.GenericTrainer` via
:class:`~bittrainer.generic.tasks.dual_branch_task.DualBranchTask` (Bitcrush ISSUE-0542 Step 7).
The dual-branch finalisation stays BESPOKE inside the task (ISSUE-0490: it builds its own
result dict and compare-vs-incumbent promotion — deliberately not merged with group
calibration). ``run_dual_branch_training`` is a thin wrapper; the module-level helpers below
stay here so the task (and the existing test/monkeypatch seams) keep reaching them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bittrainer.backbone_init import apply_backbone_init, wants_timm_pretrained
from bittrainer.dual_branch_model import DualBranchConvNeXt
from bittrainer.group_validation import compute_multiclass_metrics


@dataclass
class DualBranchTrainConfig:
    group_folder: str
    context_folder: str
    num_classes: int
    class_names: list[str]
    classifier_mode: str = "dual_branch"
    max_epochs: int = 50
    patience: int = 3
    backbone_variant: str = "nano"
    label_smoothing: float = 0.1
    device: str = "cuda"
    dtype: str = "bfloat16"
    from_scratch: bool = False
    # Bitcrush Engine backbone spec (see bittrainer.backbone_init) — governs
    # where fresh-model backbone weights come from. None = timm pretrained.
    backbone_init: dict | None = None
    best_model_name: str = "best.pt"
    checkpoint_dir: str | None = None
    progress_callback: Callable[[dict], None] | None = None
    # Manual batch size override — skips the dual-branch VRAM probe when set
    batch_size: int | None = None
    # torch.compile for forward/backward; falls back to eager without triton.
    use_compile: bool = True
    # NHWC layout — ConvNeXt stem/downsample/dwconv save permute traffic.
    channels_last: bool = True
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


def _fresh_dual_branch_model(config: "DualBranchTrainConfig") -> DualBranchConvNeXt:
    model = DualBranchConvNeXt(
        backbone_variant=config.backbone_variant,
        num_classes=config.num_classes,
        pretrained=wants_timm_pretrained(config.backbone_init),
    )
    apply_backbone_init(model.crop_branch, config.backbone_init)
    apply_backbone_init(model.context_branch, config.backbone_init)
    return model


def _collate_dual(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    crops = torch.stack([item[0] for item in batch])
    contexts = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return crops, contexts, labels


def _train_one_epoch(
    model: DualBranchConvNeXt,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype,
    label_smoothing: float,
    *,
    step_callback: Callable[[int, int, float], None] | None = None,
    stop_now_event: object | None = None,
    memory_format: torch.memory_format | None = None,
    boundary_hook: Callable[[int], str | None] | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    total_steps = len(dataloader)
    _last_report = time.monotonic()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer.zero_grad()
    for crops, contexts, labels in dataloader:
        if stop_now_event is not None and stop_now_event.is_set():
            break
        crops = crops.to(device=device, dtype=dtype)
        contexts = contexts.to(device=device, dtype=dtype)
        if memory_format is not None:
            crops = crops.contiguous(memory_format=memory_format)
            contexts = contexts.contiguous(memory_format=memory_format)
        labels = labels.to(device=device)

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            logits = model(crops, contexts)
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1
        boundary_signal = boundary_hook(num_batches) if boundary_hook is not None else None

        if step_callback is not None:
            now = time.monotonic()
            if now - _last_report >= 0.25 or num_batches == total_steps:
                _last_report = now
                step_callback(num_batches, total_steps, total_loss / num_batches)
        if boundary_signal == "stop":
            break

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def _evaluate(
    model: DualBranchConvNeXt,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype,
    memory_format: torch.memory_format | None = None,
) -> dict:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0
    num_batches = 0
    criterion = nn.CrossEntropyLoss()

    for crops, contexts, labels in dataloader:
        crops = crops.to(device=device, dtype=dtype)
        contexts = contexts.to(device=device, dtype=dtype)
        if memory_format is not None:
            crops = crops.contiguous(memory_format=memory_format)
            contexts = contexts.contiguous(memory_format=memory_format)
        labels = labels.to(device=device)

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            logits = model(crops, contexts)
            loss = criterion(logits, labels)

        preds = logits.float().argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        total_loss += loss.item()
        num_batches += 1

    metrics = compute_multiclass_metrics(all_labels, all_preds, num_classes)
    metrics["val_loss"] = total_loss / max(num_batches, 1)
    return metrics


def run_dual_branch_training(
    config: DualBranchTrainConfig,
    *,
    progress_callback: Callable[[dict], None] | None = None,
    stop_event: object | None = None,
    stop_now_event: object | None = None,
    pause_event: object | None = None,
) -> dict:
    """Run the dual-branch (crop + context) training loop.

    Thin wrapper over :class:`~bittrainer.generic.generic_trainer.GenericTrainer`
    driving :class:`~bittrainer.generic.tasks.dual_branch_task.DualBranchTask`.
    ``stop_event`` / ``stop_now_event`` / ``pause_event`` behave exactly as for the
    other trainers (graceful stop, mid-epoch interrupt, and resumable backup-pause;
    see the config's ``backup_dir`` / ``resume_from``). Resume is epoch-restart and
    the finalisation/promotion stays bespoke inside the task (ISSUE-0490).
    """
    from bittrainer.generic.generic_trainer import GenericTrainer
    from bittrainer.generic.tasks.dual_branch_task import DualBranchTask

    return GenericTrainer().run(
        DualBranchTask(config),
        progress_callback=progress_callback,
        stop_event=stop_event,
        stop_now_event=stop_now_event,
        pause_event=pause_event,
    )
