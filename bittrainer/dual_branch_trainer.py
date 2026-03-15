from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from adv_optm import Prodigy_adv
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from bittrainer.dual_branch_model import DualBranchConvNeXt
from bittrainer.dual_crop_dataset import DualCropDataset
from bittrainer.group_dataset import get_train_transform, get_val_transform
from bittrainer.group_validation import compute_multiclass_metrics

logger = logging.getLogger(__name__)


@dataclass
class DualBranchTrainConfig:
    group_folder: str
    context_folder: str
    num_classes: int
    class_names: list[str]
    classifier_mode: str = "dual_branch"
    max_epochs: int = 50
    patience: int = 3
    batch_size: int = 32
    backbone_variant: str = "nano"
    label_smoothing: float = 0.1
    device: str = "cuda"
    dtype: str = "bfloat16"
    from_scratch: bool = False
    progress_callback: Callable[[dict], None] | None = None


def _get_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _collate_dual(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    crops = torch.stack([item[0] for item in batch])
    contexts = torch.stack([item[1] for item in batch])
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return crops, contexts, labels


def _compute_effective_batch_size(requested: int, total_samples: int) -> int:
    cap = max(4, int(total_samples * 0.1))
    return max(4, min(requested, cap))


def _train_one_epoch(
    model: DualBranchConvNeXt,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype,
    label_smoothing: float,
    grad_accum_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer.zero_grad()
    for step, (crops, contexts, labels) in enumerate(dataloader):
        crops = crops.to(device=device, dtype=dtype)
        contexts = contexts.to(device=device, dtype=dtype)
        labels = labels.to(device=device)

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            logits = model(crops, contexts)
            loss = criterion(logits, labels)
            loss = loss / grad_accum_steps

        loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def _evaluate(
    model: DualBranchConvNeXt,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype,
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
) -> dict:
    cb = progress_callback or config.progress_callback or (lambda _: None)
    device = torch.device(config.device)
    dtype = _get_dtype(config.dtype)
    crops_folder = Path(config.group_folder)
    context_folder = Path(config.context_folder)
    checkpoint_dir = crops_folder / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_ds = DualCropDataset(
        crops_folder, context_folder, config.class_names,
        split="train", crop_transform=train_transform, context_transform=train_transform,
    )
    val_ds = DualCropDataset(
        crops_folder, context_folder, config.class_names,
        split="val", crop_transform=val_transform, context_transform=val_transform,
    )

    total_samples = len(train_ds)
    if total_samples == 0:
        raise RuntimeError("No training image pairs found")

    eff_bs = _compute_effective_batch_size(config.batch_size, total_samples)
    grad_accum = max(1, config.batch_size // eff_bs) if eff_bs < config.batch_size else 1

    # Build model — warm-start from existing checkpoint if available
    existing_best = checkpoint_dir / "best.pt"
    if not config.from_scratch and existing_best.exists():
        try:
            model = DualBranchConvNeXt.from_checkpoint(str(existing_best), device=device)
            logger.info("Warm-starting from existing dual-branch checkpoint: %s", existing_best)
        except (RuntimeError, KeyError, FileNotFoundError):
            logger.warning("Failed to load existing checkpoint, starting from pretrained", exc_info=True)
            model = DualBranchConvNeXt(
                backbone_variant=config.backbone_variant,
                num_classes=config.num_classes,
            ).to(device)
    else:
        model = DualBranchConvNeXt(
            backbone_variant=config.backbone_variant,
            num_classes=config.num_classes,
        ).to(device)

    optimizer = Prodigy_adv(
        model.parameters(), lr=1.0, d_coef=0.9,
        weight_decay=0.01, betas=(0.9, 0.999),
        kourkoutas_beta=True, k_warmup_steps=50,
        cautious_mask=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path: str | None = None
    best_metrics: dict = {}

    train_loader = DataLoader(
        train_ds, batch_size=eff_bs, shuffle=True,
        collate_fn=_collate_dual, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=eff_bs, shuffle=False,
        collate_fn=_collate_dual, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    for epoch in range(config.max_epochs):
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, config.num_classes,
            device, dtype, config.label_smoothing,
            grad_accum_steps=grad_accum,
        )
        scheduler.step()

        val_metrics = _evaluate(model, val_loader, config.num_classes, device, dtype)
        val_metrics["train_loss"] = train_loss

        val_macro_f1 = val_metrics["macro_f1"]
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0
            best_metrics = val_metrics.copy()

            ckpt_path = checkpoint_dir / "candidate.pt"
            model.save_checkpoint(str(ckpt_path), metadata={
                "epoch": epoch + 1,
                "val_macro_f1": val_macro_f1,
            })
            best_checkpoint_path = str(ckpt_path)
        else:
            patience_counter += 1

        cb({
            "type": "epoch_complete",
            "epoch": epoch + 1,
            "max_epochs": config.max_epochs,
            "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "val_macro_f1": val_macro_f1,
            "per_class_f1": val_metrics.get("per_class_f1", {}),
            "best_val_macro_f1": best_val_macro_f1,
            "best_epoch": best_epoch + 1,
        })

        if patience_counter >= config.patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, config.patience)
            break

    # Promote best checkpoint
    if best_checkpoint_path:
        if existing_best.exists() and best_checkpoint_path != str(existing_best):
            try:
                old_model = DualBranchConvNeXt.from_checkpoint(str(existing_best), device=device)
                old_metrics = _evaluate(old_model, val_loader, config.num_classes, device, dtype)
                old_f1 = old_metrics["macro_f1"]
                del old_model

                if old_f1 > best_val_macro_f1:
                    logger.info("Old checkpoint F1 %.4f > new F1 %.4f — keeping old", old_f1, best_val_macro_f1)
                    Path(best_checkpoint_path).unlink(missing_ok=True)
                    best_checkpoint_path = str(existing_best)
                    best_val_macro_f1 = old_f1
                    best_metrics = old_metrics
                else:
                    logger.info("New checkpoint F1 %.4f >= old F1 %.4f — promoting new", best_val_macro_f1, old_f1)
                    Path(best_checkpoint_path).replace(existing_best)
                    best_checkpoint_path = str(existing_best)
            except (RuntimeError, KeyError, FileNotFoundError):
                logger.warning("Failed to re-evaluate old checkpoint, keeping new", exc_info=True)
                Path(best_checkpoint_path).replace(existing_best)
                best_checkpoint_path = str(existing_best)
        else:
            Path(best_checkpoint_path).replace(existing_best)
            best_checkpoint_path = str(existing_best)

    # Count samples per class
    class_counts: dict[int, int] = {}
    for class_name in config.class_names:
        idx = config.class_names.index(class_name)
        class_dir = crops_folder / class_name / "train"
        if class_dir.exists():
            class_counts[idx] = sum(1 for f in class_dir.iterdir() if f.is_file())

    return {
        "epochs_completed": epoch + 1,
        "best_epoch": best_epoch + 1,
        "best_val_macro_f1": best_val_macro_f1,
        "final_val_macro_f1": best_metrics.get("macro_f1"),
        "final_val_loss": best_metrics.get("val_loss"),
        "per_class_f1": best_metrics.get("per_class_f1", {}),
        "confusion_matrix": best_metrics.get("confusion_matrix", []),
        "balanced_accuracy": best_metrics.get("balanced_accuracy"),
        "checkpoint_path": best_checkpoint_path,
        "class_counts": class_counts,
        "total_images": total_samples,
    }
