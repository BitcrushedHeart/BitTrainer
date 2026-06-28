from __future__ import annotations

import logging
import time
from dataclasses import dataclass
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
    backbone_variant: str = "nano"
    label_smoothing: float = 0.1
    device: str = "cuda"
    dtype: str = "bfloat16"
    from_scratch: bool = False
    best_model_name: str = "best.pt"
    checkpoint_dir: str | None = None
    progress_callback: Callable[[dict], None] | None = None
    # Manual batch size override — skips the dual-branch VRAM probe when set
    batch_size: int | None = None
    # torch.compile for forward/backward; falls back to eager without triton.
    use_compile: bool = True
    # NHWC layout — ConvNeXt stem/downsample/dwconv save permute traffic.
    channels_last: bool = True
    backbone_init: dict | None = None


def _get_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


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

        if step_callback is not None:
            now = time.monotonic()
            if now - _last_report >= 0.25 or num_batches == total_steps:
                _last_report = now
                step_callback(num_batches, total_steps, total_loss / num_batches)

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
) -> dict:
    from bittrainer.runtime import configure_cuda_backend, maybe_compile, prewarm_compile

    cb = progress_callback or config.progress_callback or (lambda _: None)
    device = torch.device(config.device)
    dtype = _get_dtype(config.dtype)
    configure_cuda_backend()
    crops_folder = Path(config.group_folder)
    context_folder = Path(config.context_folder)
    checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else crops_folder / "checkpoints"
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

    # Build model — warm-start from existing checkpoint if available
    existing_best = checkpoint_dir / config.best_model_name
    if not config.from_scratch and existing_best.exists():
        try:
            model = DualBranchConvNeXt.from_checkpoint(str(existing_best), device=device)
            logger.info("Warm-starting from existing dual-branch checkpoint: %s", existing_best)
        except (RuntimeError, KeyError, FileNotFoundError):
            logger.warning("Failed to load existing checkpoint, starting from pretrained", exc_info=True)
            model = DualBranchConvNeXt(
                backbone_variant=config.backbone_variant,
                num_classes=config.num_classes,
                backbone_init=config.backbone_init,
            ).to(device)
    else:
        model = DualBranchConvNeXt(
            backbone_variant=config.backbone_variant,
            num_classes=config.num_classes,
            backbone_init=config.backbone_init,
        ).to(device)
    memory_format = torch.channels_last if config.channels_last else None
    if memory_format is not None:
        model = model.to(memory_format=memory_format)

    def _dual_inputs(b: int) -> tuple[torch.Tensor, torch.Tensor]:
        crops = torch.randn(b, 3, 512, 512, device=device, dtype=dtype)
        contexts = torch.randn(b, 3, 512, 512, device=device, dtype=dtype)
        if memory_format is not None:
            crops = crops.contiguous(memory_format=memory_format)
            contexts = contexts.contiguous(memory_format=memory_format)
        return crops, contexts

    # --- Auto batch sizing (shared profile-and-fit probe) ---
    if config.batch_size is not None and config.batch_size > 0:
        eff_bs = int(config.batch_size)
        cb({"type": "autobatch", "batch_size": eff_bs, "manual_override": True})
    else:
        from bittrainer.autobatch import determine_batch_size

        def _probe_progress(attempt: int, candidate: int, cap: int, status: str) -> None:
            cb({
                "type": "training_progress", "stage": "preparing",
                "status_text": f"Probing batch size (try {attempt}: {candidate}/{cap} — {status})",
            })

        cb({
            "type": "training_progress", "stage": "preparing",
            "status_text": "Probing optimal batch size",
        })
        auto_result = determine_batch_size(
            model, {(512, 512): total_samples}, device, dtype=dtype,
            use_ema=False, make_inputs=_dual_inputs,
            progress_callback=_probe_progress,
        )
        eff_bs = auto_result["batch_size"]
        cb({"type": "autobatch", **auto_result})

    optimizer = Prodigy_adv(
        model.parameters(), lr=1.0, d_coef=0.9,
        weight_decay=0.01, betas=(0.9, 0.999),
        kourkoutas_beta=True, k_warmup_steps=50,
        cautious_wd=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    # fwd_model shares parameters with the eager model — optimizer and
    # checkpoint saves keep operating on `model`.
    fwd_model, compiled = maybe_compile(model, enabled=config.use_compile, cb=cb)
    if compiled and not prewarm_compile(
        fwd_model, {(512, 512): total_samples}, eff_bs, device, dtype,
        memory_format=memory_format,
        make_inputs=lambda b, _bucket: _dual_inputs(b), cb=cb,
    ):
        fwd_model = model

    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path: str | None = None
    best_metrics: dict = {}

    train_loader = DataLoader(
        train_ds, batch_size=eff_bs, shuffle=True,
        collate_fn=_collate_dual, num_workers=6,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True, prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=eff_bs, shuffle=False,
        collate_fn=_collate_dual, num_workers=6,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True, prefetch_factor=4,
    )

    for epoch in range(config.max_epochs):
        if stop_now_event is not None and stop_now_event.is_set():
            logger.info("Stop-now requested before epoch %d — running final comparison", epoch)
            cb({"type": "stop_now", "epoch": epoch, "max_epochs": config.max_epochs})
            break
        if stop_event is not None and stop_event.is_set():
            logger.info("Graceful stop requested after epoch %d — running final comparison", epoch)
            cb({"type": "graceful_stop", "epoch": epoch, "max_epochs": config.max_epochs})
            break

        def _on_step(step: int, total_steps: int, avg_loss: float) -> None:
            cb({
                "type": "step",
                "epoch": epoch + 1,
                "max_epochs": config.max_epochs,
                "step": step,
                "total_steps": total_steps,
                "train_loss": round(avg_loss, 4),
                "best_val_macro_f1": best_val_macro_f1 if best_val_macro_f1 >= 0 else None,
                "best_epoch": best_epoch + 1 if best_val_macro_f1 >= 0 else None,
            })

        train_loss = _train_one_epoch(
            fwd_model, train_loader, optimizer, config.num_classes,
            device, dtype, config.label_smoothing,
            step_callback=_on_step,
            stop_now_event=stop_now_event,
            memory_format=memory_format,
        )
        if stop_now_event is not None and stop_now_event.is_set():
            cb({
                "type": "stop_now",
                "epoch": epoch + 1,
                "max_epochs": config.max_epochs,
                "status_text": f"Stop-now triggered mid-epoch {epoch + 1} — finishing up",
            })
        scheduler.step()

        val_metrics = _evaluate(
            fwd_model, val_loader, config.num_classes, device, dtype,
            memory_format=memory_format,
        )
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
