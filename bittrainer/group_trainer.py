"""Training loop for ConvNeXt V2 multi-class group classifiers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import numpy as np
from adv_optm import Prodigy_adv
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from bittrainer.group_dataset import (
    GroupDataset,
    build_group_bucket_sampler,
    get_skin_normalised_train_transform,
    get_skin_normalised_val_transform,
    get_train_transform,
    get_val_transform,
)
from bittrainer.group_validation import compute_multiclass_metrics, compute_multilabel_metrics, compute_ordinal_metrics
from bittrainer.model import (
    create_model,
    freeze_backbone,
    get_stages,
    load_checkpoint,
    unfreeze_backbone,
    unfreeze_stage,
)

logger = logging.getLogger(__name__)

_NUM_STAGES = 4


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
    multi_label: bool = False
    soft_aliases: dict = field(default_factory=dict)
    device: str = "cuda"
    dtype: str = "bfloat16"
    from_scratch: bool = False
    best_model_name: str = "best.pt"
    checkpoint_dir: str | None = None
    skin_normalise: bool = False
    face_model_path: str = ""
    progress_callback: Callable[[dict], None] | None = None


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


def _build_gaussian_kernel(num_classes: int, sigma: float = 1.0) -> torch.Tensor:
    """Build a Gaussian smoothing kernel for ordinal classes.

    kernel[i, j] = exp(-(i-j)^2 / (2*sigma^2)), then normalised per row.
    """
    indices = torch.arange(num_classes, dtype=torch.float32)
    diffs = indices.unsqueeze(0) - indices.unsqueeze(1)
    kernel = torch.exp(-diffs ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum(dim=1, keepdim=True)
    return kernel


def _build_soft_targets(
    labels: torch.Tensor,
    num_classes: int,
    *,
    ordinal: bool = False,
    label_smoothing: float = 0.0,
    soft_aliases: dict | None = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Convert integer labels to soft target vectors.

    1. Start with one-hot
    2. Apply ordinal Gaussian smoothing (if ordinal)
    3. Apply uniform label smoothing (if not ordinal but smoothing > 0)
    4. Apply soft aliases
    """
    batch_size = labels.shape[0]
    targets = torch.zeros(batch_size, num_classes, device=device)
    targets.scatter_(1, labels.unsqueeze(1), 1.0)

    if ordinal and num_classes > 2:
        kernel = _build_gaussian_kernel(num_classes).to(device)
        targets = targets @ kernel
    elif label_smoothing > 0:
        targets = targets * (1.0 - label_smoothing) + label_smoothing / num_classes

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


def _soft_ce_loss(log_probs: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss against soft targets."""
    return -(soft_targets * log_probs).sum(dim=1).mean()


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
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    bce_criterion = nn.BCEWithLogitsLoss() if config.multi_label else None

    for images, labels in dataloader:
        images = images.to(device, dtype=dtype)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            logits = model(images)

            if config.multi_label:
                loss = bce_criterion(logits.float(), labels.float())
            elif use_soft_targets:
                soft = _build_soft_targets(
                    labels, config.num_classes,
                    ordinal=config.ordinal,
                    label_smoothing=config.label_smoothing,
                    soft_aliases=config.soft_aliases or None,
                    device=device,
                )
                log_probs = torch.log_softmax(logits.float(), dim=1)
                loss = _soft_ce_loss(log_probs, soft)
            else:
                criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
                loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

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
) -> dict:
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    if multi_label:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    for images, labels in dataloader:
        images = images.to(device, dtype=dtype)
        labels = labels.to(device)

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            logits = model(images)
            if multi_label:
                loss = criterion(logits.float(), labels.float())
            else:
                loss = criterion(logits, labels)

        if multi_label:
            preds = (torch.sigmoid(logits.float()) > 0.5).int()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().int().numpy())
        else:
            preds = logits.float().argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        total_loss += loss.item()
        num_batches += 1

    if multi_label:
        all_labels_arr = np.concatenate(all_labels, axis=0)
        all_preds_arr = np.concatenate(all_preds, axis=0)
        metrics = compute_multilabel_metrics(all_labels_arr, all_preds_arr, num_classes)
    else:
        metrics = compute_multiclass_metrics(all_labels, all_preds, num_classes)
        if ordinal:
            metrics.update(compute_ordinal_metrics(all_labels, all_preds, num_classes))

    metrics["val_loss"] = total_loss / max(num_batches, 1)
    return metrics


def run_group_training(
    config: GroupTrainConfig,
    *,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """Run the full multi-class training loop."""
    cb = progress_callback or config.progress_callback or (lambda _: None)
    device = torch.device(config.device)
    dtype = _get_dtype(config.dtype)
    group_folder = Path(config.group_folder)
    checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else group_folder / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    use_soft = config.ordinal or bool(config.soft_aliases)

    # Build datasets
    train_transform = get_skin_normalised_train_transform() if config.skin_normalise else get_train_transform()
    train_ds = GroupDataset(
        group_folder, config.class_names, split="train",
        transform=train_transform,
        multi_label=config.multi_label,
    )
    val_transform = get_skin_normalised_val_transform() if config.skin_normalise else get_val_transform()
    val_ds = GroupDataset(
        group_folder, config.class_names, split="val",
        transform=val_transform,
        multi_label=config.multi_label,
    )

    total_samples = len(train_ds)
    if total_samples == 0:
        raise RuntimeError("No training images found")

    # --- Face-aware cropping pre-computation ---
    face_bboxes: dict[str, list[int]] = {}
    if config.face_model_path:
        from bittrainer.face_crop import FaceBBoxCache, precompute_face_bboxes
        face_cache = FaceBBoxCache(group_folder / ".resize_cache" / "face_bboxes.json")
        all_image_paths = [s["path"] for s in train_ds.samples] + [s["path"] for s in val_ds.samples]
        precompute_face_bboxes(
            all_image_paths, face_cache, config.face_model_path,
            device=config.device,
        )
        for p in all_image_paths:
            bbox = face_cache.get(p)
            if bbox:
                face_bboxes[p] = bbox
        train_ds._face_bboxes = face_bboxes
        val_ds._face_bboxes = face_bboxes

    # --- Count samples per bucket ---
    bucket_counts: dict[tuple[int, int], int] = {}
    for s in train_ds.samples:
        b = s["bucket"]
        bucket_counts[b] = bucket_counts.get(b, 0) + 1

    # Create model — warm-start from best.pt unless from_scratch is set
    existing_best = checkpoint_dir / config.best_model_name
    if not config.from_scratch and existing_best.exists():
        try:
            model = load_checkpoint(
                str(existing_best), device=str(device), dtype=dtype,
                model_size=config.backbone_variant, num_classes=config.num_classes,
            ).to(device)
            logger.info("Warm-starting from existing checkpoint: %s", existing_best)
        except Exception:
            logger.warning("Failed to load existing checkpoint, starting from pretrained", exc_info=True)
            model = create_model(
                model_size=config.backbone_variant, pretrained=True,
                dtype=dtype, num_classes=config.num_classes,
            ).to(device)
    else:
        model = create_model(
            model_size=config.backbone_variant, pretrained=True,
            dtype=dtype, num_classes=config.num_classes,
        ).to(device)
    freeze_backbone(model)

    # --- Auto batch sizing ---
    from bittrainer.autobatch import determine_batch_size
    auto_result = determine_batch_size(model, bucket_counts, device, dtype=dtype)
    eff_bs = auto_result["batch_size"]
    cb({"type": "autobatch", **auto_result})

    class_counts = train_ds.get_class_counts()
    total_raw = sum(class_counts.values())
    use_gradual_unfreeze = total_raw < 50

    # Optimizer
    optimizer = Prodigy_adv(
        model.parameters(), lr=1.0, d_coef=0.9,
        weight_decay=0.01, betas=(0.9, 0.999),
        kourkoutas_beta=True, k_warmup_steps=50,
        cautious_mask=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    best_val_macro_f1 = -1.0
    best_val_qwk = -1.0
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path = None
    best_metrics: dict = {}

    for epoch in range(config.max_epochs):
        # Unfreezing
        if epoch == 1:
            if use_gradual_unfreeze:
                unfreeze_stage(model, _NUM_STAGES - 1)
            else:
                unfreeze_backbone(model)
                optimizer = Prodigy_adv(
                    model.parameters(), lr=1.0, d_coef=0.9,
                    weight_decay=0.01, betas=(0.9, 0.999),
                    kourkoutas_beta=True, k_warmup_steps=50,
                    cautious_mask=True,
                )
                remaining = config.max_epochs - 1
                scheduler = CosineAnnealingLR(optimizer, T_max=remaining)
        elif epoch > 1 and use_gradual_unfreeze:
            stage_idx = _NUM_STAGES - epoch
            if 0 <= stage_idx < _NUM_STAGES:
                unfreeze_stage(model, stage_idx)

        # Reshuffle for class-balanced sampling
        train_ds.reshuffle()

        # Build dataloaders
        collate_fn = _collate_multilabel_batch if config.multi_label else _collate_bucket_batch
        train_sampler = build_group_bucket_sampler(train_ds, batch_size=eff_bs)
        train_loader = DataLoader(
            train_ds, batch_sampler=train_sampler, collate_fn=collate_fn,
            num_workers=0, pin_memory=(device.type == "cuda"),
        )
        val_sampler = build_group_bucket_sampler(val_ds, batch_size=eff_bs)
        val_loader = DataLoader(
            val_ds, batch_sampler=val_sampler, collate_fn=collate_fn,
            num_workers=0, pin_memory=(device.type == "cuda"),
        )

        # Train
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, config, device, dtype,
            use_soft_targets=use_soft,
        )
        scheduler.step()

        # Validate
        val_metrics = _evaluate(
            model, val_loader, config.num_classes, device, dtype,
            multi_label=config.multi_label,
            ordinal=config.ordinal,
        )
        val_metrics["train_loss"] = train_loss

        val_macro_f1 = val_metrics["macro_f1"]
        val_qwk = val_metrics.get("qwk", 0.0)

        improved = (val_qwk > best_val_qwk) if config.ordinal else (val_macro_f1 > best_val_macro_f1)
        if improved:
            best_val_macro_f1 = val_macro_f1
            best_val_qwk = val_qwk
            best_epoch = epoch
            patience_counter = 0
            best_metrics = val_metrics.copy()

            ckpt_path = checkpoint_dir / "candidate.pt"
            ckpt_meta = {
                "state_dict": model.state_dict(),
                "num_classes": config.num_classes,
                "model_size": config.backbone_variant,
            }
            if config.multi_label:
                ckpt_meta["multi_label"] = True
            torch.save(ckpt_meta, ckpt_path)
            best_checkpoint_path = str(ckpt_path)
        else:
            patience_counter += 1

        epoch_msg = {
            "type": "epoch_complete",
            "epoch": epoch + 1,
            "max_epochs": config.max_epochs,
            "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "val_macro_f1": val_macro_f1,
            "per_class_f1": val_metrics.get("per_class_f1", {}),
            "best_val_macro_f1": best_val_macro_f1,
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

    # Checkpoint comparison — same pattern as binary trainer
    existing_best = checkpoint_dir / config.best_model_name

    if best_checkpoint_path:
        if existing_best.exists():
            try:
                old_model = create_model(
                    model_size=config.backbone_variant, pretrained=False,
                    dtype=dtype, num_classes=config.num_classes,
                ).to(device)
                old_data = torch.load(str(existing_best), map_location=device, weights_only=True)
                old_state = old_data["state_dict"] if isinstance(old_data, dict) and "state_dict" in old_data else old_data
                old_model.load_state_dict(old_state)

                old_metrics = _evaluate(
                    old_model, val_loader, config.num_classes, device, dtype,
                    multi_label=config.multi_label,
                    ordinal=config.ordinal,
                )
                old_f1 = old_metrics["macro_f1"]
                old_qwk = old_metrics.get("qwk", 0.0)
                del old_model

                old_wins = (old_qwk > best_val_qwk) if config.ordinal else (old_f1 > best_val_macro_f1)
                if old_wins:
                    logger.info("Old checkpoint F1 %.4f > new F1 %.4f — keeping old", old_f1, best_val_macro_f1)
                    Path(best_checkpoint_path).unlink(missing_ok=True)
                    best_checkpoint_path = str(existing_best)
                    best_val_macro_f1 = old_f1
                    best_val_qwk = old_qwk
                    best_metrics = old_metrics
                else:
                    logger.info("New checkpoint F1 %.4f >= old F1 %.4f — promoting new", best_val_macro_f1, old_f1)
                    Path(best_checkpoint_path).replace(existing_best)
                    best_checkpoint_path = str(existing_best)
            except Exception:
                logger.warning("Failed to re-evaluate old checkpoint, keeping new", exc_info=True)
                Path(best_checkpoint_path).replace(existing_best)
                best_checkpoint_path = str(existing_best)
        else:
            Path(best_checkpoint_path).replace(existing_best)
            best_checkpoint_path = str(existing_best)

    result = {
        "epochs_completed": epoch + 1,
        "best_epoch": best_epoch + 1,
        "best_val_macro_f1": best_val_macro_f1,
        "final_val_macro_f1": best_metrics.get("macro_f1"),
        "final_val_loss": best_metrics.get("val_loss"),
        "per_class_f1": best_metrics.get("per_class_f1", {}),
        "checkpoint_path": best_checkpoint_path,
        "class_counts": train_ds.get_class_counts(),
        "total_images": total_raw,
    }

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
