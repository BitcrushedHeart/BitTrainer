"""Training loop for ConvNeXt V2 binary classifiers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from adv_optm import Prodigy_adv
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from bittrainer.dataset import (
    ConceptDataset,
    build_bucket_batch_sampler,
    get_heavy_augment_transform,
    get_train_transform,
    get_val_transform,
)
from bittrainer.model import (
    create_model,
    freeze_backbone,
    get_stages,
    load_checkpoint,
    unfreeze_backbone,
    unfreeze_stage,
)
from bittrainer.validation import compute_metrics, find_optimal_threshold

logger = logging.getLogger(__name__)

_NUM_STAGES = 4  # ConvNeXt V2 has 4 stages


@dataclass
class TrainConfig:
    concept_folder: str
    max_epochs: int = 50
    patience: int = 3
    batch_size: int = 32
    neg_pos_ratio: float = 1.0
    model_size: str = "nano"
    device: str = "cuda"
    dtype: str = "bfloat16"
    from_scratch: bool = False
    extra_positive_dirs: list[str] = field(default_factory=list)
    progress_callback: Callable[[dict], None] | None = None


def compute_effective_batch_size(requested: int, total_samples: int) -> int:
    """Cap batch size at 10% of total samples, floor of 4."""
    cap = max(4, int(total_samples * 0.1))
    return max(4, min(requested, cap))


def _get_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _collate_bucket_batch(batch):
    """Custom collate that handles variable-size images within a bucket batch.

    All images in a bucket batch have the same target size, so standard
    stacking works. We just discard the bucket info after collation.
    """
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    *,
    grad_accum_steps: int = 1,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()
    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device, dtype=dtype)
        labels = labels.to(device)

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            logits = model(images)
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
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Evaluate on validation set. Returns loss, predictions, and labels."""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    num_batches = 0

    for images, labels in dataloader:
        images = images.to(device, dtype=dtype)
        labels = labels.to(device)

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits.float(), dim=1)[:, 1]  # P(positive)
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        total_loss += loss.item()
        num_batches += 1

    return {
        "val_loss": total_loss / max(num_batches, 1),
        "probs": all_probs,
        "labels": all_labels,
    }


def run_training(
    config: TrainConfig,
    *,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """Run the full training loop. Returns a result dict with metrics and checkpoint path."""
    cb = progress_callback or config.progress_callback or (lambda _: None)
    device = torch.device(config.device)
    dtype = _get_dtype(config.dtype)
    concept_folder = Path(config.concept_folder)
    checkpoint_dir = concept_folder / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets
    train_ds = ConceptDataset(
        concept_folder, split="train",
        neg_pos_ratio=config.neg_pos_ratio,
        transform=get_train_transform(),
        extra_positive_dirs=config.extra_positive_dirs,
    )
    # Apply heavy augmentation if negatives < positives
    if len(train_ds._all_negative_paths) < len(train_ds._positive_paths):
        train_ds.transform = get_heavy_augment_transform()

    val_ds = ConceptDataset(
        concept_folder, split="val",
        transform=get_val_transform(),
        extra_positive_dirs=config.extra_positive_dirs,
    )

    num_positives = len(train_ds._positive_paths)
    total_samples = len(train_ds)

    # Effective batch size
    eff_bs = compute_effective_batch_size(config.batch_size, total_samples)
    grad_accum = max(1, config.batch_size // eff_bs) if eff_bs < config.batch_size else 1

    # Create model — warm-start from best.pt unless from_scratch is set
    existing_best = checkpoint_dir / "best.pt"
    if not config.from_scratch and existing_best.exists():
        try:
            model = load_checkpoint(
                str(existing_best), device=str(device), dtype=dtype,
                model_size=config.model_size,
            ).to(device)
            logger.info("Warm-starting from existing checkpoint: %s", existing_best)
        except Exception:
            logger.warning("Failed to load existing checkpoint, starting from pretrained", exc_info=True)
            model = create_model(model_size=config.model_size, pretrained=True, dtype=dtype).to(device)
    else:
        model = create_model(model_size=config.model_size, pretrained=True, dtype=dtype).to(device)
    freeze_backbone(model)
    use_gradual_unfreeze = num_positives < 50

    # Optimiser: Prodigy_adv with kourkoutas beta and cautious weight decay
    optimizer = Prodigy_adv(
        model.parameters(),
        lr=1.0,
        d_coef=0.9,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        kourkoutas_beta=True,
        k_warmup_steps=50,
        cautious_mask=True,
    )

    # Scheduler: cosine annealing (stepped once per epoch)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path = None
    best_metrics: dict = {}

    for epoch in range(config.max_epochs):
        # Unfreezing logic
        if epoch == 1:
            if use_gradual_unfreeze:
                # Unfreeze last stage
                unfreeze_stage(model, _NUM_STAGES - 1)
            else:
                unfreeze_backbone(model)
                # Re-create optimizer with all params
                optimizer = Prodigy_adv(
                    model.parameters(), lr=1.0, d_coef=0.9,
                    weight_decay=0.01, betas=(0.9, 0.999),
                    kourkoutas_beta=True, k_warmup_steps=50,
                    cautious_mask=True,
                )
                remaining_epochs = config.max_epochs - 1  # 1 epoch already done
                scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs)
        elif epoch > 1 and use_gradual_unfreeze:
            stage_idx = _NUM_STAGES - epoch  # 3, 2, 1, 0
            if 0 <= stage_idx < _NUM_STAGES:
                unfreeze_stage(model, stage_idx)

        # Reshuffle negatives each epoch
        train_ds.reshuffle_negatives()

        # Build dataloaders with bucket sampling
        train_sampler = build_bucket_batch_sampler(train_ds, batch_size=eff_bs)
        train_loader = DataLoader(
            train_ds, batch_sampler=train_sampler, collate_fn=_collate_bucket_batch,
            num_workers=0, pin_memory=(device.type == "cuda"),
        )
        val_sampler = build_bucket_batch_sampler(val_ds, batch_size=eff_bs)
        val_loader = DataLoader(
            val_ds, batch_sampler=val_sampler, collate_fn=_collate_bucket_batch,
            num_workers=0, pin_memory=(device.type == "cuda"),
        )

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, dtype,
            grad_accum_steps=grad_accum,
        )
        scheduler.step()

        # Validate
        val_result = evaluate(model, val_loader, criterion, device, dtype)
        metrics = compute_metrics(val_result["labels"], val_result["probs"])
        metrics["val_loss"] = val_result["val_loss"]
        metrics["train_loss"] = train_loss

        # Check improvement
        val_f1 = metrics.get("f1", 0.0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_metrics = metrics.copy()

            # Save checkpoint as candidate (compared against existing best.pt later)
            ckpt_path = checkpoint_dir / "candidate.pt"
            torch.save(model.state_dict(), ckpt_path)
            best_checkpoint_path = str(ckpt_path)
        else:
            patience_counter += 1

        # Progress callback
        cb({
            "type": "epoch_complete",
            "epoch": epoch + 1,
            "max_epochs": config.max_epochs,
            "train_loss": train_loss,
            "val_loss": val_result["val_loss"],
            "val_f1": val_f1,
            "val_precision": metrics.get("precision", 0.0),
            "val_recall": metrics.get("recall", 0.0),
            "val_auprc": metrics.get("auprc", 0.0),
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch + 1,
        })

        # Early stopping
        if patience_counter >= config.patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, config.patience)
            break

    # Compare candidate checkpoint against existing best.pt on the CURRENT val set.
    # This ensures a fair apples-to-apples comparison even when the val set has
    # changed between training runs.
    existing_best = checkpoint_dir / "best.pt"
    optimal_threshold = 0.5

    if best_checkpoint_path:
        if existing_best.exists():
            # Re-evaluate the old best.pt on the current validation set
            try:
                old_model = create_model(model_size=config.model_size, pretrained=False, dtype=dtype).to(device)
                old_state = torch.load(
                    str(existing_best), map_location=device, weights_only=True,
                )
                old_model.load_state_dict(old_state)
                old_val_result = evaluate(
                    old_model, val_loader, criterion, device, dtype,
                )
                old_metrics = compute_metrics(
                    old_val_result["labels"], old_val_result["probs"],
                )
                old_f1 = old_metrics.get("f1", 0.0)
                del old_model  # free GPU memory

                if old_f1 > best_val_f1:
                    # Old model is better — keep existing best.pt, discard candidate
                    logger.info(
                        "Old checkpoint F1 %.4f > new F1 %.4f — keeping old",
                        old_f1, best_val_f1,
                    )
                    candidate = Path(best_checkpoint_path)
                    if candidate.exists():
                        candidate.unlink()
                    best_checkpoint_path = str(existing_best)
                    best_val_f1 = old_f1
                    best_metrics = old_metrics
                    optimal_threshold = find_optimal_threshold(
                        old_val_result["labels"], old_val_result["probs"],
                    )
                else:
                    # New model wins — promote candidate to best.pt
                    logger.info(
                        "New checkpoint F1 %.4f >= old F1 %.4f — promoting new",
                        best_val_f1, old_f1,
                    )
                    candidate = Path(best_checkpoint_path)
                    candidate.replace(existing_best)
                    best_checkpoint_path = str(existing_best)
                    model.load_state_dict(
                        torch.load(
                            best_checkpoint_path,
                            map_location=device,
                            weights_only=True,
                        ),
                    )
                    val_result = evaluate(
                        model, val_loader, criterion, device, dtype,
                    )
                    optimal_threshold = find_optimal_threshold(
                        val_result["labels"], val_result["probs"],
                    )
            except Exception:
                # Old checkpoint incompatible (e.g. architecture change) — new wins
                logger.warning(
                    "Failed to re-evaluate old checkpoint, keeping new",
                    exc_info=True,
                )
                candidate = Path(best_checkpoint_path)
                candidate.replace(existing_best)
                best_checkpoint_path = str(existing_best)
                model.load_state_dict(
                    torch.load(
                        best_checkpoint_path,
                        map_location=device,
                        weights_only=True,
                    ),
                )
                val_result = evaluate(
                    model, val_loader, criterion, device, dtype,
                )
                optimal_threshold = find_optimal_threshold(
                    val_result["labels"], val_result["probs"],
                )
        else:
            # No existing best.pt — promote candidate directly
            candidate = Path(best_checkpoint_path)
            candidate.replace(existing_best)
            best_checkpoint_path = str(existing_best)
            model.load_state_dict(
                torch.load(
                    best_checkpoint_path,
                    map_location=device,
                    weights_only=True,
                ),
            )
            val_result = evaluate(model, val_loader, criterion, device, dtype)
            optimal_threshold = find_optimal_threshold(
                val_result["labels"], val_result["probs"],
            )

    return {
        "epochs_completed": epoch + 1,
        "best_epoch": best_epoch + 1,
        "best_val_f1": best_val_f1,
        "final_val_f1": best_metrics.get("f1"),
        "final_val_precision": best_metrics.get("precision"),
        "final_val_recall": best_metrics.get("recall"),
        "final_val_auprc": best_metrics.get("auprc"),
        "final_val_loss": best_metrics.get("val_loss"),
        "optimal_threshold": optimal_threshold,
        "checkpoint_path": best_checkpoint_path,
        "positive_count": num_positives,
        "negative_count": len(train_ds._all_negative_paths),
        "confusion_matrix": best_metrics.get("confusion_matrix"),
    }
