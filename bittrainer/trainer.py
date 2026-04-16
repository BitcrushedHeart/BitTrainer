"""Training loop for ConvNeXt V2 binary classifiers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from adv_optm import Prodigy_adv
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from bittrainer.dataset import (
    ConceptDataset,
    _DimensionCache,
    build_bucket_batch_sampler,
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
    neg_pos_ratio: float = 1.0
    model_size: str = "nano"
    device: str = "cuda"
    dtype: str = "bfloat16"
    from_scratch: bool = False
    extra_positive_dirs: list[str] = field(default_factory=list)
    negative_dirs: list[str] = field(default_factory=list)
    label_smoothing: float = 0.1
    best_model_name: str = "best.pt"
    checkpoint_dir: str | None = None
    skin_normalise: bool = False
    face_model_path: str = ""
    progress_callback: Callable[[dict], None] | None = None


def _get_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _unwrap_state_dict(data: dict | object) -> dict:
    if isinstance(data, dict) and "state_dict" in data:
        return data["state_dict"]
    return data


def _collate_bucket_batch(batch):
    """Collate a bucket batch — all images should share the same dimensions.

    Includes a center-crop safety net for the rare case where dimensions
    differ (e.g. edge-case bucket assignment changes between runs).
    """
    from torchvision.transforms import functional as TF

    images = [item[0] for item in batch]
    target_h, target_w = images[0].shape[1], images[0].shape[2]
    for i in range(1, len(images)):
        if images[i].shape[1] != target_h or images[i].shape[2] != target_w:
            images[i] = TF.center_crop(images[i], [target_h, target_w])
    images = torch.stack(images)
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
    step_callback: Callable[[int, int, float], None] | None = None,
) -> float:
    from bittrainer.gpu_augment import apply_train_augment

    model.train()
    total_loss = 0.0
    num_batches = 0
    total_steps = len(dataloader)
    _last_report = time.monotonic()

    optimizer.zero_grad()
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        images = apply_train_augment(images, dtype=dtype)
        labels = labels.to(device)

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            logits = model(images)
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

        if step_callback is not None:
            now = time.monotonic()
            if now - _last_report >= 2.0 or num_batches == total_steps:
                _last_report = now
                step_callback(num_batches, total_steps, total_loss / num_batches)

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
    from bittrainer.gpu_augment import apply_val_transform

    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    num_batches = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        images = apply_val_transform(images, dtype=dtype)
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


def _rebalance_val_negatives(train_ds: ConceptDataset, val_ds: ConceptDataset) -> None:
    """Ensure the val set has enough negatives for meaningful evaluation.

    Target: at least as many negatives as positives in val.
    Cap: never take more than 40% of total negatives (training still needs them).
    """
    val_pos = len(val_ds._positive_paths)
    val_neg = len(val_ds._all_negative_paths)
    target = max(5, val_pos)

    if val_neg >= target:
        return

    needed = target - val_neg
    total_neg = len(train_ds._all_negative_paths) + val_neg
    max_donate = max(0, int(total_neg * 0.4) - val_neg)
    to_donate = min(needed, max_donate, len(train_ds._all_negative_paths))

    if to_donate <= 0:
        return

    donated = train_ds._all_negative_paths[:to_donate]
    train_ds._all_negative_paths = train_ds._all_negative_paths[to_donate:]
    val_ds._all_negative_paths = val_ds._all_negative_paths + donated

    # Ensure val_ds has bucket info for donated paths (they were precomputed by train_ds)
    val_ds._path_info.update(
        {str(p): train_ds._path_info[str(p)] for p in donated if str(p) in train_ds._path_info}
    )

    train_ds._build_samples()
    val_ds._build_samples()

    logger.info(
        "Rebalanced val set: donated %d negatives from train → val (val now %d neg / %d pos)",
        to_donate, len(val_ds._all_negative_paths), val_pos,
    )


def run_training(
    config: TrainConfig,
    *,
    progress_callback: Callable[[dict], None] | None = None,
    stop_event: object | None = None,
) -> dict:
    """Run the full training loop. Returns a result dict with metrics and checkpoint path."""
    cb = progress_callback or config.progress_callback or (lambda _: None)
    device = torch.device(config.device)
    dtype = _get_dtype(config.dtype)
    concept_folder = Path(config.concept_folder)
    checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else concept_folder / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Shared dimension cache — avoids reading image headers twice
    cache_dir = concept_folder / ".resize_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dim_cache = _DimensionCache(cache_dir / "dimensions.json")

    # Build datasets (no transform — GPU augmentation applied in training loop)
    train_ds = ConceptDataset(
        concept_folder, split="train",
        neg_pos_ratio=config.neg_pos_ratio,
        extra_positive_dirs=config.extra_positive_dirs,
        negative_dirs=config.negative_dirs,
        dim_cache=dim_cache,
    )
    val_ds = ConceptDataset(
        concept_folder, split="val",
        extra_positive_dirs=config.extra_positive_dirs,
        negative_dirs=config.negative_dirs,
        dim_cache=dim_cache,
    )

    _rebalance_val_negatives(train_ds, val_ds)

    num_positives = len(train_ds._positive_paths)

    # --- Face-aware cropping pre-computation ---
    face_bboxes: dict[str, list[int]] = {}
    if config.face_model_path:
        from bittrainer.face_crop import FaceBBoxCache, precompute_face_bboxes
        face_cache = FaceBBoxCache(cache_dir / "face_bboxes.json")
        all_image_paths = [s["path"] for s in train_ds.samples] + [s["path"] for s in val_ds.samples]

        def _face_progress(done: int, total: int) -> None:
            cb({"type": "face_detection", "processed": done, "total": total})

        precompute_face_bboxes(
            all_image_paths, face_cache, config.face_model_path,
            device=config.device,
            progress_fn=_face_progress,
        )
        for p in all_image_paths:
            bbox = face_cache.get(p)
            if bbox:
                face_bboxes[p] = bbox
        train_ds._face_bboxes = face_bboxes
        val_ds._face_bboxes = face_bboxes

    # --- Build tensor cache (after face bboxes so face-aware crops are cached) ---
    from bittrainer.tensor_cache import build_tensor_cache

    def _cache_progress(done: int, total: int) -> None:
        cb({"type": "tensor_cache", "cached": done, "total": total})

    tensor_cache_dir = concept_folder / ".tensor_cache"
    all_cache_samples = train_ds.samples + val_ds.samples
    build_tensor_cache(
        all_cache_samples, tensor_cache_dir, cache_dir,
        config.skin_normalise, face_bboxes,
        progress_fn=_cache_progress,
    )
    train_ds._use_tensor_cache = True
    train_ds._skin_normalise = config.skin_normalise
    train_ds._tensor_cache_dir = tensor_cache_dir
    val_ds._use_tensor_cache = True
    val_ds._skin_normalise = config.skin_normalise
    val_ds._tensor_cache_dir = tensor_cache_dir

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
                model_size=config.model_size,
            ).to(device)
            logger.info("Warm-starting from existing checkpoint: %s", existing_best)
        except Exception:
            logger.warning("Failed to load existing checkpoint, starting from pretrained", exc_info=True)
            model = create_model(model_size=config.model_size, pretrained=True, dtype=dtype).to(device)
    else:
        model = create_model(model_size=config.model_size, pretrained=True, dtype=dtype).to(device)
    use_gradual_unfreeze = num_positives < 50

    # Probe unfrozen = worst-case VRAM, then freeze for epoch 0
    from bittrainer.autobatch import determine_batch_size
    auto_result = determine_batch_size(model, bucket_counts, device, dtype=dtype)
    eff_bs = auto_result["batch_size"]
    cb({"type": "autobatch", **auto_result})
    freeze_backbone(model)

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

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path = None
    best_metrics: dict = {}

    # DataLoaders with num_workers=0 — tensor cache makes CPU workers unnecessary.
    # Images load as uint8 from disk cache; normalize + augment on GPU.
    val_sampler = build_bucket_batch_sampler(val_ds, batch_size=eff_bs)
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler, collate_fn=_collate_bucket_batch,
        num_workers=0, pin_memory=False,
    )

    def _rebuild_train_loader() -> DataLoader:
        sampler = build_bucket_batch_sampler(train_ds, batch_size=eff_bs)
        return DataLoader(
            train_ds, batch_sampler=sampler, collate_fn=_collate_bucket_batch,
            num_workers=0, pin_memory=False,
        )

    train_loader = _rebuild_train_loader()

    for epoch in range(config.max_epochs):
        if stop_event is not None and stop_event.is_set():
            logger.info("Graceful stop requested after epoch %d — running final comparison", epoch)
            cb({"type": "graceful_stop", "epoch": epoch, "max_epochs": config.max_epochs})
            break

        # Resample cross-concept negatives so the model sees different
        # negatives each epoch (no-op for legacy per-concept negatives)
        if epoch > 0:
            train_ds.resample_negatives()
            train_loader = _rebuild_train_loader()

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

        # Train
        def _on_step(step: int, total_steps: int, avg_loss: float) -> None:
            cb({
                "type": "step",
                "epoch": epoch + 1,
                "max_epochs": config.max_epochs,
                "step": step,
                "total_steps": total_steps,
                "train_loss": round(avg_loss, 4),
                "best_val_f1": best_val_f1 if best_val_f1 >= 0 else None,
                "best_epoch": best_epoch + 1 if best_val_f1 >= 0 else None,
            })

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, dtype,
            step_callback=_on_step,
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

            ckpt_path = checkpoint_dir / "candidate.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "num_classes": 2,
                "model_size": config.model_size,
            }, ckpt_path)
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
    existing_best = checkpoint_dir / config.best_model_name
    optimal_threshold = 0.5

    if best_checkpoint_path:
        if existing_best.exists():
            # Re-evaluate the old best.pt on the current validation set
            try:
                old_data = torch.load(
                    str(existing_best), map_location=device, weights_only=True,
                )
                old_sd = old_data["state_dict"] if isinstance(old_data, dict) and "state_dict" in old_data else old_data
                old_size = old_data.get("model_size", config.model_size) if isinstance(old_data, dict) else config.model_size
                old_model = create_model(model_size=old_size, pretrained=False, dtype=dtype).to(device)
                old_model.load_state_dict(old_sd)
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
                    model.load_state_dict(_unwrap_state_dict(
                        torch.load(best_checkpoint_path, map_location=device, weights_only=True),
                    ))
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
            model.load_state_dict(_unwrap_state_dict(
                torch.load(best_checkpoint_path, map_location=device, weights_only=True),
            ))
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
