"""Training loop for ConvNeXt V2 multi-class group classifiers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import partial
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
_NONE_CLASS_NAME = "__none__"


def _resolve_none_index(class_names: list[str]) -> int:
    """Return the position of the ``__none__`` class, or -1 if absent.

    ``__none__`` is a valid output class the model learns to predict, but it
    must be excluded from any code path that assumes class indices are
    positions on an ordinal scale (Gaussian soft-target smoothing, ordinal
    validation metrics, etc.).
    """
    try:
        return class_names.index(_NONE_CLASS_NAME)
    except ValueError:
        return -1


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
    cache_dir: str | None = None
    use_cache: bool = True
    sourceless: bool = False
    group_name: str = ""
    modeltype: str = "convnext_v2"
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


def _build_gaussian_kernel(
    num_classes: int,
    sigma: float = 1.0,
    *,
    none_index: int = -1,
) -> torch.Tensor:
    """Build a Gaussian smoothing kernel for ordinal classes.

    kernel[i, j] = exp(-(i-j)^2 / (2*sigma^2)), then normalised per row.

    When ``none_index >= 0`` the corresponding class (``__none__``) is treated
    as a separate semantic category, not a position on the ordinal scale.
    Its row and column are zeroed and the diagonal entry is set to 1, so no
    probability bleeds between ``__none__`` and its numeric neighbours during
    soft-target smoothing — without this, the model learns that ``__none__``
    is adjacent to the lowest ordinal class (e.g. ``__none__`` ↔ "Augmented
    Breasts" or ``__none__`` ↔ "0-year-old"), which corrupts predictions on
    visually-empty inputs.
    """
    indices = torch.arange(num_classes, dtype=torch.float32)
    diffs = indices.unsqueeze(0) - indices.unsqueeze(1)
    kernel = torch.exp(-diffs ** 2 / (2 * sigma ** 2))
    if 0 <= none_index < num_classes:
        kernel[none_index, :] = 0.0
        kernel[:, none_index] = 0.0
        kernel[none_index, none_index] = 1.0
    kernel = kernel / kernel.sum(dim=1, keepdim=True)
    return kernel


def _build_soft_targets(
    labels: torch.Tensor,
    num_classes: int,
    *,
    ordinal: bool = False,
    label_smoothing: float = 0.0,
    soft_aliases: dict | None = None,
    none_index: int = -1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Convert integer labels to soft target vectors.

    1. Start with one-hot
    2. Apply ordinal Gaussian smoothing (if ordinal), excluding ``none_index``
    3. Apply uniform label smoothing (if not ordinal but smoothing > 0)
    4. Apply soft aliases
    """
    batch_size = labels.shape[0]
    targets = torch.zeros(batch_size, num_classes, device=device)
    targets.scatter_(1, labels.unsqueeze(1), 1.0)

    if ordinal and num_classes > 2:
        kernel = _build_gaussian_kernel(num_classes, none_index=none_index).to(device)
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
    step_callback: Callable[[int, int, float], None] | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    total_steps = len(dataloader)
    _last_report = time.monotonic()

    bce_criterion = nn.BCEWithLogitsLoss() if config.multi_label else None

    from bittrainer.gpu_augment import apply_train_augment

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        images = apply_train_augment(images, dtype=dtype)
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
                    none_index=_resolve_none_index(config.class_names),
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

        if step_callback is not None:
            now = time.monotonic()
            if now - _last_report >= 2.0 or num_batches == total_steps:
                _last_report = now
                step_callback(num_batches, total_steps, total_loss / num_batches)

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
    none_index: int = -1,
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

    from bittrainer.gpu_augment import apply_val_transform

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        images = apply_val_transform(images, dtype=dtype)
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
            metrics.update(compute_ordinal_metrics(
                all_labels, all_preds, num_classes, none_index=none_index,
            ))

    metrics["val_loss"] = total_loss / max(num_batches, 1)
    return metrics


def run_group_training(
    config: GroupTrainConfig,
    *,
    progress_callback: Callable[[dict], None] | None = None,
    stop_event: object | None = None,
) -> dict:
    """Run the full multi-class training loop."""
    from bittrainer.smart_cache import _noop_callback, _never_stop
    from bittrainer.trainer import _stop_event_is_set
    cb = progress_callback or config.progress_callback or _noop_callback
    device = torch.device(config.device)
    dtype = _get_dtype(config.dtype)
    group_folder = Path(config.group_folder)
    checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else group_folder / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    group_name = config.group_name or group_folder.name

    use_soft = config.ordinal or bool(config.soft_aliases)

    # --- SmartCache setup ---
    smart_cache = None
    if config.use_cache:
        from bittrainer.smart_cache import SmartCache, face_model_signature
        cache_root = Path(config.cache_dir) if config.cache_dir else (group_folder / ".smart_cache")
        smart_cache = SmartCache(
            cache_root,
            modeltype=config.modeltype,
            progress_callback=cb,
            stop_check=partial(_stop_event_is_set, stop_event),
            face_model_sig=face_model_signature(config.face_model_path or None),
        )

    if config.sourceless:
        if smart_cache is None:
            raise RuntimeError("sourceless=True requires use_cache=True and a cache_dir")
        cb({
            "type": "training_progress", "stage": "validating",
            "status_text": "Loading sourceless samples from cache",
            "step": 0, "total_steps": 0,
        })
        train_ds = GroupDataset(
            group_folder, config.class_names, split="train",
            multi_label=config.multi_label,
            cache=smart_cache, sourceless=True, group_name=group_name,
        )
        val_ds = GroupDataset(
            group_folder, config.class_names, split="val",
            multi_label=config.multi_label,
            cache=smart_cache, sourceless=True, group_name=group_name,
        )
        face_bboxes: dict[str, list[int]] = {}
    else:
        train_ds = GroupDataset(
            group_folder, config.class_names, split="train",
            multi_label=config.multi_label,
            skin_normalise=config.skin_normalise, group_name=group_name,
        )
        val_ds = GroupDataset(
            group_folder, config.class_names, split="val",
            multi_label=config.multi_label,
            skin_normalise=config.skin_normalise, group_name=group_name,
        )

        # --- Face-aware cropping pre-computation ---
        face_bboxes = {}
        if config.face_model_path:
            from bittrainer.face_crop import FaceBBoxCache, precompute_face_bboxes
            face_cache = FaceBBoxCache(group_folder / ".resize_cache" / "face_bboxes.json")
            all_image_paths = [s["path"] for s in train_ds.samples] + [s["path"] for s in val_ds.samples]

            def _face_progress(done: int, total: int) -> None:
                cb({
                    "type": "training_progress", "stage": "face_detection",
                    "status_text": f"Detecting faces ({done}/{total})",
                    "step": done, "total_steps": total,
                })

            precompute_face_bboxes(
                all_image_paths, face_cache, config.face_model_path,
                device=config.device,
                progress_fn=_face_progress,
            )
            for p in all_image_paths:
                bbox = face_cache.get(p)
                if bbox:
                    face_bboxes[p] = bbox
            train_ds.refresh_face_bboxes(face_bboxes)
            val_ds.refresh_face_bboxes(face_bboxes)

        # --- Warm SmartCache ---
        if smart_cache is not None:
            from bittrainer.cache_builders import build_image_tensor
            from bittrainer.smart_cache import CachingStoppedException
            all_cache_samples = train_ds.samples + val_ds.samples
            try:
                smart_cache.prepare(
                    all_cache_samples, build_image_tensor,
                    num_workers=1, stage_label="caching",
                )
            except CachingStoppedException:
                logger.info("Caching interrupted by stop_event")
                cb({"type": "training_cancelled", "stage": "caching",
                    "status_text": "Cancelled during cache build"})
                raise
            # Callbacks are only needed during prepare(). Replace with picklable
            # no-ops so the cache (now attached to datasets) survives pickling
            # when DataLoader workers spawn on Windows — mp.Event and local
            # closures aren't picklable.
            smart_cache._progress_cb = _noop_callback
            smart_cache._stop_check = _never_stop
            train_ds.set_cache(smart_cache)
            val_ds.set_cache(smart_cache)

    total_samples = len(train_ds)
    if total_samples == 0:
        raise RuntimeError("No training images found")

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
    # --- Auto batch sizing (probe unfrozen = worst-case VRAM) ---
    from bittrainer.autobatch import determine_batch_size
    auto_result = determine_batch_size(model, bucket_counts, device, dtype=dtype)
    eff_bs = auto_result["batch_size"]
    cb({"type": "autobatch", **auto_result})
    freeze_backbone(model)

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
        if stop_event is not None and stop_event.is_set():
            logger.info("Graceful stop requested after epoch %d — running final comparison", epoch)
            cb({"type": "graceful_stop", "epoch": epoch, "max_epochs": config.max_epochs})
            break

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
            num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=3,
        )
        val_sampler = build_group_bucket_sampler(val_ds, batch_size=eff_bs)
        val_loader = DataLoader(
            val_ds, batch_sampler=val_sampler, collate_fn=collate_fn,
            num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=3,
        )

        # Train
        epoch_start_mono = time.monotonic()

        def _on_step(step: int, total_steps: int, avg_loss: float) -> None:
            elapsed = time.monotonic() - epoch_start_mono
            throughput = step / elapsed if elapsed > 0 else None
            eta_seconds = (total_steps - step) / throughput if throughput and throughput > 0 else None
            cb({
                "type": "training_progress",
                "stage": "training",
                "status_text": f"Training (epoch {epoch + 1}/{config.max_epochs}, step {step}/{total_steps})",
                "epoch": epoch + 1,
                "max_epochs": config.max_epochs,
                "step": step,
                "total_steps": total_steps,
                "eta_seconds": eta_seconds,
                "throughput": throughput,
                "throughput_unit": "batch/s",
                "train_loss": round(avg_loss, 4),
                "best_val_macro_f1": best_val_macro_f1 if best_val_macro_f1 >= 0 else None,
                "best_epoch": best_epoch + 1 if best_val_macro_f1 >= 0 else None,
            })

        train_loss = _train_one_epoch(
            model, train_loader, optimizer, config, device, dtype,
            use_soft_targets=use_soft,
            step_callback=_on_step,
        )
        scheduler.step()

        # Validate
        val_metrics = _evaluate(
            model, val_loader, config.num_classes, device, dtype,
            multi_label=config.multi_label,
            ordinal=config.ordinal,
            none_index=_resolve_none_index(config.class_names),
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
            "stage": "training",
            "status_text": f"Epoch {epoch + 1}/{config.max_epochs} complete (val macro F1 {val_macro_f1:.3f})",
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
                    none_index=_resolve_none_index(config.class_names),
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
