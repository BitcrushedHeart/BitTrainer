"""Multi-head ordinal trainer for size prediction.

Single ConvNeXt V2 backbone -> shared trunk -> {band head, size head}. The band head trains
only on the band (cup ignored); the size head trains on the full US size with volume-distance
soft labels so sister sizes are near-equivalent. A band-consistency term keeps the two heads
agreeing. Validation reports per-head and combined F1/QWK; model selection is on combined QWK.

Structurally mirrors :mod:`bittrainer.dual_branch_trainer` (warm-start, autobatch, compile,
early stopping, best-checkpoint promotion) — adapted to one input and two heads. The size
classes come in at their suite class-index order; all ordinal bookkeeping (band vocab,
size->band, size->volume, volume ranks) is derived here from the class names.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from adv_optm import Prodigy_adv
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from bittrainer.backbone_init import apply_backbone_init, wants_timm_pretrained
from bittrainer.dataset import get_train_transform, get_val_transform
from bittrainer.group_dataset import GroupDataset
from bittrainer.group_validation import compute_multihead_metrics
from bittrainer.multihead_losses import (
    BandConsistencyLoss,
    BandOrdinalSoftLabelLoss,
    VolumeSoftLabelLoss,
)
from bittrainer.multihead_model import MultiHeadConvNeXt
from bittrainer.multihead_ordinal import (
    build_band_vocab,
    size_to_band_index,
    size_to_volume,
    volume_ranks,
)

logger = logging.getLogger(__name__)

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
        if step_callback is not None:
            now = time.monotonic()
            if now - _last >= 0.25 or num_batches == total_steps:
                _last = now
                step_callback(num_batches, total_steps, total_loss / num_batches)

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


def run_multihead_training(
    config: MultiHeadTrainConfig,
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

    group_folder = Path(config.group_folder)
    checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else group_folder / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    size_classes = config.size_classes
    maps = _build_maps(size_classes)
    num_bands = len(maps.band_vocab)
    if num_bands < 1 or maps.num_ranks < 1:
        raise RuntimeError("Multi-head training needs at least one parseable size class")

    train_ds = GroupDataset(group_folder, size_classes, split="train", transform=get_train_transform())
    val_ds = GroupDataset(group_folder, size_classes, split="val", transform=get_val_transform())
    total_samples = len(train_ds)
    if total_samples == 0:
        raise RuntimeError("No training images found")

    existing_best = checkpoint_dir / config.best_model_name

    def _fresh_model() -> MultiHeadConvNeXt:
        model = MultiHeadConvNeXt(
            backbone_variant=config.backbone_variant,
            n_bands=num_bands,
            n_sizes=len(size_classes),
            band_classes=maps.band_vocab,
            size_classes=size_classes,
            pretrained=wants_timm_pretrained(config.backbone_init),
        )
        apply_backbone_init(model.backbone, config.backbone_init)
        return model.to(device)

    if not config.from_scratch and existing_best.exists():
        try:
            model = MultiHeadConvNeXt.from_checkpoint(str(existing_best), device=device)
            logger.info("Warm-starting multi-head model from %s", existing_best)
        except (RuntimeError, KeyError, FileNotFoundError):
            logger.warning("Failed to warm-start, using pretrained backbone", exc_info=True)
            model = _fresh_model()
    else:
        model = _fresh_model()

    memory_format = torch.channels_last if config.channels_last else None
    if memory_format is not None:
        model = model.to(memory_format=memory_format)

    def _make_inputs(b: int) -> torch.Tensor:
        x = torch.randn(b, 3, 512, 512, device=device, dtype=dtype)
        if memory_format is not None:
            x = x.contiguous(memory_format=memory_format)
        return x

    if config.batch_size is not None and config.batch_size > 0:
        eff_bs = int(config.batch_size)
        cb({"type": "autobatch", "batch_size": eff_bs, "manual_override": True})
    else:
        from bittrainer.autobatch import determine_batch_size

        cb({"type": "training_progress", "stage": "preparing", "status_text": "Probing optimal batch size"})
        auto_result = determine_batch_size(
            model, {(512, 512): total_samples}, device, dtype=dtype, use_ema=False,
            make_inputs=lambda b: (_make_inputs(b),),
        )
        eff_bs = auto_result["batch_size"]
        cb({"type": "autobatch", **auto_result})

    size_loss_fn = VolumeSoftLabelLoss(
        size_to_volume(size_classes), temperature=config.size_temperature, ignore_index=maps.none_index,
    ).to(device)
    band_loss_fn = BandOrdinalSoftLabelLoss(
        num_bands, temperature=config.band_temperature, ignore_index=-1,
    ).to(device)
    consistency_fn = BandConsistencyLoss(
        maps.size_to_band.tolist(), num_bands, weight=config.consistency_weight,
    ).to(device)
    band_loss_fn = _Scaled(band_loss_fn, config.band_loss_weight)
    criteria = (size_loss_fn, band_loss_fn, consistency_fn)

    optimizer = Prodigy_adv(
        model.parameters(), lr=1.0, d_coef=0.9, weight_decay=0.01, betas=(0.9, 0.999),
        kourkoutas_beta=True, k_warmup_steps=50, cautious_wd=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    fwd_model, compiled = maybe_compile(model, enabled=config.use_compile, cb=cb)
    if compiled and not prewarm_compile(
        fwd_model, {(512, 512): total_samples}, eff_bs, device, dtype,
        memory_format=memory_format, make_inputs=lambda b, _bucket: (_make_inputs(b),), cb=cb,
    ):
        fwd_model = model

    train_loader = DataLoader(
        train_ds, batch_size=eff_bs, shuffle=True, collate_fn=_collate, num_workers=6,
        pin_memory=(device.type == "cuda"), persistent_workers=True, prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=eff_bs, shuffle=False, collate_fn=_collate, num_workers=6,
        pin_memory=(device.type == "cuda"), persistent_workers=True, prefetch_factor=4,
    )

    best_qwk = -1.0
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path: str | None = None
    best_metrics: dict = {}
    epoch = 0

    for epoch in range(config.max_epochs):
        if stop_now_event is not None and stop_now_event.is_set():
            cb({"type": "stop_now", "epoch": epoch, "max_epochs": config.max_epochs})
            break
        if stop_event is not None and stop_event.is_set():
            cb({"type": "graceful_stop", "epoch": epoch, "max_epochs": config.max_epochs})
            break

        def _on_step(step, total_steps, avg_loss):
            cb({
                "type": "step", "epoch": epoch + 1, "max_epochs": config.max_epochs,
                "step": step, "total_steps": total_steps, "train_loss": round(avg_loss, 4),
                "best_val_qwk": best_qwk if best_qwk >= 0 else None,
            })

        train_loss = _train_one_epoch(
            fwd_model, train_loader, optimizer, criteria, maps, device, dtype,
            step_callback=_on_step, stop_now_event=stop_now_event, memory_format=memory_format,
        )
        scheduler.step()

        val_metrics = _evaluate(
            fwd_model, val_loader, criteria, maps, num_bands, device, dtype, memory_format=memory_format,
        )
        val_metrics["train_loss"] = train_loss
        combined_qwk = val_metrics["multi_head"]["qwk"]

        if combined_qwk > best_qwk:
            best_qwk = combined_qwk
            best_epoch = epoch
            patience_counter = 0
            best_metrics = val_metrics.copy()
            ckpt_path = checkpoint_dir / "candidate.pt"
            model.save_checkpoint(str(ckpt_path), metadata={
                "epoch": epoch + 1,
                "band_qwk": val_metrics["band"]["qwk"],
                "size_qwk": val_metrics["size"]["qwk"],
                "multi_head_qwk": combined_qwk,
            })
            best_checkpoint_path = str(ckpt_path)
        else:
            patience_counter += 1

        cb({
            "type": "epoch_complete", "epoch": epoch + 1, "max_epochs": config.max_epochs,
            "train_loss": train_loss, "val_loss": val_metrics["val_loss"],
            "band": val_metrics["band"], "size": val_metrics["size"],
            "multi_head": val_metrics["multi_head"], "best_val_qwk": best_qwk, "best_epoch": best_epoch + 1,
        })

        if patience_counter >= config.patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    if best_checkpoint_path:
        Path(best_checkpoint_path).replace(existing_best)
        best_checkpoint_path = str(existing_best)

    band = best_metrics.get("band", {})
    size = best_metrics.get("size", {})
    multi = best_metrics.get("multi_head", {})
    return {
        "epochs_completed": epoch + 1,
        "best_epoch": best_epoch + 1,
        "checkpoint_path": best_checkpoint_path,
        "total_images": total_samples,
        "final_val_loss": best_metrics.get("val_loss"),
        "band_classes": maps.band_vocab,
        # Per-head + combined metrics (the issue's required validation figures).
        "final_val_f1_band": band.get("f1"),
        "final_val_qwk_band": band.get("qwk"),
        "final_val_f1_size": size.get("f1"),
        "final_val_qwk_size": size.get("qwk"),
        "final_val_f1_multihead": multi.get("f1"),
        "final_val_qwk_multihead": multi.get("qwk"),
        # Mirror single-head keys so existing persistence keeps working.
        "qwk": multi.get("qwk"),
        "best_val_qwk": multi.get("qwk"),
        "macro_f1": multi.get("f1"),
        "final_val_macro_f1": multi.get("f1"),
    }


class _Scaled(torch.nn.Module):
    """Wrap a loss module to scale its output by a constant weight."""

    def __init__(self, loss: torch.nn.Module, weight: float):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *args) -> torch.Tensor:
        return self.weight * self.loss(*args)
