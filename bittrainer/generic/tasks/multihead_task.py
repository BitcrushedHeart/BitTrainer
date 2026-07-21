"""Multi-head ordinal (Skin Tone V2 / size) training as a :class:`TrainingTask`
(Bitcrush ISSUE-0542 Step 7).

Ports ``bittrainer.multihead_trainer.run_multihead_training`` onto the shared
:class:`~bittrainer.generic.generic_trainer.GenericTrainer`. The multi-head
trainer keeps its own mechanics — one backbone -> shared trunk -> {band, size}
heads, the volume/band ordinal soft-label loss zoo + band-consistency term, and
selection on the combined QWK — all expressed through the generic hooks. It trains
fully unfrozen throughout (no epoch-1 unfreeze) and has no EMA / SWA / soup / mixup.
Backup / pause / resume is epoch-restart (not bit-exact mid-epoch): no RNG or batch
schedule rides in the backup envelope.

The trainer's module-level helpers (``_train_one_epoch`` / ``_evaluate`` /
``_build_maps`` / ``_collate`` / ``_get_dtype`` / ``_Scaled``) stay in
``bittrainer.multihead_trainer`` and are reached through the ``mh`` module alias so
their existing import + monkeypatch seams keep firing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import bittrainer.multihead_trainer as mh
from bittrainer.backbone_init import apply_backbone_init, wants_timm_pretrained
from bittrainer.dataset import get_train_transform, get_val_transform
from bittrainer.generic.optimizer import make_optimizer
from bittrainer.generic.task import BestTracker, LoopSpec, ResumeInfo, TaskContext, TrainingTask
from bittrainer.group_dataset import GroupDataset
from bittrainer.multihead_losses import (
    BandConsistencyLoss,
    BandOrdinalSoftLabelLoss,
    VolumeSoftLabelLoss,
)
from bittrainer.multihead_model import MultiHeadConvNeXt
from bittrainer.multihead_ordinal import size_to_volume
from bittrainer.training_state import init_backup, loader_kwargs, restore_optimizer_state

logger = logging.getLogger(__name__)


class MultiHeadTask(TrainingTask):
    """Drives ``GenericTrainer`` for a single backbone with band + size heads."""

    trainer_name = "multihead"

    def __init__(self, config: mh.MultiHeadTrainConfig) -> None:
        self.config = config
        # Populated across the lifecycle hooks.
        self.size_classes: list[str] = list(config.size_classes)
        self.maps = None
        self.num_bands = 0
        self.train_ds = None
        self.val_ds = None
        self.total_samples = 0
        self.memory_format = None
        self.criteria = None
        self.fwd_model = None
        self.eff_bs = 0
        self.scheduler_t_max = 0
        self._train_loader = None
        self._val_loader = None

    # -- one-time setup ----------------------------------------------------
    def make_context(self, progress_callback, stop_event, stop_now_event, pause_event) -> TaskContext:
        from bittrainer.runtime import configure_cuda_backend

        config = self.config
        cb = progress_callback or config.progress_callback or (lambda _: None)
        device = torch.device(config.device)
        dtype = mh._get_dtype(config.dtype)
        configure_cuda_backend()

        class _RawEmitter:
            """Multi-head emits raw frames only; ``.stage`` is a no-op the core
            calls once per epoch (validating) which this trainer never surfaced."""

            def __init__(self, raw) -> None:
                self.raw = raw

            def stage(self, *_a, **_k) -> None:
                pass

        group_folder = Path(config.group_folder)
        checkpoint_dir = (
            Path(config.checkpoint_dir) if config.checkpoint_dir else group_folder / "checkpoints"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return TaskContext(
            device=device, dtype=dtype, em=_RawEmitter(cb), cb=cb, checkpoint_dir=checkpoint_dir,
            stop_event=stop_event, stop_now_event=stop_now_event, pause_event=pause_event,
        )

    def fingerprint_init(self, ctx: TaskContext) -> None:
        config = self.config
        coordinator, fingerprint, resume_state = init_backup(
            config, ctx.pause_event, ctx.cb,
            class_names=list(config.size_classes), num_classes=len(config.size_classes),
            max_epochs=config.max_epochs, multi_label=False, ordinal=True,
            best_model_name=config.best_model_name, model_size=config.backbone_variant,
        )
        ctx.coordinator = coordinator
        ctx.fingerprint = fingerprint
        ctx.resume_state = resume_state

    def loop_spec(self) -> LoopSpec:
        # Selection is on strict combined-QWK improvement (no min-delta guard).
        return LoopSpec(max_epochs=self.config.max_epochs, patience=self.config.patience,
                        selection_min_delta=0.0)

    def prepare_data(self, ctx: TaskContext) -> None:
        config = self.config
        group_folder = Path(config.group_folder)
        self.maps = mh._build_maps(self.size_classes)
        self.num_bands = len(self.maps.band_vocab)
        if self.num_bands < 1 or self.maps.num_ranks < 1:
            raise RuntimeError("Multi-head training needs at least one parseable size class")

        self.train_ds = GroupDataset(
            group_folder, self.size_classes, split="train", transform=get_train_transform(),
        )
        self.val_ds = GroupDataset(
            group_folder, self.size_classes, split="val", transform=get_val_transform(),
        )
        self.total_samples = len(self.train_ds)
        if self.total_samples == 0:
            raise RuntimeError("No training images found")

    def create_model(self, ctx: TaskContext, resume_state: dict | None):
        config = self.config
        device = ctx.device
        self.memory_format = torch.channels_last if config.channels_last else None
        existing_best = ctx.checkpoint_dir / config.best_model_name

        def _fresh_model() -> MultiHeadConvNeXt:
            model = MultiHeadConvNeXt(
                backbone_variant=config.backbone_variant,
                n_bands=self.num_bands,
                n_sizes=len(self.size_classes),
                band_classes=self.maps.band_vocab,
                size_classes=self.size_classes,
                pretrained=wants_timm_pretrained(config.backbone_init),
            )
            apply_backbone_init(model.backbone, config.backbone_init)
            return model.to(device)

        if resume_state is not None:
            model = _fresh_model()
            model.load_state_dict(resume_state["model"])
        elif not config.from_scratch and existing_best.exists():
            try:
                model = MultiHeadConvNeXt.from_checkpoint(str(existing_best), device=device)
                logger.info("Warm-starting multi-head model from %s", existing_best)
            except (RuntimeError, KeyError, FileNotFoundError):
                logger.warning("Failed to warm-start, using pretrained backbone", exc_info=True)
                model = _fresh_model()
        else:
            model = _fresh_model()

        if self.memory_format is not None:
            model = model.to(memory_format=self.memory_format)
        return model

    def resolve_batch_size(self, ctx: TaskContext, model, resume_state: dict | None) -> int:
        config = self.config
        cb = ctx.cb
        device, dtype = ctx.device, ctx.dtype

        def _make_inputs(b: int) -> torch.Tensor:
            x = torch.randn(b, 3, 512, 512, device=device, dtype=dtype)
            if self.memory_format is not None:
                x = x.contiguous(memory_format=self.memory_format)
            return x

        if resume_state is not None:
            eff_bs = int(resume_state["eff_bs"])
            cb({"type": "autobatch", "batch_size": eff_bs, "resumed": True})
        elif config.batch_size is not None and config.batch_size > 0:
            eff_bs = int(config.batch_size)
            cb({"type": "autobatch", "batch_size": eff_bs, "manual_override": True})
        else:
            from bittrainer.autobatch import determine_batch_size

            cb({"type": "training_progress", "stage": "preparing", "status_text": "Probing optimal batch size"})
            auto_result = determine_batch_size(
                model, {(512, 512): self.total_samples}, device, dtype=dtype, use_ema=False,
                make_inputs=lambda b: (_make_inputs(b),),
            )
            eff_bs = auto_result["batch_size"]
            cb({"type": "autobatch", **auto_result})
        self.eff_bs = eff_bs
        return eff_bs

    def setup_training(self, ctx: TaskContext, model, resume_state: dict | None) -> None:
        config = self.config
        device = ctx.device
        maps = self.maps
        size_loss_fn = VolumeSoftLabelLoss(
            size_to_volume(self.size_classes), temperature=config.size_temperature,
            ignore_index=maps.none_index,
        ).to(device)
        band_loss_fn = BandOrdinalSoftLabelLoss(
            self.num_bands, temperature=config.band_temperature, ignore_index=-1,
        ).to(device)
        consistency_fn = BandConsistencyLoss(
            maps.size_to_band.tolist(), self.num_bands, weight=config.consistency_weight,
        ).to(device)
        band_loss_fn = mh._Scaled(band_loss_fn, config.band_loss_weight)
        self.criteria = (size_loss_fn, band_loss_fn, consistency_fn)

    def create_optimizer(self, ctx: TaskContext, model, eff_bs: int, resume_state: dict | None):
        from bittrainer.runtime import maybe_compile, prewarm_compile

        config = self.config
        device, dtype = ctx.device, ctx.dtype
        optimizer = make_optimizer(model)
        # The multi-head trainer never recreates the scheduler mid-run.
        self.scheduler_t_max = config.max_epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=self.scheduler_t_max)

        if resume_state is not None:
            restore_optimizer_state(resume_state, optimizer, scheduler, device)

        def _make_inputs(b: int) -> torch.Tensor:
            x = torch.randn(b, 3, 512, 512, device=device, dtype=dtype)
            if self.memory_format is not None:
                x = x.contiguous(memory_format=self.memory_format)
            return x

        fwd_model, compiled = maybe_compile(model, enabled=config.use_compile, cb=ctx.cb)
        if compiled and not prewarm_compile(
            fwd_model, {(512, 512): self.total_samples}, eff_bs, device, dtype,
            memory_format=self.memory_format,
            make_inputs=lambda b, _bucket: (_make_inputs(b),), cb=ctx.cb,
        ):
            fwd_model = model
        self.fwd_model = fwd_model
        return optimizer, scheduler, self.scheduler_t_max

    def resumed_message(self, ctx: TaskContext, best: BestTracker, global_step: int, start_epoch: int) -> dict:
        return {
            "type": "training_resumed", "resumed_from": str(self.config.resume_from),
            "epoch": start_epoch, "global_step": global_step,
            "best_val_qwk": best.best_validation_score,
        }

    # -- per-epoch ---------------------------------------------------------
    def build_loaders(self, ctx: TaskContext, epoch: int, eff_bs: int, resume_info: ResumeInfo):
        config = self.config
        # Static loaders (shuffle=True reshuffles the train order each epoch):
        # build once, on the first epoch of the run.
        if self._train_loader is None:
            lk = loader_kwargs(config.dataloader_workers, pin_memory=(ctx.device.type == "cuda"))
            self._train_loader = DataLoader(
                self.train_ds, batch_size=eff_bs, shuffle=True, collate_fn=mh._collate, **lk,
            )
            self._val_loader = DataLoader(
                self.val_ds, batch_size=eff_bs, shuffle=False, collate_fn=mh._collate, **lk,
            )
        return self._train_loader, None, 0

    def make_step_callback(self, ctx: TaskContext, epoch: int, eff_bs: int, best: BestTracker, epoch_start_mono: float):
        config = self.config
        cb = ctx.cb

        def _on_step(step, total_steps, avg_loss):
            best_qwk = best.best_validation_score
            cb({
                "type": "step", "epoch": epoch + 1, "max_epochs": config.max_epochs,
                "step": step, "total_steps": total_steps, "train_loss": round(avg_loss, 4),
                "best_val_qwk": best_qwk if best_qwk >= 0 else None,
            })

        return _on_step

    def train_epoch(self, ctx: TaskContext, model, optimizer, train_loader, *, step_callback, boundary_hook, start_batch: int):
        return mh._train_one_epoch(
            self.fwd_model, train_loader, optimizer, self.criteria, self.maps, ctx.device, ctx.dtype,
            step_callback=step_callback, stop_now_event=ctx.stop_now_event,
            memory_format=self.memory_format, boundary_hook=boundary_hook,
        )

    def validate(self, ctx: TaskContext, model, epoch: int, train_result) -> dict:
        val_metrics = mh._evaluate(
            self.fwd_model, self._val_loader, self.criteria, self.maps, self.num_bands,
            ctx.device, ctx.dtype, memory_format=self.memory_format,
        )
        val_metrics["train_loss"] = train_result
        return val_metrics

    def selection_score(self, metrics: dict) -> float:
        return float(metrics["multi_head"]["qwk"])

    def save_candidate(self, ctx: TaskContext, model, epoch: int, metrics: dict, best: BestTracker) -> None:
        combined_qwk = metrics["multi_head"]["qwk"]
        best.best_val_qwk = combined_qwk
        best.best_metrics = metrics.copy()
        ckpt_path = ctx.checkpoint_dir / "candidate.pt"
        model.save_checkpoint(str(ckpt_path), metadata={
            "epoch": epoch + 1,
            "band_qwk": metrics["band"]["qwk"],
            "size_qwk": metrics["size"]["qwk"],
            "multi_head_qwk": combined_qwk,
        })
        best.best_checkpoint_path = str(ckpt_path)

    def epoch_message(self, ctx: TaskContext, epoch: int, metrics: dict, train_result, selected_score: float, best: BestTracker) -> dict:
        config = self.config
        return {
            "type": "epoch_complete", "epoch": epoch + 1, "max_epochs": config.max_epochs,
            "train_loss": train_result, "val_loss": metrics["val_loss"],
            "band": metrics["band"], "size": metrics["size"],
            "multi_head": metrics["multi_head"], "best_val_qwk": best.best_validation_score,
            "best_epoch": best.best_epoch + 1,
        }

    # -- finalisation ------------------------------------------------------
    def finalize(self, ctx: TaskContext, model, best: BestTracker, epochs_completed: int) -> dict:
        config = self.config
        existing_best = ctx.checkpoint_dir / config.best_model_name
        best_checkpoint_path = best.best_checkpoint_path
        if best_checkpoint_path:
            Path(best_checkpoint_path).replace(existing_best)
            best_checkpoint_path = str(existing_best)

        best_metrics = best.best_metrics
        band = best_metrics.get("band", {})
        size = best_metrics.get("size", {})
        multi = best_metrics.get("multi_head", {})
        return {
            "epochs_completed": epochs_completed,
            "best_epoch": best.best_epoch + 1,
            "checkpoint_path": best_checkpoint_path,
            "total_images": self.total_samples,
            "final_val_loss": best_metrics.get("val_loss"),
            "band_classes": self.maps.band_vocab,
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
