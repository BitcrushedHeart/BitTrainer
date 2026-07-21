"""Dual-branch (skin-tone crop+context) training as a :class:`TrainingTask`
(Bitcrush ISSUE-0542 Step 7).

Ports ``bittrainer.dual_branch_trainer.run_dual_branch_training`` onto the shared
:class:`~bittrainer.generic.generic_trainer.GenericTrainer`. The dual-branch
trainer keeps its own mechanics — two ConvNeXt V2 branches (crop + context) fused
to one head, label-smoothed cross-entropy, selection on macro-F1 — all expressed
through the generic hooks. It trains fully unfrozen throughout (no epoch-1
unfreeze) and has no EMA / SWA / soup / mixup. Backup / pause / resume is
epoch-restart (not bit-exact mid-epoch).

Its finalisation is BESPOKE (ISSUE-0490): :meth:`finalize` runs the dual-branch
compare-vs-incumbent promotion and builds its OWN result dict (per-class counts,
confusion matrix, balanced accuracy). This is intentionally NOT merged with the
group calibration/finalisation path.

The trainer's module-level helpers (``_train_one_epoch`` / ``_evaluate`` /
``_fresh_dual_branch_model`` / ``_collate_dual`` / ``_get_dtype``) stay in
``bittrainer.dual_branch_trainer`` and are reached through the ``db`` module alias
so their existing import + monkeypatch seams keep firing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import bittrainer.dual_branch_trainer as db
from bittrainer.dual_branch_model import DualBranchConvNeXt
from bittrainer.dual_crop_dataset import DualCropDataset
from bittrainer.generic.optimizer import make_optimizer
from bittrainer.generic.task import BestTracker, LoopSpec, ResumeInfo, TaskContext, TrainingTask
from bittrainer.group_dataset import get_train_transform, get_val_transform
from bittrainer.training_state import init_backup, loader_kwargs, restore_optimizer_state

logger = logging.getLogger(__name__)


class DualBranchTask(TrainingTask):
    """Drives ``GenericTrainer`` for a crop+context dual-branch classifier."""

    trainer_name = "dual_branch"

    def __init__(self, config: db.DualBranchTrainConfig) -> None:
        self.config = config
        # Populated across the lifecycle hooks.
        self.crops_folder = Path(config.group_folder)
        self.context_folder = Path(config.context_folder)
        self.train_ds = None
        self.val_ds = None
        self.total_samples = 0
        self.memory_format = None
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
        dtype = db._get_dtype(config.dtype)
        configure_cuda_backend()

        class _RawEmitter:
            """Dual-branch emits raw frames only; ``.stage`` is a no-op the core
            calls once per epoch (validating) which this trainer never surfaced."""

            def __init__(self, raw) -> None:
                self.raw = raw

            def stage(self, *_a, **_k) -> None:
                pass

        checkpoint_dir = (
            Path(config.checkpoint_dir) if config.checkpoint_dir else self.crops_folder / "checkpoints"
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
            class_names=list(config.class_names), num_classes=config.num_classes,
            max_epochs=config.max_epochs, multi_label=False, ordinal=False,
            best_model_name=config.best_model_name, model_size=config.backbone_variant,
        )
        ctx.coordinator = coordinator
        ctx.fingerprint = fingerprint
        ctx.resume_state = resume_state

    def loop_spec(self) -> LoopSpec:
        # Selection is on strict macro-F1 improvement (no min-delta guard).
        return LoopSpec(max_epochs=self.config.max_epochs, patience=self.config.patience,
                        selection_min_delta=0.0)

    def prepare_data(self, ctx: TaskContext) -> None:
        config = self.config
        train_transform = get_train_transform()
        val_transform = get_val_transform()
        self.train_ds = DualCropDataset(
            self.crops_folder, self.context_folder, config.class_names,
            split="train", crop_transform=train_transform, context_transform=train_transform,
        )
        self.val_ds = DualCropDataset(
            self.crops_folder, self.context_folder, config.class_names,
            split="val", crop_transform=val_transform, context_transform=val_transform,
        )
        self.total_samples = len(self.train_ds)
        if self.total_samples == 0:
            raise RuntimeError("No training image pairs found")

    def create_model(self, ctx: TaskContext, resume_state: dict | None):
        config = self.config
        device = ctx.device
        self.memory_format = torch.channels_last if config.channels_last else None
        existing_best = ctx.checkpoint_dir / config.best_model_name

        if resume_state is not None:
            model = db._fresh_dual_branch_model(config).to(device)
            model.load_state_dict(resume_state["model"])
        elif not config.from_scratch and existing_best.exists():
            try:
                model = DualBranchConvNeXt.from_checkpoint(str(existing_best), device=device)
                logger.info("Warm-starting from existing dual-branch checkpoint: %s", existing_best)
            except (RuntimeError, KeyError, FileNotFoundError):
                logger.warning("Failed to load existing checkpoint, starting from pretrained", exc_info=True)
                model = db._fresh_dual_branch_model(config).to(device)
        else:
            model = db._fresh_dual_branch_model(config).to(device)

        if self.memory_format is not None:
            model = model.to(memory_format=self.memory_format)
        return model

    def resolve_batch_size(self, ctx: TaskContext, model, resume_state: dict | None) -> int:
        config = self.config
        cb = ctx.cb
        device, dtype = ctx.device, ctx.dtype

        def _dual_inputs(b: int) -> tuple[torch.Tensor, torch.Tensor]:
            crops = torch.randn(b, 3, 512, 512, device=device, dtype=dtype)
            contexts = torch.randn(b, 3, 512, 512, device=device, dtype=dtype)
            if self.memory_format is not None:
                crops = crops.contiguous(memory_format=self.memory_format)
                contexts = contexts.contiguous(memory_format=self.memory_format)
            return crops, contexts

        if resume_state is not None:
            eff_bs = int(resume_state["eff_bs"])
            cb({"type": "autobatch", "batch_size": eff_bs, "resumed": True})
        elif config.batch_size is not None and config.batch_size > 0:
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
                model, {(512, 512): self.total_samples}, device, dtype=dtype,
                use_ema=False, make_inputs=_dual_inputs, progress_callback=_probe_progress,
            )
            eff_bs = auto_result["batch_size"]
            cb({"type": "autobatch", **auto_result})
        self.eff_bs = eff_bs
        self._dual_inputs = _dual_inputs
        return eff_bs

    def create_optimizer(self, ctx: TaskContext, model, eff_bs: int, resume_state: dict | None):
        from bittrainer.runtime import maybe_compile, prewarm_compile

        config = self.config
        device, dtype = ctx.device, ctx.dtype
        optimizer = make_optimizer(model)
        self.scheduler_t_max = config.max_epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=self.scheduler_t_max)

        if resume_state is not None:
            restore_optimizer_state(resume_state, optimizer, scheduler, device)

        fwd_model, compiled = maybe_compile(model, enabled=config.use_compile, cb=ctx.cb)
        if compiled and not prewarm_compile(
            fwd_model, {(512, 512): self.total_samples}, eff_bs, device, dtype,
            memory_format=self.memory_format,
            make_inputs=lambda b, _bucket: self._dual_inputs(b), cb=ctx.cb,
        ):
            fwd_model = model
        self.fwd_model = fwd_model
        return optimizer, scheduler, self.scheduler_t_max

    def resumed_message(self, ctx: TaskContext, best: BestTracker, global_step: int, start_epoch: int) -> dict:
        return {
            "type": "training_resumed", "resumed_from": str(self.config.resume_from),
            "epoch": start_epoch, "global_step": global_step,
            "best_val_macro_f1": best.best_validation_score,
        }

    # -- per-epoch ---------------------------------------------------------
    def build_loaders(self, ctx: TaskContext, epoch: int, eff_bs: int, resume_info: ResumeInfo):
        config = self.config
        # Static loaders (shuffle=True reshuffles the train order each epoch):
        # build once, on the first epoch of the run.
        if self._train_loader is None:
            lk = loader_kwargs(config.dataloader_workers, pin_memory=(ctx.device.type == "cuda"))
            self._train_loader = DataLoader(
                self.train_ds, batch_size=eff_bs, shuffle=True, collate_fn=db._collate_dual, **lk,
            )
            self._val_loader = DataLoader(
                self.val_ds, batch_size=eff_bs, shuffle=False, collate_fn=db._collate_dual, **lk,
            )
        return self._train_loader, None, 0

    def make_step_callback(self, ctx: TaskContext, epoch: int, eff_bs: int, best: BestTracker, epoch_start_mono: float):
        config = self.config
        cb = ctx.cb

        def _on_step(step: int, total_steps: int, avg_loss: float) -> None:
            best_f1 = best.best_validation_score
            cb({
                "type": "step",
                "epoch": epoch + 1,
                "max_epochs": config.max_epochs,
                "step": step,
                "total_steps": total_steps,
                "train_loss": round(avg_loss, 4),
                "best_val_macro_f1": best_f1 if best_f1 >= 0 else None,
                "best_epoch": best.best_epoch + 1 if best_f1 >= 0 else None,
            })

        return _on_step

    def train_epoch(self, ctx: TaskContext, model, optimizer, train_loader, *, step_callback, boundary_hook, start_batch: int):
        config = self.config
        return db._train_one_epoch(
            self.fwd_model, train_loader, optimizer, config.num_classes,
            ctx.device, ctx.dtype, config.label_smoothing,
            step_callback=step_callback, stop_now_event=ctx.stop_now_event,
            memory_format=self.memory_format, boundary_hook=boundary_hook,
        )

    def validate(self, ctx: TaskContext, model, epoch: int, train_result) -> dict:
        config = self.config
        val_metrics = db._evaluate(
            self.fwd_model, self._val_loader, config.num_classes, ctx.device, ctx.dtype,
            memory_format=self.memory_format,
        )
        val_metrics["train_loss"] = train_result
        return val_metrics

    def selection_score(self, metrics: dict) -> float:
        return float(metrics["macro_f1"])

    def save_candidate(self, ctx: TaskContext, model, epoch: int, metrics: dict, best: BestTracker) -> None:
        val_macro_f1 = metrics["macro_f1"]
        best.best_val_macro_f1 = val_macro_f1
        best.best_metrics = metrics.copy()
        ckpt_path = ctx.checkpoint_dir / "candidate.pt"
        model.save_checkpoint(str(ckpt_path), metadata={
            "epoch": epoch + 1,
            "val_macro_f1": val_macro_f1,
        })
        best.best_checkpoint_path = str(ckpt_path)

    def epoch_message(self, ctx: TaskContext, epoch: int, metrics: dict, train_result, selected_score: float, best: BestTracker) -> dict:
        config = self.config
        return {
            "type": "epoch_complete",
            "epoch": epoch + 1,
            "max_epochs": config.max_epochs,
            "train_loss": train_result,
            "val_loss": metrics["val_loss"],
            "val_macro_f1": metrics["macro_f1"],
            "per_class_f1": metrics.get("per_class_f1", {}),
            "best_val_macro_f1": best.best_validation_score,
            "best_epoch": best.best_epoch + 1,
        }

    # -- finalisation (BESPOKE — ISSUE-0490) -------------------------------
    def finalize(self, ctx: TaskContext, model, best: BestTracker, epochs_completed: int) -> dict:
        config = self.config
        device, dtype = ctx.device, ctx.dtype
        existing_best = ctx.checkpoint_dir / config.best_model_name
        val_loader = self._val_loader

        best_checkpoint_path = best.best_checkpoint_path
        best_val_macro_f1 = best.best_validation_score
        best_metrics = best.best_metrics

        # Promote best checkpoint — re-evaluate the incumbent so the stronger model
        # ships (the dual-branch trainer builds its own comparison, not group's).
        if best_checkpoint_path:
            if existing_best.exists() and best_checkpoint_path != str(existing_best):
                try:
                    old_model = DualBranchConvNeXt.from_checkpoint(str(existing_best), device=device)
                    old_metrics = db._evaluate(old_model, val_loader, config.num_classes, device, dtype)
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
            class_dir = self.crops_folder / class_name / "train"
            if class_dir.exists():
                class_counts[idx] = sum(1 for f in class_dir.iterdir() if f.is_file())

        return {
            "epochs_completed": epochs_completed,
            "best_epoch": best.best_epoch + 1,
            "best_val_macro_f1": best_val_macro_f1,
            "final_val_macro_f1": best_metrics.get("macro_f1"),
            "final_val_loss": best_metrics.get("val_loss"),
            "per_class_f1": best_metrics.get("per_class_f1", {}),
            "confusion_matrix": best_metrics.get("confusion_matrix", []),
            "balanced_accuracy": best_metrics.get("balanced_accuracy"),
            "checkpoint_path": best_checkpoint_path,
            "class_counts": class_counts,
            "total_images": self.total_samples,
        }
