"""Binary concept training as a :class:`TrainingTask` (Bitcrush ISSUE-0542 Step 5).

Ports ``bittrainer.trainer.run_training`` onto the shared
:class:`~bittrainer.generic.generic_trainer.GenericTrainer` skeleton. The binary
trainer keeps its own mechanics — no EMA / SWA / soup / probes / mixup by default,
a ConceptDataset with per-epoch cross-concept negative resampling, hard-negative
weighting, F1-at-tuned-threshold selection, its own centre-crop collate, and the
epoch-1 backbone unfreeze + scheduler rebuild — all expressed through the generic
hooks. Backup / pause / resume is epoch-restart (not bit-exact mid-epoch).

The epoch-1 unfreeze rides the generic :meth:`TrainingTask.on_epoch_start` hook
(the core swaps in the rebuilt optimizer / scheduler it returns); the resumed
epoch's reconstruction runs once in :meth:`create_optimizer`.

The trainer's module-level helpers (``train_one_epoch`` / ``evaluate`` /
``_binary_compare_promote`` / ``_tuned_val_metrics`` / ``_make_optimizer`` /
``_collate_bucket_batch`` / ``_fresh_binary_model`` / ``_rebalance_val_negatives``)
stay in ``bittrainer.trainer`` and are reached through the ``bt`` module alias so
their existing import + monkeypatch seams keep firing.
"""

from __future__ import annotations

import logging
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import bittrainer.trainer as bt
from bittrainer.dataset import ConceptDataset, _DimensionCache, build_bucket_batch_sampler
from bittrainer.ema import ModelEMA
from bittrainer.generic.task import BestTracker, LoopSpec, ResumeInfo, TaskContext, TrainingTask
from bittrainer.model import (
    freeze_backbone,
    load_checkpoint,
    unfreeze_backbone,
    unfreeze_stage,
)
from bittrainer.trainer import _NUM_STAGES
from bittrainer.training_state import init_backup, loader_kwargs, restore_optimizer_state

logger = logging.getLogger(__name__)


class BinaryTask(TrainingTask):
    """Drives ``GenericTrainer`` for a single ConvNeXt V2 binary concept."""

    trainer_name = "binary"

    def __init__(self, config: bt.TrainConfig) -> None:
        self.config = config
        # Populated across the lifecycle hooks.
        self.concept_name = ""
        self.train_ds: ConceptDataset | None = None
        self.val_ds: ConceptDataset | None = None
        self.smart_cache = None
        self.num_positives = 0
        self.bucket_counts: dict = {}
        self.use_gradual_unfreeze = False
        self.eff_bs = 0
        self.criterion: nn.Module | None = None
        self.ema: ModelEMA | None = None
        self._val_loader = None
        self._loader_kwargs: dict | None = None
        self._epoch_start_mono = 0.0
        self._first_build = True

    # -- one-time setup ----------------------------------------------------
    def make_context(self, progress_callback, stop_event, stop_now_event, pause_event) -> TaskContext:
        from bittrainer.runtime import configure_cuda_backend
        from bittrainer.smart_cache import _noop_callback

        config = self.config
        cb = progress_callback or config.progress_callback or _noop_callback
        device = torch.device(config.device)
        dtype = bt._get_dtype(config.dtype)
        configure_cuda_backend()

        class _RawEmitter:
            """Binary emits raw frames only; ``.stage`` is a no-op the core calls
            once per epoch (validating) which the binary trainer never surfaced."""

            def __init__(self, raw) -> None:
                self.raw = raw

            def stage(self, *_a, **_k) -> None:
                pass

        concept_folder = Path(config.concept_folder)
        checkpoint_dir = (
            Path(config.checkpoint_dir) if config.checkpoint_dir else concept_folder / "checkpoints"
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
            class_names=["negative", "positive"], num_classes=2,
            max_epochs=config.max_epochs, multi_label=False, ordinal=False,
            best_model_name=config.best_model_name, model_size=config.model_size,
        )
        ctx.coordinator = coordinator
        ctx.fingerprint = fingerprint
        ctx.resume_state = resume_state

    def loop_spec(self) -> LoopSpec:
        # Binary selects on strict F1 improvement (no min-delta guard).
        return LoopSpec(max_epochs=self.config.max_epochs, patience=self.config.patience,
                        selection_min_delta=0.0)

    def prepare_data(self, ctx: TaskContext) -> None:
        from bittrainer.smart_cache import _never_stop, _noop_callback

        config = self.config
        cb = ctx.cb
        concept_folder = Path(config.concept_folder)
        self.concept_name = config.concept_name or concept_folder.name

        # --- SmartCache setup ---
        smart_cache = None
        if config.use_cache:
            from bittrainer.smart_cache import SmartCache, face_model_signature
            cache_root = Path(config.cache_dir) if config.cache_dir else (concept_folder / ".smart_cache")
            smart_cache = SmartCache(
                cache_root,
                modeltype=config.modeltype,
                progress_callback=cb,
                stop_check=partial(bt._stop_event_is_set, ctx.stop_event),
                face_model_sig=face_model_signature(config.face_model_path or None),
            )

        # --- Sourceless path: reconstruct samples from cache, skip dataset indexing ---
        if config.sourceless:
            if smart_cache is None:
                raise RuntimeError("sourceless=True requires use_cache=True and a cache_dir")
            cb({
                "type": "training_progress", "stage": "validating",
                "status_text": "Loading sourceless samples from cache",
                "step": 0, "total_steps": 0,
            })
            train_ds = ConceptDataset(
                concept_folder, split="train", cache=smart_cache,
                sourceless=True, concept_name=self.concept_name,
            )
            val_ds = ConceptDataset(
                concept_folder, split="val", cache=smart_cache,
                sourceless=True, concept_name=self.concept_name,
            )
            num_positives = len(train_ds._positive_paths)
        else:
            cache_dir = concept_folder / ".resize_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            dim_cache = _DimensionCache(cache_dir / "dimensions.json")

            train_ds = ConceptDataset(
                concept_folder, split="train",
                neg_pos_ratio=config.neg_pos_ratio,
                extra_positive_dirs=config.extra_positive_dirs,
                negative_dirs=config.negative_dirs,
                hard_negative_paths=config.hard_negative_paths,
                hard_negative_weight=config.hard_negative_weight,
                dim_cache=dim_cache,
                skin_normalise=config.skin_normalise,
                concept_name=self.concept_name,
                train_resolution=config.train_resolution,
            )
            val_ds = ConceptDataset(
                concept_folder, split="val",
                extra_positive_dirs=config.extra_positive_dirs,
                negative_dirs=config.negative_dirs,
                hard_negative_paths=config.hard_negative_paths,
                hard_negative_weight=1,
                dim_cache=dim_cache,
                skin_normalise=config.skin_normalise,
                concept_name=self.concept_name,
                train_resolution=config.train_resolution,
            )

            bt._rebalance_val_negatives(train_ds, val_ds)

            num_positives = len(train_ds._positive_paths)

            # --- Face-aware cropping pre-computation ---
            if config.face_model_path:
                from bittrainer.face_crop import FaceBBoxCache, precompute_face_bboxes
                face_cache = FaceBBoxCache(cache_dir / "face_bboxes.json")
                all_image_paths = [s["path"] for s in train_ds.samples] + [s["path"] for s in val_ds.samples]

                def _face_progress(done: int, total: int) -> None:
                    cb({
                        "type": "training_progress", "stage": "face_detection",
                        "status_text": f"Detecting faces ({done}/{total})",
                        "step": done, "total_steps": total,
                    })

                precompute_face_bboxes(
                    all_image_paths, face_cache, config.face_model_path,
                    device=config.device, progress_fn=_face_progress,
                )
                face_bboxes: dict[str, list[int]] = {}
                for p in all_image_paths:
                    bbox = face_cache.get(p)
                    if bbox:
                        face_bboxes[p] = bbox
                train_ds.refresh_face_bboxes(face_bboxes)
                val_ds.refresh_face_bboxes(face_bboxes)

            # --- Warm SmartCache (validate + build missing) ---
            if smart_cache is not None:
                from bittrainer.cache_builders import build_image_tensor
                from bittrainer.smart_cache import CachingStoppedException
                all_cache_samples = train_ds.samples + val_ds.samples
                try:
                    smart_cache.prepare(
                        all_cache_samples, build_image_tensor,
                        num_workers=config.cache_workers,
                        stage_label="caching",
                    )
                except CachingStoppedException:
                    logger.info("Caching interrupted by stop_event")
                    cb({"type": "training_cancelled", "stage": "caching",
                        "status_text": "Cancelled during cache build"})
                    raise
                # Callbacks are only needed during prepare(). Replace with picklable
                # no-ops so the cache survives pickling when DataLoader workers spawn.
                smart_cache._progress_cb = _noop_callback
                smart_cache._stop_check = _never_stop
                train_ds.set_cache(smart_cache)
                val_ds.set_cache(smart_cache)

        # --- Count samples per bucket ---
        bucket_counts: dict[tuple[int, int], int] = {}
        for s in train_ds.samples:
            b = s["bucket"]
            bucket_counts[b] = bucket_counts.get(b, 0) + 1

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.smart_cache = smart_cache
        self.num_positives = num_positives
        self.bucket_counts = bucket_counts

    def create_model(self, ctx: TaskContext, resume_state: dict | None):
        config = self.config
        device, dtype = ctx.device, ctx.dtype
        checkpoint_dir = ctx.checkpoint_dir
        existing_best = checkpoint_dir / config.best_model_name
        self.use_gradual_unfreeze = self.num_positives < 50
        ctx.cb({"type": "training_progress", "stage": "preparing", "status_text": "Loading model"})

        if resume_state is not None:
            model = bt._fresh_binary_model(config, dtype=dtype).to(device)
            model.load_state_dict(resume_state["model"])
            self.use_gradual_unfreeze = bool(
                resume_state.get("use_gradual_unfreeze", self.use_gradual_unfreeze)
            )
        elif not config.from_scratch and existing_best.exists():
            try:
                model = load_checkpoint(
                    str(existing_best), device=str(device), dtype=dtype, model_size=config.model_size,
                ).to(device)
                logger.info("Warm-starting from existing checkpoint: %s", existing_best)
            except Exception:
                logger.warning("Failed to load existing checkpoint, starting from pretrained", exc_info=True)
                model = bt._fresh_binary_model(config, dtype=dtype).to(device)
        else:
            model = bt._fresh_binary_model(config, dtype=dtype).to(device)
        return model

    def resolve_batch_size(self, ctx: TaskContext, model, resume_state: dict | None) -> int:
        config = self.config
        cb = ctx.cb
        # Probe unfrozen = worst-case VRAM (freeze happens in create_optimizer).
        # Resume reuses the backed-up batch size (skip the probe).
        if resume_state is not None:
            eff_bs = int(resume_state["eff_bs"])
            cb({"type": "autobatch", "batch_size": eff_bs, "resumed": True})
        else:
            from bittrainer.autobatch import determine_batch_size

            def _probe_progress(attempt: int, candidate: int, cap: int, status: str) -> None:
                cb({
                    "type": "training_progress", "stage": "preparing",
                    "status_text": f"Probing batch size (try {attempt}: {candidate}/{cap} — {status})",
                })

            cb({"type": "training_progress", "stage": "preparing", "status_text": "Probing optimal batch size"})
            auto_result = determine_batch_size(
                model, self.bucket_counts, ctx.device, dtype=ctx.dtype,
                use_ema=config.use_ema, progress_callback=_probe_progress,
            )
            eff_bs = auto_result["batch_size"]
            cb({"type": "autobatch", **auto_result})
        self.eff_bs = eff_bs
        return eff_bs

    def create_optimizer(self, ctx: TaskContext, model, eff_bs: int, resume_state: dict | None):
        config = self.config
        device = ctx.device
        # Freeze for epoch 0 (the probe above ran unfrozen for worst-case VRAM).
        freeze_backbone(model)
        optimizer = bt._make_optimizer(model, config)
        scheduler_t_max = config.max_epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_t_max)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        # EMA mirrors all params from the start (freeze only gates gradient flow).
        self.ema = ModelEMA(model, decay=config.ema_decay) if config.use_ema else None

        if resume_state is not None:
            start_epoch = int(resume_state["epoch"])
            # Replay the gradual-unfreeze reconstruction so the optimizer param_groups
            # match the epoch we resume INTO, BEFORE loading optimizer/scheduler state.
            skip_opt_load = False
            if start_epoch >= 1:
                if self.use_gradual_unfreeze:
                    unfreeze_stage(model, _NUM_STAGES - 1)  # epoch-1 transition
                    for e in range(2, start_epoch + 1):
                        si = _NUM_STAGES - e
                        if 0 <= si < _NUM_STAGES:
                            unfreeze_stage(model, si)
                else:
                    unfreeze_backbone(model)
                    optimizer = bt._make_optimizer(model, config)
                    scheduler_t_max = config.max_epochs - 1
                    scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_t_max)
                    # Resuming INTO epoch 1: optimizer/scheduler are freshly created
                    # here exactly as the uninterrupted run does — the backup holds
                    # the now-discarded frozen optimizer, so start fresh.
                    skip_opt_load = start_epoch == 1
            if not skip_opt_load:
                restore_optimizer_state(resume_state, optimizer, scheduler, device)
            if self.ema is not None and resume_state.get("ema") is not None:
                self.ema.load_full_state_dict(resume_state["ema"])
        return optimizer, scheduler, scheduler_t_max

    def resumed_message(self, ctx: TaskContext, best: BestTracker, global_step: int, start_epoch: int) -> dict:
        return {
            "type": "training_resumed", "resumed_from": str(self.config.resume_from),
            "epoch": start_epoch, "global_step": global_step,
            "best_val_f1": best.best_validation_score, "best_epoch": best.best_epoch + 1,
        }

    # -- per-epoch ---------------------------------------------------------
    def on_epoch_start(self, ctx, model, epoch, *, optimizer, scheduler, scheduler_t_max, start_epoch):
        # The resumed epoch's reconstruction already ran in create_optimizer.
        if ctx.resume_state is not None and epoch == start_epoch:
            return None
        config = self.config
        if epoch == 1:
            if self.use_gradual_unfreeze:
                unfreeze_stage(model, _NUM_STAGES - 1)
                return None
            # Non-gradual: unfreeze everything + a fresh optimizer/scheduler for the
            # remaining epochs (the epoch-0 frozen optimizer is discarded).
            unfreeze_backbone(model)
            optimizer = bt._make_optimizer(model, config)
            scheduler_t_max = config.max_epochs - 1
            scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_t_max)
            return optimizer, scheduler, scheduler_t_max
        if epoch > 1 and self.use_gradual_unfreeze:
            stage_idx = _NUM_STAGES - epoch  # 3, 2, 1, 0
            if 0 <= stage_idx < _NUM_STAGES:
                unfreeze_stage(model, stage_idx)
        return None

    def build_loaders(self, ctx: TaskContext, epoch: int, eff_bs: int, resume_info: ResumeInfo):
        config = self.config
        if self._loader_kwargs is None:
            self._loader_kwargs = loader_kwargs(config.dataloader_workers, prefetch_factor=3)
        # Val loader is static — build it once, on the first epoch of the run.
        if self._val_loader is None:
            val_sampler = build_bucket_batch_sampler(self.val_ds, batch_size=eff_bs)
            self._val_loader = DataLoader(
                self.val_ds, batch_sampler=val_sampler,
                collate_fn=bt._collate_bucket_batch, **self._loader_kwargs,
            )
        if self._first_build:
            self._first_build = False
            ctx.cb({
                "type": "training_progress", "stage": "training",
                "status_text": f"Training (epoch {epoch}/{config.max_epochs})",
                "epoch": epoch, "max_epochs": config.max_epochs,
            })
        # Resample cross-concept negatives so the model sees different negatives
        # each epoch (no-op for legacy per-concept negatives).
        if epoch > 0:
            self.train_ds.resample_negatives()
        train_loader = DataLoader(
            self.train_ds,
            batch_sampler=build_bucket_batch_sampler(self.train_ds, batch_size=eff_bs),
            collate_fn=bt._collate_bucket_batch, **self._loader_kwargs,
        )
        return train_loader, None, 0

    def make_step_callback(self, ctx: TaskContext, epoch: int, eff_bs: int, best: BestTracker, epoch_start_mono: float):
        self._epoch_start_mono = epoch_start_mono
        config = self.config
        cb = ctx.cb

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
                "best_val_f1": best.best_validation_score if best.best_validation_score >= 0 else None,
                "best_epoch": best.best_epoch + 1 if best.best_validation_score >= 0 else None,
            })

        return _on_step

    def train_epoch(self, ctx: TaskContext, model, optimizer, train_loader, *, step_callback, boundary_hook, start_batch: int):
        config = self.config
        return bt.train_one_epoch(
            model, train_loader, optimizer, self.criterion, ctx.device, ctx.dtype,
            step_callback=step_callback,
            stop_now_event=ctx.stop_now_event,
            ema=self.ema,
            randaugment_n=config.randaugment_n,
            randaugment_m=config.randaugment_m,
            random_erasing_p=config.random_erasing_p,
            boundary_hook=boundary_hook,
        )

    def validate(self, ctx: TaskContext, model, epoch: int, train_result) -> dict:
        # Validate against EMA weights when enabled — they generalise better.
        eval_model = self.ema.module if self.ema is not None else model
        val_result = bt.evaluate(eval_model, self._val_loader, self.criterion, ctx.device, ctx.dtype)
        # Select on F1 at the tuned threshold (what inference ships), not @0.5.
        metrics, _epoch_threshold = bt._tuned_val_metrics(val_result)
        metrics["val_loss"] = val_result["val_loss"]
        metrics["train_loss"] = train_result
        return metrics

    def selection_score(self, metrics: dict) -> float:
        return float(metrics.get("f1", 0.0))

    def save_candidate(self, ctx: TaskContext, model, epoch: int, metrics: dict, best: BestTracker) -> None:
        config = self.config
        best.best_metrics = metrics.copy()
        ckpt_path = ctx.checkpoint_dir / "candidate.pt"
        primary_state = self.ema.state_dict() if self.ema is not None else model.state_dict()
        ckpt_meta = {
            "state_dict": primary_state,
            "num_classes": 2,
            "model_size": config.model_size,
        }
        if self.ema is not None:
            ckpt_meta["model_state_dict"] = model.state_dict()
            ckpt_meta["ema_decay"] = config.ema_decay
        torch.save(ckpt_meta, ckpt_path)
        best.best_checkpoint_path = str(ckpt_path)

    def epoch_message(self, ctx: TaskContext, epoch: int, metrics: dict, train_result, selected_score: float, best: BestTracker) -> dict:
        config = self.config
        val_f1 = metrics.get("f1", 0.0)
        return {
            "type": "epoch_complete",
            "stage": "training",
            "status_text": f"Epoch {epoch + 1}/{config.max_epochs} complete (val F1 {val_f1:.3f})",
            "epoch": epoch + 1,
            "max_epochs": config.max_epochs,
            "train_loss": train_result,
            "val_loss": metrics.get("val_loss"),
            "val_f1": val_f1,
            "val_precision": metrics.get("precision", 0.0),
            "val_recall": metrics.get("recall", 0.0),
            "val_auprc": metrics.get("auprc", 0.0),
            "best_val_f1": best.best_validation_score,
            "best_epoch": best.best_epoch + 1,
        }

    def collect_extra_state(self, ctx: TaskContext, *, rng_epoch_start, schedule, batch_in_epoch: int) -> dict:
        # Binary is epoch-restart: no schedule / RNG replay is stored (the
        # per-epoch scheduler keeps a restarted epoch consistent).
        return {
            "use_gradual_unfreeze": self.use_gradual_unfreeze,
            "ema": self.ema.full_state_dict() if self.ema is not None else None,
        }

    # -- finalisation ------------------------------------------------------
    def finalize(self, ctx: TaskContext, model, best: BestTracker, epochs_completed: int) -> dict:
        config = self.config
        existing_best = ctx.checkpoint_dir / config.best_model_name
        return bt._binary_compare_promote(
            config,
            best_checkpoint_path=best.best_checkpoint_path,
            existing_best=existing_best,
            model=model,
            val_loader=self._val_loader,
            criterion=self.criterion,
            device=ctx.device,
            dtype=ctx.dtype,
            best_val_f1=best.best_validation_score,
            best_metrics=best.best_metrics,
            best_epoch=best.best_epoch,
            epochs_completed=epochs_completed,
            num_positives=self.num_positives,
            train_ds=self.train_ds,
        )
