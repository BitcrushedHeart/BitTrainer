"""Group multi-class training as a :class:`TrainingTask` (Bitcrush ISSUE-0542 Step 3).

Every block that left ``run_group_training`` lives here, moved verbatim: the loss
zoo / soft targets, cached-feature warmup + sweeps, autobatch, class-balance /
DCW / MixUp / SWA setup, the per-epoch shipped-decode validation (incl. Skin
Tone V2 dual-view), greedy-soup pooling, and the SWA / soup / promotion
finalisation. All the group helpers still live in ``bittrainer.group_trainer``
and are reached through the ``gt`` module alias so their existing test/monkeypatch
seams (``gt._train_one_epoch`` / ``gt._compare_promote_finalize`` /
``gt._warmup_head_probe`` / ``gt._create_or_warmstart_model``) keep firing.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import bittrainer.group_trainer as gt
from bittrainer.dynamic_class_weights import DynamicClassWeightController
from bittrainer.ema import ModelEMA
from bittrainer.generic.task import BestTracker, LoopSpec, ResumeInfo, TaskContext, TrainingTask
from bittrainer.group_dataset import build_group_bucket_sampler
from bittrainer.model import create_model, unfreeze_backbone
from bittrainer.model_soup import greedy_soup
from bittrainer.progress import ProgressEmitter, Stage
from bittrainer.runtime import configure_cuda_backend, maybe_compile, prewarm_compile
from bittrainer.smart_cache import _noop_callback
from bittrainer.training_state import (
    _FixedBatchSampler,
    capture_rng_states,
    init_backup,
    loader_kwargs,
    restore_optimizer_state,
    restore_rng_states,
)

logger = logging.getLogger(__name__)


class GroupTask(TrainingTask):
    """Drives ``GenericTrainer`` for a single ConvNeXt V2 multi-class group."""

    trainer_name = "group"

    def __init__(self, config: gt.GroupTrainConfig) -> None:
        self.config = config
        # Populated across the lifecycle hooks (see each hook below).
        self.train_ds = None
        self.val_ds = None
        self.smart_cache = None
        self.bucket_counts: dict = {}
        self.use_soft = False
        self.head_hidden_size: int | None = None
        self.memory_format = None
        self.eff_bs = 0
        self.class_counts: dict = {}
        self.total_raw = 0
        self.class_weights: torch.Tensor | None = None
        self.dcw_controller: DynamicClassWeightController | None = None
        self.mixup_enabled = False
        self.swa = None
        self.swa_start_epoch = 0
        self.ema: ModelEMA | None = None
        self.fwd_model = None
        self.scheduler_t_max = 0
        self.soup_pool: list[tuple[float, int, str]] = []
        self.soup_dir: Path | None = None
        self._val_loader = None
        self._val_sampler = None
        self._collate_fn = None
        self._epoch_start_mono = 0.0

    # -- one-time setup ----------------------------------------------------
    def make_context(self, progress_callback, stop_event, stop_now_event, pause_event) -> TaskContext:
        config = self.config
        em = ProgressEmitter(progress_callback or config.progress_callback or _noop_callback)
        device = torch.device(config.device)
        dtype = gt._get_dtype(config.dtype)
        configure_cuda_backend()
        group_folder = Path(config.group_folder)
        checkpoint_dir = (
            Path(config.checkpoint_dir) if config.checkpoint_dir else group_folder / "checkpoints"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return TaskContext(
            device=device, dtype=dtype, em=em, cb=em.raw, checkpoint_dir=checkpoint_dir,
            stop_event=stop_event, stop_now_event=stop_now_event, pause_event=pause_event,
        )

    def fingerprint_init(self, ctx: TaskContext) -> None:
        config = self.config
        coordinator, fingerprint, resume_state = init_backup(
            config, ctx.pause_event, ctx.cb,
            class_names=config.class_names, num_classes=config.num_classes,
            max_epochs=config.max_epochs, multi_label=config.multi_label,
            ordinal=config.ordinal, best_model_name=config.best_model_name,
            model_size=config.backbone_variant,
        )
        if resume_state is not None:
            # Re-apply the sweep outcomes the interrupted run resolved (the sweeps
            # themselves are skipped) before anything reads label_smoothing /
            # ordinal_sigma / oversample_none / class_balance_mode.
            gt._apply_resolved(config, resume_state.get("resolved") or {})
            ctx.em.stage(Stage.resuming, f"Resuming from backup (epoch {resume_state.get('epoch')})")
        ctx.coordinator = coordinator
        ctx.fingerprint = fingerprint
        ctx.resume_state = resume_state

    def loop_spec(self) -> LoopSpec:
        return LoopSpec(
            max_epochs=self.config.max_epochs,
            patience=self.config.patience,
            selection_min_delta=gt._SELECTION_MIN_DELTA,
        )

    def prepare_data(self, ctx: TaskContext) -> None:
        config = self.config
        self.use_soft = (
            config.ordinal
            or bool(config.soft_aliases)
            or bool(config.class_similarity_centroids)
            or (not config.multi_label and config.label_smoothing > 0)
        )
        ctx.em.stage(Stage.scanning, "Scanning dataset")
        self.train_ds, self.val_ds, self.smart_cache, self.bucket_counts = (
            gt._prepare_datasets_and_cache(config, cb=ctx.cb, stop_event=ctx.stop_event)
        )

    def create_model(self, ctx: TaskContext, resume_state: dict | None):
        config = self.config
        device, dtype = ctx.device, ctx.dtype
        self.head_hidden_size = config.probe_mlp_hidden if config.probe_head == "mlp" else None
        self.memory_format = torch.channels_last if config.channels_last else None

        if resume_state is None:
            gt._emit_model_load_stage(ctx.em, config, ctx.checkpoint_dir)
            model = gt._create_or_warmstart_model(
                config, device=device, dtype=dtype,
                head_hidden_size=self.head_hidden_size, checkpoint_dir=ctx.checkpoint_dir,
            )
            if self.memory_format is not None:
                model = model.to(memory_format=self.memory_format)
            return model

        # Resume: rebuild the architecture directly and load the backed-up
        # weights (skip warm-start, warmup probe and the sweeps entirely).
        model = create_model(
            model_size=config.backbone_variant, pretrained=False,
            num_classes=config.num_classes, head_hidden_size=self.head_hidden_size,
        )
        if config.cell_masks:
            from bittrainer.spatial import install_spatial_head

            install_spatial_head(model, config.cell_masks, config.grid_rows * config.grid_cols)
        model.load_state_dict(resume_state["model"])
        model = model.to(device)
        if self.memory_format is not None:
            model = model.to(memory_format=self.memory_format)
        unfreeze_backbone(model)
        return model

    def pre_loop(self, ctx: TaskContext, model) -> None:
        # Head warmup on cached features (replaces the fixed 1-epoch frozen
        # warmup), then fine-tune fully unfrozen. A converged head removes the
        # feature-distortion risk a random head poses.
        gt._warmup_head_probe(
            model, self.config, self.train_ds, self.val_ds, self.smart_cache,
            device=ctx.device, dtype=ctx.dtype, cb=ctx.cb,
            stop_event=ctx.stop_event, stop_now_event=ctx.stop_now_event,
        )
        unfreeze_backbone(model)  # the probe froze the backbone — restore full grad

    def resolve_batch_size(self, ctx: TaskContext, model, resume_state: dict | None) -> int:
        config = self.config
        train_ds = self.train_ds
        # The warmup oversample sweep may have flipped config.oversample_none;
        # rebuild the train set (and bucket histogram) so the full fine-tune trains
        # on the chosen __none__ composition. No-op when the sweep was off/undecided.
        if config.oversample_none != train_ds.oversample_none:
            train_ds.set_oversample_none(config.oversample_none)
            self.bucket_counts = {}
            for s in train_ds.samples:
                b = s["bucket"]
                self.bucket_counts[b] = self.bucket_counts.get(b, 0) + 1

        if resume_state is not None:
            # Resume: reuse the backed-up batch size (skip the probe). If the caller
            # now forces a different batch_size (Engine's OOM degrade halves it) the
            # backed-up batch_schedule no longer maps, so honour the new size and
            # fall back to epoch-restart resume (schedule discarded).
            backup_bs = int(resume_state["eff_bs"])
            resume_bs_changed = False
            if config.batch_size and int(config.batch_size) > 0 and int(config.batch_size) != backup_bs:
                eff_bs = int(config.batch_size)
                resume_bs_changed = True
                logger.info("Resume batch size changed %d -> %d; epoch-restart resume", backup_bs, eff_bs)
            else:
                eff_bs = backup_bs
            resume_state["_resume_bs_changed"] = resume_bs_changed
            ctx.cb({"type": "autobatch", "batch_size": eff_bs, "resumed": True})
        elif config.batch_size is not None and config.batch_size > 0:
            eff_bs = int(config.batch_size)
            ctx.cb({"type": "autobatch", "batch_size": eff_bs, "manual_override": True})
        else:
            from bittrainer.autobatch import determine_batch_size

            def _probe_progress(attempt: int, candidate: int, cap: int, status: str) -> None:
                ctx.cb({
                    "type": "training_progress", "stage": "autobatch",
                    "status_text": f"Probing batch size (try {attempt}: {candidate}/{cap} — {status})",
                })

            ctx.em.stage(Stage.autobatch, "Probing optimal batch size")
            auto_result = determine_batch_size(
                model, self.bucket_counts, ctx.device, dtype=ctx.dtype,
                vram_fraction=config.vram_fraction, use_ema=config.use_ema,
                memory_format=self.memory_format, progress_callback=_probe_progress,
            )
            eff_bs = auto_result["batch_size"]
            ctx.cb({"type": "autobatch", **auto_result})

        self.eff_bs = eff_bs
        return eff_bs

    def setup_training(self, ctx: TaskContext, model, resume_state: dict | None) -> None:
        config = self.config
        device = ctx.device
        train_ds = self.train_ds
        self.class_counts = train_ds.get_class_counts()
        self.total_raw = sum(self.class_counts.values())

        # --- Class imbalance strategy: resample vs reweight (mutually exclusive). ---
        balance_mode = gt._resolve_class_balance(config, self.class_counts)
        class_weights: torch.Tensor | None = None
        if not config.multi_label and balance_mode == "reweight":
            train_ds.set_natural_sampling(True)
            class_weights = gt._effective_number_weights(
                self.class_counts, config.num_classes, config.class_balance_beta, device,
            )
            ctx.cb({
                "type": "training_progress", "stage": "preparing",
                "status_text": "Class balance: reweight (natural sampling + effective-number weights)",
            })

        # --- Dynamic per-class loss weighting (single-label only). ---
        dcw_controller = gt._build_dcw_controller(config, class_weights, device)
        if dcw_controller is not None:
            if resume_state is not None and resume_state.get("dcw") is not None:
                base = class_weights if class_weights is not None else torch.ones(config.num_classes, device=device)
                dcw_controller = DynamicClassWeightController.from_dict(resume_state["dcw"], base)
            class_weights = dcw_controller.current_weights()
            ctx.cb({
                "type": "training_progress", "stage": "preparing",
                "status_text": (
                    f"Dynamic per-class loss weighting ON "
                    f"(trigger={config.dcw_metric}, patience={config.dcw_patience})"
                ),
            })
        if resume_state is not None and resume_state.get("class_weights") is not None and dcw_controller is None:
            class_weights = resume_state["class_weights"].to(device)

        self.class_weights = class_weights
        self.dcw_controller = dcw_controller

        # --- MixUp/CutMix gate: skip on tiny datasets and for multi-label. ---
        self.mixup_enabled = (
            config.use_mixup and not config.multi_label and self.total_raw >= config.mixup_min_images
        )

        # --- SWA: average weights over the cosine tail. ---
        self.swa = gt._SWA() if (config.use_swa and not config.multi_label) else None
        self.swa_start_epoch = int(config.swa_start_frac * config.max_epochs)
        if self.swa is not None and resume_state is not None and resume_state.get("swa") is not None:
            self.swa.load_state_dict(resume_state["swa"]["avg"], resume_state["swa"]["n"])

    def create_optimizer(self, ctx: TaskContext, model, eff_bs: int, resume_state: dict | None):
        config = self.config
        device, dtype = ctx.device, ctx.dtype
        optimizer = gt._make_optimizer(model, config)
        # The group trainer never recreates the scheduler mid-run, so T_max is
        # always config.max_epochs; carried in the backup for schema symmetry.
        scheduler_t_max = int(resume_state["scheduler_t_max"]) if resume_state is not None else config.max_epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_t_max)

        # EMA tracks all params from the start; freeze/unfreeze only affects which
        # ones receive gradient updates.
        self.ema = ModelEMA(model, decay=config.ema_decay) if config.use_ema else None

        if resume_state is not None:
            restore_optimizer_state(resume_state, optimizer, scheduler, device)
            if self.ema is not None and resume_state.get("ema") is not None:
                self.ema.load_full_state_dict(resume_state["ema"])

        # fwd_model shares parameters with the eager model — optimizer, EMA and
        # checkpoint saves keep operating on `model`; only forward calls go through
        # the compiled wrapper.
        fwd_model, compiled = maybe_compile(model, enabled=config.use_compile, cb=ctx.cb)
        if compiled and not prewarm_compile(
            fwd_model, self.bucket_counts, eff_bs, device, dtype,
            memory_format=self.memory_format, cb=ctx.cb,
        ):
            fwd_model = model
        self.fwd_model = fwd_model
        self.scheduler_t_max = scheduler_t_max

        self.soup_pool = []
        self.soup_dir = ctx.checkpoint_dir / "soup_cands"
        return optimizer, scheduler, scheduler_t_max

    def restore_resume_extra(self, ctx: TaskContext, resume_state: dict) -> None:
        self.soup_pool = [tuple(t) for t in (resume_state.get("soup_pool") or [])]

    def resumed_message(self, ctx: TaskContext, best: BestTracker, global_step: int, start_epoch: int) -> dict:
        config = self.config
        return {
            "type": "training_resumed",
            "resumed_from": str(config.resume_from),
            "epoch": start_epoch,
            "global_step": global_step,
            "best_val_macro_f1": best.best_val_macro_f1,
            "best_validation_score": best.best_validation_score,
            "best_val_qwk": best.best_val_qwk if config.ordinal else None,
            "best_epoch": best.best_epoch + 1,
        }

    # -- per-epoch ---------------------------------------------------------
    def reshuffle(self) -> None:
        self.train_ds.reshuffle()

    def build_loaders(self, ctx: TaskContext, epoch: int, eff_bs: int, resume_info: ResumeInfo):
        config = self.config
        device = ctx.device
        collate_fn = gt._collate_multilabel_batch if config.multi_label else gt._collate_bucket_batch
        train_sampler = build_group_bucket_sampler(self.train_ds, batch_size=eff_bs)
        if resume_info.mid_resume:
            schedule = [list(b) for b in resume_info.resume_schedule]
            loader_batches = schedule[resume_info.resume_batch_in_epoch:]
            start_batch = resume_info.resume_batch_in_epoch
            # Jump the augmentation/mixup RNG to the mid-epoch backup point.
            restore_rng_states(resume_info.resume_rng_now, device)
        else:
            schedule = [list(b) for b in train_sampler]
            loader_batches = schedule
            start_batch = 0
        lk = loader_kwargs(config.dataloader_workers)
        if lk["num_workers"] == 0:
            # workers=0 (bit-exact resume mode): keep the DataLoader base-seed draw
            # off the global torch RNG so it stays purely augmentation-driven.
            lk["generator"] = torch.Generator().manual_seed(0)
        train_loader = DataLoader(
            self.train_ds, batch_sampler=_FixedBatchSampler(loader_batches),
            collate_fn=collate_fn, **lk,
        )
        val_sampler = build_group_bucket_sampler(self.val_ds, batch_size=eff_bs)
        val_loader = DataLoader(
            self.val_ds, batch_sampler=val_sampler, collate_fn=collate_fn, **lk,
        )
        self._collate_fn = collate_fn
        self._val_loader = val_loader
        self._val_sampler = val_sampler
        return train_loader, schedule, start_batch

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
                "images_per_s": round(throughput * eff_bs, 1) if throughput else None,
                "batch_size": eff_bs,
                "train_loss": round(avg_loss, 4),
                "best_val_macro_f1": best.best_val_macro_f1 if best.best_val_macro_f1 >= 0 else None,
                "best_validation_score": best.best_validation_score if best.best_validation_score >= 0 else None,
                "validation_metric": gt._primary_validation_metric(config),
                "best_val_qwk": (
                    best.best_val_qwk if config.ordinal and best.best_val_qwk > -1.0 else None
                ),
                "best_epoch": best.best_epoch + 1 if best.best_val_macro_f1 >= 0 else None,
            })

        return _on_step

    def train_epoch(self, ctx: TaskContext, model, optimizer, train_loader, *, step_callback, boundary_hook, start_batch: int):
        config = self.config
        return gt._train_one_epoch(
            self.fwd_model, train_loader, optimizer, config, ctx.device, ctx.dtype,
            use_soft_targets=self.use_soft,
            step_callback=step_callback,
            stop_now_event=ctx.stop_now_event,
            ema=self.ema,
            class_weights=self.class_weights,
            mixup_enabled=self.mixup_enabled,
            pause_event=ctx.pause_event,
            boundary_hook=boundary_hook,
            start_batch=start_batch,
        )

    def on_after_train(self, ctx: TaskContext, model, epoch: int) -> None:
        # Capture the post-epoch weights into the SWA running average over the
        # cosine tail (uniform average; LayerNorm backbone needs no BN update).
        if self.swa is not None and epoch >= self.swa_start_epoch:
            self.swa.update(model)

    def validate(self, ctx: TaskContext, model, epoch: int, train_result) -> dict:
        config = self.config
        device, dtype = ctx.device, ctx.dtype
        none_index = gt._resolve_none_index(config.class_names)
        # Validate against EMA weights when enabled — they generalise better.
        eval_model = self.ema.module if self.ema is not None else self.fwd_model
        if config.multi_label:
            val_metrics = gt._evaluate(
                eval_model, self._val_loader, config.num_classes, device, dtype,
                multi_label=True, ordinal=config.ordinal,
                none_index=none_index, channels_last=config.channels_last,
            )
        else:
            # Score the epoch under the decode the model ships with (temperature +
            # __none__ bias + ordinal cut-points) so selection and the shipped model
            # agree on what "best" means.
            epoch_logits, epoch_labels = gt._collect_val_logits(
                eval_model, self._val_loader, config, device, dtype,
            )
            val_metrics = gt._shipped_decode_metrics(epoch_logits, epoch_labels, config, none_index)
            # Skin Tone V2 dual-view (ISSUE-0217, spec §8): score the
            # colour-normalised view and the averaged-logit combination as separate
            # tracks. Selection stays on the ORIGINAL view.
            if getattr(self.val_ds, "skin_tone_views", None) is not None:
                self.val_ds.skin_tone_force_view = True
                try:
                    # A FRESH in-process loader is load-bearing: the persistent
                    # workers hold a dataset copy pickled before the flag flip.
                    view_loader = DataLoader(
                        self.val_ds, batch_sampler=self._val_sampler, collate_fn=self._collate_fn,
                        num_workers=0, pin_memory=True,
                    )
                    view_logits, view_labels = gt._collect_val_logits(
                        eval_model, view_loader, config, device, dtype,
                    )
                finally:
                    self.val_ds.skin_tone_force_view = False
                view_metrics = gt._shipped_decode_metrics(view_logits, view_labels, config, none_index)
                val_metrics["macro_f1_original"] = val_metrics["macro_f1"]
                val_metrics["macro_f1_normalized"] = view_metrics["macro_f1"]
                if torch.equal(epoch_labels, view_labels):
                    dual_metrics = gt._shipped_decode_metrics(
                        (epoch_logits + view_logits) / 2.0, epoch_labels, config, none_index,
                    )
                    val_metrics["macro_f1_dual"] = dual_metrics["macro_f1"]
        val_metrics["train_loss"] = train_result[0]
        return val_metrics

    def selection_score(self, metrics: dict) -> float:
        return gt._metric_score(metrics, self.config)

    def save_candidate(self, ctx: TaskContext, model, epoch: int, metrics: dict, best: BestTracker) -> None:
        config = self.config
        best.best_val_macro_f1 = metrics["macro_f1"]
        best.best_val_qwk = metrics.get("qwk", 0.0)
        best.best_metrics = metrics.copy()

        ckpt_path = ctx.checkpoint_dir / "candidate.pt"
        # When EMA is active, persist the EMA weights as the primary state_dict.
        primary_state = self.ema.state_dict() if self.ema is not None else model.state_dict()
        ckpt_meta = {
            "state_dict": primary_state,
            "num_classes": config.num_classes,
            "model_size": config.backbone_variant,
            "class_names": list(config.class_names),
            "validation_metric": gt._primary_validation_metric(config),
            **gt._spatial_ckpt_meta(config),
        }
        if self.head_hidden_size is not None:
            ckpt_meta["head_hidden_size"] = self.head_hidden_size
        if self.ema is not None:
            ckpt_meta["model_state_dict"] = model.state_dict()
            ckpt_meta["ema_decay"] = config.ema_decay
        if config.multi_label:
            ckpt_meta["multi_label"] = True
        torch.save(ckpt_meta, ckpt_path)
        best.best_checkpoint_path = str(ckpt_path)

    def on_epoch_end(self, ctx: TaskContext, model, epoch: int, metrics: dict, selected_score: float, best: BestTracker) -> None:
        config = self.config
        # Dynamic per-class loss weighting: fold this epoch's per-class val signal
        # into the controller and reassign class_weights for the NEXT epoch.
        if self.dcw_controller is not None:
            self.class_weights = self.dcw_controller.update(
                metrics.get("per_class_f1", {}), metrics.get("per_class_val_loss", {}),
            )

        # Per-epoch snapshot dump for snapshot-ensemble experiments. Off by default.
        if config.snapshot_dir:
            snap_dir = Path(config.snapshot_dir)
            snap_dir.mkdir(parents=True, exist_ok=True)
            snap_state = self.ema.state_dict() if self.ema is not None else model.state_dict()
            snap_meta = {
                "state_dict": snap_state,
                "num_classes": config.num_classes,
                "model_size": config.backbone_variant,
                "class_names": list(config.class_names),
                **gt._spatial_ckpt_meta(config),
            }
            if self.head_hidden_size is not None:
                snap_meta["head_hidden_size"] = self.head_hidden_size
            torch.save(snap_meta, snap_dir / f"epoch_{epoch + 1:03d}.pt")

        # Track this epoch as a greedy-soup candidate (top-N by selection score).
        if config.use_greedy_soup:
            gt._update_soup_pool(
                self.soup_pool, self.soup_dir, selected_score, epoch,
                self.ema.state_dict() if self.ema is not None else model.state_dict(),
                config.soup_max_candidates,
            )

    def epoch_message(self, ctx: TaskContext, epoch: int, metrics: dict, train_result, selected_score: float, best: BestTracker) -> dict:
        return gt._build_epoch_message(
            epoch=epoch,
            config=self.config,
            train_loss=train_result[0],
            val_metrics=metrics,
            best_val_macro_f1=best.best_val_macro_f1,
            best_val_qwk=best.best_val_qwk,
            selected_score=selected_score,
            best_validation_score=best.best_validation_score,
            best_epoch=best.best_epoch,
            per_class_train_loss=train_result[1],
            elapsed_seconds=time.monotonic() - self._epoch_start_mono,
            dcw_multipliers=(
                self.dcw_controller.multipliers() if self.dcw_controller is not None else None
            ),
        )

    def collect_extra_state(self, ctx: TaskContext, *, rng_epoch_start, schedule, batch_in_epoch: int) -> dict:
        swa_payload = None
        if self.swa is not None and self.swa.state_dict() is not None:
            swa_payload = {"avg": self.swa.state_dict(), "n": self.swa.n}
        return {
            "ema": self.ema.full_state_dict() if self.ema is not None else None,
            "swa": swa_payload,
            "soup_pool": [list(t) for t in self.soup_pool],
            "dcw": self.dcw_controller.to_dict() if self.dcw_controller is not None else None,
            "class_weights": self.class_weights.detach().cpu() if self.class_weights is not None else None,
            "resolved": gt._resolved_snapshot(self.config),
            "rng_epoch_start": rng_epoch_start,
            "rng_now": capture_rng_states(ctx.device),
            "batch_schedule": (
                [list(b) for b in schedule]
                if (schedule is not None and batch_in_epoch > 0) else None
            ),
            "head_hidden_size": self.head_hidden_size,
        }

    # -- finalisation ------------------------------------------------------
    def finalize(self, ctx: TaskContext, model, best: BestTracker, epochs_completed: int) -> dict:
        config = self.config
        device, dtype = ctx.device, ctx.dtype
        val_loader = self._val_loader
        cb = ctx.cb
        none_index = gt._resolve_none_index(config.class_names)

        # --- SWA finalisation: materialise the averaged weights, evaluate them, and
        # adopt as the candidate only when they beat the best single-epoch checkpoint. ---
        if self.swa is not None and self.swa.n >= 2 and best.best_checkpoint_path:
            try:
                swa_sd_cpu = self.swa.state_dict()
                model.load_state_dict({k: v.to(device) for k, v in swa_sd_cpu.items()})
                swa_logits, swa_labels = gt._collect_val_logits(model, val_loader, config, device, dtype)
                swa_metrics = gt._shipped_decode_metrics(swa_logits, swa_labels, config, none_index)
                swa_score = gt._metric_score(swa_metrics, config)
                cb({
                    "type": "training_progress", "stage": "validating",
                    "status_text": (
                        f"SWA ({self.swa.n} snapshots): score {swa_score:.4f} "
                        f"vs best {best.best_validation_score:.4f}"
                    ),
                })
                if swa_score > best.best_validation_score:
                    ckpt_meta = {
                        "state_dict": swa_sd_cpu,
                        "num_classes": config.num_classes,
                        "model_size": config.backbone_variant,
                        "class_names": list(config.class_names),
                        "validation_metric": gt._primary_validation_metric(config),
                        **gt._spatial_ckpt_meta(config),
                    }
                    if self.head_hidden_size is not None:
                        ckpt_meta["head_hidden_size"] = self.head_hidden_size
                    ckpt_path = ctx.checkpoint_dir / "candidate.pt"
                    torch.save(ckpt_meta, ckpt_path)
                    best.best_checkpoint_path = str(ckpt_path)
                    best.best_metrics = swa_metrics.copy()
                    best.best_val_macro_f1 = swa_metrics["macro_f1"]
                    best.best_val_qwk = swa_metrics.get("qwk", 0.0)
                    best.best_validation_score = swa_score
                    logger.info("SWA weights adopted (score %.4f)", swa_score)
            except Exception:
                logger.warning("SWA evaluation failed; keeping best single-epoch checkpoint", exc_info=True)

        # --- Greedy weight soup: average the strongest epochs into ONE model. ---
        if config.use_greedy_soup and len(self.soup_pool) >= 2 and best.best_checkpoint_path:
            def _soup_metrics(state: dict) -> dict:
                model.load_state_dict({k: v.to(device) for k, v in state.items()})
                if config.multi_label:
                    return gt._evaluate(
                        model, val_loader, config.num_classes, device, dtype,
                        multi_label=True, ordinal=config.ordinal,
                        none_index=none_index, channels_last=config.channels_last,
                    )
                logits, labels = gt._collect_val_logits(model, val_loader, config, device, dtype)
                return gt._shipped_decode_metrics(logits, labels, config, none_index)

            try:
                candidates = [
                    (score, torch.load(path, map_location="cpu")) for score, _ep, path in self.soup_pool
                ]
                soup_state, soup_score, accepted = greedy_soup(
                    candidates, lambda s: gt._metric_score(_soup_metrics(s), config),
                )
                cb({
                    "type": "training_progress", "stage": "validating",
                    "status_text": (
                        f"Greedy soup ({len(accepted)}/{len(candidates)} epochs): "
                        f"score {soup_score:.4f} vs best {best.best_validation_score:.4f}"
                    ),
                })
                if soup_score > best.best_validation_score:
                    soup_metrics = _soup_metrics(soup_state)
                    ckpt_meta = {
                        "state_dict": {k: v.detach().cpu() for k, v in soup_state.items()},
                        "num_classes": config.num_classes,
                        "model_size": config.backbone_variant,
                        "class_names": list(config.class_names),
                        "validation_metric": gt._primary_validation_metric(config),
                        **gt._spatial_ckpt_meta(config),
                    }
                    if self.head_hidden_size is not None:
                        ckpt_meta["head_hidden_size"] = self.head_hidden_size
                    if config.multi_label:
                        ckpt_meta["multi_label"] = True
                    ckpt_path = ctx.checkpoint_dir / "candidate.pt"
                    torch.save(ckpt_meta, ckpt_path)
                    best.best_checkpoint_path = str(ckpt_path)
                    best.best_metrics = soup_metrics.copy()
                    best.best_val_macro_f1 = soup_metrics.get("macro_f1", best.best_val_macro_f1)
                    best.best_val_qwk = soup_metrics.get("qwk", best.best_val_qwk)
                    best.best_validation_score = soup_score
                    logger.info("Greedy soup adopted (%d epochs, score %.4f)", len(accepted), soup_score)
            except Exception:
                logger.warning("Greedy soup failed; keeping best single-epoch checkpoint", exc_info=True)
            finally:
                for _s, _e, _p in self.soup_pool:
                    try:
                        Path(_p).unlink()
                    except OSError:
                        pass
                try:
                    self.soup_dir.rmdir()
                except OSError:
                    pass

        return gt._compare_promote_finalize(
            config,
            candidate_path=best.best_checkpoint_path,
            best_metrics=best.best_metrics,
            candidate_macro_f1=best.best_val_macro_f1,
            candidate_qwk=best.best_val_qwk,
            best_epoch_display=best.best_epoch + 1,
            epochs_completed=epochs_completed,
            val_loader=val_loader,
            device=device, dtype=dtype,
            checkpoint_dir=ctx.checkpoint_dir,
            class_counts=self.train_ds.get_class_counts(),
            effective_class_counts=self.train_ds.get_effective_class_counts(),
            total_raw=self.total_raw,
            cb=cb,
        )
