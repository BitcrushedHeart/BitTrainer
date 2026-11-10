"""Backbone Builder training as a :class:`TrainingTask` (Bitcrush ISSUE-0542 Step 6).

Ports ``bittrainer.backbone_trainer._train_backbone`` onto the shared
:class:`~bittrainer.generic.generic_trainer.GenericTrainer` skeleton. The backbone
trainer keeps its own mechanics — one shared ConvNeXt V2 backbone with a supervised
head per binary concept (BCE) and per multi-class group (CE), masked losses, EMA on
by default, and a candidate exported as SAFETENSORS with Engine-readable metadata —
all expressed through the generic hooks. Backup / pause / resume is epoch-restart.

Three behaviour changes ride this migration:

1. **Prodigy_adv** (Kourkoutas-beta) replaces AdamW, built through the shared
   :func:`bittrainer.generic.optimizer.make_optimizer`. A **flat** param group is
   used deliberately: ``build_llrd_param_groups`` buckets by ``stem`` / ``stages.N``
   / ``head`` name prefixes, which the ``_BackboneWithHeads`` wrapper's
   ``backbone.*`` / ``heads.*`` prefixes do not expose, so LLRD would collapse every
   parameter into one bucket — flat is the simplest correct choice. ``learning_rate``
   is read + tolerated but inert (Prodigy runs at ``lr=1.0`` and the cosine schedule
   scales its adapted step ``d``). Resume primes Prodigy + restores the Kourkoutas
   aux accumulators via ``restore_optimizer_state``; the optimizer identity is folded
   into the fingerprint so a stale AdamW backup mismatches and starts fresh.
2. **Heads persist**: the best-epoch (EMA-consistent) head tensors are exported
   ``heads.*``-prefixed alongside the BARE backbone keys, with metadata that lets a
   consumer rebuild ``_MultiTaskHeads``.
3. **Dedup**: ``_build_samples`` collapses records sharing a ``content_hash`` and the
   preparing payload reports ``unique_images``.

The backbone helpers stay in ``bittrainer.backbone_trainer`` and are reached through
the ``bb`` module alias so their import + monkeypatch seams keep firing.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader

import bittrainer.backbone_trainer as bb
from bittrainer.amp import amp_label, make_grad_scaler
from bittrainer.backbone_init import apply_backbone_init, wants_timm_pretrained
from bittrainer.ema import ModelEMA
from bittrainer.generic.optimizer import (
    clip_gradients,
    make_optimizer,
    optimizer_identity,
)
from bittrainer.generic.task import (
    BestTracker,
    LoopSpec,
    ResumeInfo,
    TaskContext,
    TrainingTask,
)
from bittrainer.lean_grn import use_lean_grn
from bittrainer.model import create_model, default_drop_path_rate
from bittrainer.training_state import (
    BackupCoordinator,
    capture_rng_states,
    restore_optimizer_state,
)

logger = logging.getLogger(__name__)


class BackboneTask(TrainingTask):
    """Drives ``GenericTrainer`` for the multi-task ConvNeXt V2 backbone builder."""

    trainer_name = "backbone"
    # Full backbone builds fit + export val-tuned calibration (A2); the head-only
    # subclass turns this off (cached-vector calibration is a follow-up).
    _calibration_enabled = True

    def __init__(
        self,
        request: dict,
        *,
        cancel_event: threading.Event | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.request = request
        self.config = dict(request.get("training_config") or {})
        # Cancellation (raise BackboneTrainingCancelled) is distinct from the core's
        # stop_event: it propagates out instead of finalising. max_steps rides the
        # core's stop_event via ``steps_stop_event`` (graceful boundary stop).
        self.cancel_event = cancel_event
        # A caller-supplied ``stop_event`` (Bitcrush ISSUE-0554 finish-early)
        # IS the steps event, so the task's own max_steps stop and the user's
        # request share one graceful-boundary signal.
        self.steps_stop_event = (
            stop_event if stop_event is not None else threading.Event()
        )

        c = self.config
        self.image_size = int(c.get("image_size") or 384)
        # ``batch_size`` of 0 or "auto" opts into the autobatch probe (resolved in
        # resolve_batch_size); the sentinel 0 is carried until then so the "auto"
        # string never reaches an int() here. Absent/None keeps the legacy default 8.
        raw_bs = c.get("batch_size")
        self._autobatch = str(raw_bs).lower() == "auto" or (
            isinstance(raw_bs, (int, float)) and int(raw_bs) == 0
        )
        self.batch_size = 0 if self._autobatch else int(raw_bs or 8)
        self.epochs = int(c.get("epochs") or 10)
        self.max_steps = c.get("max_steps")
        # Read + tolerated, but inert under Prodigy_adv (lr=1.0).
        self.learning_rate = float(c.get("learning_rate") or 1e-4)
        self.validation_split = float(c.get("validation_split") or 0.15)
        self.model_size = request.get("convnextv2_size") or "nano"
        self.patience = int(c.get("patience") or c.get("early_stopping_patience") or 0)
        self.use_ema = bool(c.get("use_ema", True))
        self.ema_decay = float(c.get("ema_decay") or 0.9999)
        self.use_cosine = bool(c.get("use_cosine", True))
        self.amp_enabled, self.amp_dtype = bb._amp_settings(c)
        # fp16 runs step through a GradScaler (Bitcrush ISSUE-1655). It is built
        # in make_context, once the device and so its precision are known; it
        # stays None on bf16 and on a CPU.
        self._scaler = None
        # Opt-in speed knobs, both defaulting to the legacy behaviour: decode and
        # augment on the training thread, contiguous (NCHW) memory.
        self.dataloader_workers = int(c.get("dataloader_workers") or 0)
        self.channels_last = bool(c.get("channels_last", False))
        # GRN kept in the autocast precision (see bittrainer.lean_grn): the same
        # weights and the same full-precision results, 40% faster and 29% less
        # memory for Base in fp16 on a T4. ``lean_grn: False`` asks for timm's.
        self.lean_grn = bool(c.get("lean_grn", True))
        # Heartbeat inside an epoch: a ``training_step`` frame every N optimizer
        # steps (0 = none, the legacy behaviour). An epoch on a big pool runs for
        # an hour or more; without this a detached run cannot be told from a hang.
        self.progress_every_steps = int(c.get("progress_every_steps") or 0)
        # The last epoch's wall time and image count, surfaced on its frame.
        self._train_seconds = 0.0
        self._train_images = 0

        # Optimizer layout (A4/B7): LLRD + weight-decay exclusions ON by default for
        # the backbone builder (long-horizon trunk training). The identity folds
        # into the fingerprint so a stale flat backup mismatches cleanly.
        self.llrd = bool(c.get("llrd", True))
        self.llrd_decay = float(c.get("llrd_decay") or 0.8)
        self.wd_exclusions = bool(c.get("wd_exclusions", True))
        # Gradient clipping (B8): max total grad norm before each step; 0 disables.
        self.clip_grad_norm = float(c.get("clip_grad_norm", 1.0))
        # Stochastic depth (B6): absent/None -> size-scaled default (ON), explicit 0
        # -> off. Parameter-free, so it never perturbs resume/state-dict shape.
        raw_dp = c.get("drop_path_rate")
        self.drop_path_rate = (
            default_drop_path_rate(self.model_size) if raw_dp is None else float(raw_dp)
        )
        # Loss + augmentation parity (A3). New backbone defaults: mean loss over
        # heads-present, 0.1 group label smoothing; density-first positive cap kept
        # as the behaviour-preserving default.
        self.loss_reduction = str(c.get("loss_reduction") or "mean")
        self.group_label_smoothing = float(c.get("group_label_smoothing", 0.1))
        self.positive_cap_mode = str(c.get("positive_cap_mode") or "density")
        self._aug = {
            "use_random_resized_crop": bool(c.get("use_random_resized_crop", True)),
            "rrc_scale_min": float(c.get("rrc_scale_min", 0.6)),
            "randaugment_n": int(c.get("randaugment_n", 2)),
            "randaugment_m": int(c.get("randaugment_m", 9)),
            "random_erasing_p": float(c.get("random_erasing_p", 0.25)),
        }
        # Calibration export (A2): per-head thresholds on by default; temperature
        # opt-in (a second-order win that adds a consumer-side contract).
        self.calibrate_thresholds = bool(c.get("calibrate_thresholds", True))
        self.calibrate_temperature = bool(c.get("calibrate_temperature", False))
        # Raw val logits of the winning epoch (populated by validate/save_candidate)
        # so calibration fits on the exported epoch, not the last one.
        self._val_score_sink: dict = {}
        self._best_val_scores: dict | None = None

        # Sampling layer (Bitcrush ISSUE-0545/0546) — see bb._plan_epoch_samples.
        self.neg_pos_ratio = float(c.get("neg_pos_ratio", 5.0))
        self.label_policy = dict(c.get("label_policy") or {})
        self.use_pos_weight = bool(c.get("use_pos_weight", True))
        raw_positive_cap = c.get("max_positives_per_class")
        self.positive_cap = 1000 if raw_positive_cap is None else int(raw_positive_cap)
        self.oversample_positives = bool(c.get("oversample_positives", True))
        self.min_positive_threshold = int(c.get("min_positive_threshold") or 30)
        self.max_oversample_factor = float(c.get("max_oversample_factor") or 4.0)
        self.sampling_seed = int(c.get("sampling_seed") or 0)
        self._pos_weight: dict[str, float] | None = None

        # Resolution tail: last N epochs (train + val) at a higher size.
        self.finetune_image_size = int(c.get("finetune_image_size") or 0)
        self.finetune_epochs = int(c.get("finetune_epochs") or 0)
        self._val_loader_size: int | None = None

        # Populated across the lifecycle hooks.
        self.vocab: bb._Vocab | None = None
        self.train_samples: list = []
        self.val_samples: list = []
        self._unique_images = 0
        self.model = None
        self.ema: ModelEMA | None = None
        self.feature_dim = 0
        self.step = 0
        self.seq = 1
        self._cb = None
        self._val_loader = None
        self.best_backbone_state: dict | None = None
        self.best_heads_state: dict | None = None
        self.best_metrics: dict = {}
        self.best_score = -1.0
        self.validation_metrics: dict = {}
        self.validation_score = 0.0

        # max_steps <= 0 means "no optimiser step" — stop before the first epoch.
        if self.max_steps is not None and int(self.max_steps) <= 0:
            self.steps_stop_event.set()

    # -- helpers -----------------------------------------------------------
    def _emit(self, stage: str, status_text: str, **extra) -> None:
        self.seq += 1
        self._cb(
            {
                "type": "training_progress",
                "stage": stage,
                "status_text": status_text,
                "run_id": self.request.get("run_id"),
                "seq": self.seq,
                **extra,
            }
        )

    def _eval_modules(self):
        # Validate / snapshot against EMA weights when enabled (they generalise
        # better on small datasets), else the live model.
        src = self.ema.module if self.ema is not None else self.model
        return src.backbone, src.heads

    # -- one-time setup ----------------------------------------------------
    def make_context(
        self, progress_callback, stop_event, stop_now_event, pause_event
    ) -> TaskContext:
        from bittrainer.smart_cache import _noop_callback

        cb = progress_callback or _noop_callback
        self._cb = cb
        requested_device = self.config.get("device")
        device = torch.device(
            requested_device
            if requested_device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Fit the precision to the card now that it is known: bf16 where it is
        # native, fp16 through a GradScaler where it is not (a T4).
        self.amp_enabled, self.amp_dtype = bb._amp_settings(self.config, device)
        self._scaler = make_grad_scaler(device, self.amp_dtype, self.amp_enabled)

        # Samples are built here (not in prepare_data) because loop_spec — which the
        # core calls before prepare_data — needs to know whether a val split exists.
        records = list(self.request.get("records") or [])
        self.vocab = bb._Vocab(records)
        if not self.vocab.has_targets:
            raise RuntimeError(
                "Backbone training needs at least one labelled binary concept or "
                "group with 2+ classes; the dataset audit found none."
            )
        samples, missing = bb._build_samples(records, self.vocab)
        if not samples:
            raise RuntimeError("No labelled images with existing files to train on.")
        if missing:
            logger.warning(
                "Backbone training: %d records had no existing file on disk", missing
            )
        self.train_samples, self.val_samples = bb._split_samples(
            records, samples, self.validation_split
        )
        self._unique_images = len(samples)

        class _Emitter:
            """Backbone emits raw frames only; ``.stage`` is a no-op the core calls
            once per epoch (validating) which the backbone trainer never surfaced."""

            def __init__(self, raw) -> None:
                self.raw = raw

            def stage(self, *_a, **_k) -> None:
                pass

        checkpoint_dir = Path(self.request["candidate_checkpoint_path"]).parent
        return TaskContext(
            device=device,
            dtype=self.amp_dtype,
            em=_Emitter(cb),
            cb=cb,
            checkpoint_dir=checkpoint_dir,
            stop_event=stop_event,
            stop_now_event=stop_now_event,
            pause_event=pause_event,
        )

    def fingerprint_init(self, ctx: TaskContext) -> None:
        c = self.config
        coordinator = BackupCoordinator(
            backup_dir=c.get("backup_dir"),
            backup_every_steps=int(c.get("backup_every_steps") or 0),
            pause_event=ctx.pause_event,
            cb=ctx.cb,
        )
        fingerprint = bb._backbone_fingerprint(
            self.vocab,
            self.model_size,
            self.epochs,
            resolution=self._resolution_signature(),
            optimizer_identity=optimizer_identity(
                llrd=self.llrd,
                llrd_decay=self.llrd_decay,
                wd_exclusions=self.wd_exclusions,
            ),
        )
        resume_from = c.get("resume_from")
        resume_state = (
            coordinator.load_resume(fingerprint, resume_from=resume_from)
            if resume_from
            else None
        )
        ctx.coordinator = coordinator
        ctx.fingerprint = fingerprint
        ctx.resume_state = resume_state

    def loop_spec(self) -> LoopSpec:
        # Backbone selects on strict improvement in the mean validation metric (no
        # min-delta guard). Patience only bites with a val split; disabled/no-val
        # runs get an unreachable patience so they run every epoch (legacy shape).
        eff_patience = (
            self.patience
            if (self.patience > 0 and self.val_samples)
            else (self.epochs + 1)
        )
        return LoopSpec(
            max_epochs=self.epochs, patience=eff_patience, selection_min_delta=0.0
        )

    @property
    def _has_tail(self) -> bool:
        return self.finetune_image_size > 0 and 0 < self.finetune_epochs < self.epochs

    def _epoch_image_size(self, epoch: int) -> int:
        if self._has_tail and epoch >= self.epochs - self.finetune_epochs:
            return self.finetune_image_size
        return self.image_size

    def _resolution_signature(self) -> str:
        if self._has_tail:
            return (
                f"{self.image_size}->{self.finetune_image_size}@{self.finetune_epochs}"
            )
        return str(self.image_size)

    def _plan_epoch(self, epoch: int):
        # The loss follows the epoch its labels were planned for (bb._reject_share).
        self._planned_epoch = epoch
        return bb._plan_epoch_samples(
            self.train_samples,
            self.vocab,
            epoch,
            seed=self.sampling_seed,
            neg_pos_ratio=self.neg_pos_ratio,
            label_policy=self.label_policy,
            positive_cap=self.positive_cap,
            min_positive_threshold=self.min_positive_threshold
            if self.oversample_positives
            else 0,
            max_oversample_factor=self.max_oversample_factor,
            positive_cap_mode=self.positive_cap_mode,
        )

    def prepare_data(self, ctx: TaskContext) -> None:
        # Epoch-0 plan preview: the planner is deterministic, so these stats are
        # exactly what the first epoch will train on.
        _preview, plan_stats = self._plan_epoch(0)
        self._emit(
            "preparing",
            f"Preparing backbone training on {self._unique_images} unique images "
            f"({len(self.train_samples)} train / {len(self.val_samples)} val)",
            train_samples=len(self.train_samples),
            val_samples=len(self.val_samples),
            unique_images=self._unique_images,
            concepts=len(self.vocab.concepts),
            groups=len(self.vocab.groups),
            sampling=plan_stats,
            device=ctx.device.type,
            amp_dtype=amp_label(self.amp_enabled, self.amp_dtype),
            grad_scaler=self._scaler is not None,
            lean_grn=self.lean_grn,
        )

    def create_model(self, ctx: TaskContext, resume_state: dict | None):
        spec = self.request.get("backbone_init")
        backbone = create_model(
            model_size=self.model_size,
            pretrained=wants_timm_pretrained(spec),
            num_classes=0,
            drop_path_rate=self.drop_path_rate,
        )
        apply_backbone_init(backbone, spec)
        if self.lean_grn:
            use_lean_grn(backbone)
        backbone = backbone.to(ctx.device)
        self.feature_dim = backbone.num_features
        heads = bb._MultiTaskHeads(backbone.num_features, self.vocab).to(ctx.device)
        model = bb._BackboneWithHeads(backbone, heads).to(ctx.device)
        if resume_state is not None:
            model.load_state_dict(resume_state["model"])
        if self.channels_last:
            model = model.to(memory_format=torch.channels_last)
        self.model = model
        return model

    def resolve_batch_size(
        self, ctx: TaskContext, model, resume_state: dict | None
    ) -> int:
        # Explicit batch size keeps the legacy path bit-for-bit; ``batch_size`` of
        # 0/"auto" opts into the shared VRAM probe (the backbone trains square, so
        # one bucket at image_size).
        if not self._autobatch:
            return self.batch_size
        from bittrainer.autobatch import determine_batch_size

        result = determine_batch_size(
            model,
            {(self.image_size, self.image_size): len(self.train_samples)},
            ctx.device,
            dtype=self.amp_dtype,
        )
        self._cb({"type": "autobatch", **result})
        self.batch_size = int(result["batch_size"])
        return self.batch_size

    def create_optimizer(
        self, ctx: TaskContext, model, eff_bs: int, resume_state: dict | None
    ):
        # LLRD now buckets the backbone.*/heads.* wrapper prefixes correctly (A4),
        # so layer-wise decay + weight-decay exclusions are enabled by default.
        optimizer = make_optimizer(
            model,
            llrd=self.llrd,
            llrd_decay=self.llrd_decay,
            wd_exclusions=self.wd_exclusions,
        )
        t_max = max(1, self.epochs)
        # The core steps the scheduler every epoch and serialises its state, so a
        # scheduler object is always provided: cosine when enabled, else a constant
        # multiplier (the legacy "no cosine" constant-LR behaviour).
        scheduler = (
            CosineAnnealingLR(optimizer, T_max=t_max)
            if self.use_cosine
            else LambdaLR(optimizer, lambda _epoch: 1.0)
        )
        self.ema = ModelEMA(model, decay=self.ema_decay) if self.use_ema else None

        if resume_state is not None:
            restore_optimizer_state(resume_state, optimizer, scheduler, ctx.device)
            if self.ema is not None and resume_state.get("ema") is not None:
                self.ema.load_full_state_dict(resume_state["ema"])
            self.step = int(resume_state.get("global_step", 0))
        return optimizer, scheduler, t_max

    def restore_resume_extra(self, ctx: TaskContext, resume_state: dict) -> None:
        self.best_backbone_state = resume_state.get("best_backbone_state")
        self.best_heads_state = resume_state.get("best_heads_state")
        best = resume_state.get("best") or {}
        self.best_metrics = dict(best.get("best_metrics") or {})
        self.best_score = float(best.get("best_validation_score", -1.0))
        # The scaler's running scale continues where it stopped. A backup taken
        # under bf16 carries none, and a bf16 resume has no scaler to load into.
        scaler_state = resume_state.get("grad_scaler")
        if self._scaler is not None and scaler_state:
            self._scaler.load_state_dict(scaler_state)

    def resumed_message(
        self, ctx: TaskContext, best: BestTracker, global_step: int, start_epoch: int
    ) -> dict:
        return {
            "type": "training_resumed",
            "run_id": self.request.get("run_id"),
            "resumed_from": str(self.config.get("resume_from")),
            "epoch": start_epoch,
            "global_step": global_step,
            "best_score": best.best_validation_score,
        }

    # -- per-epoch ---------------------------------------------------------
    def _loader_kwargs(self, device, *, persistent: bool) -> dict:
        """DataLoader kwargs for this run.

        ``dataloader_workers`` (default 0, the legacy path) moves decode and
        augmentation off the training thread. The train loader is rebuilt every
        epoch, so only the long-lived val loader asks to keep its workers.
        """
        workers = max(0, self.dataloader_workers)
        kwargs = {
            "batch_size": self.batch_size,
            "collate_fn": bb._collate,
            "num_workers": workers,
            "pin_memory": workers > 0 and torch.device(device).type == "cuda",
        }
        if workers > 0:
            kwargs["persistent_workers"] = persistent
            kwargs["prefetch_factor"] = 4
        return kwargs

    def build_loaders(
        self, ctx: TaskContext, epoch: int, eff_bs: int, resume_info: ResumeInfo
    ):
        image_size = self._epoch_image_size(epoch)
        # The finetune tail switches the VAL resolution too — the tail's scores
        # (and the exported candidate's selection) must be measured at the
        # resolution the model ships at.
        if self._val_loader is None or self._val_loader_size != image_size:
            self._val_loader = DataLoader(
                bb._BackboneDataset(self.val_samples, bb._val_transform(image_size)),
                shuffle=False,
                **self._loader_kwargs(ctx.device, persistent=True),
            )
            self._val_loader_size = image_size
        # Per-epoch label plan (ISSUE-0545/0546): fresh negative draw each epoch
        # sweeps the full negative pool over training without ever concentrating
        # it; validation samples stay uncapped and masked-unknown.
        epoch_samples, plan_stats = self._plan_epoch(epoch)
        self._pos_weight = (
            bb._head_pos_weights(plan_stats) if self.use_pos_weight else None
        )
        train_loader = DataLoader(
            bb._BackboneDataset(
                epoch_samples, bb._train_transform(image_size, self._aug)
            ),
            shuffle=True,
            **self._loader_kwargs(ctx.device, persistent=False),
        )
        # Backbone resume is epoch-restart: no schedule replay, no partial start.
        return train_loader, None, 0

    def make_step_callback(
        self,
        ctx: TaskContext,
        epoch: int,
        eff_bs: int,
        best: BestTracker,
        epoch_start_mono,
    ):
        """The in-epoch heartbeat, or ``None`` when ``progress_every_steps`` is 0."""
        if self.progress_every_steps <= 0:
            return None

        def _step(batch: int, batches: int, loss: float, images: int, seconds: float):
            self._emit(
                "training_step",
                f"Epoch {epoch + 1}/{self.epochs}, batch {batch}/{batches}",
                epoch=epoch + 1,
                epochs=self.epochs,
                batch=batch,
                batches=batches,
                steps=self.step,
                loss=loss,
                images_per_second=images / seconds if seconds > 0 else None,
            )

        return _step

    def train_epoch(
        self,
        ctx: TaskContext,
        model,
        optimizer,
        train_loader,
        *,
        step_callback,
        boundary_hook,
        start_batch: int,
    ):
        device = ctx.device
        scaler = self._scaler
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        images_seen = 0
        total_batches = len(train_loader)
        started = time.perf_counter()
        for images, binary_labels, group_labels in train_loader:
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise bb.BackboneTrainingCancelled
            if self.max_steps is not None and self.step >= int(self.max_steps):
                # Step cap reached — ask the core to stop at the next epoch boundary.
                self.steps_stop_event.set()
                break
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device, non_blocking=True)
            if self.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.amp.autocast(
                device_type=device.type, dtype=self.amp_dtype, enabled=self.amp_enabled
            ):
                features = model.backbone(images)
                loss = bb._batch_loss(
                    features.float(),
                    model.heads,
                    binary_labels,
                    group_labels,
                    device,
                    pos_weight=self._pos_weight,
                    group_label_smoothing=self.group_label_smoothing,
                    reduction=self.loss_reduction,
                    label_policy=self.label_policy,
                    epoch=self._planned_epoch,
                )
            if loss is None:
                continue
            clip = self.clip_grad_norm and self.clip_grad_norm > 0
            if scaler is not None:
                # fp16: backward on the scaled loss, then unscale so the clipper
                # sees true gradient norms. The scaler skips the optimizer step
                # (and backs its scale off) when a gradient came back inf/nan.
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if clip:
                    clip_gradients(model, self.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if clip:
                    clip_gradients(model, self.clip_grad_norm)
                optimizer.step()
            if self.ema is not None:
                self.ema.update(model)
            self.step += 1
            epoch_loss += float(loss.detach())
            epoch_batches += 1
            images_seen += int(images.shape[0])
            if (
                step_callback is not None
                and epoch_batches % self.progress_every_steps == 0
            ):
                step_callback(
                    epoch_batches,
                    total_batches,
                    epoch_loss / epoch_batches,
                    images_seen,
                    time.perf_counter() - started,
                )
            if boundary_hook(epoch_batches) == "stop":  # pause backed up mid-epoch
                break
        self._train_seconds = time.perf_counter() - started
        self._train_images = images_seen
        return epoch_loss / max(epoch_batches, 1)

    def validate(self, ctx: TaskContext, model, epoch: int, train_result) -> dict:
        eval_backbone, eval_heads = self._eval_modules()
        # Fresh sink each validate: holds this epoch's raw logits so a winning epoch
        # can be snapshotted for calibration (save_candidate copies it).
        self._val_score_sink = {}
        metrics = (
            bb._evaluate(
                eval_backbone,
                eval_heads,
                self._val_loader,
                self.vocab,
                ctx.device,
                amp_enabled=self.amp_enabled,
                amp_dtype=self.amp_dtype,
                score_sink=self._val_score_sink,
            )
            if self.val_samples
            else {}
        )
        self.validation_metrics = metrics
        self.validation_score = self.selection_score(metrics)
        return metrics

    def selection_score(self, metrics: dict) -> float:
        # Prefer the A1 ``selection`` aggregate; fall back to the legacy blind mean
        # so fakes returning bare accuracy dicts (existing tests) still score right.
        if "selection" in metrics:
            return float(metrics["selection"])
        return (sum(metrics.values()) / len(metrics)) if metrics else 0.0

    def save_candidate(
        self, ctx: TaskContext, model, epoch: int, metrics: dict, best: BestTracker
    ) -> None:
        # No val signal: never snapshot a "best" — finalize serialises the final
        # EMA weights instead (legacy behaviour).
        if not self.val_samples:
            return
        eval_backbone, eval_heads = self._eval_modules()
        # clone() is load-bearing: on CPU (and whenever the source already lives on
        # the target device) ``.to("cpu")`` returns the SAME tensor, so a bare
        # snapshot would alias the live EMA weights and drift as later epochs update
        # them — the "best" would silently become the last. clone() freezes it.
        self.best_backbone_state = {
            k: v.detach().cpu().clone() for k, v in eval_backbone.state_dict().items()
        }
        self.best_heads_state = {
            k: v.detach().cpu().clone() for k, v in eval_heads.state_dict().items()
        }
        self.best_metrics = dict(metrics)
        self.best_score = best.best_validation_score
        best.best_metrics = dict(metrics)
        # Snapshot the winning epoch's raw val logits so calibration is fitted on
        # the exported epoch, not whatever the last epoch happened to produce.
        self._best_val_scores = (
            dict(self._val_score_sink) if self._val_score_sink else None
        )

    def on_epoch_end(
        self,
        ctx: TaskContext,
        model,
        epoch: int,
        metrics: dict,
        selected_score: float,
        best: BestTracker,
    ) -> None:
        # Resolution-tail switch: low-res and high-res validation scores are not
        # comparable, so reset the best tracker on the LAST pre-tail epoch. The
        # first tail epoch then always re-wins (score > -1), the exported
        # candidate is selected at the deployment resolution, and patience
        # accumulated at low res can't early-stop the tail. Prodigy re-adapts
        # its step within a few batches; EMA and the cosine schedule continue
        # uninterrupted (ConvNeXt weights are resolution-agnostic).
        if self._has_tail and epoch == self.epochs - self.finetune_epochs - 1:
            best.best_validation_score = -1.0
            best.patience_counter = 0
            self._emit(
                "training",
                f"Switching to finetune resolution {self.finetune_image_size}px for the "
                f"last {self.finetune_epochs} epoch(s)",
                finetune_image_size=self.finetune_image_size,
                finetune_epochs=self.finetune_epochs,
            )
            return
        # Surface the early-stop notice the legacy loop emitted (the core breaks on
        # the same condition immediately after this hook).
        if (
            self.val_samples
            and self.patience > 0
            and best.patience_counter >= self.patience
        ):
            self._emit(
                "training",
                f"Early stopping at epoch {epoch + 1} (patience {self.patience})",
                early_stop=True,
                epoch=epoch + 1,
            )

    def epoch_message(
        self,
        ctx: TaskContext,
        epoch: int,
        metrics: dict,
        train_result,
        selected_score: float,
        best: BestTracker,
    ) -> None:
        have_best = best.best_validation_score >= 0
        # Training speed for the epoch (decode + augment + step, validation not
        # included). Absent when the epoch trained nothing or the task does not
        # time itself (head-only retraining).
        speed = {}
        if self._train_seconds > 0 and self._train_images > 0:
            speed = {
                "train_seconds": self._train_seconds,
                "train_images": self._train_images,
                "images_per_second": self._train_images / self._train_seconds,
            }
        self._emit(
            "training",
            f"Epoch {epoch + 1}/{self.epochs}",
            epoch=epoch + 1,
            epochs=self.epochs,
            steps=self.step,
            loss=train_result,
            validation_score=self.validation_score,
            best_score=best.best_validation_score if have_best else None,
            best_epoch=best.best_epoch + 1 if have_best else None,
            **speed,
        )
        return None

    def collect_extra_state(
        self, ctx: TaskContext, *, rng_epoch_start, schedule, batch_in_epoch: int
    ) -> dict:
        # Backbone is epoch-restart; carry the EMA, the RNG stream, and the
        # in-memory best backbone/heads snapshot so a resume keeps the incumbent.
        return {
            "ema": self.ema.full_state_dict() if self.ema is not None else None,
            "rng_now": capture_rng_states(ctx.device),
            "rng_epoch_start": rng_epoch_start,
            "best_backbone_state": self.best_backbone_state,
            "best_heads_state": self.best_heads_state,
            "grad_scaler": self._scaler.state_dict()
            if self._scaler is not None
            else None,
        }

    # -- finalisation ------------------------------------------------------
    def finalize(
        self, ctx: TaskContext, model, best: BestTracker, epochs_completed: int
    ) -> dict:
        self._emit("validating", "Validating backbone candidate")
        if self.val_samples and self.best_backbone_state is not None:
            validation_metrics = self.best_metrics
            validation_score = self.best_score
            backbone_state = self.best_backbone_state
            heads_state = self.best_heads_state
        else:
            # No validation signal (or no improved epoch): serialise the final
            # (EMA) backbone + heads.
            eval_backbone, eval_heads = self._eval_modules()
            backbone_state = {
                k: v.detach().to("cpu") for k, v in eval_backbone.state_dict().items()
            }
            heads_state = {
                k: v.detach().to("cpu") for k, v in eval_heads.state_dict().items()
            }
            validation_metrics = self.validation_metrics
            validation_score = self.validation_score

        candidate_path = Path(self.request["candidate_checkpoint_path"])
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self._candidate_metadata(validation_score, validation_metrics)
        from safetensors.torch import save_file

        # BARE backbone keys (apply_backbone_init's unprefix branch requires the
        # backbone keys NOT be prefixed) plus heads.<key> for the multi-task heads.
        # contiguous(): a channels-last run holds its conv weights in NHWC
        # strides, which safetensors refuses to write. A no-op otherwise.
        state = {
            key: value.detach().to("cpu").contiguous()
            for key, value in backbone_state.items()
        }
        for key, value in (heads_state or {}).items():
            state[f"heads.{key}"] = value.detach().to("cpu")
        save_file(
            state,
            str(candidate_path),
            metadata={
                key: bb._stringify(value)
                for key, value in metadata.items()
                if value is not None
            },
        )
        self._emit(
            "saving",
            "Backbone candidate checkpoint written",
            candidate_checkpoint_path=str(candidate_path),
            validation_score=validation_score,
        )

        return {
            "candidate_checkpoint_path": str(candidate_path),
            "validation_score": float(validation_score),
            "validation_metrics": validation_metrics,
            "heads": self.request.get("heads") or {},
            "release_blocking": bool(self.request.get("release_blocking")),
            "epochs_completed": int(epochs_completed),
            "best_epoch": int(best.best_epoch + 1)
            if self.val_samples
            else int(epochs_completed),
        }

    def _candidate_metadata(self, validation_score, validation_metrics) -> dict:
        """Engine-readable candidate metadata, shared with BackboneHeadsTask."""
        metadata = {
            "family_name": self.request.get("family_name"),
            "architecture": self.request.get("architecture"),
            "size_alias": self.request.get("size_alias"),
            "display_size": self.request.get("display_size"),
            "convnextv2_size": self.request.get("convnextv2_size"),
            "version": "1",
            "status": "candidate",
            "created_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "training_run_id": self.request.get("run_id"),
            "dataset_snapshot_id": self.request.get("dataset_snapshot_id"),
            "content_hash_index_id": self.request.get("content_hash_index_id"),
            "license_provenance": self.request.get("license_provenance")
            or "locally_trained",
            "initialization_source": self.request.get("initialization_source")
            or (self.request.get("backbone_init") or {}).get("source"),
            "non_commercial_only": bool(self.request.get("non_commercial_only")),
            "external_pretrained_used": bool(
                self.request.get("external_pretrained_used")
            ),
            "temporary_timm_fallback_used": bool(
                self.request.get("temporary_timm_fallback_used")
            ),
            "release_blocking": bool(self.request.get("release_blocking")),
            "validation_score": validation_score,
            "validation_metrics_json": validation_metrics,
            "heads_json": self.request.get("heads") or {},
            "training_config_json": self.config,
            # Head persistence (Bitcrush ISSUE-0542): the tensors ride the state
            # dict below; these fields let a consumer rebuild _MultiTaskHeads.
            "heads_state_present": "1",
            "backbone_feature_dim": self.feature_dim,
            "heads_concepts_json": list(self.vocab.concepts),
            "heads_groups_json": {g: list(cs) for g, cs in self.vocab.groups.items()},
        }
        for key in (
            "pretrained_model_id",
            "pretrained_license",
            "pretrained_source_url",
        ):
            value = self.request.get(key)
            if value is not None:
                metadata[key] = value
        # Per-head calibration fitted on the WINNING epoch's val logits (A2). Keys
        # are absent (additive) unless enabled and val scores were captured, so a
        # consumer decoding ``sigmoid(logit / T) >= threshold`` keeps working with
        # T=1 / threshold=0.5 defaults. Head-only retraining opts out this round.
        scores = self._best_val_scores if self._calibration_enabled else None
        binary_scores = (scores or {}).get("binary") or {}
        if binary_scores:
            if self.calibrate_thresholds:
                metadata["binary_thresholds_json"] = bb._tune_binary_thresholds(
                    binary_scores
                )
            if self.calibrate_temperature:
                metadata["binary_temperatures_json"] = {
                    concept: bb._fit_binary_temperature(logits, targets)
                    for concept, (logits, targets) in binary_scores.items()
                }
        return metadata
