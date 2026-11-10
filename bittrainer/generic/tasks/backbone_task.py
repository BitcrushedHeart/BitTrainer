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

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader

import bittrainer.backbone_trainer as bb
from bittrainer.amp import amp_label, make_grad_scaler
from bittrainer.autobatch import (
    _OPT_STATE_MULT,
    _allocator_cap_bytes,
    _apply_trust_bound,
    _linear_fit,
)
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
    _looks_like_oom,
    capture_rng_states,
    restore_optimizer_state,
    restore_rng_states,
)

logger = logging.getLogger(__name__)

# The share of the memory a card can give the run that one training step may
# take. The rest is for what a single step does not show: validation, the last
# partial batch, and the allocator's fragmentation over days.
_PREFLIGHT_SHARE = 0.95
# The share an automatic batch size aims for (the shared probe's own figure).
_AUTO_SHARE = 0.85
# Once the GradScaler has settled it skips about one step in 2000 as it probes
# upwards. More than this share of an epoch's steps is reported loudly.
_SKIP_WARN_SHARE = 0.02


class BackboneRunRefused(RuntimeError):
    """Raised before the first epoch when the run cannot work as configured:
    a training step does not fit the card. Nothing has been trained."""


def _wants_auto(raw) -> bool:
    """A batch size of 0 or "auto" asks for one to be measured."""
    return str(raw).lower() == "auto" or (
        isinstance(raw, (int, float)) and int(raw) == 0
    )


def _gb(size: float) -> str:
    return f"{size / 2**30:.1f}"


def _out_of_memory(exc: BaseException) -> bool:
    """Running out of GPU memory, however it is worded: torch's own error, or
    cuDNN / cuBLAS failing to get their workspace on a card that is full."""
    return _looks_like_oom(exc) or "ALLOC_FAILED" in str(exc)


def _vram_room(device) -> int | None:
    """The most this process can hold on *device*, in bytes: what it holds now
    plus what is free, under any cap set on the allocator. None off a GPU."""
    if torch.device(device).type != "cuda":
        return None
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    room = free + torch.cuda.memory_reserved(device)
    cap = _allocator_cap_bytes(device, total)
    return room if cap is None else min(room, cap)


def _largest_fitting_batch(measure, budget: float, max_batch: int) -> dict | None:
    """The largest batch whose training step stays inside *budget* bytes.

    ``measure(batch)`` runs one real step and returns its peak in bytes, or
    None when it ran out of memory. Batches double from 1 for as long as a
    straight line through the measurements says the next one fits, so a step
    expected to overflow the card is never tried and the answer is at most
    twice a batch that was really run. None when not even one image fits.
    """
    sizes: list[int] = []
    peaks: list[float] = []
    batch, ran_out = 1, False
    while batch <= max_batch:
        peak = measure(batch)
        if peak is None:
            ran_out = True
            break
        sizes.append(batch)
        peaks.append(float(peak))
        fit = _linear_fit(sizes, peaks)
        # From one measurement: as if the whole cost doubled with the batch.
        ahead = fit[0] * batch * 2 + fit[1] if fit else peak * 2
        if ahead > budget:
            break
        batch *= 2
    if not sizes:
        return None
    fit = _linear_fit(sizes, peaks)
    limit = sizes[-1]
    if fit is not None and fit[0] > 0:
        limit = int((budget - fit[1]) / fit[0])
    limit, trust_cap = _apply_trust_bound(limit, sizes[-1], ran_out)
    return {
        "batch_size": max(1, min(limit, max_batch)),
        "max_fitted_rung": sizes[-1],
        "trust_cap": trust_cap,
        "fit_slope": fit[0] if fit else None,
        "fit_intercept": fit[1] if fit else None,
    }


def save_val_scores(path, scores: dict) -> None:
    """Write one epoch's raw validation logits and targets as a compressed
    ``.npz``, so thresholds can be worked out again later under any rule,
    without the images or a GPU.

    Heads are stored by position, their names in a JSON list beside them
    (``binary_names_json`` / ``group_names_json``), so a name may hold any
    character. The yes/no heads lie end to end in ``binary_logits`` /
    ``binary_targets``, head ``i`` at ``binary_offsets[i]:binary_offsets[i + 1]``.
    Group ``i`` has ``group_<i>_logits`` (a column per class) and
    ``group_<i>_targets``. :func:`load_val_scores` reads it back.
    """
    binary = scores.get("binary") or {}
    groups = scores.get("groups") or {}
    empty = [np.zeros(0)]
    arrays = {
        "binary_names_json": np.array(json.dumps(list(binary))),
        "binary_offsets": np.cumsum([0, *(len(pair[0]) for pair in binary.values())]),
        "binary_logits": np.concatenate([p[0] for p in binary.values()] or empty),
        "binary_targets": np.concatenate([p[1] for p in binary.values()] or empty),
        "group_names_json": np.array(json.dumps(list(groups))),
    }
    for index, (logits, targets) in enumerate(groups.values()):
        arrays[f"group_{index}_logits"] = np.asarray(logits)
        arrays[f"group_{index}_targets"] = np.asarray(targets)
    np.savez_compressed(path, **arrays)


def load_val_scores(path) -> dict:
    """Read :func:`save_val_scores`'s file back into
    ``{"binary": {head: (logits, targets)}, "groups": {group: (logits, targets)}}``."""
    with np.load(path, allow_pickle=False) as data:
        offsets = data["binary_offsets"]
        logits, targets = data["binary_logits"], data["binary_targets"]
        binary = {
            name: (logits[start:stop], targets[start:stop])
            for name, start, stop in zip(
                json.loads(str(data["binary_names_json"])), offsets, offsets[1:]
            )
        }
        groups = {
            name: (data[f"group_{index}_logits"], data[f"group_{index}_targets"])
            for index, name in enumerate(json.loads(str(data["group_names_json"])))
        }
    return {"binary": binary, "groups": groups}


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
        self._autobatch = _wants_auto(raw_bs)
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
        # One real training step at every size the run will use, before the
        # first epoch (see _preflight). Unset: on a GPU only.
        self.preflight = c.get("preflight")
        self._dry_run = False
        # fp16 only: optimizer steps the GradScaler skipped (see _count_skip),
        # over the run and inside the current epoch, and its current scale.
        self.skipped_steps = 0
        self._scaler_settled = False
        self._grad_scale: float | None = None
        self._epoch_skipped = 0
        self._epoch_late_skipped = 0
        self._epoch_first_step = 0
        self._skip_warned = False

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
        # The tail's own batch size: a 384px step takes about three times the
        # memory of a 224px one. Unset follows ``batch_size``; 0 / "auto" is
        # measured at the tail's resolution (see resolve_batch_size).
        raw_tail_bs = c.get("finetune_batch_size")
        self._auto_tail = (
            self._autobatch if raw_tail_bs is None else _wants_auto(raw_tail_bs)
        )
        self.finetune_batch_size = (
            0 if self._auto_tail else int(raw_tail_bs or self.batch_size)
        )
        self._val_loader_size: int | None = None
        # What the export records about itself (see finalize).
        self._exported_epoch: int | None = None
        self._val_scores_file: str | None = None

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

    def _epoch_batch_size(self, epoch: int) -> int:
        if self._has_tail and epoch >= self.epochs - self.finetune_epochs:
            return self.finetune_batch_size
        return self.batch_size

    def _phases(self, start_epoch: int = 0) -> list[tuple[str, int, int]]:
        """(name, image size, batch size) of each stretch of the run that
        still has an epoch to train from *start_epoch* on."""
        tail_epochs = self.finetune_epochs if self._has_tail else 0
        phases = []
        if start_epoch < self.epochs - tail_epochs:
            phases.append(("main", self.image_size, self.batch_size))
        if tail_epochs and start_epoch < self.epochs:
            phases.append(("tail", self.finetune_image_size, self.finetune_batch_size))
        return phases

    # The tail's batch size is deliberately not part of this: nothing a backup
    # holds depends on a batch size (an epoch restarts from its beginning), so
    # a paused run can come back with a smaller tail batch instead of being
    # told its backups belong to another run.
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
        # An explicit batch size is used as given. 0 / "auto" is measured, each
        # phase at its own resolution: a batch sized at 224px is far too large
        # for a 384px tail.
        if self._autobatch:
            self.batch_size = self._auto_batch_size(ctx, model, self.image_size, "main")
        if not self._has_tail:
            self.finetune_batch_size = self.batch_size
        elif self._auto_tail:
            self.finetune_batch_size = self._auto_batch_size(
                ctx, model, self.finetune_image_size, "tail"
            )
        return self.batch_size

    def _auto_batch_size(self, ctx: TaskContext, model, image_size: int, phase: str):
        """The largest batch that fits at *image_size*, measured with real
        training steps (so in the run's precision, with its GRN layer, its loss
        and a distillation's teacher), leaving 15% of the card free.

        The shared probe could not do this: it calls ``model(inputs)``, the
        wrapper here has no ``forward``, and the probe read the error that
        raises as "out of memory" at a batch of one and fell back to 8. It
        still answers off a GPU (a fixed 32) and for head-only retraining,
        whose loop runs on cached vectors.
        """
        device = ctx.device
        train_images = len(self.train_samples)
        room = _vram_room(device)
        if room is None or self.trainer_name != BackboneTask.trainer_name:
            from bittrainer.autobatch import determine_batch_size

            found = determine_batch_size(
                model,
                {(image_size, image_size): train_images},
                device,
                dtype=self.amp_dtype,
            )
        else:
            # Still to come when the batch is chosen, and resident from then
            # on: the optimizer's state and the EMA's copy of the model.
            weights = sum(p.numel() * p.element_size() for p in model.parameters())
            later = (_OPT_STATE_MULT + (1.0 if self.use_ema else 0.0)) * weights

            def _measure(batch: int):
                try:
                    peak = self._dry_step(ctx, model, image_size, batch)
                except RuntimeError as exc:
                    if not _out_of_memory(exc):
                        raise  # a broken step must stop the run, not size it
                    return None
                # Past what the card has, Windows spills into shared memory
                # instead of failing: that is not fitting either.
                return None if peak > room else peak

            found = _largest_fitting_batch(
                _measure, _AUTO_SHARE * room - later, train_images
            )
            if found is None:
                raise BackboneRunRefused(
                    f"Automatic batch size: not even one {image_size}px image fits in "
                    f"a training step on this card ({_gb(room)} GB for the run)."
                )
        self._cb(
            {"type": "autobatch", "phase": phase, "image_size": image_size, **found}
        )
        return int(found["batch_size"])

    # -- before the first epoch: a step with nothing kept -------------------
    def _dry_step(self, ctx: TaskContext, model, image_size: int, batch_size: int):
        """One real training step at this size, with nothing kept of it.

        ``train_epoch`` itself runs on one made-up batch, so the precision, the
        loss and a subclass's teacher are all in it; ``_dry_run`` makes the loop
        stop after the backward pass, before the optimizer, the scaler's update
        and the EMA. Weights, RNG streams and counters are as they were
        afterwards. Returns the step's peak in bytes on a GPU, else None.
        """
        device = ctx.device
        on_gpu = device.type == "cuda"
        images = torch.zeros(batch_size, 3, image_size, image_size)
        # Every head labelled on every image: the most a batch can ask of the loss.
        binary = [
            {
                name: float((row + column) % 2)
                for column, name in enumerate(self.vocab.concepts)
            }
            for row in range(batch_size)
        ]
        groups = [
            {name: row % len(classes) for name, classes in self.vocab.groups.items()}
            for row in range(batch_size)
        ]
        rng = capture_rng_states(device)
        was_training = model.training
        timing = (self._train_seconds, self._train_images)
        self._dry_run = True
        if on_gpu:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        try:
            self.train_epoch(
                ctx,
                model,
                # The loop only clears gradients before the step it never takes.
                SimpleNamespace(zero_grad=model.zero_grad),
                [(images, binary, groups)],
                step_callback=None,
                boundary_hook=None,
                start_batch=0,
            )
            return torch.cuda.max_memory_allocated(device) if on_gpu else None
        finally:
            self._dry_run = False
            model.zero_grad(set_to_none=True)
            model.train(was_training)
            restore_rng_states(rng, device)
            self._train_seconds, self._train_images = timing
            if on_gpu:
                torch.cuda.empty_cache()

    def _preflight(self, ctx: TaskContext, model, optimizer, start_epoch: int) -> None:
        """Refuse, before any training, a run with a step that does not fit.

        One real step at each (resolution, batch size) still ahead. A tail too
        large for the card is then reported in the run's first minute, by
        name, instead of failing at the switch a day in. On by default on a
        GPU; ``preflight`` True forces it, False skips it.
        """
        wanted = ctx.device.type == "cuda" if self.preflight is None else self.preflight
        if not wanted or (
            self.max_steps is not None and self.step >= int(self.max_steps)
        ):
            return
        room = _vram_room(ctx.device)
        # A fresh optimizer builds its state on its first real step, and keeps
        # it: the tail has to fit beside it. A resumed one already holds it.
        to_come = 0
        if not optimizer.state:
            to_come = int(
                _OPT_STATE_MULT
                * sum(
                    p.numel() * p.element_size()
                    for p in model.parameters()
                    if p.requires_grad
                )
            )
        precision = amp_label(self.amp_enabled, self.amp_dtype)
        for phase, image_size, batch_size in self._phases(start_epoch):
            tail = phase == "tail"
            what = (
                f"{'tail' if tail else 'main phase'} ({image_size}px, batch size "
                f"{batch_size}, precision {precision})"
            )
            lower = (
                "Lower finetune_batch_size (--finetune-batch-size)."
                if tail
                else "Lower batch_size (--batch-size)."
            )
            try:
                peak = self._dry_step(ctx, model, image_size, batch_size)
            except RuntimeError as exc:
                if not _out_of_memory(exc):
                    raise
                raise BackboneRunRefused(
                    f"Preflight: one training step of the {what} ran out of GPU "
                    f"memory. {lower}"
                ) from exc
            needed = None if peak is None else peak + to_come
            if needed is not None and room and needed > _PREFLIGHT_SHARE * room:
                raise BackboneRunRefused(
                    f"Preflight: one training step of the {what} needs "
                    f"{_gb(needed)} GB of the {_gb(room)} GB this card can give the "
                    f"run, more than the {_PREFLIGHT_SHARE:.0%} a long run can rely "
                    f"on. {lower}"
                )
            fits = f"Preflight: the {what} fits"
            if needed is not None and room:
                fits += f": {_gb(needed)} of {_gb(room)} GB"
            self._emit(
                "preflight",
                fits,
                phase=phase,
                image_size=image_size,
                batch_size=batch_size,
                peak_vram_mb=None if needed is None else round(needed / 2**20),
            )

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
        # Here, not inside the epoch loop: a refusal then leaves no backup
        # behind for the next attempt to trip over.
        start_epoch = int(resume_state["epoch"]) if resume_state is not None else 0
        self._preflight(ctx, model, optimizer, start_epoch)
        return optimizer, scheduler, t_max

    def restore_resume_extra(self, ctx: TaskContext, resume_state: dict) -> None:
        self.best_backbone_state = resume_state.get("best_backbone_state")
        self.best_heads_state = resume_state.get("best_heads_state")
        # The winning epoch's raw validation scores: without them a resumed run
        # that never beats that epoch would export it with no thresholds.
        self._best_val_scores = resume_state.get("best_val_scores")
        self.skipped_steps = int(resume_state.get("skipped_steps", 0))
        # A backup from before the count existed is from a run under way.
        self._scaler_settled = bool(resume_state.get("scaler_settled", True))
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
            # What the epoch it resumes into trains at.
            "image_size": self._epoch_image_size(start_epoch),
            "batch_size": self._epoch_batch_size(start_epoch),
        }

    # -- per-epoch ---------------------------------------------------------
    def _loader_kwargs(
        self, device, *, persistent: bool, batch_size: int | None = None
    ) -> dict:
        """DataLoader kwargs for this run.

        ``dataloader_workers`` (default 0, the legacy path) moves decode and
        augmentation off the training thread. The train loader is rebuilt every
        epoch, so only the long-lived val loader asks to keep its workers.
        """
        workers = max(0, self.dataloader_workers)
        kwargs = {
            "batch_size": batch_size or self.batch_size,
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
        # The tail's batch size goes with its resolution, for both loaders.
        batch_size = self._epoch_batch_size(epoch)
        # The finetune tail switches the VAL resolution too — the tail's scores
        # (and the exported candidate's selection) must be measured at the
        # resolution the model ships at.
        if self._val_loader is None or self._val_loader_size != image_size:
            self._val_loader = DataLoader(
                bb._BackboneDataset(self.val_samples, bb._val_transform(image_size)),
                shuffle=False,
                **self._loader_kwargs(
                    ctx.device, persistent=True, batch_size=batch_size
                ),
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
            **self._loader_kwargs(ctx.device, persistent=False, batch_size=batch_size),
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
            self._warn_if_skipping(epoch)
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
                image_size=self._epoch_image_size(epoch),
                batch_size=self._epoch_batch_size(epoch),
                skipped_steps=self._epoch_skipped,
                grad_scale=self._grad_scale,
            )

        return _step

    # -- steps the fp16 GradScaler skipped ----------------------------------
    def _start_skip_count(self, scaler) -> None:
        """Start an epoch's count. Called at the top of every ``train_epoch``."""
        self._epoch_first_step = self.step
        self._epoch_skipped = 0
        self._epoch_late_skipped = 0
        self._skip_warned = False
        self._grad_scale = scaler.get_scale() if scaler is not None else None

    def _count_skip(self, scaler) -> None:
        """Called after every ``scaler.update()``.

        The scaler lowers its scale only when a gradient was not finite, and
        then it has left the optimizer alone: that step was skipped. The first
        step that lands ends the opening descent from the scaler's start value
        (expected, once per run); a skip after it is a late one.
        """
        scale = scaler.get_scale()
        if scale < self._grad_scale:
            self.skipped_steps += 1
            self._epoch_skipped += 1
            if self._scaler_settled:
                self._epoch_late_skipped += 1
        else:
            self._scaler_settled = True
        self._grad_scale = scale

    def _warn_if_skipping(self, epoch: int) -> None:
        """Say so, once an epoch, when late skips pass a small share of its
        steps: a run that keeps skipping has stopped learning without failing."""
        steps = self.step - self._epoch_first_step
        late = self._epoch_late_skipped
        if self._skip_warned or late < 3 or late <= _SKIP_WARN_SHARE * steps:
            return
        self._skip_warned = True
        text = (
            f"{late} of the {steps} steps so far in epoch {epoch + 1} were skipped: "
            f"gradients were not finite in half precision (scale now "
            f"{self._grad_scale:g}). Those steps trained nothing."
        )
        logger.warning("Backbone training: %s", text)
        self._emit(
            "warning",
            text,
            warning="skipped_steps",
            epoch=epoch + 1,
            skipped_steps=late,
            steps=steps,
            grad_scale=self._grad_scale,
        )

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
        self._start_skip_count(scaler)
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
            if self._dry_run:
                # A step for its memory only (see _dry_step): the backward pass
                # is the peak. No update of the weights, the scaler or the EMA.
                (scaler.scale(loss) if scaler is not None else loss).backward()
                break
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
                self._count_skip(scaler)
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
                f"Switching to finetune resolution {self.finetune_image_size}px "
                f"(batch size {self.finetune_batch_size}) for the last "
                f"{self.finetune_epochs} epoch(s)",
                finetune_image_size=self.finetune_image_size,
                finetune_epochs=self.finetune_epochs,
                finetune_batch_size=self.finetune_batch_size,
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
        self._warn_if_skipping(epoch)
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
            image_size=self._epoch_image_size(epoch),
            batch_size=self._epoch_batch_size(epoch),
            # fp16 only: steps the GradScaler skipped this epoch and so far,
            # and its scale now (None in bf16 and on a CPU).
            skipped_steps=self._epoch_skipped,
            skipped_steps_total=self.skipped_steps,
            grad_scale=self._grad_scale,
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
            "best_val_scores": self._best_val_scores,
            "grad_scaler": self._scaler.state_dict()
            if self._scaler is not None
            else None,
            "skipped_steps": self.skipped_steps,
            "scaler_settled": self._scaler_settled,
        }

    # -- finalisation ------------------------------------------------------
    def finalize(
        self, ctx: TaskContext, model, best: BestTracker, epochs_completed: int
    ) -> dict:
        self._emit("validating", "Validating backbone candidate")
        val_scores = None
        if self.val_samples and self.best_backbone_state is not None:
            validation_metrics = self.best_metrics
            validation_score = self.best_score
            backbone_state = self.best_backbone_state
            heads_state = self.best_heads_state
            val_scores = self._best_val_scores
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
        best_epoch = (
            int(best.best_epoch + 1) if self.val_samples else int(epochs_completed)
        )
        self._exported_epoch = best_epoch
        # The exported epoch's raw validation scores go beside the candidate,
        # so thresholds can be worked out again later under any rule. An older
        # run's file must never sit beside a new candidate.
        scores_path = candidate_path.with_name(f"{candidate_path.stem}.val_scores.npz")
        if val_scores and (val_scores.get("binary") or val_scores.get("groups")):
            save_val_scores(scores_path, val_scores)
            self._val_scores_file = scores_path.name
            scores_note = f"; validation scores in {scores_path.name}"
        else:
            scores_path.unlink(missing_ok=True)
            self._val_scores_file = None
            why = "none were collected" if self.val_samples else "no validation images"
            scores_note = f"; no validation scores to keep ({why})"
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
            f"Backbone candidate checkpoint written{scores_note}",
            candidate_checkpoint_path=str(candidate_path),
            validation_score=validation_score,
            val_scores_file=self._val_scores_file,
        )

        return {
            "candidate_checkpoint_path": str(candidate_path),
            "validation_score": float(validation_score),
            "validation_metrics": validation_metrics,
            "heads": self.request.get("heads") or {},
            "release_blocking": bool(self.request.get("release_blocking")),
            "epochs_completed": int(epochs_completed),
            "best_epoch": best_epoch,
            # The file beside the candidate with the exported epoch's raw
            # validation scores; None when there were none to keep.
            "val_scores_file": self._val_scores_file,
            # fp16 only: optimizer steps the GradScaler skipped, and its scale.
            "skipped_steps": int(self.skipped_steps),
            "grad_scale": self._grad_scale,
        }

    def _training_actual(self) -> dict:
        """What the run really used, defaults included, for the candidate's
        metadata (``training_config_json`` holds only what was passed in)."""
        tail = self._has_tail
        # The size the exported epoch was validated at: the shorter side is
        # resized to it, then a centre crop of it is taken (bb._val_transform).
        val_size = self._val_loader_size or self.image_size
        return {
            "validation_split": self.validation_split,
            "train_images": len(self.train_samples),
            "val_images": len(self.val_samples),
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "finetune_image_size": self.finetune_image_size if tail else None,
            "finetune_epochs": self.finetune_epochs if tail else None,
            "finetune_batch_size": self.finetune_batch_size if tail else None,
            "val_resize": val_size,
            "val_center_crop": val_size,
            "amp_dtype": amp_label(self.amp_enabled, self.amp_dtype),
            "grad_scaler": self._scaler is not None,
            "lean_grn": self.lean_grn,
            # Steps of the loop, and how many of them the scaler skipped.
            "optimizer_steps": self.step,
            "skipped_steps": self.skipped_steps,
            "best_epoch": self._exported_epoch,
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
        # How the run was really done, and where its validation scores are.
        # Head-only retraining shares this method but not the trunk loop these
        # describe, so it does not claim them.
        if self.trainer_name == BackboneTask.trainer_name:
            metadata["training_actual_json"] = self._training_actual()
            if self._val_scores_file:
                metadata["val_scores_file"] = self._val_scores_file
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
