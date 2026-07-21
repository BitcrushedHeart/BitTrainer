"""Generic training-task contract (Bitcrush ISSUE-0542 Step 3).

:class:`GenericTrainer` owns the training *lifecycle* — dataset prep, model
create/warm-start, warmup, autobatch, the epoch loop with pause / stop / backup /
resume / best / patience mechanics, and finalisation. Everything trainer-specific
(the group loss zoo, SWA / greedy-soup / DCW, skin-tone dual-view scoring,
promotion) lives behind the :class:`TrainingTask` hooks the core calls.

This step migrates only the group trainer; the binary / head-only / multihead /
dual-branch trainers keep their own loops for now and move onto ``TrainingTask``
later. The hook surface is therefore sized for those future tasks: the core never
branches on a task type, and every group-specific richness rides in a hook
(``on_epoch_end``, ``finalize``, ``collect_extra_state``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TaskContext:
    """Per-run environment the core threads through every hook.

    ``coordinator`` / ``fingerprint`` / ``resume_state`` start as ``None`` and are
    populated by :meth:`TrainingTask.fingerprint_init` (which calls
    ``training_state.init_backup`` with the task's fingerprint kwargs).
    """

    device: Any
    dtype: Any
    em: Any
    cb: Callable[[dict], None]
    checkpoint_dir: Any
    stop_event: object | None = None
    stop_now_event: object | None = None
    pause_event: object | None = None
    coordinator: Any = None
    fingerprint: dict | None = None
    resume_state: dict | None = None


@dataclass
class LoopSpec:
    """The handful of scalars the core epoch loop needs from the task."""

    max_epochs: int
    patience: int
    # Minimum composite gain required to replace the incumbent best. Guards the
    # selection against epoch-to-epoch noise (one-standard-error-rule spirit;
    # Prechelt 1998). The group default matches the pre-refactor constant.
    selection_min_delta: float = 0.002


@dataclass
class BestTracker:
    """The best-checkpoint / early-stop state cluster the core mutates.

    Mirrors the scalars ``run_group_training`` carried inline; the core owns
    ``best_validation_score`` / ``best_epoch`` / ``patience_counter`` (pure
    mechanics), while the task's :meth:`TrainingTask.save_candidate` fills the
    metric-derived fields (``best_val_macro_f1`` / ``best_val_qwk`` /
    ``best_metrics`` / ``best_checkpoint_path``).
    """

    best_val_macro_f1: float = -1.0
    best_val_qwk: float = -1.0
    best_validation_score: float = -1.0
    best_epoch: int = 0
    patience_counter: int = 0
    best_checkpoint_path: str | None = None
    best_metrics: dict = field(default_factory=dict)

    def as_backup_dict(self) -> dict:
        """The ``best`` sub-envelope ``collect_epoch_state`` expects."""
        return {
            "best_val_macro_f1": self.best_val_macro_f1,
            "best_val_qwk": self.best_val_qwk,
            "best_validation_score": self.best_validation_score,
            "best_epoch": self.best_epoch,
            "patience_counter": self.patience_counter,
            "best_checkpoint_path": self.best_checkpoint_path,
            "best_metrics": self.best_metrics,
        }

    def restore_from(self, best: dict) -> None:
        """Reload the cluster from a backup's ``best`` sub-envelope (resume)."""
        self.best_val_macro_f1 = best["best_val_macro_f1"]
        self.best_val_qwk = best["best_val_qwk"]
        self.best_validation_score = best["best_validation_score"]
        self.best_epoch = best["best_epoch"]
        self.patience_counter = best["patience_counter"]
        self.best_checkpoint_path = best["best_checkpoint_path"]
        self.best_metrics = dict(best.get("best_metrics") or {})


@dataclass
class ResumeInfo:
    """Mid-epoch continuation state handed to :meth:`TrainingTask.build_loaders`.

    ``mid_resume`` is True only for the FIRST resumed epoch of a mid-epoch backup;
    the loader then replays ``resume_schedule[resume_batch_in_epoch:]`` and jumps
    the augmentation RNG to ``resume_rng_now`` so the run is bit-exact.
    """

    mid_resume: bool = False
    resume_schedule: list[list[int]] | None = None
    resume_batch_in_epoch: int = 0
    resume_rng_now: dict | None = None


class TrainingTask(ABC):
    """The seam between :class:`GenericTrainer` and a concrete trainer.

    Hooks are grouped by lifecycle phase. Only the ones a trainer genuinely needs
    are abstract; the rest have no-op / minimal defaults so a simple task (or the
    unit-test spy) stays short. The core calls them in the order documented on
    :meth:`GenericTrainer.run`.
    """

    trainer_name: str = "generic"

    # -- one-time setup ----------------------------------------------------
    @abstractmethod
    def make_context(
        self, progress_callback, stop_event, stop_now_event, pause_event
    ) -> TaskContext:
        """Build the run's :class:`TaskContext` (emitter, device/dtype, checkpoint
        dir, events). Backup wiring is deferred to :meth:`fingerprint_init`."""

    @abstractmethod
    def fingerprint_init(self, ctx: TaskContext) -> None:
        """Populate ``ctx.coordinator`` / ``ctx.fingerprint`` / ``ctx.resume_state``
        (via ``training_state.init_backup``) and, on resume, re-apply resolved
        sweep outcomes + emit the resuming stage."""

    @abstractmethod
    def loop_spec(self) -> LoopSpec:
        """Return the epoch-loop scalars (max_epochs / patience / min-delta)."""

    @abstractmethod
    def prepare_data(self, ctx: TaskContext) -> None:
        """Build datasets + cache; store them on the task for later hooks."""

    @abstractmethod
    def create_model(self, ctx: TaskContext, resume_state: dict | None):
        """Return the (eager, fp32-master) model: warm-start for a fresh run,
        rebuild-and-load for a resume."""

    def pre_loop(self, ctx: TaskContext, model) -> None:
        """Fresh-run only (skipped on resume): head-probe warmup + sweeps."""

    @abstractmethod
    def resolve_batch_size(self, ctx: TaskContext, model, resume_state: dict | None) -> int:
        """Decide the effective batch size (manual / resumed / autobatch probe)."""

    def setup_training(self, ctx: TaskContext, model, resume_state: dict | None) -> None:
        """Class-balance / DCW / mixup / SWA setup before the optimizer is built."""

    @abstractmethod
    def create_optimizer(self, ctx: TaskContext, model, eff_bs: int, resume_state: dict | None):
        """Return ``(optimizer, scheduler, scheduler_t_max)``. Also builds any
        EMA / compiled-forward wrapper and restores optimizer/EMA state on resume."""

    def restore_resume_extra(self, ctx: TaskContext, resume_state: dict) -> None:
        """Resume only: reload task-owned run state (e.g. the greedy-soup pool)."""

    def resumed_message(self, ctx: TaskContext, best: BestTracker, global_step: int, start_epoch: int) -> dict | None:
        """The ``training_resumed`` payload, or None to skip emitting one."""
        return None

    # -- per-epoch ---------------------------------------------------------
    def reshuffle(self) -> None:
        """Reshuffle the train dataset for a new epoch (post RNG capture)."""

    @abstractmethod
    def build_loaders(self, ctx: TaskContext, epoch: int, eff_bs: int, resume_info: ResumeInfo):
        """Build this epoch's loaders. Store the val loader on the task and return
        ``(train_loader, schedule, start_batch)`` for the core."""

    def make_step_callback(self, ctx: TaskContext, epoch: int, eff_bs: int, best: BestTracker, epoch_start_mono: float):
        """Return the per-step progress callback passed into :meth:`train_epoch`."""
        return None

    @abstractmethod
    def train_epoch(
        self, ctx: TaskContext, model, optimizer, train_loader, *, step_callback, boundary_hook, start_batch: int
    ):
        """Run one training epoch and return an opaque ``train_result`` the core
        threads back into ``validate`` / ``on_epoch_end`` / ``epoch_message``."""

    def on_after_train(self, ctx: TaskContext, model, epoch: int) -> None:
        """After the scheduler steps (e.g. fold weights into the SWA average)."""

    @abstractmethod
    def validate(self, ctx: TaskContext, model, epoch: int, train_result) -> dict:
        """Evaluate the model on the stored val loader and return metrics."""

    @abstractmethod
    def selection_score(self, metrics: dict) -> float:
        """The scalar epoch/candidate selection score derived from ``metrics``."""

    @abstractmethod
    def save_candidate(self, ctx: TaskContext, model, epoch: int, metrics: dict, best: BestTracker) -> None:
        """Persist the improved candidate and fill ``best``'s metric-derived fields."""

    def on_epoch_end(self, ctx: TaskContext, model, epoch: int, metrics: dict, selected_score: float, best: BestTracker) -> None:
        """Post-selection per-epoch work (DCW update, snapshot dump, soup pool)."""

    def epoch_message(self, ctx: TaskContext, epoch: int, metrics: dict, train_result, selected_score: float, best: BestTracker) -> dict | None:
        """The ``epoch_complete`` payload, or None to skip emitting one."""
        return None

    def collect_extra_state(self, ctx: TaskContext, *, rng_epoch_start, schedule, batch_in_epoch: int) -> dict:
        """Task-specific fields folded into the backup envelope (ema/swa/dcw/soup/
        rng/schedule/...). Empty for tasks without resumable extras."""
        return {}

    # -- finalisation ------------------------------------------------------
    @abstractmethod
    def finalize(self, ctx: TaskContext, model, best: BestTracker, epochs_completed: int) -> dict:
        """SWA / soup finalisation, promotion vs incumbent, build the result dict."""
