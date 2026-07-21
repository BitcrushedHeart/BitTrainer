"""Generic training lifecycle (Bitcrush ISSUE-0542 Step 3).

``GenericTrainer.run`` is the trainer-agnostic core extracted from
``run_group_training``: it owns the lifecycle *order* and the pause / stop /
backup / resume / best / patience mechanics, delegating every trainer-specific
step to :class:`~bittrainer.generic.task.TrainingTask` hooks. The group trainer's
``run_group_training`` is now a thin wrapper: ``GenericTrainer().run(GroupTask(cfg), ...)``.
"""

from __future__ import annotations

import logging
import time
from functools import partial

from bittrainer.generic.task import BestTracker, ResumeInfo, TrainingTask
from bittrainer.progress import Stage
from bittrainer.training_state import (
    capture_rng_states,
    collect_epoch_state,
    paused_result,
    restore_rng_states,
)

logger = logging.getLogger(__name__)


class GenericTrainer:
    """Runs a :class:`TrainingTask` through the full training lifecycle."""

    def run(
        self,
        task: TrainingTask,
        *,
        progress_callback=None,
        stop_event=None,
        stop_now_event=None,
        pause_event=None,
    ) -> dict:
        """Execute ``task``'s training run and return its result dict.

        Lifecycle order (hooks in call order):
        ``make_context`` → ``fingerprint_init`` → ``prepare_data`` →
        ``create_model`` → ``pre_loop`` (fresh only) → ``resolve_batch_size`` →
        ``setup_training`` → ``create_optimizer`` → [resume restore] → per epoch
        (``reshuffle`` → ``build_loaders`` → ``train_epoch`` → ``on_after_train``
        → ``validate`` → best/patience + ``save_candidate`` → ``on_epoch_end`` →
        ``epoch_message`` → epoch-boundary backup) → ``finalize``.

        ``stop_event`` breaks at the next epoch boundary; ``stop_now_event`` also
        interrupts the epoch mid-batch — both still finalise. ``pause_event``
        backs up and returns ``{"paused": True, ...}`` WITHOUT finalising.
        """
        ctx = task.make_context(progress_callback, stop_event, stop_now_event, pause_event)
        task.fingerprint_init(ctx)
        cb = ctx.cb
        coordinator = ctx.coordinator
        resume_state = ctx.resume_state
        spec = task.loop_spec()

        _paused = partial(
            paused_result, cb,
            stage="backing_up", status_text="Training paused — state backed up",
        )

        # --- Datasets + cache ---
        task.prepare_data(ctx)
        if coordinator.paused:  # pause during scan/cache — nothing to finalise
            return _paused(resume_state["epoch"] if resume_state else 0, 0, None)

        # --- Model (warm-start, or rebuild-and-load on resume) ---
        model = task.create_model(ctx, resume_state)
        if resume_state is None:
            task.pre_loop(ctx, model)
            if coordinator.paused:  # pause during warmup/sweeps
                return _paused(0, 0, None)

        # --- Batch size / balance / optimizer ---
        eff_bs = task.resolve_batch_size(ctx, model, resume_state)
        task.setup_training(ctx, model, resume_state)
        optimizer, scheduler, scheduler_t_max = task.create_optimizer(ctx, model, eff_bs, resume_state)

        best = BestTracker()
        global_step = 0
        start_epoch = 0
        # Mid-epoch resume state (one-shot; cleared after the first resumed epoch).
        resume_batch_in_epoch = 0
        resume_schedule: list[list[int]] | None = None
        resume_rng_epoch_start: dict | None = None
        resume_rng_now: dict | None = None

        if resume_state is not None:
            best.restore_from(resume_state["best"])
            task.restore_resume_extra(ctx, resume_state)
            global_step = int(resume_state.get("global_step", 0))
            resume_batch_in_epoch = int(resume_state.get("batch_in_epoch", 0))
            resume_rng_epoch_start = resume_state.get("rng_epoch_start")
            resume_rng_now = resume_state.get("rng_now")
            # ``epoch`` in the envelope is always the epoch to resume INTO (a
            # boundary backup stores epoch+1, so it never re-runs a finished epoch).
            start_epoch = int(resume_state["epoch"])
            resume_bs_changed = bool(resume_state.get("_resume_bs_changed"))
            if resume_batch_in_epoch > 0 and not resume_bs_changed:
                resume_schedule = resume_state.get("batch_schedule")
            else:
                resume_batch_in_epoch = 0
            msg = task.resumed_message(ctx, best, global_step, start_epoch)
            if msg is not None:
                cb(msg)

        # Shared mutable epoch-start RNG snapshot (folded into every backup).
        rng_epoch_start: dict | None = None

        def _collect_state(cur_epoch: int, batch_in_epoch: int, schedule) -> dict:
            """Assemble the schema-v1 backup envelope from the live run state."""
            return collect_epoch_state(
                fingerprint=ctx.fingerprint, trainer=task.trainer_name, epoch=cur_epoch,
                batch_in_epoch=batch_in_epoch, global_step=global_step, eff_bs=eff_bs,
                scheduler_t_max=scheduler_t_max,
                model=model, optimizer=optimizer, scheduler=scheduler,
                best=best.as_backup_dict(),
                **task.collect_extra_state(
                    ctx, rng_epoch_start=rng_epoch_start,
                    schedule=schedule, batch_in_epoch=batch_in_epoch,
                ),
            )

        epoch = start_epoch - 1  # so epochs_completed is defined if the loop is empty
        exc_epoch = [start_epoch]  # boxed so the exception hook sees the live value

        with coordinator.backup_on_exception(lambda: _collect_state(exc_epoch[0], 0, None)):
            for epoch in range(start_epoch, spec.max_epochs):
                exc_epoch[0] = epoch
                if stop_now_event is not None and stop_now_event.is_set():
                    logger.info("Stop-now requested before epoch %d — running final comparison", epoch)
                    cb({"type": "stop_now", "epoch": epoch, "max_epochs": spec.max_epochs})
                    break
                if stop_event is not None and stop_event.is_set():
                    logger.info("Graceful stop requested after epoch %d — running final comparison", epoch)
                    cb({"type": "graceful_stop", "epoch": epoch, "max_epochs": spec.max_epochs})
                    break
                if coordinator.paused:
                    # Pause at an epoch boundary (before any training this epoch):
                    # back up a clean batch_in_epoch=0 snapshot and stop.
                    path = coordinator.save(_collect_state(epoch, 0, None), reason="pause")
                    return _paused(epoch, global_step, path)

                # Exact mid-epoch continuation only for the FIRST resumed epoch.
                mid_resume = (
                    epoch == start_epoch and resume_schedule is not None and resume_batch_in_epoch > 0
                )

                # Capture the epoch-start RNG BEFORE reshuffle so the sample layout
                # is reproducible; on a mid-epoch resume restore the stored
                # epoch-start RNG instead so reshuffle rebuilds the identical list.
                if mid_resume:
                    restore_rng_states(resume_rng_epoch_start, ctx.device)
                    rng_epoch_start = resume_rng_epoch_start
                else:
                    if epoch == start_epoch and resume_rng_now is not None:
                        # Epoch-boundary resume: continue the RNG stream from the
                        # backup point so this fresh epoch matches the control run.
                        restore_rng_states(resume_rng_now, ctx.device)
                    rng_epoch_start = capture_rng_states(ctx.device)
                task.reshuffle()

                if epoch == 0:
                    cb({
                        "type": "training_progress", "stage": "preparing",
                        "status_text": f"Batch size {eff_bs} — spawning data workers",
                    })

                resume_info = ResumeInfo(
                    mid_resume=mid_resume,
                    resume_schedule=resume_schedule,
                    resume_batch_in_epoch=resume_batch_in_epoch,
                    resume_rng_now=resume_rng_now,
                )
                train_loader, schedule, start_batch = task.build_loaders(ctx, epoch, eff_bs, resume_info)
                # One-shot: subsequent epochs are ordinary.
                resume_schedule = None

                # Top-of-epoch model/optimizer reshape (binary's epoch-1 unfreeze
                # + scheduler rebuild). A returned tuple replaces the core's three;
                # None keeps them. The nested backup collectors close over these
                # run-locals, so the reassignment is picked up by later backups.
                rebuilt = task.on_epoch_start(
                    ctx, model, epoch,
                    optimizer=optimizer, scheduler=scheduler,
                    scheduler_t_max=scheduler_t_max, start_epoch=start_epoch,
                )
                if rebuilt is not None:
                    optimizer, scheduler, scheduler_t_max = rebuilt

                epoch_start_mono = time.monotonic()
                step_callback = task.make_step_callback(ctx, epoch, eff_bs, best, epoch_start_mono)

                def _boundary_hook(num_batches: int, _epoch=epoch, _schedule=schedule) -> str | None:
                    # Fires at every gradient-accumulation boundary. Owns the global
                    # optimizer-step counter and the pause/periodic backup cadence.
                    nonlocal global_step
                    global_step += 1
                    return coordinator.on_boundary(
                        lambda: _collect_state(_epoch, num_batches, _schedule), global_step,
                    )

                train_result = task.train_epoch(
                    ctx, model, optimizer, train_loader,
                    step_callback=step_callback, boundary_hook=_boundary_hook, start_batch=start_batch,
                )

                if coordinator.paused:
                    # Pause fired mid-epoch — the boundary hook already wrote the
                    # backup. Return WITHOUT finalisation so a pause never ships.
                    return _paused(epoch, global_step, coordinator.last_backup_path)

                if stop_now_event is not None and stop_now_event.is_set():
                    cb({
                        "type": "stop_now", "epoch": epoch + 1, "max_epochs": spec.max_epochs,
                        "status_text": f"Stop-now triggered mid-epoch {epoch + 1} — finishing up",
                    })
                scheduler.step()
                task.on_after_train(ctx, model, epoch)

                ctx.em.stage(
                    Stage.validating,
                    f"Validating (epoch {epoch + 1}/{spec.max_epochs})",
                    epoch=epoch + 1, max_epochs=spec.max_epochs,
                )
                metrics = task.validate(ctx, model, epoch, train_result)

                selected_score = task.selection_score(metrics)
                # Require a minimum gain so epoch-to-epoch noise can't flip the
                # selection onto a marginally-higher but less robust checkpoint.
                improved = selected_score > best.best_validation_score + spec.selection_min_delta
                if improved:
                    best.best_validation_score = selected_score
                    best.best_epoch = epoch
                    best.patience_counter = 0
                    task.save_candidate(ctx, model, epoch, metrics, best)
                else:
                    best.patience_counter += 1

                task.on_epoch_end(ctx, model, epoch, metrics, selected_score, best)

                msg = task.epoch_message(ctx, epoch, metrics, train_result, selected_score, best)
                if msg is not None:
                    cb(msg)

                # Epoch-boundary backup: the cleanest resume point. Stored as
                # epoch+1/batch_in_epoch=0 so a resume starts the NEXT epoch fresh.
                if coordinator.enabled:
                    coordinator.save(_collect_state(epoch + 1, 0, None), reason="periodic")

                if best.patience_counter >= spec.patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, spec.patience)
                    break

            result = task.finalize(ctx, model, best, epochs_completed=epoch + 1)

        # Training completed successfully (no pause, no exception): the backups are
        # obsolete — clear them so a later run doesn't resume a finished job.
        coordinator.delete_backups()
        return result
