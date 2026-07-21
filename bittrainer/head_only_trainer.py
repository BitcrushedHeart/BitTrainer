"""Function 1: train_head_only — train the classifier head on cached features.

Trains the head to convergence on cached backbone features (the backbone itself is
never touched), then resolves the result through the same promote-if-better path as
full fine-tune: a head-only model that beats the current one becomes the group's
deployed model; a worse one leaves the incumbent in place. Cheap by design — the
embedding cache is built once per backbone era, so re-running (or running after a
full fine-tune adapts the backbone) reuses or rebuilds vectors automatically.

The implementation lives in :class:`~bittrainer.generic.tasks.head_only_task.HeadOnlyTask`
(Bitcrush ISSUE-0542): this module is a thin wrapper that drives it through the
shared :class:`~bittrainer.generic.generic_trainer.GenericTrainer` skeleton.
"""

from __future__ import annotations

from typing import Any, Callable

from bittrainer.group_trainer import GroupTrainConfig


def run_head_only_training(
    config: GroupTrainConfig,
    *,
    progress_callback: Callable[[dict], None] | None = None,
    stop_event: Any | None = None,
    stop_now_event: Any | None = None,
    pause_event: Any | None = None,
) -> dict:
    """Train a cached-feature head probe and report per-class scores. Terminal.

    ``pause_event`` (Bitcrush ISSUE-0405) is accepted for signature uniformity
    with the full trainers; head-only training has no backup/resume (out of
    scope), so a set pause_event simply behaves like ``stop_now`` — the probe
    finishes early and returns its partial ``{"cancelled": True, ...}`` result.
    """
    from bittrainer.generic.generic_trainer import GenericTrainer
    from bittrainer.generic.tasks.head_only_task import HeadOnlyTask

    return GenericTrainer().run(
        HeadOnlyTask(config),
        progress_callback=progress_callback,
        stop_event=stop_event,
        stop_now_event=stop_now_event,
        pause_event=pause_event,
    )
