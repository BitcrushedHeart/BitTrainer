"""Cached-feature head-only training for binary concepts.

Unlike ``frozen_backbone``, this path forwards each image through the backbone
once per backbone era, persists the pooled vectors, and trains only the tiny
classifier tail on those vectors. It mirrors the group head-only contract while
retaining binary threshold tuning and promotion semantics.
"""

from __future__ import annotations

from typing import Any, Callable

from bittrainer.trainer import TrainConfig


def run_binary_head_only_training(
    config: TrainConfig,
    *,
    progress_callback: Callable[[dict], None] | None = None,
    stop_event: Any | None = None,
    stop_now_event: Any | None = None,
    pause_event: Any | None = None,
) -> dict:
    """Build/reuse pooled features and train the binary classifier head."""
    from bittrainer.generic.generic_trainer import GenericTrainer
    from bittrainer.generic.tasks.binary_head_only_task import BinaryHeadOnlyTask

    return GenericTrainer().run(
        BinaryHeadOnlyTask(config),
        progress_callback=progress_callback,
        stop_event=stop_event,
        stop_now_event=stop_now_event,
        pause_event=pause_event,
    )
