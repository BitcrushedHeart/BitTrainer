"""``run_pairwise_reward_training`` — caption-conditioned reward models
(Bitcrush Optimise P8, ISSUE-1133).

Thin wrapper driving :class:`~bittrainer.generic.tasks.pairwise_reward_task
.PairwiseRewardTask` through the shared :class:`GenericTrainer` skeleton, the
same way ``run_head_only_training`` wraps ``HeadOnlyTask``. The config is the
whole contract Optimise's training runner fills in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class RewardTrainConfig:
    """Everything one reward-model training run needs.

    ``manifest_path`` is the pair manifest Optimise exported
    (:func:`bittrainer.pair_dataset.load_pair_manifest`); ``output_dir`` receives
    the checkpoint (``reward_best.pt``) and, unless ``split_path`` names an
    existing persisted split, ``split.json``. ``image_model_size`` must be a
    SigLIP2 size; when its text tower cannot be loaded and ``fallback_to_base``
    is on, both towers fall back to ``siglip2_base`` and the result records it.
    ``vae_prep`` is the provenance returned by
    :func:`bittrainer.pair_dataset.vae_roundtrip_prepare` when the manifest
    points at round-tripped copies; it is stored in the checkpoint untouched.
    """

    manifest_path: str
    output_dir: str
    image_model_size: str = "siglip2_so400m"
    fallback_to_base: bool = True
    image_size: int = 256
    hidden_dim: int = 512
    dropout: float = 0.1
    max_epochs: int = 30
    patience: int = 6
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    val_fraction: float = 0.2
    seed: int = 0
    split_path: str | None = None
    embedding_cache_dir: str | None = None
    embed_batch_size: int = 32
    vae_prep: dict[str, Any] | None = None
    device: str = "cuda"
    dtype: str = "float32"
    # Backup/resume plumbing fields the generic lifecycle reads; reward runs
    # train a head over cached vectors in minutes, so backups are off.
    backup_dir: str | None = None
    backup_every_steps: int = 0
    resume_from: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def run_pairwise_reward_training(
    config: RewardTrainConfig,
    *,
    progress_callback: Callable[[dict], None] | None = None,
    stop_event: Any | None = None,
    stop_now_event: Any | None = None,
    pause_event: Any | None = None,
) -> dict:
    """Train a caption-conditioned reward head on cached SigLIP2 embeddings.

    Returns the result dict described on
    :meth:`~bittrainer.generic.tasks.pairwise_reward_task.PairwiseRewardTask.finalize`
    (checkpoint path, held-out preference accuracy and margin, per-type
    accuracy, the persisted split, the fixed normalisation mapping). A set
    ``pause_event`` behaves like ``stop_now``: there is no backup/resume.
    """
    from bittrainer.generic.generic_trainer import GenericTrainer
    from bittrainer.generic.tasks.pairwise_reward_task import PairwiseRewardTask

    return GenericTrainer().run(
        PairwiseRewardTask(config),
        progress_callback=progress_callback,
        stop_event=stop_event,
        stop_now_event=stop_now_event,
        pause_event=pause_event,
    )
