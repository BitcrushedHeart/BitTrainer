"""BitTrainer — binary concept classifier training pipeline."""

from bittrainer.model import (
    create_model,
    freeze_backbone,
    get_stages,
    load_checkpoint,
    unfreeze_backbone,
    unfreeze_stage,
)
from bittrainer.trainer import TrainConfig, run_training
from bittrainer.validation import compute_metrics, find_optimal_threshold
from bittrainer.checkpoint import compare_checkpoints, save_if_better
from bittrainer.dataset import (
    ASPECT_RATIO_BUCKETS,
    BucketBatchSampler,
    ConceptDataset,
    build_bucket_batch_sampler,
    find_nearest_bucket,
    get_heavy_augment_transform,
    get_train_transform,
    get_val_transform,
)
from bittrainer.smart_cache import CachingStoppedException, SmartCache

__all__ = [
    # model
    "create_model",
    "freeze_backbone",
    "get_stages",
    "load_checkpoint",
    "unfreeze_backbone",
    "unfreeze_stage",
    # trainer
    "TrainConfig",
    "run_training",
    # validation
    "compute_metrics",
    "find_optimal_threshold",
    # checkpoint
    "compare_checkpoints",
    "save_if_better",
    # dataset
    "ASPECT_RATIO_BUCKETS",
    "BucketBatchSampler",
    "ConceptDataset",
    "build_bucket_batch_sampler",
    "find_nearest_bucket",
    "get_heavy_augment_transform",
    "get_train_transform",
    "get_val_transform",
    # smart cache
    "SmartCache",
    "CachingStoppedException",
]
