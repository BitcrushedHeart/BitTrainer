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
from bittrainer.group_trainer import GroupTrainConfig, run_group_training
from bittrainer.head_only_trainer import run_head_only_training
from bittrainer.oft_trainer import run_oft_training
from bittrainer.oft import (
    OFTConv2d,
    OFTLinear,
    merge_oft_into_model,
    merged_state_dict,
    oft_parameters,
    skew_to_rotation,
    wrap_backbone_with_oft,
)
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
    "GroupTrainConfig",
    "run_group_training",
    "run_head_only_training",
    # oft
    "run_oft_training",
    "OFTLinear",
    "OFTConv2d",
    "wrap_backbone_with_oft",
    "merge_oft_into_model",
    "merged_state_dict",
    "oft_parameters",
    "skew_to_rotation",
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
