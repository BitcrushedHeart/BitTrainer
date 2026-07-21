"""Extraction shims (Bitcrush ISSUE-0542): moved helpers stay importable.

The unification splits group_trainer/trainer helpers into dedicated modules
(selection, soft_labels, probes, priors, finalize, collate, generic.optimizer,
generic.evaluation). Every previously-importable name must remain importable
from its old home AND be the very same object as the new home's export, so
head_only_trainer, Engine shims and older pickles keep working.
"""

from __future__ import annotations

import importlib

import pytest

# (new module, names) — the mapping the extraction must satisfy. Each name must
# also still resolve on bittrainer.group_trainer to the identical object.
_MOVES = [
    ("bittrainer.selection", ["_metric_score", "_primary_validation_metric"]),
    ("bittrainer.soft_labels", ["_build_gaussian_kernel", "_build_soft_targets", "_soft_ce_loss"]),
    ("bittrainer.probes", ["_run_auto_softness_probe", "_run_auto_oversample_probe"]),
    ("bittrainer.priors", ["_apply_and_persist_priors"]),
    ("bittrainer.finalize", ["_compare_promote_finalize", "_finalise_ordinal_decode"]),
    ("bittrainer.collate", ["_collate_bucket_batch", "_collate_multilabel_batch"]),
    ("bittrainer.generic.evaluation", ["_evaluate", "_metrics_from_logits", "_collect_val_logits", "_shipped_decode_metrics"]),
]


@pytest.mark.parametrize(("module_name", "names"), _MOVES)
def test_moved_names_identical_in_both_homes(module_name, names):
    new_mod = importlib.import_module(module_name)
    old_mod = importlib.import_module("bittrainer.group_trainer")
    for name in names:
        new_obj = getattr(new_mod, name)
        old_obj = getattr(old_mod, name)
        assert new_obj is old_obj, f"{name}: {module_name} and group_trainer diverged"


def test_head_only_import_surface_survives():
    """The exact import list head_only_trainer relies on."""
    from bittrainer.group_trainer import (  # noqa: F401
        GroupTrainConfig,
        _collate_bucket_batch,
        _collate_multilabel_batch,
        _compare_promote_finalize,
        _create_or_warmstart_model,
        _evaluate,
        _get_dtype,
        _metric_score,
        _prepare_datasets_and_cache,
        _primary_validation_metric,
        _resolve_none_index,
        _run_auto_oversample_probe,
        _run_auto_softness_probe,
        _spatial_ckpt_meta,
    )


def test_engine_shim_names_importable():
    """Engine's services/training shims import these by name."""
    from bittrainer.group_trainer import GroupTrainConfig, run_group_training  # noqa: F401
    from bittrainer.head_only_trainer import run_head_only_training  # noqa: F401
    from bittrainer.trainer import (  # noqa: F401
        TrainConfig,
        evaluate,
        run_training,
        train_one_epoch,
    )
