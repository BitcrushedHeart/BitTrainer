"""End-to-end migration guard for the dual-branch trainer on GenericTrainer
(Bitcrush ISSUE-0542 Step 7).

A tiny CPU run through ``run_dual_branch_training`` (now a thin
``GenericTrainer().run(DualBranchTask(cfg))`` wrapper) pins the result-dict keys
Engine's ``_group_training_worker`` consumes for ``classifier_mode == "dual_branch"``
and that the promoted checkpoint carries the dual-branch metadata unchanged. The
dual-branch finalisation builds its OWN result dict + compare-promote (ISSUE-0490)
and must stay separate from group calibration.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image


def _seed(n: int = 0) -> None:
    torch.manual_seed(n)
    np.random.seed(n)
    random.seed(n)


def _make_labelled_group(root, classes, *, per_class=6, seed=0):
    rng = np.random.default_rng(seed)
    for split, n in (("train", per_class), ("val", max(2, per_class // 2))):
        for cname in classes:
            d = root / cname / split
            d.mkdir(parents=True, exist_ok=True)
            for j in range(n):
                Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
                    d / f"{cname}_{j}.png"
                )


_CLASSES = ["a", "b", "c"]

# Result keys Engine persists for a dual-branch run (see the trainer's result dict).
_RESULT_KEYS = {
    "epochs_completed",
    "best_epoch",
    "best_val_macro_f1",
    "final_val_macro_f1",
    "final_val_loss",
    "per_class_f1",
    "confusion_matrix",
    "balanced_accuracy",
    "checkpoint_path",
    "class_counts",
    "total_images",
}


def _cfg(tmp_path, checkpoint_dir, **kw):
    from bittrainer.dual_branch_trainer import DualBranchTrainConfig

    base = dict(
        group_folder=str(tmp_path / "crops"),
        context_folder=str(tmp_path / "context"),
        num_classes=len(_CLASSES),
        class_names=list(_CLASSES),
        checkpoint_dir=str(checkpoint_dir),
        max_epochs=2,
        patience=99,
        backbone_variant="atto",
        device="cpu",
        dtype="float32",
        batch_size=4,
        use_compile=False,
        channels_last=False,
        from_scratch=True,
        dataloader_workers=0,
        backbone_init={"source": "random_init", "checkpoint_path": None},
    )
    base.update(kw)
    return DualBranchTrainConfig(**base)


def test_dual_branch_task_is_used_by_the_wrapper():
    from bittrainer.generic.tasks.dual_branch_task import DualBranchTask

    assert DualBranchTask.trainer_name == "dual_branch"


def test_dual_branch_run_result_keys_and_checkpoint(tmp_path):
    from bittrainer.dual_branch_trainer import run_dual_branch_training

    _make_labelled_group(tmp_path / "crops", _CLASSES, per_class=6, seed=0)
    _make_labelled_group(tmp_path / "context", _CLASSES, per_class=6, seed=0)
    _seed(0)
    result = run_dual_branch_training(_cfg(tmp_path, tmp_path / "ck"))

    assert "paused" not in result
    assert _RESULT_KEYS.issubset(result.keys()), _RESULT_KEYS - result.keys()
    assert result["epochs_completed"] == 2
    assert result["total_images"] == len(_CLASSES) * 6
    # Per-class train sample counts land keyed by class index.
    assert set(result["class_counts"]) == set(range(len(_CLASSES)))

    # The promoted checkpoint is best.pt and carries the dual-branch metadata
    # (the finalisation persists epoch + val_macro_f1 into the checkpoint).
    ckpt_path = result["checkpoint_path"]
    assert ckpt_path is not None and ckpt_path.endswith("best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert ckpt["classifier_mode"] == "dual_branch"
    meta = ckpt["metadata"]
    assert meta["num_classes"] == len(_CLASSES)
    assert "val_macro_f1" in meta
    assert {"crop_branch", "context_branch", "head"}.issubset(ckpt.keys())
