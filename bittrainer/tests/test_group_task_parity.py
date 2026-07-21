"""GroupTask end-to-end parity (Bitcrush ISSUE-0542 Step 3).

``run_group_training`` is now a thin wrapper over ``GenericTrainer().run(GroupTask(...))``.
This drives a tiny CPU group run through the public entry point and asserts the
result dict still carries every key the Engine reads, and that the resume
fingerprint inputs are byte-stable (make_fingerprint over a known config).
"""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image

from bittrainer.group_trainer import GroupTrainConfig, run_group_training
from bittrainer.training_state import make_fingerprint

_CLASSES = ["a", "b", "__none__"]


def _make_group(root, *, per_class=6, seed=0):
    rng = np.random.default_rng(seed)
    for split, n in (("train", per_class), ("val", max(2, per_class // 2))):
        for cname in _CLASSES:
            d = root / cname / split
            d.mkdir(parents=True, exist_ok=True)
            for j in range(n):
                Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
                    d / f"{cname}_{j}.png"
                )


def _cfg(group_folder, checkpoint_dir, **kw) -> GroupTrainConfig:
    base = dict(
        group_folder=str(group_folder),
        num_classes=len(_CLASSES),
        class_names=list(_CLASSES),
        checkpoint_dir=str(checkpoint_dir),
        max_epochs=2,
        patience=99,
        backbone_variant="atto",
        device="cpu",
        dtype="float32",
        batch_size=2,
        use_cache=False,
        use_compile=False,
        channels_last=False,
        auto_label_softness=False,
        auto_oversample_none=False,
        use_greedy_soup=False,
        dataloader_workers=0,
        head_max_epochs=2,
        head_patience=1,
        backbone_init={"source": "random_init", "checkpoint_path": None},
    )
    base.update(kw)
    return GroupTrainConfig(**base)


def _seed(n: int = 0) -> None:
    torch.manual_seed(n)
    np.random.seed(n)
    random.seed(n)


def test_group_run_returns_engine_result_keys(tmp_path):
    _make_group(tmp_path / "grp", per_class=6, seed=1)
    _seed(0)
    result = run_group_training(_cfg(tmp_path / "grp", tmp_path / "ck"))

    assert "paused" not in result
    # Keys the Engine reads off the run result (see _compare_promote_finalize).
    for key in (
        "checkpoint_path",
        "epochs_completed",
        "best_epoch",
        "best_val_macro_f1",
        "validation_metric",
        "selected_validation_score",
        "final_val_macro_f1",
        "promotion_reason",
        "per_class_f1",
        "class_counts",
        "total_images",
    ):
        assert key in result, f"missing result key: {key}"

    assert result["epochs_completed"] == 2
    assert result["checkpoint_path"].endswith("best.pt")
    # First-ever training promotes (no incumbent).
    assert result["promotion_reason"] is not None


def test_fingerprint_inputs_are_byte_stable():
    """The resume fingerprint schema/values for a known config must not drift
    across the Step-3 refactor (the resume suite is the deeper net)."""
    fp = make_fingerprint(
        class_names=["a", "b", "__none__"],
        num_classes=3,
        max_epochs=2,
        multi_label=False,
        ordinal=False,
        best_model_name="best.pt",
        model_size="atto",
    )
    assert fp == {
        "class_names": ["a", "b", "__none__"],
        "num_classes": 3,
        "max_epochs": 2,
        "multi_label": False,
        "ordinal": False,
        "best_model_name": "best.pt",
        "model_size": "atto",
    }
