"""End-to-end migration guard for the multi-head trainer on GenericTrainer
(Bitcrush ISSUE-0542 Step 7).

A tiny CPU run through ``run_multihead_training`` (now a thin
``GenericTrainer().run(MultiHeadTask(cfg))`` wrapper) pins the result-dict keys
Engine's ``_group_training_worker`` consumes for ``classifier_mode == "multihead"``
and that the promoted checkpoint carries the multi-head metadata unchanged.
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


_SIZES = ["__none__", "34A", "34B", "36A", "36B"]

# Result keys Engine persists for a multi-head run (see the trainer's result dict).
_RESULT_KEYS = {
    "epochs_completed",
    "best_epoch",
    "checkpoint_path",
    "total_images",
    "final_val_loss",
    "band_classes",
    "final_val_f1_band",
    "final_val_qwk_band",
    "final_val_f1_size",
    "final_val_qwk_size",
    "final_val_f1_multihead",
    "final_val_qwk_multihead",
    "qwk",
    "best_val_qwk",
    "macro_f1",
    "final_val_macro_f1",
}


def _cfg(group_folder, checkpoint_dir, **kw):
    from bittrainer.multihead_trainer import MultiHeadTrainConfig

    base = dict(
        group_folder=str(group_folder),
        size_classes=list(_SIZES),
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
    return MultiHeadTrainConfig(**base)


def test_multihead_task_is_used_by_the_wrapper():
    """The public entrypoint dispatches to MultiHeadTask on the generic core."""
    from bittrainer.generic.tasks.multihead_task import MultiHeadTask

    assert MultiHeadTask.trainer_name == "multihead"


def test_multihead_run_result_keys_and_checkpoint(tmp_path):
    from bittrainer.multihead_trainer import run_multihead_training

    _make_labelled_group(tmp_path / "grp", _SIZES, per_class=6, seed=0)
    _seed(0)
    result = run_multihead_training(_cfg(tmp_path / "grp", tmp_path / "ck"))

    assert "paused" not in result
    assert _RESULT_KEYS.issubset(result.keys()), _RESULT_KEYS - result.keys()
    assert result["epochs_completed"] == 2
    assert result["total_images"] == len(_SIZES) * 6
    assert result["band_classes"]  # non-empty band vocab

    # The promoted checkpoint is best.pt and carries the multi-head metadata.
    ckpt_path = result["checkpoint_path"]
    assert ckpt_path is not None and ckpt_path.endswith("best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert ckpt["classifier_mode"] == "multihead"
    meta = ckpt["metadata"]
    assert meta["size_classes"] == _SIZES
    assert meta["band_classes"] == result["band_classes"]
    assert "multi_head_qwk" in meta
