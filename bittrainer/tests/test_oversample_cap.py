"""The hardcoded global oversample cap bounds minority-class replication.

CPU-only (just reads image sizes and builds the sample list) — safe to run with
CUDA hidden alongside a running GPU train.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from PIL import Image

from bittrainer.group_dataset import _MAX_OVERSAMPLE_FACTOR, GroupDataset


def _build(root, counts):
    for ci, (cname, n) in enumerate(counts.items()):
        d = root / cname / "train"
        d.mkdir(parents=True, exist_ok=True)
        for j in range(n):
            rng = np.random.default_rng(ci * 1000 + j)
            Image.fromarray(rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)).save(
                d / f"img_{ci}_{j}.png"
            )


def _counts(ds):
    return dict(Counter(s["label"] for s in ds.samples))


def test_cap_is_four():
    assert _MAX_OVERSAMPLE_FACTOR == 4.0


def test_sparse_class_capped(tmp_path):
    root = tmp_path / "g"
    _build(root, {"a": 2, "b": 20})  # max_count = 20
    random.seed(0)
    ds = GroupDataset(root, ["a", "b"], split="train", group_name="g")
    c = _counts(ds)
    assert c[0] == 8  # ceil(4 * 2), not the full 20
    assert c[1] == 20  # largest class untouched


def test_mild_imbalance_still_equalised(tmp_path):
    root = tmp_path / "g"
    _build(root, {"a": 10, "b": 20})  # within 4x -> full equalisation
    random.seed(0)
    ds = GroupDataset(root, ["a", "b"], split="train", group_name="g")
    c = _counts(ds)
    assert c[0] == 20
    assert c[1] == 20
