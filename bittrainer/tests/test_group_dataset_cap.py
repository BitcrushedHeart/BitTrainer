"""Disk-level test that GroupDataset honours the replication cap on the train split.

Imports torch via GroupDataset but is CPU-only (just reads image sizes and builds the
sample list) — safe to run with CUDA hidden alongside a running GPU train.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from PIL import Image

from bittrainer.group_dataset import GroupDataset


def _make_image(path, seed):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)).save(path)


def _build(root, counts):
    for ci, (cname, n) in enumerate(counts.items()):
        d = root / cname / "train"
        d.mkdir(parents=True, exist_ok=True)
        for j in range(n):
            _make_image(d / f"img_{ci}_{j}.png", seed=ci * 1000 + j)


def _label_counts(ds) -> dict[int, int]:
    return dict(Counter(s["label"] for s in ds.samples))


def test_uncapped_full_equalisation(tmp_path):
    root = tmp_path / "g"
    _build(root, {"a": 2, "b": 20})  # max_count = 20
    random.seed(0)
    ds = GroupDataset(root, ["a", "b"], split="train", group_name="g")
    counts = _label_counts(ds)
    assert counts[0] == 20  # minority replicated all the way up (legacy default)
    assert counts[1] == 20


def test_cap_limits_minority_replication(tmp_path):
    root = tmp_path / "g"
    _build(root, {"a": 2, "b": 20})  # max_count = 20
    random.seed(0)
    ds = GroupDataset(
        root, ["a", "b"], split="train", group_name="g", oversample_max_ratio=4.0
    )
    counts = _label_counts(ds)
    assert counts[0] == 8  # ceil(4 * 2), not 20
    assert counts[1] == 20  # largest class untouched


def test_cap_leaves_mild_imbalance_fully_equalised(tmp_path):
    root = tmp_path / "g"
    _build(root, {"a": 10, "b": 20})  # within 4x -> still equalised
    random.seed(0)
    ds = GroupDataset(
        root, ["a", "b"], split="train", group_name="g", oversample_max_ratio=4.0
    )
    counts = _label_counts(ds)
    assert counts[0] == 20
    assert counts[1] == 20
