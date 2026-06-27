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


# --- __none__ is outside class balancing -----------------------------------


def test_none_excluded_from_positive_balance(tmp_path):
    # __none__ must not drive the positive equalisation target, and positives must
    # not scale up to match a large __none__. Negatives stay at natural count.
    root = tmp_path / "g"
    _build(root, {"a": 2, "b": 20, "__none__": 100})
    random.seed(0)
    ds = GroupDataset(
        root, ["a", "b", "__none__"], split="train", group_name="g",
        oversample_max_ratio=4.0,  # oversample_none defaults off
    )
    counts = _label_counts(ds)
    assert counts[0] == 8  # ceil(4*2) vs positive max_count=20, NOT vs 100
    assert counts[1] == 20  # largest positive class
    assert counts[2] == 100  # every negative, natural count, not equalised/capped


def test_none_oversampled_to_one_to_one_when_underpopulated(tmp_path):
    # Toggle on: __none__ raised to the combined positive total (10 + 10 = 20).
    root = tmp_path / "g"
    _build(root, {"a": 10, "b": 10, "__none__": 5})
    random.seed(0)
    ds = GroupDataset(
        root, ["a", "b", "__none__"], split="train", group_name="g",
        oversample_none=True,
    )
    counts = _label_counts(ds)
    assert counts[0] == 10
    assert counts[1] == 10
    assert counts[2] == 20  # 5 -> 20 (1:1 vs 20 positives)


def test_none_never_downsampled(tmp_path):
    # Toggle on but negatives already exceed the positive total: no-op.
    root = tmp_path / "g"
    _build(root, {"a": 10, "b": 10, "__none__": 50})
    random.seed(0)
    ds = GroupDataset(
        root, ["a", "b", "__none__"], split="train", group_name="g",
        oversample_none=True,
    )
    counts = _label_counts(ds)
    assert counts[2] == 50  # kept all, not reduced to 20
