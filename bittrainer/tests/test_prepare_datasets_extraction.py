"""Characterization test pinning _prepare_datasets_and_cache to direct dataset construction.

The helper was extracted out of run_group_training. With caching/face-crop off it
must produce exactly the datasets a direct GroupDataset(...) pair would, so the
full-train path can't silently drift when both functions start sharing it.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image

from bittrainer.group_dataset import GroupDataset
from bittrainer.group_trainer import GroupTrainConfig, _prepare_datasets_and_cache


def _make_image(path, w, h, seed):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _build_group(root, classes, per_class_split, sizes):
    for ci, cname in enumerate(classes):
        for split, count in per_class_split.items():
            d = root / cname_dir(cname) / split
            d.mkdir(parents=True, exist_ok=True)
            for j in range(count):
                w, h = sizes[(ci + j) % len(sizes)]
                _make_image(d / f"img_{ci}_{split}_{j}.png", w, h, seed=ci * 100 + j)


def cname_dir(name: str) -> str:
    return name


def _key(sample) -> tuple:
    label = sample["label"]
    if hasattr(label, "tolist"):
        label = tuple(label.tolist())
    return (sample["path"], label, tuple(sample["bucket"]))


def test_helper_matches_direct_construction(tmp_path):
    classes = ["a", "b", "c"]
    sizes = [(64, 64), (48, 80), (96, 64)]
    root = tmp_path / "group"
    _build_group(root, classes, {"train": 5, "val": 3}, sizes)

    config = GroupTrainConfig(
        group_folder=str(root),
        num_classes=len(classes),
        class_names=classes,
        use_cache=False,
        face_model_path="",
        group_name="group",
    )

    # Reference: what run_group_training used to build inline.
    random.seed(1234)
    ref_train = GroupDataset(
        root, classes, split="train", multi_label=False,
        skin_normalise=False, group_name="group",
        oversample_none=False, extra_paths={},
    )
    ref_val = GroupDataset(
        root, classes, split="val", multi_label=False,
        skin_normalise=False, group_name="group",
        oversample_none=False, extra_paths={},
    )

    random.seed(1234)
    train_ds, val_ds, smart_cache, bucket_counts = _prepare_datasets_and_cache(
        config, cb=lambda _msg: None, stop_event=None,
    )

    assert smart_cache is None
    assert [_key(s) for s in train_ds.samples] == [_key(s) for s in ref_train.samples]
    assert sorted(_key(s) for s in val_ds.samples) == sorted(_key(s) for s in ref_val.samples)

    # bucket_counts is derived from the train samples
    expected_counts: dict = {}
    for s in ref_train.samples:
        b = tuple(s["bucket"])
        expected_counts[b] = expected_counts.get(b, 0) + 1
    assert {tuple(k): v for k, v in bucket_counts.items()} == expected_counts


def test_helper_matches_direct_construction_multilabel(tmp_path):
    classes = ["x", "y", "z"]
    sizes = [(64, 64), (80, 48)]
    root = tmp_path / "ml_group"
    _build_group(root, classes, {"train": 4, "val": 2}, sizes)

    config = GroupTrainConfig(
        group_folder=str(root),
        num_classes=len(classes),
        class_names=classes,
        multi_label=True,
        use_cache=False,
        face_model_path="",
        group_name="ml_group",
    )

    random.seed(99)
    ref_train = GroupDataset(
        root, classes, split="train", multi_label=True,
        skin_normalise=False, group_name="ml_group",
        oversample_none=False, extra_paths={},
    )

    random.seed(99)
    train_ds, val_ds, smart_cache, _ = _prepare_datasets_and_cache(
        config, cb=lambda _msg: None, stop_event=None,
    )

    assert smart_cache is None
    assert sorted(_key(s) for s in train_ds.samples) == sorted(_key(s) for s in ref_train.samples)
