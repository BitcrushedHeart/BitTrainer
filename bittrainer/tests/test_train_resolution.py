"""Per-group / per-concept training-resolution override (Bitcrush ISSUE-0549).

``train_resolution`` scales the canonical ~512px aspect-bucket table for one
group/concept (snapped to /32); 512 is the identity and keeps every existing
SmartCache entry valid (cache keys embed the bucket dims, so any other value
simply produces fresh entries). The embedding-cache era is namespaced by
resolution through ``preproc_sig`` so head probes never silently reuse vectors
built from differently-sized buckets.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from bittrainer.dataset import (
    ASPECT_RATIO_BUCKETS,
    DEFAULT_TRAIN_RESOLUTION,
    ConceptDataset,
    find_nearest_bucket,
    scaled_buckets,
)
from bittrainer.group_dataset import GroupDataset
from bittrainer.group_trainer import (
    GroupTrainConfig,
    _embedding_preproc_sig,
    _prepare_datasets_and_cache,
)
from bittrainer.trainer import TrainConfig


def test_default_resolution_is_identity():
    assert scaled_buckets(512) is ASPECT_RATIO_BUCKETS
    assert scaled_buckets(0) is ASPECT_RATIO_BUCKETS
    assert scaled_buckets(None) is ASPECT_RATIO_BUCKETS


def test_scaled_buckets_snap_to_32():
    table = scaled_buckets(768)  # 1.5x
    assert table[0] == (768, 768)
    for (w, h), (bw, bh) in zip(table, ASPECT_RATIO_BUCKETS):
        assert w % 32 == 0 and h % 32 == 0
        assert abs(w / h - bw / bh) < 0.25  # aspect roughly preserved
    # Downscale too.
    assert scaled_buckets(256)[0] == (256, 256)


def test_find_nearest_bucket_resolution_kwarg():
    # Square image: canonical 512 bucket vs scaled.
    assert find_nearest_bucket(1000, 1000) == (512, 512)
    assert find_nearest_bucket(1000, 1000, 768) == (768, 768)
    # Wide image keeps its aspect bucket, scaled.
    canonical = find_nearest_bucket(1600, 900)
    scaled = find_nearest_bucket(1600, 900, 768)
    idx = ASPECT_RATIO_BUCKETS.index(canonical)
    assert scaled == scaled_buckets(768)[idx]


def test_config_defaults():
    assert TrainConfig(concept_folder="x").train_resolution == 512
    assert (
        GroupTrainConfig(group_folder="x", num_classes=2, class_names=["a", "b"]).train_resolution
        == 512
    )


def test_embedding_preproc_sig():
    # 512 keeps the historical default sig -> bare-hash era dir, existing
    # caches stay valid.
    assert _embedding_preproc_sig(512) == "val_imagenet"
    assert _embedding_preproc_sig(0) == "val_imagenet"
    assert _embedding_preproc_sig(768) == "val_imagenet@768"


def _write_images(folder, n, seed=0):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        arr = np.random.default_rng(seed + i).integers(0, 256, (64, 64, 3)).astype(np.uint8)
        Image.fromarray(arr).save(folder / f"img{i}.png")


def test_concept_dataset_uses_resolution(tmp_path):
    root = tmp_path / "concept"
    _write_images(root / "train", 3)
    _write_images(root / "negative" / "train", 3, seed=50)
    ds = ConceptDataset(root, split="train", train_resolution=256)
    assert ds.samples
    assert all(s["bucket"] == (256, 256) for s in ds.samples)
    # Default stays canonical.
    ds_default = ConceptDataset(root, split="train")
    assert all(s["bucket"] == (512, 512) for s in ds_default.samples)


def test_group_dataset_uses_resolution(tmp_path):
    root = tmp_path / "group"
    _write_images(root / "a" / "train", 3)
    _write_images(root / "b" / "train", 3, seed=50)
    ds = GroupDataset(root, ["a", "b"], split="train", group_name="g", train_resolution=768)
    assert ds.samples
    assert all(s["bucket"] == (768, 768) for s in ds.samples)


def test_group_config_resolution_reaches_datasets(tmp_path):
    """End-to-end through _prepare_datasets_and_cache: the config field lands
    on every train AND val sample bucket."""
    root = tmp_path / "group"
    _write_images(root / "a" / "train", 3)
    _write_images(root / "b" / "train", 3, seed=50)
    _write_images(root / "a" / "val", 2, seed=100)
    _write_images(root / "b" / "val", 2, seed=150)
    config = GroupTrainConfig(
        group_folder=str(root),
        num_classes=2,
        class_names=["a", "b"],
        use_cache=False,
        train_resolution=256,
        group_name="g",
    )
    train_ds, val_ds, smart_cache, bucket_counts = _prepare_datasets_and_cache(
        config, cb=lambda _msg: None, stop_event=None,
    )
    assert smart_cache is None
    assert all(s["bucket"] == (256, 256) for s in train_ds.samples)
    assert all(s["bucket"] == (256, 256) for s in val_ds.samples)
    assert set(bucket_counts) == {(256, 256)}
