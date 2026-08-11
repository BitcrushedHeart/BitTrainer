"""Bitcrush ISSUE-0742: 3:1 implied-negative floor + embedding-cache-seeded negatives."""

from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image

from bittrainer.dataset import (
    MIN_BINARY_NEG_POS_RATIO,
    effective_binary_neg_pos_ratio,
)
from bittrainer.embedding_cache import _sig_suffix, cached_hashes_for_sig
from bittrainer.generic.tasks.binary_head_only_task import (
    BinaryHeadOnlyTask,
    _cached_negative_predicate,
    _sample_negative_pool,
)
from bittrainer.smart_cache import CACHE_VERSION
from bittrainer.trainer import TrainConfig


def test_binary_floor_is_three_to_one() -> None:
    assert MIN_BINARY_NEG_POS_RATIO == 3.0
    assert effective_binary_neg_pos_ratio(None) == 3.0
    assert effective_binary_neg_pos_ratio(2.0) == 3.0
    assert effective_binary_neg_pos_ratio(5.0) == 5.0
    assert TrainConfig(concept_folder="unused").neg_pos_ratio == 3.0


def _era(root: Path, name: str, hashes: list[str]) -> None:
    era = root / name
    era.mkdir(parents=True)
    for content_hash in hashes:
        (era / f"{content_hash}.npy").write_bytes(b"")


def test_cached_hashes_for_sig_scans_only_matching_era_dirs(tmp_path: Path) -> None:
    _era(tmp_path, "a" * 16, ["hash-one", "hash-two"])
    _era(tmp_path, "b" * 16, ["hash-three"])
    other_sig = "val_imagenet@448"
    _era(tmp_path, "c" * 16 + _sig_suffix(other_sig), ["hash-four"])
    junk = tmp_path / "not-an-era"
    junk.mkdir()
    (junk / "hash-five.npy").write_bytes(b"")

    assert cached_hashes_for_sig(tmp_path, "val_imagenet") == {
        "hash-one",
        "hash-two",
        "hash-three",
    }
    assert cached_hashes_for_sig(tmp_path, other_sig) == {"hash-four"}
    assert cached_hashes_for_sig(tmp_path / "missing", "val_imagenet") == frozenset()


def test_sample_negative_pool_prefers_cached_then_fills() -> None:
    pool = [f"cached-{index}.png" for index in range(30)] + [
        f"fresh-{index}.png" for index in range(70)
    ]

    def is_cached(path: str) -> bool:
        return path.startswith("cached-")

    # Quota fits inside the cache-hit tier: sample entirely within it.
    sampled = _sample_negative_pool(
        pool, positive_count=5, neg_pos_ratio=3.0, is_cached=is_cached
    )
    assert len(sampled) == 15
    assert all(is_cached(path) for path in sampled)

    # Quota exceeds the tier: every hit is kept, remainder from the fresh tier.
    sampled = _sample_negative_pool(
        pool, positive_count=20, neg_pos_ratio=3.0, is_cached=is_cached
    )
    assert len(sampled) == 60
    assert sum(is_cached(path) for path in sampled) == 30

    # Pool no larger than the quota: everything is used, no preference needed.
    small = pool[:10]
    assert (
        _sample_negative_pool(
            small, positive_count=20, neg_pos_ratio=3.0, is_cached=is_cached
        )
        == small
    )


def test_cached_negative_predicate_reads_smart_cache_index(tmp_path: Path) -> None:
    smart_root = tmp_path / "image"
    smart_root.mkdir()
    cached_image = tmp_path / "cached.png"
    uncached_image = tmp_path / "uncached.png"
    unknown_image = tmp_path / "unknown.png"
    (smart_root / "cache.json").write_text(
        json.dumps(
            {
                "version": CACHE_VERSION,
                "entries": {
                    os.path.normpath(str(cached_image)): {"hash": "hash-one"},
                    os.path.normpath(str(uncached_image)): {"hash": "hash-nine"},
                },
                "hash_index": {},
            }
        ),
        encoding="utf-8",
    )
    _era(tmp_path / "embedding", "a" * 16, ["hash-one"])

    config = TrainConfig(
        concept_folder=str(tmp_path / "concept"),
        use_cache=True,
        cache_dir=str(smart_root),
        embedding_cache_dir=str(tmp_path / "embedding"),
    )

    predicate = _cached_negative_predicate(config)

    assert predicate is not None
    assert predicate(str(cached_image))
    assert not predicate(str(uncached_image))
    assert not predicate(str(unknown_image))


def test_cached_negative_predicate_is_none_without_usable_caches(
    tmp_path: Path,
) -> None:
    empty = TrainConfig(
        concept_folder=str(tmp_path),
        use_cache=True,
        cache_dir=str(tmp_path / "image"),
        embedding_cache_dir=str(tmp_path / "embedding"),
    )
    assert _cached_negative_predicate(empty) is None

    _era(tmp_path / "embedding", "a" * 16, ["hash-one"])
    cache_disabled = TrainConfig(
        concept_folder=str(tmp_path),
        use_cache=False,
        cache_dir=str(tmp_path / "image"),
        embedding_cache_dir=str(tmp_path / "embedding"),
    )
    assert _cached_negative_predicate(cache_disabled) is None


def _image(path: Path, colour: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), colour).save(path)
    return path


def test_head_only_prepare_data_seeds_negatives_through_the_predicate(
    tmp_path: Path, monkeypatch
) -> None:
    import bittrainer.generic.tasks.binary_head_only_task as module

    def marker(_path: str) -> bool:
        return False

    monkeypatch.setattr(module, "_cached_negative_predicate", lambda _config: marker)
    seen: list = []
    original = module._sample_negative_pool

    def spy(paths, **kwargs):
        seen.append(kwargs.get("is_cached"))
        return original(paths, **kwargs)

    monkeypatch.setattr(module, "_sample_negative_pool", spy)

    concept_folder = tmp_path / "concept"
    config = TrainConfig(
        concept_folder=str(concept_folder),
        use_cache=False,
        train_positive_paths=[str(_image(concept_folder / "train-pos.png", (255, 0, 0)))],
        val_positive_paths=[str(_image(concept_folder / "val-pos.png", (0, 255, 0)))],
        train_negative_paths=[
            str(_image(tmp_path / "negatives" / f"train-neg-{index}.png", (0, 0, index)))
            for index in range(6)
        ],
        val_negative_paths=[
            str(_image(tmp_path / "negatives" / f"val-neg-{index}.png", (index, 0, 255)))
            for index in range(6)
        ],
        train_hard_negative_paths=[],
        val_hard_negative_paths=[],
    )
    task = BinaryHeadOnlyTask(config)

    task.prepare_data(task.make_context(None, None, None, None))

    assert seen and all(is_cached is marker for is_cached in seen)
