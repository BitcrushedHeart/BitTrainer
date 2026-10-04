"""Consolidated vector packs for the embedding cache (Bitcrush ISSUE-1110).

A head-only run used to read every pooled vector from its own ``.npy`` file —
once in ``ensure``'s audit and again in every probe gather (about 64k small
reads per run on NTFS). A *packed* cache keeps the run's vectors in memory and
writes one consolidated ``pack-*.npz`` per era, so the next run (or the next
comparison candidate) loads them with a single read.

The per-photo ``.npy`` files stay the durable source of truth, so older
readers keep working. A pack is only ever written under a fresh name (never
replaced in place — a reader holding it open on Windows cannot block a
writer), is read fully into memory (never memory-mapped), and is ignored when
it fails validation.

CPU-only, tiny random SigLIP trunk; no download.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from bittrainer.embedding_cache import EmbeddingCache, _content_hash
from bittrainer.model import backbone_feature_hash, create_model
from bittrainer.tests.siglip_fixtures import TINY_DIM, install_tiny_trunk

_DEV = torch.device("cpu")
_DT = torch.float32


@pytest.fixture
def model(monkeypatch):
    install_tiny_trunk(monkeypatch)
    return create_model(model_size="siglip2_base", pretrained=False, num_classes=3).eval()


def _samples(tmp_path, n=6, name="imgs"):
    d = tmp_path / name
    d.mkdir()
    samples = []
    for i in range(n):
        p = d / f"im{i}.png"
        arr = np.random.default_rng(i + 100 * len(name)).integers(0, 256, (80, 64, 3))
        Image.fromarray(arr.astype(np.uint8)).save(p)
        samples.append(
            {"path": str(p), "bucket": (64, 80), "label": i % 3,
             "skin_normalise": False, "face_bbox": None}
        )
    return samples


def _cache(root, model, *, packed):
    return EmbeddingCache(root, backbone_feature_hash(model), TINY_DIM, packed=packed)


def _packs(cache):
    return sorted(cache.root.glob("pack-*.npz"))


def test_unpacked_cache_writes_no_pack(model, tmp_path):
    """The default (binary trainer, resolution probe, ...) is unchanged."""
    samples = _samples(tmp_path)
    cache = _cache(tmp_path / "embed", model, packed=False)
    cache.ensure(samples, model, None, device=_DEV, dtype=_DT, batch_size=4)
    assert _packs(cache) == []


def test_packed_ensure_writes_one_pack_and_keeps_per_photo_files(model, tmp_path):
    samples = _samples(tmp_path)
    cache = _cache(tmp_path / "embed", model, packed=True)
    stats = cache.ensure(samples, model, None, device=_DEV, dtype=_DT, batch_size=4)
    assert stats == {"built": 6, "reused": 0, "total": 6}
    assert len(_packs(cache)) == 1

    # Older readers (per-photo files, no pack awareness) see the same vectors.
    old_reader = _cache(tmp_path / "embed", model, packed=False)
    for s in samples:
        np.testing.assert_array_equal(
            old_reader.get_vector(s["path"], None), cache.get_vector(s["path"], None)
        )


def test_second_run_reads_the_pack_not_the_per_photo_files(model, tmp_path, monkeypatch):
    samples = _samples(tmp_path)
    first = _cache(tmp_path / "embed", model, packed=True)
    first.ensure(samples, model, None, device=_DEV, dtype=_DT, batch_size=4)
    expected = {s["path"]: first.get_vector(s["path"], None) for s in samples}

    def _no_file_reads(self, path):
        raise AssertionError(f"per-photo read of {path}")

    monkeypatch.setattr(EmbeddingCache, "_load_vector_path", _no_file_reads)
    second = _cache(tmp_path / "embed", model, packed=True)
    stats = second.ensure(samples, model, None, device=_DEV, dtype=_DT)
    assert stats == {"built": 0, "reused": 6, "total": 6}
    matrix, present = second.vectors_for(samples, None)
    assert present.all()
    for row, s in zip(matrix, samples, strict=True):
        np.testing.assert_array_equal(row, expected[s["path"]])
    # The run's set was already covered: no rewrite.
    assert len(_packs(second)) == 1


def test_vectors_for_aligns_rows_and_marks_missing(model, tmp_path):
    samples = _samples(tmp_path)
    cache = _cache(tmp_path / "embed", model, packed=True)
    cache.ensure(samples[:4], model, None, device=_DEV, dtype=_DT)
    ask = [samples[1], samples[5], samples[1], samples[0]]
    matrix, present = cache.vectors_for(ask, None)
    assert matrix.shape == (4, TINY_DIM) and matrix.dtype == np.float32
    assert present.tolist() == [True, False, True, True]
    np.testing.assert_array_equal(matrix[0], matrix[2])
    np.testing.assert_array_equal(matrix[0], cache.get_vector(samples[1]["path"], None))


def test_new_images_write_a_new_pack_of_the_run_set_and_drop_the_old(model, tmp_path):
    samples = _samples(tmp_path)
    cache = _cache(tmp_path / "embed", model, packed=True)
    cache.ensure(samples[:4], model, None, device=_DEV, dtype=_DT)
    (old_pack,) = _packs(cache)

    later = _cache(tmp_path / "embed", model, packed=True)
    stats = later.ensure(samples[2:], model, None, device=_DEV, dtype=_DT)
    assert stats == {"built": 2, "reused": 2, "total": 4}
    (new_pack,) = _packs(later)
    assert new_pack.name != old_pack.name, "a pack is never replaced in place"
    with np.load(new_pack, allow_pickle=False) as z:
        hashes = {str(h) for h in z["hashes"]}
    assert hashes == {_content_hash(s["path"], None) for s in samples[2:]}


def test_a_corrupt_or_torn_pack_is_ignored(model, tmp_path):
    samples = _samples(tmp_path)
    cache = _cache(tmp_path / "embed", model, packed=True)
    cache.ensure(samples, model, None, device=_DEV, dtype=_DT)
    expected = [cache.get_vector(s["path"], None) for s in samples]
    (pack,) = _packs(cache)
    pack.write_bytes(pack.read_bytes()[: pack.stat().st_size // 2])  # torn write

    fresh = _cache(tmp_path / "embed", model, packed=True)
    stats = fresh.ensure(samples, model, None, device=_DEV, dtype=_DT)
    assert stats["built"] == 0 and stats["reused"] == 6
    matrix, present = fresh.vectors_for(samples, None)
    assert present.all()
    np.testing.assert_array_equal(matrix, np.stack(expected))
    # The unreadable pack is superseded by a good one.
    good = [p for p in _packs(fresh) if p != pack]
    assert len(good) == 1


def test_a_pack_with_a_bad_row_is_rejected(model, tmp_path):
    samples = _samples(tmp_path)
    cache = _cache(tmp_path / "embed", model, packed=True)
    cache.ensure(samples, model, None, device=_DEV, dtype=_DT)
    (pack,) = _packs(cache)
    with np.load(pack, allow_pickle=False) as z:
        meta, hashes, vectors = z["meta"], z["hashes"], z["vectors"].copy()
    vectors[2] = np.nan
    with pack.open("wb") as fh:
        np.savez(fh, meta=meta, hashes=hashes, vectors=vectors)

    fresh = _cache(tmp_path / "embed", model, packed=True)
    matrix, present = fresh.vectors_for(samples, None)
    assert present.all() and np.isfinite(matrix).all()


def test_a_pack_from_another_era_is_rejected(model, tmp_path):
    samples = _samples(tmp_path)
    cache = _cache(tmp_path / "embed", model, packed=True)
    cache.ensure(samples, model, None, device=_DEV, dtype=_DT)
    (pack,) = _packs(cache)
    other = EmbeddingCache(tmp_path / "embed", "0" * 16, TINY_DIM, packed=True)
    other.root.mkdir(parents=True)
    (other.root / pack.name).write_bytes(pack.read_bytes())
    assert other._load_pack() is None


def test_get_vector_serves_the_pack(model, tmp_path, monkeypatch):
    samples = _samples(tmp_path)
    _cache(tmp_path / "embed", model, packed=True).ensure(
        samples, model, None, device=_DEV, dtype=_DT
    )
    monkeypatch.setattr(
        EmbeddingCache, "_load_vector_path",
        lambda self, path: (_ for _ in ()).throw(AssertionError("per-photo read")),
    )
    fresh = _cache(tmp_path / "embed", model, packed=True)
    assert fresh.get_vector(samples[3]["path"], None).shape == (TINY_DIM,)


def test_check_progress_reports_the_audit(model, tmp_path):
    samples = _samples(tmp_path)
    _cache(tmp_path / "embed", model, packed=False).ensure(
        samples, model, None, device=_DEV, dtype=_DT
    )
    seen: list[tuple[int, int]] = []
    fresh = _cache(tmp_path / "embed", model, packed=True)
    fresh.ensure(samples, model, None, device=_DEV, dtype=_DT,
                 check_progress_cb=lambda d, t: seen.append((d, t)))
    assert seen and seen[-1] == (6, 6)
