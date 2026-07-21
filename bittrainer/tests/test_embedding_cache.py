"""Tests for EmbeddingCache: build/reuse, the mandatory verify self-test, and
stale-era pruning on backbone change. All CPU, no SmartCache (on-the-fly build).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from bittrainer.embedding_cache import EmbeddingCache, EmbeddingCacheMismatch, _content_hash
from bittrainer.model import backbone_feature_hash, create_model

_DEV = torch.device("cpu")
_DT = torch.float32


def _samples(tmp_path, n=6):
    samples = []
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(n):
        p = d / f"im{i}.png"
        arr = np.random.default_rng(i).integers(0, 256, (80, 80, 3)).astype(np.uint8)
        Image.fromarray(arr).save(p)
        samples.append({"path": str(p), "bucket": (64, 64), "label": i % 3,
                        "skin_normalise": False, "face_bbox": None})
    return samples


def test_build_reuse_and_verify(tmp_path):
    samples = _samples(tmp_path)
    model = create_model(model_size="nano", pretrained=False, num_classes=3).eval()
    cache = EmbeddingCache(tmp_path / "embed", backbone_feature_hash(model), 640)

    stats = cache.ensure(samples, model, None, device=_DEV, dtype=_DT, batch_size=4)
    assert stats["built"] == 6 and stats["reused"] == 0

    # Stored at float32 — lossless capture, not an fp16 truncation step.
    h = _content_hash(samples[0]["path"], None)
    assert np.load(cache._vec_path(h)).dtype == np.float32

    assert cache.verify(samples, model, None, device=_DEV, dtype=_DT) == 6

    stats2 = cache.ensure(samples, model, None, device=_DEV, dtype=_DT)
    assert stats2["built"] == 0 and stats2["reused"] == 6


def test_verify_raises_on_corrupt_vector(tmp_path):
    samples = _samples(tmp_path)
    model = create_model(model_size="nano", pretrained=False, num_classes=3).eval()
    cache = EmbeddingCache(tmp_path / "embed", backbone_feature_hash(model), 640)
    cache.ensure(samples, model, None, device=_DEV, dtype=_DT)

    h = _content_hash(samples[0]["path"], None)
    np.save(cache._vec_path(h), np.zeros(640, dtype=np.float16))

    with pytest.raises(EmbeddingCacheMismatch):
        cache.verify(samples, model, None, device=_DEV, dtype=_DT)


def test_verify_raises_when_empty(tmp_path):
    samples = _samples(tmp_path)
    model = create_model(model_size="nano", pretrained=False, num_classes=3).eval()
    cache = EmbeddingCache(tmp_path / "embed", backbone_feature_hash(model), 640)
    # nothing built yet
    with pytest.raises(EmbeddingCacheMismatch):
        cache.verify(samples, model, None, device=_DEV, dtype=_DT)


def test_stale_era_pruned_on_backbone_change(tmp_path):
    """A new backbone hash invalidates the old era; ensure() prunes the previous
    namespace so the cache never accumulates dead vectors for a hash that will
    never recur. (Cache point includes head.norm, so mutating it re-hashes.)"""
    samples = _samples(tmp_path)
    model = create_model(model_size="nano", pretrained=False, num_classes=3).eval()
    root = tmp_path / "embed"
    h1 = backbone_feature_hash(model)
    EmbeddingCache(root, h1, 640).ensure(samples, model, None, device=_DEV, dtype=_DT)

    with torch.no_grad():
        model.head.norm.weight.add_(0.5)  # changes feature hash
    h2 = backbone_feature_hash(model)
    assert h2 != h1
    EmbeddingCache(root, h2, 640).ensure(samples, model, None, device=_DEV, dtype=_DT)

    # Old era removed, only the active backbone's namespace remains.
    eras = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert eras == [h2]


def test_preproc_sig_namespaces_on_disk(tmp_path):
    """preproc_sig is cache IDENTITY: a non-default sig gets its own era dir
    (resolution changes pooled-vector VALUES at fixed dim, so vectors built
    under different preprocessing must never be silently reused). The default
    sig keeps the bare backbone-hash dir name — existing caches stay valid."""
    model = create_model(model_size="nano", pretrained=False, num_classes=3).eval()
    h = backbone_feature_hash(model)
    default = EmbeddingCache(tmp_path / "embed", h, 640)
    sized = EmbeddingCache(tmp_path / "embed", h, 640, preproc_sig="val_imagenet@256sq")
    assert default.root.name == h
    assert sized.root.name != h
    assert sized.root.name.startswith(h + "-")
    assert default.root != sized.root


def test_prune_reclaims_other_sig_same_hash(tmp_path):
    """Two sigs of the same backbone hash are mutually pruning: establishing
    one era reclaims the other (a preprocessing switch invalidates the old
    vectors just like a weight change does)."""
    samples = _samples(tmp_path)
    model = create_model(model_size="nano", pretrained=False, num_classes=3).eval()
    root = tmp_path / "embed"
    h = backbone_feature_hash(model)
    EmbeddingCache(root, h, 640).ensure(samples, model, None, device=_DEV, dtype=_DT)
    sized = EmbeddingCache(root, h, 640, preproc_sig="val_imagenet@256sq")
    sized.ensure(samples, model, None, device=_DEV, dtype=_DT)
    eras = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert eras == [sized.root.name]


def test_era_regex_matches_suffixed_dirs(tmp_path):
    from bittrainer.embedding_cache import _ERA_DIR_RE

    assert _ERA_DIR_RE.match("0123456789abcdef")
    assert _ERA_DIR_RE.match("0123456789abcdef-89abcdef")
    assert not _ERA_DIR_RE.match("not-an-era")


def test_prune_leaves_unrelated_dirs(tmp_path):
    """Pruning only touches recognisable era namespaces — an unrelated sibling
    directory under the cache root is never deleted."""
    samples = _samples(tmp_path)
    model = create_model(model_size="nano", pretrained=False, num_classes=3).eval()
    root = tmp_path / "embed"
    root.mkdir()
    (root / "not-an-era").mkdir()  # 16-hex regex won't match, no meta.json

    EmbeddingCache(root, backbone_feature_hash(model), 640).ensure(
        samples, model, None, device=_DEV, dtype=_DT
    )
    assert (root / "not-an-era").is_dir()
