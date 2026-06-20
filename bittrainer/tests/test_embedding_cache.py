"""Tests for EmbeddingCache: build/reuse, the mandatory verify self-test, and
per-backbone-era namespace isolation. All CPU, no SmartCache (on-the-fly build).
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


def test_namespaces_isolated_by_backbone(tmp_path):
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

    eras = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert eras == sorted([h1, h2])
