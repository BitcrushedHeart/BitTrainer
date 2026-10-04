"""SigLIP models through the embedding cache and the cached head probe
(Bitcrush ISSUE-1106).

Head-only SigLIP training is the existing head-only path: the frozen trunk's
pooled vectors (MAP pool output, ``num_features`` wide) go into the
``EmbeddingCache`` keyed by the backbone hash, the mandatory ``verify`` replays a
live forward against them, and the probe trains ``head.fc`` on the vectors.

CPU-only, tiny random trunk through the ``_build_trunk`` seam; no download.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from bittrainer.embedding_cache import EmbeddingCache, _content_hash
from bittrainer.gpu_augment import apply_val_transform
from bittrainer.model import (
    backbone_feature_hash,
    create_model,
    head_tail_logits,
    pooled_features,
)
from bittrainer.tests.siglip_fixtures import TINY_DIM, install_tiny_trunk

_DEV = torch.device("cpu")
_DT = torch.float32


@pytest.fixture
def model(monkeypatch):
    install_tiny_trunk(monkeypatch)
    return create_model(
        model_size="siglip2_base", pretrained=False, num_classes=3
    ).eval()


def _samples(tmp_path, n=5):
    d = tmp_path / "imgs"
    d.mkdir()
    samples = []
    for i in range(n):
        p = d / f"im{i}.png"
        arr = np.random.default_rng(i).integers(0, 256, (80, 64, 3)).astype(np.uint8)
        Image.fromarray(arr).save(p)
        samples.append(
            {
                "path": str(p),
                "bucket": (64, 80),
                "label": i % 3,
                "skin_normalise": False,
                "face_bbox": None,
            }
        )
    return samples


def test_cache_builds_reuses_and_verifies_siglip_vectors(model, tmp_path):
    samples = _samples(tmp_path)
    cache = EmbeddingCache(
        tmp_path / "embed", backbone_feature_hash(model), int(model.num_features)
    )
    stats = cache.ensure(samples, model, None, device=_DEV, dtype=_DT, batch_size=2)
    assert stats["built"] == len(samples)
    assert cache.verify(samples, model, None, device=_DEV, dtype=_DT) == len(samples)
    vec = np.load(cache._vec_path(_content_hash(samples[0]["path"], None)))
    assert vec.shape == (TINY_DIM,)
    again = cache.ensure(samples, model, None, device=_DEV, dtype=_DT)
    assert again["reused"] == len(samples) and again["built"] == 0


def test_cached_vector_is_the_pooled_feature_the_head_reads(model, tmp_path):
    samples = _samples(tmp_path, n=1)
    cache = EmbeddingCache(tmp_path / "embed", backbone_feature_hash(model), TINY_DIM)
    cache.ensure(samples, model, None, device=_DEV, dtype=_DT)
    vec = torch.from_numpy(
        np.load(cache._vec_path(_content_hash(samples[0]["path"], None)))
    )

    from bittrainer.cache_builders import build_image_tensor

    batch = torch.from_numpy(np.ascontiguousarray(build_image_tensor(samples[0])))[None]
    with torch.no_grad():
        x = apply_val_transform(batch, dtype=_DT)
        torch.testing.assert_close(
            pooled_features(model, x)[0], vec, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            head_tail_logits(model, vec[None]), model(x), atol=1e-5, rtol=1e-5
        )


def test_a_moved_trunk_is_a_new_cache_era(model):
    before = backbone_feature_hash(model)
    with torch.no_grad():
        model.trunk.attn_pool.latent.add_(0.1)
    assert backbone_feature_hash(model) != before
