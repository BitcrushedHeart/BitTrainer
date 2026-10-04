"""Cached features are loaded ONCE per head-only run (Bitcrush ISSUE-1110).

Job 1593 showed "Loading cached features (24784/25343)" again after the
soft-label sweep: the softness sweep and the ``__none__`` oversample sweep each
re-read every vector from its own ``.npy`` file. The run now loads the train
and val features once and hands the same tensors to both sweeps, and the load
itself reads each distinct image once (the replication-expanded train schedule
repeats rows).

Guards: the tensors (and therefore every probe result) are identical to the
per-sample loader's.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

import bittrainer.group_trainer as gt
import bittrainer.head_probe as head_probe
import bittrainer.probes as probes
from bittrainer.embedding_cache import EmbeddingCache
from bittrainer.model import backbone_feature_hash, create_model
from bittrainer.tests.siglip_fixtures import TINY_DIM, install_tiny_trunk

_DEV = torch.device("cpu")
_DT = torch.float32
_CLASSES = ["__none__", "low", "mid", "high"]


@pytest.fixture
def model(monkeypatch):
    install_tiny_trunk(monkeypatch)
    return create_model(model_size="siglip2_base", pretrained=False, num_classes=4).eval()


def _samples(tmp_path, n, name):
    d = tmp_path / name
    d.mkdir()
    out = []
    for i in range(n):
        p = d / f"im{i}.png"
        arr = np.random.default_rng(i + 31 * len(name)).integers(0, 256, (64, 64, 3))
        Image.fromarray(arr.astype(np.uint8)).save(p)
        out.append({"path": str(p), "bucket": (64, 64), "label": i % 4,
                    "skin_normalise": False, "face_bbox": None})
    return out


class _LegacyCache:
    """A cache without the bulk API: the per-sample ``get_vector`` path."""

    def __init__(self, inner: EmbeddingCache) -> None:
        self.inner = inner

    def get_vector(self, path, smart_cache):
        return self.inner.get_vector(path, smart_cache)


def _config(**kw):
    return gt.GroupTrainConfig(
        group_folder="/tmp/g", num_classes=4, class_names=_CLASSES, ordinal=True,
        device="cpu", dtype="float32", head_max_epochs=3, head_patience=2, **kw,
    )


def _built_cache(model, tmp_path, samples, *, packed=True):
    cache = EmbeddingCache(tmp_path / "embed", backbone_feature_hash(model), TINY_DIM,
                           packed=packed)
    cache.ensure(samples, model, None, device=_DEV, dtype=_DT)
    return cache


def test_bulk_load_matches_the_per_sample_loader(model, tmp_path):
    train = _samples(tmp_path, 10, "tr")
    val = _samples(tmp_path, 5, "va")
    cache = _built_cache(model, tmp_path, train + val)
    # A replication-expanded schedule repeats rows; one is missing a vector.
    epoch = train + train[:4] + [{**train[0], "path": str(tmp_path / "missing.png")}]

    fast = head_probe.prepare_head_probe_tensors(cache, None, epoch, val, _config())
    slow = head_probe.prepare_head_probe_tensors(_LegacyCache(cache), None, epoch, val,
                                                 _config())
    for a, b in zip(fast, slow, strict=True):
        assert a.dtype == b.dtype
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert fast[0].shape[0] == len(train) + 4


def test_unpacked_bulk_load_matches_too(model, tmp_path):
    train = _samples(tmp_path, 6, "tr")
    val = _samples(tmp_path, 3, "va")
    cache = _built_cache(model, tmp_path, train + val, packed=False)
    fast = head_probe.prepare_head_probe_tensors(cache, None, train, val, _config())
    slow = head_probe.prepare_head_probe_tensors(_LegacyCache(cache), None, train, val,
                                                 _config())
    for a, b in zip(fast, slow, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_sweeps_reuse_preloaded_tensors_and_results_are_unchanged(model, tmp_path,
                                                                  monkeypatch):
    train = _samples(tmp_path, 16, "tr")
    val = _samples(tmp_path, 8, "va")
    cache = _built_cache(model, tmp_path, train + val)
    none_index = 0

    def _run(tensors):
        torch.manual_seed(7)
        m = create_model(model_size="siglip2_base", pretrained=False, num_classes=4).eval()
        cfg = _config()
        soft = gt._run_auto_softness_probe(
            m, cfg, cache, None, train, val, device=_DEV, none_index=none_index,
            cb=lambda _m: None, stop_event=None, tensors=tensors,
        )
        over = gt._run_auto_oversample_probe(
            m, cfg, cache, None, train, val, device=_DEV, none_index=none_index,
            cb=lambda _m: None, stop_event=None, tensors=tensors,
        )
        return soft, over, cfg, {k: v.clone() for k, v in m.head.state_dict().items()}

    baseline = _run(None)

    loads: list[int] = []
    real = head_probe.prepare_head_probe_tensors

    def _counting(*a, **k):
        loads.append(1)
        return real(*a, **k)

    monkeypatch.setattr(probes, "prepare_head_probe_tensors", _counting)
    tensors = real(cache, None, train, val, _config())
    reused = _run(tensors)
    assert loads == [], "a sweep re-read the features although they were preloaded"

    for key in ("macro_f1", "qwk", "val_loss", "best_epoch", "epochs_completed"):
        assert baseline[0].get(key) == reused[0].get(key)
        assert baseline[1].get(key) == reused[1].get(key)
    assert baseline[2].ordinal_sigma == reused[2].ordinal_sigma
    assert baseline[2].oversample_none == reused[2].oversample_none
    for k in baseline[3]:
        torch.testing.assert_close(baseline[3][k], reused[3][k], rtol=0, atol=0)


def test_no_sweep_path_uses_the_preloaded_tensors(model, tmp_path, monkeypatch):
    train = _samples(tmp_path, 8, "tr")
    val = _samples(tmp_path, 4, "va")
    cache = _built_cache(model, tmp_path, train + val)
    tensors = head_probe.prepare_head_probe_tensors(cache, None, train, val, _config())
    monkeypatch.setattr(
        probes, "train_head_probe",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-gathered")),
    )
    cfg = _config(auto_label_softness=False)
    m = create_model(model_size="siglip2_base", pretrained=False, num_classes=4).eval()
    result = gt._run_auto_softness_probe(
        m, cfg, cache, None, train, val, device=_DEV, none_index=0,
        cb=lambda _m: None, stop_event=None, tensors=tensors,
    )
    assert result["epochs_completed"] >= 1


def test_train_head_probe_equals_the_from_tensors_path(model, tmp_path):
    """``train_head_probe`` (gather + train) is the from-tensors trainer — the
    two copies of the training loop are one."""
    train = _samples(tmp_path, 12, "tr")
    val = _samples(tmp_path, 6, "va")
    cache = _built_cache(model, tmp_path, train + val)
    cfg = _config()

    torch.manual_seed(3)
    a = create_model(model_size="siglip2_base", pretrained=False, num_classes=4).eval()
    torch.manual_seed(11)
    res_a = head_probe.train_head_probe(a, cache, None, train, val, cfg, device=_DEV,
                                        none_index=0)
    torch.manual_seed(3)
    b = create_model(model_size="siglip2_base", pretrained=False, num_classes=4).eval()
    tensors = head_probe.prepare_head_probe_tensors(cache, None, train, val, cfg)
    torch.manual_seed(11)
    res_b = head_probe.train_head_probe_from_tensors(b, *tensors, cfg, device=_DEV,
                                                     none_index=0)
    assert res_a == res_b
