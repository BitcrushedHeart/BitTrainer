"""Resolution probe (Bitcrush ISSUE-0550): paired k-fold linear probes over
per-resolution embedding eras, without ever pruning the production cache.

CPU, ``atto``, synthetic PNGs, tiny probe resolutions so the whole file runs in
seconds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from bittrainer.model import create_model
from bittrainer.resolution_probe import ResolutionProbeError, run_resolution_probe


def _write_images(folder, n, seed=0, size=(80, 64)):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        arr = (
            np.random.default_rng(seed + i)
            .integers(0, 256, (size[1], size[0], 3))
            .astype(np.uint8)
        )
        Image.fromarray(arr).save(folder / f"img{seed + i}.png")


def _make_group(tmp_path):
    root = tmp_path / "group"
    _write_images(root / "a" / "train", 10, seed=0)
    _write_images(root / "b" / "train", 10, seed=100)
    _write_images(root / "a" / "val", 3, seed=200)
    _write_images(root / "b" / "val", 3, seed=300)
    return root


def _checkpoint(tmp_path):
    donor = create_model(model_size="atto", pretrained=False, num_classes=0)
    path = tmp_path / "backbone.safetensors"
    save_file(donor.state_dict(), str(path))
    return str(path)


def _request(tmp_path, **overrides):
    request = {
        "mode": "group",
        "folder": str(_make_group(tmp_path)),
        "class_names": ["a", "b"],
        "backbone_init": {"source": "local_active", "checkpoint_path": _checkpoint(tmp_path)},
        "model_size": "atto",
        "resolutions": [64, 96],
        "baseline_resolution": 64,
        "sample_size": 30,
        "folds": 3,
        "seed": 7,
        "device": "cpu",
    }
    request.update(overrides)
    return request


def test_probe_end_to_end_structure_and_pairing(tmp_path):
    result = run_resolution_probe(_request(tmp_path))

    assert result["metric_name"] == "macro_f1"
    assert result["baseline_resolution"] == 64
    resolutions = [row["resolution"] for row in result["results"]]
    assert resolutions == [64, 96]
    for row in result["results"]:
        assert 0.0 <= row["metric_mean"] <= 1.0
        assert len(row["fold_scores"]) == 3  # same folds at every resolution
        assert "delta_mean" in row and "delta_std" in row
        assert row["compute_multiplier"] > 0
    baseline_row = result["results"][0]
    assert baseline_row["delta_mean"] == 0.0  # paired delta vs itself
    assert result["verdict"]["kind"] in ("winner", "no_clear_winner")
    assert result["native_resolution"]["median_px"] > 0
    # 80x64 sources are below both candidates -> the audit says so.
    assert result["native_resolution"]["pct_below"][96] == 100.0


def test_probe_eras_coexist_and_second_run_reuses_cache(tmp_path):
    request = _request(tmp_path)
    first = run_resolution_probe(request)
    assert any(row["cache_built"] > 0 for row in first["results"])

    cache_root = tmp_path / "group" / ".embedding_cache"
    eras = [p.name for p in cache_root.iterdir() if p.is_dir()]
    # One era per non-default resolution (64 and 96 both differ from 512 ->
    # both suffixed), NOT mutually pruned.
    assert len(eras) == 2

    second = run_resolution_probe(request)
    assert all(row["cache_built"] == 0 for row in second["results"])
    # Determinism: identical sample, folds and head seeds -> identical scores.
    assert [row["fold_scores"] for row in second["results"]] == [
        row["fold_scores"] for row in first["results"]
    ]


def test_probe_never_prunes_a_production_era(tmp_path):
    request = _request(tmp_path)
    cache_root = tmp_path / "group" / ".embedding_cache"
    # A fake production era (bare backbone-hash dir with vectors).
    production = cache_root / "0123456789abcdef"
    production.mkdir(parents=True)
    (production / "meta.json").write_text("{}")
    np.save(production / "aabb.npy", np.zeros(4, dtype=np.float32))

    run_resolution_probe(request)
    assert production.is_dir()
    assert (production / "aabb.npy").is_file()


def test_probe_refuses_random_init(tmp_path):
    request = _request(tmp_path, backbone_init={"source": "random_init", "checkpoint_path": None})
    with pytest.raises(ResolutionProbeError, match="random-init|random_init|from-scratch"):
        run_resolution_probe(request)


def test_probe_refuses_single_class(tmp_path):
    request = _request(tmp_path, class_names=["a"])
    with pytest.raises(ResolutionProbeError, match="2 classes"):
        run_resolution_probe(request)


def test_binary_mode(tmp_path):
    root = tmp_path / "concept"
    _write_images(root / "train", 10, seed=0)
    _write_images(root / "val", 3, seed=50)
    _write_images(root / "negative" / "train", 10, seed=100)
    _write_images(root / "negative" / "val", 3, seed=150)
    request = _request(
        tmp_path, mode="binary", folder=str(root), class_names=None, sample_size=24
    )
    result = run_resolution_probe(request)
    assert result["metric_name"] == "balanced_accuracy"
    assert result["per_class_counts"]["positive"] > 0
    assert result["per_class_counts"]["negative"] > 0


def test_ensure_prune_false_keeps_sibling_eras(tmp_path):
    from bittrainer.embedding_cache import EmbeddingCache
    from bittrainer.model import backbone_feature_hash

    samples = []
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(3):
        arr = np.random.default_rng(i).integers(0, 256, (80, 80, 3)).astype(np.uint8)
        p = d / f"im{i}.png"
        Image.fromarray(arr).save(p)
        samples.append({"path": str(p), "bucket": (64, 64), "skin_normalise": False,
                        "face_bbox": None})
    model = create_model(model_size="nano", pretrained=False, num_classes=3).eval()
    h = backbone_feature_hash(model)
    root = tmp_path / "embed"
    kwargs = {"device": torch.device("cpu"), "dtype": torch.float32, "prune": False}
    EmbeddingCache(root, h, 640).ensure(samples, model, None, **kwargs)
    EmbeddingCache(root, h, 640, preproc_sig="val_imagenet@256").ensure(
        samples, model, None, **kwargs
    )
    EmbeddingCache(root, h, 640, preproc_sig="val_imagenet@768").ensure(
        samples, model, None, **kwargs
    )
    assert len([p for p in root.iterdir() if p.is_dir()]) == 3
