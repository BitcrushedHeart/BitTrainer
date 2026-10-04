"""Head-only candidates are scored on the CACHED val features (Bitcrush ISSUE-1110).

Job 1593's ``pre_loop`` pushed all 2,267 val images through the full SigLIP
so400m model after the probe (>20 min under GPU contention, no progress frames)
and finalisation's calibration ran a second full pass after reloading the
1.6 GB checkpoint — although the frozen trunk's pooled vector for every val
image was already in the embedding cache. ``model(x)`` is exactly
``head_tail_logits(model, pooled_features(model, x))``, so scoring the head on
the cached vectors IS the full pass, minus the trunk.

These tests prove that equivalence on tiny real models (the same batches, the
same RNG draws, the same metrics), that a checkpoint whose trunk differs from
the cache's still gets the full pass, and that a head-only run no longer runs
the trunk over the val set.

CPU-only, float32, tiny random trunks; no download.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

import bittrainer.finalize as finalize_mod
import bittrainer.group_trainer as gt
from bittrainer.cached_eval import (
    CachedValScorer,
    collect_cached_logits,
    evaluate_cached,
    val_inputs_are_cache_inputs,
)
from bittrainer.collate import _collate_bucket_batch, _collate_multilabel_batch
from bittrainer.embedding_cache import EmbeddingCache
from bittrainer.generic.evaluation import _collect_val_logits, _evaluate
from bittrainer.group_dataset import GroupDataset, build_group_bucket_sampler
from bittrainer.model import backbone_feature_hash, create_model
from bittrainer.ordinal_threshold import install_ordinal_threshold_head
from bittrainer.tests.siglip_fixtures import install_tiny_trunk

_DEV = torch.device("cpu")
_DT = torch.float32
_CLASSES = ["__none__", "low", "mid", "high"]
_SIZES = [(96, 96), (128, 80), (80, 128)]


def _build_group(root, per_class=(5, 4)):
    for ci, c in enumerate(_CLASSES):
        for split, n in (("train", per_class[0]), ("val", per_class[1])):
            d = root / c / split
            d.mkdir(parents=True)
            for j in range(n):
                w, h = _SIZES[(ci + j) % len(_SIZES)]
                rng = np.random.default_rng(ci * 100 + j + (7 if split == "val" else 0))
                arr = rng.integers(0, 90, (h, w, 3)).astype(np.uint8)
                arr[..., ci % 3] = np.clip(arr[..., ci % 3] + 120, 0, 255)
                Image.fromarray(arr).save(d / f"i{j}.png")


def _siglip(monkeypatch, *, threshold=False, mlp=False):
    install_tiny_trunk(monkeypatch)
    torch.manual_seed(0)
    model = create_model(model_size="siglip2_base", pretrained=False, num_classes=4,
                         head_hidden_size=16 if mlp else None)
    if threshold:
        install_ordinal_threshold_head(model, 4, 0)
    return model.eval()


def _val(root, *, multi_label=False):
    return GroupDataset(root, _CLASSES, split="val", train_resolution=128,
                        multi_label=multi_label, group_name="g")


def _features(model, val_ds, tmp_path):
    cache = EmbeddingCache(tmp_path / "embed", backbone_feature_hash(model),
                           int(model.num_features), packed=True)
    cache.ensure(val_ds.samples, model, None, device=_DEV, dtype=_DT, batch_size=3)
    matrix, present = cache.vectors_for(val_ds.samples, None)
    assert present.all()
    return torch.from_numpy(matrix)


def _loader(val_ds, *, multi_label=False, bs=3):
    return DataLoader(
        val_ds, batch_sampler=build_group_bucket_sampler(val_ds, batch_size=bs),
        collate_fn=_collate_multilabel_batch if multi_label else _collate_bucket_batch,
        num_workers=0,
    )


class _Cfg:
    channels_last = False


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True  # NaN (e.g. an undefined kappa) on both sides
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def _assert_same_metrics(full: dict, cached: dict) -> None:
    assert set(full) == set(cached)
    for key, value in full.items():
        if key == "val_loss":
            assert cached[key] == pytest.approx(value, rel=1e-5, abs=1e-6)
        elif isinstance(value, np.ndarray):
            np.testing.assert_allclose(cached[key], value, rtol=1e-5, atol=1e-6)
        else:
            assert _same(cached[key], value), key


@pytest.mark.parametrize("variant", ["linear", "threshold", "mlp"])
def test_cached_eval_equals_the_full_model_eval(monkeypatch, tmp_path, variant):
    root = tmp_path / "grp"
    _build_group(root)
    model = _siglip(monkeypatch, threshold=variant == "threshold", mlp=variant == "mlp")
    val_ds = _val(root)
    assert len({s["bucket"] for s in val_ds.samples}) > 1, "fixture must mix buckets"
    feats = _features(model, val_ds, tmp_path)
    loader = _loader(val_ds)

    random.seed(5)
    full = _evaluate(model, loader, 4, _DEV, _DT, ordinal=True, none_index=0)
    after_full = random.random()
    random.seed(5)
    cached = evaluate_cached(model, feats, loader, 4, _DEV, _DT, ordinal=True, none_index=0)
    after_cached = random.random()
    _assert_same_metrics(full, cached)
    assert after_full == after_cached, "the cached pass must draw the same RNG"

    random.seed(9)
    logits_f, labels_f = _collect_val_logits(model, loader, _Cfg(), _DEV, _DT)
    random.seed(9)
    logits_c, labels_c = collect_cached_logits(model, feats, loader, _DEV, _DT)
    torch.testing.assert_close(logits_c, logits_f, rtol=1e-5, atol=1e-5)
    assert torch.equal(labels_c, labels_f)


def test_cached_eval_equals_full_eval_multilabel(monkeypatch, tmp_path):
    root = tmp_path / "grp"
    _build_group(root)
    model = _siglip(monkeypatch)
    val_ds = _val(root, multi_label=True)
    feats = _features(model, val_ds, tmp_path)
    loader = _loader(val_ds, multi_label=True)
    random.seed(1)
    full = _evaluate(model, loader, 4, _DEV, _DT, multi_label=True)
    random.seed(1)
    cached = evaluate_cached(model, feats, loader, 4, _DEV, _DT, multi_label=True)
    _assert_same_metrics(full, cached)


def test_cached_eval_equals_full_eval_convnext(tmp_path):
    root = tmp_path / "grp"
    _build_group(root, per_class=(2, 2))
    torch.manual_seed(0)
    model = create_model(model_size="atto", pretrained=False, num_classes=4).eval()
    val_ds = _val(root)
    feats = _features(model, val_ds, tmp_path)
    loader = _loader(val_ds)
    random.seed(2)
    full = _evaluate(model, loader, 4, _DEV, _DT, ordinal=True, none_index=0)
    random.seed(2)
    cached = evaluate_cached(model, feats, loader, 4, _DEV, _DT, ordinal=True, none_index=0)
    _assert_same_metrics(full, cached)


def test_scorer_serves_a_same_trunk_checkpoint_and_refuses_another(monkeypatch, tmp_path):
    root = tmp_path / "grp"
    _build_group(root)
    model = _siglip(monkeypatch)
    val_ds = _val(root)
    feats = _features(model, val_ds, tmp_path)
    loader = _loader(val_ds)
    scorer = CachedValScorer(model, feats, backbone_feature_hash(model))

    # An incumbent with the same frozen trunk and its own head.
    incumbent = _siglip(monkeypatch)
    incumbent.trunk.load_state_dict(model.trunk.state_dict())
    with torch.no_grad():
        incumbent.head.fc.weight.normal_()
    same = scorer.model_for_state(incumbent.state_dict())
    assert same is not None
    random.seed(4)
    want, labels_w = _collect_val_logits(incumbent, loader, _Cfg(), _DEV, _DT)
    random.seed(4)
    got, labels_g = scorer.collect_logits(same, loader, _DEV, _DT)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    assert torch.equal(labels_g, labels_w)

    # A moved trunk is a different feature space: the scorer must refuse it.
    other = _siglip(monkeypatch)
    with torch.no_grad():
        other.trunk.attn_pool.latent.add_(0.5)
    assert scorer.model_for_state(other.state_dict()) is None
    # A head of another shape (MLP vs linear) cannot be served either.
    mlp = _siglip(monkeypatch, mlp=True)
    mlp.trunk.load_state_dict(model.trunk.state_dict())
    assert scorer.model_for_state(mlp.state_dict()) is None


def test_val_inputs_check():
    class _DS:
        transform = None
        _forensic = None
        skin_tone_views = None
        skin_tone_view_prob = 0.0
        skin_tone_force_view = False

    ds = _DS()
    assert val_inputs_are_cache_inputs(ds)
    ds.skin_tone_views = object()
    assert val_inputs_are_cache_inputs(ds), "a bank at probability 0 changes nothing"
    ds.skin_tone_force_view = True
    assert not val_inputs_are_cache_inputs(ds)
    ds.skin_tone_force_view = False
    ds.transform = lambda img: img
    assert not val_inputs_are_cache_inputs(ds)


# --------------------------------------------------------------------------
# Finalisation and the head-only run
# --------------------------------------------------------------------------


def _head_only_cfg(root, ckpt_dir, embed_dir, **kw):
    return gt.GroupTrainConfig(
        group_folder=str(root), num_classes=4, class_names=_CLASSES, ordinal=True,
        backbone_variant="siglip2_base", device="cpu", dtype="float32", use_cache=False,
        checkpoint_dir=str(ckpt_dir), embedding_cache_dir=str(embed_dir),
        probe_head="linear", head_max_epochs=4, head_patience=2, train_resolution=128,
        **kw,
    )


def _count_full_passes(monkeypatch):
    calls: list[str] = []
    real_eval = gt._evaluate
    real_collect = finalize_mod._collect_val_logits

    def _eval(*a, **k):
        calls.append("evaluate")
        return real_eval(*a, **k)

    def _collect(*a, **k):
        calls.append("collect")
        return real_collect(*a, **k)

    monkeypatch.setattr(gt, "_evaluate", _eval)
    monkeypatch.setattr(finalize_mod, "_evaluate", _eval)
    monkeypatch.setattr(finalize_mod, "_collect_val_logits", _collect)
    return calls


def test_head_only_run_scores_on_cached_features_with_same_result(monkeypatch, tmp_path):
    """No full-trunk val pass in a head-only run, and the outcome is the one
    the full passes produce (the same RNG seed drives both runs)."""
    from bittrainer.generic.generic_trainer import GenericTrainer
    from bittrainer.generic.tasks.head_only_task import HeadOnlyTask

    install_tiny_trunk(monkeypatch)
    root = tmp_path / "grp"
    _build_group(root)
    calls = _count_full_passes(monkeypatch)

    def _run(name, cached):
        ckpt_dir = tmp_path / f"ck_{name}"
        ckpt_dir.mkdir()
        random.seed(13)
        np.random.seed(13)
        torch.manual_seed(13)
        task = HeadOnlyTask(_head_only_cfg(root, ckpt_dir, tmp_path / f"emb_{name}"),
                            cached_val_eval=cached)
        return GenericTrainer().run(task)

    calls.clear()
    full = _run("full", False)
    assert calls.count("evaluate") == 1 and calls.count("collect") == 1
    calls.clear()
    fast = _run("fast", True)
    assert calls == [], f"full-trunk val passes still ran: {calls}"

    for key in ("best_val_macro_f1", "final_val_macro_f1", "qwk", "best_val_qwk",
                "calibration_temperature", "none_logit_bias", "final_val_loss"):
        assert fast.get(key) == pytest.approx(full.get(key), rel=1e-5, abs=1e-6), key
    for key in ("promotion_reason", "confusion_matrix", "ordinal_decode",
                "selected_softness_value", "selected_oversample_none"):
        assert fast.get(key) == full.get(key), key
    assert fast["per_class_f1"] == pytest.approx(full["per_class_f1"], rel=1e-5, abs=1e-6)


def test_incumbent_with_the_same_trunk_is_scored_on_cached_features(monkeypatch, tmp_path):
    """A head-only group's next run: the incumbent shares the frozen trunk, so
    the head-to-head comparison needs no checkpoint load and no trunk pass."""
    from bittrainer.generic.generic_trainer import GenericTrainer
    from bittrainer.generic.tasks.head_only_task import HeadOnlyTask

    install_tiny_trunk(monkeypatch)
    root = tmp_path / "grp"
    _build_group(root)
    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    cfg = _head_only_cfg(root, ckpt_dir, tmp_path / "emb")
    first = GenericTrainer().run(HeadOnlyTask(cfg))
    assert first["checkpoint_path"].endswith("best.pt")

    calls = _count_full_passes(monkeypatch)
    loads: list[str] = []
    real_load = finalize_mod.load_checkpoint
    monkeypatch.setattr(finalize_mod, "load_checkpoint",
                        lambda *a, **k: loads.append(a[0]) or real_load(*a, **k))
    second = GenericTrainer().run(HeadOnlyTask(_head_only_cfg(root, ckpt_dir,
                                                              tmp_path / "emb")))
    assert calls == [] and loads == []
    assert second["promotion_reason"] is not None


def test_incumbent_with_another_trunk_still_gets_a_full_pass_with_progress(monkeypatch,
                                                                          tmp_path):
    from bittrainer.generic.generic_trainer import GenericTrainer
    from bittrainer.generic.tasks.head_only_task import HeadOnlyTask

    install_tiny_trunk(monkeypatch)
    root = tmp_path / "grp"
    _build_group(root)
    ckpt_dir = tmp_path / "ck"
    ckpt_dir.mkdir()
    other = create_model(model_size="siglip2_base", pretrained=False, num_classes=4)
    install_ordinal_threshold_head(other, 4, 0)
    with torch.no_grad():
        other.trunk.attn_pool.latent.add_(0.5)
    torch.save({"state_dict": other.state_dict(), "num_classes": 4,
                "model_size": "siglip2_base", "class_names": _CLASSES},
               ckpt_dir / "best.pt")
    calls = _count_full_passes(monkeypatch)
    frames: list[dict] = []
    cfg = _head_only_cfg(root, ckpt_dir, tmp_path / "emb", from_scratch=True,
                         ordinal_head="threshold")
    GenericTrainer().run(HeadOnlyTask(cfg), progress_callback=frames.append)
    assert calls.count("collect") >= 1, "the other-trunk incumbent needs its own pass"
    texts = [f.get("status_text", "") for f in frames]
    assert any("Scoring the current model" in t for t in texts), texts
