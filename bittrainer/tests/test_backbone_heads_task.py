"""Backbone head-only retraining from the embedding cache.

``run_backbone_head_training`` retrains the multi-task concept/group heads
against a FROZEN existing backbone checkpoint, on cached pooled features
(one embedding pass per backbone era, then head epochs in seconds). The
candidate keeps the 0542 convention — BARE backbone keys byte-equal to the
source checkpoint plus fresh ``heads.*`` tensors — so every existing
consumer (``apply_backbone_init``, promotion) reads it unchanged. The same
per-epoch sampling plan (neg:pos cap, oversample, pos_weight, label policy)
applies to head training.

CPU, ``atto``, synthetic PNGs.
"""

from __future__ import annotations

import asyncio

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from bittrainer.backbone_trainer import run_backbone_head_training, run_backbone_training
from bittrainer.tests.test_backbone_generic import _request


def _train_source(tmp_path):
    """Produce a real backbone candidate to retrain heads against."""
    request = _request(tmp_path, epochs=1, max_steps=4, n=8)
    result = asyncio.run(run_backbone_training(request))
    return result["candidate_checkpoint_path"]


def _heads_request(tmp_path, source_path, *, epochs=1, max_steps=8, n=8):
    request = _request(tmp_path, epochs=epochs, max_steps=max_steps, n=n)
    request["run_id"] = "run_heads_test"
    request["candidate_checkpoint_path"] = str(
        tmp_path / "candidates" / "candidate_heads.safetensors"
    )
    request["backbone_init"] = {"source": "local_active", "checkpoint_path": str(source_path)}
    request["training_config"]["embedding_cache_dir"] = str(tmp_path / "embed")
    return request


def _run_heads(request, progress_callback=None):
    return asyncio.run(run_backbone_head_training(request, progress_callback=progress_callback))


def test_head_only_trains_from_cache_and_exports_convention(tmp_path):
    source = _train_source(tmp_path)
    result = _run_heads(_heads_request(tmp_path, source))

    saved = load_file(result["candidate_checkpoint_path"])
    source_state = load_file(source)
    backbone_keys = [k for k in saved if not k.startswith("heads.")]
    head_keys = [k for k in saved if k.startswith("heads.")]
    assert backbone_keys and head_keys
    # Trunk tensors are byte-equal to the source checkpoint — the backbone
    # never trains in head-only mode.
    for key in backbone_keys:
        assert torch.equal(saved[key], source_state[key]), key
    # Heads are freshly trained, not copies of the source's.
    assert any(
        not torch.equal(saved[k], source_state[k]) for k in head_keys if k in source_state
    ) or any(k not in source_state for k in head_keys)

    with safe_open(result["candidate_checkpoint_path"], framework="pt") as f:
        metadata = f.metadata()
    assert metadata["head_only_retrain"] == "1"
    assert metadata["heads_state_present"] == "1"
    assert metadata["source_backbone_checkpoint"] == str(source)

    # The mixed-key candidate loads through apply_backbone_init unchanged.
    from bittrainer.backbone_init import apply_backbone_init
    from bittrainer.model import create_model

    fresh = create_model(model_size="atto", pretrained=False, num_classes=0)
    assert apply_backbone_init(
        fresh, {"source": "local_active", "checkpoint_path": result["candidate_checkpoint_path"]}
    )


def test_backbone_frozen_during_head_training(tmp_path, monkeypatch):
    from bittrainer.generic.tasks.backbone_heads_task import BackboneHeadsTask

    source = _train_source(tmp_path)
    captured = {}
    real = BackboneHeadsTask.create_model

    def _spy(self, ctx, resume_state):
        model = real(self, ctx, resume_state)
        captured["model"] = model
        return model

    monkeypatch.setattr(BackboneHeadsTask, "create_model", _spy)
    _run_heads(_heads_request(tmp_path, source))

    model = captured["model"]
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.heads.parameters())


def test_cache_reused_on_second_run(tmp_path):
    source = _train_source(tmp_path)
    first = _run_heads(_heads_request(tmp_path, source))
    assert first["embedding_cache_stats"]["built"] > 0
    second_request = _heads_request(tmp_path, source)
    second_request["candidate_checkpoint_path"] = str(
        tmp_path / "candidates" / "candidate_heads2.safetensors"
    )
    second = _run_heads(second_request)
    assert second["embedding_cache_stats"]["built"] == 0
    assert second["embedding_cache_stats"]["reused"] > 0


def test_sampling_plan_applies_to_head_training(tmp_path, monkeypatch):
    import bittrainer.backbone_trainer as bb

    source = _train_source(tmp_path)
    calls = []
    real = bb._plan_epoch_samples

    def _spy(samples, vocab, epoch, **kw):
        calls.append((epoch, kw.get("neg_pos_ratio")))
        return real(samples, vocab, epoch, **kw)

    monkeypatch.setattr(bb, "_plan_epoch_samples", _spy)
    request = _heads_request(tmp_path, source)
    request["training_config"]["neg_pos_ratio"] = 2.0
    _run_heads(request)
    assert calls and all(ratio == 2.0 for _epoch, ratio in calls)


def test_result_contract(tmp_path):
    source = _train_source(tmp_path)
    request = _heads_request(tmp_path, source)
    result = _run_heads(request)
    assert result["candidate_checkpoint_path"] == request["candidate_checkpoint_path"]
    assert isinstance(result["validation_score"], float)
    assert isinstance(result["validation_metrics"], dict)
    assert result["mode"] == "backbone_head_only"
    assert isinstance(result["backbone_hash"], str) and len(result["backbone_hash"]) == 16
    assert isinstance(result["epochs_completed"], int)
