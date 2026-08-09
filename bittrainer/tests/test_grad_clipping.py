"""Gradient clipping at the real optimizer-step boundary (round B8).

``clip_gradients`` (shared helper in ``bittrainer.generic.optimizer``) is
wired, config-gated (``clip_grad_norm``, default 1.0, ``0`` disables), into
BOTH training loops: ``group_trainer._train_one_epoch`` clips ONLY at a
gradient-accumulation boundary (immediately before ``optimizer.step()``) and
``BackboneTask.train_epoch`` clips before each step. The loops call the helper
through their own module namespace (``gt.clip_gradients`` /
``backbone_task.clip_gradients``) — the same monkeypatch-seam convention as
``bb._evaluate``.
"""

from __future__ import annotations

import asyncio

import pytest
import torch
import torch.nn as nn

import bittrainer.backbone_trainer as bb
import bittrainer.group_trainer as gt
from bittrainer.generic.optimizer import make_optimizer
from bittrainer.model import create_model
from bittrainer.tests.test_backbone_generic import _request


# --------------------------------------------------------------------------- #
# The shared helper                                                           #
# --------------------------------------------------------------------------- #


def test_clip_gradients_bounds_the_total_norm():
    from bittrainer.generic.optimizer import clip_gradients

    layer = nn.Linear(4, 4)
    for p in layer.parameters():
        p.grad = torch.full_like(p, 100.0)
    clip_gradients(layer, 1.0)
    total = torch.sqrt(sum((p.grad**2).sum() for p in layer.parameters()))
    assert float(total) == pytest.approx(1.0, rel=1e-4)


def test_clip_gradients_skips_gradless_params():
    from bittrainer.generic.optimizer import clip_gradients

    layer = nn.Linear(4, 4)
    layer.weight.grad = torch.full_like(layer.weight, 100.0)
    layer.bias.grad = None  # frozen / untouched param must not break clipping
    clip_gradients(layer, 1.0)
    assert float(layer.weight.grad.norm()) == pytest.approx(1.0, rel=1e-4)


# --------------------------------------------------------------------------- #
# Group loop: boundary-only under gradient accumulation                       #
# --------------------------------------------------------------------------- #


def _group_config(**kw) -> gt.GroupTrainConfig:
    base = dict(
        group_folder=".",
        num_classes=3,
        class_names=["a", "b", "c"],
        device="cpu",
        dtype="float32",
        channels_last=False,
        use_compile=False,
    )
    base.update(kw)
    return gt.GroupTrainConfig(**base)


def _batches(n: int, batch: int = 2):
    torch.manual_seed(0)
    return [
        (
            torch.randint(0, 255, (batch, 3, 64, 64), dtype=torch.uint8),
            torch.randint(0, 3, (batch,)),
        )
        for _ in range(n)
    ]


def _run_group_epoch(monkeypatch, config):
    events: list[str] = []

    def _spy_clip(module_or_params, max_norm):
        events.append(f"clip:{max_norm}")

    monkeypatch.setattr(gt, "clip_gradients", _spy_clip)

    model = create_model(model_size="atto", pretrained=False, num_classes=3)
    optimizer = make_optimizer(model)
    real_step = optimizer.step

    def _step(*a, **k):
        events.append("step")
        return real_step(*a, **k)

    optimizer.step = _step
    gt._train_one_epoch(
        model,
        _batches(4),
        optimizer,
        config,
        torch.device("cpu"),
        torch.float32,
    )
    return events


def test_group_clips_only_at_accumulation_boundaries(monkeypatch):
    events = _run_group_epoch(
        monkeypatch, _group_config(grad_accum_steps=2, clip_grad_norm=1.0)
    )
    # 4 batches, accum 2 -> exactly 2 optimizer steps, each preceded by ONE clip
    # (clipping inside the accumulation window would clip half-summed grads).
    assert events == ["clip:1.0", "step", "clip:1.0", "step"]


def test_group_clip_disabled_at_zero(monkeypatch):
    events = _run_group_epoch(
        monkeypatch, _group_config(grad_accum_steps=2, clip_grad_norm=0.0)
    )
    assert events == ["step", "step"]


def test_group_config_default_is_one():
    assert _group_config().clip_grad_norm == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Backbone loop                                                               #
# --------------------------------------------------------------------------- #


def _run_backbone(tmp_path, monkeypatch, **config_overrides):
    import bittrainer.generic.tasks.backbone_task as bt

    calls: list[float] = []

    def _spy_clip(module_or_params, max_norm):
        calls.append(float(max_norm))

    monkeypatch.setattr(bt, "clip_gradients", _spy_clip)
    request = _request(tmp_path, epochs=1, max_steps=4, n=8)
    request["training_config"].update(config_overrides)
    asyncio.run(bb.run_backbone_training(request))
    return calls


def test_backbone_clips_before_each_step_by_default(tmp_path, monkeypatch):
    calls = _run_backbone(tmp_path, monkeypatch)
    assert calls, "backbone loop never called clip_gradients"
    assert set(calls) == {1.0}


def test_backbone_clip_disabled_at_zero(tmp_path, monkeypatch):
    assert _run_backbone(tmp_path, monkeypatch, clip_grad_norm=0) == []
