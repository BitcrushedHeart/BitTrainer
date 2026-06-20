"""Tests for runtime.py — compile gating, eager fallback, layout helpers."""

import torch
import torch.nn as nn

import bittrainer.runtime as runtime
from bittrainer.gpu_augment import apply_train_augment, apply_val_transform
from bittrainer.runtime import maybe_compile, prewarm_compile, unwrap_compiled


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Conv2d(3, 4, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))


class TestMaybeCompile:
    def test_disabled_returns_eager(self):
        model = _tiny_model()
        fwd, compiled = maybe_compile(model, enabled=False)
        assert fwd is model
        assert compiled is False

    def test_missing_triton_falls_back_with_status(self, monkeypatch):
        monkeypatch.setattr(runtime, "compile_supported", lambda: False)
        messages = []
        model = _tiny_model()
        fwd, compiled = maybe_compile(model, enabled=True, cb=messages.append)
        assert fwd is model
        assert compiled is False
        assert any("eager" in m["status_text"] for m in messages)

    def test_compiled_wrapper_shares_parameters(self, monkeypatch):
        monkeypatch.setattr(runtime, "compile_supported", lambda: True)
        model = _tiny_model()
        fwd, compiled = maybe_compile(model, enabled=True)
        assert compiled is True
        # Optimizer/EMA iterate parameters() — ordering and identity must match
        # the eager module exactly.
        for eager_p, wrapped_p in zip(model.parameters(), fwd.parameters()):
            assert eager_p is wrapped_p
        assert unwrap_compiled(fwd) is model

    def test_unwrap_passthrough_for_eager(self):
        model = _tiny_model()
        assert unwrap_compiled(model) is model


class TestPrewarm:
    def test_empty_buckets_is_noop_success(self):
        assert prewarm_compile(
            _tiny_model(), {}, 4, torch.device("cpu"), torch.float32,
        ) is True

    def test_eager_cpu_prewarm_succeeds_and_emits(self):
        messages = []
        model = _tiny_model()
        ok = prewarm_compile(
            model, {(32, 32): 10, (64, 16): 2}, 4, torch.device("cpu"), torch.float32,
            cb=messages.append,
        )
        assert ok is True
        compiling = [m for m in messages if m["stage"] == "compiling"]
        assert len(compiling) == 4  # 2 shapes x 2 batch sizes
        assert all(p.grad is None for p in model.parameters())

    def test_failure_returns_false(self):
        def _broken_inputs(b, bucket):
            raise RuntimeError("inductor exploded")

        messages = []
        ok = prewarm_compile(
            _tiny_model(), {(32, 32): 10}, 4, torch.device("cpu"), torch.float32,
            make_inputs=_broken_inputs, cb=messages.append,
        )
        assert ok is False
        assert any("eager" in m["status_text"] for m in messages)


class TestAugmentMemoryFormat:
    def test_train_augment_channels_last(self):
        batch = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
        out = apply_train_augment(
            batch, dtype=torch.float32, memory_format=torch.channels_last,
        )
        assert out.is_contiguous(memory_format=torch.channels_last)

    def test_val_transform_channels_last(self):
        batch = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
        out = apply_val_transform(batch, memory_format=torch.channels_last)
        assert out.is_contiguous(memory_format=torch.channels_last)

    def test_default_layout_unchanged(self):
        batch = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
        out = apply_val_transform(batch)
        assert out.is_contiguous()
