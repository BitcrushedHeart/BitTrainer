"""Tests for runtime.py — compile gating, eager fallback, layout helpers."""

import torch
import torch.nn as nn

import bittrainer.runtime as runtime
from bittrainer.gpu_augment import apply_train_augment, apply_val_transform
from bittrainer.runtime import (
    compile_regions,
    configure_compile_cache,
    maybe_compile,
    prewarm_compile,
    uncompile_regions,
    unwrap_compiled,
)


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Conv2d(3, 4, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))


class _FakeBlock(nn.Module):
    """Stand-in for a timm ConvNeXt block — identical across a stage."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dwconv(x)


class _FakeStage(nn.Module):
    def __init__(self, dim: int, depth: int) -> None:
        super().__init__()
        self.downsample = nn.Identity()
        self.blocks = nn.Sequential(*[_FakeBlock(dim) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.downsample(x))


class _FakeConvNeXt(nn.Module):
    """Mirrors timm's ``model.stages[i].blocks[j]`` layout."""

    def __init__(self, depths: tuple[int, ...] = (2, 2, 6, 2), dim: int = 4) -> None:
        super().__init__()
        self.stages = nn.Sequential(*[_FakeStage(dim, d) for d in depths])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(x)


class _FakeMultiHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _FakeConvNeXt(depths=(1, 1, 3, 1))


class _FakeDualBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.crop_branch = _FakeConvNeXt(depths=(1, 1, 2, 1))
        self.context_branch = _FakeConvNeXt(depths=(1, 1, 2, 1))


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

    def test_regional_compile_returns_same_module(self, monkeypatch):
        monkeypatch.setattr(runtime, "compile_supported", lambda: True)
        model = _FakeConvNeXt()
        fwd, compiled = maybe_compile(model, enabled=True)
        assert compiled is True
        # Regional compilation mutates submodules in place — no OptimizedModule
        # wrapper, so the eager module stays the single source of truth.
        assert fwd is model
        assert unwrap_compiled(fwd) is model
        # Every repeated block was handed to torch.compile.
        blocks = [m for m in model.modules() if isinstance(m, _FakeBlock)]
        assert blocks and all(b._compiled_call_impl is not None for b in blocks)

    def test_unwrap_passthrough_for_eager(self):
        model = _tiny_model()
        assert unwrap_compiled(model) is model


class TestRegionalCompile:
    def test_targets_cover_every_block(self):
        model = _FakeConvNeXt(depths=(2, 2, 6, 2))
        assert len(runtime._compile_targets(model)) == 12

    def test_compile_regions_compiles_each_block(self, monkeypatch):
        monkeypatch.setattr(runtime, "compile_supported", lambda: True)
        model = _FakeConvNeXt(depths=(1, 1, 3, 1))
        n = compile_regions(model)
        assert n == 6
        blocks = [m for m in model.modules() if isinstance(m, _FakeBlock)]
        assert all(b._compiled_call_impl is not None for b in blocks)

    def test_uncompile_regions_reverts(self):
        model = _FakeConvNeXt(depths=(1, 1, 2, 1))
        compile_regions(model)
        uncompile_regions(model)
        blocks = [m for m in model.modules() if isinstance(m, _FakeBlock)]
        assert all(b._compiled_call_impl is None for b in blocks)

    def test_multihead_backbone_blocks_found(self):
        model = _FakeMultiHead()
        assert len(runtime._compile_targets(model)) == 6

    def test_dual_branch_both_branches_found(self):
        model = _FakeDualBranch()
        # 5 blocks per branch (1+1+2+1), two branches.
        assert len(runtime._compile_targets(model)) == 10

    def test_no_stages_yields_no_targets(self):
        assert runtime._compile_targets(_tiny_model()) == []


class TestCompileCache:
    def test_sets_cache_dirs_under_override(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
        monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
        monkeypatch.setenv("BITTRAINER_COMPILE_CACHE", str(tmp_path))
        configure_compile_cache()
        inductor = tmp_path / "inductor"
        triton = tmp_path / "triton"
        assert inductor.is_dir() and triton.is_dir()
        import os

        assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(inductor)
        assert os.environ["TRITON_CACHE_DIR"] == str(triton)

    def test_respects_preset_cache_dir(self, tmp_path, monkeypatch):
        preset = str(tmp_path / "preset")
        monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", preset)
        monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
        monkeypatch.setenv("BITTRAINER_COMPILE_CACHE", str(tmp_path))
        configure_compile_cache()
        import os

        # setdefault must not clobber a user-supplied value.
        assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == preset


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

    def test_failure_uncompiles_regions(self):
        def _broken_inputs(b, bucket):
            raise RuntimeError("inductor exploded")

        model = _FakeConvNeXt(depths=(1, 1, 2, 1))
        compile_regions(model)
        ok = prewarm_compile(
            model, {(32, 32): 10}, 4, torch.device("cpu"), torch.float32,
            make_inputs=_broken_inputs,
        )
        assert ok is False
        blocks = [m for m in model.modules() if isinstance(m, _FakeBlock)]
        assert all(b._compiled_call_impl is None for b in blocks)


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
