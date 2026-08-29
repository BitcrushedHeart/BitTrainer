"""Tests for autobatch.py — sparse buckets must not clamp the batch size."""

import torch
import torch.nn as nn

import bittrainer.autobatch as autobatch
from bittrainer.autobatch import (
    _allocator_cap_bytes,
    _apply_trust_bound,
    _linear_fit,
    _make_default_inputs,
    _probe_budget_bytes,
    determine_batch_size,
    profile_vram_batch_size,
)


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Conv2d(3, 4, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))


def _fake_probe(vram_limit: int, max_fitted_rung: int = 32, trust_cap: int = 64):
    def probe(model, make_inputs, device, **kwargs):
        return {
            "vram_limit": vram_limit,
            "max_fitted_rung": max_fitted_rung,
            "trust_cap": trust_cap,
            "fit_slope": 1.0e7,
            "fit_intercept": 1.5e8,
            "predicted_fraction": 0.7,
        }
    return probe


class TestTrustBound:
    def test_no_oom_trusts_fit_to_double_top_rung(self):
        bounded, cap = _apply_trust_bound(100, 32, oomed=False)
        assert (bounded, cap) == (64, 64)

    def test_no_oom_keeps_fit_below_cap(self):
        bounded, cap = _apply_trust_bound(28, 32, oomed=False)
        assert (bounded, cap) == (28, 64)

    def test_oom_falls_back_to_last_fitting_rung(self):
        bounded, cap = _apply_trust_bound(28, 8, oomed=True)
        assert (bounded, cap) == (8, 8)


class TestDetermineBatchSize:
    def test_sparse_bucket_does_not_clamp(self, monkeypatch):
        # The old data_floor clamped every run to 4 when one aspect bucket was
        # sparse — the regression this file exists to prevent.
        monkeypatch.setattr(autobatch, "profile_vram_batch_size", _fake_probe(28))
        result = determine_batch_size(
            _tiny_model(), {(512, 512): 500, (800, 320): 3}, torch.device("cpu"),
        )
        assert result["batch_size"] == 28

    def test_capped_by_total_train_samples(self, monkeypatch):
        monkeypatch.setattr(autobatch, "profile_vram_batch_size", _fake_probe(28))
        result = determine_batch_size(
            _tiny_model(), {(512, 512): 7, (800, 320): 3}, torch.device("cpu"),
        )
        assert result["batch_size"] == 10
        assert result["total_train_samples"] == 10

    def test_minimum_of_four(self, monkeypatch):
        monkeypatch.setattr(autobatch, "profile_vram_batch_size", _fake_probe(2))
        result = determine_batch_size(
            _tiny_model(), {(512, 512): 500}, torch.device("cpu"),
        )
        assert result["batch_size"] == 4

    def test_probe_fields_surface_in_result(self, monkeypatch):
        monkeypatch.setattr(autobatch, "profile_vram_batch_size", _fake_probe(28))
        result = determine_batch_size(
            _tiny_model(), {(512, 512): 500}, torch.device("cpu"),
        )
        assert result["max_fitted_rung"] == 32
        assert result["trust_cap"] == 64
        assert "data_floor" not in result


class TestProbeInputs:
    def test_cpu_device_skips_probe(self):
        result = profile_vram_batch_size(
            _tiny_model(), lambda b: (torch.randn(b, 3, 8, 8),), torch.device("cpu"),
        )
        assert result["vram_limit"] == 32

    def test_channels_last_inputs(self):
        (x,) = _make_default_inputs(
            2, (64, 32), torch.device("cpu"), torch.float32,
            memory_format=torch.channels_last,
        )
        assert x.shape == (2, 3, 32, 64)
        assert x.is_contiguous(memory_format=torch.channels_last)

    def test_default_inputs_contiguous(self):
        (x,) = _make_default_inputs(2, (64, 32), torch.device("cpu"), torch.float32)
        assert x.is_contiguous()


class TestLinearFit:
    def test_recovers_slope_and_intercept(self):
        fit = _linear_fit([1, 2, 4, 8], [3.0, 5.0, 9.0, 17.0])
        assert fit is not None
        slope, intercept = fit
        assert abs(slope - 2.0) < 1e-9
        assert abs(intercept - 1.0) < 1e-9

    def test_degenerate_returns_none(self):
        assert _linear_fit([4], [9.0]) is None
        assert _linear_fit([4, 4], [9.0, 9.0]) is None


class TestAllocatorCapBudget:
    """Bitcrush ISSUE-0870: the probe must size the batch against the allocator
    cap the host set (``set_per_process_memory_fraction``), not against "85% of
    whatever is free" — otherwise a batch that fits the probe still walks the
    caching allocator to the cap and OOMs (or, uncapped on Windows, spills into
    shared RAM at PCIe speed)."""

    GB = 1024 ** 3

    def test_uncapped_budget_is_fraction_of_free(self):
        budget = _probe_budget_bytes(
            free_vram=20 * self.GB, fraction=0.85, param_overhead_bytes=self.GB,
            cap_bytes=None, resident_bytes=0,
        )
        assert budget == 20 * self.GB * 0.85 - self.GB

    def test_cap_below_free_wins(self):
        # 24 GB card, 23 GB free, allocator capped at 19 GB with 1 GB of model
        # already resident: the batch may only use 0.85 * 19 - 1 - overhead.
        budget = _probe_budget_bytes(
            free_vram=23 * self.GB, fraction=0.85, param_overhead_bytes=0.5 * self.GB,
            cap_bytes=19 * self.GB, resident_bytes=1 * self.GB,
        )
        assert budget == 19 * self.GB * 0.85 - 1 * self.GB - 0.5 * self.GB
        assert budget < 23 * self.GB * 0.85 - 0.5 * self.GB

    def test_cap_above_free_is_inert(self):
        budget = _probe_budget_bytes(
            free_vram=8 * self.GB, fraction=0.85, param_overhead_bytes=0,
            cap_bytes=19 * self.GB, resident_bytes=0,
        )
        assert budget == 8 * self.GB * 0.85

    def test_no_cap_when_fraction_unset(self, monkeypatch):
        monkeypatch.setattr(
            torch.cuda, "get_per_process_memory_fraction", lambda device: 1.0, raising=False
        )
        assert _allocator_cap_bytes(torch.device("cpu"), 24 * self.GB) is None

    def test_cap_reads_allocator_fraction(self, monkeypatch):
        monkeypatch.setattr(
            torch.cuda, "get_per_process_memory_fraction", lambda device: 0.5, raising=False
        )
        assert _allocator_cap_bytes(torch.device("cpu"), 24 * self.GB) == 12 * self.GB

    def test_cap_getter_missing_or_broken_is_none(self, monkeypatch):
        def boom(device):
            raise RuntimeError("no cuda")

        monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", boom, raising=False)
        assert _allocator_cap_bytes(torch.device("cpu"), 24 * self.GB) is None
        monkeypatch.delattr(torch.cuda, "get_per_process_memory_fraction", raising=False)
        assert _allocator_cap_bytes(torch.device("cpu"), 24 * self.GB) is None
