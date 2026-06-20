"""Tests for autobatch.py — sparse buckets must not clamp the batch size."""

import torch
import torch.nn as nn

import bittrainer.autobatch as autobatch
from bittrainer.autobatch import (
    _apply_trust_bound,
    _linear_fit,
    _make_default_inputs,
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
