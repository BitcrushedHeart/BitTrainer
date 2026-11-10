"""Tests for autobatch.py — sparse buckets must not clamp the batch size."""

import pytest
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
    is_out_of_memory,
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


class _FailsFrom(nn.Module):
    """Runs a real forward / backward until the batch reaches ``at``, then
    raises ``message`` the way a full card (or a broken step) would."""

    def __init__(self, at: int, message: str) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.at, self.message = at, message
        self.seen: list[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen.append(int(x.shape[0]))
        if x.shape[0] >= self.at:
            raise RuntimeError(self.message)
        return x.mean(dim=(1, 2, 3)) * self.weight


@pytest.fixture
def _fake_card(monkeypatch):
    """A 15 GiB card with 12 GiB free, simulated on the CPU: each image of a
    rung costs 64 MiB at its peak."""
    gib = 1024**3
    peak = {"bytes": 0}
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (12 * gib, 15 * gib))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: peak["bytes"])
    monkeypatch.setattr(
        torch.cuda, "get_per_process_memory_fraction", lambda device: 1.0, raising=False
    )

    def _inputs(batch: int):
        peak["bytes"] = batch * 64 * 1024**2
        return (torch.ones(batch, 3, 4, 4),)

    return _inputs


def _probe(model, make_inputs):
    return profile_vram_batch_size(
        model, make_inputs, torch.device("cuda"), dtype=torch.float32
    )


class TestOnlyMemoryErrorsAreSizedAround:
    """Bitcrush ISSUE-1690: the probe read every ``RuntimeError`` as "out of
    memory". A broken step at a batch of one then fell back to batch 8 and
    failed later, somewhere else. Only running out of memory says "does not
    fit"; anything else is raised as itself."""

    def test_a_broken_step_is_raised_not_sized_around(self, _fake_card):
        model = _FailsFrom(1, "mat1 and mat2 shapes cannot be multiplied")
        model.eval()
        with pytest.raises(RuntimeError, match="cannot be multiplied"):
            _probe(model, _fake_card)
        assert model.seen == [1]
        assert model.training is False  # left as it was found
        assert model.weight.grad is None

    def test_running_out_of_memory_still_ends_the_ladder(self, _fake_card):
        model = _FailsFrom(4, "CUDA out of memory. Tried to allocate 2.00 GiB.")
        result = _probe(model, _fake_card)
        assert model.seen == [1, 2, 4]
        assert result["max_fitted_rung"] == 2
        assert result["vram_limit"] == 2  # the last rung that fitted is the ceiling

    def test_the_t4s_cudnn_wording_counts_as_out_of_memory(self, _fake_card):
        model = _FailsFrom(8, "cuDNN error: CUDNN_STATUS_ALLOC_FAILED")
        result = _probe(model, _fake_card)
        assert model.seen == [1, 2, 4, 8]
        assert result["max_fitted_rung"] == 4

    def test_the_shared_test_reads_every_wording(self):
        assert is_out_of_memory(torch.cuda.OutOfMemoryError("CUDA out of memory"))
        assert is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate"))
        assert is_out_of_memory(RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED"))
        assert is_out_of_memory(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling"))
        assert not is_out_of_memory(RuntimeError("shapes cannot be multiplied"))
        assert not is_out_of_memory(NotImplementedError("missing the forward function"))

    def test_a_full_card_in_cudnns_and_cublas_other_words_is_out_of_memory(self):
        # With cudnn.benchmark on, a card with no room left for the algorithm
        # search says the first; cuBLAS failing to start on a full card says
        # the second. Neither mentions memory.
        assert is_out_of_memory(
            RuntimeError("Unable to find a valid cuDNN algorithm to run convolution")
        )
        assert is_out_of_memory(
            RuntimeError(
                "CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)`"
            )
        )

    def test_an_unknown_error_after_a_batch_has_fitted_ends_the_ladder(self, _fake_card):
        """Home training must not start failing where it used to carry on. Once
        a smaller batch has run the step works, so whatever a larger batch
        raises is the card giving out, in words the shared test does not know."""
        model = _FailsFrom(8, "cuDNN error: CUDNN_STATUS_INTERNAL_ERROR")
        result = _probe(model, _fake_card)
        assert model.seen == [1, 2, 4, 8]
        assert result["max_fitted_rung"] == 4

    def test_every_trainer_asks_the_same_test(self):
        import bittrainer.embedding_cache as embedding_cache
        import bittrainer.generic.tasks.backbone_task as backbone_task
        import bittrainer.generic.tasks.group_task as group_task
        import bittrainer.training_state as training_state

        assert backbone_task._out_of_memory is is_out_of_memory
        for module in (embedding_cache, group_task, training_state):
            assert module.is_out_of_memory is is_out_of_memory
