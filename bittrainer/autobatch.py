"""Automatic batch size determination.

One profile-and-fit VRAM probe, shared by every trainer. It profiles a small
ladder of batch sizes (forward+backward), linearly fits the *incremental* peak
memory against batch size, then extrapolates the largest batch whose predicted
cost — plus the optimizer/EMA state that is only allocated *after* sizing — fits
within a fraction of free VRAM. The large batch is solved for, never allocated,
so the probe can't trip NVIDIA's Windows sysmem-fallback stall and runs in
seconds on any card. Extrapolation is trusted up to 2x the largest rung that
actually fit; the only data-derived cap is the total training-set size.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Prodigy_adv keeps exp_avg + exp_avg_sq (1x params each) plus two ~1/11-size
# slice buffers (s, p0) ≈ 2.2x param bytes, in the parameter dtype. Sizing the
# *unfrozen* model means budgeting full-backbone optimizer state — the exact
# memory that the epoch-1 unfreeze + optimizer rebuild allocates.
_OPT_STATE_MULT = 2.2
# Ladder of batch sizes to profile. The largest is extrapolated from these, not
# tested, so we never allocate beyond the top rung.
_PROBE_LADDER = (1, 2, 4, 8, 16)
_PROBE_LADDER_BIG = (1, 2, 4, 8, 16, 32)
_BIG_CARD_BYTES = 16 * (1024 ** 3)
_FALLBACK_BATCH = 8


def _make_default_inputs(
    batch: int, bucket: tuple[int, int], device: torch.device, dtype: torch.dtype,
    memory_format: torch.memory_format | None = None,
) -> tuple[torch.Tensor, ...]:
    w, h = bucket
    x = torch.randn(batch, 3, h, w, device=device, dtype=dtype)
    if memory_format is not None:
        x = x.contiguous(memory_format=memory_format)
    return (x,)


def _apply_trust_bound(vram_limit: int, max_fitted_rung: int, oomed: bool) -> tuple[int, int]:
    """Bound an extrapolated batch by what the probe actually measured.

    Returns ``(bounded_limit, trust_cap)``. After an OOM the last fitting rung
    is the ceiling (Ultralytics' "prior safe point"); otherwise the linear fit
    is trusted up to 2x the largest measured rung.
    """
    trust_cap = max_fitted_rung if oomed else 2 * max_fitted_rung
    return min(vram_limit, trust_cap), trust_cap


def _linear_fit(xs: list[int], ys: list[float]) -> tuple[float, float] | None:
    """Closed-form least-squares (slope, intercept). None if degenerate."""
    n = len(xs)
    if n < 2:
        return None
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def profile_vram_batch_size(
    model: nn.Module,
    make_inputs: Callable[[int], tuple[torch.Tensor, ...]],
    device: torch.device,
    *,
    dtype: torch.dtype = torch.bfloat16,
    fraction: float = 0.85,
    param_overhead_bytes: int = 0,
    max_batch: int | None = None,
    progress_callback: Callable[[int, int, int, str], None] | None = None,
) -> dict:
    """Profile a ladder of batch sizes and extrapolate the largest safe batch.

    Targets ``fraction`` of *currently free* VRAM, minus ``param_overhead_bytes``
    (optimizer + EMA state, which don't exist yet at probe time). Each rung runs
    forward+backward in train mode and records the *incremental* peak allocation
    (delta over the resident model), which a linear fit maps to per-sample and
    fixed costs. The chosen batch is never allocated; the extrapolation is
    trusted only up to 2x the largest rung that actually fit.

    ``progress_callback(attempt, candidate_batch, top_rung, status)`` is invoked
    before each rung ("trying") and after ("fits" / "oom").

    Returns ``{vram_limit, max_fitted_rung, trust_cap, fit_slope, fit_intercept,
    predicted_fraction}``.
    """
    if device.type != "cuda":
        logger.info("VRAM probe skipped (device=%s), defaulting to 32", device)
        return {
            "vram_limit": 32, "max_fitted_rung": None, "trust_cap": None,
            "fit_slope": None, "fit_intercept": None, "predicted_fraction": None,
        }

    torch.cuda.empty_cache()
    free_vram, total_vram = torch.cuda.mem_get_info(device)

    ladder = _PROBE_LADDER_BIG if total_vram >= _BIG_CARD_BYTES else _PROBE_LADDER
    if max_batch is not None:
        ladder = tuple(b for b in ladder if b <= max_batch) or (1,)
    top_rung = ladder[-1]

    was_training = model.training
    model.train()

    xs: list[int] = []
    ys: list[float] = []
    failed_at: int | None = None
    attempt = 0
    for b in ladder:
        attempt += 1
        if progress_callback is not None:
            try:
                progress_callback(attempt, b, top_rung, "trying")
            except Exception:
                pass
        status = "fits"
        try:
            torch.cuda.reset_peak_memory_stats(device)
            alloc_before = torch.cuda.memory_allocated(device)
            inputs = make_inputs(b)
            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                out = model(*inputs)
                loss = out.sum()
            loss.backward()
            peak = torch.cuda.max_memory_allocated(device)
            free_after, _ = torch.cuda.mem_get_info(device)
            model.zero_grad(set_to_none=True)
            del inputs, out, loss
            torch.cuda.empty_cache()

            # On Windows the sysmem fallback may keep allocations succeeding past
            # true VRAM exhaustion by spilling to system RAM via PCIe. If free
            # memory dropped to near zero, treat as effective OOM.
            if free_after < (total_vram * 0.05):
                failed_at = b
                status = "oom"
            else:
                xs.append(b)
                ys.append(float(max(0, peak - alloc_before)))
        except RuntimeError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            failed_at = b
            status = "oom"
        if progress_callback is not None:
            try:
                progress_callback(attempt, b, top_rung, status)
            except Exception:
                pass
        if failed_at is not None:
            break

    model.train(was_training)

    fit = _linear_fit(xs, ys)
    budget = free_vram * fraction - param_overhead_bytes
    if fit is None or fit[0] <= 0 or budget <= 0:
        logger.warning(
            "VRAM probe: degenerate fit (%d valid points, budget=%.2f GB) — falling back to batch %d",
            len(xs), budget / 1e9, _FALLBACK_BATCH,
        )
        return {
            "vram_limit": _FALLBACK_BATCH,
            "max_fitted_rung": max(xs) if xs else None,
            "trust_cap": None,
            "fit_slope": fit[0] if fit else None,
            "fit_intercept": fit[1] if fit else None,
            "predicted_fraction": None,
        }

    slope, intercept = fit
    max_fitted_rung = max(xs)
    vram_limit = max(1, int((budget - intercept) / slope))
    vram_limit, trust_cap = _apply_trust_bound(vram_limit, max_fitted_rung, failed_at is not None)

    used_now = total_vram - free_vram
    predicted = slope * vram_limit + intercept + param_overhead_bytes
    predicted_fraction = (predicted + used_now) / total_vram

    logger.info(
        "VRAM probe: batch=%d at %.1f GB free / %.1f GB total "
        "(slope=%.1f MB/img, intercept=%.2f GB, overhead=%.2f GB, ~%.0f%% of total)",
        vram_limit, free_vram / 1e9, total_vram / 1e9,
        slope / 1e6, intercept / 1e9, param_overhead_bytes / 1e9,
        predicted_fraction * 100,
    )
    return {
        "vram_limit": vram_limit,
        "max_fitted_rung": max_fitted_rung,
        "trust_cap": trust_cap,
        "fit_slope": slope,
        "fit_intercept": intercept,
        "predicted_fraction": predicted_fraction,
    }


def determine_batch_size(
    model: nn.Module,
    bucket_counts: dict[tuple[int, int], int],
    device: torch.device,
    *,
    dtype: torch.dtype = torch.bfloat16,
    vram_fraction: float = 0.85,
    use_ema: bool = False,
    make_inputs: Callable[[int], tuple[torch.Tensor, ...]] | None = None,
    memory_format: torch.memory_format | None = None,
    progress_callback: Callable[[int, int, int, str], None] | None = None,
) -> dict:
    """Determine the optimal batch size from data distribution and VRAM.

    The final batch is ``max(4, min(vram_limit, total_train_samples))`` — sparse
    aspect buckets simply emit partial batches, so they no longer cap the global
    batch size. Pass ``use_ema`` so the EMA shadow copy is budgeted,
    ``memory_format`` so the probe measures the same layout training uses
    (e.g. channels_last), and ``make_inputs`` to override the dummy forward
    inputs (e.g. the dual-branch crops+context pair); the default builds a
    single ``(batch, 3, h, w)`` tensor at the largest active bucket.
    """
    from bittrainer.dataset import ASPECT_RATIO_BUCKETS

    total_train_samples = sum(c for c in bucket_counts.values() if c > 0)

    active_buckets = [b for b in bucket_counts if bucket_counts[b] > 0]
    if not active_buckets:
        active_buckets = ASPECT_RATIO_BUCKETS
    largest_bucket = max(active_buckets, key=lambda b: b[0] * b[1])

    # Optimizer (Prodigy_adv ~2.2x params) + optional EMA (1x params) are
    # allocated after sizing, on the unfrozen model — budget for them explicitly
    # so the epoch-1 unfreeze spike doesn't OOM.
    try:
        param = next(model.parameters())
        num_params = sum(p.numel() for p in model.parameters())
        bytes_per_elem = param.element_size()
    except StopIteration:
        num_params, bytes_per_elem = 0, 2
    overhead = int((_OPT_STATE_MULT + (1.0 if use_ema else 0.0)) * num_params * bytes_per_elem)

    if make_inputs is None:
        def make_inputs(b: int) -> tuple[torch.Tensor, ...]:
            return _make_default_inputs(b, largest_bucket, device, dtype, memory_format)

    probe = profile_vram_batch_size(
        model, make_inputs, device,
        dtype=dtype, fraction=vram_fraction,
        param_overhead_bytes=overhead,
        progress_callback=progress_callback,
    )
    vram_limit = probe["vram_limit"]

    batch_size = max(4, min(vram_limit, total_train_samples))

    sparse = {f"{w}x{h}": n for (w, h), n in bucket_counts.items()
              if 0 < n < batch_size // 4}
    if sparse:
        logger.warning(
            "Sparse buckets (< %d images): %s — these will emit partial batches",
            batch_size // 4, sparse,
        )

    logger.info(
        "Auto batch size: %d (vram_limit=%d, total_train_samples=%d, overhead=%.2f GB)",
        batch_size, vram_limit, total_train_samples, overhead / 1e9,
    )

    return {
        "batch_size": batch_size,
        "total_train_samples": total_train_samples,
        "vram_limit": vram_limit,
        "max_fitted_rung": probe["max_fitted_rung"],
        "trust_cap": probe["trust_cap"],
        "largest_bucket": list(largest_bucket),
        "bucket_counts": {f"{w}x{h}": n for (w, h), n in bucket_counts.items()},
        "overhead_gb": overhead / 1e9,
        "fit_slope": probe["fit_slope"],
        "fit_intercept": probe["fit_intercept"],
        "predicted_fraction": probe["predicted_fraction"],
    }
