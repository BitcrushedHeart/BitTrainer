"""Automatic batch size determination.

Calculates the largest batch size that:
1. Doesn't exceed available data per bucket (data floor)
2. Fits within 75% of available GPU VRAM (VRAM cap)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_data_floor(bucket_counts: dict[tuple[int, int], int]) -> int:
    """Calculate the maximum batch size from bucket sample distribution.

    batch_size <= min(smallest_bucket * 1.5, largest_bucket)

    The 1.5x on smallest accounts for horizontal flip augmentation
    effectively increasing that bucket's capacity.
    """
    if not bucket_counts:
        return 4

    counts = [c for c in bucket_counts.values() if c > 0]
    if not counts:
        return 4

    smallest = min(counts)
    largest = max(counts)
    floor = int(min(smallest * 1.5, largest))
    return max(4, floor)


def probe_vram_batch_size(
    model: nn.Module,
    largest_bucket: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    fraction: float = 0.6,
) -> int:
    """Binary search for the largest batch size fitting within free VRAM.

    Targets ``fraction`` of *currently free* VRAM, not total. Total-based
    sizing is unsafe on Windows: NVIDIA's CUDA sysmem fallback policy spills
    over-allocations into system RAM via PCIe instead of OOMing, which stalls
    the process for tens of minutes. Free-based sizing also leaves room for
    the desktop compositor, browser, and any other GPU consumers.
    """
    if device.type != "cuda":
        logger.info("VRAM probe skipped (device=%s), defaulting to 32", device)
        return 32

    torch.cuda.empty_cache()
    free_vram, total_vram = torch.cuda.mem_get_info(device)
    target_usage = int(free_vram * fraction)

    w, h = largest_bucket
    was_training = model.training
    model.train()

    # Cap the upper bound at a batch size whose dummy tensor alone wouldn't
    # exceed half of target — anything larger is guaranteed not to fit once
    # activations + gradients land on top.
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    bytes_per_sample = 3 * h * w * bytes_per_elem
    upper_cap = max(4, min(512, target_usage // (2 * max(1, bytes_per_sample))))

    low, high, best = 1, int(upper_cap), 4
    while low <= high:
        mid = (low + high) // 2
        try:
            torch.cuda.reset_peak_memory_stats(device)
            free_before, _ = torch.cuda.mem_get_info(device)
            dummy = torch.randn(mid, 3, h, w, device=device, dtype=dtype)
            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                logits = model(dummy)
                loss = logits.sum()
            loss.backward()
            model.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated(device)
            free_after, _ = torch.cuda.mem_get_info(device)
            del dummy, logits, loss
            torch.cuda.empty_cache()

            # On Windows the sysmem fallback may keep allocations succeeding
            # past true VRAM exhaustion. If free memory dropped to near zero,
            # treat as effective OOM regardless of what max_memory_allocated says.
            sysmem_overflow = free_after < (total_vram * 0.05)
            if sysmem_overflow or peak > target_usage:
                high = mid - 1
            else:
                best = mid
                low = mid + 1
        except RuntimeError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            high = mid - 1

    model.train(was_training)

    logger.info(
        "VRAM probe: best=%d at %dx%d (%.1f GB free / %.1f GB total, %.0f%% of free targeted, upper_cap=%d)",
        best, w, h,
        free_vram / 1e9,
        total_vram / 1e9,
        fraction * 100,
        upper_cap,
    )
    return max(4, best)


def determine_batch_size(
    model: nn.Module,
    bucket_counts: dict[tuple[int, int], int],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    vram_fraction: float = 0.6,
) -> dict:
    """Determine the optimal batch size from data distribution and VRAM.

    Returns a dict with:
        batch_size: final chosen batch size
        data_floor: max from data distribution
        vram_limit: max from VRAM probe
        largest_bucket: (w, h) used for VRAM probe
        bucket_counts: the input counts
    """
    from bittrainer.dataset import ASPECT_RATIO_BUCKETS

    data_floor = compute_data_floor(bucket_counts)

    # Find the largest bucket by pixel count for VRAM probing
    active_buckets = [b for b in bucket_counts if bucket_counts[b] > 0]
    if not active_buckets:
        active_buckets = ASPECT_RATIO_BUCKETS

    largest_bucket = max(active_buckets, key=lambda b: b[0] * b[1])

    vram_limit = probe_vram_batch_size(
        model, largest_bucket, device, dtype=dtype, fraction=vram_fraction,
    )

    batch_size = max(4, vram_limit)

    # Warn about very sparse buckets (would get partial batches)
    sparse = {f"{w}x{h}": n for (w, h), n in bucket_counts.items()
              if 0 < n < batch_size // 4}
    if sparse:
        logger.warning(
            "Sparse buckets (< %d images): %s — these will emit partial batches",
            batch_size // 4, sparse,
        )

    logger.info(
        "Auto batch size: %d (data_floor=%d, vram_limit=%d)",
        batch_size, data_floor, vram_limit,
    )

    return {
        "batch_size": batch_size,
        "data_floor": data_floor,
        "vram_limit": vram_limit,
        "largest_bucket": list(largest_bucket),
        "bucket_counts": {f"{w}x{h}": n for (w, h), n in bucket_counts.items()},
    }
