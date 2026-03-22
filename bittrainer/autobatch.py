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
    fraction: float = 0.75,
) -> int:
    """Binary search for the largest batch size fitting within VRAM fraction.

    Uses the largest bucket dimensions (worst case per sample) to ensure
    no OOM regardless of which bucket is being processed.

    The model must already be on the target device.
    """
    total_vram = torch.cuda.get_device_properties(device).total_mem
    target_usage = int(total_vram * fraction)

    w, h = largest_bucket

    low, high, best = 1, 512, 4
    while low <= high:
        mid = (low + high) // 2
        try:
            torch.cuda.reset_peak_memory_stats(device)
            dummy = torch.randn(mid, 3, h, w, device=device, dtype=dtype)
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                    _ = model(dummy)
            peak = torch.cuda.max_memory_allocated(device)
            del dummy
            torch.cuda.empty_cache()

            if peak <= target_usage:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        except RuntimeError:
            torch.cuda.empty_cache()
            high = mid - 1

    logger.info(
        "VRAM probe: best=%d at %dx%d (%.1f/%.1f GB, %.0f%% target)",
        best, w, h,
        torch.cuda.max_memory_allocated(device) / 1e9,
        total_vram / 1e9,
        fraction * 100,
    )
    return max(4, best)


def determine_batch_size(
    model: nn.Module,
    bucket_counts: dict[tuple[int, int], int],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    vram_fraction: float = 0.75,
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

    batch_size = max(4, min(data_floor, vram_limit))

    logger.info(
        "Auto batch size: %d (data_floor=%d, vram_limit=%d)",
        batch_size, data_floor, vram_limit,
    )

    return {
        "batch_size": batch_size,
        "data_floor": data_floor,
        "vram_limit": vram_limit,
        "largest_bucket": largest_bucket,
        "bucket_counts": dict(bucket_counts),
    }
