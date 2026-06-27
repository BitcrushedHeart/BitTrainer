"""Bounded class-balance maths — pure, torch-free, unit-testable.

The training default historically equalised every class by *replicating* minority
samples up to the largest class with **no ceiling** on the replication factor (a
10-image class against a 1000-image class was replicated ~100x). Exact-duplicate
over-replication memorises the few distinct minority images and hurts real-world
(deployment-distribution) F1, because the dataset skew is a genuine prior, not noise.

These helpers bound the correction at ``max_ratio`` so a minority class is lifted at
most ``max_ratio`` x — enough to be learnable, without pretending it is as frequent as
the majority. The same ratio bounds both balancing mechanisms:

* ``resample`` mode — :func:`capped_equalise_target` caps the *replication target*.
* ``reweight`` mode — :func:`cap_weight_ratio` caps the *loss-weight ratio*.

Kept in its own module (importing only :mod:`math`) so the maths can be tested without
importing torch — i.e. without touching a GPU that may be mid-training.
"""

from __future__ import annotations

import math


def capped_equalise_target(n: int, max_count: int, max_ratio: float) -> int:
    """Target sample count for a class when balancing by replication.

    ``n`` is the class's natural (deduplicated) size, ``max_count`` the largest class's
    size. With ``max_ratio <= 0`` this is the legacy behaviour — full equalisation to
    ``max_count``. Otherwise the class is replicated to at most ``ceil(max_ratio * n)``,
    never above ``max_count`` and never below its natural size ``n`` (so ``max_ratio < 1``
    cannot *undersample*).

    Examples (``max_ratio=4``): n=10,  max=1000 -> 40 (capped); n=500, max=1000 -> 1000
    (within 4x, full equalise); n=1000, max=1000 -> 1000 (the largest class).
    """
    if n <= 0:
        return 0
    if max_ratio and max_ratio > 0:
        capped = max(n, math.ceil(max_ratio * n))
        return min(max_count, capped)
    return max_count


def cap_weight_ratio(
    weights: list[float], counts: list[int], max_ratio: float
) -> list[float]:
    """Clamp per-class loss weights so max/min over *non-empty* classes <= ``max_ratio``.

    ``weights[i]`` is the raw (pre-normalisation) weight for class ``i`` and ``counts[i]``
    its sample count; empty classes (count 0) are passed through untouched. The most
    common class has the smallest weight, so we cap every active weight at
    ``min_active_weight * max_ratio``. ``max_ratio <= 0`` is a no-op (uncapped).
    """
    if not max_ratio or max_ratio <= 0:
        return list(weights)
    active = [w for w, c in zip(weights, counts) if c > 0]
    if not active:
        return list(weights)
    ceiling = min(active) * max_ratio
    return [min(w, ceiling) if c > 0 else w for w, c in zip(weights, counts)]
