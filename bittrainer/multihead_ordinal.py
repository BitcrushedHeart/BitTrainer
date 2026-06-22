"""Ordinal bookkeeping for the multi-head size model.

The size head predicts the full US size (e.g. ``34DD``); the band head predicts the
band only (e.g. ``34``). Both are ordinal, but the size scale is ordered by **volume**,
not by class-creation order, so that *sister sizes* (e.g. ``32D``, ``34C``,
``36B`` — same volume, different band) are treated as equivalent on the ordinal scale and
in the soft-label targets. The band head distinguishes them.

This module is self-contained (no torch) so the mappings are cheap to unit-test.
"""

from __future__ import annotations

import re

# US cup ladder: index = volume step (one step ~= 1 inch bust-minus-underbust). Matches
# the suite's services/bra_size_conversion US ladder and the demo's CUP_VOLUME_INDEX.
US_CUP_LADDER: list[str] = [
    "AAA", "AA", "A", "B", "C", "D", "DD", "DDD", "DDDD",
    "G", "H", "I", "J", "K", "L", "M", "N", "O",
]
_CUP_INDEX: dict[str, int] = {c: i for i, c in enumerate(US_CUP_LADDER)}

# Band step in inches that equals one cup volume step.
_BAND_BASELINE = 28
_BAND_STEP = 2

_SIZE_RE = re.compile(r"^(\d{2,3})([A-Z]+)$")


def parse_size(size: str) -> tuple[int, str] | None:
    """Split a US size string like ``"34DD"`` into ``(34, "DD")``, or None if unparseable."""
    m = _SIZE_RE.match(size.strip().upper())
    if not m:
        return None
    cup = m.group(2)
    if cup not in _CUP_INDEX:
        return None
    return int(m.group(1)), cup


def volume_index(size: str) -> float | None:
    """Breast-volume index for a size: ``(band - 28) / 2 + cup_index``.

    Sister sizes share a volume (``32D`` = ``34C`` = ``36B``), which is exactly what the
    ordinal scale and soft-label targets should treat as equivalent.
    """
    parsed = parse_size(size)
    if parsed is None:
        return None
    band, cup = parsed
    return (band - _BAND_BASELINE) / _BAND_STEP + _CUP_INDEX[cup]


def band_of(size: str) -> int | None:
    """Band number for a size (``"34DD"`` -> ``34``), or None if unparseable."""
    parsed = parse_size(size)
    return None if parsed is None else parsed[0]


def build_band_vocab(size_classes: list[str], *, none_name: str = "__none__") -> list[str]:
    """Sorted ascending list of distinct band labels (as strings) across the size classes.

    ``__none__`` and unparseable size classes contribute no band.
    """
    bands: set[int] = set()
    for name in size_classes:
        if name == none_name:
            continue
        b = band_of(name)
        if b is not None:
            bands.add(b)
    return [str(b) for b in sorted(bands)]


def size_to_band_index(
    size_classes: list[str],
    band_vocab: list[str],
    *,
    none_name: str = "__none__",
    ignore_index: int = -1,
) -> list[int]:
    """For each size class, the band head's target index (``ignore_index`` for ``__none__``).

    The band head trains only on the band, ignoring the cup — so ``__none__`` and
    unparseable classes are ignored (their band target is ``ignore_index``).
    """
    band_pos = {b: i for i, b in enumerate(band_vocab)}
    out: list[int] = []
    for name in size_classes:
        if name == none_name:
            out.append(ignore_index)
            continue
        b = band_of(name)
        out.append(band_pos.get(str(b), ignore_index) if b is not None else ignore_index)
    return out


def size_to_volume(
    size_classes: list[str],
    *,
    none_name: str = "__none__",
    none_value: float = -100.0,
) -> list[float]:
    """Per-size-class volume index; ``none_value`` for ``__none__``/unparseable.

    Used by the volume-distance soft-label loss so equivalent sizes get partial credit.
    """
    out: list[float] = []
    for name in size_classes:
        if name == none_name:
            out.append(none_value)
            continue
        v = volume_index(name)
        out.append(v if v is not None else none_value)
    return out


def volume_ranks(
    size_classes: list[str],
    *,
    none_name: str = "__none__",
    none_index: int = -1,
) -> list[int]:
    """Map each size class to an integer ordinal rank by volume (ties share a rank).

    Sister sizes share a rank, so the size-head QWK treats sister-size confusion as zero
    ordinal error. ``__none__`` maps to ``none_index`` (off the ordinal scale).
    """
    volumes = sorted(
        {volume_index(n) for n in size_classes if n != none_name and volume_index(n) is not None}
    )
    rank_of = {v: i for i, v in enumerate(volumes)}
    out: list[int] = []
    for name in size_classes:
        if name == none_name:
            out.append(none_index)
            continue
        v = volume_index(name)
        out.append(rank_of.get(v, none_index) if v is not None else none_index)
    return out
