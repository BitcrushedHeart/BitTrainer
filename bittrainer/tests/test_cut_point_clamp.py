"""Bitcrush ISSUE-0562: fitted ordinal cut-points must stay near neutral.

Unconstrained coordinate ascent on validation QWK let a boundary drift far
from its half-integer neutral (a shipped Horizontal Angle model had the
Side/TQR boundary at 3.84 vs neutral 3.5, squeezing one class band to width
0.32). The fit is now clamped to neutral +/- 0.25 so no band can shrink below
half its natural width, and the decode stays defensible per image.
"""

from __future__ import annotations

import numpy as np

from bittrainer.group_validation import (
    compute_ordinal_metrics,
    find_ordinal_cut_points,
    ordinal_decode,
)

CLAMP = 0.25


def _biased_val_data(rng, num_classes, n, none_index=-1):
    """Diffuse posteriors whose labels are shifted up for a large slice —
    aggregate-QWK bait that pulls boundaries far below neutral when the fit
    is unconstrained."""
    real = [i for i in range(num_classes) if i != none_index]
    labels = rng.choice(real, size=n)
    logits = rng.normal(0.0, 1.0, size=(n, num_classes))
    logits[np.arange(n), labels] += 0.8
    if none_index >= 0:
        logits[:, none_index] -= 2.0
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    shifted = labels.copy()
    shifted[: n // 3] = np.minimum(shifted[: n // 3] + 1, max(real))
    return probs, shifted


def test_fitted_boundaries_within_clamp_of_neutral():
    rng = np.random.default_rng(7)
    num_classes = 5
    probs, labels = _biased_val_data(rng, num_classes, 600)
    cuts = find_ordinal_cut_points(probs, labels, num_classes, none_index=-1)
    assert cuts is not None
    neutral = np.arange(num_classes - 1) + 0.5
    assert np.all(np.abs(np.asarray(cuts) - neutral) <= CLAMP + 1e-9)


def test_clamp_respects_none_offset_neutrals():
    """With __none__ at index 0 the real scale is 1..5 and neutrals sit at
    1.5, 2.5, 3.5, 4.5 — the clamp anchors there, not at 0.5-offsets."""
    rng = np.random.default_rng(11)
    num_classes = 6
    probs, labels = _biased_val_data(rng, num_classes, 600, none_index=0)
    cuts = find_ordinal_cut_points(probs, labels, num_classes, none_index=0)
    assert cuts is not None
    neutral = np.arange(1, num_classes - 1) + 0.5
    assert np.all(np.abs(np.asarray(cuts) - neutral) <= CLAMP + 1e-9)


def test_clamped_cuts_monotonic_and_not_worse_than_neutral():
    rng = np.random.default_rng(3)
    num_classes = 5
    probs, labels = _biased_val_data(rng, num_classes, 400)
    cuts = find_ordinal_cut_points(probs, labels, num_classes, none_index=-1)
    assert cuts is not None
    assert all(cuts[i] < cuts[i + 1] for i in range(len(cuts) - 1))

    neutral = (np.arange(num_classes - 1) + 0.5).tolist()
    labels_list = labels.tolist()
    fitted = compute_ordinal_metrics(
        labels_list,
        ordinal_decode(probs, none_index=-1, cut_points=cuts),
        num_classes,
    )["qwk"]
    base = compute_ordinal_metrics(
        labels_list,
        ordinal_decode(probs, none_index=-1, cut_points=neutral),
        num_classes,
    )["qwk"]
    assert fitted >= base - 1e-9
