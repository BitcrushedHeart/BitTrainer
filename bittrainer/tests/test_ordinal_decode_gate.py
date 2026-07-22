"""ISSUE-0540: confidence-gated ordinal decode + the None => argmax contract.

The QWK-fitted cut-points can push a boundary past a high-confidence argmax
peak (aggregate-QWK-optimal, per-image indefensible: a 96%-confident "Large"
decoded as "Very large"). The gate makes argmax authoritative whenever one
real class holds a majority of the renormalised ordinal mass; EV + cut-points
only decide genuinely ambiguous rows. Separately, ``cut_points=None`` now
means plain argmax — the contract every caller (trainer selection, Engine
inference, strictness curves) already assumed — instead of the silent
EV-midpoint fallback.
"""

from __future__ import annotations

import numpy as np

from bittrainer.group_validation import (
    compute_ordinal_metrics,
    find_ordinal_cut_points,
    ordinal_decode,
)


def test_none_cut_points_is_argmax():
    """No cut-points shipped => plain argmax, even where EV-midpoints differ."""
    # Diffuse row: argmax is 1, EV ~= 1.7 would round to 2 under midpoints.
    probs = np.array([[0.10, 0.35, 0.30, 0.25]])
    assert ordinal_decode(probs, none_index=-1) == [1]
    assert ordinal_decode(probs, none_index=-1, cut_points=None) == [1]


def test_malformed_cut_points_are_argmax():
    """Wrong-length cut-points behave exactly like None (argmax), never crash."""
    probs = np.array([[0.10, 0.35, 0.30, 0.25]])
    assert ordinal_decode(probs, none_index=-1, cut_points=[1.0]) == [1]


def test_confident_peak_never_flipped_by_cut_points():
    """The filed ISSUE-0540 distribution: 96% on "Large" with symmetric 2%
    tails (E[j] exactly on the class) must decode to the argmax class even
    when an adversarial boundary sits at/below it."""
    # none_index=0, real classes 1..7; mass at 5 (0.02), 6 (0.96), 7 (0.02).
    row = np.zeros((1, 8))
    row[0, 5] = 0.02
    row[0, 6] = 0.96
    row[0, 7] = 0.02
    # Boundary 6<->7 placed below 6.0: searchsorted(side="right") buckets
    # E[j]=6.0 into class 7 without the gate.
    cuts = [1.5, 2.5, 3.5, 4.5, 5.5, 5.9]
    assert ordinal_decode(row, none_index=0, cut_points=cuts) == [6]


def test_low_confidence_row_uses_cut_points():
    """Rows below the gate still get the EV + cut-point decode."""
    probs = np.array([[0.10, 0.35, 0.30, 0.25]])  # top real prob 0.35 < 0.5
    # Midpoint boundaries: EV ~= 1.7 -> class 2 (differs from argmax 1).
    assert ordinal_decode(probs, none_index=-1, cut_points=[0.5, 1.5, 2.5]) == [2]


def test_gate_threshold_is_on_renormalised_real_mass():
    """The gate reads the ordinal (real-class) distribution the EV decode
    operates on, after excluding __none__ mass."""
    # Raw top real prob is 0.45 (< 0.5) but with none's 0.2 removed the
    # renormalised mass is 0.45/0.8 = 0.5625 -> gated to argmax.
    probs = np.array([[0.20, 0.45, 0.20, 0.15]])
    cuts = [1.4, 2.5]  # boundary 1<->2 below EV to tempt a flip
    assert ordinal_decode(probs, none_index=0, cut_points=cuts) == [1]


def test_none_override_survives_gate():
    """Overall argmax == __none__ still decodes to __none__ regardless of the
    gate or cut-points."""
    probs = np.array([[0.50, 0.05, 0.25, 0.20]])
    assert ordinal_decode(probs, none_index=0, cut_points=[1.4, 2.5]) == [0]


def test_bimodal_top2_two_apart_decodes_argmax():
    """Top-2 real classes >= 2 apart: E[j] lands between the modes where almost
    no mass lives, so argmax is authoritative (Bitcrush ISSUE-0562)."""
    # Modes at 0 (0.48) and 2 (0.47); EV ~= 0.99 would decode the near-empty
    # middle class under midpoint boundaries. Confidence gate does not fire.
    probs = np.array([[0.48, 0.05, 0.47]])
    assert ordinal_decode(probs, none_index=-1, cut_points=[0.5, 1.5]) == [0]


def test_bimodal_adjacent_top2_still_ev_decoded():
    """Top-2 real classes adjacent: the EV + cut-point decode is untouched."""
    probs = np.array([[0.10, 0.35, 0.30, 0.25]])  # top-2 are classes 1 and 2
    assert ordinal_decode(probs, none_index=-1, cut_points=[0.5, 1.5, 2.5]) == [2]


def test_bimodal_gate_reads_renormalised_real_mass():
    """The top-2 separation is measured on the real (non-none) ordinal scale."""
    # none_index=0; real modes at 1 (0.45) and 3 (0.43), two apart. EV ~= 1.98
    # would decode the middle class 2; renormalised max 0.45/0.94 < 0.5 so the
    # confidence gate alone would not save it.
    probs = np.array([[0.06, 0.45, 0.06, 0.43]])
    assert ordinal_decode(probs, none_index=0, cut_points=[1.5, 2.5]) == [1]


def test_none_override_survives_bimodal_gate():
    """Overall argmax == __none__ still decodes __none__, bimodal or not."""
    probs = np.array([[0.40, 0.31, 0.01, 0.28]])
    assert ordinal_decode(probs, none_index=0, cut_points=[1.5, 2.5]) == [0]


def test_bimodal_and_confidence_gates_compose():
    """A row tripping both gates decodes argmax, same as tripping either."""
    probs = np.array([[0.55, 0.00, 0.45]])  # confident AND bimodal
    # Adversarial boundary at 0.9 would bucket EV=0.90 into class 1.
    assert ordinal_decode(probs, none_index=-1, cut_points=[0.9, 1.1]) == [0]


def test_fitter_never_flips_majority_confident_samples():
    """find_ordinal_cut_points scores through the gated decode, so no fitted
    boundary can flip a row whose renormalised top real prob >= 0.5."""
    rng = np.random.default_rng(42)
    num_classes = 5
    n = 300
    labels = rng.integers(0, num_classes, size=n)
    logits = rng.normal(0.0, 1.0, size=(n, num_classes))
    logits[np.arange(n), labels] += 1.5
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    # Aggregate-QWK bait: mislabel a slice so the fitter is tempted to shove a
    # boundary past confident peaks.
    labels[: n // 10] = np.clip(labels[: n // 10] + 1, 0, num_classes - 1)

    cuts = find_ordinal_cut_points(probs, labels, num_classes, none_index=-1)
    assert cuts is not None
    preds = np.asarray(ordinal_decode(probs, none_index=-1, cut_points=cuts))
    confident = probs.max(axis=1) >= 0.5
    assert np.array_equal(preds[confident], probs.argmax(axis=1)[confident])
