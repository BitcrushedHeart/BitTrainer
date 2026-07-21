"""Tests for the QWK-optimal ordinal decode and cut-point fitting.

Under quadratic-weighted kappa the Bayes-optimal prediction is round(E[j]),
not argmax. These guard that ordinal_decode realises that gain, leaves sharp
posteriors untouched, respects the __none__ gate, and that find_ordinal_cut_points
never scores below the round-to-nearest baseline.
"""

from __future__ import annotations

import numpy as np

from bittrainer.group_validation import (
    compute_ordinal_metrics,
    find_ordinal_cut_points,
    ordinal_decode,
)


def test_ev_decode_beats_argmax_on_diffuse_posterior():
    """A skewed-but-diffuse posterior: argmax misses, E[j] lands on the truth.

    Since ISSUE-0540 the EV decode only runs with shipped cut-points (None
    means argmax), so the round-to-nearest midpoints are passed explicitly.
    """
    # True class is 2 (middle). Mode is 1, but mass leans high -> E[j] ~ 1.7.
    probs = np.array([[0.10, 0.35, 0.30, 0.25]])
    assert int(np.argmax(probs)) == 1
    assert ordinal_decode(probs, none_index=-1, cut_points=[0.5, 1.5, 2.5]) == [2]


def test_ev_decode_matches_argmax_when_confident():
    """Sharp posteriors: gated to the mode, decode == argmax."""
    probs = np.array(
        [[0.90, 0.05, 0.03, 0.02], [0.02, 0.03, 0.05, 0.90]]
    )
    assert ordinal_decode(probs, none_index=-1, cut_points=[0.5, 1.5, 2.5]) == [0, 3]


def test_tuned_decode_never_worse_than_argmax_on_a_batch():
    """The SHIPPED decode (EV + fitted cut-points) must never lose to argmax.

    Raw round-to-nearest E[j] can regress vs argmax (it is biased inward at the
    scale edges for symmetric posteriors); the fitted cut-points correct that, so
    the guarantee we actually rely on is tuned-QWK >= argmax-QWK.
    """
    rng = np.random.default_rng(0)
    num_classes = 5
    labels = rng.integers(0, num_classes, size=400)
    # Noisy logits centred on the true class => diffuse posteriors.
    logits = rng.normal(0.0, 1.0, size=(400, num_classes))
    logits[np.arange(400), labels] += 1.2
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    argmax_qwk = compute_ordinal_metrics(
        labels.tolist(), probs.argmax(axis=1).tolist(), num_classes
    )["qwk"]
    cuts = find_ordinal_cut_points(probs, labels, num_classes, none_index=-1)
    tuned_qwk = compute_ordinal_metrics(
        labels.tolist(), ordinal_decode(probs, none_index=-1, cut_points=cuts), num_classes
    )["qwk"]
    assert tuned_qwk >= argmax_qwk - 1e-9


def test_none_gate_preserved():
    """Where overall argmax is __none__, decode returns __none__ (index 0 here)."""
    # none_index = 0; row picks none, second row decodes among real classes.
    probs = np.array([[0.70, 0.10, 0.10, 0.10], [0.05, 0.10, 0.35, 0.50]])
    preds = ordinal_decode(probs, none_index=0)
    assert preds[0] == 0  # none gate fires
    assert preds[1] != 0  # real-class decode


def test_malformed_cut_points_fall_back():
    """Wrong-length cut-points are ignored (argmax, ISSUE-0540), never crash."""
    probs = np.array([[0.10, 0.35, 0.30, 0.25]])
    assert ordinal_decode(probs, none_index=-1, cut_points=[1.0]) == [1]


def test_fewer_than_two_real_classes_is_argmax():
    """Binary (1 real + none) has no ordinal scale -> argmax fallback."""
    probs = np.array([[0.3, 0.7]])
    assert ordinal_decode(probs, none_index=0) == [1]


def test_find_cut_points_monotonic_and_not_worse():
    rng = np.random.default_rng(1)
    num_classes = 4
    labels = rng.integers(0, num_classes, size=300)
    logits = rng.normal(0.0, 1.0, size=(300, num_classes))
    logits[np.arange(300), labels] += 0.8
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    cuts = find_ordinal_cut_points(probs, labels, num_classes, none_index=-1)
    assert cuts is not None
    assert len(cuts) == num_classes - 1
    assert all(cuts[i] < cuts[i + 1] for i in range(len(cuts) - 1))

    base = compute_ordinal_metrics(
        labels.tolist(), ordinal_decode(probs, none_index=-1), num_classes
    )["qwk"]
    tuned = compute_ordinal_metrics(
        labels.tolist(),
        ordinal_decode(probs, none_index=-1, cut_points=cuts),
        num_classes,
    )["qwk"]
    assert tuned >= base - 1e-9


def test_find_cut_points_none_for_degenerate():
    probs = np.array([[0.3, 0.7]])
    assert find_ordinal_cut_points(probs, [1], 2, none_index=0) is None
