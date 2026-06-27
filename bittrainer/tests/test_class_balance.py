"""Unit tests for bounded class-balance maths (pure, torch-free)."""

from __future__ import annotations

from bittrainer.class_balance import cap_weight_ratio, capped_equalise_target


# --- capped_equalise_target ------------------------------------------------


def test_uncapped_full_equalisation():
    # max_ratio <= 0 -> legacy behaviour: every non-empty class -> max_count.
    assert capped_equalise_target(10, 1000, 0.0) == 1000
    assert capped_equalise_target(1000, 1000, 0.0) == 1000


def test_cap_limits_extreme_minority():
    # 10 images, largest 1000, 4x cap -> at most 40 (not 1000).
    assert capped_equalise_target(10, 1000, 4.0) == 40
    assert capped_equalise_target(100, 1000, 4.0) == 400


def test_cap_does_not_reduce_mild_imbalance():
    # Within max_ratio of the largest class -> still full equalisation.
    assert capped_equalise_target(500, 1000, 4.0) == 1000
    assert capped_equalise_target(250, 1000, 4.0) == 1000  # exactly 4x boundary


def test_largest_class_unchanged():
    assert capped_equalise_target(1000, 1000, 4.0) == 1000


def test_ratio_one_means_no_oversampling():
    assert capped_equalise_target(100, 1000, 1.0) == 100


def test_ratio_below_one_never_undersamples():
    # A pathological <1 ratio must not drop below the natural count.
    assert capped_equalise_target(100, 1000, 0.5) == 100


def test_empty_class_is_zero():
    assert capped_equalise_target(0, 1000, 4.0) == 0


def test_cap_uses_ceiling():
    # ceil(4 * 7) = 28, well under max_count.
    assert capped_equalise_target(7, 1000, 4.0) == 28


# --- cap_weight_ratio ------------------------------------------------------


def test_weight_cap_bounds_ratio():
    # Rarest class wants 100x the commonest; cap at 4x.
    weights = [0.01, 1.0]  # class 0 common (low weight), class 1 rare (high weight)
    counts = [1000, 10]
    out = cap_weight_ratio(weights, counts, 4.0)
    assert out[0] == 0.01  # commonest untouched
    assert out[1] == 0.01 * 4.0  # rare clamped to 4x the min active weight
    assert max(out) / min(out) == 4.0


def test_weight_cap_noop_when_within_bound():
    weights = [0.5, 1.0]
    counts = [100, 60]
    out = cap_weight_ratio(weights, counts, 4.0)
    assert out == [0.5, 1.0]  # 2:1 already within 4x


def test_weight_cap_ignores_empty_classes():
    # Empty class (count 0) keeps its placeholder weight and is excluded from min().
    weights = [0.01, 1.0, 1.0]
    counts = [1000, 10, 0]
    out = cap_weight_ratio(weights, counts, 4.0)
    assert out[2] == 1.0  # empty passed through
    assert out[1] == 0.04  # capped against the active min (0.01), not the empty 1.0


def test_weight_cap_disabled():
    weights = [0.01, 1.0]
    counts = [1000, 10]
    assert cap_weight_ratio(weights, counts, 0.0) == weights
    assert cap_weight_ratio(weights, counts, -1.0) == weights
