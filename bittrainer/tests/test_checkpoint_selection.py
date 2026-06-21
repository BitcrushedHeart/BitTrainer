"""Tests for the epoch checkpoint-selection score.

Regression guard for the failure mode where argmax(QWK) picked a marginally
higher-QWK epoch (0.89) whose exact-match macro-F1 had collapsed (0.45) over a
balanced epoch (QWK 0.88 / F1 0.70). Ordinal selection now uses a weighted
harmonic mean of the ordinal metric and macro-F1 (_composite_selection_score)
so a marginal QWK gain cannot override an F1 collapse.
"""

from __future__ import annotations

from bittrainer.group_trainer import (
    GroupTrainConfig,
    _SELECTION_SECONDARY_WEIGHT,
    _composite_selection_score,
    _metric_score,
)


def _cfg(**kw) -> GroupTrainConfig:
    base = dict(group_folder="/tmp/grp", num_classes=4, class_names=["a", "b", "c", "d"])
    base.update(kw)
    return GroupTrainConfig(**base)


def _ordinal_cfg() -> GroupTrainConfig:
    return _cfg(ordinal=True, validation_metric="qwk")


def test_balanced_epoch_beats_collapsed_epoch():
    """The exact reported regression: QWK 0.88/F1 0.70 must beat QWK 0.89/F1 0.45."""
    cfg = _ordinal_cfg()
    balanced = _metric_score({"qwk": 0.88, "macro_f1": 0.70}, cfg)
    collapsed = _metric_score({"qwk": 0.89, "macro_f1": 0.45}, cfg)
    assert balanced > collapsed


def test_genuine_qwk_gain_still_wins():
    """A real QWK improvement with only a small F1 dip should still be selected."""
    cfg = _ordinal_cfg()
    baseline = _metric_score({"qwk": 0.88, "macro_f1": 0.70}, cfg)
    improved = _metric_score({"qwk": 0.92, "macro_f1": 0.68}, cfg)
    assert improved > baseline


def test_composite_is_bounded_by_components():
    """A harmonic mean never exceeds the larger component and is pulled toward the smaller."""
    score = _composite_selection_score(0.9, 0.5)
    assert 0.5 < score < 0.9
    # Weight sanity: heavier secondary weight pulls the composite further toward F1.
    assert 0.0 < _SELECTION_SECONDARY_WEIGHT < 1.0


def test_degenerate_components_do_not_raise():
    """Zero/negative components (worse-than-chance QWK) fall back to a finite score."""
    assert _composite_selection_score(0.0, 0.0) == 0.0
    assert _composite_selection_score(-0.3, 0.5) >= 0.0  # negative QWK clamped


def test_non_ordinal_unchanged():
    """Non-ordinal groups still select on plain macro-F1."""
    cfg = _cfg(ordinal=False)
    assert _metric_score({"macro_f1": 0.62, "qwk": 0.99}, cfg) == 0.62


def test_guarded_qwk_includes_none_term():
    """guarded_qwk ordinal groups fold the __none__ guard into the primary before compositing."""
    cfg = _cfg(
        ordinal=True,
        validation_metric="guarded_qwk",
        num_classes=4,
        class_names=["a", "b", "c", "__none__"],
    )
    with_none = _metric_score({"qwk": 0.80, "macro_f1": 0.60, "none_f1": 0.9}, cfg)
    without_none = _metric_score({"qwk": 0.80, "macro_f1": 0.60, "none_f1": 0.0}, cfg)
    assert with_none > without_none
