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
    _guarded_metric_enabled,
    _metric_score,
    _score_metric_label,
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


def test_none_guard_off_by_default_for_non_ordinal_none_group():
    """A non-ordinal group with a __none__ class selects on RAW macro-F1.

    Regression guard for the half-removed guard: ``_guarded_metric_enabled``
    used to auto-activate ``guarded_macro_f1`` (macro_f1 + 0.1*none_f1) for any
    non-ordinal group that happened to own a __none__ class, silently demoting
    raw macro-F1. With ``none_guard`` off (the default) the __none__ F1 term
    must not enter selection.
    """
    cfg = _cfg(ordinal=False, num_classes=4, class_names=["a", "b", "c", "__none__"])
    assert not _guarded_metric_enabled(cfg)
    assert _score_metric_label(cfg) == "macro_f1"
    high_none = _metric_score({"macro_f1": 0.62, "none_f1": 0.99}, cfg)
    low_none = _metric_score({"macro_f1": 0.62, "none_f1": 0.0}, cfg)
    assert high_none == low_none == 0.62


def test_none_guard_off_by_default_for_ordinal_group():
    """Ordinal groups select on the qwk/macro-F1 composite with NO __none__ term."""
    cfg = _cfg(
        ordinal=True,
        validation_metric="guarded_qwk",
        num_classes=4,
        class_names=["a", "b", "c", "__none__"],
    )
    assert not _guarded_metric_enabled(cfg)
    with_none = _metric_score({"qwk": 0.80, "macro_f1": 0.60, "none_f1": 0.9}, cfg)
    without_none = _metric_score({"qwk": 0.80, "macro_f1": 0.60, "none_f1": 0.0}, cfg)
    assert with_none == without_none


def test_none_guard_opt_in_folds_in_none_term():
    """When explicitly enabled, the __none__ guard is restored for both paths."""
    nonordinal = _cfg(
        ordinal=False, num_classes=4, class_names=["a", "b", "c", "__none__"], none_guard=True
    )
    assert _guarded_metric_enabled(nonordinal)
    assert _score_metric_label(nonordinal) == "guarded_macro_f1"
    assert _metric_score({"macro_f1": 0.62, "none_f1": 0.99}, nonordinal) > 0.62

    ordinal = _cfg(
        ordinal=True,
        validation_metric="guarded_qwk",
        num_classes=4,
        class_names=["a", "b", "c", "__none__"],
        none_guard=True,
    )
    with_none = _metric_score({"qwk": 0.80, "macro_f1": 0.60, "none_f1": 0.9}, ordinal)
    without_none = _metric_score({"qwk": 0.80, "macro_f1": 0.60, "none_f1": 0.0}, ordinal)
    assert with_none > without_none
