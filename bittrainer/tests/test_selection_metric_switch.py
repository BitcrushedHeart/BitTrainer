"""selection_metric plumbing into non-ordinal checkpoint selection (ISSUE-0490 B).

groups.selection_metric drives the non-ordinal epoch/candidate comparison through
_metric_score: 'macro_f1' (default), 'weighted_f1', or 'balanced' (harmonic mean
of the two). Ordinal groups keep their QWK+macro composite and ignore the setting.
"""

from __future__ import annotations

from bittrainer.group_trainer import GroupTrainConfig, _metric_score


def _cfg(**kw) -> GroupTrainConfig:
    base = dict(group_folder="/tmp/grp", num_classes=4, class_names=["a", "b", "c", "d"])
    base.update(kw)
    return GroupTrainConfig(**base)


# Two epochs where macro and weighted disagree on the winner.
EPOCH_A = {"macro_f1": 0.80, "weighted_f1": 0.60}
EPOCH_B = {"macro_f1": 0.70, "weighted_f1": 0.75}


def test_default_is_macro_f1() -> None:
    cfg = _cfg()
    assert _metric_score(EPOCH_A, cfg) > _metric_score(EPOCH_B, cfg)
    assert _metric_score(EPOCH_A, cfg) == 0.80


def test_weighted_f1_picks_its_own_winner() -> None:
    cfg = _cfg(selection_metric="weighted_f1")
    assert _metric_score(EPOCH_B, cfg) > _metric_score(EPOCH_A, cfg)
    assert _metric_score(EPOCH_B, cfg) == 0.75


def test_balanced_is_harmonic_mean_and_flips() -> None:
    cfg = _cfg(selection_metric="balanced")
    hm_a = 2 * 0.80 * 0.60 / (0.80 + 0.60)
    hm_b = 2 * 0.70 * 0.75 / (0.70 + 0.75)
    assert _metric_score(EPOCH_A, cfg) == hm_a
    assert _metric_score(EPOCH_B, cfg) == hm_b
    assert _metric_score(EPOCH_B, cfg) > _metric_score(EPOCH_A, cfg)


def test_ordinal_ignores_selection_metric() -> None:
    metrics = {"macro_f1": 0.80, "weighted_f1": 0.60, "qwk": 0.9}
    macro_cfg = _cfg(ordinal=True, validation_metric="qwk", selection_metric="macro_f1")
    weighted_cfg = _cfg(ordinal=True, validation_metric="qwk", selection_metric="weighted_f1")
    assert _metric_score(metrics, macro_cfg) == _metric_score(metrics, weighted_cfg)
