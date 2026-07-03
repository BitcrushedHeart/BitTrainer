"""Auto-Promote: the candidate wins unconditionally, bypassing the gate.

Pure (torch-free) unit tests for ``decide_promotion``'s ``auto_promote``
short-circuit — the escape hatch for a known-leaky incumbent (e.g. a re-split
group whose incumbent trained on images now in the current validation split).
"""

from __future__ import annotations

from bittrainer.promotion import PromotionReason, decide_promotion


def test_auto_promote_wins_over_a_better_incumbent():
    """Even a strictly higher-scoring, class-compatible incumbent is bypassed."""
    promote, reason = decide_promotion(
        incumbent_exists=True,
        incumbent_class_names=["a", "b", "c"],
        candidate_class_names=["a", "b", "c"],
        incumbent_score=0.99,
        candidate_score=0.10,
        eval_ok=True,
        auto_promote=True,
    )
    assert promote is True
    assert reason == PromotionReason.auto_promote


def test_auto_promote_reported_reason_is_distinct():
    """The reason is auto_promote — not higher_score/no_incumbent — so the UI
    and logs can show *why* the incumbent was skipped."""
    _, reason = decide_promotion(
        incumbent_exists=False,
        incumbent_class_names=None,
        candidate_class_names=["a", "b"],
        incumbent_score=None,
        candidate_score=0.5,
        eval_ok=False,
        auto_promote=True,
    )
    assert reason == PromotionReason.auto_promote


def test_off_by_default_preserves_existing_gate():
    """auto_promote defaults False; the head-to-head comparison still governs."""
    promote, reason = decide_promotion(
        incumbent_exists=True,
        incumbent_class_names=["a", "b", "c"],
        candidate_class_names=["a", "b", "c"],
        incumbent_score=0.99,
        candidate_score=0.10,
        eval_ok=True,
    )
    assert promote is False
    assert reason == PromotionReason.incumbent_wins
