"""Tests for promotion.py — the candidate-vs-incumbent resolution order."""

from bittrainer.promotion import PromotionReason, decide_promotion


class TestNoIncumbent:
    def test_promotes_when_nothing_to_beat(self):
        promote, reason = decide_promotion(
            incumbent_exists=False,
            incumbent_class_names=None,
            candidate_class_names=["a", "b"],
            incumbent_score=None,
            candidate_score=0.5,
            eval_ok=False,
        )
        assert promote is True
        assert reason is PromotionReason.no_incumbent


class TestClassMismatch:
    def test_name_mismatch_promotes(self):
        promote, reason = decide_promotion(
            incumbent_exists=True,
            incumbent_class_names=["a", "b"],
            candidate_class_names=["a", "c"],
            incumbent_score=0.9,
            candidate_score=0.1,
            eval_ok=True,
        )
        assert promote is True
        assert reason is PromotionReason.class_mismatch

    def test_count_mismatch_promotes_for_legacy_incumbent(self):
        # Legacy incumbent has no class_names but a different class count — the
        # mismatch is still provable and must resolve cleanly (not crash into an
        # eval error). This is the BMC_Age 71-vs-102 case.
        promote, reason = decide_promotion(
            incumbent_exists=True,
            incumbent_class_names=None,
            candidate_class_names=["c"] * 102,
            incumbent_score=None,
            candidate_score=0.058,
            eval_ok=False,
            incumbent_num_classes=71,
            candidate_num_classes=102,
        )
        assert promote is True
        assert reason is PromotionReason.class_mismatch

    def test_matching_counts_do_not_force_mismatch(self):
        promote, reason = decide_promotion(
            incumbent_exists=True,
            incumbent_class_names=None,
            candidate_class_names=["a", "b"],
            incumbent_score=0.4,
            candidate_score=0.6,
            eval_ok=True,
            incumbent_num_classes=2,
            candidate_num_classes=2,
        )
        assert promote is True
        assert reason is PromotionReason.higher_score


class TestEvalFailurePromotes:
    def test_eval_not_ok_promotes_candidate(self):
        promote, reason = decide_promotion(
            incumbent_exists=True,
            incumbent_class_names=["a", "b"],
            candidate_class_names=["a", "b"],
            incumbent_score=None,
            candidate_score=0.3,
            eval_ok=False,
        )
        assert promote is True
        assert reason is PromotionReason.eval_error

    def test_missing_incumbent_score_promotes_candidate(self):
        promote, reason = decide_promotion(
            incumbent_exists=True,
            incumbent_class_names=None,
            candidate_class_names=["a", "b"],
            incumbent_score=None,
            candidate_score=0.3,
            eval_ok=True,
        )
        assert promote is True
        assert reason is PromotionReason.eval_error


class TestScoreComparison:
    def test_higher_candidate_wins(self):
        promote, reason = decide_promotion(
            incumbent_exists=True,
            incumbent_class_names=["a", "b"],
            candidate_class_names=["a", "b"],
            incumbent_score=0.5,
            candidate_score=0.7,
            eval_ok=True,
        )
        assert promote is True
        assert reason is PromotionReason.higher_score

    def test_tie_promotes_candidate(self):
        promote, reason = decide_promotion(
            incumbent_exists=True,
            incumbent_class_names=["a", "b"],
            candidate_class_names=["a", "b"],
            incumbent_score=0.5,
            candidate_score=0.5,
            eval_ok=True,
        )
        assert promote is True
        assert reason is PromotionReason.higher_score

    def test_lower_candidate_keeps_incumbent(self):
        promote, reason = decide_promotion(
            incumbent_exists=True,
            incumbent_class_names=["a", "b"],
            candidate_class_names=["a", "b"],
            incumbent_score=0.8,
            candidate_score=0.2,
            eval_ok=True,
        )
        assert promote is False
        assert reason is PromotionReason.incumbent_wins
