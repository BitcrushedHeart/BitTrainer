"""Model promotion decision logic shared by local and cloud training.

Encodes the resolution order that stops a newly trained candidate from
overwriting a good incumbent without evidence it is worse. Pure (no torch / no
IO) so it is unit-testable and importable on a training pod alongside the
cloud orchestrator.

Resolution order:
  0. Auto-Promote requested      -> promote candidate unconditionally, without
                                    loading or scoring the incumbent at all. The
                                    caller has asserted the freshly trained model
                                    should ship regardless of the incumbent (e.g.
                                    the incumbent is known-leaky on the current
                                    validation split after a re-split). Skips the
                                    class-setup and eval checks below.
  1. No incumbent                -> promote candidate (nothing to beat).
  2. Class-setup mismatch        -> promote candidate; the models are
                                    incomparable and the incumbent is stale
                                    relative to the current task definition. A
                                    mismatch is proven by differing class names
                                    OR differing class counts -- the latter
                                    catches legacy incumbents that predate the
                                    ``class_names`` metadata.
  3. Genuine eval error          -> promote candidate; if the incumbent cannot
                                    be loaded or scored on the candidate's
                                    validation split the two are effectively
                                    incomparable, so the freshly trained
                                    candidate wins. Promotion never overwrites a
                                    *demonstrably* better incumbent -- only one we
                                    could not fairly score.
  4. Compatible + clean eval     -> higher score wins (a tie promotes the
                                    candidate, matching the long-standing local
                                    behaviour).
"""

from __future__ import annotations

from enum import Enum


class PromotionReason(str, Enum):
    auto_promote = "auto_promote"
    no_incumbent = "no_incumbent"
    class_mismatch = "class_mismatch"
    higher_score = "higher_score"
    incumbent_wins = "incumbent_wins"
    eval_error = "eval_error"


class IncumbentEvalError(Exception):
    """The incumbent could not be fairly evaluated against the candidate.

    Retained for backward compatibility (callers/pod orchestrators may still
    import or catch it). The standard promotion flow no longer raises it: an
    eval failure now promotes the candidate (see ``PromotionReason.eval_error``)
    rather than keeping a model that could not be scored.
    """

    def __init__(self, message: str, *, reason_key: str = "OTHER") -> None:
        super().__init__(message)
        self.reason_key = reason_key


def decide_promotion(
    *,
    incumbent_exists: bool,
    incumbent_class_names: list[str] | None,
    candidate_class_names: list[str],
    incumbent_score: float | None,
    candidate_score: float,
    eval_ok: bool,
    incumbent_num_classes: int | None = None,
    candidate_num_classes: int | None = None,
    auto_promote: bool = False,
) -> tuple[bool, PromotionReason]:
    """Decide whether to promote the candidate over the incumbent.

    ``auto_promote`` short-circuits the whole gate: the candidate wins
    unconditionally, before (and instead of) any class-setup or head-to-head
    check. The caller is expected to skip loading/scoring the incumbent entirely
    when it passes this, so the other arguments are ignored.

    ``incumbent_class_names`` is ``None`` for legacy checkpoints that predate
    class-name metadata; a name mismatch cannot be proven, so detection falls
    back to ``num_classes`` when both counts are known. A differing count is
    itself a class-setup mismatch (the label spaces cannot line up).

    ``eval_ok`` must be ``False`` whenever the caller could not load or evaluate
    the incumbent on the same validation split as the candidate. An eval failure
    promotes the candidate: a model that cannot be scored is not a model the
    candidate must beat.
    """
    if auto_promote:
        return True, PromotionReason.auto_promote

    if not incumbent_exists:
        return True, PromotionReason.no_incumbent

    # Class-setup mismatch is resolved up front, before (and regardless of) any
    # head-to-head eval: a different label space makes the models incomparable.
    # Names are the strongest signal; class counts are the fallback that still
    # catches legacy incumbents lacking class-name metadata.
    if incumbent_class_names is not None and list(incumbent_class_names) != list(candidate_class_names):
        return True, PromotionReason.class_mismatch
    if (
        incumbent_num_classes is not None
        and candidate_num_classes is not None
        and incumbent_num_classes != candidate_num_classes
    ):
        return True, PromotionReason.class_mismatch

    # Could not fairly compare -> the candidate wins. A crash/load failure is not
    # evidence the candidate is worse, and the freshly trained candidate is the
    # safer thing to ship than an incumbent we could not even score.
    if not eval_ok or incumbent_score is None:
        return True, PromotionReason.eval_error

    promote = candidate_score >= incumbent_score
    return promote, (PromotionReason.higher_score if promote else PromotionReason.incumbent_wins)
