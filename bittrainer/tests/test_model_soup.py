"""Unit tests for the greedy weight soup (ISSUE-0392)."""

from __future__ import annotations

import pytest
import torch

from bittrainer.model_soup import average_state_dicts, greedy_soup


def test_average_two_state_dicts():
    a = {"w": torch.tensor([2.0, 4.0]), "n": torch.tensor(3)}
    b = {"w": torch.tensor([0.0, 0.0]), "n": torch.tensor(7)}
    avg = average_state_dicts([a, b])
    assert torch.allclose(avg["w"], torch.tensor([1.0, 2.0]))
    # non-float buffer taken from the first dict, not averaged
    assert avg["n"].item() == 3


def test_average_single_is_copy():
    a = {"w": torch.tensor([1.0, 2.0])}
    out = average_state_dicts([a])
    assert torch.allclose(out["w"], a["w"])
    out["w"][0] = 99.0
    assert a["w"][0] == 1.0  # a clone, not a view


def test_average_preserves_dtype():
    a = {"w": torch.tensor([1.0, 3.0], dtype=torch.bfloat16)}
    b = {"w": torch.tensor([3.0, 1.0], dtype=torch.bfloat16)}
    out = average_state_dicts([a, b])
    assert out["w"].dtype == torch.bfloat16
    assert torch.allclose(out["w"].float(), torch.tensor([2.0, 2.0]))


def test_empty_raises():
    with pytest.raises(ValueError):
        average_state_dicts([])
    with pytest.raises(ValueError):
        greedy_soup([], lambda s: 0.0)


# eval_fn: higher is better == closer to the target [0, 0].
_TARGET = torch.tensor([0.0, 0.0])


def _score(state: dict) -> float:
    return -float(torch.linalg.vector_norm(state["w"] - _TARGET))


def test_greedy_accepts_helpful_rejects_harmful():
    # A and B bracket the target -> their average IS the target (best). C is far
    # and drags the soup away -> must be rejected.
    A = {"w": torch.tensor([1.0, 0.0])}   # score -1
    B = {"w": torch.tensor([-1.0, 0.0])}  # score -1  (avg(A,B) -> [0,0] score 0)
    C = {"w": torch.tensor([5.0, 5.0])}   # score ~-7.07
    cands = [(_score(A), A), (_score(B), B), (_score(C), C)]

    soup, score, accepted = greedy_soup(cands, _score)

    assert set(accepted) == {0, 1}          # A and B in, C out
    assert score == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(soup["w"], _TARGET, atol=1e-6)


def test_soup_never_worse_than_best_single():
    torch.manual_seed(0)
    cands = [(_score(s := {"w": torch.randn(2)}), s) for _ in range(6)]
    best_single = max(_score(s) for _, s in cands)
    _, soup_score, accepted = greedy_soup(cands, _score)
    assert soup_score >= best_single - 1e-6
    assert len(accepted) >= 1


def test_all_harmful_keeps_only_best():
    # Candidates spread far apart so any averaging worsens the score -> soup
    # collapses to just the single best candidate.
    A = {"w": torch.tensor([1.0, 0.0])}
    B = {"w": torch.tensor([0.0, 8.0])}
    C = {"w": torch.tensor([9.0, 9.0])}
    cands = [(_score(A), A), (_score(B), B), (_score(C), C)]
    soup, score, accepted = greedy_soup(cands, _score)
    assert accepted == [0]  # only A (best single); averaging B/C only hurts
    assert torch.allclose(soup["w"], A["w"])
