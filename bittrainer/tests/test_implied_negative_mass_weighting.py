"""Bitcrush ISSUE-0773: mass-normalised implied negatives + pinned val prevalence.

Implied negatives are other concepts' positives — unlabelled for the concept
being trained, not verified negative. Training them at full weight against a
10:1 pool collapsed recall to 24.7% at the served threshold on a live concept,
because the model learned that anything resembling the rest of the dataset was
negative. These tests pin the corrected arithmetic.
"""

from __future__ import annotations

import pytest
import torch

from bittrainer.dataset import DEFAULT_VAL_NEG_POS_RATIO, MAX_BINARY_NEG_POS_RATIO
from bittrainer.generic.tasks.binary_head_only_task import (
    _weighted_cross_entropy,
    _weighted_train_tensors,
    implied_negative_weights,
)
from bittrainer.trainer import TrainConfig


def test_implied_pool_carries_alpha_times_the_positive_mass() -> None:
    # 10 positives against 200 implied negatives: the pathological shape.
    y = torch.cat([torch.ones(10, dtype=torch.long), torch.zeros(200, dtype=torch.long)])
    weights = implied_negative_weights(y, alpha=1.0)

    assert weights[y == 1].sum().item() == pytest.approx(10.0)
    assert weights[y == 0].sum().item() == pytest.approx(10.0)
    # Every implied negative is individually quiet, but the pool is not silent.
    assert weights[y == 0][0].item() == pytest.approx(0.05)


def test_alpha_scales_the_pool_mass_proportionally() -> None:
    y = torch.cat([torch.ones(20, dtype=torch.long), torch.zeros(100, dtype=torch.long)])
    for alpha in (0.5, 1.0, 2.0):
        weights = implied_negative_weights(y, alpha=alpha)
        assert weights[y == 0].sum().item() == pytest.approx(alpha * 20)


def test_alpha_zero_restores_historical_full_weight() -> None:
    y = torch.cat([torch.ones(5, dtype=torch.long), torch.zeros(50, dtype=torch.long)])
    assert torch.equal(implied_negative_weights(y, alpha=0.0), torch.ones(55))


def test_weights_are_safe_when_a_class_is_missing() -> None:
    only_positive = torch.ones(4, dtype=torch.long)
    only_negative = torch.zeros(4, dtype=torch.long)
    assert torch.equal(implied_negative_weights(only_positive, alpha=1.0), torch.ones(4))
    assert torch.equal(implied_negative_weights(only_negative, alpha=1.0), torch.ones(4))


def test_explicit_negatives_keep_full_weight_through_repetition() -> None:
    """Explicit negatives are user-verified, so normalisation must not touch them."""
    x_base = torch.zeros(6, 3)
    y_base = torch.cat([torch.ones(2, dtype=torch.long), torch.zeros(4, dtype=torch.long)])
    w_base = implied_negative_weights(y_base, alpha=1.0)
    x_explicit = torch.ones(3, 3)
    y_explicit = torch.zeros(3, dtype=torch.long)

    x, y, w = _weighted_train_tensors(x_base, y_base, x_explicit, y_explicit, 3, w_base)

    assert x.shape[0] == 6 + 9
    assert y.shape[0] == 15
    # The nine appended explicit rows all carry weight 1.0 ...
    assert torch.equal(w[6:], torch.ones(9))
    # ... while the implied block keeps its normalised weight.
    assert w[2:6].sum().item() == pytest.approx(2.0)


def test_weighted_train_tensors_without_weights_stays_backward_compatible() -> None:
    x_base = torch.zeros(4, 2)
    y_base = torch.cat([torch.ones(2, dtype=torch.long), torch.zeros(2, dtype=torch.long)])
    x, y, w = _weighted_train_tensors(
        x_base, y_base, torch.ones(1, 2), torch.zeros(1, dtype=torch.long), 2
    )
    assert w is None
    assert x.shape[0] == 6 and y.shape[0] == 6


def test_weighted_cross_entropy_matches_plain_ce_when_weights_are_uniform() -> None:
    torch.manual_seed(0)
    logits = torch.randn(16, 2)
    labels = torch.randint(0, 2, (16,))
    plain = _weighted_cross_entropy(logits, labels, None, 0.1)
    uniform = _weighted_cross_entropy(logits, labels, torch.ones(16), 0.1)
    assert torch.allclose(plain, uniform, atol=1e-6)


def test_weighted_cross_entropy_normalises_by_total_weight_not_count() -> None:
    """Halving every weight must not halve the loss — only balance may shift."""
    torch.manual_seed(1)
    logits = torch.randn(12, 2)
    labels = torch.randint(0, 2, (12,))
    full = _weighted_cross_entropy(logits, labels, torch.ones(12), 0.1)
    halved = _weighted_cross_entropy(logits, labels, torch.full((12,), 0.5), 0.1)
    assert torch.allclose(full, halved, atol=1e-6)


def test_weighted_cross_entropy_follows_the_emphasised_class() -> None:
    # Two samples, one of each class; the model is confident and wrong on #1.
    logits = torch.tensor([[5.0, -5.0], [5.0, -5.0]])
    labels = torch.tensor([1, 0])
    emphasise_positive = _weighted_cross_entropy(
        logits, labels, torch.tensor([10.0, 1.0]), 0.0
    )
    emphasise_negative = _weighted_cross_entropy(
        logits, labels, torch.tensor([1.0, 10.0]), 0.0
    )
    assert emphasise_positive > emphasise_negative


def test_zero_total_weight_falls_back_to_the_mean() -> None:
    logits = torch.randn(4, 2)
    labels = torch.randint(0, 2, (4,))
    result = _weighted_cross_entropy(logits, labels, torch.zeros(4), 0.0)
    assert torch.isfinite(result)


def test_config_defaults_encode_the_issue_0773_contract() -> None:
    config = TrainConfig(concept_folder="unused")
    assert config.implied_negative_mass_alpha == 1.0
    assert config.neg_pos_ratio == MAX_BINARY_NEG_POS_RATIO == 5.0
    # None means "follow the train ratio"; Engine pins it explicitly.
    assert config.val_neg_pos_ratio is None
    assert DEFAULT_VAL_NEG_POS_RATIO == 3.0
