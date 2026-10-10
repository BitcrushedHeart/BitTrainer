"""Caption-conditioned pairwise reward head (Bitcrush Optimise P8, ISSUE-1133).

``RewardHead`` scores an (image embedding, text embedding) pair with an MLP over
``[i, t, i*t, |i-t|]``; the training objective is the pairwise ranking loss
``softplus(-(r_chosen - r_rejected))`` = ``-log sigmoid(r_c - r_r)``; a
published model carries a FIXED normalisation mapping from raw rewards to the
0-100 Suite score scale. CPU only, tiny dims.
"""

from __future__ import annotations

import math

import pytest
import torch

from bittrainer.reward_head import (
    RewardHead,
    apply_norm_mapping,
    fit_norm_mapping,
    pair_features,
    pairwise_ranking_loss,
    preference_accuracy,
    preference_margin,
)

DIM = 8


def test_pair_features_are_the_four_blocks_in_order():
    i = torch.tensor([[3.0, 4.0]])
    t = torch.tensor([[0.0, 1.0]])
    feats = pair_features(i, t, normalise=False)
    assert feats.shape == (1, 8)
    torch.testing.assert_close(
        feats, torch.tensor([[3.0, 4.0, 0.0, 1.0, 0.0, 4.0, 3.0, 3.0]])
    )
    # Normalised: both vectors become unit length before combining.
    normed = pair_features(i, t, normalise=True)
    torch.testing.assert_close(normed[0, :2], torch.tensor([0.6, 0.8]))


def test_head_scores_one_scalar_per_pair_and_roundtrips_its_arch():
    head = RewardHead(DIM, hidden_dim=16, dropout=0.0).eval()
    i = torch.randn(5, DIM)
    t = torch.randn(5, DIM)
    r = head(i, t)
    assert r.shape == (5,)
    assert head.arch == {
        "family": "mlp4",
        "embed_dim": DIM,
        "hidden_dim": 16,
        "dropout": 0.0,
        "normalise": True,
    }
    clone = RewardHead.from_arch(head.arch)
    clone.load_state_dict(head.state_dict())
    clone.eval()
    torch.testing.assert_close(clone(i, t), r)


def test_head_rejects_mismatched_embedding_dims():
    head = RewardHead(DIM, hidden_dim=16)
    with pytest.raises((RuntimeError, ValueError)):
        head(torch.randn(2, DIM), torch.randn(2, DIM + 1))


def test_ranking_loss_is_negative_log_sigmoid_of_the_margin():
    r_c = torch.tensor([2.0, 0.0, -1.0])
    r_r = torch.tensor([0.0, 0.0, 1.0])
    loss = pairwise_ranking_loss(r_c, r_r)
    want = torch.stack(
        [-torch.log(torch.sigmoid(r_c[k] - r_r[k])) for k in range(3)]
    ).mean()
    torch.testing.assert_close(loss, want)
    # Larger margins -> smaller loss; loss at zero margin is log 2.
    assert pairwise_ranking_loss(torch.tensor([5.0]), torch.tensor([0.0])) < pairwise_ranking_loss(
        torch.tensor([1.0]), torch.tensor([0.0])
    )
    torch.testing.assert_close(
        pairwise_ranking_loss(torch.tensor([1.0]), torch.tensor([1.0])),
        torch.tensor(math.log(2.0)),
    )


def test_preference_accuracy_and_margin():
    r_c = torch.tensor([1.0, 0.5, -1.0, 2.0])
    r_r = torch.tensor([0.0, 0.5, 0.0, 1.0])
    # Ties count as not preferred.
    assert preference_accuracy(r_c, r_r) == pytest.approx(0.5)
    assert preference_margin(r_c, r_r) == pytest.approx((1.0 + 0.0 - 1.0 + 1.0) / 4)
    assert preference_accuracy(torch.tensor([]), torch.tensor([])) is None


def test_head_learns_a_separable_preference_on_cpu():
    torch.manual_seed(0)
    n = 128
    t = torch.randn(n, DIM)
    # Chosen images point along their caption, rejected ones away from it.
    i_c = t + 0.1 * torch.randn(n, DIM)
    i_r = -t + 0.1 * torch.randn(n, DIM)
    head = RewardHead(DIM, hidden_dim=32, dropout=0.0)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(150):
        opt.zero_grad()
        loss = pairwise_ranking_loss(head(i_c, t), head(i_r, t))
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        acc = preference_accuracy(head(i_c, t), head(i_r, t))
    assert acc > 0.95


def test_norm_mapping_is_fixed_monotone_and_bounded():
    rewards = [-2.0, -1.0, 0.0, 1.0, 2.0, 10.0]
    mapping = fit_norm_mapping(rewards)
    assert mapping["type"] == "sigmoid"
    assert mapping["version"] == 1
    assert mapping["centre"] == pytest.approx(0.5)  # median
    assert mapping["scale"] > 0
    scores = [apply_norm_mapping(mapping, r) for r in rewards]
    assert scores == sorted(scores)
    assert all(0.0 <= s <= 100.0 for s in scores)
    assert apply_norm_mapping(mapping, mapping["centre"]) == pytest.approx(50.0)
    # The mapping is data the model carries; applying it never refits.
    frozen = dict(mapping)
    apply_norm_mapping(mapping, 123.0)
    assert mapping == frozen
    # Degenerate (constant) rewards still give a usable, finite mapping.
    flat = fit_norm_mapping([3.0, 3.0, 3.0])
    assert flat["scale"] > 0
    assert apply_norm_mapping(flat, 3.0) == pytest.approx(50.0)
    with pytest.raises(ValueError):
        fit_norm_mapping([])
    with pytest.raises(ValueError):
        apply_norm_mapping({"type": "unknown", "version": 1}, 0.0)
