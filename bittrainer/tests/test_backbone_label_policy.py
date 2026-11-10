"""What the backbone trainer does with unlabelled images (Bitcrush ISSUE-1655).

For each concept some images carry a human yes, fewer a human no, and the rest
are unlabelled. Under ``soft_implicit_negative`` an unlabelled image can join a
head's training as an implied negative. Three options, all off by default,
change what happens to those pairs and to nothing else:

* large-loss rejection: from the second epoch, the highest-scoring implied
  negatives of a batch take no part in the loss;
* a Hill term in place of BCE towards the soft target;
* no implied negatives at all for a concept that is common.

A human yes and a human no always count in full, with plain BCE. To compare the
options, validation reports recall at precision 0.90 and 0.95 per head.

Small tensors and generated scores only: no images, no training run.
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import bittrainer.backbone_trainer as bb
from bittrainer.tests.test_backbone_head_vectorised import (
    CPU,
    FEATURES,
    _features,
    _heads,
    _reference_batch_loss,
)
from bittrainer.tests.test_backbone_sampling import _vocab

_SOFT = {"mode": "soft_implicit_negative", "implicit_negative_value": 0.1}
_HILL = {**_SOFT, "implicit_negative_loss": "hill"}


def _implied(value: float = 0.1) -> float:
    return bb._ImpliedNegative(value)


# -- the plan: which pairs are implied ----------------------------------------


def _pool():
    """Concept "c": 10 human yes, 4 human no, and 100 images nobody labelled
    for it (they carry concept "b" only)."""
    samples = [bb._Sample(f"p{i}.png", {"c": 1.0}, {}) for i in range(10)]
    samples += [bb._Sample(f"n{i}.png", {"c": 0.0}, {}) for i in range(4)]
    samples += [bb._Sample(f"u{i}.png", {"b": float(i % 2)}, {}) for i in range(100)]
    return samples


def test_implied_negatives_are_marked_and_human_labels_are_not():
    planned, stats = bb._plan_epoch_samples(
        _pool(), _vocab(("c", "b")), epoch=0, label_policy=_SOFT
    )

    implied = [
        sample.binary["c"]
        for sample in planned
        if isinstance(sample.binary.get("c"), bb._ImpliedNegative)
    ]
    assert len(implied) == stats["c"]["neg_implicit"] == 6
    assert all(value == 0.1 for value in implied)
    for sample in planned:
        if sample.path[0] in "pn":  # the images a person answered for "c"
            assert type(sample.binary["c"]) is float, sample.path


def test_the_mark_is_the_type_never_the_value():
    """An implied target of 0.0 looks like a human no; it is still implied."""
    planned, _stats = bb._plan_epoch_samples(
        _pool(),
        _vocab(("c", "b")),
        epoch=0,
        label_policy={**_SOFT, "implicit_negative_value": 0.0},
    )

    values = [sample.binary["c"] for sample in planned if "c" in sample.binary]
    assert sum(value == 0.0 for value in values) == 10  # 4 human, 6 implied
    assert sum(isinstance(value, bb._ImpliedNegative) for value in values) == 6


def test_the_mark_survives_the_trip_to_the_loss():
    """Loader workers pickle every batch, and the pinning thread rebuilds its
    lists and dicts; the labels must come out of both as they went in."""
    from torch.utils.data._utils.pin_memory import pin_memory

    _images, binary, _groups = bb._collate(
        [
            (torch.zeros(1), {"c": _implied(), "b": 0.0}, {}),
            (torch.zeros(1), {"c": 1.0}, {}),
        ]
    )

    for carried in (binary, pickle.loads(pickle.dumps(binary)), pin_memory(binary)):
        assert isinstance(carried[0]["c"], bb._ImpliedNegative)
        assert carried[0]["c"] == 0.1
        assert type(carried[0]["b"]) is float and type(carried[1]["c"]) is float


# -- the plan: common concepts take no implied negatives ----------------------


def _two_concepts():
    """200 training images: "common" has a human yes on 40 of them, "rare" on
    4, and nobody said no to either."""
    samples = [bb._Sample(f"c{i}.png", {"common": 1.0}, {}) for i in range(40)]
    samples += [bb._Sample(f"r{i}.png", {"rare": 1.0}, {}) for i in range(4)]
    samples += [
        bb._Sample(f"o{i}.png", {"other": float(i % 2)}, {}) for i in range(156)
    ]
    return samples


_THREE = ("common", "rare", "other")


def test_prior_is_the_share_of_training_images_with_a_human_yes():
    vocab = _vocab(_THREE)

    _planned, stats = bb._plan_epoch_samples(
        _two_concepts(), vocab, epoch=0, positive_cap=10
    )

    # Of every image, not of this epoch's capped draw.
    assert stats["common"]["pos"] == 10
    assert stats["common"]["prior"] == pytest.approx(40 / 200)
    assert stats["rare"]["prior"] == pytest.approx(4 / 200)
    # Kept on the vocabulary, for validation to band its figures by.
    assert vocab.priors == pytest.approx({"common": 0.2, "rare": 0.02, "other": 0.39})


def test_a_common_concept_gets_no_implied_negatives():
    samples = _two_concepts()

    _planned, as_today = bb._plan_epoch_samples(
        samples, _vocab(_THREE), epoch=0, label_policy=_SOFT
    )
    planned, masked = bb._plan_epoch_samples(
        samples,
        _vocab(_THREE),
        epoch=0,
        label_policy={**_SOFT, "implicit_negative_max_prior": 0.05},
    )

    assert as_today["common"]["neg_implicit"] == 160
    assert as_today["rare"]["neg_implicit"] == 4
    assert masked["common"]["neg_implicit"] == 0  # prior 0.20
    assert masked["other"]["neg_implicit"] == 0  # prior 0.39
    assert masked["rare"]["neg_implicit"] == 4  # prior 0.02
    assert not any(
        isinstance(sample.binary.get("common"), bb._ImpliedNegative)
        for sample in planned
    )
    # Its human labels are all still there.
    assert masked["common"]["pos"] == as_today["common"]["pos"] == 40


def test_a_concept_exactly_at_the_threshold_keeps_its_implied_negatives():
    _planned, stats = bb._plan_epoch_samples(
        _two_concepts(),
        _vocab(_THREE),
        epoch=0,
        label_policy={**_SOFT, "implicit_negative_max_prior": 0.2},
    )
    assert stats["common"]["neg_implicit"] == 160


# -- the loss -----------------------------------------------------------------


def _heads_with_logits(logits: torch.Tensor):
    """Heads and features whose concept logits are exactly ``logits``
    ``[rows, concepts]``: row r is a one-hot feature picking column r of the
    head matrix."""
    rows, concepts = logits.shape
    heads, _vocab_unused = _heads(concepts=concepts, groups=())
    features = torch.zeros(rows, FEATURES)
    features[torch.arange(rows), torch.arange(rows)] = 1.0
    with torch.no_grad():
        heads.binary_weight.zero_()
        heads.binary_weight[:, :rows] = logits.T
        heads.binary_bias.zero_()
    return heads, features


def _name(column: int) -> str:
    return f"concept_{column:02d}"


def _expected(logits, labels, *, hill=False, dropped=(), reduction="sum"):
    """The loss written out pair by pair: BCE on every label, or the Hill term
    on an implied one; the mean over each head's own labels; ``dropped`` pairs
    left out as if nobody had labelled them."""
    bce = nn.functional.binary_cross_entropy_with_logits
    per_head: dict[str, list[torch.Tensor]] = {}
    for row, row_labels in enumerate(labels):
        for concept, target in row_labels.items():
            if (row, concept) in dropped:
                continue
            logit = logits[row, int(concept[-2:])]
            if hill and isinstance(target, bb._ImpliedNegative):
                prob = torch.sigmoid(logit)
                value = (1.5 - prob) * prob * prob
            else:
                value = bce(logit, torch.tensor(float(target)))
            per_head.setdefault(concept, []).append(value)
    means = torch.stack([torch.stack(values).mean() for values in per_head.values()])
    return means.mean() if reduction == "mean" else means.sum()


def _mixed_batch():
    """Six rows over five heads: human yes and no, and implied negatives."""
    logits = torch.randn(6, 5, generator=torch.Generator().manual_seed(11)) * 2
    labels = [
        {_name(0): 1.0, _name(1): _implied(), _name(4): _implied()},
        {_name(0): 0.0, _name(2): _implied()},
        {_name(1): 1.0, _name(2): 1.0, _name(3): _implied()},
        {_name(0): _implied(), _name(3): 0.0},
        {_name(1): _implied(), _name(2): 0.0},
        {},
    ]
    return logits, labels


def _loss(logits, labels, **kwargs):
    heads, features = _heads_with_logits(logits)
    return bb._batch_loss(features, heads, labels, [{}] * len(labels), CPU, **kwargs)


def test_with_no_option_set_the_loss_is_what_it_was():
    logits, labels = _mixed_batch()
    heads, features = _heads_with_logits(logits)
    groups = [{}] * 6

    plain = bb._batch_loss(features, heads, labels, groups, CPU)

    spelled_out = {**_SOFT, "implicit_negative_loss": "bce", "reject_rate": 0.0}
    for policy, epoch in ((None, None), (_SOFT, 7), (spelled_out, 7)):
        again = bb._batch_loss(
            features, heads, labels, groups, CPU, label_policy=policy, epoch=epoch
        )
        assert torch.equal(again, plain)
    torch.testing.assert_close(
        plain, _reference_batch_loss(features, heads, labels, groups, CPU)
    )
    torch.testing.assert_close(plain, _expected(logits, labels))


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_the_hill_term_replaces_the_loss_on_implied_negatives_only(reduction):
    logits, labels = _mixed_batch()

    hill = _loss(logits, labels, label_policy=_HILL, reduction=reduction)

    torch.testing.assert_close(
        hill, _expected(logits, labels, hill=True, reduction=reduction)
    )
    assert not torch.isclose(hill, _loss(logits, labels, reduction=reduction))


def test_the_hill_term_leaves_a_batch_of_human_labels_alone():
    logits, labels = _mixed_batch()
    human = [
        {
            concept: float(value)
            for concept, value in row.items()
            if not isinstance(value, bb._ImpliedNegative)
        }
        for row in labels
    ]

    assert torch.equal(
        _loss(logits, human, label_policy=_HILL),
        _loss(logits, human, label_policy=None),
    )


def _push(logit: float, policy: dict) -> tuple[float, float]:
    """The loss on one implied negative scored ``logit``, and how hard it
    pushes that logit down."""
    heads, features = _heads_with_logits(torch.tensor([[logit]]))
    loss = bb._batch_loss(
        features, heads, [{_name(0): _implied()}], [{}], CPU, label_policy=policy
    )
    (grad,) = torch.autograd.grad(loss, heads.binary_bias)
    return float(loss.detach()), float(grad[0])


def test_the_hill_term_lets_go_once_the_model_is_sure_of_a_yes():
    """Towards a 0.1 target, BCE pushes an implied negative harder the higher
    the model scores it. The Hill term pushes hardest near 0.5 and lets go as
    the score nears 1; its value levels off at 0.5."""
    unsure_loss, unsure_push = _push(0.0, _HILL)
    sure_loss, sure_push = _push(7.0, _HILL)

    assert unsure_loss == pytest.approx((1.5 - 0.5) * 0.25)
    assert unsure_push == pytest.approx(3 * 0.25 * 0.25)  # 3 p^2 (1 - p)^2
    assert unsure_loss < sure_loss < 0.5
    assert sure_push < 1e-4

    _bce_loss, bce_push = _push(7.0, _SOFT)
    assert bce_push == pytest.approx(float(torch.sigmoid(torch.tensor(7.0))) - 0.1)


@pytest.mark.parametrize("logit", [-1e4, -50.0, 50.0, 1e4, float("inf")])
def test_the_hill_term_is_finite_for_extreme_logits(logit):
    heads, features = _heads_with_logits(torch.tensor([[logit]]))

    loss = bb._batch_loss(
        features, heads, [{_name(0): _implied()}], [{}], CPU, label_policy=_HILL
    )
    weight_grad, bias_grad = torch.autograd.grad(
        loss, [heads.binary_weight, heads.binary_bias]
    )

    assert torch.isfinite(loss) and 0.0 <= float(loss.detach()) <= 0.5
    assert torch.isfinite(weight_grad).all() and torch.isfinite(bias_grad).all()


def test_the_new_terms_stay_in_float32_under_autocast():
    """Training calls the loss inside autocast; the head product comes back in
    half precision and the loss is worked out from float32 logits."""
    logits, labels = _mixed_batch()
    logits[5] = torch.tensor([900.0, -900.0, 30.0, -30.0, 0.0])
    labels[5] = {_name(column): _implied() for column in range(5)}
    policy = {**_HILL, "reject_rate": 0.25, "reject_from_epoch": 1}

    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = _loss(logits, labels, label_policy=policy, epoch=0, reduction="mean")

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_rejection_starts_in_the_second_epoch_and_grows():
    policy = {"reject_rate": 0.05}

    # Epochs count from 0 here, as everywhere in the trainer.
    shares = [bb._reject_share(policy, epoch) for epoch in range(4)]

    assert shares == pytest.approx([0.0, 0.05, 0.10, 0.15])
    assert bb._reject_share({**policy, "reject_from_epoch": 1}, 0) == pytest.approx(
        0.05
    )
    assert bb._reject_share({**policy, "reject_from_epoch": 4}, 2) == 0.0
    assert bb._reject_share({"reject_rate": 0.6}, 5) == 1.0  # never more than all
    assert bb._reject_share({}, 9) == 0.0  # off unless a rate is set
    assert bb._reject_share(policy, None) == 0.0  # a caller that gives no epoch


def _ten_implied():
    """Row 0: ten implied negatives scored -5 to 4. Row 1: a human no scored
    above every one of them on three of the heads."""
    logits = torch.stack([torch.arange(10.0) - 5.0, torch.full((10,), 6.0)])
    labels = [
        {_name(column): _implied() for column in range(10)},
        {_name(0): 0.0, _name(8): 0.0, _name(9): 0.0},
    ]
    return logits, labels


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_rejection_drops_the_highest_scoring_implied_negatives(reduction):
    logits, labels = _ten_implied()
    policy = {**_SOFT, "reject_rate": 0.1}

    # Third epoch: a share of 0.2, so two of the ten. Head 7 loses its only
    # label and takes no part; heads 8 and 9 keep their human no.
    two = _loss(logits, labels, label_policy=policy, epoch=2, reduction=reduction)
    three = _loss(logits, labels, label_policy=policy, epoch=3, reduction=reduction)

    torch.testing.assert_close(
        two,
        _expected(
            logits, labels, dropped={(0, _name(8)), (0, _name(9))}, reduction=reduction
        ),
    )
    torch.testing.assert_close(
        three,
        _expected(
            logits,
            labels,
            dropped={(0, _name(7)), (0, _name(8)), (0, _name(9))},
            reduction=reduction,
        ),
    )


def test_nothing_is_rejected_in_the_first_epoch_or_without_an_epoch():
    logits, labels = _ten_implied()
    policy = {**_SOFT, "reject_rate": 0.1}
    plain = _loss(logits, labels)

    assert torch.equal(_loss(logits, labels, label_policy=policy, epoch=0), plain)
    assert torch.equal(_loss(logits, labels, label_policy=policy), plain)
    from_the_start = {**policy, "reject_from_epoch": 1}
    assert not torch.isclose(
        _loss(logits, labels, label_policy=from_the_start, epoch=0), plain
    )


def test_rejection_never_drops_a_human_label():
    """Every implied negative rejected: what is left is the human labels, the
    highest-scoring pairs of the batch among them."""
    logits, labels = _ten_implied()
    everything = {**_SOFT, "reject_rate": 1.0}

    loss = _loss(logits, labels, label_policy=everything, epoch=1)

    human_only = [{}, labels[1]]
    torch.testing.assert_close(loss, _expected(logits, human_only))
    # And with only human labels in the batch there is nothing to reject.
    assert torch.equal(
        _loss(logits, human_only, label_policy=everything, epoch=1),
        _loss(logits, human_only),
    )


def test_a_batch_whose_labels_were_all_rejected_is_a_batch_with_none():
    logits, labels = _ten_implied()

    loss = _loss(
        logits, [labels[0], {}], label_policy={**_SOFT, "reject_rate": 1.0}, epoch=1
    )

    assert loss is None


def test_rejection_ranks_by_score_not_by_the_soft_target_loss():
    """Towards a 0.1 target, BCE is larger for an implied negative the model is
    certain of (3.0 at logit -30) than for one it scores 0.73 (1.2 at logit 1).
    Rejecting by that loss would drop the surest negatives."""
    logits = torch.tensor([[-30.0, 1.0]])
    labels = [{_name(0): _implied(), _name(1): _implied()}]

    loss = _loss(logits, labels, label_policy={**_SOFT, "reject_rate": 0.5}, epoch=1)

    torch.testing.assert_close(loss, _expected(logits, labels, dropped={(0, _name(1))}))


def test_a_batch_too_small_for_its_share_rejects_nothing():
    """The share is rounded to whole pairs: 0.1 of four is none, 0.2 is one."""
    logits = torch.tensor([[-2.0, -1.0, 1.0, 2.0]])
    labels = [{_name(column): _implied() for column in range(4)}]
    policy = {**_SOFT, "reject_rate": 0.1}

    assert torch.equal(
        _loss(logits, labels, label_policy=policy, epoch=1), _loss(logits, labels)
    )
    torch.testing.assert_close(
        _loss(logits, labels, label_policy=policy, epoch=2),
        _expected(logits, labels, dropped={(0, _name(3))}),
    )


def test_rejection_and_the_hill_term_work_together():
    logits, labels = _ten_implied()
    policy = {**_HILL, "reject_rate": 0.1}

    loss = _loss(logits, labels, label_policy=policy, epoch=2)

    torch.testing.assert_close(
        loss,
        _expected(logits, labels, hill=True, dropped={(0, _name(8)), (0, _name(9))}),
    )


# -- recall at fixed precision ------------------------------------------------


def _ranked(pattern: str):
    """A head's validation scores, best first: ``+`` a human yes, ``-`` a no."""
    targets = np.array([1.0 if mark == "+" else 0.0 for mark in pattern])
    return np.linspace(4.0, -4.0, len(pattern)), targets


_CLEAN = "+" * 10 + "-" * 10
_ONE_SLIP = "+" * 9 + "-" + "+" + "-" * 9
_TOP_IS_WRONG = "-" + "+" * 10 + "-" * 9
_BACKWARDS = "-" * 10 + "+" * 10


def test_recall_at_fixed_precision_on_hand_made_rankings():
    metrics = bb._backbone_metrics(
        {
            "clean": _ranked(_CLEAN),
            "one_slip": _ranked(_ONE_SLIP),
            "top_is_wrong": _ranked(_TOP_IS_WRONG),
            "backwards": _ranked(_BACKWARDS),
        },
        {},
    )

    assert metrics["binary_r_at_p90/clean"] == 1.0
    assert metrics["binary_r_at_p95/clean"] == 1.0
    # Nine clean yes, then a no: all ten yes are found at precision 10/11.
    assert metrics["binary_r_at_p90/one_slip"] == 1.0
    assert metrics["binary_r_at_p95/one_slip"] == 0.9
    # A no on top: precision is at best 10/11, never 0.95.
    assert metrics["binary_r_at_p90/top_is_wrong"] == 1.0
    assert metrics["binary_r_at_p95/top_is_wrong"] == 0.0
    assert metrics["binary_r_at_p90/backwards"] == 0.0
    assert metrics["binary_macro_r_at_p90"] == pytest.approx(0.75)
    assert metrics["binary_macro_r_at_p95"] == pytest.approx((1.0 + 0.9) / 4)
    assert metrics["binary_r_at_p_heads"] == 4


def test_recall_at_fixed_precision_matches_the_precision_recall_curve():
    from sklearn.metrics import precision_recall_curve

    rng = np.random.default_rng(5)
    scores = {}
    for index in range(30):
        size = int(rng.integers(30, 400))
        targets = (rng.random(size) < rng.uniform(0.2, 0.8)).astype(np.float64)
        targets[:10], targets[10:20] = 1.0, 0.0
        logits = rng.normal(0, 2, size) + rng.uniform(1.0, 5.0) * targets
        if index % 3 == 0:
            logits = np.round(logits)  # many images share a score
        scores[f"head_{index}"] = (logits, targets)

    metrics = bb._backbone_metrics(scores, {})

    for tag, target in ((90, 0.90), (95, 0.95)):
        expected = []
        for concept, (logits, targets) in scores.items():
            probs = 1.0 / (1.0 + np.exp(-logits))
            precision, recall, _ = precision_recall_curve(targets.astype(int), probs)
            expected.append(float(recall[precision >= target].max()))
            assert metrics[f"binary_r_at_p{tag}/{concept}"] == pytest.approx(
                expected[-1], abs=1e-12
            ), concept
        assert metrics[f"binary_macro_r_at_p{tag}"] == pytest.approx(np.mean(expected))


def test_a_head_with_too_few_labels_of_either_kind_gets_no_figure():
    metrics = bb._backbone_metrics(
        {
            "enough": _ranked(_CLEAN),
            "nine_yes": _ranked("+" * 9 + "-" * 30),
            "nine_no": _ranked("+" * 30 + "-" * 9),
        },
        {},
    )

    assert "binary_r_at_p90/enough" in metrics
    for concept in ("nine_yes", "nine_no"):
        assert f"binary_r_at_p90/{concept}" not in metrics
        assert f"binary_r_at_p95/{concept}" not in metrics
        assert f"binary_ap/{concept}" in metrics  # the older figures are unchanged
    assert metrics["binary_r_at_p_heads"] == 1
    assert metrics["binary_macro_r_at_p90"] == 1.0

    none = bb._backbone_metrics({"nine_yes": _ranked("+" * 9 + "-" * 30)}, {})
    assert not any("r_at_p" in key for key in none)


def test_recall_at_fixed_precision_is_averaged_by_how_common_a_concept_is():
    scores = {
        "rare_a": _ranked(_CLEAN),  # 1.0 and 1.0
        "rare_b": _ranked(_BACKWARDS),  # 0.0 and 0.0
        "medium": _ranked(_ONE_SLIP),  # 1.0 and 0.9
        "common": _ranked(_TOP_IS_WRONG),  # 1.0 and 0.0
        "unknown": _ranked(_CLEAN),  # no prior: in the overall figure only
    }
    priors = {"rare_a": 0.001, "rare_b": 0.0099, "medium": 0.01, "common": 0.05}

    metrics = bb._backbone_metrics(scores, {}, priors=priors)

    assert metrics["binary_r_at_p_heads"] == 5
    assert metrics["binary_r_at_p_heads_rare"] == 2
    assert metrics["binary_r_at_p_heads_medium"] == 1
    assert metrics["binary_r_at_p_heads_common"] == 1
    assert metrics["binary_macro_r_at_p90_rare"] == pytest.approx(0.5)
    assert metrics["binary_macro_r_at_p95_rare"] == pytest.approx(0.5)
    assert metrics["binary_macro_r_at_p95_medium"] == pytest.approx(0.9)
    assert metrics["binary_macro_r_at_p90_common"] == pytest.approx(1.0)
    assert metrics["binary_macro_r_at_p95_common"] == pytest.approx(0.0)
    assert metrics["binary_macro_r_at_p90"] == pytest.approx(4 / 5)

    # Without priors there are no bands, only the overall figures.
    plain = bb._backbone_metrics(scores, {})
    assert plain["binary_macro_r_at_p90"] == pytest.approx(4 / 5)
    assert not any(key.endswith(("_rare", "_medium", "_common")) for key in plain)


def test_validation_bands_by_the_priors_the_planner_recorded():
    heads, vocab = _heads(concepts=2, groups=())
    logits = torch.stack(
        [torch.linspace(4.0, -4.0, 20), torch.linspace(-4.0, 4.0, 20)], dim=1
    )
    with torch.no_grad():
        heads.binary_weight.zero_()
        heads.binary_weight[:, :20] = logits.T
        heads.binary_bias.zero_()
    labels = [
        {"concept_00": float(row < 10), "concept_01": float(row < 10)}
        for row in range(20)
    ]
    loader = [(torch.eye(20, FEATURES), labels, [{}] * 20)]

    vocab.priors = {"concept_00": 0.002, "concept_01": 0.3}
    metrics = bb._evaluate(nn.Identity(), heads, loader, vocab, CPU)

    assert metrics["binary_macro_r_at_p95_rare"] == 1.0
    assert metrics["binary_macro_r_at_p95_common"] == 0.0
    assert metrics["binary_macro_r_at_p95"] == 0.5

    # A vocabulary no plan has touched: the overall figures, no bands.
    bare = SimpleNamespace(concepts=vocab.concepts, groups={})
    assert "binary_macro_r_at_p95_rare" not in bb._evaluate(
        nn.Identity(), heads, loader, bare, CPU
    )


# -- the two training loops ---------------------------------------------------


def _spy_on_the_loss(monkeypatch) -> dict:
    seen: dict = {}
    real = bb._batch_loss

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(bb, "_batch_loss", _spy)
    return seen


def _one_batch():
    return _features(2, seed=1), [{"concept_00": 1.0}, {"concept_00": 0.0}], [{}, {}]


def test_the_training_loop_gives_the_loss_its_policy_and_epoch(monkeypatch):
    from bittrainer.generic.tasks.backbone_task import BackboneTask

    policy = {**_SOFT, "reject_rate": 0.01}
    task = BackboneTask(
        {
            "run_id": "run_policy",
            "convnextv2_size": "atto",
            "candidate_checkpoint_path": "unused/candidate.safetensors",
            "records": [],
            "training_config": {
                "device": "cpu",
                "use_amp": False,
                "clip_grad_norm": 0,
                "label_policy": policy,
            },
        }
    )
    heads, vocab = _heads(concepts=2, groups=())
    task.vocab = vocab
    task.train_samples = [bb._Sample("a.png", {"concept_00": 1.0}, {})]
    task._plan_epoch(3)  # what build_loaders does before each epoch
    seen = _spy_on_the_loss(monkeypatch)
    model = bb._BackboneWithHeads(nn.Identity(), heads)

    task.train_epoch(
        SimpleNamespace(device=CPU),
        model,
        torch.optim.SGD(model.parameters(), lr=0.0),
        [_one_batch()],
        step_callback=None,
        boundary_hook=lambda _batches: None,
        start_batch=0,
    )

    assert seen["label_policy"] == policy
    assert seen["epoch"] == 3


def test_the_distillation_loop_gives_the_loss_its_policy_and_epoch(monkeypatch):
    from bittrainer.generic.tasks.backbone_distill_task import BackboneDistillTask

    task = SimpleNamespace(
        supervised_weight=1.0,
        logit_weight=0.0,
        feature_weight=0.0,
        _pos_weight=None,
        group_label_smoothing=0.0,
        loss_reduction="mean",
        label_policy=_HILL,
        _planned_epoch=4,
    )
    heads, _vocab_unused = _heads(concepts=2, groups=())
    seen = _spy_on_the_loss(monkeypatch)
    features, binary, groups = _one_batch()

    loss, terms = BackboneDistillTask._step_loss(
        task, SimpleNamespace(heads=heads), features, None, None, binary, groups, CPU
    )

    assert loss is not None and set(terms) == {"supervised"}
    assert seen["label_policy"] == _HILL
    assert seen["epoch"] == 4


# -- the pack runner ----------------------------------------------------------


def test_runner_options_map_onto_the_label_policy(tmp_path):
    import bittrainer.tools.backbone_pack_train as runner
    from bittrainer.tests.test_backbone_pack_train import _make_pack, _options

    pack = _make_pack(tmp_path)

    def policy(**options):
        request, _counts = runner.build_request(
            pack, tmp_path / "run", **_options(**options)
        )
        return request["training_config"].get("label_policy")

    assert policy() is None  # nothing asked for: the trainer's own default
    assert policy(soft_negatives=True) == _SOFT
    # Each option is about implied negatives, so each one turns them on.
    assert policy(hill_negatives=True) == _HILL
    assert policy(reject_rate=0.01) == {**_SOFT, "reject_rate": 0.01}
    assert policy(mask_common=0.05) == {**_SOFT, "implicit_negative_max_prior": 0.05}
    assert policy(hill_negatives=True, reject_rate=0.02, mask_common=0.5) == {
        **_HILL,
        "reject_rate": 0.02,
        "implicit_negative_max_prior": 0.5,
    }


def test_runner_command_line_carries_the_three_options(monkeypatch, tmp_path):
    import bittrainer.tools.backbone_pack_train as runner

    seen: dict = {}

    def _fake(pack, out, **kw):
        seen.update(kw)
        return {"status": "finished"}

    monkeypatch.setattr(runner, "train_from_pack", _fake)
    base = ["--pack", str(tmp_path), "--out", str(tmp_path / "run")]

    assert runner.main(base) == 0
    assert seen["hill_negatives"] is False
    assert seen["reject_rate"] == 0.0
    assert seen["mask_common"] is None

    flags = ["--hill-negatives", "--reject-rate", "0.01", "--mask-common", "0.05"]
    assert runner.main([*base, *flags]) == 0
    assert seen["hill_negatives"] is True
    assert seen["reject_rate"] == 0.01
    assert seen["mask_common"] == 0.05
