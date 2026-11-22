"""Scaled implied negatives for the concept heads (Bitcrush ISSUE-1753).

Most concept heads have a few dozen human yes and no human no at all, so their
negative side is whatever unlabelled images the planner draws. Under the plain
``soft_implicit_negative`` policy that is at most the neg:pos cap, and the cap
tightens to 1:1 for a tiny head: a 10-yes "Hallway" sees about 10 "not a
hallway" images an epoch and fires on most indoor scenes.

``scaled_implied_negatives`` (off by default) draws many more of them, about
20 per yes, fresh every epoch, and weighs each one down so that together they
carry a bounded share of the loss: ``implied_mass`` times the positive mass,
divided by ``1 + human no / implied_mass_real_no_scale``. The more a person has
said no to a concept, the less the implied side counts.

Small synthetic samples and generated logits only: no images, no training run.
"""

from __future__ import annotations

import hashlib
import json
import pickle

import pytest
import torch
import torch.nn as nn

import bittrainer.backbone_trainer as bb
from bittrainer.tests.test_backbone_head_vectorised import CPU
from bittrainer.tests.test_backbone_label_policy import _heads_with_logits, _name
from bittrainer.tests.test_backbone_sampling import _vocab

_SOFT = {"mode": "soft_implicit_negative", "implicit_negative_value": 0.1}
_SCALED = {**_SOFT, "scaled_implied_negatives": True}
_CONCEPTS = ("hall", "common", "other")


def _pool(real_no: int = 0) -> list[bb._Sample]:
    """ "hall": 10 human yes and ``real_no`` human no. 400 images nobody
    labelled for it (they carry "other" and a group), 5 more holding a
    validation answer back."""
    samples = [bb._Sample(f"h{i}.png", {"hall": 1.0}, {}) for i in range(10)]
    samples += [bb._Sample(f"n{i}.png", {"hall": 0.0}, {}) for i in range(real_no)]
    samples += [
        bb._Sample(f"o{i}.png", {"other": float(i % 2)}, {"g": i % 3})
        for i in range(400)
    ]
    samples += [
        bb._Sample(f"x{i}.png", {"other": 1.0}, {}, held_binary={"hall": 1.0})
        for i in range(5)
    ]
    return samples


def _hall_weights(planned) -> list[float]:
    return [
        sample.binary["hall"].weight
        for sample in planned
        if isinstance(sample.binary.get("hall"), bb._ImpliedNegative)
    ]


# -- the plan -----------------------------------------------------------------


def test_a_head_with_ten_yes_and_no_no_gets_many_implied_negatives_of_bounded_mass():
    _planned, today = bb._plan_epoch_samples(
        _pool(), _vocab(_CONCEPTS), epoch=0, label_policy=_SOFT
    )
    planned, stats = bb._plan_epoch_samples(
        _pool(), _vocab(_CONCEPTS), epoch=0, label_policy=_SCALED
    )

    assert today["hall"]["neg_implicit"] == 10  # the 1:1 cap of a tiny head
    hall = stats["hall"]
    assert hall["neg_implicit"] == 200  # 20 per yes, the cap does not apply
    assert hall["neg_explicit"] == 0
    weights = _hall_weights(planned)
    assert len(weights) == 200
    # The positive mass is every yes the epoch trains, replicas included:
    # 10 yes oversampled 3x. With no human no, the implied side carries
    # implied_mass (1.0) times that, spread over all 200.
    positive_mass = hall["pos"] * hall["oversample_factor"]
    assert positive_mass == 30
    assert sum(weights) == pytest.approx(30.0)
    assert hall["implied_mass"] == pytest.approx(30.0)
    assert hall["implied_weight"] == pytest.approx(0.15)
    assert all(weight == pytest.approx(0.15) for weight in weights)
    assert hall["implied_mass_share"] == pytest.approx(0.5)
    # The target is still the soft 0.1, and the label still reads as implied.
    implied = [s.binary["hall"] for s in planned if "hall" in s.binary]
    assert sum(isinstance(v, bb._ImpliedNegative) for v in implied) == 200
    assert all(v == 0.1 for v in implied if isinstance(v, bb._ImpliedNegative))


def test_the_implied_mass_fades_as_human_no_answers_grow():
    masses = []
    for real_no in (0, 10, 20, 100):
        planned, stats = bb._plan_epoch_samples(
            _pool(real_no), _vocab(_CONCEPTS), epoch=0, label_policy=_SCALED
        )
        hall = stats["hall"]
        # The human no answers keep their own full-weight, capped draw.
        assert hall["neg_explicit"] == min(real_no, 10)
        assert hall["neg_implicit"] == 200
        assert sum(_hall_weights(planned)) == pytest.approx(hall["implied_mass"])
        masses.append(hall["implied_mass"])

    # 30 / (1 + no / 20)
    assert masses == pytest.approx([30.0, 20.0, 15.0, 5.0])


def test_the_mass_constants_come_from_the_policy():
    policy = {
        **_SCALED,
        "implied_per_positive": 5,
        "implied_mass": 0.5,
        "implied_mass_real_no_scale": 10,
    }
    _planned, stats = bb._plan_epoch_samples(
        _pool(10), _vocab(_CONCEPTS), epoch=0, label_policy=policy
    )
    assert stats["hall"]["neg_implicit"] == 50
    assert stats["hall"]["implied_mass"] == pytest.approx(0.5 * 30 / 2)


def test_an_implied_negative_never_outweighs_a_human_label():
    """Too few unlabelled images to spread the mass over: each one stops at 1."""
    samples = [bb._Sample(f"h{i}.png", {"hall": 1.0}, {}) for i in range(40)]
    samples += [bb._Sample(f"o{i}.png", {"other": 1.0}, {}) for i in range(8)]
    _planned, stats = bb._plan_epoch_samples(
        samples, _vocab(("hall", "other")), epoch=0, label_policy=_SCALED
    )
    assert stats["hall"]["neg_implicit"] == 8
    assert stats["hall"]["implied_weight"] == 1.0
    assert stats["hall"]["implied_mass"] == pytest.approx(8.0)


def test_a_common_concept_still_gets_no_implied_negatives():
    samples = _pool()
    samples += [bb._Sample(f"c{i}.png", {"common": 1.0}, {}) for i in range(60)]
    planned, stats = bb._plan_epoch_samples(
        samples,
        _vocab(_CONCEPTS),
        epoch=0,
        label_policy={**_SCALED, "implicit_negative_max_prior": 0.05},
    )
    assert stats["common"]["neg_implicit"] == 0  # prior 60 / 475
    assert stats["common"]["implied_mass"] == 0.0
    assert not any(
        isinstance(sample.binary.get("common"), bb._ImpliedNegative)
        for sample in planned
    )
    assert stats["hall"]["neg_implicit"] == 200  # prior 10 / 470


def test_each_epoch_draws_afresh_and_never_an_image_holding_an_answer_back():
    def drawn(epoch):
        planned, _stats = bb._plan_epoch_samples(
            _pool(), _vocab(_CONCEPTS), epoch=epoch, seed=3, label_policy=_SCALED
        )
        return {
            sample.path
            for sample in planned
            if isinstance(sample.binary.get("hall"), bb._ImpliedNegative)
        }

    first = drawn(0)
    assert first == drawn(0)  # a pure function of (seed, epoch)
    assert first != drawn(1)
    assert not any(path.startswith("x") for path in first | drawn(1))


def _fingerprint_pool() -> list[bb._Sample]:
    samples = [bb._Sample(f"h{i}.png", {"hall": 1.0}, {}) for i in range(10)]
    samples += [bb._Sample(f"n{i}.png", {"hall": 0.0}, {}) for i in range(3)]
    samples += [bb._Sample(f"c{i}.png", {"common": 1.0}, {}) for i in range(60)]
    samples += [
        bb._Sample(f"o{i}.png", {"other": float(i % 2)}, {"g": i % 3})
        for i in range(400)
    ]
    samples += [
        bb._Sample(f"x{i}.png", {"other": 1.0}, {}, held_binary={"hall": 1.0})
        for i in range(5)
    ]
    return samples


def _fingerprint(policy) -> str:
    """A digest of two epochs' plans and stats over a fixed pool."""
    out = []
    for epoch in (0, 1):
        planned, stats = bb._plan_epoch_samples(
            _fingerprint_pool(),
            _vocab(_CONCEPTS),
            epoch,
            seed=7,
            label_policy=policy,
        )
        rows = [
            (
                s.path,
                sorted(
                    (k, type(v).__name__, repr(float(v))) for k, v in s.binary.items()
                ),
                sorted(s.groups.items()),
            )
            for s in planned
        ]
        out.append(repr(rows))
        out.append(json.dumps(stats, sort_keys=True))
    return hashlib.sha256("\n".join(out).encode()).hexdigest()


# Digests of the plan the trainer made before ISSUE-1753 (BitTrainer f3631fe).
_BEFORE = {
    "none": "c5c3a48759da58e328820d7c5ef082e64877ad084bb308ef53b58894567bb26b",
    "soft": "6752971f5c24ca06f36376968eaf3fade84a82f6909f08171a2dd0a9bd36cc55",
    "soft_mask": "2d94b11b3901665ad90368e7bdea4f00f2d706188075d180f8f676153ef00e31",
}


@pytest.mark.parametrize(
    ("name", "policy"),
    [
        ("none", None),
        ("soft", _SOFT),
        ("soft_mask", {**_SOFT, "implicit_negative_max_prior": 0.05}),
        ("soft", {**_SOFT, "scaled_implied_negatives": False}),
        # Without soft negatives there is nothing to scale.
        ("none", {"scaled_implied_negatives": True}),
    ],
)
def test_with_the_option_off_the_plan_is_byte_for_byte_what_it_was(name, policy):
    assert _fingerprint(policy) == _BEFORE[name]


# -- the label on its way to the loss ------------------------------------------


def test_the_weight_survives_the_trip_to_the_loss():
    from torch.utils.data._utils.pin_memory import pin_memory

    value = bb._WeightedImpliedNegative(0.1, 0.15)
    _images, binary, _groups = bb._collate([(torch.zeros(1), {"hall": value}, {})])
    for carried in (binary, pickle.loads(pickle.dumps(binary)), pin_memory(binary)):
        label = carried[0]["hall"]
        assert isinstance(label, bb._ImpliedNegative)
        assert label == 0.1
        assert label.weight == pytest.approx(0.15)


def test_pos_weight_counts_the_implied_side_by_its_mass():
    stats = {
        "scaled": {
            "pos": 10,
            "oversample_factor": 3,
            "neg_explicit": 0,
            "neg_implicit": 200,
            "implied_mass": 30.0,
        },
        "plain": {
            "pos": 10,
            "oversample_factor": 1,
            "neg_explicit": 0,
            "neg_implicit": 40,
        },
    }
    weights = bb._head_pos_weights(stats)
    assert weights["scaled"] == pytest.approx(1.0)
    assert weights["plain"] == pytest.approx(4.0)


# -- the loss -----------------------------------------------------------------


def _w(weight: float) -> float:
    return bb._WeightedImpliedNegative(0.1, weight)


def _bce(logit, target) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(float(logit)), torch.tensor(float(target))
    )


def _weighted_batch():
    logits = torch.randn(4, 3, generator=torch.Generator().manual_seed(5)) * 2
    labels = [
        {_name(0): 1.0, _name(1): _w(0.1)},
        {_name(0): _w(0.25), _name(1): _w(0.1)},
        {_name(0): _w(0.25), _name(2): 0.0},
        {_name(0): 0.0},
    ]
    return logits, labels


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_a_weighted_implied_negative_counts_by_its_weight(reduction):
    logits, labels = _weighted_batch()
    heads, features = _heads_with_logits(logits)

    loss = bb._batch_loss(
        features,
        heads,
        labels,
        [{}] * 4,
        CPU,
        reduction=reduction,
        label_policy=_SCALED,
    )

    lg = logits
    # Head 0: a yes, a no and two implied at 0.25: a weighted mean over 2.5.
    head0 = (
        _bce(lg[0, 0], 1)
        + 0.25 * _bce(lg[1, 0], 0.1)
        + 0.25 * _bce(lg[2, 0], 0.1)
        + _bce(lg[3, 0], 0)
    ) / 2.5
    # Head 1: two implied at 0.1 and nothing else: their weighted sum, which
    # counts as a fifth of a head.
    head1 = 0.1 * (_bce(lg[0, 1], 0.1) + _bce(lg[1, 1], 0.1))
    head2 = _bce(lg[2, 2], 0)
    total = head0 + head1 + head2
    expected = total if reduction == "sum" else total / (1 + 0.2 + 1)
    torch.testing.assert_close(loss, expected)


def test_rejection_still_drops_the_highest_scoring_weighted_implied_negatives():
    logits = torch.tensor([[0.0], [3.0], [-3.0]])
    labels = [{_name(0): 1.0}, {_name(0): _w(0.5)}, {_name(0): _w(0.5)}]
    heads, features = _heads_with_logits(logits)

    loss = bb._batch_loss(
        features,
        heads,
        labels,
        [{}] * 3,
        CPU,
        label_policy={**_SCALED, "reject_rate": 0.5},
        epoch=1,
    )

    expected = (_bce(0.0, 1) + 0.5 * _bce(-3.0, 0.1)) / 1.5
    torch.testing.assert_close(loss, expected)


# -- the pack runner ----------------------------------------------------------


def test_runner_option_and_flag_turn_on_scaled_implied_negatives(monkeypatch, tmp_path):
    import bittrainer.tools.backbone_pack_train as runner
    from bittrainer.tests.test_backbone_pack_train import _make_pack, _options

    pack = _make_pack(tmp_path)

    def policy(**options):
        request, _counts = runner.build_request(
            pack, tmp_path / "run", **_options(**options)
        )
        return request["training_config"].get("label_policy")

    assert policy() is None
    assert policy(scaled_implied_negatives=True) == _SCALED
    assert policy(scaled_implied_negatives=True, mask_common=0.05) == {
        **_SCALED,
        "implicit_negative_max_prior": 0.05,
    }

    seen: dict = {}

    def _fake(pack, out, **kw):
        seen.update(kw)
        return {"status": "finished"}

    monkeypatch.setattr(runner, "train_from_pack", _fake)
    base = ["--pack", str(tmp_path), "--out", str(tmp_path / "run")]
    assert runner.main(base) == 0
    assert seen["scaled_implied_negatives"] is False
    assert runner.main([*base, "--scaled-implied-negatives"]) == 0
    assert seen["scaled_implied_negatives"] is True
