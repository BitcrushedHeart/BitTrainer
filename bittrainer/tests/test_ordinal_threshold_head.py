"""All-threshold ordinal head (Bitcrush ISSUE-1106, Phase B).

``OrdinalThresholdFC`` replaces ``head.fc`` like ``SpatialCellFC`` does. It has
K-1 outputs — R-1 thresholds ("is the step above k?") plus one ``__none__``
logit when the group has a none class — and emits K exact log-probabilities via
the thermometer construction, so softmax, temperature, priors, calibration and
every Engine consumer keep the one-probability-per-class contract.

CPU-only: random-init ConvNeXt atto and a tiny SigLIP trunk; no download.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import bittrainer.group_trainer as gt
from bittrainer.model import (
    backbone_feature_hash,
    create_model,
    head_tail_logits,
    load_checkpoint,
    pooled_features,
)
from bittrainer.ordinal_threshold import (
    OrdinalThresholdFC,
    cumulative_targets,
    has_threshold_head,
    install_ordinal_threshold_head,
    ordinal_threshold_loss,
    threshold_logits_from_scores,
)
from bittrainer.tests.siglip_fixtures import install_tiny_trunk


def _explicit_scores(
    raw: torch.Tensor, num_classes: int, none_index: int
) -> torch.Tensor:
    """The plan's formula, written out literally (slow, obviously right)."""
    has_none = 0 <= none_index < num_classes
    ranks = num_classes - int(has_none)
    z = raw[:, : ranks - 1].double()
    therm = []
    for j in range(ranks):
        on = F.logsigmoid(z[:, :j]).sum(1)
        off = F.logsigmoid(-z[:, j:]).sum(1)
        therm.append(on + off)
    therm = torch.stack(therm, 1)
    rank_lp = therm - torch.logsumexp(therm, 1, keepdim=True)
    if not has_none:
        return rank_lp
    u = raw[:, ranks - 1].double()
    out = torch.empty(raw.shape[0], num_classes, dtype=torch.float64)
    rank_idx = [i for i in range(num_classes) if i != none_index]
    out[:, rank_idx] = rank_lp + F.logsigmoid(-u)[:, None]
    out[:, none_index] = F.logsigmoid(u)
    return out


def _fc(num_classes=5, none_index=-1, in_features=8):
    torch.manual_seed(0)
    return OrdinalThresholdFC(in_features, num_classes, none_index)


@pytest.mark.parametrize("none_index", [-1, 0, 2, 4])
def test_scores_are_exact_log_probabilities(none_index):
    fc = _fc(5, none_index)
    raw = torch.randn(6, 4) * 3
    scores = fc.decode(raw)
    assert scores.shape == (6, 5)
    torch.testing.assert_close(
        torch.logsumexp(scores, 1), torch.zeros(6), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        scores.double(), _explicit_scores(raw, 5, none_index), atol=1e-5, rtol=1e-5
    )


def test_output_layout_is_thresholds_then_the_none_logit():
    fc = _fc(5, none_index=0)
    assert fc.threshold_fc.out_features == 4
    assert fc.num_ranks == 4
    raw = torch.zeros(1, 4)
    raw[0, 3] = 30.0  # the none logit
    assert fc.decode(raw).argmax(1).item() == 0
    plain = _fc(5, none_index=-1)
    assert plain.threshold_fc.out_features == 4 and plain.num_ranks == 5


def test_argmax_is_the_count_rule_for_monotone_thresholds():
    fc = _fc(6)
    torch.manual_seed(3)
    for _ in range(50):
        z = torch.sort(torch.randn(1, 5) * 4, descending=True).values
        assert fc.decode(z).argmax(1).item() == int((z > 0).sum())


def test_argmax_among_ranks_ignores_the_none_gate():
    fc = _fc(6, none_index=5)
    z = torch.tensor([[3.0, 1.0, -0.5, -2.0, -40.0]])  # none logit very negative
    assert fc.decode(z).argmax(1).item() == 2


@pytest.mark.parametrize("none_index", [-1, 0, 3])
def test_threshold_logits_round_trip_through_the_scores(none_index):
    fc = _fc(4, none_index)
    raw = torch.randn(5, 3) * 2
    torch.testing.assert_close(
        threshold_logits_from_scores(fc.decode(raw), none_index),
        raw,
        atol=1e-4,
        rtol=1e-4,
    )


def test_cumulative_targets_of_hard_labels():
    soft = F.one_hot(torch.tensor([0, 2, 4, 1]), 5).float()
    soft[3] = 0.0
    soft[3, 1] = 1.0
    targets, mass = cumulative_targets(soft, none_index=-1)
    assert targets.tolist() == [
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 1],
        [1, 0, 0, 0],
    ]
    assert mass.tolist() == [1, 1, 1, 1]


def test_cumulative_targets_with_a_none_class():
    # classes: __none__, r0, r1, r2
    soft = torch.tensor([[1.0, 0, 0, 0], [0, 0, 0.5, 0.5], [0.25, 0, 0.75, 0]])
    targets, mass = cumulative_targets(soft, none_index=0)
    torch.testing.assert_close(mass, torch.tensor([0.0, 1.0, 0.75]))
    torch.testing.assert_close(
        targets[:, -1], torch.tensor([1.0, 0.0, 0.25])
    )  # none target
    torch.testing.assert_close(targets[1, :2], torch.tensor([1.0, 0.5]))
    torch.testing.assert_close(targets[2, :2], torch.tensor([1.0, 0.0]))


def test_loss_is_mean_bce_against_cumulative_targets():
    torch.manual_seed(1)
    z = torch.randn(4, 4)
    soft = torch.softmax(torch.randn(4, 5), 1)
    targets, _ = cumulative_targets(soft, none_index=-1)
    want = F.binary_cross_entropy_with_logits(z, targets)
    torch.testing.assert_close(ordinal_threshold_loss(z, soft, none_index=-1), want)


def test_none_rows_only_train_the_none_logit():
    z = torch.zeros(2, 4, requires_grad=True)  # 3 thresholds + none logit
    soft = torch.tensor([[1.0, 0, 0, 0, 0], [0, 0, 1.0, 0, 0]])  # row 0 is __none__
    ordinal_threshold_loss(z, soft, none_index=0).backward()
    assert z.grad[0, :3].abs().sum() == 0
    assert z.grad[0, 3] != 0
    assert z.grad[1, :3].abs().sum() > 0 and z.grad[1, 3] != 0


def test_class_weights_weigh_rows_by_their_target():
    torch.manual_seed(2)
    z = torch.randn(3, 3)
    soft = F.one_hot(torch.tensor([0, 1, 3]), 4).float()
    weights = torch.tensor([1.0, 2.0, 1.0, 4.0])
    per_row = torch.stack(
        [
            ordinal_threshold_loss(z[i : i + 1], soft[i : i + 1], none_index=-1)
            for i in range(3)
        ]
    )
    want = (per_row * torch.tensor([1.0, 2.0, 4.0])).sum() / 7.0
    got = ordinal_threshold_loss(z, soft, none_index=-1, class_weights=weights)
    torch.testing.assert_close(got, want)


def test_stable_at_extreme_logits():
    fc = _fc(6, none_index=5)
    raw = torch.tensor(
        [[50.0, 50.0, -50.0, -50.0, 50.0], [-50.0] * 5], requires_grad=True
    )
    scores = fc.decode(raw)
    assert torch.isfinite(scores).all()
    torch.testing.assert_close(
        torch.logsumexp(scores, 1), torch.zeros(2), atol=1e-4, rtol=0
    )
    soft = F.one_hot(torch.tensor([2, 0]), 6).float()
    loss = ordinal_threshold_loss(
        threshold_logits_from_scores(scores, 5), soft, none_index=5
    )
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(raw.grad).all()


def test_refuses_fewer_than_two_classes():
    with pytest.raises(ValueError):
        OrdinalThresholdFC(8, 1)


def test_install_on_convnext_keeps_the_head_contract():
    torch.manual_seed(0)
    model = create_model(model_size="atto", pretrained=False, num_classes=5).eval()
    before = backbone_feature_hash(model)
    install_ordinal_threshold_head(model, 5, none_index=4)
    assert has_threshold_head(model)
    assert backbone_feature_hash(model) == before
    keys = set(model.state_dict())
    assert {
        "head.fc.threshold_fc.weight",
        "head.fc.threshold_fc.bias",
        "head.fc.none_mask",
    } <= keys
    assert "head.fc.weight" not in keys
    assert model.head.fc.weight is model.head.fc.threshold_fc.weight
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
        assert out.shape == (2, 5)
        torch.testing.assert_close(
            head_tail_logits(model, pooled_features(model, x)), out
        )


def test_install_on_a_tiny_siglip(monkeypatch):
    install_tiny_trunk(monkeypatch)
    model = create_model(
        model_size="siglip2_base", pretrained=False, num_classes=4
    ).eval()
    install_ordinal_threshold_head(model, 4, none_index=-1)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 32, 32))
    torch.testing.assert_close(
        torch.logsumexp(out, 1), torch.zeros(1), atol=1e-5, rtol=0
    )


@pytest.mark.parametrize("with_names", [True, False])
def test_load_checkpoint_reinstalls_the_head(tmp_path, with_names):
    torch.manual_seed(0)
    names = ["__none__", "a", "b", "c"]
    model = create_model(model_size="atto", pretrained=False, num_classes=4).eval()
    install_ordinal_threshold_head(model, 4, none_index=0)
    meta = {
        "state_dict": model.state_dict(),
        "model_size": "atto",
        "ordinal_head": "threshold",
    }
    if with_names:
        meta.update(class_names=names, num_classes=4)
    torch.save(meta, tmp_path / "best.pt")
    loaded = load_checkpoint(str(tmp_path / "best.pt")).eval()
    assert has_threshold_head(loaded) and loaded.head.fc.none_index == 0
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(model(x), loaded(x))


def test_warm_start_from_a_softmax_incumbent_keeps_the_trunk(tmp_path):
    (tmp_path / "ckpt").mkdir()
    torch.manual_seed(7)
    incumbent = create_model(model_size="atto", pretrained=False, num_classes=4)
    torch.save(
        {"state_dict": incumbent.state_dict(), "model_size": "atto", "num_classes": 4},
        tmp_path / "ckpt" / "best.pt",
    )
    config = gt.GroupTrainConfig(
        group_folder=str(tmp_path),
        num_classes=4,
        class_names=["a", "b", "c", "__none__"],
        backbone_variant="atto",
        ordinal=True,
        ordinal_head="threshold",
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    model = gt._create_or_warmstart_model(
        config,
        device=torch.device("cpu"),
        dtype=torch.float32,
        head_hidden_size=None,
        checkpoint_dir=tmp_path / "ckpt",
    )
    assert has_threshold_head(model) and model.head.fc.none_index == 3
    assert torch.equal(model.stem[0].weight, incumbent.stem[0].weight)


def test_loading_keeps_the_modules_own_none_position():
    """A warm start from a threshold incumbent whose __none__ sat elsewhere must
    not overwrite the run's own none position (the trunk and thresholds may)."""
    source = _fc(4, none_index=0)
    target = _fc(4, none_index=3)
    target.load_state_dict(source.state_dict(), strict=False)
    assert target.none_index == 3
    assert target.none_mask.tolist() == [0.0, 0.0, 0.0, 1.0]
