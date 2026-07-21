"""Single Prodigy_adv factory (Bitcrush ISSUE-0542 unification).

All trainers must build their optimizer through one factory carrying the
canonical Prodigy_adv + Kourkoutas hyperparameters; the per-trainer copies
delegate to it so the three optimizer stories collapse into one.
"""

from __future__ import annotations

import torch.nn as nn
from adv_optm import Prodigy_adv

from bittrainer.generic.optimizer import make_optimizer
from bittrainer.model import build_llrd_param_groups, create_model


def _tiny_model() -> nn.Module:
    return create_model(model_size="atto", pretrained=False, num_classes=3)


def test_make_optimizer_is_prodigy_kourkoutas():
    opt = make_optimizer(_tiny_model())
    assert isinstance(opt, Prodigy_adv)
    defaults = opt.defaults
    assert defaults["lr"] == 1.0
    assert defaults["d_coef"] == 0.9
    assert defaults["weight_decay"] == 0.01
    assert defaults["betas"] == (0.9, 0.999)
    assert defaults["kourkoutas_beta"] is True
    assert defaults["k_warmup_steps"] == 50
    assert defaults["cautious_wd"] is True


def test_llrd_param_groups_match_model_helper():
    model = _tiny_model()
    opt = make_optimizer(model, llrd=True, llrd_decay=0.8)
    expected = build_llrd_param_groups(model, 0.8)
    assert len(opt.param_groups) == len(expected)
    for got, want in zip(opt.param_groups, expected, strict=True):
        assert [id(p) for p in got["params"]] == [id(p) for p in want["params"]]
        assert got["lr"] == want["lr"]  # per-group multiplier on Prodigy's d
        assert got["name"] == want["name"]


def test_flat_params_without_llrd():
    model = _tiny_model()
    opt = make_optimizer(model)
    assert len(opt.param_groups) == 1
    assert len(opt.param_groups[0]["params"]) == len(list(model.parameters()))


def test_trainer_factories_delegate_to_shared():
    """The per-trainer _make_optimizer names must be the shared factory's
    output path — group and binary configs both produce identical defaults."""
    from bittrainer.group_trainer import GroupTrainConfig
    from bittrainer.group_trainer import _make_optimizer as group_make
    from bittrainer.trainer import TrainConfig
    from bittrainer.trainer import _make_optimizer as binary_make

    model = _tiny_model()
    g = group_make(model, GroupTrainConfig(group_folder=".", num_classes=3, class_names=["a", "b", "c"], llrd=False))
    b = binary_make(model, TrainConfig(concept_folder=".", llrd=False))
    ref = make_optimizer(model)
    assert isinstance(g, Prodigy_adv) and isinstance(b, Prodigy_adv)
    assert g.defaults == ref.defaults == b.defaults
