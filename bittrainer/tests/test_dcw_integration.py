"""Wiring tests for the dynamic-class-weight controller inside group_trainer.

These cover the branching that could actually be wrong — the enable/multi-label
gate, the base-weight selection, and the critical *no-op at epoch 1* property
(enabling the controller must not perturb training until a class overfits). The
full run_group_training end-to-end path is exercised by the CPU A/B experiment
(ISSUE-0392, Phase 5), not here, to keep this test off the network/compile path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bittrainer.group_trainer import GroupTrainConfig, _build_dcw_controller

_DEVICE = torch.device("cpu")


def _cfg(**kw) -> GroupTrainConfig:
    base = dict(group_folder="/tmp/grp", num_classes=4, class_names=["a", "b", "c", "d"])
    base.update(kw)
    return GroupTrainConfig(**base)


def test_disabled_returns_none():
    assert _build_dcw_controller(_cfg(dynamic_class_weighting=False), None, _DEVICE) is None


def test_multilabel_gate_returns_none():
    cfg = _cfg(dynamic_class_weighting=True, multi_label=True)
    assert _build_dcw_controller(cfg, None, _DEVICE) is None


def test_singlelabel_ones_base_is_numerical_noop():
    cfg = _cfg(dynamic_class_weighting=True)
    ctrl = _build_dcw_controller(cfg, None, _DEVICE)
    assert ctrl is not None
    w = ctrl.current_weights()
    assert torch.allclose(w, torch.ones(4), atol=1e-6)


def test_initial_weights_match_unweighted_ce():
    # The stronger property: CE(weight=initial_weights) == plain CE at epoch 1.
    cfg = _cfg(dynamic_class_weighting=True)
    ctrl = _build_dcw_controller(cfg, None, _DEVICE)
    torch.manual_seed(0)
    logits = torch.randn(16, 4)
    labels = torch.randint(0, 4, (16,))
    weighted = nn.CrossEntropyLoss(weight=ctrl.current_weights())(logits, labels)
    plain = nn.CrossEntropyLoss()(logits, labels)
    assert torch.allclose(weighted, plain, atol=1e-6)


def test_reweight_base_is_modulated_and_renormalised():
    cfg = _cfg(dynamic_class_weighting=True)
    base = torch.tensor([1.6, 0.4, 0.6, 1.4])  # mean 1 effective-number-style weights
    ctrl = _build_dcw_controller(cfg, base, _DEVICE)
    w = ctrl.current_weights()
    assert abs(float(w.mean()) - 1.0) < 1e-6
    # ordering preserved at start (no throttling yet)
    assert float(w[0]) > float(w[3]) > float(w[2]) > float(w[1])


def test_config_knobs_propagate_to_controller():
    cfg = _cfg(
        dynamic_class_weighting=True,
        dcw_metric="both",
        dcw_patience=3,
        dcw_decay=0.5,
        dcw_floor=0.1,
        dcw_cooldown=2,
        dcw_min_delta=0.02,
        dcw_ema_decay=0.9,
    )
    ctrl = _build_dcw_controller(cfg, None, _DEVICE)
    assert ctrl.metric == "both"
    assert ctrl.patience == 3
    assert ctrl.decay == 0.5
    assert ctrl.floor == 0.1
    assert ctrl.cooldown_period == 2
    assert ctrl.min_delta == 0.02
    assert ctrl.ema_decay == 0.9
