"""Unit tests for DynamicClassWeightController (ISSUE-0392).

Pure-logic tests (no model, no disk) driven by hand-built per-class metric
dicts — the template is test_checkpoint_selection.py. EMA smoothing is disabled
(``ema_decay=0.0`` => the EMA follows the current value exactly) and
``min_delta=0.0`` so the trigger sequence is deterministic and readable.
"""

from __future__ import annotations

import pytest
import torch

from bittrainer.dynamic_class_weights import DynamicClassWeightController


def _ctrl(num_classes=4, base=None, **kw):
    if base is None:
        base = torch.ones(num_classes)
    params = dict(
        metric="val_f1", patience=2, ema_decay=0.0, decay=0.8,
        floor=0.25, ceiling=1.0, cooldown=1, min_delta=0.0,
    )
    params.update(kw)
    return DynamicClassWeightController(num_classes, base, **params)


def _f1(**vals):
    """Full 4-class F1 dict; unspecified classes held at a steady 0.9."""
    return {str(c): vals.get(f"c{c}", 0.9) for c in range(4)}


def _loss(**vals):
    return {str(c): vals.get(f"c{c}", 0.1) for c in range(4)}


def test_rejects_unknown_metric():
    with pytest.raises(ValueError):
        _ctrl(metric="nonsense")


def test_no_op_while_improving():
    c = _ctrl()
    for f in (0.5, 0.6, 0.7, 0.8, 0.9):
        w = c.update(_f1(c0=f), _loss())
    assert c.multiplier == [1.0, 1.0, 1.0, 1.0]
    assert c.adjustments == 0
    # all-ones base * all-ones multipliers, renormalised -> exactly the base
    assert torch.allclose(w, torch.ones(4), atol=1e-6)


def test_plateau_does_not_trigger():
    c = _ctrl()
    for _ in range(6):
        c.update(_f1(c0=0.8), _loss())
    assert c.multiplier[0] == 1.0
    assert c.adjustments == 0


def test_shrinks_after_sustained_decline():
    c = _ctrl()
    c.update(_f1(c0=0.8), _loss())  # peak
    c.update(_f1(c0=0.7), _loss())  # decline, stale=1 (< patience)
    assert c.multiplier[0] == 1.0
    c.update(_f1(c0=0.6), _loss())  # decline, stale=2 -> throttle
    assert c.multiplier[0] == pytest.approx(0.8)
    # only the offending class is touched
    assert c.multiplier[1:] == [1.0, 1.0, 1.0]
    assert c.adjustments == 1


def test_cooldown_blocks_consecutive_throttle():
    c = _ctrl(cooldown=1)
    for f in (0.8, 0.7, 0.6):  # -> throttle at the third (mult 0.8)
        c.update(_f1(c0=f), _loss())
    assert c.multiplier[0] == pytest.approx(0.8)
    c.update(_f1(c0=0.5), _loss())  # decline but cooldown active -> no throttle
    assert c.multiplier[0] == pytest.approx(0.8)
    c.update(_f1(c0=0.4), _loss())  # cooldown elapsed + stale>=patience -> throttle
    assert c.multiplier[0] == pytest.approx(0.64)


def test_respects_floor():
    c = _ctrl(cooldown=0, floor=0.25)
    f = 0.9
    for _ in range(40):
        f -= 0.02
        c.update(_f1(c0=f), _loss())
    assert c.multiplier[0] == pytest.approx(0.25)
    assert c.multiplier[0] >= 0.25  # never breaches the floor


def test_renormalises_to_mean_one():
    c = _ctrl()
    for f in (0.8, 0.7, 0.6):
        w = c.update(_f1(c0=f), _loss())
    assert c.multiplier[0] < 1.0
    assert float(w.mean()) == pytest.approx(1.0, abs=1e-6)
    # a throttled class stays relatively down-weighted vs the others
    assert float(w[0]) < float(w[1])


def test_both_needs_loss_corroboration():
    # F1 declines but val loss FALLS (model still generalising) -> no throttle.
    c = _ctrl(metric="both")
    c.update(_f1(c0=0.8), _loss(c0=0.20))
    c.update(_f1(c0=0.7), _loss(c0=0.15))
    c.update(_f1(c0=0.6), _loss(c0=0.10))
    assert c.multiplier[0] == 1.0
    assert c.adjustments == 0


def test_both_throttles_when_loss_rises():
    c = _ctrl(metric="both")
    c.update(_f1(c0=0.8), _loss(c0=0.20))  # peak, loss_at_peak=0.20
    c.update(_f1(c0=0.7), _loss(c0=0.30))  # F1 down + loss up -> stale=1
    c.update(_f1(c0=0.6), _loss(c0=0.40))  # stale=2 -> throttle
    assert c.multiplier[0] == pytest.approx(0.8)
    assert c.adjustments == 1


def test_val_loss_metric_throttles_on_rising_loss():
    c = _ctrl(metric="val_loss")
    c.update(_f1(), _loss(c0=0.20))  # peak (lowest loss)
    c.update(_f1(), _loss(c0=0.30))  # loss up -> stale=1
    c.update(_f1(), _loss(c0=0.40))  # stale=2 -> throttle
    assert c.multiplier[0] == pytest.approx(0.8)


def test_zero_support_class_not_penalised():
    # Class 3 is absent from the loss dict (no val support). compute_multiclass
    # would report its F1 as 0.0 — which must NOT be read as a decline.
    c = _ctrl()
    for _ in range(5):
        f1 = {"0": 0.9, "1": 0.9, "2": 0.9, "3": 0.0}
        loss = {"0": 0.1, "1": 0.1, "2": 0.1}  # no "3"
        c.update(f1, loss)
    assert c.multiplier[3] == 1.0
    assert c.adjustments == 0


def test_effective_number_base_is_modulated():
    # Non-uniform base weights (mean 1) compose with the multiplier + renorm.
    base = torch.tensor([2.0, 0.5, 0.75, 1.75])  # mean 1.25 (arbitrary)
    c = _ctrl(base=base)
    for f in (0.8, 0.7, 0.6):
        w = c.update(_f1(c0=f), _loss())
    # class 0 throttled -> its share of the (renormalised) weight budget drops
    # relative to the untouched classes, but the base ordering among the rest holds.
    assert c.multiplier[0] == pytest.approx(0.8)
    assert float(w.mean()) == pytest.approx(1.0, abs=1e-6)
    assert float(w[3]) > float(w[1])  # base 1.75 vs 0.5 preserved among untouched
