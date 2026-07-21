"""Bayes logit adjustment: adjusted = logits + tau*(log_natural - log_effective).

A near-boundary rare-class prediction flips back to the majority once the
oversample-induced rare-class boost is divided out; tau=0 is the identity.
"""

from __future__ import annotations

import numpy as np

from bittrainer.group_trainer import _prior_logit_delta


def _log_priors(counts, num_classes):
    from bittrainer.group_dataset import compute_class_log_priors

    return compute_class_log_priors(counts, num_classes)


def test_adjustment_flips_near_boundary_rare_prediction_to_majority():
    num_classes = 2
    # Natural stream: class 0 majority (95%), class 1 rare (5%).
    natural = _log_priors({0: 950, 1: 50}, num_classes)
    # Effective (post 4x oversample): rare class boosted toward parity.
    effective = _log_priors({0: 950, 1: 200}, num_classes)

    delta = _prior_logit_delta(natural, effective, num_classes, tau=1.0)

    # A near-boundary image the balanced model tips slightly toward rare class 1.
    logits = np.array([2.00, 2.10])
    assert int(logits.argmax()) == 1  # pre-adjustment: rare class wins

    adjusted = logits + delta
    assert int(adjusted.argmax()) == 0  # post-adjustment: majority recovers


def test_tau_zero_is_identity():
    num_classes = 3
    natural = _log_priors({0: 100, 1: 10, 2: 1}, num_classes)
    effective = _log_priors({0: 100, 1: 40, 2: 4}, num_classes)
    delta = _prior_logit_delta(natural, effective, num_classes, tau=0.0)
    assert np.allclose(delta, 0.0)


def test_delta_sign_pushes_oversampled_classes_down():
    num_classes = 2
    natural = _log_priors({0: 950, 1: 50}, num_classes)
    effective = _log_priors({0: 950, 1: 200}, num_classes)
    delta = _prior_logit_delta(natural, effective, num_classes, tau=1.0)
    # Rare class was up-weighted in training => negative adjustment; majority
    # was relatively down-weighted => positive adjustment.
    assert delta[1] < 0
    assert delta[0] > 0
