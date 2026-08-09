"""Backbone calibration export: per-head val-tuned thresholds (+ optional temperature).

Trainer-parity round, workstream A2. Engine consumers historically assumed
``sigmoid(logit) > 0.5`` per binary head; the candidate now ships per-head
decision thresholds tuned on the (masked-unknown) val split of the WINNING
epoch — ``binary_thresholds_json`` in the safetensors metadata — plus an
optional per-head temperature (``binary_temperatures_json``, config-gated,
default OFF). Consumers decode ``sigmoid(logit / T) >= threshold`` with T
defaulting to 1.0 and threshold to 0.5, so absent keys mean legacy behaviour
(additive wire contract).

Pure tuning tests + one tiny end-to-end atto run (house style of
test_backbone_generic).
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

import bittrainer.backbone_trainer as bb
from bittrainer.backbone_trainer import run_backbone_training
from bittrainer.tests.test_backbone_generic import _request


def _get(name: str):
    fn = getattr(bb, name, None)
    assert fn is not None, f"bb.{name} is missing (workstream A2 not implemented)"
    return fn


# --------------------------------------------------------------------------- #
# Threshold tuning (pure)                                                     #
# --------------------------------------------------------------------------- #


def test_threshold_tuning_beats_default_on_shifted_scores():
    """Positives live at sigmoid ~0.25-0.45, negatives below 0.2: the F1-optimal
    threshold must drop clearly below the naive 0.5."""
    tune = _get("_tune_binary_thresholds")
    probs = np.array([0.25, 0.30, 0.40, 0.45, 0.05, 0.10, 0.15, 0.18])
    targets = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    logits = np.log(probs / (1.0 - probs))
    thresholds = tune({"c": (logits, targets)})
    assert set(thresholds) == {"c"}
    assert 0.18 < thresholds["c"] < 0.25
    # At the tuned threshold the split is perfect; at 0.5 recall would be zero.
    preds = (probs >= thresholds["c"]).astype(int)
    assert preds.tolist() == targets.astype(int).tolist()


def test_threshold_tuning_zero_support_falls_back_to_half():
    tune = _get("_tune_binary_thresholds")
    thresholds = tune(
        {
            "empty": (np.array([-1.0, -2.0, -3.0]), np.array([0.0, 0.0, 0.0])),
            "good": (np.array([2.0, -2.0]), np.array([1.0, 0.0])),
        }
    )
    assert thresholds["empty"] == pytest.approx(0.5)
    assert 0.0 < thresholds["good"] < 1.0


def test_threshold_tuning_is_deterministic():
    tune = _get("_tune_binary_thresholds")
    rng = np.random.default_rng(7)
    logits = rng.normal(size=64)
    targets = (rng.random(64) < 0.3).astype(np.float64)
    a = tune({"c": (logits, targets)})
    b = tune({"c": (logits.copy(), targets.copy())})
    assert a == b


# --------------------------------------------------------------------------- #
# Temperature fitting (pure)                                                  #
# --------------------------------------------------------------------------- #


def _bce_nll(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    p = 1.0 / (1.0 + np.exp(-logits / temperature))
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(-(targets * np.log(p) + (1.0 - targets) * np.log(1.0 - p)).mean())


def test_temperature_fit_reduces_nll_on_overconfident_logits():
    fit = _get("_fit_binary_temperature")
    rng = np.random.default_rng(3)
    # Well-separated "true" logits, then blown up 5x: overconfident, T* ~ 5.
    base = np.where(rng.random(200) < 0.5, 1.0, -1.0)
    targets = (base > 0).astype(np.float64)
    noisy = base + rng.normal(scale=1.5, size=200)
    overconfident = noisy * 5.0
    t = fit(overconfident, targets)
    assert t > 1.5  # clearly detected the overconfidence
    assert _bce_nll(overconfident, targets, t) < _bce_nll(overconfident, targets, 1.0)


def test_temperature_fit_near_one_for_calibrated_logits():
    fit = _get("_fit_binary_temperature")
    rng = np.random.default_rng(11)
    logits = rng.normal(scale=2.0, size=400)
    # Sample targets FROM sigmoid(logits): perfectly calibrated by construction.
    targets = (rng.random(400) < 1.0 / (1.0 + np.exp(-logits))).astype(np.float64)
    t = fit(logits, targets)
    assert 0.6 < t < 1.7


# --------------------------------------------------------------------------- #
# End-to-end export                                                           #
# --------------------------------------------------------------------------- #


def _run(request):
    return asyncio.run(run_backbone_training(request))


def test_candidate_metadata_carries_tuned_thresholds(tmp_path):
    from safetensors import safe_open

    request = _request(tmp_path, epochs=1, max_steps=4, n=12)
    request["training_config"]["validation_split"] = 0.5
    result = _run(request)
    with safe_open(result["candidate_checkpoint_path"], framework="pt") as f:
        metadata = f.metadata()
    assert "binary_thresholds_json" in metadata, (
        "candidate metadata lacks binary_thresholds_json (calibration export missing)"
    )
    thresholds = json.loads(metadata["binary_thresholds_json"])
    assert "watermark" in thresholds
    assert 0.0 < float(thresholds["watermark"]) < 1.0
    # Temperature is OFF by default — key absent, consumers use T=1.
    assert "binary_temperatures_json" not in metadata


def test_temperature_export_is_config_gated(tmp_path):
    from safetensors import safe_open

    request = _request(tmp_path, epochs=1, max_steps=4, n=12)
    request["training_config"]["validation_split"] = 0.5
    request["training_config"]["calibrate_temperature"] = True
    result = _run(request)
    with safe_open(result["candidate_checkpoint_path"], framework="pt") as f:
        metadata = f.metadata()
    temperatures = json.loads(metadata["binary_temperatures_json"])
    assert "watermark" in temperatures
    assert float(temperatures["watermark"]) > 0.0
