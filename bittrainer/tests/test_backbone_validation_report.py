"""Validation reports for the Backbone tab (Bitcrush ISSUE-1655).

Pure functions over the validation scores ``_collect_val_scores`` returns,
beside the flat metrics dict that feeds epoch selection and is left alone:

* per single-label group: precision, recall, F1 and support per class, and the
  confusion matrix;
* per yes/no head: precision and recall at the threshold it will use, the
  label counts behind them, and its recall at precision 0.90 and 0.95;
* thresholds that are only tuned for a head with enough validation yes and no
  labels, with a record of which heads were tuned;
* a temperature per single-label group.

Generated scores only.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import bittrainer.backbone_trainer as bb

# -- groups -------------------------------------------------------------------


def _one_hot(predictions, classes: int) -> np.ndarray:
    """Logits whose argmax is each given prediction."""
    logits = np.zeros((len(predictions), classes))
    logits[np.arange(len(predictions)), predictions] = 5.0
    return logits


def test_group_report_on_a_hand_made_group():
    targets = np.array([0, 0, 0, 1, 1, 2])
    predicted = [0, 0, 1, 1, 1, 0]  # one "a" called "b", the "c" called "a"

    report = bb._group_validation_report(
        {"shot": (_one_hot(predicted, 4), targets)}, {"shot": ["a", "b", "c", "d"]}
    )

    assert report == {
        "shot": {
            "classes": ["a", "b", "c", "d"],
            "support": [3, 2, 1, 0],
            "precision": [pytest.approx(2 / 3), pytest.approx(2 / 3), 0.0, 0.0],
            "recall": [pytest.approx(2 / 3), 1.0, 0.0, 0.0],
            "f1": [pytest.approx(2 / 3), pytest.approx(0.8), 0.0, 0.0],
            # Rows are the true class, columns the class it was called.
            "confusion": [[2, 1, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
        }
    }
    json.dumps(report)  # plain lists and numbers


def test_group_report_matches_scikit_learn():
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    rng = np.random.default_rng(3)
    scores = {}
    for index in range(8):
        size, classes = int(rng.integers(1, 300)), int(rng.integers(2, 8))
        # Every other group has a class no validation image belongs to.
        targets = rng.integers(0, classes - index % 2, size)
        scores[f"group_{index}"] = (rng.normal(0, 1, (size, classes)), targets)

    report = bb._group_validation_report(scores)

    assert list(report) == list(scores)
    for group, (logits, targets) in scores.items():
        labels = list(range(logits.shape[1]))
        predicted = logits.argmax(axis=1)
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predicted, labels=labels, zero_division=0
        )
        entry = report[group]
        assert entry["classes"] == [str(label) for label in labels]  # no names given
        assert entry["support"] == support.tolist()
        assert entry["precision"] == pytest.approx(precision.tolist(), abs=1e-12)
        assert entry["recall"] == pytest.approx(recall.tolist(), abs=1e-12)
        assert entry["f1"] == pytest.approx(f1.tolist(), abs=1e-12)
        assert (
            entry["confusion"]
            == confusion_matrix(targets, predicted, labels=labels).tolist()
        )


# -- yes/no heads -------------------------------------------------------------


def _ranked(pattern: str):
    """A head's validation scores, best first: ``+`` a human yes, ``-`` a no.
    Logits run from 4 down to -4, so the top half scores above 0.5."""
    targets = np.array([1.0 if mark == "+" else 0.0 for mark in pattern])
    return np.linspace(4.0, -4.0, len(pattern)), targets


_ONE_SLIP = "+" * 9 + "-" + "+" + "-" * 9


def test_binary_report_at_the_default_and_at_a_tuned_threshold():
    scores = {"head": _ranked(_ONE_SLIP)}

    at_half = bb._binary_validation_report(scores)["head"]
    lower = bb._binary_validation_report(scores, {"head": 0.3})["head"]

    # The ten best scores are above 0.5: nine yes and the one no.
    assert at_half == {
        "yes": 10,
        "no": 10,
        "threshold": 0.5,
        "called": 10,
        "hits": 9,
        "precision": pytest.approx(0.9),
        "recall": pytest.approx(0.9),
        "recall_at_p90": 1.0,
        "recall_at_p95": 0.9,
    }
    # Two more are above 0.3: the last yes and another no.
    assert (lower["threshold"], lower["called"], lower["hits"]) == (0.3, 12, 10)
    assert lower["precision"] == pytest.approx(10 / 12)
    assert lower["recall"] == 1.0
    json.dumps([at_half, lower])


def test_binary_report_says_none_where_there_is_nothing_to_divide_by():
    report = bb._binary_validation_report(
        {
            "nothing_called": (np.full(25, -6.0), np.array([1.0] * 12 + [0.0] * 13)),
            "no_yes": (np.linspace(3, -3, 12), np.zeros(12)),
            "few_labels": _ranked("+" * 9 + "-" * 30),
        }
    )

    assert report["nothing_called"]["called"] == 0
    assert report["nothing_called"]["precision"] is None
    assert report["nothing_called"]["recall"] == 0.0
    assert report["no_yes"]["yes"] == 0 and report["no_yes"]["recall"] is None
    assert report["no_yes"]["precision"] == 0.0
    # Too few yes labels for a recall at fixed precision; the rest is reported.
    assert report["few_labels"]["recall_at_p90"] is None
    assert report["few_labels"]["recall_at_p95"] is None
    assert report["few_labels"]["recall"] == 1.0
    json.dumps(report)


def test_binary_report_agrees_with_the_flat_metrics():
    rng = np.random.default_rng(8)
    scores = {}
    for index in range(12):
        size = int(rng.integers(25, 300))
        targets = (rng.random(size) < 0.5).astype(np.float64)
        targets[:10], targets[10:20] = 1.0, 0.0
        scores[f"head_{index}"] = (rng.normal(0, 2, size) + 2.0 * targets, targets)

    report = bb._binary_validation_report(scores)
    metrics = bb._backbone_metrics(scores, {})

    for concept, entry in report.items():
        assert entry["recall_at_p90"] == metrics[f"binary_r_at_p90/{concept}"]
        assert entry["recall_at_p95"] == metrics[f"binary_r_at_p95/{concept}"]
        assert entry["yes"] == metrics[f"binary_support/{concept}"]


# -- thresholds ---------------------------------------------------------------


def _threshold_scores():
    rng = np.random.default_rng(2)

    def head(yes: int, no: int):
        targets = np.array([1.0] * yes + [0.0] * no)
        return rng.normal(0, 1.5, yes + no) + 2.0 * targets, targets

    return {
        "plenty": head(40, 40),
        "few_yes": head(4, 40),
        "few_no": head(40, 4),
        "no_yes": head(0, 40),
    }


def test_a_threshold_is_only_tuned_with_enough_yes_and_no_labels():
    scores = _threshold_scores()

    thresholds, tuned = bb._binary_threshold_tuning(
        scores, min_positive=5, min_negative=5
    )

    assert tuned == {"plenty": True, "few_yes": False, "few_no": False, "no_yes": False}
    assert thresholds["few_yes"] == thresholds["few_no"] == thresholds["no_yes"] == 0.5
    assert thresholds["plenty"] == bb._tune_binary_thresholds(scores)["plenty"]
    assert thresholds == bb._tune_binary_thresholds(
        scores, min_positive=5, min_negative=5
    )


def test_existing_callers_get_what_they_always_got():
    scores = _threshold_scores()

    thresholds = bb._tune_binary_thresholds(scores)
    same, tuned = bb._binary_threshold_tuning(scores)

    # One validation yes is enough by default, and a no is not asked for.
    assert isinstance(thresholds, dict) and thresholds == same
    assert tuned == {"plenty": True, "few_yes": True, "few_no": True, "no_yes": False}
    assert thresholds["no_yes"] == 0.5


# -- group temperature --------------------------------------------------------


def _mean_cross_entropy(logits: np.ndarray, targets: np.ndarray, t: float) -> float:
    return float(
        torch.nn.functional.cross_entropy(
            torch.from_numpy(logits / t), torch.from_numpy(targets)
        )
    )


def test_group_temperature_is_the_grid_value_with_the_lowest_cross_entropy():
    rng = np.random.default_rng(4)
    grid = np.geomspace(0.25, 4.0, 33)  # the grid of _fit_binary_temperature
    for classes in (2, 5, 9):
        logits = rng.normal(0, 3, (400, classes))
        targets = rng.integers(0, classes, 400)
        targets[:300] = logits[:300].argmax(axis=1)  # mostly right, overconfident

        fitted = bb._fit_group_temperature(logits, targets)

        losses = [_mean_cross_entropy(logits, targets, t) for t in grid]
        assert fitted == pytest.approx(float(grid[int(np.argmin(losses))]))


def test_group_temperature_undoes_a_known_sharpening():
    """Labels drawn from softmax(z): z is calibrated, so 2z needs T = 2."""
    rng = np.random.default_rng(6)
    z = rng.normal(0, 1.5, (20_000, 4))
    probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    drawn = (probs.cumsum(axis=1) < rng.random((20_000, 1))).sum(axis=1)
    targets = np.minimum(drawn, 3)

    assert bb._fit_group_temperature(z, targets) == pytest.approx(1.0)
    assert bb._fit_group_temperature(2.0 * z, targets) == pytest.approx(2.0)


def test_group_temperature_of_a_group_with_no_validation_image_is_one():
    assert bb._fit_group_temperature(np.zeros((0, 3)), np.zeros(0)) == 1.0
