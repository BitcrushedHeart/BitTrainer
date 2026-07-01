"""Selection/shipping decode alignment.

Per-epoch selection used to score checkpoints on uncalibrated argmax while the
shipped model applies temperature -> __none__ logit bias -> ordinal EV
cut-point decode (all fitted only at finalisation), so the "best" epoch was
chosen under a decode that never ships. `_shipped_decode_metrics` scores every
epoch under the shipped decode; `_incumbent_decode_metrics` scores the
incumbent under its OWN persisted calibration so the fair comparison stays
fair across the change (argmax fallback for pre-change checkpoints).
"""

from __future__ import annotations

import torch

from bittrainer.group_trainer import (
    GroupTrainConfig,
    _incumbent_decode_metrics,
    _metric_score,
    _metrics_from_logits,
    _shipped_decode_metrics,
)


def _cfg(**kw) -> GroupTrainConfig:
    base = dict(group_folder="/tmp/grp", num_classes=4, class_names=["a", "b", "c", "d"])
    base.update(kw)
    return GroupTrainConfig(**base)


def _edge_biased_logits(n_per_class: int = 30) -> tuple[torch.Tensor, torch.Tensor]:
    """Ordinal posteriors systematically biased inward at the scale edges.

    Middle classes are predicted sharply; edge classes have their mass split
    with the adjacent inner class so argmax lands inward while E[j] with a
    fitted boundary recovers the edge. This is exactly the shape the EV
    cut-point decode exists to fix.
    """
    rows: list[list[float]] = []
    labels: list[int] = []
    per_true = {
        0: [0.44, 0.46, 0.10, 0.00],  # argmax -> 1, EV = 0.66
        1: [0.10, 0.80, 0.10, 0.00],
        2: [0.00, 0.10, 0.80, 0.10],
        3: [0.00, 0.10, 0.46, 0.44],  # argmax -> 2, EV = 2.34
    }
    for true, probs in per_true.items():
        for _ in range(n_per_class):
            rows.append(probs)
            labels.append(true)
    probs_t = torch.tensor(rows, dtype=torch.float32).clamp_min(1e-9)
    return torch.log(probs_t), torch.tensor(labels, dtype=torch.long)


class TestShippedDecodeMetrics:
    def test_ordinal_shipped_decode_beats_argmax_on_edge_bias(self) -> None:
        config = _cfg(ordinal=True, validation_metric="qwk")
        logits, labels = _edge_biased_logits()

        argmax_metrics = _metrics_from_logits(logits, labels, config, none_index=-1)
        shipped = _shipped_decode_metrics(logits, labels, config, none_index=-1)

        assert argmax_metrics["macro_f1"] < 1.0  # edge classes decoded inward
        assert shipped["macro_f1"] == 1.0  # cut-point decode recovers the edges
        assert _metric_score(shipped, config) > _metric_score(argmax_metrics, config)
        assert shipped["selection_decode"] == "shipped"

    def test_selection_flip_between_epochs(self) -> None:
        """The epoch that ships better must win selection under the new scorer."""
        config = _cfg(ordinal=True, validation_metric="qwk")

        # Epoch A: mediocre-but-argmax-friendly. One edge class fully wrong.
        rows_a, labels_a = [], []
        per_true_a = {
            0: [0.85, 0.15, 0.00, 0.00],
            1: [0.15, 0.70, 0.15, 0.00],
            2: [0.00, 0.15, 0.70, 0.15],
            3: [0.00, 0.10, 0.80, 0.10],  # argmax AND EV land on 2
        }
        for true, probs in per_true_a.items():
            for _ in range(30):
                rows_a.append(probs)
                labels_a.append(true)
        logits_a = torch.log(torch.tensor(rows_a, dtype=torch.float32).clamp_min(1e-9))
        labels_a_t = torch.tensor(labels_a, dtype=torch.long)

        # Epoch B: edge-biased posteriors -- poor under argmax, perfect shipped.
        logits_b, labels_b_t = _edge_biased_logits()

        old_score_a = _metric_score(_metrics_from_logits(logits_a, labels_a_t, config, -1), config)
        old_score_b = _metric_score(_metrics_from_logits(logits_b, labels_b_t, config, -1), config)
        new_score_a = _metric_score(_shipped_decode_metrics(logits_a, labels_a_t, config, -1), config)
        new_score_b = _metric_score(_shipped_decode_metrics(logits_b, labels_b_t, config, -1), config)

        assert old_score_a > old_score_b  # argmax selection picked the wrong epoch
        assert new_score_b > new_score_a  # shipped-decode selection flips it

    def test_non_ordinal_no_none_reduces_to_argmax(self) -> None:
        config = _cfg(ordinal=False, validation_metric="macro_f1")
        logits = torch.tensor(
            [[2.0, 0.1, 0.0, 0.0], [0.0, 2.0, 0.1, 0.0], [0.0, 0.1, 2.0, 0.0]],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 1, 2], dtype=torch.long)
        shipped = _shipped_decode_metrics(logits, labels, config, none_index=-1)
        argmax_metrics = _metrics_from_logits(logits, labels, config, none_index=-1)
        assert shipped["macro_f1"] == argmax_metrics["macro_f1"]
        assert shipped["per_class_f1"] == argmax_metrics["per_class_f1"]


class TestIncumbentDecodeMetrics:
    def test_persisted_none_bias_is_applied(self) -> None:
        config = _cfg(class_names=["__none__", "b", "c", "d"])
        none_index = 0
        # Borderline sample: real class narrowly beats __none__ pre-bias.
        logits = torch.tensor([[1.0, 1.2, 0.0, 0.0]], dtype=torch.float32)
        labels = torch.tensor([0], dtype=torch.long)

        no_calib = _incumbent_decode_metrics(logits, labels, config, none_index, {})
        assert no_calib["per_class_recall"]["0"] == 0.0  # argmax fallback: misses none

        ckpt = {"temperature": 1.0, "class_logit_bias": [0.5, 0.0, 0.0, 0.0]}
        with_bias = _incumbent_decode_metrics(logits, labels, config, none_index, ckpt)
        assert with_bias["per_class_recall"]["0"] == 1.0  # bias flips it to none

    def test_persisted_cut_points_are_applied_for_ordinal(self) -> None:
        config = _cfg(ordinal=True, validation_metric="qwk")
        logits, labels = _edge_biased_logits(n_per_class=5)
        # Boundaries fitted outward, matching what finalisation would persist.
        ckpt = {"ordinal_cut_points": [0.7, 1.5, 2.3]}
        metrics = _incumbent_decode_metrics(logits, labels, config, -1, ckpt)
        assert metrics["macro_f1"] == 1.0
        assert metrics["selection_decode"] == "shipped"

    def test_non_dict_checkpoint_falls_back_to_argmax(self) -> None:
        config = _cfg(ordinal=True, validation_metric="qwk")
        logits, labels = _edge_biased_logits(n_per_class=5)
        metrics = _incumbent_decode_metrics(logits, labels, config, -1, None)
        argmax_metrics = _metrics_from_logits(logits, labels, config, -1)
        assert metrics["macro_f1"] == argmax_metrics["macro_f1"]
