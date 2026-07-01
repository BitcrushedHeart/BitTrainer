"""Report-only macro-F1 variants: supported-classes and __none__-excluded.

The raw macro-F1 averages over ALL defined classes (labels=range(num_classes),
zero_division=0), so a class with zero validation support contributes a
permanent 0 — for a group like Age (102 classes, ~68 with val support) the raw
number has a hard ceiling far below 1.0 even for a perfect model. The variants
make the honest number visible without touching selection.
"""

from __future__ import annotations

import numpy as np

from bittrainer.group_validation import (
    compute_multiclass_metrics,
    compute_multilabel_metrics,
    macro_f1_variants,
)


class TestPerClassSupport:
    def test_support_counts_true_labels_per_class(self) -> None:
        metrics = compute_multiclass_metrics([0, 0, 1], [0, 0, 1], num_classes=3)
        assert metrics["per_class_support"] == {"0": 2, "1": 1, "2": 0}

    def test_empty_labels_returns_empty_support(self) -> None:
        metrics = compute_multiclass_metrics([], [], num_classes=3)
        assert metrics["per_class_support"] == {}

    def test_support_is_independent_of_predictions(self) -> None:
        # Predicting an unsupported class must not give it support.
        metrics = compute_multiclass_metrics([0, 0, 1], [2, 2, 2], num_classes=3)
        assert metrics["per_class_support"] == {"0": 2, "1": 1, "2": 0}

    def test_multilabel_support_counts_positive_labels(self) -> None:
        labels = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]])
        preds = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]])
        metrics = compute_multilabel_metrics(labels, preds, num_classes=3)
        assert metrics["per_class_support"] == {"0": 2, "1": 2, "2": 0}


class TestMacroF1Variants:
    def test_zero_support_class_caps_raw_but_not_supported(self) -> None:
        # Perfect predictions on the two supported classes; class 2 is empty.
        metrics = compute_multiclass_metrics([0, 0, 1], [0, 0, 1], num_classes=3)
        assert metrics["macro_f1"] < 1.0  # raw: dragged to 2/3 by the empty class
        variants = macro_f1_variants(
            metrics["per_class_f1"], metrics["per_class_support"],
            num_classes=3, none_index=-1,
        )
        assert variants["macro_f1_supported"] == 1.0

    def test_excl_none_drops_the_none_class(self) -> None:
        # none (index 0) predicted badly; real classes perfect.
        per_class_f1 = {"0": 0.2, "1": 1.0, "2": 1.0}
        support = {"0": 5, "1": 5, "2": 5}
        variants = macro_f1_variants(per_class_f1, support, num_classes=3, none_index=0)
        assert variants["macro_f1_excl_none"] == 1.0
        assert abs(variants["macro_f1_supported"] - (0.2 + 1.0 + 1.0) / 3) < 1e-9

    def test_supported_excl_none_combines_both_filters(self) -> None:
        per_class_f1 = {"0": 0.0, "1": 0.8, "2": 0.0, "3": 0.6}
        support = {"0": 3, "1": 4, "2": 0, "3": 2}
        variants = macro_f1_variants(per_class_f1, support, num_classes=4, none_index=0)
        # supported: mean over classes 0,1,3 -> (0.0 + 0.8 + 0.6)/3
        assert abs(variants["macro_f1_supported"] - (0.0 + 0.8 + 0.6) / 3) < 1e-9
        # excl none: mean over classes 1,2,3 -> (0.8 + 0.0 + 0.6)/3
        assert abs(variants["macro_f1_excl_none"] - (0.8 + 0.0 + 0.6) / 3) < 1e-9
        # both: mean over classes 1,3 -> (0.8 + 0.6)/2
        assert abs(variants["macro_f1_supported_excl_none"] - 0.7) < 1e-9

    def test_all_zero_support_falls_back_to_unfiltered_mean(self) -> None:
        per_class_f1 = {"0": 0.5, "1": 0.7}
        support = {"0": 0, "1": 0}
        variants = macro_f1_variants(per_class_f1, support, num_classes=2, none_index=-1)
        assert abs(variants["macro_f1_supported"] - 0.6) < 1e-9

    def test_no_none_class_makes_excl_none_equal_raw_mean(self) -> None:
        per_class_f1 = {"0": 0.5, "1": 0.7, "2": 0.9}
        support = {"0": 1, "1": 1, "2": 1}
        variants = macro_f1_variants(per_class_f1, support, num_classes=3, none_index=-1)
        assert abs(variants["macro_f1_excl_none"] - 0.7) < 1e-9
        assert abs(variants["macro_f1_supported"] - 0.7) < 1e-9

    def test_supported_ceiling_scenario_age_like(self) -> None:
        # 10-class group, only 4 classes have val support, all predicted
        # perfectly: raw macro-F1 is capped at 0.4 while supported reads 1.0.
        labels = [0, 1, 2, 3]
        preds = [0, 1, 2, 3]
        metrics = compute_multiclass_metrics(labels, preds, num_classes=10)
        assert abs(metrics["macro_f1"] - 0.4) < 1e-9
        variants = macro_f1_variants(
            metrics["per_class_f1"], metrics["per_class_support"],
            num_classes=10, none_index=-1,
        )
        assert variants["macro_f1_supported"] == 1.0
