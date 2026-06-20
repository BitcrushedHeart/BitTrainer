"""Characterization tests pinning build_group_loss_fn to the pre-refactor inline loss.

These reconstruct the exact loss each branch of _train_one_epoch used to compute
inline, independently of the extracted helper, and assert the helper reproduces
them numerically. A subtle extraction bug that changed full-train behaviour would
fail here rather than slipping through a green smoke test.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bittrainer.group_trainer import (
    GroupTrainConfig,
    _build_soft_targets,
    _resolve_none_index,
    _soft_ce_loss,
    build_group_loss_fn,
)
from bittrainer.losses import AsymmetricLoss

_DEVICE = torch.device("cpu")


def _cfg(**kw) -> GroupTrainConfig:
    base = dict(group_folder="/tmp/grp", num_classes=4, class_names=["a", "b", "c", "d"])
    base.update(kw)
    return GroupTrainConfig(**base)


def _multiclass_batch(num_classes: int, n: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    logits = torch.randn(n, num_classes)
    labels = torch.randint(0, num_classes, (n,))
    return logits, labels


def _multilabel_batch(num_classes: int, n: int = 8, seed: int = 1):
    torch.manual_seed(seed)
    logits = torch.randn(n, num_classes)
    labels = (torch.rand(n, num_classes) > 0.5).float()
    return logits, labels


class TestPlainCrossEntropy:
    def test_plain_ce_ignores_config_smoothing_when_soft_targets_disabled(self):
        config = _cfg(label_smoothing=0.1, multi_label=False)
        logits, labels = _multiclass_batch(config.num_classes)

        reference = nn.CrossEntropyLoss(label_smoothing=0.0)(logits, labels)

        loss_fn = build_group_loss_fn(
            config, use_soft_targets=False,
            none_index=_resolve_none_index(config.class_names), device=_DEVICE,
        )
        assert torch.allclose(loss_fn(logits, labels), reference, atol=1e-6)

    def test_zero_smoothing(self):
        config = _cfg(label_smoothing=0.0, multi_label=False)
        logits, labels = _multiclass_batch(config.num_classes, seed=3)
        reference = nn.CrossEntropyLoss(label_smoothing=0.0)(logits, labels)
        loss_fn = build_group_loss_fn(
            config, use_soft_targets=False,
            none_index=_resolve_none_index(config.class_names), device=_DEVICE,
        )
        assert torch.allclose(loss_fn(logits, labels), reference, atol=1e-6)


class TestSoftCrossEntropy:
    def _reference(self, config, logits, labels, none_index):
        soft = _build_soft_targets(
            labels, config.num_classes,
            ordinal=config.ordinal,
            label_smoothing=config.label_smoothing,
            soft_aliases=config.soft_aliases or None,
            none_index=none_index,
            device=_DEVICE,
        )
        log_probs = torch.log_softmax(logits.float(), dim=1)
        return _soft_ce_loss(log_probs, soft)

    def test_ordinal_with_none_index(self):
        config = _cfg(
            num_classes=4, class_names=["0", "1", "2", "__none__"],
            ordinal=True, label_smoothing=0.1,
        )
        logits, labels = _multiclass_batch(config.num_classes, seed=5)
        none_index = _resolve_none_index(config.class_names)
        assert none_index == 3

        reference = self._reference(config, logits, labels, none_index)
        loss_fn = build_group_loss_fn(
            config, use_soft_targets=True, none_index=none_index, device=_DEVICE,
        )
        assert torch.allclose(loss_fn(logits, labels), reference, atol=1e-6)

    def test_soft_aliases_non_ordinal(self):
        config = _cfg(
            num_classes=4, class_names=["a", "b", "c", "d"],
            ordinal=False, label_smoothing=0.05,
            soft_aliases={"0": [(1, 0.3)], "2": [(3, 0.2)]},
        )
        logits, labels = _multiclass_batch(config.num_classes, seed=7)
        none_index = _resolve_none_index(config.class_names)

        reference = self._reference(config, logits, labels, none_index)
        loss_fn = build_group_loss_fn(
            config, use_soft_targets=True, none_index=none_index, device=_DEVICE,
        )
        assert torch.allclose(loss_fn(logits, labels), reference, atol=1e-6)

    def test_non_ordinal_smoothing_excludes_none_class(self):
        config = _cfg(
            num_classes=4, class_names=["__none__", "a", "b", "c"],
            ordinal=False, label_smoothing=0.12,
        )
        labels = torch.tensor([0, 1])
        targets = _build_soft_targets(
            labels, config.num_classes,
            ordinal=False, label_smoothing=config.label_smoothing,
            none_index=_resolve_none_index(config.class_names), device=_DEVICE,
        )
        assert torch.allclose(targets[0], torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
        assert targets[1, 0] == 0.0
        assert torch.allclose(targets[1, 1:], torch.tensor([0.88, 0.06, 0.06]), atol=1e-6)


class TestMultiLabel:
    def test_bce(self):
        config = _cfg(multi_label=True, use_asl=False)
        logits, labels = _multilabel_batch(config.num_classes)
        reference = nn.BCEWithLogitsLoss()(logits.float(), labels.float())
        loss_fn = build_group_loss_fn(
            config, use_soft_targets=False,
            none_index=_resolve_none_index(config.class_names), device=_DEVICE,
        )
        assert torch.allclose(loss_fn(logits, labels), reference, atol=1e-6)

    def test_asl(self):
        config = _cfg(
            multi_label=True, use_asl=True,
            asl_gamma_neg=4.0, asl_gamma_pos=0.0, asl_clip=0.05,
        )
        logits, labels = _multilabel_batch(config.num_classes, seed=9)
        reference = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)(
            logits.float(), labels.float()
        )
        loss_fn = build_group_loss_fn(
            config, use_soft_targets=False,
            none_index=_resolve_none_index(config.class_names), device=_DEVICE,
        )
        assert torch.allclose(loss_fn(logits, labels), reference, atol=1e-6)


class TestSoftTargetsInvariants:
    def test_rows_sum_to_one_and_none_isolated(self):
        num_classes = 4
        none_index = 3
        labels = torch.tensor([0, 1, 2, 3])
        targets = _build_soft_targets(
            labels, num_classes, ordinal=True, none_index=none_index, device=_DEVICE,
        )
        assert torch.allclose(targets.sum(dim=1), torch.ones(4), atol=1e-6)
        # __none__ row stays one-hot — no probability bleeds to ordinal neighbours
        assert torch.allclose(targets[3], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-6)
        # ordinal smoothing spreads weight off the diagonal for a mid class
        assert targets[1, 0] > 0.0 and targets[1, 2] > 0.0

    def test_ordinal_sigma_zero_bypasses_gaussian(self):
        num_classes = 4
        none_index = 3
        labels = torch.tensor([0])
        targets = _build_soft_targets(
            labels, num_classes, ordinal=True, ordinal_sigma=0.0, label_smoothing=0.1,
            none_index=none_index, device=_DEVICE,
        )
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(targets[0], expected, atol=1e-6)

    def test_ordinal_sigma_modulates_bleed_width(self):
        num_classes = 4
        none_index = 3
        labels = torch.tensor([1])  # Mid class
        # Narrow sigma (0.4)
        targets_narrow = _build_soft_targets(
            labels, num_classes, ordinal=True, ordinal_sigma=0.4, label_smoothing=0.1,
            none_index=none_index, device=_DEVICE,
        )
        # Standard sigma (1.0)
        targets_std = _build_soft_targets(
            labels, num_classes, ordinal=True, ordinal_sigma=1.0, label_smoothing=0.1,
            none_index=none_index, device=_DEVICE,
        )
        # Bleed to adjacent class (class 0 or 2) should be much smaller in narrow sigma
        assert targets_narrow[0, 0] < targets_std[0, 0]
        # And the target class itself should retain more probability in the narrow case
        assert targets_narrow[0, 1] > targets_std[0, 1]
