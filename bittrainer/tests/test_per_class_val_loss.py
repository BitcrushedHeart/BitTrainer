"""Per-class validation-loss telemetry (single-label path).

Pins ``_per_class_val_loss`` to a manual per-true-class mean cross-entropy, and
guards that wiring it into ``_metrics_from_logits`` does NOT change the aggregate
``val_loss`` (byte-for-byte the old ``nn.CrossEntropyLoss`` scalar). The
support-weighted mean of the per-class losses must reconcile with that aggregate.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bittrainer.group_trainer import (
    GroupTrainConfig,
    _metrics_from_logits,
    _per_class_val_loss,
    _resolve_none_index,
)

_DEVICE = torch.device("cpu")


def _cfg(**kw) -> GroupTrainConfig:
    base = dict(group_folder="/tmp/grp", num_classes=4, class_names=["a", "b", "c", "d"])
    base.update(kw)
    return GroupTrainConfig(**base)


def _batch(num_classes: int, n: int = 32, seed: int = 0):
    torch.manual_seed(seed)
    logits = torch.randn(n, num_classes)
    labels = torch.randint(0, num_classes, (n,))
    return logits, labels


def test_per_class_matches_manual_grouped_ce():
    logits, labels = _batch(4, seed=1)
    per_ex = nn.functional.cross_entropy(logits.float(), labels.long(), reduction="none")

    out = _per_class_val_loss(logits, labels, 4)

    for c in range(4):
        mask = labels == c
        if bool(mask.any()):
            expected = float(per_ex[mask].mean().item())
            assert out[str(c)] == expected
        else:  # pragma: no cover - seed keeps all classes present
            assert str(c) not in out


def test_keys_are_stringified_indices():
    logits, labels = _batch(4, seed=2)
    out = _per_class_val_loss(logits, labels, 4)
    assert all(isinstance(k, str) for k in out)
    assert set(out).issubset({"0", "1", "2", "3"})


def test_empty_class_is_omitted():
    # labels only ever hit classes 0 and 2 -> classes 1 and 3 have no support.
    logits = torch.randn(6, 4)
    labels = torch.tensor([0, 2, 0, 2, 0, 2])
    out = _per_class_val_loss(logits, labels, 4)
    assert set(out) == {"0", "2"}


def test_empty_labels_return_empty_dict():
    logits = torch.zeros(0, 4)
    labels = torch.zeros(0, dtype=torch.long)
    assert _per_class_val_loss(logits, labels, 4) == {}


def test_support_weighted_mean_reconciles_with_aggregate():
    logits, labels = _batch(4, n=40, seed=3)
    out = _per_class_val_loss(logits, labels, 4)

    aggregate = float(nn.CrossEntropyLoss()(logits.float(), labels.long()).item())
    total = 0.0
    for c in range(4):
        mask = labels == c
        support = int(mask.sum().item())
        if support:
            total += out[str(c)] * support
    weighted = total / labels.numel()
    assert abs(weighted - aggregate) < 1e-5


def test_metrics_from_logits_adds_per_class_and_preserves_val_loss():
    config = _cfg()
    logits, labels = _batch(config.num_classes, seed=4)
    none_index = _resolve_none_index(config.class_names)

    metrics = _metrics_from_logits(logits, labels, config, none_index)

    # aggregate val_loss unchanged from the pre-telemetry CrossEntropyLoss scalar
    expected_val_loss = float(nn.CrossEntropyLoss()(logits.float(), labels.long()).item())
    assert metrics["val_loss"] == expected_val_loss
    # per-class telemetry present and consistent with the standalone helper
    assert metrics["per_class_val_loss"] == _per_class_val_loss(
        logits, labels, config.num_classes
    )
