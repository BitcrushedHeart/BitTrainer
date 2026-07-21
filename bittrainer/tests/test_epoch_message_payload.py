"""The per-epoch emit payload carries the ISSUE-0491 run-history keys.

_build_epoch_message is the exact contract Engine buffers into
group_training_epochs, so it must expose weighted/micro F1, both losses,
per-class F1, and elapsed wall-time.
"""

from __future__ import annotations

from bittrainer.group_trainer import GroupTrainConfig, _build_epoch_message


def _cfg(**kw) -> GroupTrainConfig:
    base = dict(group_folder="/tmp/grp", num_classes=3, class_names=["a", "b", "c"])
    base.update(kw)
    return GroupTrainConfig(**base)


VAL_METRICS = {
    "val_loss": 0.42,
    "macro_f1": 0.71,
    "weighted_f1": 0.66,
    "micro_f1": 0.80,
    "per_class_f1": {"0": 0.9, "1": 0.5, "2": 0.7},
    "qwk": 0.85,
    "adjacent_accuracy": 0.95,
}


def test_payload_has_all_history_keys():
    msg = _build_epoch_message(
        epoch=2,
        config=_cfg(),
        train_loss=0.33,
        val_metrics=VAL_METRICS,
        best_val_macro_f1=0.71,
        best_val_qwk=0.0,
        selected_score=0.71,
        best_validation_score=0.71,
        best_epoch=2,
        per_class_train_loss={},
        elapsed_seconds=12.5,
    )
    assert msg["type"] == "epoch_complete"
    assert msg["epoch"] == 3
    assert msg["train_loss"] == 0.33
    assert msg["val_loss"] == 0.42
    assert msg["val_macro_f1"] == 0.71
    assert msg["val_weighted_f1"] == 0.66
    assert msg["val_micro_f1"] == 0.80
    assert msg["per_class_f1"] == {"0": 0.9, "1": 0.5, "2": 0.7}
    assert msg["elapsed_seconds"] == 12.5
    assert msg["best_epoch"] == 3


def test_ordinal_payload_adds_qwk():
    msg = _build_epoch_message(
        epoch=0,
        config=_cfg(ordinal=True, validation_metric="qwk"),
        train_loss=0.5,
        val_metrics=VAL_METRICS,
        best_val_macro_f1=0.71,
        best_val_qwk=0.85,
        selected_score=0.8,
        best_validation_score=0.8,
        best_epoch=0,
        per_class_train_loss={},
        elapsed_seconds=5.0,
    )
    assert msg["val_qwk"] == 0.85
    assert msg["val_adjacent_accuracy"] == 0.95
    assert msg["val_weighted_f1"] == 0.66
