from __future__ import annotations

import torch

import bittrainer.group_trainer as gt


class TinyHeadModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(2, 2)


def _patch_probe(monkeypatch, original_weight, scores):
    monkeypatch.setattr(
        gt,
        "prepare_head_probe_tensors",
        lambda *a, **k: (
            torch.zeros(2, 2),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(2, 2),
            torch.zeros(2, dtype=torch.long),
        ),
    )

    starts_clean: list[bool] = []

    def fake_probe(model, _x_train, _y_train, _x_val, _y_val, config, **_kwargs):
        starts_clean.append(torch.allclose(model.head.weight, original_weight))
        value = config.ordinal_sigma if config.ordinal else config.label_smoothing
        model.head.weight.data.fill_(float(value))
        macro_f1, loss = scores[float(value)]
        return {
            "macro_f1": macro_f1,
            "qwk": macro_f1 - 0.1 if config.ordinal else None,
            "val_loss": loss,
            "best_epoch": 1,
            "epochs_completed": 1,
        }

    monkeypatch.setattr(gt, "train_head_probe_from_tensors", fake_probe)
    return starts_clean


def test_ordinal_softness_sweep_resets_head_and_selects_macro_f1(monkeypatch):
    monkeypatch.setattr(gt, "_ORDINAL_SIGMA_CANDIDATES", [0.0, 0.2, 0.5])
    model = TinyHeadModel()
    original_weight = model.head.weight.detach().clone()
    starts_clean = _patch_probe(
        monkeypatch,
        original_weight,
        {0.0: (0.4, 0.7), 0.2: (0.8, 0.6), 0.5: (0.7, 0.5)},
    )
    config = gt.GroupTrainConfig(
        group_folder="/tmp/group",
        num_classes=3,
        class_names=["a", "b", "c"],
        ordinal=True,
    )

    result = gt._run_auto_softness_probe(
        model, config, object(), None, [], [],
        device=torch.device("cpu"), none_index=-1,
        cb=lambda _msg: None, stop_event=None,
    )

    assert starts_clean == [True, True, True]
    assert result["macro_f1"] == 0.8
    assert config.selected_softness_kind == "ordinal_sigma"
    assert config.selected_softness_value == 0.2
    assert config.ordinal_sigma == 0.2
    assert len(config.soft_label_tuning_results) == 3
    assert torch.all(model.head.weight == 0.2)


def test_label_smoothing_sweep_tiebreaks_to_lower_value(monkeypatch):
    monkeypatch.setattr(gt, "_LABEL_SMOOTHING_CANDIDATES", [0.0, 0.05, 0.1])
    model = TinyHeadModel()
    original_weight = model.head.weight.detach().clone()
    _patch_probe(
        monkeypatch,
        original_weight,
        {0.0: (0.6, 0.4), 0.05: (0.7, 0.3), 0.1: (0.7, 0.3)},
    )
    config = gt.GroupTrainConfig(
        group_folder="/tmp/group",
        num_classes=3,
        class_names=["a", "b", "c"],
        ordinal=False,
    )

    gt._run_auto_softness_probe(
        model, config, object(), None, [], [],
        device=torch.device("cpu"), none_index=-1,
        cb=lambda _msg: None, stop_event=None,
    )

    assert config.selected_softness_kind == "label_smoothing"
    assert config.selected_softness_value == 0.05
    assert config.label_smoothing == 0.05


def test_multi_label_uses_plain_probe_without_sweep(monkeypatch):
    model = TinyHeadModel()
    called = {"plain": 0, "sweep": 0}

    def fake_plain(*_args, **_kwargs):
        called["plain"] += 1
        return {"macro_f1": 0.3, "best_epoch": 1, "epochs_completed": 1}

    def fake_sweep(*_args, **_kwargs):
        called["sweep"] += 1
        raise AssertionError("multi-label groups must not use the softmax sweep")

    monkeypatch.setattr(gt, "train_head_probe", fake_plain)
    monkeypatch.setattr(gt, "train_head_probe_from_tensors", fake_sweep)
    config = gt.GroupTrainConfig(
        group_folder="/tmp/group",
        num_classes=3,
        class_names=["a", "b", "c"],
        multi_label=True,
    )

    result = gt._run_auto_softness_probe(
        model, config, object(), None, [], [],
        device=torch.device("cpu"), none_index=-1,
        cb=lambda _msg: None, stop_event=None,
    )

    assert result["macro_f1"] == 0.3
    assert called == {"plain": 1, "sweep": 0}
    assert config.selected_softness_kind is None
