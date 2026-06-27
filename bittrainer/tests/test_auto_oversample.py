from __future__ import annotations

import torch

import bittrainer.group_trainer as gt

_NONE_INDEX = 2
# 2 each of classes 0/1/__none__ → max_count 2, target ceil(1.5 * 2*2) = 6,
# so the oversampled train set has 6 __none__ rows (10 total vs 6 base).
_BASE_Y = torch.tensor([0, 0, 1, 1, _NONE_INDEX, _NONE_INDEX], dtype=torch.long)
_BASE_N = _BASE_Y.numel()


class TinyHeadModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(2, 2)


def _patch_probe(monkeypatch, original_weight, scores):
    """scores maps oversampled-bool -> (macro_f1, val_loss)."""
    x_train = torch.arange(_BASE_N * 4, dtype=torch.float32).reshape(_BASE_N, 4)
    monkeypatch.setattr(
        gt,
        "prepare_head_probe_tensors",
        lambda *a, **k: (x_train, _BASE_Y.clone(), x_train[:2], _BASE_Y[:2].clone()),
    )

    starts_clean: list[bool] = []
    seen_sizes: list[int] = []

    def fake_probe(model, x_tr, _y_tr, _x_val, _y_val, config, **_kwargs):
        starts_clean.append(torch.allclose(model.head.weight, original_weight))
        oversampled = x_tr.shape[0] > _BASE_N
        seen_sizes.append(int(x_tr.shape[0]))
        model.head.weight.data.fill_(1.0 if oversampled else 0.0)
        macro_f1, loss = scores[oversampled]
        return {
            "macro_f1": macro_f1,
            "qwk": None,
            "none_f1": macro_f1,
            "val_loss": loss,
            "best_epoch": 1,
            "epochs_completed": 1,
        }

    monkeypatch.setattr(gt, "train_head_probe_from_tensors", fake_probe)
    return starts_clean, seen_sizes


def _config(**kw) -> gt.GroupTrainConfig:
    return gt.GroupTrainConfig(
        group_folder="/tmp/group",
        num_classes=3,
        class_names=["a", "b", "__none__"],
        **kw,
    )


def test_oversample_sweep_selects_winner_and_sets_config(monkeypatch):
    model = TinyHeadModel()
    original_weight = model.head.weight.detach().clone()
    starts_clean, seen_sizes = _patch_probe(
        monkeypatch, original_weight, {False: (0.5, 0.4), True: (0.8, 0.3)},
    )
    config = _config()

    result = gt._run_auto_oversample_probe(
        model, config, object(), None, [], [],
        device=torch.device("cpu"), none_index=_NONE_INDEX,
        cb=lambda _msg: None, stop_event=None,
    )

    # Each candidate starts from the same (soft-label-selected) head state.
    assert starts_clean == [True, True]
    # The 1.5x candidate trained on the larger, oversampled tensor set.
    assert seen_sizes == [_BASE_N, 10]
    assert result["macro_f1"] == 0.8
    assert config.oversample_none is True
    assert config.selected_oversample_none is True
    assert len(config.oversample_tuning_results) == 2
    assert config.oversample_tuning_elapsed_ms is not None
    assert torch.all(model.head.weight == 1.0)  # best (oversampled) head loaded


def test_oversample_sweep_tiebreak_prefers_off(monkeypatch):
    model = TinyHeadModel()
    original_weight = model.head.weight.detach().clone()
    _patch_probe(monkeypatch, original_weight, {False: (0.7, 0.3), True: (0.7, 0.3)})
    config = _config()

    gt._run_auto_oversample_probe(
        model, config, object(), None, [], [],
        device=torch.device("cpu"), none_index=_NONE_INDEX,
        cb=lambda _msg: None, stop_event=None,
    )

    assert config.oversample_none is False
    assert config.selected_oversample_none is False
    assert torch.all(model.head.weight == 0.0)  # off head loaded


def test_oversample_sweep_skips_when_disabled(monkeypatch):
    model = TinyHeadModel()
    called = {"prep": 0}
    monkeypatch.setattr(
        gt, "prepare_head_probe_tensors",
        lambda *a, **k: called.__setitem__("prep", called["prep"] + 1) or (None,) * 4,
    )
    config = _config(auto_oversample_none=False)

    result = gt._run_auto_oversample_probe(
        model, config, object(), None, [], [],
        device=torch.device("cpu"), none_index=_NONE_INDEX,
        cb=lambda _msg: None, stop_event=None,
    )

    assert result == {}
    assert called["prep"] == 0
    assert config.selected_oversample_none is None


def test_oversample_sweep_skips_multilabel():
    model = TinyHeadModel()
    config = _config(multi_label=True)
    result = gt._run_auto_oversample_probe(
        model, config, object(), None, [], [],
        device=torch.device("cpu"), none_index=_NONE_INDEX,
        cb=lambda _msg: None, stop_event=None,
    )
    assert result == {}
    assert config.selected_oversample_none is None


def test_oversample_sweep_skips_without_none_class():
    model = TinyHeadModel()
    config = gt.GroupTrainConfig(
        group_folder="/tmp/group", num_classes=2, class_names=["a", "b"],
    )
    result = gt._run_auto_oversample_probe(
        model, config, object(), None, [], [],
        device=torch.device("cpu"), none_index=-1,
        cb=lambda _msg: None, stop_event=None,
    )
    assert result == {}


def test_build_oversampled_tensors_reaches_target():
    x = torch.arange(_BASE_N * 4, dtype=torch.float32).reshape(_BASE_N, 4)
    x_os, y_os = gt._build_oversampled_tensors(x, _BASE_Y.clone(), _NONE_INDEX)
    # target __none__ = ceil(1.5 * (max_count=2 * non_none_classes=2)) = 6.
    assert int((y_os == _NONE_INDEX).sum()) == 6
    assert y_os.shape[0] == 10
    # Non-__none__ rows are untouched (first _BASE_N rows preserved).
    assert torch.equal(x_os[:_BASE_N], x)
    # Appended rows are all __none__ feature rows drawn from the originals.
    appended = x_os[_BASE_N:]
    none_rows = x[_BASE_Y == _NONE_INDEX]
    assert all(any(torch.equal(r, nr) for nr in none_rows) for r in appended)


def test_build_oversampled_tensors_noop_without_none():
    y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    x = torch.zeros(4, 3)
    x_os, y_os = gt._build_oversampled_tensors(x, y, _NONE_INDEX)
    assert x_os.shape[0] == 4
    assert torch.equal(y_os, y)
