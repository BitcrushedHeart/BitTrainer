"""A yes/no (concept) run asked to auto-promote ships its model whatever the
incumbent scores (Bitcrush ISSUE-1732): the model it replaces sits on a trunk
Engine no longer starts from, so a higher score does not keep it.
"""

from __future__ import annotations

from types import SimpleNamespace

import bittrainer.trainer as bt


class _Dataset:
    _all_negative_paths: list = []


def _promote(tmp_path, monkeypatch, *, auto_promote: bool):
    existing = tmp_path / "best.pt"
    existing.write_bytes(b"old")
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"new")
    loaded = SimpleNamespace(load_state_dict=lambda _state: None)
    monkeypatch.setattr(bt.torch, "load", lambda *_a, **_k: {"state_dict": {}})
    monkeypatch.setattr(bt, "create_model", lambda **_k: SimpleNamespace(to=lambda _d: loaded))
    monkeypatch.setattr(bt, "evaluate", lambda *_a, **_k: {"labels": [], "probs": []})
    # The incumbent scores far above the candidate's 0.5.
    monkeypatch.setattr(bt, "_tuned_val_metrics", lambda _result: ({"f1": 0.9}, 0.4))
    monkeypatch.setattr(bt, "find_optimal_threshold", lambda _labels, _probs: 0.3)
    result = bt._binary_compare_promote(
        SimpleNamespace(model_size="atto", auto_promote=auto_promote),
        best_checkpoint_path=str(candidate),
        existing_best=existing,
        model=loaded,
        val_loader=None,
        criterion=None,
        device="cpu",
        dtype=None,
        best_val_f1=0.5,
        best_metrics={"f1": 0.5},
        best_epoch=0,
        epochs_completed=1,
        num_positives=1,
        train_ds=_Dataset(),
    )
    return result, existing


def test_auto_promote_replaces_a_better_scoring_incumbent(tmp_path, monkeypatch):
    result, existing = _promote(tmp_path, monkeypatch, auto_promote=True)
    assert existing.read_bytes() == b"new"
    assert result["best_val_f1"] == 0.5
    assert result["checkpoint_path"] == str(existing)


def test_without_it_the_better_incumbent_stays(tmp_path, monkeypatch):
    result, existing = _promote(tmp_path, monkeypatch, auto_promote=False)
    assert existing.read_bytes() == b"old"
    assert result["best_val_f1"] == 0.9


def test_train_config_takes_auto_promote():
    assert bt.TrainConfig(concept_folder="x").auto_promote is False
    assert bt.TrainConfig(concept_folder="x", auto_promote=True).auto_promote is True
