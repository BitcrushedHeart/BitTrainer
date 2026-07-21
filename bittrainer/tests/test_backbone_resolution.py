"""Backbone low-res era + high-res finetune tail (Bitcrush trainer-efficiency round).

``finetune_image_size`` + ``finetune_epochs`` switch the LAST N epochs (train
AND validation) to a higher resolution. At the switch the best tracker resets:
low-res and high-res validation scores are not comparable, so the tail always
re-wins and the exported candidate is selected at the deployment resolution.
Prodigy/EMA/cosine continue uninterrupted across the switch. Absent config =
legacy single-resolution behaviour, including the fingerprint's resolution key
staying stable for a given config.
"""

from __future__ import annotations

import asyncio

import torch
from safetensors.torch import load_file

import bittrainer.backbone_trainer as bb
import bittrainer.generic.tasks.backbone_task as bt
from bittrainer.backbone_trainer import run_backbone_training
from bittrainer.tests.test_backbone_generic import _request


def _run(request):
    return asyncio.run(run_backbone_training(request))


def _spy_transforms(monkeypatch):
    calls = {"train": [], "val": []}
    real_train, real_val = bb._train_transform, bb._val_transform

    def _train(size):
        calls["train"].append(size)
        return real_train(size)

    def _val(size):
        calls["val"].append(size)
        return real_val(size)

    monkeypatch.setattr(bb, "_train_transform", _train)
    monkeypatch.setattr(bb, "_val_transform", _val)
    return calls


def test_tail_switches_train_and_val_size(tmp_path, monkeypatch):
    calls = _spy_transforms(monkeypatch)
    request = _request(tmp_path, epochs=3, max_steps=1000, n=12)
    request["training_config"].update(
        {"image_size": 48, "finetune_image_size": 64, "finetune_epochs": 1}
    )
    _run(request)
    assert calls["train"] == [48, 48, 64]
    # Val loader is built lazily at epoch 0 and rebuilt once at the switch.
    assert calls["val"] == [48, 64]


def test_no_tail_config_is_legacy_single_resolution(tmp_path, monkeypatch):
    calls = _spy_transforms(monkeypatch)
    request = _request(tmp_path, epochs=2, max_steps=1000, n=12)
    request["training_config"]["image_size"] = 48
    _run(request)
    assert calls["train"] == [48, 48]
    assert calls["val"] == [48]


def test_tail_resets_best_and_exports_tail_candidate(tmp_path, monkeypatch):
    """Pre-tail epoch scores 0.9, tail epoch 0.1 — without the reset the
    low-res epoch would win; with it the tail always re-wins."""
    scores = iter([{"binary/watermark": 0.9}, {"binary/watermark": 0.1}])
    monkeypatch.setattr(bb, "_evaluate", lambda *a, **k: next(scores))

    snapshots = []
    real_save = bt.BackboneTask.save_candidate

    def _spy_save(self, ctx, model, epoch, metrics, best):
        real_save(self, ctx, model, epoch, metrics, best)
        snapshots.append({k: v.clone() for k, v in (self.best_heads_state or {}).items()})

    monkeypatch.setattr(bt.BackboneTask, "save_candidate", _spy_save)

    request = _request(tmp_path, epochs=2, max_steps=1000, n=12)
    request["training_config"].update(
        {"image_size": 48, "finetune_image_size": 64, "finetune_epochs": 1}
    )
    result = _run(request)
    # Both epochs snapshot: epoch 1 improves vs -1, then the reset lets the
    # tail epoch improve vs -1 again.
    assert len(snapshots) == 2
    assert result["best_epoch"] == 2
    saved = load_file(result["candidate_checkpoint_path"])
    for key, tensor in snapshots[1].items():
        assert torch.equal(saved[f"heads.{key}"], tensor)


def test_without_tail_best_epoch_stays_first(tmp_path, monkeypatch):
    """Control for the reset test: same scores, no tail config -> epoch 1 wins."""
    scores = iter([{"binary/watermark": 0.9}, {"binary/watermark": 0.1}])
    monkeypatch.setattr(bb, "_evaluate", lambda *a, **k: next(scores))
    request = _request(tmp_path, epochs=2, max_steps=1000, n=12)
    result = _run(request)
    assert result["best_epoch"] == 1


def test_fingerprint_carries_resolution():
    vocab = bb._Vocab([{"binary": {"c": "positive"}}, {"binary": {"c": "negative"}}])
    plain = bb._backbone_fingerprint(vocab, "atto", 3, resolution="384")
    tailed = bb._backbone_fingerprint(vocab, "atto", 3, resolution="256->384@2")
    assert plain["resolution"] == "384"
    assert tailed["resolution"] == "256->384@2"
    assert plain != tailed
