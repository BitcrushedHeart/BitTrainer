"""Calibration is fit on prior-adjusted logits (ISSUE-0490 A ordering).

_apply_and_persist_priors is the finalisation seam that runs BEFORE
_tune_softmax_calibration, so temperature / none-bias are fit on the exact
logits inference will see (raw -> prior adjustment -> temperature -> none bias).
"""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image

from bittrainer.group_trainer import (
    GroupTrainConfig,
    _apply_and_persist_priors,
    _compute_prior_vectors,
    _prior_logit_delta,
)
from bittrainer.group_dataset import GroupDataset


def _build(root, counts):
    for ci, (cname, n) in enumerate(counts.items()):
        d = root / cname / "train"
        d.mkdir(parents=True, exist_ok=True)
        for j in range(n):
            rng = np.random.default_rng(ci * 1000 + j)
            Image.fromarray(rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)).save(
                d / f"img_{ci}_{j}.png"
            )


def _cfg(**kw) -> GroupTrainConfig:
    base = dict(group_folder="/tmp/grp", num_classes=2, class_names=["a", "b"])
    base.update(kw)
    return GroupTrainConfig(**base)


def test_returned_logits_are_raw_plus_delta_and_priors_persisted(tmp_path):
    root = tmp_path / "g"
    _build(root, {"a": 60, "b": 5})  # imbalance => nonzero delta after 4x cap
    random.seed(0)
    ds = GroupDataset(root, ["a", "b"], split="train", group_name="g")
    cfg = _cfg(prior_tau=1.0)

    ckpt_path = str(tmp_path / "cand.pt")
    torch.save({"state_dict": {}, "num_classes": 2}, ckpt_path)

    raw = torch.tensor([[1.5, 1.4], [0.2, 3.0]], dtype=torch.float32)
    natural = ds.get_class_counts()
    effective = ds.get_effective_class_counts()
    adjusted = _apply_and_persist_priors(raw.clone(), natural, effective, cfg, ckpt_path)

    log_natural, log_effective = _compute_prior_vectors(ds, cfg)
    delta = _prior_logit_delta(log_natural, log_effective, 2, 1.0)
    expected = raw.float() + torch.tensor(delta, dtype=torch.float32)
    assert torch.allclose(adjusted, expected, atol=1e-6)

    # Genuinely shifts the logits calibration will receive.
    assert not torch.allclose(adjusted, raw.float())

    reloaded = torch.load(ckpt_path, weights_only=True)
    assert "class_log_prior_natural" in reloaded
    assert "class_log_prior_train_effective" in reloaded
    assert reloaded["prior_tau"] == 1.0


def test_multi_label_returns_logits_unchanged(tmp_path):
    root = tmp_path / "g"
    _build(root, {"a": 10, "b": 10})
    ds = GroupDataset(root, ["a", "b"], split="train", multi_label=True, group_name="g")
    cfg = _cfg(multi_label=True)
    raw = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    out = _apply_and_persist_priors(
        raw.clone(), ds.get_class_counts(), ds.get_effective_class_counts(), cfg, None
    )
    assert torch.allclose(out, raw)
