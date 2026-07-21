"""Effective + natural prior vectors and their landing in checkpoint meta.

Balanced oversampling (4x cap) makes the model's effective class exposure far
flatter than the natural stream; prior correction needs BOTH distributions to
undo the shift at decode. CPU-only (reads image sizes, builds sample lists).
"""

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np
import torch
from PIL import Image

from bittrainer.group_dataset import GroupDataset, compute_class_log_priors
from bittrainer.group_trainer import (
    GroupTrainConfig,
    _compute_prior_vectors,
    _persist_class_priors,
)


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
    base = dict(group_folder="/tmp/grp", num_classes=3, class_names=["a", "b", "c"])
    base.update(kw)
    return GroupTrainConfig(**base)


def test_effective_reflects_oversample_natural_reflects_raw(tmp_path):
    root = tmp_path / "g"
    # 12:1 ratio; class "c" empty (no folder) to exercise Laplace smoothing.
    _build(root, {"a": 60, "b": 5})
    random.seed(0)
    ds = GroupDataset(root, ["a", "b", "c"], split="train", group_name="g")

    natural = {int(k): int(v) for k, v in ds.get_class_counts().items()}
    effective = dict(Counter(s["label"] for s in ds.samples))

    # Natural = raw disk counts, un-oversampled.
    assert natural[0] == 60
    assert natural[1] == 5
    assert natural.get(2, 0) == 0

    # Effective: minority "b" capped at 4x its natural size (20), not equalised
    # to 60, so exposure is flatter than natural but not uniform.
    assert effective[0] == 60
    assert effective[1] == 20
    assert 2 not in effective


def test_log_priors_finite_for_empty_class(tmp_path):
    counts = {0: 60, 1: 20}  # class 2 empty
    log_priors = compute_class_log_priors(counts, num_classes=3)
    assert set(log_priors) == {"0", "1", "2"}
    assert all(math.isfinite(v) for v in log_priors.values())
    # Laplace: class 2 == log(1 / (61 + 21 + 1)).
    assert log_priors["2"] == math.log(1.0 / (61 + 21 + 1))


def test_vectors_land_in_checkpoint_meta(tmp_path):
    root = tmp_path / "g"
    _build(root, {"a": 60, "b": 5})
    random.seed(0)
    ds = GroupDataset(root, ["a", "b", "c"], split="train", group_name="g")
    cfg = _cfg(prior_tau=1.0)

    vectors = _compute_prior_vectors(ds, cfg)
    assert vectors is not None
    log_natural, log_effective = vectors

    ckpt_path = str(tmp_path / "cand.pt")
    torch.save({"state_dict": {}, "num_classes": 3}, ckpt_path)
    _persist_class_priors(
        ckpt_path, log_natural=log_natural, log_effective=log_effective, tau=cfg.prior_tau
    )

    reloaded = torch.load(ckpt_path, weights_only=True)
    assert reloaded["prior_tau"] == 1.0
    assert set(reloaded["class_log_prior_natural"]) == {"0", "1", "2"}
    assert set(reloaded["class_log_prior_train_effective"]) == {"0", "1", "2"}
    # Rare class "b" is up-weighted in effective vs natural, so the correction
    # log(natural) - log(effective) is negative for it (pushes it back down).
    delta_b = reloaded["class_log_prior_natural"]["1"] - reloaded[
        "class_log_prior_train_effective"
    ]["1"]
    assert delta_b < 0


def test_multi_label_has_no_priors(tmp_path):
    root = tmp_path / "g"
    _build(root, {"a": 10, "b": 10})
    ds = GroupDataset(root, ["a", "b"], split="train", multi_label=True, group_name="g")
    cfg = _cfg(num_classes=2, class_names=["a", "b"], multi_label=True)
    assert _compute_prior_vectors(ds, cfg) is None
