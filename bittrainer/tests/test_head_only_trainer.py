"""Integration tests for run_head_only_training (Function 1).

Runs the full path on CPU with caching off (single-process, no SmartCache
workers): prepare datasets -> warm-start model -> build+verify embedding cache
-> train probe -> evaluate candidate on val -> promote-if-better. Head-only is a
first-class training mode: a trained head that beats the current model becomes
the group's deployed model (best.pt); a worse one leaves the incumbent in place.
Covers linear and MLP probe heads, and the no-incumbent (fresh group) path.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from PIL import Image

from bittrainer.group_trainer import GroupTrainConfig
from bittrainer.head_only_trainer import run_head_only_training
from bittrainer.model import create_model, load_checkpoint

_CLASSES = ["a", "b", "c"]
_PROMOTED = {"higher_score", "no_incumbent", "class_mismatch", "eval_error"}


def _build_group(root):
    for ci, c in enumerate(_CLASSES):
        for split, n in (("train", 8), ("val", 4)):
            d = root / c / split
            d.mkdir(parents=True)
            for j in range(n):
                arr = np.random.default_rng(ci * 1000 + j + hash(split) % 50).integers(
                    0, 80, (96, 96, 3)
                ).astype(np.uint8)
                arr[..., ci] = np.clip(arr[..., ci] + 150, 0, 255)  # class-separable
                Image.fromarray(arr).save(d / f"i{j}.png")


def _seed_best(ckpt_dir):
    m = create_model(model_size="nano", pretrained=False, num_classes=3)
    path = ckpt_dir / "best.pt"
    torch.save({"state_dict": m.state_dict(), "num_classes": 3,
                "model_size": "nano", "class_names": _CLASSES}, path)
    return hashlib.md5(path.read_bytes()).hexdigest()


def _cfg(root, ckpt_dir, embed_dir, probe_head):
    return GroupTrainConfig(
        group_folder=str(root), num_classes=3, class_names=_CLASSES,
        device="cpu", dtype="float32", use_cache=False,
        checkpoint_dir=str(ckpt_dir), embedding_cache_dir=str(embed_dir),
        probe_head=probe_head, head_max_epochs=20, head_patience=6,
    )


def test_head_only_promotes_over_random_incumbent(tmp_path):
    root = tmp_path / "grp"
    _build_group(root)
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    before = _seed_best(ckpt_dir)

    res = run_head_only_training(_cfg(root, ckpt_dir, tmp_path / "embed", "linear"))

    assert res["mode"] == "head_only"
    assert len(res["per_class_f1"]) == 3
    # the winner is always returned as best.pt, whether promoted or kept
    assert res["checkpoint_path"].endswith("best.pt")
    # a trained head beats the seeded random-head incumbent -> it deploys
    assert res["promotion_reason"] in _PROMOTED
    assert hashlib.md5((ckpt_dir / "best.pt").read_bytes()).hexdigest() != before
    # the deployed model reloads via the standard inference path
    m = load_checkpoint(res["checkpoint_path"], device="cpu", num_classes=3, model_size="nano")
    assert "head.pre_logits.fc.weight" not in m.state_dict()
    # the candidate scratch file is cleaned up
    assert not (ckpt_dir / "candidate.pt").exists()


def test_head_only_mlp_fresh_group(tmp_path):
    # No incumbent: head-only training of a fresh group produces the group model.
    root = tmp_path / "grp"
    _build_group(root)
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()

    res = run_head_only_training(_cfg(root, ckpt_dir, tmp_path / "embed", "mlp"))

    assert res["probe_head"] == "mlp"
    assert res["promotion_reason"] == "no_incumbent"
    assert res["checkpoint_path"].endswith("best.pt")
    m = load_checkpoint(res["checkpoint_path"], device="cpu", num_classes=3, model_size="nano")
    assert m.state_dict()["head.pre_logits.fc.weight"].shape[0] == 512


def test_head_only_reuses_cache_across_runs(tmp_path):
    root = tmp_path / "grp"
    _build_group(root)
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    _seed_best(ckpt_dir)
    embed_dir = tmp_path / "embed"

    run_head_only_training(_cfg(root, ckpt_dir, embed_dir, "linear"))
    res2 = run_head_only_training(_cfg(root, ckpt_dir, embed_dir, "linear"))
    # backbone (and head.norm) unchanged across runs -> same era, vectors reused
    assert res2["embedding_cache_stats"]["built"] == 0
    assert res2["embedding_cache_stats"]["reused"] > 0
