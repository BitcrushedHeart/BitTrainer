"""The embedding-cache era names everything that shapes the embedded pixels
(Bitcrush ISSUE-1110).

Before: the era was ``{backbone_hash}{sig(train_resolution)}`` only, so a group
that turned face crop (or skin normalisation) on or off silently reused vectors
of the OTHER crop. That mattered once backbone-comparison candidates began to
share the group's cache: a candidate may only reuse vectors embedded from the
same pixels. Runs without crop / skin normalisation keep the historical sig, so
existing caches stay valid.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

import bittrainer.group_trainer as gt
from bittrainer.embedding_cache import EmbeddingCache
from bittrainer.smart_cache import region_signature


def _cfg(**kw):
    return gt.GroupTrainConfig(group_folder="/tmp/g", num_classes=2, class_names=["a", "b"],
                               **kw)


def test_plain_runs_keep_the_historical_sig():
    assert gt._group_embedding_preproc_sig(_cfg()) == "val_imagenet"
    assert gt._group_embedding_preproc_sig(_cfg(train_resolution=384)) == "val_imagenet@384"


def test_face_crop_region_crop_and_skin_normalise_are_identity(tmp_path):
    face = tmp_path / "BMDSeg_Face_s.pt"
    face.write_bytes(b"x")
    region = tmp_path / "BMD_hand.pt"
    region.write_bytes(b"y")
    base = gt._group_embedding_preproc_sig(_cfg(train_resolution=384))
    crop = gt._group_embedding_preproc_sig(_cfg(train_resolution=384,
                                                face_model_path=str(face)))
    assert crop != base and crop.startswith(base)
    assert region_signature(str(face)) in crop

    region_a = gt._group_embedding_preproc_sig(_cfg(
        region_model_path=str(region), region_classes=["hand"], region_selection="highest_conf",
    ))
    region_b = gt._group_embedding_preproc_sig(_cfg(
        region_model_path=str(region), region_classes=["hand", "arm"],
        region_selection="highest_conf",
    ))
    assert region_a != region_b

    skin = gt._group_embedding_preproc_sig(_cfg(skin_normalise=True))
    assert skin != "val_imagenet"

    # A new detector file (mtime) is a new crop.
    os.utime(face, (1_000_000, 1_000_000))
    later = gt._group_embedding_preproc_sig(_cfg(train_resolution=384,
                                                 face_model_path=str(face)))
    assert later != crop


def test_crop_and_no_crop_runs_get_different_eras(tmp_path):
    face = tmp_path / "face.pt"
    face.write_bytes(b"x")
    a = EmbeddingCache(tmp_path / "e", "0123456789abcdef", 8,
                       preproc_sig=gt._group_embedding_preproc_sig(_cfg(train_resolution=384)))
    b = EmbeddingCache(tmp_path / "e", "0123456789abcdef", 8,
                       preproc_sig=gt._group_embedding_preproc_sig(
                           _cfg(train_resolution=384, face_model_path=str(face))))
    assert a.root != b.root


def _build_group(root):
    for ci, c in enumerate(["a", "b"]):
        for split, n in (("train", 4), ("val", 2)):
            d = root / c / split
            d.mkdir(parents=True)
            for j in range(n):
                arr = np.random.default_rng(ci * 10 + j).integers(0, 255, (64, 64, 3))
                Image.fromarray(arr.astype(np.uint8)).save(d / f"i{j}.png")


def test_prune_false_keeps_the_other_eras(tmp_path, monkeypatch):
    """A comparison candidate (``embedding_cache_prune=False``) shares the
    group's cache root without deleting the group's own era."""
    from bittrainer.head_only_trainer import run_head_only_training
    from bittrainer.tests.siglip_fixtures import install_tiny_trunk

    install_tiny_trunk(monkeypatch)
    root = tmp_path / "grp"
    _build_group(root)
    embed = tmp_path / "embed"
    foreign = embed / "fedcba9876543210"
    foreign.mkdir(parents=True)
    (foreign / "meta.json").write_text("{}")

    def _run(prune):
        ckpt = tmp_path / f"ck_{prune}"
        ckpt.mkdir()
        return run_head_only_training(gt.GroupTrainConfig(
            group_folder=str(root), num_classes=2, class_names=["a", "b"],
            backbone_variant="siglip2_base", device="cpu", dtype="float32",
            use_cache=False, checkpoint_dir=str(ckpt), embedding_cache_dir=str(embed),
            head_max_epochs=2, head_patience=1, train_resolution=128,
            embedding_cache_prune=prune,
        ))

    _run(False)
    assert foreign.is_dir(), "a non-pruning run deleted another era"
    _run(True)
    assert not foreign.exists(), "the group's own run still prunes stale eras"
