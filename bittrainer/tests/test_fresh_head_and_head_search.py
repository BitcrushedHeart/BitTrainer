"""Fresh heads on a kept trunk, and the softmax-vs-threshold head search (Bitcrush ISSUE-1496).

Engine's "Train Head" keeps the live model's trunk (a fine-tuned one included)
but trains the classifier tail (``head.pre_logits`` + ``head.fc``) from scratch;
on an ordinal group it trains a softmax and an all-threshold tail on the same
cached features and keeps the better. "Finetune Head" keeps warm-starting the
incumbent's head exactly as before.

CPU-only: random-init ConvNeXt; no download.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

import bittrainer.group_trainer as gt
from bittrainer.cached_eval import CachedValScorer
from bittrainer.group_trainer import GroupTrainConfig
from bittrainer.head_only_trainer import run_head_only_training
from bittrainer.model import (
    backbone_feature_hash,
    create_model,
    head_tail_logits,
    load_checkpoint,
    pooled_features,
    trunk_state,
)
from bittrainer.ordinal_threshold import (
    SOFTMAX_HEAD,
    THRESHOLD_HEAD,
    has_threshold_head,
    install_ordinal_threshold_head,
)

_CLASSES = ["low", "mid", "high"]


def _save_incumbent(ckpt_dir, *, seed=7, threshold=False, size="atto"):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = create_model(model_size=size, pretrained=False, num_classes=3)
    meta = {"model_size": size, "num_classes": 3, "class_names": list(_CLASSES)}
    if threshold:
        install_ordinal_threshold_head(model, 3, none_index=-1)
        meta["ordinal_head"] = THRESHOLD_HEAD
    meta["state_dict"] = model.state_dict()
    torch.save(meta, ckpt_dir / "best.pt")
    return model


def _group_cfg(tmp_path, **kw):
    base = dict(
        group_folder=str(tmp_path),
        num_classes=3,
        class_names=list(_CLASSES),
        backbone_variant="atto",
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    base.update(kw)
    return GroupTrainConfig(**base)


def _build(config, tmp_path):
    return gt._create_or_warmstart_model(
        config,
        device=torch.device("cpu"),
        dtype=torch.float32,
        head_hidden_size=None,
        checkpoint_dir=tmp_path / "ckpt",
    )


# --------------------------------------------------------------------------- #
# trunk_state + the group warm start
# --------------------------------------------------------------------------- #


def test_trunk_state_drops_only_the_head_tail():
    state = {
        "stem.0.weight": torch.zeros(1),
        "head.norm.weight": torch.zeros(1),
        "head.fc.weight": torch.zeros(1),
        "head.fc.threshold_fc.weight": torch.zeros(1),
        "head.pre_logits.fc.weight": torch.zeros(1),
        "region_head.conv.weight": torch.zeros(1),
    }
    assert sorted(trunk_state(state)) == [
        "head.norm.weight",
        "region_head.conv.weight",
        "stem.0.weight",
    ]


def test_fresh_head_keeps_the_trunk_and_starts_the_tail_fresh(tmp_path):
    incumbent = _save_incumbent(tmp_path / "ckpt")
    model = _build(_group_cfg(tmp_path, fresh_head=True), tmp_path)

    assert torch.equal(model.stem[0].weight, incumbent.stem[0].weight)
    assert torch.equal(model.head.norm.weight, incumbent.head.norm.weight)
    assert not torch.equal(model.head.fc.weight, incumbent.head.fc.weight)


def test_without_fresh_head_the_incumbents_head_carries_over(tmp_path):
    incumbent = _save_incumbent(tmp_path / "ckpt")
    model = _build(_group_cfg(tmp_path), tmp_path)

    assert torch.equal(model.head.fc.weight, incumbent.head.fc.weight)


def test_fresh_head_from_a_threshold_incumbent_builds_the_configured_head(tmp_path):
    incumbent = _save_incumbent(tmp_path / "ckpt", threshold=True)
    model = _build(_group_cfg(tmp_path, ordinal=True, fresh_head=True), tmp_path)

    assert not has_threshold_head(model)
    assert torch.equal(model.stem[0].weight, incumbent.stem[0].weight)


# --------------------------------------------------------------------------- #
# head choice
# --------------------------------------------------------------------------- #


def test_pick_head_takes_the_higher_score_and_softmax_on_a_tie():
    from bittrainer.generic.tasks.head_only_task import pick_head

    softmax = {"head": SOFTMAX_HEAD, "score": 0.5}
    threshold = {"head": THRESHOLD_HEAD, "score": 0.6}
    assert pick_head([softmax, threshold]) is threshold
    assert pick_head([softmax, {"head": THRESHOLD_HEAD, "score": 0.5}]) is softmax
    assert pick_head([{"head": THRESHOLD_HEAD, "score": 0.4}, softmax]) is softmax


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"ordinal": False}, False),
        ({"ordinal_head_search": False}, False),
        ({"multi_label": True}, False),
        ({"cell_masks": [[0], [1], [2]]}, False),
        ({"use_focal": True}, False),
    ],
)
def test_head_search_applies_only_where_the_threshold_head_can_train(overrides, expected):
    from bittrainer.generic.tasks.head_only_task import head_search_enabled

    config = GroupTrainConfig(
        group_folder="x",
        num_classes=3,
        class_names=list(_CLASSES),
        ordinal=True,
        ordinal_head_search=True,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    assert head_search_enabled(config) is expected


# --------------------------------------------------------------------------- #
# the cached scorer serves an incumbent with the other head
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("incumbent_threshold", [True, False])
def test_cached_scorer_serves_an_incumbent_with_the_other_head(incumbent_threshold):
    torch.manual_seed(3)
    run_model = create_model(model_size="atto", pretrained=False, num_classes=3).eval()
    if not incumbent_threshold:
        install_ordinal_threshold_head(run_model, 3, none_index=-1)
    incumbent = create_model(model_size="atto", pretrained=False, num_classes=3).eval()
    incumbent.load_state_dict(
        {k: v for k, v in run_model.state_dict().items() if not k.startswith("head.fc.")},
        strict=False,
    )
    if incumbent_threshold:
        install_ordinal_threshold_head(incumbent, 3, none_index=-1)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        feats = pooled_features(run_model, x)
    scorer = CachedValScorer(run_model, feats, backbone_feature_hash(run_model))

    head_model = scorer.model_for_state({"state_dict": incumbent.state_dict()})

    assert head_model is not None
    assert has_threshold_head(head_model) is incumbent_threshold
    with torch.no_grad():
        torch.testing.assert_close(
            head_tail_logits(head_model, feats), head_tail_logits(incumbent, feats)
        )


# --------------------------------------------------------------------------- #
# end to end: the head-only run
# --------------------------------------------------------------------------- #


def _build_group(root):
    for ci, c in enumerate(_CLASSES):
        for split, n in (("train", 8), ("val", 4)):
            d = root / c / split
            d.mkdir(parents=True)
            for j in range(n):
                arr = np.random.default_rng(ci * 1000 + j + (7 if split == "val" else 0)).integers(
                    0, 80, (96, 96, 3)
                ).astype(np.uint8)
                arr[..., ci] = np.clip(arr[..., ci] + 150, 0, 255)  # class-separable
                Image.fromarray(arr).save(d / f"i{j}.png")


def _run_cfg(root, ckpt_dir, embed_dir, **kw):
    base = dict(
        group_folder=str(root),
        num_classes=3,
        class_names=list(_CLASSES),
        backbone_variant="atto",
        device="cpu",
        dtype="float32",
        use_cache=False,
        checkpoint_dir=str(ckpt_dir),
        embedding_cache_dir=str(embed_dir),
        head_max_epochs=8,
        head_patience=4,
        auto_label_softness=False,
        auto_oversample_none=False,
        backbone_init={"source": "random_init", "checkpoint_path": None},
    )
    base.update(kw)
    return GroupTrainConfig(**base)


def test_head_search_trains_both_heads_and_ships_the_winner(tmp_path):
    root = tmp_path / "grp"
    _build_group(root)
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()

    res = run_head_only_training(
        _run_cfg(
            root,
            ckpt_dir,
            tmp_path / "embed",
            ordinal=True,
            ordinal_head_search=True,
            fresh_head=True,
        )
    )

    rows = res["head_search"]
    assert [row["head"] for row in rows] == [SOFTMAX_HEAD, THRESHOLD_HEAD]
    assert all(isinstance(row["score"], float) for row in rows)
    winner = next(row["head"] for row in rows if row["selected"])
    assert res["promotion_reason"] == "no_incumbent"
    assert res["ordinal_head"] == winner
    shipped = load_checkpoint(res["checkpoint_path"], device="cpu")
    assert has_threshold_head(shipped) is (winner == THRESHOLD_HEAD)


def test_head_search_off_reports_no_search(tmp_path):
    root = tmp_path / "grp"
    _build_group(root)
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()

    res = run_head_only_training(_run_cfg(root, ckpt_dir, tmp_path / "embed", ordinal=True))

    assert "head_search" not in res


def test_fresh_head_run_keeps_a_fine_tuned_trunk(tmp_path):
    root = tmp_path / "grp"
    _build_group(root)
    ckpt_dir = tmp_path / "ckpts"
    incumbent = _save_incumbent(ckpt_dir, seed=11)

    res = run_head_only_training(
        _run_cfg(root, ckpt_dir, tmp_path / "embed", fresh_head=True, auto_promote=True)
    )

    shipped = load_checkpoint(res["checkpoint_path"], device="cpu")
    assert torch.equal(shipped.stem[0].weight, incumbent.stem[0].weight)


# --------------------------------------------------------------------------- #
# binary concepts
# --------------------------------------------------------------------------- #


def _binary_task(tmp_path, **kw):
    from bittrainer.generic.tasks.binary_task import BinaryTask
    from bittrainer.trainer import TrainConfig

    config = TrainConfig(
        concept_folder=str(tmp_path),
        checkpoint_dir=str(tmp_path / "ckpt"),
        model_size="atto",
        device="cpu",
        dtype="float32",
        training_mode="head_only",
        backbone_init={"source": "random_init", "checkpoint_path": None},
        **kw,
    )
    task = BinaryTask(config)
    task.num_positives = 100
    return task


def _binary_ctx(tmp_path):
    from bittrainer.generic.task import TaskContext

    return TaskContext(
        device=torch.device("cpu"),
        dtype=torch.float32,
        em=None,
        cb=lambda _msg: None,
        checkpoint_dir=tmp_path / "ckpt",
    )


def _save_binary_incumbent(tmp_path):
    (tmp_path / "ckpt").mkdir()
    torch.manual_seed(5)
    model = create_model(model_size="atto", pretrained=False)
    torch.save(
        {"state_dict": model.state_dict(), "model_size": "atto"},
        tmp_path / "ckpt" / "best.pt",
    )
    return model


def test_binary_fresh_head_keeps_the_trunk_and_starts_the_tail_fresh(tmp_path):
    incumbent = _save_binary_incumbent(tmp_path)

    model = _binary_task(tmp_path, fresh_head=True).create_model(_binary_ctx(tmp_path), None)

    assert torch.equal(model.stem[0].weight, incumbent.stem[0].weight)
    assert not torch.equal(model.head.fc.weight, incumbent.head.fc.weight)


def test_binary_warm_start_keeps_the_incumbents_head(tmp_path):
    incumbent = _save_binary_incumbent(tmp_path)

    model = _binary_task(tmp_path).create_model(_binary_ctx(tmp_path), None)

    assert torch.equal(model.head.fc.weight, incumbent.head.fc.weight)
