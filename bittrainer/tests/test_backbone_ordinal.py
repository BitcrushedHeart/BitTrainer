"""Ordinal groups train as ordinal in the backbone (Bitcrush ISSUE-1721).

A pack's manifest may name its ordinal groups with their class order
(``"ordinal": {"g6": {"order": ["k3", "k10", ...], "sigma": 1.0}}``). Those
groups then train against Gaussian soft targets over that order, the same idea
as the group trainer's ordinal smoothing, with ``__none__`` (any class not in
the order) kept off the scale. Validation reports QWK and mean error for them;
the macro F1 and the selection score are what they were, so a run that resumes
with the change keeps its best score.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

import bittrainer.backbone_trainer as bb
from bittrainer.backbone_pack import PackError, pack_ordinal

# Classes sort as strings in the vocab ("k10" < "k2"), not in their order.
_ORDER = ["k2", "k3", "k10"]


def _vocab() -> bb._Vocab:
    records = [
        {"groups": {"g6": name}, "binary": {}} for name in ("k10", "k2", "k3", "k99")
    ]
    vocab = bb._Vocab(records)
    assert vocab.groups["g6"] == ["k10", "k2", "k3", "k99"]  # k99 = the none class
    return vocab


def _write_manifest(tmp_path, extra: dict) -> None:
    manifest = {"format": 2, "images": 0, "home_validation": {"answers": 0, "images": 0}}
    manifest.update(extra)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


# -- the pack ------------------------------------------------------------------

def test_the_manifest_names_its_ordinal_groups(tmp_path):
    _write_manifest(tmp_path, {"ordinal": {"g6": {"order": _ORDER, "sigma": 1.5}}})
    assert pack_ordinal(tmp_path) == {"g6": {"order": _ORDER, "sigma": 1.5}}


def test_a_pack_without_the_key_has_no_ordinal_groups(tmp_path):
    _write_manifest(tmp_path, {})
    assert pack_ordinal(tmp_path) is None


def test_a_malformed_ordinal_entry_is_refused(tmp_path):
    _write_manifest(tmp_path, {"ordinal": {"g6": ["k2", "k3"]}})
    with pytest.raises(PackError):
        pack_ordinal(tmp_path)


# -- the soft targets -------------------------------------------------------------

def test_ranks_follow_the_order_not_the_sorted_vocab():
    vocab = _vocab()
    bb._attach_ordinal(vocab, {"g6": {"order": _ORDER, "sigma": 1.0}})
    # vocab order k10, k2, k3, k99 -> ranks 2, 0, 1, off the scale
    assert vocab.ordinal["g6"] == [2, 0, 1, None]


def test_groups_not_in_the_vocab_or_with_one_ranked_class_are_ignored():
    vocab = _vocab()
    bb._attach_ordinal(
        vocab, {"g7": {"order": _ORDER, "sigma": 1.0}, "g6": {"order": ["k2"], "sigma": 1.0}}
    )
    assert vocab.ordinal == {}


def test_the_kernel_spreads_over_neighbours_and_never_onto_none():
    vocab = _vocab()
    bb._attach_ordinal(vocab, {"g6": {"order": _ORDER, "sigma": 1.0}})
    kernel = bb._ordinal_kernels(vocab, torch.device("cpu"))["g6"]
    k10, k2, k3, none = 0, 1, 2, 3
    assert torch.allclose(kernel.sum(1), torch.ones(4))
    # A k2 answer: most on k2, then its neighbour k3, then k10 two steps away.
    assert kernel[k2, k2] > kernel[k2, k3] > kernel[k2, k10] > 0
    assert kernel[k2, none] == 0 and kernel[none, k2] == 0
    assert kernel[none, none] == 1
    weight = [math.exp(-(d * d) / 2) for d in (0, 1, 2)]
    assert kernel[k2, k3].item() == pytest.approx(weight[1] / sum(weight), abs=1e-6)


def test_a_sigma_of_zero_is_a_one_hot_target(tmp_path):
    _write_manifest(tmp_path, {"ordinal": {"g6": {"order": _ORDER, "sigma": 0.0}}})
    spec = pack_ordinal(tmp_path)
    assert spec["g6"]["sigma"] == 0.0
    vocab = _vocab()
    bb._attach_ordinal(vocab, spec)
    assert torch.equal(bb._ordinal_kernels(vocab, torch.device("cpu"))["g6"], torch.eye(4))


def test_ordinal_groups_train_on_soft_targets_and_others_stay_plain():
    vocab = _vocab()
    bb._attach_ordinal(vocab, {"g6": {"order": _ORDER, "sigma": 1.0}})
    soft = bb._ordinal_kernels(vocab, torch.device("cpu"))
    torch.manual_seed(0)
    heads = bb._MultiTaskHeads(8, vocab)
    features = torch.randn(3, 8)
    labels = [{"g6": 1}, {"g6": 2}, {"g6": 3}]
    loss = bb._batch_loss(features, heads, [{}] * 3, labels, torch.device("cpu"),
                          group_soft_targets=soft)
    log_probs = torch.log_softmax(heads.groups["g6"](features).float(), dim=1)
    expected = -(soft["g6"][torch.tensor([1, 2, 3])] * log_probs).sum(1).mean()
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)
    plain = bb._batch_loss(features, heads, [{}] * 3, labels, torch.device("cpu"))
    assert plain.item() == pytest.approx(
        torch.nn.functional.cross_entropy(
            heads.groups["g6"](features), torch.tensor([1, 2, 3])
        ).item(),
        rel=1e-5,
    )


# -- validation ---------------------------------------------------------------------

def test_validation_adds_qwk_and_mean_error_and_keeps_f1_and_selection():
    # vocab order k10, k2, k3, none; ranks 2, 0, 1, None
    targets = np.array([1, 2, 0, 1, 3])
    preds = np.array([1, 0, 0, 2, 3])
    logits = np.full((5, 4), -5.0)
    logits[np.arange(5), preds] = 5.0
    scores = {"g6": (logits, targets)}
    plain = bb._backbone_metrics({}, scores)
    ranked = bb._backbone_metrics({}, scores, ordinal={"g6": [2, 0, 1, None]})
    assert ranked["group_f1/g6"] == plain["group_f1/g6"]
    assert ranked["selection"] == plain["selection"]
    # On the scale: ranks (true, pred) = (0,0) (1,0) (2,2) (0,1); none row left out.
    assert ranked["group_mae/g6"] == pytest.approx(0.5)
    assert "group_qwk/g6" in ranked and -1.0 <= ranked["group_qwk/g6"] <= 1.0
    assert "group_qwk/g6" not in plain
