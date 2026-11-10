"""Concept heads in one matrix product, not one call per head (Bitcrush ISSUE-1655).

``_batch_loss`` and ``_collect_val_scores`` used to visit every concept head in
Python: a row scan, an index, a Linear and a loss per head and per batch, and in
validation a GPU-to-CPU copy per head as well. With several hundred concepts
that is thousands of tiny calls per batch. Both now run the concept heads as a
single matrix product.

The old loops are kept here, verbatim, as the reference: the new code must give
the same loss, the same gradients, the same heads left untouched, and the same
validation scores in the same order.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import bittrainer.backbone_trainer as bb

CPU = torch.device("cpu")
FEATURES = 24


# -- the old per-head loops, as the reference --------------------------------


def _reference_batch_loss(
    features,
    heads,
    binary_labels,
    group_labels,
    device,
    *,
    pos_weight=None,
    group_label_smoothing=0.0,
    reduction="sum",
):
    losses = []
    bce = nn.functional.binary_cross_entropy_with_logits
    ce = nn.functional.cross_entropy
    for concept, head in heads.binary.items():
        rows = [i for i, labels in enumerate(binary_labels) if concept in labels]
        if not rows:
            continue
        logits = head(features[rows]).squeeze(-1)
        targets = torch.tensor(
            [binary_labels[i][concept] for i in rows], device=device, dtype=logits.dtype
        )
        weight = (pos_weight or {}).get(concept)
        if weight is not None and weight != 1.0:
            losses.append(
                bce(
                    logits,
                    targets,
                    pos_weight=torch.tensor(weight, device=device, dtype=logits.dtype),
                )
            )
        else:
            losses.append(bce(logits, targets))
    for group, head in heads.groups.items():
        rows = [i for i, labels in enumerate(group_labels) if group in labels]
        if not rows:
            continue
        logits = head(features[rows])
        targets = torch.tensor([group_labels[i][group] for i in rows], device=device)
        losses.append(ce(logits, targets, label_smoothing=group_label_smoothing))
    if not losses:
        return None
    stacked = torch.stack(losses)
    return stacked.mean() if reduction == "mean" else stacked.sum()


@torch.no_grad()
def _reference_val_scores(backbone, heads, loader, device):
    bin_logits = {c: [] for c in heads.binary}
    bin_targets = {c: [] for c in heads.binary}
    grp_logits = {g: [] for g in heads.groups}
    grp_targets = {g: [] for g in heads.groups}
    for images, binary_labels, group_labels in loader:
        features = backbone(images.to(device)).float()
        for concept, head in heads.binary.items():
            rows = [i for i, labels in enumerate(binary_labels) if concept in labels]
            if not rows:
                continue
            logits = head(features[rows]).squeeze(-1)
            bin_logits[concept].append(
                logits.detach().cpu().numpy().astype(np.float64).reshape(-1)
            )
            bin_targets[concept].append(
                np.asarray([binary_labels[i][concept] for i in rows], dtype=np.float64)
            )
        for group, head in heads.groups.items():
            rows = [i for i, labels in enumerate(group_labels) if group in labels]
            if not rows:
                continue
            logits = head(features[rows])
            grp_logits[group].append(logits.detach().cpu().numpy().astype(np.float64))
            grp_targets[group].append(
                np.asarray([group_labels[i][group] for i in rows], dtype=np.int64)
            )
    binary_scores = {
        c: (np.concatenate(bin_logits[c]), np.concatenate(bin_targets[c]))
        for c in heads.binary
        if bin_logits[c]
    }
    group_scores = {
        g: (np.concatenate(grp_logits[g]), np.concatenate(grp_targets[g]))
        for g in heads.groups
        if grp_logits[g]
    }
    return binary_scores, group_scores


# -- fixtures -----------------------------------------------------------------


def _heads(concepts=12, groups=(3, 5, 2), seed=0):
    vocab = SimpleNamespace(
        concepts=[f"concept_{i:02d}" for i in range(concepts)],
        groups={
            f"group_{i}": [f"class_{j}" for j in range(n)] for i, n in enumerate(groups)
        },
    )
    torch.manual_seed(seed)
    return bb._MultiTaskHeads(FEATURES, vocab), vocab


def _labels(vocab, batch, seed, *, concept_rate=0.35, group_rate=0.6):
    """Partly-labelled rows: hard and soft binary targets, and group classes."""
    rng = random.Random(seed)
    binary, group = [], []
    for _ in range(batch):
        binary.append(
            {
                concept: rng.choice([1.0, 0.0, 0.0, 0.1])
                for concept in vocab.concepts
                if rng.random() < concept_rate
            }
        )
        group.append(
            {
                name: rng.randrange(len(classes))
                for name, classes in vocab.groups.items()
                if rng.random() < group_rate
            }
        )
    return binary, group


def _features(batch, seed):
    return torch.randn(batch, FEATURES, generator=torch.Generator().manual_seed(seed))


def _grads(loss, heads, features):
    params = list(heads.parameters())
    grads = torch.autograd.grad(loss, [features, *params], allow_unused=True)
    return grads[0], dict(
        zip([name for name, _ in heads.named_parameters()], grads[1:])
    )


# -- training loss ------------------------------------------------------------


@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("smoothing", [0.0, 0.1])
@pytest.mark.parametrize("weighted", [False, True])
def test_loss_matches_the_per_head_loop(reduction, smoothing, weighted):
    heads, vocab = _heads()
    binary, group = _labels(vocab, batch=16, seed=1)
    features = _features(16, seed=2)
    # Weights for some heads only, 1.0 for one of them, none for the rest.
    pos_weight = (
        {"concept_00": 3.0, "concept_03": 1.0, "concept_07": 9.5} if weighted else None
    )
    kwargs = {
        "pos_weight": pos_weight,
        "group_label_smoothing": smoothing,
        "reduction": reduction,
    }

    new = bb._batch_loss(features, heads, binary, group, CPU, **kwargs)
    old = _reference_batch_loss(features, heads, binary, group, CPU, **kwargs)

    torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6)


def test_gradients_match_the_per_head_loop():
    heads, vocab = _heads()
    binary, group = _labels(vocab, batch=16, seed=3)
    kwargs = {
        "pos_weight": {"concept_02": 4.0},
        "group_label_smoothing": 0.1,
        "reduction": "mean",
    }

    features = _features(16, seed=4).requires_grad_()
    new_f, new_p = _grads(
        bb._batch_loss(features, heads, binary, group, CPU, **kwargs), heads, features
    )
    old_f, old_p = _grads(
        _reference_batch_loss(features, heads, binary, group, CPU, **kwargs),
        heads,
        features,
    )

    torch.testing.assert_close(new_f, old_f, rtol=1e-4, atol=1e-6)
    assert new_p.keys() == old_p.keys()
    for name, old_grad in old_p.items():
        if old_grad is None:
            assert new_p[name] is None, name
        else:
            torch.testing.assert_close(new_p[name], old_grad, rtol=1e-4, atol=1e-6)


def test_heads_with_no_label_in_the_batch_get_no_gradient():
    """An absent head must stay out of the graph: a zero gradient would still
    move it (momentum, weight decay) on a step it has no evidence for."""
    heads, vocab = _heads()
    binary = [{"concept_01": 1.0}, {"concept_01": 0.0, "concept_04": 1.0}, {}]
    group = [{"group_0": 2}, {}, {"group_0": 0}]
    features = _features(3, seed=5).requires_grad_()

    _feature_grad, grads = _grads(
        bb._batch_loss(features, heads, binary, group, CPU), heads, features
    )

    touched = {name for name, grad in grads.items() if grad is not None}
    assert touched == {
        "binary.concept_01.weight",
        "binary.concept_01.bias",
        "binary.concept_04.weight",
        "binary.concept_04.bias",
        "groups.group_0.weight",
        "groups.group_0.bias",
    }


def test_concept_heads_are_not_called_one_by_one():
    heads, vocab = _heads()
    binary, group = _labels(vocab, batch=16, seed=6)
    calls = []
    for head in heads.binary.values():
        head.register_forward_hook(lambda *_args: calls.append(1))

    assert bb._batch_loss(_features(16, seed=7), heads, binary, group, CPU) is not None
    assert calls == []


def test_loss_edge_cases_match_the_per_head_loop():
    heads, vocab = _heads()
    features = _features(4, seed=8)
    cases = {
        "nothing labelled": ([{}, {}, {}, {}], [{}, {}, {}, {}]),
        "concepts only": ([{"concept_00": 1.0}, {"concept_00": 0.0}, {}, {}], [{}] * 4),
        "groups only": ([{}] * 4, [{"group_1": 4}, {}, {"group_1": 0}, {}]),
        # A label for a concept or group that has no head is ignored.
        "unknown names": (
            [{"not_a_head": 1.0, "concept_05": 0.1}, {}, {}, {}],
            [{"not_a_group": 1, "group_2": 1}, {}, {}, {}],
        ),
    }
    for name, (binary, group) in cases.items():
        new = bb._batch_loss(features, heads, binary, group, CPU, reduction="mean")
        old = _reference_batch_loss(
            features, heads, binary, group, CPU, reduction="mean"
        )
        if old is None:
            assert new is None, name
        else:
            torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6, msg=name)


def test_loss_runs_under_autocast():
    """Training calls the loss inside autocast with fp32 features."""
    heads, vocab = _heads()
    binary, group = _labels(vocab, batch=8, seed=9)
    features = _features(8, seed=10)
    old = _reference_batch_loss(features, heads, binary, group, CPU, reduction="mean")

    with torch.autocast("cpu", dtype=torch.bfloat16):
        new = bb._batch_loss(features, heads, binary, group, CPU, reduction="mean")

    assert new.dtype == torch.float32
    torch.testing.assert_close(new, old, rtol=5e-2, atol=5e-2)


# -- validation scores --------------------------------------------------------


def _loader(vocab, batches=4, batch=8):
    out = []
    for index in range(batches):
        binary, group = _labels(vocab, batch=batch, seed=100 + index, concept_rate=0.2)
        out.append((_features(batch, seed=200 + index), binary, group))
    return out


def test_validation_scores_match_the_per_head_loop():
    heads, vocab = _heads(concepts=15)
    loader = _loader(vocab)
    # One concept and one group never appear: they must be left out, as before.
    for _images, binary, group in loader:
        for labels in binary:
            labels.pop("concept_14", None)
        for labels in group:
            labels.pop("group_2", None)

    new_binary, new_group = bb._collect_val_scores(nn.Identity(), heads, loader, CPU)
    old_binary, old_group = _reference_val_scores(nn.Identity(), heads, loader, CPU)

    assert list(new_binary) == list(old_binary)
    assert "concept_14" not in new_binary
    for concept, (old_logits, old_targets) in old_binary.items():
        logits, targets = new_binary[concept]
        assert logits.dtype == np.float64 and targets.dtype == np.float64
        assert logits.shape == old_logits.shape
        np.testing.assert_array_equal(targets, old_targets)
        np.testing.assert_allclose(logits, old_logits, rtol=1e-5, atol=1e-6)

    assert list(new_group) == list(old_group)
    assert "group_2" not in new_group
    for name, (old_logits, old_targets) in old_group.items():
        logits, targets = new_group[name]
        assert logits.dtype == np.float64 and targets.dtype == np.int64
        assert logits.shape == old_logits.shape
        np.testing.assert_array_equal(targets, old_targets)
        np.testing.assert_allclose(logits, old_logits, rtol=1e-5, atol=1e-6)


def test_validation_does_not_call_heads_one_by_one():
    heads, vocab = _heads()
    calls = []
    for head in [*heads.binary.values(), *heads.groups.values()]:
        head.register_forward_hook(lambda *_args: calls.append(1))

    binary_scores, group_scores = bb._collect_val_scores(
        nn.Identity(), heads, _loader(vocab), CPU
    )

    assert binary_scores and group_scores
    assert calls == []


def test_validation_with_nothing_labelled_returns_nothing():
    heads, _vocab = _heads()
    loader = [(_features(4, seed=1), [{}] * 4, [{}] * 4)]
    assert bb._collect_val_scores(nn.Identity(), heads, loader, CPU) == ({}, {})


def test_validation_leaves_the_modules_in_training_mode():
    heads, vocab = _heads()
    backbone = nn.Identity()
    bb._collect_val_scores(backbone, heads, _loader(vocab, batches=1), CPU)
    assert backbone.training and heads.training
