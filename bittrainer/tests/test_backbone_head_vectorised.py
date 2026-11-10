"""Concept heads in one matrix product, not one call per head (Bitcrush ISSUE-1655).

``_batch_loss`` and ``_collect_val_scores`` used to visit every concept head in
Python: a row scan, an index, a Linear and a loss per head and per batch, and in
validation a GPU-to-CPU copy per head as well. With several hundred concepts
that is thousands of tiny calls per batch. Both now run the concept heads as a
single matrix product.

The old loops are kept here as the reference: the new code must give the same
loss, the same gradients and the same validation scores in the same order.

Since the first run on the T4 the concept heads are also STORED as one matrix
(a row per concept) instead of a ``Linear`` each. With 900 concepts the
optimiser, the gradient scaler, the clipper and the EMA were walking 1,800 small
tensors one by one every step, and a Base step took 4.8 s instead of 1.1 s. What
is saved keeps its layout: one weight and one bias per concept.
"""

from __future__ import annotations

import copy
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


def _one_head(heads, concept, features):
    """One concept's logits on its own: what the ``Linear`` per concept computed."""
    weight, bias = heads.binary_rows([concept])
    return nn.functional.linear(features, weight, bias)


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
    for concept in heads.binary_names:
        rows = [i for i, labels in enumerate(binary_labels) if concept in labels]
        if not rows:
            continue
        logits = _one_head(heads, concept, features[rows]).squeeze(-1)
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
    bin_logits = {c: [] for c in heads.binary_names}
    bin_targets = {c: [] for c in heads.binary_names}
    grp_logits = {g: [] for g in heads.groups}
    grp_targets = {g: [] for g in heads.groups}
    for images, binary_labels, group_labels in loader:
        features = backbone(images.to(device)).float()
        for concept in heads.binary_names:
            rows = [i for i, labels in enumerate(binary_labels) if concept in labels]
            if not rows:
                continue
            logits = _one_head(heads, concept, features[rows]).squeeze(-1)
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
        for c in heads.binary_names
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


def test_a_head_with_no_label_in_the_batch_gets_a_row_of_zeros():
    """The concept heads are rows of one matrix, so a head with no label in the
    batch gets a zero row rather than no gradient at all, and the optimiser's
    momentum may still nudge it. That is how a multi-label layer trains
    everywhere else; these heads only exist to teach the trunk and are retrained
    afterwards. A group with no label still stays out of the graph."""
    heads, vocab = _heads()
    binary = [{"concept_01": 1.0}, {"concept_01": 0.0, "concept_04": 1.0}, {}]
    group = [{"group_0": 2}, {}, {"group_0": 0}]
    features = _features(3, seed=5).requires_grad_()

    _feature_grad, grads = _grads(
        bb._batch_loss(features, heads, binary, group, CPU), heads, features
    )

    touched = {name for name, grad in grads.items() if grad is not None}
    assert touched == {
        "binary_weight",
        "binary_bias",
        "groups.group_0.weight",
        "groups.group_0.bias",
    }
    learning = {"concept_01", "concept_04"}
    for column, concept in enumerate(vocab.concepts):
        moved = bool(grads["binary_weight"][column].abs().sum() > 0)
        assert moved == (concept in learning), concept
        assert bool(grads["binary_bias"][column] != 0) == (concept in learning), concept


def test_concept_heads_are_two_tensors_however_many_there_are():
    few, _ = _heads(concepts=3)
    many, _ = _heads(concepts=900)

    names = [name for name, _param in many.named_parameters()]
    assert names[:2] == ["binary_weight", "binary_bias"]
    # Two for every concept together, two per group (three groups).
    assert len(names) == len(list(few.parameters())) == 2 + 2 * 3
    assert many.binary_weight.shape == (900, FEATURES)
    assert many.binary_bias.shape == (900,)


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
    for head in heads.groups.values():
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


# -- what is saved ------------------------------------------------------------


def _small():
    return _heads(concepts=3, groups=(2,))


def test_saved_layout_is_still_a_weight_and_a_bias_per_concept():
    heads, _vocab = _small()

    state = heads.state_dict()

    assert list(state) == [
        "binary.concept_00.weight",
        "binary.concept_00.bias",
        "binary.concept_01.weight",
        "binary.concept_01.bias",
        "binary.concept_02.weight",
        "binary.concept_02.bias",
        "groups.group_0.weight",
        "groups.group_0.bias",
    ]
    assert state["binary.concept_01.weight"].shape == (1, FEATURES)
    assert state["binary.concept_01.bias"].shape == (1,)
    assert torch.equal(state["binary.concept_01.weight"][0], heads.binary_weight[1])
    assert torch.equal(state["binary.concept_01.bias"][0], heads.binary_bias[1])
    assert not state["binary.concept_01.weight"].requires_grad


def test_a_file_from_the_linear_per_concept_layout_loads():
    """What every earlier run saved, and what a candidate file still holds."""
    _heads_unused, vocab = _small()
    torch.manual_seed(3)
    old = nn.ModuleDict(
        {
            "binary": nn.ModuleDict(
                {c: nn.Linear(FEATURES, 1) for c in vocab.concepts}
            ),
            "groups": nn.ModuleDict(
                {g: nn.Linear(FEATURES, len(cs)) for g, cs in vocab.groups.items()}
            ),
        }
    )
    heads = bb._MultiTaskHeads(FEATURES, vocab)

    heads.load_state_dict(old.state_dict())

    features = _features(4, seed=1)
    for concept in vocab.concepts:
        torch.testing.assert_close(
            _one_head(heads, concept, features), old["binary"][concept](features)
        )
    torch.testing.assert_close(
        heads.groups["group_0"](features), old["groups"]["group_0"](features)
    )


def test_loading_names_the_heads_that_are_missing_or_unknown():
    heads, _vocab = _small()
    state = dict(heads.state_dict())
    del state["binary.concept_01.bias"]
    state["binary.stranger.weight"] = torch.zeros(1, FEATURES)

    with pytest.raises(RuntimeError) as refused:
        heads.load_state_dict(state)
    assert "binary.concept_01.bias" in str(refused.value)
    assert "binary.stranger.weight" in str(refused.value)

    result = heads.load_state_dict(state, strict=False)
    assert result.missing_keys == ["binary.concept_01.bias"]
    assert result.unexpected_keys == ["binary.stranger.weight"]


def test_a_head_of_another_width_is_refused():
    heads, _vocab = _small()
    state = dict(heads.state_dict())
    state["binary.concept_02.weight"] = torch.zeros(1, FEATURES + 1)

    with pytest.raises(RuntimeError, match="binary.concept_02.weight"):
        heads.load_state_dict(state)


def test_the_layout_holds_inside_a_model_and_for_its_copy():
    heads, _vocab = _small()
    model = bb._BackboneWithHeads(nn.Linear(4, FEATURES), heads)

    keys = list(model.state_dict())
    assert "heads.binary.concept_00.weight" in keys
    assert not any("binary_weight" in key for key in keys)

    clone = copy.deepcopy(model)  # how the EMA takes its copy
    with torch.no_grad():
        clone.heads.binary_weight.add_(1.0)
    clone.load_state_dict(model.state_dict())
    assert torch.equal(clone.heads.binary_weight, heads.binary_weight)


def test_backups_of_the_older_layout_do_not_resume():
    """Their optimiser state has a tensor per concept; the fingerprint keeps a
    run from loading it."""
    vocab = SimpleNamespace(concepts=["a"], groups={})
    assert bb._backbone_fingerprint(vocab, "atto", 1)["heads"] == "matrix"


# -- thresholds ---------------------------------------------------------------


def test_thresholds_match_the_shared_search_without_its_cost():
    """The per-head threshold search called scikit-learn 36 times per head: 108 s
    for 902 heads on the T4. The same search in numpy must pick the same values."""
    from bittrainer.group_validation import find_per_class_thresholds

    rng = np.random.default_rng(0)
    scores = {}
    for index in range(40):
        size = int(rng.integers(3, 200))
        targets = (rng.random(size) < rng.uniform(0.05, 0.9)).astype(np.float64)
        logits = rng.normal(0, 2, size) + 2.5 * targets
        scores[f"head_{index}"] = (logits, targets)
    scores["no_yes"] = (rng.normal(0, 1, 30), np.zeros(30))
    scores["all_yes"] = (rng.normal(0, 1, 30), np.ones(30))
    scores["one_row"] = (np.array([0.3]), np.array([1.0]))

    for min_positive in (1, 5):
        tuned = bb._tune_binary_thresholds(scores, min_positive=min_positive)
        assert list(tuned) == list(scores)
        for concept, (logits, targets) in scores.items():
            probs = 1.0 / (1.0 + np.exp(-logits))
            expected = find_per_class_thresholds(
                probs.reshape(-1, 1), targets.reshape(-1, 1), min_positive=min_positive
            )[0]
            assert tuned[concept] == pytest.approx(float(expected), abs=1e-12), concept


# -- the metric panel ---------------------------------------------------------


def _reference_metrics(binary_scores, group_scores):
    """The panel as it was: two scikit-learn calls per concept head and a dozen
    per group. 4.75 s for 900 heads on 582 validation images on the T4, and more
    with every validation image."""
    from sklearn.metrics import average_precision_score, f1_score

    from bittrainer.group_validation import compute_multiclass_metrics

    metrics = {}
    binary_f1s = []
    for concept, (logits, targets) in binary_scores.items():
        logits = np.asarray(logits, dtype=np.float64)
        tgt_int = np.asarray(targets, dtype=np.float64).astype(np.int64)
        preds = (logits > 0).astype(np.int64)
        n = int(tgt_int.shape[0])
        metrics[f"binary/{concept}"] = float((preds == tgt_int).mean()) if n else 0.0
        support = int(tgt_int.sum())
        metrics[f"binary_support/{concept}"] = support
        if support >= 1:
            f1 = float(f1_score(tgt_int, preds, zero_division=0))
            probs = 1.0 / (1.0 + np.exp(-logits))
            metrics[f"binary_f1/{concept}"] = f1
            metrics[f"binary_ap/{concept}"] = float(
                average_precision_score(tgt_int, probs)
            )
            binary_f1s.append(f1)
    group_f1s = []
    for group, (logits, targets) in group_scores.items():
        logits = np.asarray(logits, dtype=np.float64)
        tgt = np.asarray(targets, dtype=np.int64)
        n = int(tgt.shape[0])
        preds = logits.argmax(axis=1)
        metrics[f"group/{group}"] = float((preds == tgt).mean()) if n else 0.0
        metrics[f"group_val_support/{group}"] = n
        if n:
            gm = compute_multiclass_metrics(
                tgt.tolist(), preds.tolist(), int(logits.shape[1])
            )
            metrics[f"group_f1/{group}"] = float(gm["macro_f1"])
            group_f1s.append(metrics[f"group_f1/{group}"])
    aggregates = []
    if binary_f1s:
        metrics["binary_macro_f1"] = float(np.mean(binary_f1s))
        aggregates.append(metrics["binary_macro_f1"])
    if group_f1s:
        metrics["group_macro_f1"] = float(np.mean(group_f1s))
        aggregates.append(metrics["group_macro_f1"])
    metrics["selection"] = float(np.mean(aggregates)) if aggregates else 0.0
    return metrics


def _panel_scores():
    rng = np.random.default_rng(1)
    binary = {}
    for index in range(40):
        size = int(rng.integers(1, 300))
        targets = (rng.random(size) < rng.uniform(0.05, 0.9)).astype(np.float64)
        logits = rng.normal(0, 2, size) + 2.5 * targets
        if index % 3 == 0:
            logits = np.round(logits)  # many images share a score
        binary[f"head_{index}"] = (logits, targets)
    binary["no_yes"] = (rng.normal(0, 1, 30), np.zeros(30))
    binary["all_yes"] = (rng.normal(0, 1, 30), np.ones(30))
    binary["one_row"] = (np.array([0.3]), np.array([1.0]))
    # Different logits that are all 1.0 as a probability: one tied score.
    binary["saturated"] = (
        np.array([50.0, 45.0, 40.0, -50.0, 60.0]),
        np.array([1.0, 0.0, 1.0, 0.0, 0.0]),
    )
    groups = {}
    for index in range(6):
        size, classes = int(rng.integers(1, 200)), int(rng.integers(2, 7))
        # Every other group has a class no validation image belongs to.
        targets = rng.integers(0, classes - index % 2, size)
        groups[f"group_{index}"] = (rng.normal(0, 1, (size, classes)), targets)
    return binary, groups


def test_metric_panel_matches_the_scikit_learn_one():
    binary, groups = _panel_scores()

    new = bb._backbone_metrics(binary, groups)
    old = _reference_metrics(binary, groups)

    # Everything the old panel reported, under the same keys; what is new is
    # the recall at fixed precision.
    assert set(old) <= set(new)
    assert all("r_at_p" in key for key in set(new) - set(old))
    for key, value in old.items():
        assert type(new[key]) is type(value), key
        assert new[key] == pytest.approx(value, rel=1e-12, abs=1e-12), key


def test_metric_panel_makes_no_scikit_learn_call(monkeypatch):
    import sklearn.metrics

    import bittrainer.group_validation as group_validation

    def _refuse(*_args, **_kwargs):
        raise AssertionError("the metric panel called scikit-learn")

    for name in ("f1_score", "average_precision_score", "precision_recall_curve"):
        monkeypatch.setattr(sklearn.metrics, name, _refuse)
    monkeypatch.setattr(group_validation, "compute_multiclass_metrics", _refuse)
    binary, groups = _panel_scores()

    assert bb._backbone_metrics(binary, groups)["selection"] > 0
