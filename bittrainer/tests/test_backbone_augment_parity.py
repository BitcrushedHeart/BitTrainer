"""Backbone augmentation parity with the group trainer (trainer-parity round, A3).

The backbone's train transform gains RandomResizedCrop (replacing the squash
Resize), RandAugment and RandomErasing — the recipe the group trainer already
runs unconditionally — plus label smoothing for the group-CE heads, all
config-gated with parity defaults. Validation stays deterministic and moves to
the aspect-preserving Resize(shorter side) + CenterCrop convention. Also pins
the two audit fold-ins that ride this file: loss normalisation over
heads-present (``reduction="mean"``, the new task default) and the opt-in
random positive-cap selection (function default stays density-first), plus the
opt-in autobatch probe (``batch_size: "auto"``).

Pure transform/loss/plan tests + one tiny end-to-end run for autobatch.
"""

from __future__ import annotations

import asyncio

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

import bittrainer.backbone_trainer as bb
from bittrainer.backbone_trainer import run_backbone_training
from bittrainer.generic.tasks.backbone_task import BackboneTask
from bittrainer.tests.test_backbone_generic import _request


def _classes(compose) -> list[str]:
    return [type(t).__name__ for t in compose.transforms]


_LEGACY_AUG = {
    "use_random_resized_crop": False,
    "randaugment_n": 0,
    "random_erasing_p": 0.0,
}


# --------------------------------------------------------------------------- #
# Train transform                                                             #
# --------------------------------------------------------------------------- #


def test_train_transform_defaults_carry_the_group_recipe():
    names = _classes(bb._train_transform(64))
    assert "RandomResizedCrop" in names
    assert "RandAugment" in names
    assert "RandomErasing" in names
    # The squash Resize is REPLACED by the crop, not stacked on top of it.
    assert "Resize" not in names


def test_train_transform_gates_off_reproduce_legacy_shape():
    names = _classes(bb._train_transform(64, _LEGACY_AUG))
    assert "RandomResizedCrop" not in names
    assert "RandAugment" not in names
    assert "RandomErasing" not in names
    resizes = [
        t
        for t in bb._train_transform(64, _LEGACY_AUG).transforms
        if type(t).__name__ == "Resize"
    ]
    assert len(resizes) == 1 and tuple(resizes[0].size) == (64, 64)  # legacy squash


def test_train_transform_output_shape_and_erasing_gate():
    img = Image.new("RGB", (100, 80), color=(120, 60, 30))
    out = bb._train_transform(64)(img)
    assert out.shape == (3, 64, 64)
    # random_erasing_p gates independently of the other knobs.
    names = _classes(bb._train_transform(64, {"random_erasing_p": 0.0}))
    assert "RandomErasing" not in names and "RandAugment" in names


# --------------------------------------------------------------------------- #
# Val transform: deterministic, aspect-preserving Resize + CenterCrop         #
# --------------------------------------------------------------------------- #


def test_val_transform_is_resize_center_crop():
    tf = bb._val_transform(64)
    names = _classes(tf)
    assert "CenterCrop" in names
    resizes = [t for t in tf.transforms if type(t).__name__ == "Resize"]
    assert len(resizes) == 1
    size = resizes[0].size
    # Aspect-preserving: a single int (shorter side), never the (s, s) squash.
    assert isinstance(size, int) or (hasattr(size, "__len__") and len(size) == 1)


def test_val_transform_is_deterministic_and_square():
    tf = bb._val_transform(64)
    img = Image.new("RGB", (128, 96))
    img.putpixel((10, 10), (255, 0, 0))
    a, b = tf(img), tf(img)
    assert torch.equal(a, b)
    assert a.shape == (3, 64, 64)


# --------------------------------------------------------------------------- #
# _batch_loss: group-CE label smoothing + heads-present normalisation         #
# --------------------------------------------------------------------------- #


def _group_fixture():
    vocab = bb._Vocab(
        [
            {"groups": {"g": "a"}},
            {"groups": {"g": "b"}},
            {"groups": {"g": "c"}},
        ]
    )
    torch.manual_seed(0)
    heads = bb._MultiTaskHeads(8, vocab)
    features = torch.randn(4, 8)
    group_labels = [{"g": 0}, {"g": 1}, {"g": 2}, {"g": 0}]
    return heads, features, group_labels


def test_group_ce_label_smoothing_matches_reference():
    heads, features, group_labels = _group_fixture()
    binary_labels: list[dict] = [{}, {}, {}, {}]
    device = torch.device("cpu")
    smoothed = bb._batch_loss(
        features,
        heads,
        binary_labels,
        group_labels,
        device,
        group_label_smoothing=0.1,
    )
    plain = bb._batch_loss(features, heads, binary_labels, group_labels, device)
    logits = heads.groups["g"](features)
    targets = torch.tensor([0, 1, 2, 0])
    reference = F.cross_entropy(logits, targets, label_smoothing=0.1)
    assert torch.allclose(smoothed, reference, atol=1e-6)
    assert not torch.allclose(smoothed, plain, atol=1e-6)


def test_batch_loss_mean_reduction_normalises_over_heads_present():
    vocab = bb._Vocab(
        [
            {"binary": {"c1": "positive"}},
            {"binary": {"c1": "negative"}},
            {"binary": {"c2": "positive"}},
            {"binary": {"c2": "negative"}},
            {"groups": {"g": "a"}},
            {"groups": {"g": "b"}},
        ]
    )
    torch.manual_seed(1)
    heads = bb._MultiTaskHeads(8, vocab)
    features = torch.randn(4, 8)
    binary_labels = [
        {"c1": 1.0, "c2": 0.0},
        {"c1": 0.0, "c2": 1.0},
        {"c1": 1.0},
        {"c2": 0.0},
    ]
    group_labels = [{"g": 0}, {"g": 1}, {}, {"g": 0}]
    device = torch.device("cpu")
    total = bb._batch_loss(features, heads, binary_labels, group_labels, device)
    mean = bb._batch_loss(
        features,
        heads,
        binary_labels,
        group_labels,
        device,
        reduction="mean",
    )
    # Three contributing losses (c1, c2, g): mean == sum / 3, and the bare call
    # stays the legacy sum so direct callers are unchanged.
    assert torch.allclose(mean * 3.0, total, atol=1e-6)


def test_backbone_task_defaults_for_loss_and_sampling():
    request = _request_no_files()
    task = BackboneTask(request)
    assert task.loss_reduction == "mean"  # new backbone default (audit fold-in)
    assert task.group_label_smoothing == pytest.approx(0.1)
    assert task.positive_cap_mode == "density"  # behaviour-preserving default
    legacy = _request_no_files()
    legacy["training_config"].update(
        {
            "loss_reduction": "sum",
            "group_label_smoothing": 0.0,
            "positive_cap_mode": "random",
        }
    )
    task2 = BackboneTask(legacy)
    assert task2.loss_reduction == "sum"
    assert task2.group_label_smoothing == 0.0
    assert task2.positive_cap_mode == "random"


def _request_no_files() -> dict:
    return {
        "run_id": "run_aug",
        "convnextv2_size": "atto",
        "candidate_checkpoint_path": "unused/candidate.safetensors",
        "records": [],
        "training_config": {"device": "cpu"},
        "heads": {},
    }


# --------------------------------------------------------------------------- #
# Positive-cap selection mode                                                 #
# --------------------------------------------------------------------------- #


def _cap_samples():
    dense = [
        bb._Sample(f"dense{i}.png", {"c": 1.0, "d": 1.0}, {"g": 0}) for i in range(10)
    ]
    sparse = [bb._Sample(f"sparse{i}.png", {"c": 1.0}, {}) for i in range(50)]
    negatives = [bb._Sample(f"neg{i}.png", {"c": 0.0}, {}) for i in range(100)]
    vocab = bb._Vocab(
        [
            {"binary": {"c": "positive"}},
            {"binary": {"c": "negative"}},
            {"binary": {"d": "positive"}},
            {"binary": {"d": "negative"}},
            {"groups": {"g": "a"}},
            {"groups": {"g": "b"}},
        ]
    )
    return dense + sparse + negatives, vocab


def _selected_positive_paths(planned) -> set:
    return {s.path for s in planned if s.binary.get("c") == 1.0}


def test_density_mode_keeps_label_dense_positives_first():
    samples, vocab = _cap_samples()
    planned, stats = bb._plan_epoch_samples(
        samples,
        vocab,
        epoch=0,
        positive_cap=20,
        min_positive_threshold=0,
    )
    selected = _selected_positive_paths(planned)
    assert stats["c"]["pos"] == 20
    assert all(f"dense{i}.png" in selected for i in range(10))


def test_random_mode_draws_a_seeded_uniform_sample():
    samples, vocab = _cap_samples()
    planned, stats = bb._plan_epoch_samples(
        samples,
        vocab,
        epoch=0,
        positive_cap=20,
        min_positive_threshold=0,
        positive_cap_mode="random",
    )
    selected = _selected_positive_paths(planned)
    assert stats["c"]["pos"] == 20
    # A uniform 20-of-60 draw keeping ALL 10 dense images has probability
    # ~2e-5 — the density bias must be gone.
    assert not all(f"dense{i}.png" in selected for i in range(10))
    # Deterministic: the same (seed, epoch) rebuilds the identical plan.
    replay, _ = bb._plan_epoch_samples(
        samples,
        vocab,
        epoch=0,
        positive_cap=20,
        min_positive_threshold=0,
        positive_cap_mode="random",
    )
    assert _selected_positive_paths(replay) == selected


# --------------------------------------------------------------------------- #
# Autobatch opt-in                                                            #
# --------------------------------------------------------------------------- #


def test_batch_size_auto_runs_the_probe(tmp_path):
    request = _request(tmp_path, epochs=1, max_steps=4, n=8)
    request["training_config"]["batch_size"] = "auto"
    frames: list[dict] = []
    result = asyncio.run(
        run_backbone_training(request, progress_callback=frames.append)
    )
    auto = [f for f in frames if f.get("type") == "autobatch"]
    assert auto, "no autobatch frame emitted for batch_size='auto'"
    assert int(auto[0]["batch_size"]) >= 4
    assert result["candidate_checkpoint_path"]
