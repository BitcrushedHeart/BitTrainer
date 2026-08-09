"""Stochastic depth through the shared model factory (round B6).

``create_model`` gains ``drop_path_rate``: ``None`` (the default) is
bit-identical to today (nothing forwarded to timm), a float is forwarded to
timm's ConvNeXt ``drop_path_rate``. ``default_drop_path_rate`` provides the
size-scaled recipe values. The backbone task turns it ON by default (auto by
size, 0 disables); ``GroupTrainConfig`` carries it OFF by default (short
fine-tunes; the repo's A/B caution).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import bittrainer.backbone_trainer as bb
from bittrainer.generic.tasks.backbone_task import BackboneTask
from bittrainer.model import create_model


def _drop_probs(model) -> list[float]:
    from timm.layers import DropPath

    return [
        float(m.drop_prob)
        for m in model.modules()
        if isinstance(m, DropPath) and m.drop_prob and m.drop_prob > 0
    ]


def test_default_drop_path_rate_table():
    from bittrainer.model import default_drop_path_rate

    for size in ("atto", "femto", "pico", "nano"):
        assert default_drop_path_rate(size) == pytest.approx(0.1)
    assert default_drop_path_rate("tiny") == pytest.approx(0.2)
    for size in ("base", "large", "huge"):
        assert default_drop_path_rate(size) == pytest.approx(0.3)


def test_create_model_forwards_drop_path_rate():
    model = create_model(
        model_size="atto",
        pretrained=False,
        num_classes=2,
        drop_path_rate=0.2,
    )
    probs = _drop_probs(model)
    assert probs, "no active DropPath modules — drop_path_rate not forwarded to timm"
    # timm scales the rate linearly across depth up to the configured maximum.
    assert max(probs) == pytest.approx(0.2, abs=1e-6)


def test_create_model_default_stays_bit_identical():
    model = create_model(model_size="atto", pretrained=False, num_classes=2)
    assert _drop_probs(model) == []


def _request(drop_path_rate=None) -> dict:
    config: dict = {"device": "cpu"}
    if drop_path_rate is not None:
        config["drop_path_rate"] = drop_path_rate
    return {
        "run_id": "run_dp",
        "convnextv2_size": "atto",
        "candidate_checkpoint_path": "unused/candidate.safetensors",
        "records": [],
        "training_config": config,
        "heads": {},
        "backbone_init": {"source": "random_init", "checkpoint_path": None},
    }


def test_backbone_task_defaults_to_size_scaled_rate():
    task = BackboneTask(_request())
    assert task.drop_path_rate == pytest.approx(0.1)  # atto auto default
    assert BackboneTask(_request(drop_path_rate=0)).drop_path_rate == pytest.approx(0.0)
    assert BackboneTask(_request(drop_path_rate=0.25)).drop_path_rate == pytest.approx(
        0.25
    )


def test_backbone_task_create_model_applies_the_rate(tmp_path):
    from bittrainer.generic.task import TaskContext

    task = BackboneTask(_request())
    task.vocab = bb._Vocab(
        [
            {"binary": {"c": "positive"}},
            {"binary": {"c": "negative"}},
        ]
    )
    ctx = TaskContext(
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        em=None,
        cb=lambda _m: None,
        checkpoint_dir=Path(tmp_path),
    )
    model = task.create_model(ctx, None)
    probs = _drop_probs(model.backbone)
    assert probs and max(probs) == pytest.approx(0.1, abs=1e-6)


def test_group_config_field_defaults_off():
    from bittrainer.group_trainer import GroupTrainConfig

    config = GroupTrainConfig(group_folder=".", num_classes=2, class_names=["a", "b"])
    assert config.drop_path_rate == pytest.approx(0.0)
