"""backbone_init: Engine's Backbone Builder contract (Bitcrush ISSUE-0342).

Engine resolves a backbone spec (local checkpoint / timm fallback / random init)
and passes it to every trainer config as ``backbone_init``. These tests pin:
  * all four train configs accept the field (Engine's
    test_bittrainer_configs_accept_backbone_init_specs mirrors this),
  * the pretrained/local/random routing logic,
  * safetensors round-trip into a fresh model's backbone.
"""

from __future__ import annotations

import pytest
import torch

from bittrainer.backbone_init import apply_backbone_init, wants_timm_pretrained
from bittrainer.model import create_model

SPEC = {
    "source": "local_active",
    "checkpoint_path": "C:/backbones/active.safetensors",
    "size_alias": "pro",
    "convnextv2_size": "base",
}


def test_all_train_configs_accept_backbone_init():
    from bittrainer.dual_branch_trainer import DualBranchTrainConfig
    from bittrainer.group_trainer import GroupTrainConfig
    from bittrainer.multihead_trainer import MultiHeadTrainConfig
    from bittrainer.trainer import TrainConfig

    assert TrainConfig(concept_folder="concept", backbone_init=SPEC).backbone_init == SPEC
    assert (
        GroupTrainConfig(
            group_folder="group", num_classes=2, class_names=["a", "b"], backbone_init=SPEC
        ).backbone_init
        == SPEC
    )
    assert (
        MultiHeadTrainConfig(
            group_folder="group", size_classes=["__none__", "34A"], backbone_init=SPEC
        ).backbone_init
        == SPEC
    )
    assert (
        DualBranchTrainConfig(
            group_folder="group",
            context_folder="context",
            num_classes=2,
            class_names=["a", "b"],
            backbone_init=SPEC,
        ).backbone_init
        == SPEC
    )


def test_backbone_init_defaults_to_none():
    from bittrainer.trainer import TrainConfig

    assert TrainConfig(concept_folder="concept").backbone_init is None


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (None, True),  # legacy behaviour: no spec -> timm pretrained
        ({}, True),
        ({"source": "temporary_timm_pretrained_fallback", "checkpoint_path": None}, True),
        ({"source": "local_active", "checkpoint_path": "x.safetensors"}, False),
        ({"source": "local_candidate", "checkpoint_path": "x.safetensors"}, False),
        ({"source": "local_active", "checkpoint_path": None}, True),  # broken spec -> fallback
        ({"source": "random_init", "checkpoint_path": None}, False),
    ],
)
def test_wants_timm_pretrained(spec, expected):
    assert wants_timm_pretrained(spec) is expected


def _atto(num_classes: int = 2) -> torch.nn.Module:
    return create_model(model_size="atto", pretrained=False, num_classes=num_classes)


def test_apply_backbone_init_loads_local_checkpoint(tmp_path):
    from safetensors.torch import save_file

    donor = _atto()
    ckpt = tmp_path / "active.safetensors"
    backbone_state = {
        k: v for k, v in donor.state_dict().items() if not k.startswith("head.")
    }
    save_file(backbone_state, str(ckpt), metadata={"family_name": "test"})

    target = _atto()
    spec = {"source": "local_active", "checkpoint_path": str(ckpt)}
    assert apply_backbone_init(target, spec) is True

    donor_state = donor.state_dict()
    target_state = target.state_dict()
    for key in backbone_state:
        assert torch.equal(target_state[key], donor_state[key]), key


def test_apply_backbone_init_strips_backbone_prefix(tmp_path):
    from safetensors.torch import save_file

    donor = _atto()
    ckpt = tmp_path / "prefixed.safetensors"
    prefixed = {
        f"backbone.{k}": v
        for k, v in donor.state_dict().items()
        if not k.startswith("head.")
    }
    save_file(prefixed, str(ckpt))

    target = _atto()
    assert apply_backbone_init(target, {"source": "local_active", "checkpoint_path": str(ckpt)})
    assert torch.equal(
        target.state_dict()["stem.0.weight"], donor.state_dict()["stem.0.weight"]
    )


def test_apply_backbone_init_noop_for_non_local_sources(tmp_path):
    target = _atto()
    before = {k: v.clone() for k, v in target.state_dict().items()}
    assert apply_backbone_init(target, None) is False
    assert apply_backbone_init(target, {"source": "random_init"}) is False
    assert (
        apply_backbone_init(
            target, {"source": "temporary_timm_pretrained_fallback", "checkpoint_path": None}
        )
        is False
    )
    after = target.state_dict()
    for key, tensor in before.items():
        assert torch.equal(after[key], tensor), key


def test_apply_backbone_init_missing_checkpoint_raises(tmp_path):
    target = _atto()
    spec = {"source": "local_active", "checkpoint_path": str(tmp_path / "missing.safetensors")}
    with pytest.raises(FileNotFoundError):
        apply_backbone_init(target, spec)
