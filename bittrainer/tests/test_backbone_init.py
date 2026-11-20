"""backbone_init: Engine's Backbone Builder contract (Bitcrush ISSUE-0342).

Engine resolves a backbone spec (a local BitVision trunk file, or random init
for a deliberate from-nothing run) and passes it to every trainer config as
``backbone_init``. These tests pin:
  * the train configs accept the field,
  * a local file loads into a fresh model's backbone,
  * nothing starts from random weights unless it says so (Bitcrush ISSUE-1733):
    a fresh model with no local trunk is refused, and so is a warm start from an
    incumbent that shares no tensor with the model.
"""

from __future__ import annotations

import pytest
import torch

from bittrainer.backbone_init import apply_backbone_init
from bittrainer.model import create_model

SPEC = {
    "source": "local_active",
    "checkpoint_path": "C:/backbones/active.safetensors",
    "size_alias": "pro",
    "convnextv2_size": "base",
}

RANDOM = {"source": "random_init"}

# Every spec that names no local trunk file and does not ask for random weights.
NO_TRUNK_SPECS = [
    None,
    {},
    {"source": "local_active", "checkpoint_path": None},
    {"source": "timm_pretrained", "checkpoint_path": None},
    {"source": "temporary_timm_pretrained_fallback", "checkpoint_path": None},
]


def test_all_train_configs_accept_backbone_init():
    from bittrainer.group_trainer import GroupTrainConfig
    from bittrainer.trainer import TrainConfig

    assert TrainConfig(concept_folder="concept", backbone_init=SPEC).backbone_init == SPEC
    assert (
        GroupTrainConfig(
            group_folder="group", num_classes=2, class_names=["a", "b"], backbone_init=SPEC
        ).backbone_init
        == SPEC
    )


def test_backbone_init_defaults_to_none():
    from bittrainer.trainer import TrainConfig

    assert TrainConfig(concept_folder="concept").backbone_init is None


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


def test_random_init_loads_nothing_and_is_allowed():
    target = _atto()
    before = {k: v.clone() for k, v in target.state_dict().items()}
    assert apply_backbone_init(target, RANDOM) is False
    after = target.state_dict()
    for key, tensor in before.items():
        assert torch.equal(after[key], tensor), key


@pytest.mark.parametrize("spec", NO_TRUNK_SPECS)
def test_no_local_trunk_is_refused_unless_random_init(spec):
    with pytest.raises(ValueError, match="random_init"):
        apply_backbone_init(_atto(), spec)


def test_apply_backbone_init_missing_checkpoint_raises(tmp_path):
    target = _atto()
    spec = {"source": "local_active", "checkpoint_path": str(tmp_path / "missing.safetensors")}
    with pytest.raises(FileNotFoundError):
        apply_backbone_init(target, spec)


# ---------------------------------------------------------------------------
# The trainers' fresh-model and warm-start paths (Bitcrush ISSUE-1733)
# ---------------------------------------------------------------------------


def _group_config(tmp_path, **overrides):
    from bittrainer.group_trainer import GroupTrainConfig

    (tmp_path / "ckpt").mkdir(exist_ok=True)
    return GroupTrainConfig(
        group_folder=str(tmp_path),
        num_classes=2,
        class_names=["a", "b"],
        backbone_variant="atto",
        checkpoint_dir=str(tmp_path / "ckpt"),
        **overrides,
    )


def _group_model(config, tmp_path):
    import bittrainer.group_trainer as gt

    return gt._create_or_warmstart_model(
        config,
        device=torch.device("cpu"),
        dtype=torch.float32,
        head_hidden_size=None,
        checkpoint_dir=tmp_path / "ckpt",
    )


@pytest.mark.parametrize("spec", NO_TRUNK_SPECS)
def test_fresh_group_model_without_a_trunk_is_refused(tmp_path, spec):
    """Group and head-only training build their fresh model here."""
    with pytest.raises(ValueError, match="random_init"):
        _group_model(_group_config(tmp_path, backbone_init=spec), tmp_path)


def test_fresh_group_model_missing_trunk_file_is_refused(tmp_path):
    spec = {"source": "local_active", "checkpoint_path": str(tmp_path / "gone.safetensors")}
    with pytest.raises(FileNotFoundError):
        _group_model(_group_config(tmp_path, backbone_init=spec), tmp_path)


def test_fresh_group_model_random_init_builds(tmp_path):
    model = _group_model(_group_config(tmp_path, backbone_init=RANDOM), tmp_path)
    assert model.head.fc.out_features == 2


def test_fresh_group_model_starts_from_the_local_trunk(tmp_path):
    from safetensors.torch import save_file

    donor = _atto()
    trunk = tmp_path / "active.safetensors"
    save_file({k: v for k, v in donor.state_dict().items() if not k.startswith("head.")}, str(trunk))
    spec = {"source": "local_active", "checkpoint_path": str(trunk)}
    model = _group_model(_group_config(tmp_path, backbone_init=spec), tmp_path)
    assert torch.equal(model.stem[0].weight, donor.stem[0].weight)


def test_group_warm_start_sharing_no_tensors_is_refused(tmp_path):
    """An incumbent of another architecture (a retired SigLIP ``best.pt``)
    matches nothing; training on from the untouched random model is refused."""
    config = _group_config(tmp_path, backbone_init=RANDOM)
    foreign = {"trunk.embeds.proj.weight": torch.zeros(768, 768), "head.weight": torch.zeros(2, 768)}
    torch.save({"state_dict": foreign}, tmp_path / "ckpt" / "best.pt")
    with pytest.raises(RuntimeError, match="shares no tensors"):
        _group_model(config, tmp_path)
    # The same incumbent naming its retired size is refused by name.
    torch.save(
        {"state_dict": foreign, "model_size": "siglip2_base"}, tmp_path / "ckpt" / "best.pt"
    )
    with pytest.raises(ValueError, match="siglip2_base"):
        _group_model(config, tmp_path)


def test_group_warm_start_keeps_a_matching_incumbent(tmp_path):
    config = _group_config(tmp_path)  # no backbone_init: the incumbent is the start
    torch.manual_seed(3)
    incumbent = _atto()
    torch.save(
        {"state_dict": incumbent.state_dict(), "model_size": "atto"},
        tmp_path / "ckpt" / "best.pt",
    )
    model = _group_model(config, tmp_path)
    assert torch.equal(model.stem[0].weight, incumbent.stem[0].weight)


@pytest.mark.parametrize("spec", NO_TRUNK_SPECS)
def test_fresh_binary_model_without_a_trunk_is_refused(spec):
    """Binary (concept) training, head-only included, builds its fresh model here."""
    import bittrainer.trainer as bt

    config = bt.TrainConfig(concept_folder="concept", model_size="atto", backbone_init=spec)
    with pytest.raises(ValueError, match="random_init"):
        bt._fresh_binary_model(config, dtype=torch.float32)


def test_fresh_binary_model_random_init_builds():
    import bittrainer.trainer as bt

    config = bt.TrainConfig(concept_folder="concept", model_size="atto", backbone_init=RANDOM)
    assert bt._fresh_binary_model(config, dtype=torch.float32).head.fc.out_features == 2


def test_load_checkpoint_refuses_a_siglip_checkpoint(tmp_path):
    from bittrainer.model import load_checkpoint

    foreign = {"trunk.embeds.proj.weight": torch.zeros(768, 768), "head.fc.weight": torch.zeros(2, 768)}
    torch.save({"state_dict": foreign, "model_size": "siglip2_base"}, tmp_path / "best.pt")
    with pytest.raises(ValueError, match="siglip2_base"):
        load_checkpoint(str(tmp_path / "best.pt"))
