"""Start a ConvNeXt V2 trunk from ConvNeXt V1 weights (Bitcrush ISSUE-1655).

Every published ConvNeXt V2 weight is licensed non-commercial. ConvNeXt V1 has
Apache-2.0 weights, and the two blocks differ in exactly two places: V1 scales
each block's output per channel (LayerScale), V2 adds GRN inside the MLP. So a
V1 trunk becomes a V2 trunk exactly: fold the scale into the block's last
layer, and leave GRN at zero, where it passes its input through.

These tests pin that the conversion is exact. They use tiny randomly
initialised models: nothing is downloaded and nothing trains.
"""

from __future__ import annotations

import pytest
import timm
import torch
from safetensors import safe_open
from timm.models.convnext import ConvNeXt

import bittrainer.convnext_v1_init as v1init
from bittrainer.backbone_init import apply_backbone_init
from bittrainer.model import create_model

_SHAPE = {"depths": (1, 2, 1, 1), "dims": (8, 16, 32, 64)}


def _randomise(v1: torch.nn.Module) -> None:
    """Make the parts the conversion touches matter: timm starts LayerScale at
    1e-6 and the biases at zero, where a wrong fold would go unnoticed."""
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, parameter in v1.named_parameters():
            if name.endswith(".gamma"):
                parameter.copy_(0.5 + torch.rand(parameter.shape, generator=generator))
            elif name.endswith(".mlp.fc2.bias"):
                parameter.copy_(torch.randn(parameter.shape, generator=generator))


def _pair(conv_mlp: bool, **v1_options):
    torch.manual_seed(0)
    v1 = ConvNeXt(conv_mlp=conv_mlp, num_classes=0, **_SHAPE, **v1_options).eval()
    v2 = ConvNeXt(
        conv_mlp=conv_mlp, num_classes=0, use_grn=True, ls_init_value=None, **_SHAPE
    ).eval()
    _randomise(v1)
    return v1, v2


def _images() -> torch.Tensor:
    return torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(2))


# -- the conversion -----------------------------------------------------------


@pytest.mark.parametrize("conv_mlp", [True, False])
def test_converted_trunk_gives_the_same_features(conv_mlp):
    """Both MLP flavours: 1x1 convolutions (atto to nano) and Linear (tiny up)."""
    v1, v2 = _pair(conv_mlp)

    v2.load_state_dict(v1init.convert_v1_state(v1.state_dict(), v2.state_dict()))

    with torch.no_grad():
        torch.testing.assert_close(v2(_images()), v1(_images()), rtol=1e-4, atol=1e-5)


def test_layerscale_is_folded_away_and_grn_passes_through():
    v1, v2 = _pair(conv_mlp=False)

    converted = v1init.convert_v1_state(v1.state_dict(), v2.state_dict())

    assert set(converted) == set(v2.state_dict())
    assert not [key for key in converted if key.endswith(".gamma")]
    grn = [value for key, value in converted.items() if ".mlp.grn." in key]
    assert grn and all(bool((value == 0).all()) for value in grn)
    # The source is left untouched.
    assert any(key.endswith(".gamma") for key in v1.state_dict())


def test_a_trunk_without_layerscale_converts_as_it_is():
    v1, v2 = _pair(conv_mlp=True, ls_init_value=None)

    v2.load_state_dict(v1init.convert_v1_state(v1.state_dict(), v2.state_dict()))

    with torch.no_grad():
        torch.testing.assert_close(v2(_images()), v1(_images()), rtol=1e-4, atol=1e-5)


def test_the_classifier_is_left_behind():
    torch.manual_seed(0)
    v1 = ConvNeXt(conv_mlp=False, num_classes=10, **_SHAPE)
    _v1, v2 = _pair(conv_mlp=False)

    converted = v1init.convert_v1_state(v1.state_dict(), v2.state_dict())

    assert not [key for key in converted if key.startswith("head.fc.")]
    v2.load_state_dict(converted)


def test_a_trunk_of_another_shape_is_refused():
    v1, _v2 = _pair(conv_mlp=False)
    other = ConvNeXt(
        conv_mlp=False,
        num_classes=0,
        use_grn=True,
        ls_init_value=None,
        depths=(1, 1, 1, 1),
        dims=(8, 16, 32, 48),
    )

    with pytest.raises(ValueError, match="do not fit"):
        v1init.convert_v1_state(v1.state_dict(), other.state_dict())


# -- which weights ------------------------------------------------------------


def test_default_weights_exist_and_are_permissive():
    """Reads timm's own catalogue: no weight is downloaded."""
    published = set(timm.list_pretrained("convnext_*"))
    assert set(v1init.V1_WEIGHTS) >= {"atto", "nano", "tiny", "base", "large"}
    for size, name in v1init.V1_WEIGHTS.items():
        assert name in published, (size, name)
        assert name.startswith(f"convnext_{size}.")
        assert v1init.declared_licence(name) in v1init.PERMISSIVE_LICENCES


def test_v2_weights_are_never_called_permissive():
    assert (
        v1init.declared_licence("convnextv2_base.fcmae_ft_in22k_in1k") == "cc-by-nc-4.0"
    )
    assert "cc-by-nc-4.0" not in v1init.PERMISSIVE_LICENCES


def test_weights_under_another_licence_are_refused_before_any_download(monkeypatch):
    monkeypatch.setattr(v1init, "declared_licence", lambda name: "cc-by-nc-4.0")
    monkeypatch.setattr(
        v1init, "_load_v1", lambda name: pytest.fail("the weights were fetched")
    )

    with pytest.raises(ValueError, match="cc-by-nc-4.0"):
        v1init.build_v1_trunk("atto")


# -- building a trunk for a run (the real weights are stood in for) -----------


def _fake_v1(monkeypatch) -> None:
    """Stand a randomly initialised V1 atto in for the downloaded weights."""

    def _load(name: str):
        torch.manual_seed(3)
        model = timm.create_model("convnext_atto", pretrained=False, num_classes=0)
        _randomise(model)
        return model.eval()

    monkeypatch.setattr(v1init, "_load_v1", _load)


def test_a_built_trunk_checks_itself_and_names_its_source(monkeypatch):
    _fake_v1(monkeypatch)

    state, info = v1init.build_v1_trunk("atto")

    assert info["model_id"] == "timm/convnext_atto.d2_in1k"
    assert info["license"] == "apache-2.0"
    assert info["max_abs_difference"] < 1e-3
    assert not [key for key in state if key.endswith(".gamma")]


def test_a_conversion_that_changes_the_features_is_not_handed_over(monkeypatch):
    _fake_v1(monkeypatch)
    real = v1init.convert_v1_state

    def _forgets_the_fold(v1_state, v2_state):
        unfolded = dict(v1_state)
        for key in [key for key in unfolded if key.endswith(".gamma")]:
            unfolded[key] = torch.ones_like(unfolded[key])  # the scale is lost
        return real(unfolded, v2_state)

    monkeypatch.setattr(v1init, "convert_v1_state", _forgets_the_fold)

    with pytest.raises(RuntimeError, match="does not reproduce"):
        v1init.build_v1_trunk("atto")


def test_a_written_trunk_loads_as_a_local_backbone(monkeypatch, tmp_path):
    _fake_v1(monkeypatch)
    path = tmp_path / "trunk.safetensors"

    info = v1init.write_v1_trunk("atto", path)

    with safe_open(path, framework="pt") as trunk:
        metadata = trunk.metadata()
    assert metadata["converted_from"] == "timm/convnext_atto.d2_in1k"
    assert metadata["license"] == "apache-2.0"
    assert info["path"] == str(path)

    backbone = create_model(
        model_size="atto", pretrained=False, num_classes=0, drop_path_rate=0.0
    )
    loaded = apply_backbone_init(
        backbone, {"source": "local_candidate", "checkpoint_path": str(path)}
    )
    assert loaded is True
