"""SigLIP2 classifier wrapper (Bitcrush ISSUE-1106).

``SigLIPClassifier`` holds a timm NaFlex ViT ``trunk`` (MAP pooling stays in the
trunk) and a ``head`` with the timm ``NormMlpClassifierHead`` surface, so every
helper written against a timm ConvNeXt — pooled features, the cached head
probe, the backbone hash, the embedding cache — works on it unchanged. Inputs
arrive ImageNet-normalised and are converted to SigLIP's 0.5/0.5 inside.

CPU-only, a tiny random trunk through the ``_build_trunk`` seam: no download.
"""

from __future__ import annotations

import pytest
import torch

import bittrainer.siglip_model as sm
from bittrainer.model import (
    _infer_head_hidden_size,
    backbone_feature_hash,
    create_model,
    head_tail_logits,
    head_tail_parameters,
    pooled_features,
)
from bittrainer.tests.siglip_fixtures import TINY_DIM, install_tiny_trunk, tiny_trunk

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@pytest.fixture
def calls(monkeypatch):
    recorded: list = []
    install_tiny_trunk(monkeypatch, calls=recorded)
    return recorded


def _model(num_classes: int = 3, **kwargs):
    return create_model(
        model_size="siglip2_base", pretrained=False, num_classes=num_classes, **kwargs
    ).eval()


def test_create_model_dispatches_siglip_sizes(calls):
    model = _model()
    assert isinstance(model, sm.SigLIPClassifier)
    assert model.backbone_family == "siglip2"
    assert model.model_size == "siglip2_base"
    assert model.num_features == TINY_DIM
    assert calls == [("naflexvit_base_patch16_siglip.v2_webli", False, None)]


def test_pretrained_and_drop_path_reach_the_trunk_builder(calls):
    create_model(
        model_size="siglip2_so400m", pretrained=True, num_classes=2, drop_path_rate=0.1
    )
    assert calls == [("naflexvit_so400m_patch16_siglip.v2_webli", True, 0.1)]


def test_default_trunk_builder_asks_timm_for_a_headless_model(monkeypatch):
    seen: dict = {}

    def _create(name, **kwargs):
        seen["name"], seen["kwargs"] = name, kwargs
        return tiny_trunk()

    monkeypatch.setattr(sm.timm, "create_model", _create)
    sm._build_trunk(
        "naflexvit_base_patch16_siglip.v2_webli", pretrained=True, drop_path_rate=None
    )
    assert seen["name"] == "naflexvit_base_patch16_siglip.v2_webli"
    assert seen["kwargs"] == {"pretrained": True, "num_classes": 0}
    sm._build_trunk(
        "naflexvit_base_patch16_siglip.v2_webli", pretrained=False, drop_path_rate=0.2
    )
    assert seen["kwargs"] == {
        "pretrained": False,
        "num_classes": 0,
        "drop_path_rate": 0.2,
    }


def test_state_dict_is_trunk_plus_head_only(calls):
    keys = list(_model().state_dict())
    assert keys
    assert all(k.startswith(("trunk.", "head.")) for k in keys), keys
    # The normalisation constants are not weights.
    assert not any("scale" in k or "shift" in k for k in keys)
    assert "trunk.embeds.proj.weight" in keys
    assert {"head.fc.weight", "head.fc.bias"} <= set(keys)


def test_head_has_the_timm_classifier_head_surface(calls):
    head = _model().head
    for name in ("global_pool", "norm", "flatten", "pre_logits", "drop", "fc"):
        assert hasattr(head, name), name
    assert head.fc.in_features == TINY_DIM and head.fc.out_features == 3


def test_wrapper_matches_the_trunk_on_siglip_normalised_input(calls):
    """ImageNet-normalised in, SigLIP 0.5/0.5 inside: same pooled vector."""
    model = _model()
    torch.manual_seed(0)
    pixels = torch.rand(2, 3, 64, 48)
    imagenet = (pixels - _MEAN) / _STD
    with torch.no_grad():
        want = model.trunk((pixels - 0.5) / 0.5)
        got = model.forward_features(imagenet)
    assert got.shape == (2, TINY_DIM)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_cached_head_tail_reproduces_the_forward(calls):
    model = _model()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        pooled = pooled_features(model, x)
        assert pooled.shape == (2, TINY_DIM)
        assert torch.equal(head_tail_logits(model, pooled), model(x))


def test_backbone_hash_ignores_the_classifier_tail(calls):
    model = _model()
    before = backbone_feature_hash(model)
    with torch.no_grad():
        model.head.fc.weight.add_(1.0)
    assert backbone_feature_hash(model) == before
    with torch.no_grad():
        model.trunk.blocks[0].attn.qkv.weight.add_(0.01)
    assert backbone_feature_hash(model) != before


def test_mlp_head_matches_the_convnext_key_layout(calls):
    model = _model(head_hidden_size=16)
    state = model.state_dict()
    assert _infer_head_hidden_size(state) == 16
    assert state["head.fc.weight"].shape == (3, 16)
    names = {id(p) for p in head_tail_parameters(model)}
    assert names == {id(p) for p in model.head.parameters()}


def test_non_square_and_channels_last_inputs(calls):
    model = _model()
    x = torch.randn(2, 3, 96, 64)
    with torch.no_grad():
        plain = model(x)
        cl = model(x.contiguous(memory_format=torch.channels_last))
    assert plain.shape == (2, 3)
    torch.testing.assert_close(plain, cl)


def test_dtype_argument_casts_the_model(calls):
    model = create_model(
        model_size="siglip2_base", pretrained=False, num_classes=2, dtype=torch.bfloat16
    )
    assert next(model.parameters()).dtype == torch.bfloat16
    with torch.no_grad():
        out = model(torch.randn(1, 3, 32, 32, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16


def test_get_classifier_is_the_final_projection(calls):
    model = _model()
    assert model.get_classifier() is model.head.fc


def test_trainable_param_count(calls):
    model = _model()
    for p in model.parameters():
        p.requires_grad_(False)
    assert sm.trainable_param_count(model) == 0
    for p in model.head.parameters():
        p.requires_grad_(True)
    assert sm.trainable_param_count(model) == TINY_DIM * 3 + 3
