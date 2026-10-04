"""Backbone size registry (Bitcrush ISSUE-1106).

One size namespace covers both backbone families: the historical ConvNeXt V2
sizes ("atto".."huge") and the SigLIP2 sizes ("siglip2_so400m", "siglip2_base").
``bittrainer.backbones`` is the single place that maps a size to its family, its
timm model and its pretrained repo. It is stdlib-only so Engine can keep a
torch-free copy (with a parity test) for config, routing and UI decisions.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from bittrainer import backbones
from bittrainer.convnextv2_catalog import CONVNEXTV2_PRETRAINED_SOURCES


def test_siglip_sizes_map_to_the_timm_siglip2_naflex_models():
    assert backbones.SIGLIP_SIZES == ("siglip2_so400m", "siglip2_base")
    so400m = backbones.siglip_spec("siglip2_so400m")
    assert so400m.model_name == "naflexvit_so400m_patch16_siglip.v2_webli"
    assert (so400m.embed_dim, so400m.depth, so400m.patch_size) == (1152, 27, 16)
    base = backbones.siglip_spec("siglip2_base")
    assert base.model_name == "naflexvit_base_patch16_siglip.v2_webli"
    assert (base.embed_dim, base.depth, base.patch_size) == (768, 12, 16)


def test_default_siglip_size_is_so400m():
    assert backbones.DEFAULT_SIGLIP_SIZE == "siglip2_so400m"


@pytest.mark.parametrize("size", ["siglip2_so400m", "siglip2_base"])
def test_siglip_sizes_are_the_siglip_family(size):
    assert backbones.is_siglip(size)
    assert backbones.backbone_family(size) == backbones.SIGLIP_FAMILY == "siglip2"


@pytest.mark.parametrize("size", sorted(CONVNEXTV2_PRETRAINED_SOURCES))
def test_convnext_sizes_are_the_convnext_family(size):
    assert not backbones.is_siglip(size)
    assert backbones.backbone_family(size) == backbones.CONVNEXT_FAMILY == "convnext_v2"


@pytest.mark.parametrize("size", [None, "", "mystery"])
def test_unknown_or_missing_sizes_are_the_legacy_convnext_family(size):
    """Old rows and checkpoints carry no family: they are ConvNeXt."""
    assert not backbones.is_siglip(size)
    assert backbones.backbone_family(size) == "convnext_v2"


def test_pretrained_repo_covers_both_families():
    assert (
        backbones.pretrained_repo("siglip2_so400m")
        == "timm/naflexvit_so400m_patch16_siglip.v2_webli"
    )
    assert (
        backbones.pretrained_repo("siglip2_base")
        == "timm/naflexvit_base_patch16_siglip.v2_webli"
    )
    for size, source in CONVNEXTV2_PRETRAINED_SOURCES.items():
        assert backbones.pretrained_repo(size) == source.model_id


def test_pretrained_repo_refuses_an_unknown_size():
    with pytest.raises(ValueError, match="mystery"):
        backbones.pretrained_repo("mystery")


def test_siglip_spec_refuses_a_convnext_size():
    with pytest.raises(ValueError, match="nano"):
        backbones.siglip_spec("nano")


def test_siglip_size_from_dim():
    assert backbones.siglip_size_from_dim(1152) == "siglip2_so400m"
    assert backbones.siglip_size_from_dim(768) == "siglip2_base"
    assert backbones.siglip_size_from_dim(32) is None


def test_registry_names_exist_in_the_installed_timm():
    """The pinned timm must ship the architectures and their pretrained tags."""
    timm = pytest.importorskip("timm")
    pretrained = set(timm.list_pretrained("naflexvit_*siglip*"))
    for size in backbones.SIGLIP_SIZES:
        name = backbones.siglip_spec(size).model_name
        assert name in pretrained, name
        assert timm.is_model(name.split(".")[0])


def test_module_is_stdlib_only():
    """Engine mirrors this module torch-free; keep heavy imports out."""
    path = Path(backbones.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    allowed_local = {"bittrainer.convnextv2_catalog"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            root = name.split(".")[0]
            assert (
                name in allowed_local
                or root in sys.stdlib_module_names
                or root == "__future__"
            ), f"non-stdlib import in backbones.py: {name}"
