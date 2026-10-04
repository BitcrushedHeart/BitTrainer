"""SigLIP checkpoints load back through ``load_checkpoint`` (Bitcrush ISSUE-1106).

A SigLIP group checkpoint is an ordinary BitTrainer checkpoint: the full fp32
``state_dict`` (``trunk.*`` + ``head.*``) with ``model_size`` naming the SigLIP2
size. ``load_checkpoint`` must rebuild the wrapper from that metadata, or from
the weights alone (the ``trunk.embeds.proj.weight`` width) when the metadata is
missing or wrong, and must keep loading ConvNeXt checkpoints exactly as before.

CPU-only, tiny random trunks through the ``_build_trunk`` seam: no download.
"""

from __future__ import annotations

import pytest
import torch

from bittrainer.model import _infer_model_size, create_model, load_checkpoint
from bittrainer.siglip_model import SigLIPClassifier
from bittrainer.tests.siglip_fixtures import install_tiny_trunk


def _save(path, model, **meta):
    torch.save({"state_dict": model.state_dict(), "num_classes": 3, **meta}, path)
    return str(path)


def test_infer_model_size_reads_the_siglip_patch_projection():
    assert (
        _infer_model_size({"trunk.embeds.proj.weight": torch.zeros(1152, 768)})
        == "siglip2_so400m"
    )
    assert (
        _infer_model_size({"trunk.embeds.proj.weight": torch.zeros(768, 768)})
        == "siglip2_base"
    )
    assert _infer_model_size({"trunk.embeds.proj.weight": torch.zeros(32, 768)}) is None


def test_infer_model_size_still_reads_convnext_stems():
    assert _infer_model_size({"stem.0.weight": torch.zeros(40, 3, 4, 4)}) == "atto"
    assert _infer_model_size({"head.fc.weight": torch.zeros(3, 8)}) is None


def test_round_trip_restores_the_wrapper_and_its_outputs(monkeypatch, tmp_path):
    install_tiny_trunk(monkeypatch)
    model = create_model(
        model_size="siglip2_base", pretrained=False, num_classes=3
    ).eval()
    with torch.no_grad():
        model.head.fc.weight.normal_()
    path = _save(
        tmp_path / "best.pt",
        model,
        model_size="siglip2_base",
        backbone_family="siglip2",
    )

    loaded = load_checkpoint(path).eval()
    assert isinstance(loaded, SigLIPClassifier)
    assert loaded.model_size == "siglip2_base"
    assert set(loaded.state_dict()) == set(model.state_dict())
    x = torch.randn(2, 3, 64, 48)
    with torch.no_grad():
        assert torch.equal(model(x), loaded(x))


def test_mlp_head_round_trips(monkeypatch, tmp_path):
    install_tiny_trunk(monkeypatch)
    model = create_model(
        model_size="siglip2_so400m",
        pretrained=False,
        num_classes=3,
        head_hidden_size=16,
    ).eval()
    loaded = load_checkpoint(
        _save(tmp_path / "m.pt", model, model_size="siglip2_so400m")
    ).eval()
    assert loaded.head.hidden_size == 16
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(model(x), loaded(x))


def test_weights_name_the_size_when_the_metadata_is_wrong(monkeypatch, tmp_path):
    """A full-width checkpoint saved under a stale ConvNeXt size still loads."""
    install_tiny_trunk(monkeypatch, embed_dim=768, depth=1)
    model = create_model(
        model_size="siglip2_base", pretrained=False, num_classes=3
    ).eval()
    path = _save(tmp_path / "stale.pt", model, model_size="nano")
    loaded = load_checkpoint(path)
    assert isinstance(loaded, SigLIPClassifier) and loaded.model_size == "siglip2_base"


def test_bare_state_dict_is_recognised_from_the_weights(monkeypatch, tmp_path):
    install_tiny_trunk(monkeypatch, embed_dim=768, depth=1)
    model = create_model(
        model_size="siglip2_base", pretrained=False, num_classes=3
    ).eval()
    path = tmp_path / "bare.pt"
    torch.save(model.state_dict(), path)
    loaded = load_checkpoint(str(path), model_size="nano", num_classes=3)
    assert isinstance(loaded, SigLIPClassifier)


def test_unrecognisable_siglip_weights_fail_loudly(monkeypatch, tmp_path):
    """trunk.* weights of an unknown width under a ConvNeXt size: never guess."""
    install_tiny_trunk(monkeypatch)
    model = create_model(model_size="siglip2_base", pretrained=False, num_classes=3)
    with pytest.raises(ValueError, match="SigLIP"):
        load_checkpoint(_save(tmp_path / "odd.pt", model, model_size="nano"))


def test_convnext_checkpoint_loads_exactly_as_before(tmp_path):
    torch.manual_seed(0)
    model = create_model(model_size="atto", pretrained=False, num_classes=3).eval()
    loaded = load_checkpoint(_save(tmp_path / "c.pt", model, model_size="atto")).eval()
    assert not isinstance(loaded, SigLIPClassifier)
    assert set(loaded.state_dict()) == set(model.state_dict())
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(model(x), loaded(x))
