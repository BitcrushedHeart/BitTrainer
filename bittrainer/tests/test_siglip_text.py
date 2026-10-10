"""The SigLIP2 text tower that pairs with the timm SigLIP2 image trunk
(Bitcrush Optimise P8, ISSUE-1133).

The image side is the existing :class:`SigLIPClassifier` trunk (timm NaFlex
ViT); the text side is the matching Hugging Face SigLIP2 text encoder, reached
through the ``_build_text_model`` seam so no test downloads weights. Parity
means the two towers share one embedding space: equal widths for the matching
size, and when the matching text tower cannot be loaded the pair falls back to
``siglip2_base`` on BOTH sides rather than mixing families.
"""

from __future__ import annotations

import os

import pytest
import torch

import bittrainer.siglip_text as st
from bittrainer.backbones import SIGLIP_SPECS
from bittrainer.model import create_model
from bittrainer.tests.siglip_fixtures import TINY_DIM, install_tiny_trunk
from bittrainer.tests.text_fixtures import install_tiny_text_model


def test_every_siglip_size_has_a_matching_text_tower_spec():
    for size, spec in SIGLIP_SPECS.items():
        text = st.text_spec(size)
        assert text.size == size
        assert text.embed_dim == spec.embed_dim
        assert text.model_id.startswith("google/siglip2-")
    with pytest.raises(ValueError):
        st.text_spec("nano")


def test_tower_encodes_captions_to_pooled_vectors(monkeypatch):
    calls = install_tiny_text_model(monkeypatch, embed_dim=TINY_DIM)
    tower = st.create_text_tower("siglip2_base")
    assert calls == ["google/siglip2-base-patch16-naflex"]
    assert tower.size == "siglip2_base"
    assert tower.embed_dim == TINY_DIM
    assert tower.fell_back is False
    with torch.no_grad():
        vectors = tower.encode(["a yellow mug", "a blue notebook on a desk"])
    assert vectors.shape == (2, TINY_DIM)
    assert torch.isfinite(vectors).all()
    # Deterministic for the same caption, different for different captions.
    with torch.no_grad():
        again = tower.encode(["a yellow mug"])
    torch.testing.assert_close(again[0], vectors[0])
    assert not torch.allclose(vectors[0], vectors[1])


def test_matching_towers_share_one_embedding_width(monkeypatch):
    install_tiny_trunk(monkeypatch, embed_dim=TINY_DIM)
    install_tiny_text_model(monkeypatch, embed_dim=TINY_DIM)
    image = create_model(model_size="siglip2_base", pretrained=False, num_classes=0).eval()
    tower = st.create_text_tower("siglip2_base")
    st.check_embedding_parity(image, tower)  # no raise
    with torch.no_grad():
        pooled = image.forward_features(torch.randn(1, 3, 64, 64))
        text = tower.encode(["anything"])
    assert pooled.shape[-1] == text.shape[-1]


def test_parity_check_refuses_mismatched_widths(monkeypatch):
    install_tiny_trunk(monkeypatch, embed_dim=TINY_DIM)
    install_tiny_text_model(monkeypatch, embed_dim=TINY_DIM * 2)
    image = create_model(model_size="siglip2_base", pretrained=False, num_classes=0)
    tower = st.create_text_tower("siglip2_base")
    with pytest.raises(ValueError, match="width"):
        st.check_embedding_parity(image, tower)


def test_unavailable_matching_tower_falls_back_to_base_on_both_sides(monkeypatch):
    calls = install_tiny_text_model(
        monkeypatch,
        embed_dim=TINY_DIM,
        unavailable={"google/siglip2-so400m-patch16-naflex"},
    )
    image_size, tower = st.resolve_reward_towers("siglip2_so400m")
    assert calls == [
        "google/siglip2-so400m-patch16-naflex",
        "google/siglip2-base-patch16-naflex",
    ]
    assert image_size == "siglip2_base"
    assert tower.size == "siglip2_base"
    assert tower.fell_back is True
    assert tower.requested_size == "siglip2_so400m"


def test_fallback_can_be_disabled(monkeypatch):
    install_tiny_text_model(
        monkeypatch,
        embed_dim=TINY_DIM,
        unavailable={"google/siglip2-so400m-patch16-naflex"},
    )
    with pytest.raises(st.TextTowerUnavailable):
        st.resolve_reward_towers("siglip2_so400m", fallback=False)


def test_text_tower_is_frozen_and_in_eval_mode(monkeypatch):
    install_tiny_text_model(monkeypatch, embed_dim=TINY_DIM)
    tower = st.create_text_tower("siglip2_base")
    assert not any(p.requires_grad for p in tower.parameters())
    assert not tower.training
    tower.train()  # a frozen tower stays a fixed feature extractor
    assert not tower.encoder.training


@pytest.mark.skipif(
    os.environ.get("BITTRAINER_REAL_SIGLIP") != "1",
    reason="downloads the real SigLIP2 base weights; set BITTRAINER_REAL_SIGLIP=1",
)
def test_real_base_towers_agree_on_a_simple_caption():
    """Opt-in parity with the real weights: the timm image trunk and the HF text
    tower must land in one space, so a matching caption scores above a
    mismatched one on a synthetic solid-colour image."""
    from PIL import Image

    from bittrainer.pair_dataset import ImageEmbedder

    image_size, tower = st.resolve_reward_towers("siglip2_base")
    image = create_model(model_size=image_size, pretrained=True, num_classes=0).eval()
    st.check_embedding_parity(image, tower)
    embedder = ImageEmbedder(image, image_size=256)
    red = Image.new("RGB", (256, 256), (220, 30, 30))
    vec = torch.from_numpy(embedder.embed_images([red]))
    with torch.no_grad():
        texts = tower.encode(["a plain red square", "a photo of a dog in a park"])
    sims = torch.nn.functional.cosine_similarity(vec, texts)
    assert sims[0] > sims[1]
