"""Approved ConvNeXt V2 pretrained source catalog."""

from __future__ import annotations

import pytest

from bittrainer.convnextv2_catalog import get_convnextv2_pretrained_source


@pytest.mark.parametrize(
    ("size", "tag"),
    [
        ("atto", "fcmae_ft_in1k"),
        ("femto", "fcmae_ft_in1k"),
        ("pico", "fcmae_ft_in1k"),
        ("nano", "fcmae_ft_in22k_in1k"),
        ("tiny", "fcmae_ft_in22k_in1k"),
        ("base", "fcmae_ft_in22k_in1k"),
        ("large", "fcmae_ft_in22k_in1k"),
    ],
)
def test_catalog_pins_exact_tagged_timm_models(size: str, tag: str):
    source = get_convnextv2_pretrained_source(size)

    assert source.model_name == f"convnextv2_{size}.{tag}"
    assert source.model_id == f"timm/convnextv2_{size}.{tag}"
    assert source.provider == "timm / Hugging Face"
    assert source.license == "CC-BY-NC-4.0"
    assert source.non_commercial_only is True
    assert source.source_url == f"https://huggingface.co/{source.model_id}"


def test_catalog_rejects_unknown_size():
    with pytest.raises(ValueError, match="Unknown ConvNeXt V2 size"):
        get_convnextv2_pretrained_source("gigantic")
