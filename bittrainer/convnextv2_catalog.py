"""Canonical ConvNeXt V2 pretrained sources used by BitTrainer and Engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConvNeXtV2PretrainedSource:
    size: str
    family: str
    model_name: str
    model_id: str
    provider: str
    license: str
    source_url: str
    non_commercial_only: bool


def _source(size: str, tag: str) -> ConvNeXtV2PretrainedSource:
    model_name = f"convnextv2_{size}.{tag}"
    model_id = f"timm/{model_name}"
    return ConvNeXtV2PretrainedSource(
        size=size,
        family="ConvNeXt V2",
        model_name=model_name,
        model_id=model_id,
        provider="timm / Hugging Face",
        license="CC-BY-NC-4.0",
        source_url=f"https://huggingface.co/{model_id}",
        non_commercial_only=True,
    )


CONVNEXTV2_PRETRAINED_SOURCES: dict[str, ConvNeXtV2PretrainedSource] = {
    size: _source(size, tag)
    for size, tag in {
        "atto": "fcmae_ft_in1k",
        "femto": "fcmae_ft_in1k",
        "pico": "fcmae_ft_in1k",
        "nano": "fcmae_ft_in22k_in1k",
        "tiny": "fcmae_ft_in22k_in1k",
        "base": "fcmae_ft_in22k_in1k",
        "large": "fcmae_ft_in22k_in1k",
        "huge": "fcmae_ft_in22k_in1k",
    }.items()
}


def get_convnextv2_pretrained_source(size: str) -> ConvNeXtV2PretrainedSource:
    try:
        return CONVNEXTV2_PRETRAINED_SOURCES[size]
    except KeyError as exc:
        known = ", ".join(CONVNEXTV2_PRETRAINED_SOURCES)
        raise ValueError(
            f"Unknown ConvNeXt V2 size {size!r}. Expected one of: {known}"
        ) from exc
