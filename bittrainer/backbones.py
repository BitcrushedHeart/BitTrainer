"""Backbone size registry: one size namespace for every backbone family.

Group models are sized by a single string (``GroupTrainConfig.backbone_variant``,
checkpoint ``model_size``). The historical ConvNeXt V2 sizes ("atto".."huge")
keep their meaning; the SigLIP2 family adds its own sizes ("siglip2_so400m",
"siglip2_base") to the same namespace instead of a separate family column,
because ConvNeXt already has a size called "base" and a family column would be
misread by every caller that only looks at the size (Bitcrush ISSUE-1106).

This module is stdlib-only on purpose: Engine keeps a torch-free copy of the
same rules for configuration, routing and UI decisions, pinned by a parity test.
"""

from __future__ import annotations

from dataclasses import dataclass

from bittrainer.convnextv2_catalog import CONVNEXTV2_PRETRAINED_SOURCES

CONVNEXT_FAMILY = "convnext_v2"
SIGLIP_FAMILY = "siglip2"


@dataclass(frozen=True)
class SigLIPSpec:
    """A SigLIP2 size: the timm NaFlex ViT that implements it and its shape."""

    size: str
    model_name: str
    embed_dim: int
    depth: int
    patch_size: int = 16


SIGLIP_SPECS: dict[str, SigLIPSpec] = {
    "siglip2_so400m": SigLIPSpec(
        size="siglip2_so400m",
        model_name="naflexvit_so400m_patch16_siglip.v2_webli",
        embed_dim=1152,
        depth=27,
    ),
    "siglip2_base": SigLIPSpec(
        size="siglip2_base",
        model_name="naflexvit_base_patch16_siglip.v2_webli",
        embed_dim=768,
        depth=12,
    ),
}

SIGLIP_SIZES: tuple[str, ...] = tuple(SIGLIP_SPECS)
DEFAULT_SIGLIP_SIZE = "siglip2_so400m"


def is_siglip(size: str | None) -> bool:
    """True for a SigLIP2 size string."""
    return size in SIGLIP_SPECS


def backbone_family(size: str | None) -> str:
    """Family of a size string.

    Anything that is not a SigLIP2 size is ConvNeXt V2 — including missing or
    unknown sizes, which only ever come from rows and checkpoints written
    before there was more than one family.
    """
    return SIGLIP_FAMILY if is_siglip(size) else CONVNEXT_FAMILY


def siglip_spec(size: str) -> SigLIPSpec:
    """The :class:`SigLIPSpec` for a SigLIP2 size; ``ValueError`` otherwise."""
    try:
        return SIGLIP_SPECS[size]
    except KeyError as exc:
        raise ValueError(
            f"Unknown SigLIP2 size {size!r}. Expected one of: {', '.join(SIGLIP_SIZES)}"
        ) from exc


def pretrained_repo(size: str) -> str:
    """Hugging Face repo id the size's pretrained weights download from."""
    if is_siglip(size):
        return f"timm/{SIGLIP_SPECS[size].model_name}"
    source = CONVNEXTV2_PRETRAINED_SOURCES.get(size)
    if source is None:
        known = ", ".join([*CONVNEXTV2_PRETRAINED_SOURCES, *SIGLIP_SIZES])
        raise ValueError(f"Unknown backbone size {size!r}. Expected one of: {known}")
    return source.model_id


def siglip_size_from_dim(dim: int) -> str | None:
    """SigLIP2 size whose embedding width is ``dim``, or ``None``.

    Used to recognise a SigLIP checkpoint from its weights alone (the width of
    ``trunk.embeds.proj.weight``) when its metadata is missing.
    """
    for spec in SIGLIP_SPECS.values():
        if spec.embed_dim == int(dim):
            return spec.size
    return None
