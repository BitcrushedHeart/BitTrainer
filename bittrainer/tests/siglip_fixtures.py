"""Tiny SigLIP trunks for tests (Bitcrush ISSUE-1106).

Tests never download SigLIP2 weights: ``install_tiny_trunk`` monkeypatches the
``bittrainer.siglip_model._build_trunk`` seam so every SigLIP size builds a
random-init NaFlex ViT a few kilobytes big, with the same structure (patch 16,
MAP pooling, gelu_tanh MLPs) as the real ``naflexvit_*_patch16_siglip`` models.
"""

from __future__ import annotations

import torch

TINY_DIM = 32
TINY_DEPTH = 4


def tiny_trunk(
    *, embed_dim: int = TINY_DIM, depth: int = TINY_DEPTH, drop_path_rate: float = 0.0
):
    from timm.models.naflexvit import NaFlexVit, NaFlexVitCfg

    cfg = NaFlexVitCfg(
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=2,
        global_pool="map",
        act_layer="gelu_tanh",
        drop_path_rate=drop_path_rate,
    )
    return NaFlexVit(cfg=cfg, num_classes=0)


def install_tiny_trunk(
    monkeypatch, *, embed_dim: int = TINY_DIM, depth: int = TINY_DEPTH, calls=None
):
    """Route every SigLIP trunk build to :func:`tiny_trunk`.

    ``calls`` (a list) records each build's ``(model_name, pretrained,
    drop_path_rate)`` so tests can assert what the factory asked timm for.
    """
    import bittrainer.siglip_model as sm

    def _build(
        model_name: str, *, pretrained: bool, drop_path_rate: float | None = None
    ):
        if calls is not None:
            calls.append((model_name, pretrained, drop_path_rate))
        torch.manual_seed(1234)
        return tiny_trunk(
            embed_dim=embed_dim, depth=depth, drop_path_rate=drop_path_rate or 0.0
        )

    monkeypatch.setattr(sm, "_build_trunk", _build)
    return _build
