"""Caption-conditioned pairwise reward head (Bitcrush Optimise P8, ISSUE-1133).

A reward model scores an image *under its caption*: ``r = head(i, t)`` where
``i`` is the frozen SigLIP2 image embedding and ``t`` the matching text
embedding (:mod:`bittrainer.siglip_text`). The head is a small MLP over the
four interaction blocks ``[i, t, i*t, |i-t|]`` (both inputs L2-normalised first),
trained with the pairwise ranking objective the spec names::

    loss = -log(sigmoid(r(caption, chosen) - r(caption, rejected)))
         = softplus(-(r_chosen - r_rejected))

so the model learns that the chosen image is preferable to the rejected one
under the SAME caption — never an image-only good/bad classifier.

A published model also carries a FIXED normalisation mapping from raw rewards
to the Suite's 0-100 score scale (:func:`fit_norm_mapping`, fitted once at
publish time over the training pool's rewards and stored in the checkpoint;
:func:`apply_norm_mapping` only ever reads it).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

REWARD_HEAD_FAMILY = "mlp4"
NORM_MAPPING_VERSION = 1
_MIN_SCALE = 1e-3


def pair_features(image: torch.Tensor, text: torch.Tensor, *, normalise: bool = True) -> torch.Tensor:
    """``[i, t, i*t, |i-t|]`` for ``[B, D]`` image and text embeddings -> ``[B, 4D]``."""
    if image.shape != text.shape:
        raise ValueError(
            f"image and text embeddings must share a shape, got {tuple(image.shape)} "
            f"and {tuple(text.shape)}"
        )
    if normalise:
        image = F.normalize(image, dim=-1)
        text = F.normalize(text, dim=-1)
    return torch.cat([image, text, image * text, (image - text).abs()], dim=-1)


class RewardHead(nn.Module):
    """``(image_embedding, text_embedding) -> scalar reward`` MLP."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 512,
        *,
        dropout: float = 0.1,
        normalise: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.normalise = bool(normalise)
        self.net = nn.Sequential(
            nn.Linear(4 * self.embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].bias)

    @property
    def arch(self) -> dict[str, Any]:
        """The architecture descriptor a checkpoint stores as ``reward_head_arch``."""
        return {
            "family": REWARD_HEAD_FAMILY,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "normalise": self.normalise,
        }

    @classmethod
    def from_arch(cls, arch: dict[str, Any]) -> RewardHead:
        family = arch.get("family", REWARD_HEAD_FAMILY)
        if family != REWARD_HEAD_FAMILY:
            raise ValueError(f"Unknown reward head family {family!r}")
        return cls(
            int(arch["embed_dim"]),
            int(arch.get("hidden_dim", 512)),
            dropout=float(arch.get("dropout", 0.1)),
            normalise=bool(arch.get("normalise", True)),
        )

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return self.net(pair_features(image, text, normalise=self.normalise)).squeeze(-1)


def pairwise_ranking_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """``softplus(-(r_c - r_r))`` averaged over the batch (= ``-log sigmoid(margin)``)."""
    return F.softplus(-(r_chosen - r_rejected)).mean()


def preference_accuracy(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> float | None:
    """Fraction of pairs whose chosen side scores strictly higher; ``None`` when empty."""
    if r_chosen.numel() == 0:
        return None
    return float((r_chosen > r_rejected).float().mean().item())


def preference_margin(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> float | None:
    """Mean ``r_chosen - r_rejected``; ``None`` when empty."""
    if r_chosen.numel() == 0:
        return None
    return float((r_chosen - r_rejected).mean().item())


def fit_norm_mapping(rewards: Sequence[float]) -> dict[str, Any]:
    """Fit the fixed raw-reward -> 0-100 mapping over a pool of rewards.

    ``score = 100 * sigmoid((r - centre) / scale)`` with ``centre`` the pool's
    median and ``scale`` its standard deviation (floored so a degenerate pool
    still maps). Fitted once at publish time and then frozen in the checkpoint.
    """
    values = [float(r) for r in rewards if math.isfinite(float(r))]
    if not values:
        raise ValueError("fit_norm_mapping needs at least one finite reward")
    centre = statistics.median(values)
    scale = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "type": "sigmoid",
        "version": NORM_MAPPING_VERSION,
        "centre": float(centre),
        "scale": float(max(scale, _MIN_SCALE)),
        "fitted_on": len(values),
    }


def apply_norm_mapping(mapping: dict[str, Any], reward: float) -> float:
    """Map one raw reward through a published mapping to the 0-100 scale."""
    if mapping.get("type") != "sigmoid":
        raise ValueError(f"Unknown norm mapping type {mapping.get('type')!r}")
    z = (float(reward) - float(mapping["centre"])) / float(mapping["scale"])
    z = max(-60.0, min(60.0, z))
    return 100.0 / (1.0 + math.exp(-z))
