"""SigLIP2 text tower matching the timm SigLIP2 image trunk (Bitcrush Optimise P8).

:mod:`bittrainer.siglip_model` builds the image side from timm
(``naflexvit_*_patch16_siglip.v2_webli``). Those weights are the vision half of
Google's SigLIP2 NaFlex checkpoints; the text half lives in the matching Hugging
Face repo (``google/siglip2-<size>-patch16-naflex``) and is loaded here through
``transformers`` (an optional dependency: ``BitTrainer[text]``). Both towers
project into one embedding space of width ``SIGLIP_SPECS[size].embed_dim``, which
:func:`check_embedding_parity` verifies before a reward head is built on them.

When the matching text tower cannot be loaded (no ``transformers``, no weights,
no network) and ``fallback`` is on, :func:`resolve_reward_towers` falls back to
``siglip2_base`` on BOTH sides — image trunk and text tower — so the two never
come from different checkpoints. The tower reports ``fell_back`` /
``requested_size`` so the caller can record what actually ran.

The text model build is behind the ``_build_text_model`` seam, which tests
monkeypatch to a tiny encoder (``bittrainer.tests.text_fixtures``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from bittrainer.backbones import SIGLIP_SPECS, is_siglip

logger = logging.getLogger(__name__)

FALLBACK_SIZE = "siglip2_base"
#: SigLIP2's text encoder was trained with 64-token captions.
TEXT_MAX_LENGTH = 64


class TextTowerUnavailable(RuntimeError):
    """The matching SigLIP2 text encoder could not be built (missing dependency,
    weights or network)."""


@dataclass(frozen=True)
class TextTowerSpec:
    size: str
    model_id: str
    embed_dim: int
    max_length: int = TEXT_MAX_LENGTH


def text_spec(size: str) -> TextTowerSpec:
    """The text tower that pairs with SigLIP2 image size ``size``."""
    if not is_siglip(size):
        raise ValueError(
            f"{size!r} is not a SigLIP2 size; reward models need one of: "
            + ", ".join(SIGLIP_SPECS)
        )
    image = SIGLIP_SPECS[size]
    suffix = size.removeprefix("siglip2_")
    return TextTowerSpec(
        size=size,
        model_id=f"google/siglip2-{suffix}-patch16-naflex",
        embed_dim=image.embed_dim,
    )


def _build_text_model(model_id: str) -> tuple[nn.Module, Any]:
    """Load the Hugging Face SigLIP2 text encoder + tokenizer for ``model_id``.

    Raises :class:`TextTowerUnavailable` when ``transformers`` is missing or the
    weights cannot be fetched. Tests monkeypatch this seam.
    """
    try:
        from transformers import AutoTokenizer, Siglip2TextModel
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise TextTowerUnavailable(
            "transformers is not installed; install BitTrainer[text] for reward models"
        ) from exc
    try:
        encoder = Siglip2TextModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:  # noqa: BLE001 - any load failure means "unavailable"
        raise TextTowerUnavailable(f"could not load {model_id}: {exc}") from exc
    return encoder, tokenizer


class SigLIPTextTower(nn.Module):
    """Frozen SigLIP2 text encoder: ``encode(captions) -> [B, D]`` pooled embeddings."""

    def __init__(
        self,
        encoder: nn.Module,
        tokenizer: Any,
        spec: TextTowerSpec,
        *,
        requested_size: str | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.spec = spec
        self.requested_size = requested_size or spec.size
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.encoder.eval()
        super().train(False)
        self._embed_dim: int | None = None

    # -- identity -----------------------------------------------------------
    @property
    def size(self) -> str:
        return self.spec.size

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def fell_back(self) -> bool:
        return self.requested_size != self.spec.size

    @property
    def embed_dim(self) -> int:
        """Width of the pooled embedding, measured on a probe caption once."""
        if self._embed_dim is None:
            with torch.no_grad():
                self._embed_dim = int(self.encode(["probe"]).shape[-1])
        return self._embed_dim

    # -- behaviour ----------------------------------------------------------
    def train(self, mode: bool = True) -> SigLIPTextTower:
        """A frozen tower is a fixed feature extractor: it never leaves eval."""
        super().train(False)
        self.encoder.eval()
        return self

    def device(self) -> torch.device:
        try:
            return next(self.encoder.parameters()).device
        except StopIteration:  # pragma: no cover - parameter-free test doubles
            return torch.device("cpu")

    def encode(self, captions: list[str]) -> torch.Tensor:
        """Pooled text embeddings ``[len(captions), D]`` (float32, no grad)."""
        tokens = self.tokenizer(
            list(captions),
            padding="max_length",
            truncation=True,
            max_length=self.spec.max_length,
            return_tensors="pt",
        )
        device = self.device()
        tokens = {k: v.to(device) for k, v in tokens.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            out = self.encoder(**tokens)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:  # pragma: no cover - HF always exposes it for SigLIP
            pooled = out.last_hidden_state[:, -1]
        return pooled.float()


def create_text_tower(size: str, *, fallback: bool = True) -> SigLIPTextTower:
    """Build the text tower for SigLIP2 size ``size``.

    With ``fallback`` the ``siglip2_base`` tower is returned when the matching
    one cannot be built (the result then reports ``fell_back``); callers that
    need the image trunk to follow use :func:`resolve_reward_towers`.
    """
    spec = text_spec(size)
    try:
        encoder, tokenizer = _build_text_model(spec.model_id)
    except TextTowerUnavailable as exc:
        if not fallback or size == FALLBACK_SIZE:
            raise
        logger.warning(
            "SigLIP2 text tower %s unavailable (%s); falling back to %s",
            spec.model_id, exc, FALLBACK_SIZE,
        )
        base = text_spec(FALLBACK_SIZE)
        encoder, tokenizer = _build_text_model(base.model_id)
        return SigLIPTextTower(encoder, tokenizer, base, requested_size=size)
    return SigLIPTextTower(encoder, tokenizer, spec)


def resolve_reward_towers(size: str, *, fallback: bool = True) -> tuple[str, SigLIPTextTower]:
    """``(image_size, text_tower)`` for a requested size, falling back together.

    The returned image size is the one whose trunk pairs with the tower that
    actually loaded, so a reward head is never built across two checkpoints.
    """
    tower = create_text_tower(size, fallback=fallback)
    return tower.size, tower


def check_embedding_parity(image_model: nn.Module, tower: SigLIPTextTower) -> None:
    """Raise ``ValueError`` unless the image trunk and text tower share a width."""
    image_dim = int(getattr(image_model, "num_features"))
    text_dim = tower.embed_dim
    if image_dim != text_dim:
        raise ValueError(
            f"SigLIP2 image/text embedding width mismatch: image trunk is {image_dim} wide, "
            f"text tower {tower.model_id} is {text_dim} wide"
        )
