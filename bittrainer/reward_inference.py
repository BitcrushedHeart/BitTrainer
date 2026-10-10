"""Score (image, caption) pairs with a published reward checkpoint
(Bitcrush Optimise P8, ISSUE-1133).

A reward checkpoint stores only the head; the SigLIP2 trunk and text tower are
rebuilt from their pretrained weights by name (``image_model`` / ``text_model``)
and the images are embedded under the checkpoint's ``embed_preproc_sig`` recipe.
Raw rewards come back; Optimise maps them through the checkpoint's fixed
``norm_mapping`` (:func:`bittrainer.reward_head.apply_norm_mapping`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from bittrainer.pair_dataset import ImageEmbedder
from bittrainer.reward_head import RewardHead


def read_reward_checkpoint(path: str | Path) -> dict[str, Any]:
    """The checkpoint dict (metadata + head weights) without building any model."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or ckpt.get("trainer") != "pairwise_reward":
        raise ValueError(f"{path} is not a pairwise reward checkpoint")
    return ckpt


class RewardModel:
    """Frozen SigLIP2 towers + trained head, ready to score new pairs."""

    def __init__(self, ckpt: dict[str, Any], embedder: ImageEmbedder, tower: Any, head: RewardHead) -> None:
        self.ckpt = ckpt
        self.embedder = embedder
        self.tower = tower
        self.head = head.eval()

    @property
    def image_model(self) -> str:
        return str(self.ckpt["image_model"])

    @property
    def text_model(self) -> str:
        return str(self.ckpt["text_model"])

    @property
    def embed_preproc_sig(self) -> str:
        return str(self.ckpt["embed_preproc_sig"])

    @property
    def norm_mapping(self) -> dict[str, Any] | None:
        return self.ckpt.get("norm_mapping")

    @torch.no_grad()
    def score(self, image_paths: list[str], captions: list[str]) -> list[float]:
        """Raw rewards for each ``(image_paths[k], captions[k])``."""
        if len(image_paths) != len(captions):
            raise ValueError("image_paths and captions must align")
        if not image_paths:
            return []
        device = next(self.head.parameters()).device
        images = torch.from_numpy(self.embedder.embed_paths(list(image_paths))).to(device)
        texts = self.tower.encode(list(captions)).to(device)
        return [float(r) for r in self.head(images, texts).cpu().tolist()]


def load_reward_model(path: str | Path, *, device: str = "cpu") -> RewardModel:
    """Rebuild the towers named by the checkpoint and load the head."""
    from bittrainer.model import create_model
    from bittrainer.siglip_text import check_embedding_parity, create_text_tower

    ckpt = read_reward_checkpoint(path)
    dev = torch.device(device if not (str(device).startswith("cuda") and not torch.cuda.is_available()) else "cpu")
    trunk = create_model(model_size=ckpt["image_model"], pretrained=True, num_classes=0).to(dev).eval()
    tower = create_text_tower(ckpt["image_model"], fallback=False).to(dev)
    if tower.model_id != ckpt["text_model"]:
        raise ValueError(
            f"checkpoint was trained with text tower {ckpt['text_model']}, "
            f"but {ckpt['image_model']} pairs with {tower.model_id}"
        )
    check_embedding_parity(trunk, tower)
    embedder = ImageEmbedder(trunk, image_size=int(ckpt.get("image_size", 256)), device=dev)
    if embedder.preproc_sig != ckpt["embed_preproc_sig"]:
        raise ValueError(
            f"preprocessing drift: checkpoint expects {ckpt['embed_preproc_sig']}, "
            f"this build produces {embedder.preproc_sig}"
        )
    head = RewardHead.from_arch(ckpt["reward_head_arch"])
    head.load_state_dict(ckpt["head_state_dict"])
    return RewardModel(ckpt, embedder, tower, head.to(dev))
