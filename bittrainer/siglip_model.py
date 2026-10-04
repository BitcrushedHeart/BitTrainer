"""SigLIP2 group classifiers that look like a timm ConvNeXt (Bitcrush ISSUE-1106).

A :class:`SigLIPClassifier` wraps a timm NaFlex ViT (``naflexvit_*_patch16_siglip``,
``num_classes=0``) as ``trunk`` and adds a :class:`SigLIPHead` with the surface of
timm's ``NormMlpClassifierHead`` (``global_pool / norm / flatten / pre_logits /
drop / fc``). Everything BitTrainer and Engine wrote against a timm ConvNeXt
therefore keeps working:

* ``model.forward_features(x)`` returns the pooled ``[B, D]`` vector (SigLIP's
  MAP attention pool lives in the trunk), and the head's ``global_pool`` /
  ``norm`` / ``flatten`` are identities, so ``pooled_features`` and
  ``head_tail_logits`` read the cache point exactly as they do for ConvNeXt;
* the state dict is ``trunk.*`` + ``head.*`` (+ ``region_head.*``), so the
  backbone hash (everything but ``head.pre_logits`` / ``head.fc`` / the region
  head) covers the whole trunk and the embedding cache keys on it;
* spatial / ordinal-threshold heads replace ``head.fc`` and the region head
  attaches as ``model.region_head``, just like on ConvNeXt.

Inputs arrive ImageNet-normalised from every existing path (GPU augment, the
embedding cache, Engine's transforms and batch lanes); the wrapper converts them
to SigLIP's ``(x - 0.5) / 0.5`` inside :meth:`SigLIPClassifier.to_siglip_input`,
so no input pipeline needs to know which family it feeds.
"""

from __future__ import annotations

from collections import OrderedDict

import timm
import torch
import torch.nn as nn

from bittrainer.backbones import SIGLIP_FAMILY, siglip_spec

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class SigLIPHead(nn.Module):
    """Classifier head with timm's ``NormMlpClassifierHead`` attribute surface.

    ``global_pool``, ``norm`` and ``flatten`` are identities because the trunk
    already pools (MAP) and normalises; ``pre_logits`` is the optional MLP
    (``fc`` + GELU, the same ``head.pre_logits.fc.*`` keys a ConvNeXt MLP head
    has) and ``fc`` the final projection.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_size: int | None = None,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_size = hidden_size
        self.global_pool = nn.Identity()
        self.norm = nn.Identity()
        self.flatten = nn.Identity()
        if hidden_size:
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(self.in_features, int(hidden_size))),
                        ("act", nn.GELU()),
                    ]
                )
            )
            self.num_features = int(hidden_size)
        else:
            self.pre_logits = nn.Identity()
            self.num_features = self.in_features
        self.drop = nn.Dropout(drop_rate)
        self.fc = (
            nn.Linear(self.num_features, int(num_classes))
            if num_classes > 0
            else nn.Identity()
        )
        if isinstance(self.fc, nn.Linear):
            nn.init.trunc_normal_(self.fc.weight, std=0.02)
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.flatten(self.norm(self.global_pool(x)))
        x = self.drop(self.pre_logits(x))
        return x if pre_logits else self.fc(x)


class SigLIPClassifier(nn.Module):
    """``trunk`` (timm NaFlex ViT, MAP pool inside) + ``head`` (:class:`SigLIPHead`)."""

    backbone_family = SIGLIP_FAMILY

    def __init__(
        self,
        trunk: nn.Module,
        *,
        model_size: str,
        num_classes: int,
        head_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        self.trunk = trunk
        self.model_size = str(model_size)
        self.num_classes = int(num_classes)
        self.num_features = int(trunk.num_features)
        self.head = SigLIPHead(self.num_features, num_classes, head_hidden_size)
        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        # ImageNet-normalised x -> pixels p = x*std + mean -> SigLIP (p - 0.5)/0.5
        # = x * 2std + (2mean - 1). Non-persistent: constants, not weights.
        self.register_buffer("_in_scale", 2.0 * std, persistent=False)
        self.register_buffer("_in_shift", 2.0 * mean - 1.0, persistent=False)

    @property
    def depth(self) -> int:
        return len(self.trunk.blocks)

    def to_siglip_input(self, x: torch.Tensor) -> torch.Tensor:
        """Convert an ImageNet-normalised batch to SigLIP's 0.5/0.5 normalisation."""
        return x * self._in_scale.to(x.dtype) + self._in_shift.to(x.dtype)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled ``[B, D]`` trunk output (final norm + MAP attention pool)."""
        tokens = self.trunk.forward_features(self.to_siglip_input(x))
        return self.trunk.forward_head(tokens)

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def trunk_frozen(self) -> bool:
        """True when no trunk weight is trainable (head-only / ``finetune_blocks=0``)."""
        return not any(p.requires_grad for p in self.trunk.parameters())

    def train(self, mode: bool = True) -> SigLIPClassifier:
        """A fully frozen trunk is a fixed feature extractor: it stays in eval
        mode (no drop-path) while the heads train, so the features the heads
        learn on are the ones inference and the embedding cache see."""
        super().train(mode)
        if mode and self.trunk_frozen():
            self.trunk.eval()
        return self


def _build_trunk(
    model_name: str, *, pretrained: bool, drop_path_rate: float | None = None
) -> nn.Module:
    """Build the headless timm NaFlex ViT for ``model_name``.

    The seam tests monkeypatch to a tiny random trunk, so no test downloads the
    ~0.4-1.7 GB SigLIP2 weights.
    """
    extra: dict = {}
    if drop_path_rate is not None:
        extra["drop_path_rate"] = drop_path_rate
    return timm.create_model(model_name, pretrained=pretrained, num_classes=0, **extra)


def create_siglip_model(
    *,
    model_size: str,
    pretrained: bool = True,
    dtype: torch.dtype = torch.float32,
    num_classes: int = 2,
    head_hidden_size: int | None = None,
    drop_path_rate: float | None = None,
) -> SigLIPClassifier:
    """Create a :class:`SigLIPClassifier` for a SigLIP2 size string."""
    spec = siglip_spec(model_size)
    trunk = _build_trunk(
        spec.model_name, pretrained=pretrained, drop_path_rate=drop_path_rate
    )
    model = SigLIPClassifier(
        trunk,
        model_size=model_size,
        num_classes=num_classes,
        head_hidden_size=head_hidden_size,
    )
    if dtype != torch.float32:
        model = model.to(dtype=dtype)
    return model


def is_siglip_model(model: nn.Module) -> bool:
    return isinstance(model, SigLIPClassifier)


def prepare_for_finetune(model: SigLIPClassifier, finetune_blocks: int) -> int:
    """Set which SigLIP weights the in-loop optimiser trains.

    The head (and region head, when installed) always trains. ``finetune_blocks``
    > 0 additionally unfreezes the LAST ``finetune_blocks`` transformer blocks
    plus everything after them (final norm, MAP pool); 0 keeps the whole trunk
    frozen — the "head-only" setting crop-assisted groups train in the loop so
    their region head gets gradients. Values at or above the depth unfreeze the
    whole trunk, patch embedding included (a full fine-tune). Returns the
    number of unfrozen blocks.
    """
    trunk = model.trunk
    depth = len(trunk.blocks)
    n = max(0, min(int(finetune_blocks), depth))
    for param in model.parameters():
        param.requires_grad_(False)
    trainable: list[nn.Module] = [model.head]
    region_head = getattr(model, "region_head", None)
    if isinstance(region_head, nn.Module):
        trainable.append(region_head)
    if n:
        trainable.extend(trunk.blocks[depth - n :])
        tail = ["norm", "attn_pool", "fc_norm", "head"]
        if n == depth:
            tail += ["embeds", "norm_pre"]
        for name in tail:
            module = getattr(trunk, name, None)
            if isinstance(module, nn.Module):
                trainable.append(module)
    for module in trainable:
        for param in module.parameters():
            param.requires_grad_(True)
    return n


def build_siglip_llrd_param_groups(model: SigLIPClassifier, decay: float) -> list[dict]:
    """Layer-wise lr decay over the TRAINABLE SigLIP parameters.

    Block ``i`` of a depth-``D`` trunk sits at depth ``D - i`` (the last block
    moves at ``decay``), the patch embedding at ``D + 1``, and everything after
    the blocks — final norm, MAP pool, head, region head — at depth 0 with the
    full rate. Frozen parameters are left out entirely so the optimiser never
    allocates state for them (a frozen so400m trunk is ~430M parameters).
    Group names (``siglip.embeds`` / ``siglip.block.{i}`` / ``head``) mirror the
    ConvNeXt buckets so the weight-decay split can suffix them the same way.
    """
    depth = len(model.trunk.blocks)
    embeds: list[nn.Parameter] = []
    blocks: dict[int, list[nn.Parameter]] = {}
    head: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("trunk.blocks."):
            blocks.setdefault(int(name.split(".")[2]), []).append(param)
        elif name.startswith(("trunk.embeds.", "trunk.norm_pre.")):
            embeds.append(param)
        else:
            head.append(param)
    groups: list[dict] = []
    if embeds:
        groups.append(
            {"params": embeds, "lr": decay ** (depth + 1), "name": "siglip.embeds"}
        )
    for index in sorted(blocks):
        groups.append(
            {
                "params": blocks[index],
                "lr": decay ** (depth - index),
                "name": f"siglip.block.{index}",
            }
        )
    if head:
        groups.append({"params": head, "lr": 1.0, "name": "head"})
    return groups


def trainable_param_count(model: nn.Module) -> int:
    """Number of parameters the optimiser will train (``requires_grad``)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
