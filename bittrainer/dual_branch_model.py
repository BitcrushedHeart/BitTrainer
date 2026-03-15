"""Dual-branch ConvNeXt V2 model for crop + context classification."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from bittrainer.model import _MODEL_REGISTRY


class DualBranchConvNeXt(nn.Module):

    def __init__(
        self,
        backbone_variant: str = "nano",
        num_classes: int = 2,
        drop_rate: float = 0.3,
    ):
        super().__init__()
        model_name = _MODEL_REGISTRY.get(backbone_variant)
        if model_name is None:
            raise ValueError(f"Unknown backbone_variant '{backbone_variant}'. Valid: {list(_MODEL_REGISTRY.keys())}")

        self.crop_branch = timm.create_model(
            model_name, pretrained=True, num_classes=0,
        )
        self.context_branch = timm.create_model(
            model_name, pretrained=True, num_classes=0,
        )

        feature_dim = self.crop_branch.num_features
        fused_dim = feature_dim * 2

        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(fused_dim // 2, num_classes),
        )

        self.num_classes = num_classes
        self.backbone_variant = backbone_variant

    def forward(self, crop: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        crop_features = self.crop_branch(crop)
        context_features = self.context_branch(context)
        fused = torch.cat([crop_features, context_features], dim=1)
        return self.head(fused)

    def save_checkpoint(self, path: str, metadata: dict | None = None) -> None:
        checkpoint = {
            "classifier_mode": "dual_branch",
            "crop_branch": self.crop_branch.state_dict(),
            "context_branch": self.context_branch.state_dict(),
            "head": self.head.state_dict(),
            "metadata": {
                "backbone_variant": self.backbone_variant,
                "num_classes": self.num_classes,
                **(metadata or {}),
            },
        }
        torch.save(checkpoint, path)

    @classmethod
    def from_checkpoint(cls, path: str, device: torch.device | None = None) -> DualBranchConvNeXt:
        checkpoint = torch.load(path, map_location=device or "cpu", weights_only=True)
        meta = checkpoint["metadata"]
        model = cls(
            backbone_variant=meta["backbone_variant"],
            num_classes=meta["num_classes"],
        )
        model.crop_branch.load_state_dict(checkpoint["crop_branch"])
        model.context_branch.load_state_dict(checkpoint["context_branch"])
        model.head.load_state_dict(checkpoint["head"])
        if device:
            model = model.to(device)
        return model
