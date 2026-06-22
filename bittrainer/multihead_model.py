"""Multi-head ConvNeXt V2 model for ordinal size prediction.

A single ConvNeXt V2 backbone feeds a shared trunk, which splits into two ordinal heads:

* ``band`` — the band size (e.g. ``34``), the more definitive visual separator.
* ``size`` — the full US size (e.g. ``34DD``).

A band + size two-head design (a third "group" head is intentionally dropped). Owns its
own checkpoint format like :class:`bittrainer.dual_branch_model.DualBranchConvNeXt`.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from bittrainer.model import _MODEL_REGISTRY


class MultiHeadConvNeXt(nn.Module):
    def __init__(
        self,
        *,
        backbone_variant: str = "nano",
        n_bands: int,
        n_sizes: int,
        band_classes: list[str] | None = None,
        size_classes: list[str] | None = None,
        drop_rate: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        model_name = _MODEL_REGISTRY.get(backbone_variant)
        if model_name is None:
            raise ValueError(
                f"Unknown backbone_variant '{backbone_variant}'. Valid: {list(_MODEL_REGISTRY.keys())}"
            )

        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features

        self.shared = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
        )
        self.head_dropout = nn.Dropout(drop_rate)
        self.fc_band = nn.Linear(512, n_bands)
        self.fc_size = nn.Linear(512, n_sizes)

        self.backbone_variant = backbone_variant
        self.n_bands = n_bands
        self.n_sizes = n_sizes
        self.band_classes = band_classes or []
        self.size_classes = size_classes or []

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        shared = self.shared(features)
        return {
            "band": self.fc_band(self.head_dropout(shared)),
            "size": self.fc_size(self.head_dropout(shared)),
        }

    def save_checkpoint(self, path: str, metadata: dict | None = None) -> None:
        checkpoint = {
            "classifier_mode": "multihead",
            "backbone": self.backbone.state_dict(),
            "shared": self.shared.state_dict(),
            "fc_band": self.fc_band.state_dict(),
            "fc_size": self.fc_size.state_dict(),
            "metadata": {
                "backbone_variant": self.backbone_variant,
                "n_bands": self.n_bands,
                "n_sizes": self.n_sizes,
                "band_classes": self.band_classes,
                "size_classes": self.size_classes,
                **(metadata or {}),
            },
        }
        torch.save(checkpoint, path)

    @classmethod
    def from_checkpoint(cls, path: str, device: torch.device | None = None) -> MultiHeadConvNeXt:
        checkpoint = torch.load(path, map_location=device or "cpu", weights_only=True)
        meta = checkpoint["metadata"]
        model = cls(
            backbone_variant=meta["backbone_variant"],
            n_bands=meta["n_bands"],
            n_sizes=meta["n_sizes"],
            band_classes=meta.get("band_classes", []),
            size_classes=meta.get("size_classes", []),
            pretrained=False,
        )
        model.backbone.load_state_dict(checkpoint["backbone"])
        model.shared.load_state_dict(checkpoint["shared"])
        model.fc_band.load_state_dict(checkpoint["fc_band"])
        model.fc_size.load_state_dict(checkpoint["fc_size"])
        if device:
            model = model.to(device)
        return model
