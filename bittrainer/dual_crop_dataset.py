"""Dataset that loads paired crop + context images for dual-branch training."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


class DualCropDataset(Dataset):
    def __init__(
        self,
        crops_dir: str | Path,
        context_dir: str | Path,
        class_names: list[str],
        split: str = "train",
        crop_transform=None,
        context_transform=None,
    ):
        self.crops_dir = Path(crops_dir)
        self.context_dir = Path(context_dir)
        self.crop_transform = crop_transform
        self.context_transform = context_transform
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        self.samples: list[tuple[Path, Path, int]] = []
        for class_name in class_names:
            crop_split_dir = self.crops_dir / class_name / split
            ctx_split_dir = self.context_dir / class_name / split
            if not crop_split_dir.exists():
                continue
            for crop_path in sorted(crop_split_dir.iterdir()):
                if not crop_path.is_file() or crop_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                    continue
                ctx_path = ctx_split_dir / crop_path.name
                if not ctx_path.exists():
                    continue
                self.samples.append((crop_path, ctx_path, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        crop_path, ctx_path, label = self.samples[idx]
        crop_img = Image.open(crop_path).convert("RGB")
        ctx_img = Image.open(ctx_path).convert("RGB")

        if self.crop_transform:
            crop_img = self.crop_transform(crop_img)
        if self.context_transform:
            ctx_img = self.context_transform(ctx_img)

        return crop_img, ctx_img, label
