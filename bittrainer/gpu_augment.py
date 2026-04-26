"""GPU-side image normalisation and augmentation.

Operates on batched uint8 CHW tensors already on GPU.  Replaces the
CPU-side torchvision transforms pipeline used by DataLoader workers.
"""

from __future__ import annotations

import torch

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def gpu_normalize(
    batch: torch.Tensor,
    mean: torch.Tensor = _IMAGENET_MEAN,
    std: torch.Tensor = _IMAGENET_STD,
) -> torch.Tensor:
    """uint8 [B,3,H,W] → float32 [B,3,H,W], ImageNet-normalised."""
    mean = mean.to(batch.device, dtype=torch.float32).view(1, 3, 1, 1)
    std = std.to(batch.device, dtype=torch.float32).view(1, 3, 1, 1)
    return batch.float().div_(255.0).sub_(mean).div_(std)


def gpu_random_flip(batch: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Per-image random horizontal flip."""
    mask = torch.rand(batch.shape[0], device=batch.device) < p
    if mask.any():
        batch[mask] = batch[mask].flip(-1)
    return batch


def gpu_color_jitter(
    batch: torch.Tensor,
    brightness: float = 0.1,
    contrast: float = 0.1,
    saturation: float = 0.1,
) -> torch.Tensor:
    """Per-image random brightness, contrast, and saturation adjustment."""
    B = batch.shape[0]
    device = batch.device

    if brightness > 0:
        bf = 1.0 + (torch.rand(B, 1, 1, 1, device=device) * 2 * brightness - brightness)
        batch = batch * bf

    if contrast > 0:
        mean = batch.mean(dim=(2, 3), keepdim=True)
        cf = 1.0 + (torch.rand(B, 1, 1, 1, device=device) * 2 * contrast - contrast)
        batch = mean + (batch - mean) * cf

    if saturation > 0:
        gray = 0.2989 * batch[:, 0:1] + 0.5870 * batch[:, 1:2] + 0.1140 * batch[:, 2:3]
        sf = 1.0 + (torch.rand(B, 1, 1, 1, device=device) * 2 * saturation - saturation)
        batch = gray + (batch - gray) * sf

    return batch


def apply_train_augment(
    batch: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalize uint8 batch and apply training augmentation on GPU."""
    out = gpu_normalize(batch)
    out = gpu_random_flip(out)
    out = gpu_color_jitter(out, brightness=0.1, contrast=0.1, saturation=0.1)
    return out.to(dtype=dtype) if dtype != torch.float32 else out


def apply_val_transform(
    batch: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalize uint8 batch for validation (no augmentation)."""
    out = gpu_normalize(batch)
    return out.to(dtype=dtype) if dtype != torch.float32 else out
