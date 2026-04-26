"""Batch-level MixUp and CutMix augmentation."""

from __future__ import annotations

import random

import numpy as np
import torch


def _to_soft(
    labels: torch.Tensor,
    num_classes: int,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Convert integer labels to one-hot soft targets.

    If *labels* is already 2-D (soft targets), returns as-is.
    """
    if labels.dim() == 2:
        return labels.float()
    soft = torch.zeros(
        labels.shape[0], num_classes,
        device=labels.device, dtype=torch.float32,
    )
    soft.scatter_(1, labels.unsqueeze(1), 1.0)
    if label_smoothing > 0:
        soft = soft * (1.0 - label_smoothing) + label_smoothing / num_classes
    return soft


def mixup(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    alpha: float = 0.2,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate pairs of images and their labels."""
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(images.shape[0], device=images.device)
    mixed = lam * images + (1.0 - lam) * images[index]
    soft = _to_soft(labels, num_classes, label_smoothing)
    mixed_labels = lam * soft + (1.0 - lam) * soft[index]
    return mixed, mixed_labels


def cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    alpha: float = 1.0,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut and paste rectangular patches between image pairs."""
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(images.shape[0], device=images.device)

    _, _, h, w = images.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = max(1, int(w * cut_rat))
    cut_h = max(1, int(h * cut_rat))

    cx = np.random.randint(w)
    cy = np.random.randint(h)
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(w, cx + cut_w // 2)
    y2 = min(h, cy + cut_h // 2)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    # Adjust lambda to actual area ratio
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (w * h)
    soft = _to_soft(labels, num_classes, label_smoothing)
    mixed_labels = lam * soft + (1.0 - lam) * soft[index]
    return mixed, mixed_labels


def apply_mixing(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    *,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly apply MixUp or CutMix (50/50 chance)."""
    if random.random() < 0.5:
        return mixup(images, labels, num_classes, alpha=mixup_alpha, label_smoothing=label_smoothing)
    return cutmix(images, labels, num_classes, alpha=cutmix_alpha, label_smoothing=label_smoothing)
