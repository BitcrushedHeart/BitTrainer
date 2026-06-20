"""Focal loss and asymmetric loss for multi-label / multi-class training."""

from __future__ import annotations

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal loss that handles both hard labels and soft targets.

    When *targets* are 1-D integer indices, label smoothing is applied
    internally.  When *targets* are already 2-D soft distributions
    (e.g. from MixUp/CutMix or ordinal smoothing), they are used
    as-is with no additional smoothing.

    ``gamma`` controls focal modulation: ``(1 - p_t)^gamma`` down-weights
    easy/confident examples so that the model focuses its gradient budget
    on hard, ambiguous cases.
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        log_probs = torch.log_softmax(logits.float(), dim=1)
        probs = torch.exp(log_probs)

        if targets.dim() == 1:
            soft = torch.zeros_like(logits, dtype=torch.float32)
            soft.scatter_(1, targets.unsqueeze(1), 1.0)
            if self.label_smoothing > 0:
                soft = soft * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes
        else:
            soft = targets.float()

        p_t = (probs * soft).sum(dim=1)
        focal_weight = (1.0 - p_t).pow(self.gamma)
        ce = -(soft * log_probs).sum(dim=1)
        loss = focal_weight * ce
        return loss.mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for multi-label classification (Ridnik et al., ICCV 2021).

    Down-weights easy negatives more aggressively than easy positives, which
    addresses the dominant positive/negative imbalance in multi-label tasks
    (most images carry only a handful of the available labels). Drop-in
    replacement for ``nn.BCEWithLogitsLoss``.

    Default ``gamma_neg=4, gamma_pos=0, clip=0.05`` matches the paper's
    recommended starting point for COCO/Pascal-VOC-scale label spaces.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))

        pt = xs_pos * targets + xs_neg * (1.0 - targets)
        one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)

        loss = -(los_pos + los_neg) * one_sided_w
        return loss.mean()
