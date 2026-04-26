"""Focal loss with soft target and label smoothing support."""

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
