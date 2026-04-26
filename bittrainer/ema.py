"""Exponential Moving Average of model weights."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


class ModelEMA:
    """Maintains an exponential moving average of model parameters.

    At inference time, use ``ema.module`` (or ``ema.state_dict()``) for
    smoother predictions that generalise better than the raw training
    weights — especially on small datasets where per-batch noise is high.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.module.load_state_dict(state_dict)
