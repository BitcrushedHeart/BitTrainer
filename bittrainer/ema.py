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

    Uses an adaptive decay warmup: the effective decay is the minimum of the
    target ``decay`` and ``(1 + n) / (warmup + n)`` after ``n`` updates. This
    keeps EMA tracking close to the live model in the early steps (when the
    model is moving fast and the running average should follow) and converges
    to the target decay once the warmup buffer is exhausted. Without this,
    short training runs (a few hundred steps) would never get past the random
    initialisation because target decay 0.9999 has a time constant of 10,000
    steps.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup: int = 10):
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.warmup = warmup
        self.n_updates = 0

    def _effective_decay(self) -> float:
        return min(self.decay, (1.0 + self.n_updates) / (self.warmup + self.n_updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.n_updates += 1
        d = self._effective_decay()
        for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(model_p.data, alpha=1.0 - d)

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.module.load_state_dict(state_dict)

    def full_state_dict(self) -> dict:
        """EMA weights *plus* the adaptive-decay bookkeeping.

        ``state_dict()`` alone loses ``n_updates``, which the warmup schedule in
        ``_effective_decay`` depends on — restoring it would reset the effective
        decay back to the early, fast-tracking regime and let a resumed EMA jump
        toward the live weights. Used by the training-state backup so a resumed
        run continues the EMA exactly where it left off.
        """
        return {
            "state_dict": self.module.state_dict(),
            "n_updates": self.n_updates,
            "decay": self.decay,
            "warmup": self.warmup,
        }

    def load_full_state_dict(self, data: dict) -> None:
        """Restore weights and adaptive-decay bookkeeping from :meth:`full_state_dict`."""
        self.module.load_state_dict(data["state_dict"])
        self.n_updates = int(data.get("n_updates", self.n_updates))
        if "decay" in data:
            self.decay = float(data["decay"])
        if "warmup" in data:
            self.warmup = int(data["warmup"])
