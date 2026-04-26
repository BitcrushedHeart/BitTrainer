"""Post-hoc temperature scaling for confidence calibration."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.optim import LBFGS
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def calibrate_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_iter: int = 50,
    lr: float = 0.01,
) -> float:
    """Learn optimal temperature *T* that minimises NLL on validation logits.

    Temperature scaling stretches the softmax distribution without
    changing the argmax (predictions stay the same), so it costs
    nothing in accuracy while dramatically improving calibration.
    """
    temperature = nn.Parameter(torch.tensor(1.5))
    nll = nn.CrossEntropyLoss()
    optimizer = LBFGS([temperature], lr=lr, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scaled = logits / temperature.clamp(min=0.01)
        loss = nll(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    t = float(temperature.item())
    t = max(0.1, min(t, 10.0))
    logger.info("Calibrated temperature: %.4f", t)
    return t


@torch.no_grad()
def collect_val_logits(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather raw logits and integer labels from a validation dataloader."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for images, labels in dataloader:
        images = images.to(device, dtype=dtype)
        labels = labels.to(device)
        with torch.amp.autocast(
            device_type=device.type, dtype=dtype,
            enabled=(dtype != torch.float32),
        ):
            logits = model(images)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_logits), torch.cat(all_labels)
