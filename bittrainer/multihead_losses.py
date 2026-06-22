"""Losses for the multi-head size model.

* :class:`VolumeSoftLabelLoss` — soft targets over the size head where each class gets
  credit proportional to its volume closeness to the true size, so *sister sizes*
  (equal volume) are treated as near-equivalent instead of as unrelated classes.
* :class:`BandOrdinalSoftLabelLoss` — Gaussian-ish ordinal soft labels over the band head
  (band is a clean ordinal scale).
* :class:`BandConsistencyLoss` — penalises disagreement between the band head and the band
  *implied* by the size head, so the two heads stay internally consistent.

All losses honour ``ignore_index`` (``__none__`` / unlabelled-for-this-head samples).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VolumeSoftLabelLoss(nn.Module):
    """Cross-entropy against volume-distance soft targets for the size head."""

    def __init__(self, volumes: list[float], *, temperature: float = 2.0, ignore_index: int = -1):
        super().__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.register_buffer("volumes", torch.tensor(volumes, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = targets != self.ignore_index
        if not mask.any():
            return logits.new_tensor(0.0)

        logits_m = logits[mask]
        targets_m = targets[mask]
        volumes = self.volumes.to(logits.device)

        target_volumes = volumes[targets_m]  # [B]
        dist = torch.abs(volumes.unsqueeze(0) - target_volumes.unsqueeze(1))  # [B, C]
        soft = F.softmax(-dist / self.temperature, dim=1)

        log_probs = F.log_softmax(logits_m.float(), dim=1)
        return -(soft * log_probs).sum(dim=1).mean()


class BandOrdinalSoftLabelLoss(nn.Module):
    """Cross-entropy against ordinal soft targets for the band head."""

    def __init__(self, num_bands: int, *, temperature: float = 1.5, ignore_index: int = -1):
        super().__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.register_buffer("positions", torch.arange(num_bands, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = targets != self.ignore_index
        if not mask.any():
            return logits.new_tensor(0.0)

        logits_m = logits[mask]
        targets_m = targets[mask].float()
        positions = self.positions.to(logits.device)

        dist = torch.abs(positions.unsqueeze(0) - targets_m.unsqueeze(1))  # [B, n_bands]
        soft = F.softmax(-dist / self.temperature, dim=1)

        log_probs = F.log_softmax(logits_m.float(), dim=1)
        return -(soft * log_probs).sum(dim=1).mean()


class BandConsistencyLoss(nn.Module):
    """Penalise disagreement between the band head and the band implied by the size head.

    ``size_to_band`` maps each size-class index to its band-head index (or ``-1`` for
    ``__none__`` / unparseable). The implied band distribution is the size-probability mass
    summed per band; the loss is MSE between it and the band head's distribution.
    """

    def __init__(self, size_to_band: list[int], num_bands: int, *, weight: float = 0.5):
        super().__init__()
        self.weight = weight
        self.num_bands = num_bands
        self.register_buffer("size_to_band", torch.tensor(size_to_band, dtype=torch.long))

    def forward(self, band_logits: torch.Tensor, size_logits: torch.Tensor) -> torch.Tensor:
        band_probs = F.softmax(band_logits.float(), dim=1)
        size_probs = F.softmax(size_logits.float(), dim=1)

        batch = size_probs.shape[0]
        implied = torch.zeros(batch, self.num_bands, device=size_probs.device)
        size_to_band = self.size_to_band.to(size_probs.device)
        for band_idx in range(self.num_bands):
            col_mask = (size_to_band == band_idx).float()
            implied[:, band_idx] = (size_probs * col_mask.unsqueeze(0)).sum(dim=1)

        implied = implied / (implied.sum(dim=1, keepdim=True) + 1e-8)
        return self.weight * F.mse_loss(implied, band_probs)
