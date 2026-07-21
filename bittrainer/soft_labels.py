"""Soft-target construction and loss (Bitcrush ISSUE-0542).

Extracted verbatim from ``bittrainer.group_trainer``: the ordinal Gaussian
kernel, the perceptual (Oklab ΔE) kernel, soft-target assembly, and the
soft-cross-entropy reduction. ``group_trainer`` re-imports every name from here,
so the old import paths keep working and the objects stay identical.
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)


def _build_gaussian_kernel(
    num_classes: int,
    sigma: float = 1.0,
    *,
    none_index: int = -1,
) -> torch.Tensor:
    """Build a Gaussian smoothing kernel for ordinal classes.

    kernel[i, j] = exp(-(i-j)^2 / (2*sigma^2)), then normalised per row.

    When ``none_index >= 0`` the corresponding class (``__none__``) is treated
    as a separate semantic category, not a position on the ordinal scale.
    Its row and column are zeroed and the diagonal entry is set to 1, so no
    probability bleeds between ``__none__`` and its numeric neighbours during
    soft-target smoothing â€” without this, the model learns that ``__none__``
    is adjacent to the lowest ordinal class (e.g. ``__none__`` â†” "Augmented
    Breasts" or ``__none__`` â†” "0-year-old"), which corrupts predictions on
    visually-empty inputs.
    """
    indices = torch.arange(num_classes, dtype=torch.float32)
    diffs = indices.unsqueeze(0) - indices.unsqueeze(1)
    kernel = torch.exp(-diffs ** 2 / (2 * sigma ** 2))
    if 0 <= none_index < num_classes:
        kernel[none_index, :] = 0.0
        kernel[:, none_index] = 0.0
        kernel[none_index, none_index] = 1.0
    kernel = kernel / kernel.sum(dim=1, keepdim=True)
    return kernel


def _build_perceptual_kernel(
    class_names: list[str],
    centroids_by_name: dict,
    sigma: float,
    *,
    none_index: int = -1,
) -> torch.Tensor | None:
    """Gaussian kernel over perceptual (Oklab ΔE) centroid distance.

    kernel[i, j] = exp(-ΔE(c_i, c_j)^2 / (2*sigma^2)), row-normalised.
    ``__none__`` and any class without a centroid stay hard (identity row and
    zeroed column) — no probability bleeds to or from them. Returns None when
    fewer than two classes carry centroids (feature off).
    """
    n = len(class_names)
    pts: list[list[float] | None] = []
    for name in class_names:
        c = centroids_by_name.get(name)
        pts.append([float(v) for v in c] if c is not None and len(c) == 3 else None)
    if sum(1 for p in pts if p is not None) < 2 or sigma <= 0:
        return None
    kernel = torch.eye(n, dtype=torch.float32)
    for i in range(n):
        if pts[i] is None or i == none_index:
            continue
        for j in range(n):
            if j == i or pts[j] is None or j == none_index:
                continue
            de2 = sum((a - b) ** 2 for a, b in zip(pts[i], pts[j]))
            kernel[i, j] = math.exp(-de2 / (2.0 * sigma * sigma))
    return kernel / kernel.sum(dim=1, keepdim=True)


def _build_soft_targets(
    labels: torch.Tensor,
    num_classes: int,
    *,
    ordinal: bool = False,
    ordinal_sigma: float = 1.0,
    label_smoothing: float = 0.0,
    soft_aliases: dict | None = None,
    none_index: int = -1,
    device: torch.device = torch.device("cpu"),
    perceptual_kernel: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert integer labels to soft target vectors.

    1. Start with one-hot
    2. Apply the perceptual (ΔE-centroid) kernel when supplied — it REPLACES
       ordinal smoothing and label smoothing (Skin Tone V2: softness follows
       colour-space distance, never ordinal rank)
    3. Else ordinal Gaussian smoothing (if ordinal and sigma > 0),
       excluding ``none_index``
    4. Else global label smoothing for non-ordinal softmax groups, excluding
       ``none_index`` from both directions
    5. Apply soft aliases
    """
    batch_size = labels.shape[0]
    targets = torch.zeros(batch_size, num_classes, device=device)
    targets.scatter_(1, labels.unsqueeze(1), 1.0)

    if perceptual_kernel is not None:
        targets = targets @ perceptual_kernel.to(device)
    elif ordinal and num_classes > 2 and ordinal_sigma > 0:
        kernel = _build_gaussian_kernel(num_classes, sigma=ordinal_sigma, none_index=none_index).to(device)
        targets = targets @ kernel
    elif not ordinal and label_smoothing > 0:
        real_indices = [i for i in range(num_classes) if i != none_index]
        if len(real_indices) > 1:
            smoothed = targets.clone()
            real = torch.tensor(real_indices, device=device, dtype=torch.long)
            for idx in real_indices:
                mask = labels == idx
                if not mask.any():
                    continue
                peer_count = len(real_indices) - 1
                smoothed[mask, :] = 0.0
                smoothed[mask, idx] = 1.0 - label_smoothing
                peer_mask = torch.zeros(num_classes, device=device, dtype=torch.bool)
                peer_mask[real[real != idx]] = True
                row_idx = mask.nonzero(as_tuple=True)[0]
                col_idx = peer_mask.nonzero(as_tuple=True)[0]
                smoothed[row_idx.unsqueeze(1), col_idx] = label_smoothing / peer_count
        targets = smoothed

    # Soft aliases: redistribute weight
    if soft_aliases:
        for src_str, alias_list in soft_aliases.items():
            src = int(src_str)
            for tgt, weight in alias_list:
                mask = labels == src
                if mask.any():
                    transfer = targets[mask, src] * weight
                    targets[mask, src] -= transfer
                    targets[mask, tgt] += transfer

    # Re-normalise to sum to 1
    targets = targets / targets.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return targets


def _soft_ce_loss(
    log_probs: torch.Tensor,
    soft_targets: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy against soft targets, with optional focal + class weights.

    Single source of truth for the soft-target loss, so the full-FT loop, the
    head probe, and the MixUp path all reduce identically. ``focal_gamma`` adds
    ``(1 - p_t)^gamma`` modulation (p_t = expected prob under the soft target);
    ``class_weights`` applies the per-class weight as the expected weight under
    the soft target, normalised so the batch loss scale is invariant.
    """
    ce = -(soft_targets * log_probs).sum(dim=1)  # [N]
    if focal_gamma > 0:
        p_t = (soft_targets * log_probs.exp()).sum(dim=1).clamp(0.0, 1.0)
        ce = (1.0 - p_t).pow(focal_gamma) * ce
    if class_weights is not None:
        w = (soft_targets * class_weights.unsqueeze(0)).sum(dim=1)
        return (ce * w).sum() / w.sum().clamp(min=1e-8)
    return ce.mean()
