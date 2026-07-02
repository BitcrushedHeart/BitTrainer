"""Spatial-grid group support (e.g. Subject Location).

Spatial groups classify *where* the subject sits on an ``rows x cols`` grid;
each class is a set of covered cells (``cell_mask``). Two things break when
such a group is trained like an ordinary single-label classifier:

1. Horizontal flip augmentation mirrors the image but keeps the label, so
   left/right classes are actively taught to collapse into each other.
   :func:`build_hflip_class_map` + :func:`spatial_hflip_batch` flip the label
   *with* the image instead (the mirror class is derivable from the masks).

2. One softmax class per composition means classes share no parameters, so
   sparsely-labelled compositions never learn. :class:`SpatialCellFC` replaces
   the classifier ``fc`` with a per-cell head: ``num_cells`` Bernoulli logits,
   decoded to per-class scores as the log-likelihood of each class's exact
   mask. Every image trains every cell, so compositions share all their
   evidence; the module still outputs ``num_classes`` scores, keeping the
   trainer's selection/calibration and the suite's inference unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def mirror_cell_mask(mask: list[int], rows: int, cols: int) -> list[int]:
    """Cell mask of the horizontally mirrored composition."""
    del rows  # symmetry is horizontal: only the column index changes
    return sorted((c // cols) * cols + (cols - 1 - (c % cols)) for c in mask)


def build_hflip_class_map(
    cell_masks: list[list[int]], rows: int, cols: int
) -> list[int]:
    """Per-class index of the horizontally mirrored class.

    Symmetric compositions map to themselves. A class whose mirror mask is not
    in the class set maps to ``-1`` — such samples must not be flipped at all,
    because there is no label that describes the mirrored image.
    """
    key_to_index = {tuple(sorted(m)): i for i, m in enumerate(cell_masks)}
    return [
        key_to_index.get(tuple(mirror_cell_mask(m, rows, cols)), -1)
        for m in cell_masks
    ]


def spatial_hflip_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    flip_map: torch.Tensor,
    *,
    p: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random horizontal flip that remaps labels through ``flip_map``.

    Replaces the label-blind ``gpu_random_flip`` for spatial groups. Samples
    whose class has no mirror (``flip_map == -1``) are never flipped. Operates
    on the batch in place (same contract as ``gpu_random_flip``) and returns a
    new labels tensor.
    """
    flip_map = flip_map.to(labels.device)
    eligible = flip_map[labels] >= 0
    flip = (torch.rand(labels.shape[0], device=labels.device) < p) & eligible
    if flip.any():
        images[flip] = images[flip].flip(-1)
        labels = torch.where(flip, flip_map[labels].clamp(min=0), labels)
    return images, labels


class SpatialCellFC(nn.Module):
    """Cell-structured classifier ``fc``: per-cell logits, per-class decode.

    ``forward`` returns ``[B, num_classes]`` scores where class ``k``'s score
    is the log-likelihood of its exact cell mask under independent Bernoulli
    cells: ``sum_on logsigmoid(z_c) + sum_off logsigmoid(-z_c)``. Softmax over
    these scores is the posterior over compositions (uniform prior), so every
    downstream consumer — CE loss, argmax decode, temperature calibration,
    suite inference — treats them exactly like ordinary class logits.
    """

    def __init__(self, in_features: int, cell_masks: list[list[int]], num_cells: int):
        super().__init__()
        self.in_features = in_features
        self.num_cells = num_cells
        self.num_classes = len(cell_masks)
        self.cell_fc = nn.Linear(in_features, num_cells)
        masks = torch.zeros(len(cell_masks), num_cells)
        for i, mask in enumerate(cell_masks):
            for c in mask:
                masks[i, c] = 1.0
        self.register_buffer("masks", masks)

    @property
    def weight(self) -> torch.Tensor:
        """The trainable weight matrix (head probes read ``.weight.dtype``)."""
        return self.cell_fc.weight

    def cell_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.cell_fc(x)

    def decode_cells(self, cell_logits: torch.Tensor) -> torch.Tensor:
        z = cell_logits.float()
        masks = self.masks.float()
        return F.logsigmoid(z) @ masks.t() + F.logsigmoid(-z) @ (1.0 - masks).t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_cells(self.cell_fc(x))


def install_spatial_head(
    model: nn.Module, cell_masks: list[list[int]], num_cells: int
) -> None:
    """Replace ``model.head.fc`` with a :class:`SpatialCellFC`.

    Composes with both timm head layouts (plain linear and MLP ``pre_logits``):
    only the final projection is swapped. Device/dtype follow the layer being
    replaced.
    """
    fc = model.head.fc
    head = SpatialCellFC(fc.in_features, cell_masks, num_cells)
    head = head.to(device=fc.weight.device, dtype=fc.weight.dtype)
    model.head.fc = head
