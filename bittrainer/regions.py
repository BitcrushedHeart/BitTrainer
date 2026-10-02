"""Region proposals for crop-assisted groups (Bitcrush ISSUE-0993).

A crop-assisted group is labelled with *regions*: boxes the user drew on an
image, each with a region type ("Face") and usually a class. The classes train
the ordinary classifier (the crop files join the dataset as extra samples); the
boxes train :class:`RegionHead`, a small detector riding on the same trunk, so
the shipped model can propose those crops itself at inference.

The head is CenterNet-style ("Objects as Points", Zhou et al. 2019) on a
stride-16 feature map fused from the trunk's stage 3 and its upsampled stage 4:
a per-region-type centre heatmap plus a box size and a sub-cell offset at every
cell. That shape is chosen for this data, not for benchmark accuracy: several
boxes per image need no anchors and no NMS (a 3x3 max-pool finds the peaks), the
head is a few hundred thousand parameters, and it learns from the hundreds of
boxes a person can realistically draw.

Supervision is *partial*. An image nobody drew on is unknown to the detector,
never a negative; drawing a region type on an image asserts that every instance
of that type on it has been drawn. :class:`RegionTargets.supervised` carries
that per (image, type), and :func:`region_loss` masks by it.

The head is attached as ``model.region_head`` so its weights ride in the
ordinary ``state_dict`` (EMA, SWA, soup and checkpoint saves need no special
case) while ``model.forward`` stays exactly the classifier it always was —
:func:`forward_with_regions` is the only path that runs the head.

Boxes everywhere in this module are normalised ``(x1, y1, x2, y2)`` in the frame
of the tensor the model sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

REGION_HEAD_PREFIX = "region_head."
REGION_STRIDE = 16
REGION_HEAD_CHANNELS = 128

DEFAULT_SCORE_THRESHOLD = 0.3
DEFAULT_MAX_REGIONS = 8

# CenterNet's weighting of the three terms: the size regression is in feature
# cells (tens of units), so it is scaled down against the heatmap focal loss.
_SIZE_LOSS_WEIGHT = 0.1
_OFFSET_LOSS_WEIGHT = 1.0
_FOCAL_ALPHA = 2.0
_FOCAL_BETA = 4.0
# Prior probability the heatmap starts at (bias = -log((1 - p) / p)), so an
# untrained head does not open with a focal loss dominated by background cells.
_HEATMAP_PRIOR = 0.1
# IoU a Gaussian-radius-shifted box must still reach with its ground truth.
_GAUSSIAN_MIN_OVERLAP = 0.7


class RegionOutput(NamedTuple):
    """Raw head output at stride 16: heatmap logits ``[B, K, H, W]``, box size in
    cells ``[B, 2, H, W]`` (width, height) and centre offset ``[B, 2, H, W]``."""

    heatmap: torch.Tensor
    size: torch.Tensor
    offset: torch.Tensor


class RegionProposal(NamedTuple):
    type: int
    score: float
    box: tuple[float, float, float, float]


class RegionHead(nn.Module):
    """Centre-point detector over stage-3 features fused with upsampled stage 4."""

    def __init__(
        self,
        in_channels: tuple[int, int],
        num_types: int,
        channels: int = REGION_HEAD_CHANNELS,
    ) -> None:
        super().__init__()
        if num_types < 1:
            raise ValueError("a region head needs at least one region type")
        c3, c4 = int(in_channels[0]), int(in_channels[1])
        self.in_channels = (c3, c4)
        self.num_types = int(num_types)
        self.channels = int(channels)
        groups = math.gcd(8, self.channels)
        self.lateral3 = nn.Conv2d(c3, self.channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(c4, self.channels, kernel_size=1)
        self.tower = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, self.channels),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, self.channels),
            nn.GELU(),
        )
        self.heatmap = nn.Conv2d(self.channels, self.num_types, kernel_size=1)
        self.size = nn.Conv2d(self.channels, 2, kernel_size=1)
        self.offset = nn.Conv2d(self.channels, 2, kernel_size=1)
        nn.init.constant_(self.heatmap.bias, -math.log((1.0 - _HEATMAP_PRIOR) / _HEATMAP_PRIOR))
        nn.init.zeros_(self.size.bias)
        nn.init.zeros_(self.offset.bias)

    def forward(self, stage3: torch.Tensor, stage4: torch.Tensor) -> RegionOutput:
        fused = self.lateral3(stage3) + F.interpolate(
            self.lateral4(stage4), size=stage3.shape[-2:], mode="nearest"
        )
        feat = self.tower(fused)
        return RegionOutput(self.heatmap(feat), self.size(feat), self.offset(feat))


def has_region_head(model: nn.Module) -> bool:
    return isinstance(getattr(model, "region_head", None), RegionHead)


def _trunk_channels(model: nn.Module) -> tuple[int, int]:
    info = getattr(model, "feature_info", None)
    if not info or len(info) < 4:
        raise RuntimeError("region head needs a four-stage ConvNeXt trunk")
    return int(info[2]["num_chs"]), int(info[3]["num_chs"])


def install_region_head(
    model: nn.Module, num_types: int, channels: int = REGION_HEAD_CHANNELS
) -> RegionHead:
    """Attach a fresh :class:`RegionHead` as ``model.region_head``.

    Device and dtype follow the trunk. ``model.forward`` is not touched: the
    classifier keeps emitting exactly what it did.
    """
    head = RegionHead(_trunk_channels(model), num_types, channels)
    ref = next(model.parameters())
    model.region_head = head.to(device=ref.device, dtype=ref.dtype)
    return model.region_head


def region_head_ckpt_meta(model: nn.Module, type_names: list[str] | None) -> dict:
    """Checkpoint metadata ``load_checkpoint`` rebuilds the head from.

    Empty for a model without a region head, so every save site can spread it
    unconditionally.
    """
    if not has_region_head(model):
        return {}
    head: RegionHead = model.region_head
    names = [str(n) for n in (type_names or [])][: head.num_types]
    names += [f"region_{i}" for i in range(len(names), head.num_types)]
    return {
        "region_head": {
            "num_types": head.num_types,
            "types": names,
            "channels": head.channels,
            "in_channels": list(head.in_channels),
            "stride": REGION_STRIDE,
        }
    }


def forward_with_regions(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, RegionOutput]:
    """Classifier logits and region maps from ONE trunk pass.

    Unrolls timm's ``forward_features`` (``stem -> stages -> norm_pre``) to tap
    the stage-3 and stage-4 maps; the logits are bit-identical to ``model(x)``.
    """
    if not has_region_head(model):
        raise RuntimeError("model has no region head installed")
    feat = model.stem(x)
    taps: list[torch.Tensor] = []
    for stage in model.stages:
        feat = stage(feat)
        taps.append(feat)
    regions = model.region_head(taps[2], taps[3])
    logits = model.forward_head(model.norm_pre(feat))
    return logits, regions


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


@dataclass
class RegionTargets:
    """One batch's region supervision — the collate's third element.

    ``boxes`` / ``batch_index`` / ``types`` list every drawn box in the batch.
    ``supervised[b, k]`` says image ``b`` was reviewed for region type ``k``
    (drawn or dismissed), so its whole ``k`` heatmap is trustworthy.
    ``cls_mask[b]`` is False for a detection-only image: it has boxes but no
    usable class label, so the classification loss must skip it.
    """

    boxes: torch.Tensor
    batch_index: torch.Tensor
    types: torch.Tensor
    supervised: torch.Tensor
    cls_mask: torch.Tensor

    def to(self, device: torch.device | str) -> RegionTargets:
        return RegionTargets(
            boxes=self.boxes.to(device),
            batch_index=self.batch_index.to(device),
            types=self.types.to(device),
            supervised=self.supervised.to(device),
            cls_mask=self.cls_mask.to(device),
        )

    def pin_memory(self) -> RegionTargets:
        return RegionTargets(
            boxes=self.boxes.pin_memory(),
            batch_index=self.batch_index.pin_memory(),
            types=self.types.pin_memory(),
            supervised=self.supervised.pin_memory(),
            cls_mask=self.cls_mask.pin_memory(),
        )

    @property
    def detector_rows(self) -> torch.Tensor:
        """``[B]`` bool: rows whose pixels must stay where the boxes say."""
        return self.supervised.any(dim=1)


def collate_region_targets(infos: list[dict | None]) -> RegionTargets | None:
    """Stack the per-item region info a region-aware dataset returns.

    ``None`` for a batch of plain items, which is what keeps every ordinary
    group's batches the two-tuples they have always been.
    """
    present = [i for i in infos if i is not None]
    if not present:
        return None
    num_types = int(present[0]["supervised"].shape[0])
    boxes: list[torch.Tensor] = []
    batch_index: list[torch.Tensor] = []
    types: list[torch.Tensor] = []
    supervised = torch.zeros(len(infos), num_types, dtype=torch.bool)
    cls_mask = torch.ones(len(infos), dtype=torch.bool)
    for row, info in enumerate(infos):
        if info is None:
            continue
        n = int(info["boxes"].shape[0])
        if n:
            boxes.append(info["boxes"].to(torch.float32))
            types.append(info["types"].to(torch.long))
            batch_index.append(torch.full((n,), row, dtype=torch.long))
        supervised[row] = info["supervised"]
        cls_mask[row] = bool(info["cls"])
    return RegionTargets(
        boxes=torch.cat(boxes) if boxes else torch.zeros(0, 4, dtype=torch.float32),
        batch_index=torch.cat(batch_index) if batch_index else torch.zeros(0, dtype=torch.long),
        types=torch.cat(types) if types else torch.zeros(0, dtype=torch.long),
        supervised=supervised,
        cls_mask=cls_mask,
    )


class RegionMaps(NamedTuple):
    """Dense targets for one batch. ``cells`` is ``[M, 3]`` (batch, y, x) of each
    box's centre cell; ``size`` / ``offset`` are ``[M, 2]`` in cells."""

    heat: torch.Tensor
    cells: torch.Tensor
    size: torch.Tensor
    offset: torch.Tensor
    types: torch.Tensor


def _gaussian_radius(height: float, width: float, min_overlap: float = _GAUSSIAN_MIN_OVERLAP) -> int:
    """Largest centre shift (in cells) that keeps IoU >= ``min_overlap`` (CornerNet)."""
    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    r1 = (b1 - math.sqrt(max(0.0, b1 * b1 - 4.0 * a1 * c1))) / (2.0 * a1)
    a2 = 4.0
    b2 = 2.0 * (height + width)
    c2 = (1.0 - min_overlap) * width * height
    r2 = (b2 - math.sqrt(max(0.0, b2 * b2 - 4.0 * a2 * c2))) / (2.0 * a2)
    a3 = 4.0 * min_overlap
    b3 = -2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1.0) * width * height
    r3 = (b3 + math.sqrt(max(0.0, b3 * b3 - 4.0 * a3 * c3))) / (2.0 * a3)
    return max(0, int(min(r1, r2, r3)))


def build_region_maps(targets: RegionTargets, height: int, width: int) -> RegionMaps:
    """Splat every box as a Gaussian onto its type's ``height x width`` heatmap."""
    device = targets.supervised.device
    batch, num_types = targets.supervised.shape
    heat = torch.zeros(batch, num_types, height, width, dtype=torch.float32, device=device)
    cells: list[list[int]] = []
    sizes: list[list[float]] = []
    offsets: list[list[float]] = []
    ys = torch.arange(height, dtype=torch.float32, device=device).view(-1, 1)
    xs = torch.arange(width, dtype=torch.float32, device=device).view(1, -1)
    rows = zip(
        targets.boxes.detach().cpu().double().tolist(),
        targets.batch_index.tolist(),
        targets.types.tolist(),
        strict=True,
    )
    for (x1, y1, x2, y2), b, k in rows:
        w_cells = max(0.0, x2 - x1) * width
        h_cells = max(0.0, y2 - y1) * height
        cx = min(max((x1 + x2) / 2.0 * width, 0.0), width - 1e-4)
        cy = min(max((y1 + y2) / 2.0 * height, 0.0), height - 1e-4)
        ix, iy = int(cx), int(cy)
        radius = _gaussian_radius(max(h_cells, 1.0), max(w_cells, 1.0))
        sigma = max(radius, 1) / 3.0 + 1e-6
        blob = torch.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2.0 * sigma * sigma))
        heat[b, k] = torch.maximum(heat[b, k], blob)
        cells.append([b, iy, ix])
        sizes.append([w_cells, h_cells])
        offsets.append([cx - ix, cy - iy])
    return RegionMaps(
        heat=heat,
        cells=torch.tensor(cells, dtype=torch.long, device=device).reshape(-1, 3),
        size=torch.tensor(sizes, dtype=torch.float32, device=device).reshape(-1, 2),
        offset=torch.tensor(offsets, dtype=torch.float32, device=device).reshape(-1, 2),
        types=targets.types,
    )


def region_loss(output: RegionOutput, targets: RegionTargets) -> torch.Tensor:
    """Penalty-reduced focal loss on the heatmaps + L1 on size/offset at centres.

    Only ``supervised`` (image, type) heatmaps count. With nothing supervised
    the result is a zero that still depends on the head, so ``backward`` works
    on a batch of plain samples.
    """
    heatmap = output.heatmap.float()
    height, width = heatmap.shape[-2:]
    mask = targets.supervised.to(heatmap.device)
    if not bool(mask.any()):
        return heatmap.sum() * 0.0 + output.size.sum() * 0.0 + output.offset.sum() * 0.0

    maps = build_region_maps(targets, height, width)
    prob = torch.sigmoid(heatmap).clamp(1e-4, 1.0 - 1e-4)
    positive = maps.heat.eq(1.0)
    weight = mask[:, :, None, None].float()
    pos_term = torch.log(prob) * (1.0 - prob).pow(_FOCAL_ALPHA) * positive.float()
    neg_term = (
        torch.log(1.0 - prob)
        * prob.pow(_FOCAL_ALPHA)
        * (1.0 - maps.heat).pow(_FOCAL_BETA)
        * (~positive).float()
    )
    num_pos = max(1.0, float((positive.float() * weight).sum().item()))
    focal = -((pos_term + neg_term) * weight).sum() / num_pos

    if maps.cells.shape[0] == 0:
        return focal + output.size.sum() * 0.0 + output.offset.sum() * 0.0
    b, y, x = maps.cells.unbind(dim=1)
    pred_size = output.size.float()[b, :, y, x]
    pred_offset = output.offset.float()[b, :, y, x]
    boxes = float(maps.cells.shape[0])
    size_l1 = F.l1_loss(pred_size, maps.size, reduction="sum") / boxes
    offset_l1 = F.l1_loss(pred_offset, maps.offset, reduction="sum") / boxes
    return focal + _SIZE_LOSS_WEIGHT * size_l1 + _OFFSET_LOSS_WEIGHT * offset_l1


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


@torch.no_grad()
def decode_regions(
    output: RegionOutput,
    *,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    max_regions: int = DEFAULT_MAX_REGIONS,
) -> list[list[RegionProposal]]:
    """Per image, the peak cells of every type's heatmap as scored boxes.

    A cell is a proposal when it is the maximum of its 3x3 neighbourhood and
    its score reaches ``score_threshold``; the best ``max_regions`` are kept,
    highest score first.
    """
    prob = torch.sigmoid(output.heatmap.float())
    batch, num_types, height, width = prob.shape
    peaks = prob * (F.max_pool2d(prob, kernel_size=3, stride=1, padding=1) == prob).float()
    flat = peaks.reshape(batch, -1)
    k = min(max(1, int(max_regions)), flat.shape[1])
    scores, index = flat.topk(k, dim=1)
    size = output.size.float().cpu()
    offset = output.offset.float().cpu()
    scores, index = scores.cpu(), index.cpu()

    proposals: list[list[RegionProposal]] = []
    for b in range(batch):
        row: list[RegionProposal] = []
        for score, idx in zip(scores[b].tolist(), index[b].tolist(), strict=True):
            if score < score_threshold:
                continue
            kind, cell = divmod(idx, height * width)
            y, x = divmod(cell, width)
            cx = (x + float(offset[b, 0, y, x])) / width
            cy = (y + float(offset[b, 1, y, x])) / height
            half_w = max(0.0, float(size[b, 0, y, x])) / width / 2.0
            half_h = max(0.0, float(size[b, 1, y, x])) / height / 2.0
            box = (
                min(1.0, max(0.0, cx - half_w)),
                min(1.0, max(0.0, cy - half_h)),
                min(1.0, max(0.0, cx + half_w)),
                min(1.0, max(0.0, cy + half_h)),
            )
            row.append(RegionProposal(type=int(kind), score=float(score), box=box))
        proposals.append(row)
    return proposals


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def region_hflip_batch(
    images: torch.Tensor,
    targets: RegionTargets,
    *,
    p: float = 0.5,
) -> tuple[torch.Tensor, RegionTargets]:
    """Random horizontal flip that mirrors the boxes with the pixels.

    Replaces the label-blind ``gpu_random_flip`` for batches that carry region
    targets (the same job ``spatial_hflip_batch`` does for spatial groups).
    Images are flipped in place; the targets come back as a new object.
    """
    flip = torch.rand(images.shape[0], device=images.device) < p
    if not bool(flip.any()):
        return images, targets
    images[flip] = images[flip].flip(-1)
    boxes = targets.boxes.clone()
    if boxes.shape[0]:
        mirrored = flip.to(targets.batch_index.device)[targets.batch_index]
        x1 = boxes[mirrored, 0].clone()
        boxes[mirrored, 0] = 1.0 - boxes[mirrored, 2]
        boxes[mirrored, 2] = 1.0 - x1
    return images, RegionTargets(
        boxes=boxes,
        batch_index=targets.batch_index,
        types=targets.types,
        supervised=targets.supervised,
        cls_mask=targets.cls_mask,
    )
