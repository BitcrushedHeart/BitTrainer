"""DataLoader collate functions (Bitcrush ISSUE-0542).

Extracted from ``bittrainer.group_trainer`` — the group trainer's bucket /
multi-label batch collates. ``trainer.py`` keeps its own bucket collate
(it carries a center-crop safety net), so these are the group-side pair only.
``group_trainer`` re-imports both names from here, keeping the objects identical.

Crop-assisted groups (Bitcrush ISSUE-0993): a region-aware ``GroupDataset``
returns a fourth item element, the sample's region info. Both collates then
return ``(images, labels, RegionTargets)``; a batch of plain items stays the
``(images, labels)`` pair every other consumer unpacks.
"""

from __future__ import annotations

import logging

import torch

from bittrainer.regions import RegionTargets, collate_region_targets

logger = logging.getLogger(__name__)


def _region_targets(batch) -> RegionTargets | None:
    if len(batch[0]) < 4:
        return None
    return collate_region_targets([item[3] if len(item) > 3 else None for item in batch])


def _collate_bucket_batch(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    targets = _region_targets(batch)
    if targets is None:
        return images, labels
    return images, labels, targets


def _collate_multilabel_batch(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    targets = _region_targets(batch)
    if targets is None:
        return images, labels
    return images, labels, targets
