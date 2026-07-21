"""DataLoader collate functions (Bitcrush ISSUE-0542).

Extracted verbatim from ``bittrainer.group_trainer`` — the group trainer's
bucket / multi-label batch collates. ``trainer.py`` keeps its own bucket collate
(it carries a center-crop safety net), so these are the group-side pair only.
``group_trainer`` re-imports both names from here, keeping the objects identical.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def _collate_bucket_batch(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


def _collate_multilabel_batch(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return images, labels
