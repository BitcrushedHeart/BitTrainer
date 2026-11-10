"""A stand-in trunk for tests of the backbone trainer's loop (Bitcrush ISSUE-1655).

Even the smallest real trunk (atto) keeps every core busy for seconds over a few
training steps, and a file of such tests adds up to minutes. What those tests
check (batch sizes, counters, progress frames, backups, resume, what is written
beside the candidate) does not depend on what the trunk computes. They run on
one small convolution instead: about 1,500 weights, a whole two-epoch run in a
fraction of a second, so they are safe to run anywhere.

A test that needs a real trunk is marked to run only where a GPU is visible.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import bittrainer.generic.tasks.backbone_task as bt


class StubTrunk(nn.Module):
    """Images in, an 8-number feature per image out: the shape of a timm trunk
    built with ``num_classes=0``, and nothing more."""

    num_features = 8

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, self.num_features, kernel_size=8, stride=8)
        # Draws random numbers while training, as a real trunk's drop-path does.
        self.drop = nn.Dropout(0.1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.drop(self.stem(images).mean(dim=(2, 3)))


def use_stub_trunk(monkeypatch) -> None:
    """Make the backbone task build a :class:`StubTrunk` instead of a ConvNeXt."""
    monkeypatch.setattr(bt, "create_model", lambda **_kwargs: StubTrunk())
