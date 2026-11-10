"""A GRN layer that stays in the precision the run trains in (Bitcrush ISSUE-1655).

ConvNeXt V2's Global Response Normalisation is
``x + bias + weight * x * x_n``, where ``x_n`` is each channel's norm over the
image divided by the mean of those norms. timm writes it as
``x + addcmul(bias, weight, x * x_n)``.

Under autocast the norm comes back in fp32, so ``x * x_n``, the ``addcmul`` and
the sum all run in fp32 too, on the widest tensor of the block (four times the
block's channels), and ``x * x_n`` is kept for the backward pass. Measured on a
T4 with ConvNeXt V2 Base in fp16 (224px, batch 64): 55 images/s and 15.5 GB.

The same value is ``x * (1 + weight * x_n) + bias``. The factor in brackets is
one number per image and channel, so it is computed in the weights' precision
and rounded once; the only wide operation left runs in the input's precision
and keeps nothing extra for the backward pass. Same card, same model:
77 images/s and 11.0 GB.

When the input already has the weights' precision (full precision training and
inference, a model cast to half) the stock expression runs, so those results do
not change by a bit.
"""

from __future__ import annotations

import torch
from timm.layers.grn import GlobalResponseNorm


class LeanGlobalResponseNorm(GlobalResponseNorm):
    """timm's GRN with the same weights, names and full-precision results."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == self.weight.dtype:
            return super().forward(x)
        # The norm sums squares over the whole image: always in the wider of the
        # two precisions (the weights' under autocast), whatever the device's
        # autocast rules would pick.
        wide = torch.promote_types(x.dtype, self.weight.dtype)
        x_g = torch.linalg.vector_norm(
            x, ord=2, dim=self.spatial_dim, keepdim=True, dtype=wide
        )
        x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        scale = (1.0 + self.weight.view(self.wb_shape) * x_n).to(x.dtype)
        return torch.addcmul(self.bias.view(self.wb_shape).to(x.dtype), x, scale)


def use_lean_grn(model: torch.nn.Module) -> int:
    """Switch every stock GRN layer in *model* to the lean one, in place.

    Weights, state-dict names and copies of the model are unaffected. Returns
    how many layers were switched (0 for a model that has none, or is already
    switched).
    """
    switched = 0
    for module in model.modules():
        if type(module) is GlobalResponseNorm:
            module.__class__ = LeanGlobalResponseNorm
            switched += 1
    return switched
