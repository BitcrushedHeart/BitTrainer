"""Single Prodigy_adv optimizer factory (Bitcrush ISSUE-0542).

All trainers build their optimizer through one factory carrying the canonical
Prodigy_adv + Kourkoutas hyperparameters; the per-trainer ``_make_optimizer``
copies (group / binary) and the multihead / dual-branch inline constructions
delegate here so the optimizer story is defined in exactly one place.
"""

from __future__ import annotations

import logging

import torch.nn as nn
from adv_optm import Prodigy_adv

from bittrainer.model import build_llrd_param_groups

logger = logging.getLogger(__name__)


def make_optimizer(
    model: nn.Module,
    *,
    llrd: bool = False,
    llrd_decay: float = 0.8,
) -> Prodigy_adv:
    """Build the canonical Prodigy_adv (Kourkoutas-β, cautious weight decay).

    ``llrd`` splits the model into layer-wise learning-rate-decayed param groups
    via :func:`bittrainer.model.build_llrd_param_groups` (per-group multiplier on
    Prodigy's adapted step ``d``); otherwise a single flat param group is used.
    """
    if llrd:
        params = build_llrd_param_groups(model, llrd_decay)
    else:
        params = model.parameters()
    return Prodigy_adv(
        params, lr=1.0, d_coef=0.9,
        weight_decay=0.01, betas=(0.9, 0.999),
        kourkoutas_beta=True, k_warmup_steps=50,
        cautious_wd=True,
    )
