"""Single Prodigy_adv optimizer factory (Bitcrush ISSUE-0542).

All trainers build their optimizer through one factory carrying the canonical
Prodigy_adv + Kourkoutas hyperparameters; the per-trainer ``_make_optimizer``
copies (group / binary) and the multihead / dual-branch inline constructions
delegate here so the optimizer story is defined in exactly one place.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from adv_optm import Prodigy_adv

from bittrainer.model import build_llrd_param_groups

logger = logging.getLogger(__name__)

# Canonical Prodigy_adv weight decay (matrix params); 1-D params (norm scales,
# biases, GRN gains) are excluded from decay when ``wd_exclusions`` is on.
_WEIGHT_DECAY = 0.01


def _split_no_decay_flat(model: nn.Module) -> list[dict]:
    """Two flat groups: matrix params (decayed) + 1-D params (no decay).

    ConvNeXtV2's norm scales, biases and GRN gains are all 1-D; decaying them
    pulls scale/shift parameters toward zero for no benefit. The decayed group
    omits ``weight_decay`` so it inherits the optimizer default (0.01).
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for param in model.parameters():
        (no_decay if param.ndim < 2 else decay).append(param)
    groups: list[dict] = []
    if decay:
        groups.append({"params": decay, "name": "decay"})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, "name": "no_decay"})
    return groups


def _split_no_decay_llrd(base_groups: list[dict]) -> list[dict]:
    """Split each LLRD depth bucket into decay / no-decay subgroups.

    Each subgroup keeps its bucket's ``lr`` multiplier (so both halves of a depth
    move at the same rate); the no-decay half zeroes ``weight_decay`` and suffixes
    ``/no_decay`` onto the bucket name so callers can still recover the base depth
    via ``name.split("/")[0]``.
    """
    groups: list[dict] = []
    for group in base_groups:
        params = list(group["params"])
        decay = [p for p in params if p.ndim >= 2]
        no_decay = [p for p in params if p.ndim < 2]
        name = str(group.get("name", ""))
        lr = group["lr"]
        if decay:
            groups.append({"params": decay, "lr": lr, "name": name})
        if no_decay:
            groups.append(
                {
                    "params": no_decay,
                    "lr": lr,
                    "weight_decay": 0.0,
                    "name": f"{name}/no_decay",
                }
            )
    return groups


def make_optimizer(
    model: nn.Module,
    *,
    llrd: bool = False,
    llrd_decay: float = 0.8,
    wd_exclusions: bool = False,
) -> Prodigy_adv:
    """Build the canonical Prodigy_adv (Kourkoutas-β, cautious weight decay).

    ``llrd`` splits the model into layer-wise learning-rate-decayed param groups
    via :func:`bittrainer.model.build_llrd_param_groups` (per-group multiplier on
    Prodigy's adapted step ``d``); otherwise a single flat param group is used.

    ``wd_exclusions`` moves every 1-D parameter (norm scales, biases, GRN gains)
    into a ``weight_decay=0.0`` group. The bare call (both flags off) stays a
    single flat group so existing callers are byte-identical; trainers opt in via
    config. The changed param-group layout is why callers fold an optimizer
    identity into their resume fingerprint (see :func:`optimizer_identity`).
    """
    if llrd:
        base_groups = build_llrd_param_groups(model, llrd_decay)
        params = _split_no_decay_llrd(base_groups) if wd_exclusions else base_groups
    elif wd_exclusions:
        params = _split_no_decay_flat(model)
    else:
        params = model.parameters()
    return Prodigy_adv(
        params,
        lr=1.0,
        d_coef=0.9,
        weight_decay=_WEIGHT_DECAY,
        betas=(0.9, 0.999),
        kourkoutas_beta=True,
        k_warmup_steps=50,
        cautious_wd=True,
    )


def optimizer_identity(*, llrd: bool, llrd_decay: float, wd_exclusions: bool) -> str:
    """Stable identity string for the optimizer's param-group layout.

    Folded into the run fingerprint so a backup whose optimizer state has a
    different group layout (a stale flat-Prodigy run vs a new LLRD+no-decay run)
    cleanly mismatches and starts fresh instead of crashing inside
    ``optimizer.load_state_dict``. ``(False, *, False)`` returns the historical
    ``"Prodigy_adv"`` so untouched runs keep their existing fingerprint; the
    decay VALUE is part of the identity because it changes the per-group ``lr``.
    """
    parts = ["Prodigy_adv"]
    if llrd:
        parts.append(f"llrd{llrd_decay:g}")
    if wd_exclusions:
        parts.append("nd")
    return "+".join(parts)


def clip_gradients(parameters_or_module, max_norm: float) -> None:
    """Clip the total gradient norm in place (thin wrapper over torch's clipper).

    Accepts a module (clips ``p for p in parameters() if p.grad is not None``) or
    an iterable of parameters. Gradient-less params (frozen trunks) are skipped so
    a partially-frozen model never trips the clipper. Wired into the training
    loops through their own module namespace so tests can monkeypatch the seam.
    """
    if isinstance(parameters_or_module, nn.Module):
        params = [p for p in parameters_or_module.parameters() if p.grad is not None]
    else:
        params = [p for p in parameters_or_module if p.grad is not None]
    if params:
        torch.nn.utils.clip_grad_norm_(params, max_norm)
