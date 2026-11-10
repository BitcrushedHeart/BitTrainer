"""Mixed-precision choices shared by the trainers (Bitcrush ISSUE-1655).

The trainers autocast in bfloat16 and never needed a GradScaler: bf16 keeps
fp32's exponent range, so gradients cannot underflow. That only holds on cards
that run bf16 natively (compute capability 8.0 and up: Ampere, Ada). A Turing
card such as the T4 (7.5) merely emulates it, without tensor-core kernels. On
those cards the fast path is fp16, and fp16 needs a GradScaler.

This module decides which precision a device gets and whether the loop must
step through a scaler. It never changes what a bf16-capable card or a CPU does.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# First CUDA compute-capability major with native bf16 (the same line torch's
# own ``is_bf16_supported(including_emulation=False)`` draws).
_NATIVE_BF16_MAJOR = 8


def native_bf16(device: torch.device) -> bool:
    """True when *device* runs bfloat16 autocast natively.

    A CPU counts as native: CPU autocast has always used bf16 in these trainers
    and must not flip to fp16. Any non-CUDA accelerator is left alone too.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return True
    return torch.cuda.get_device_properties(device).major >= _NATIVE_BF16_MAJOR


def resolve_amp_dtype(
    requested: torch.dtype | None, device: torch.device
) -> torch.dtype:
    """The autocast dtype a run should use on *device*.

    ``requested=None`` means "pick for the card": bf16 where it is native, fp16
    where it is not. An explicit bf16 on a card without it also becomes fp16
    (emulated bf16 would crawl); explicit fp16 / fp32 are honoured as given.
    """
    if requested is None or requested == torch.bfloat16:
        if native_bf16(device):
            return torch.bfloat16
        if requested is not None:
            logger.warning(
                "bfloat16 autocast was requested but %s has no native bf16; using float16",
                device,
            )
        return torch.float16
    return requested


def wants_grad_scaler(device: torch.device, dtype: torch.dtype, enabled: bool) -> bool:
    """fp16 autocast on a GPU is the one case that needs a GradScaler."""
    return (
        bool(enabled) and dtype == torch.float16 and torch.device(device).type == "cuda"
    )


def make_grad_scaler(device: torch.device, dtype: torch.dtype, enabled: bool):
    """A ``torch.amp.GradScaler`` when the run needs one, else ``None``."""
    if not wants_grad_scaler(device, dtype, enabled):
        return None
    return torch.amp.GradScaler("cuda")


def amp_label(enabled: bool, dtype: torch.dtype) -> str:
    """Short name of the precision a run trains in, for progress frames."""
    if not enabled:
        return "off"
    return str(dtype).removeprefix("torch.")
