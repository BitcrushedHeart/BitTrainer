"""CUDA backend configuration and torch.compile helpers.

Shared by every trainer entry point. The compile helpers keep the eager model
as the source of truth — optimizer, EMA, checkpoint saves, freeze/unfreeze and
the backbone hash all operate on the eager module; the OptimizedModule returned
by ``maybe_compile`` shares its parameters and is used for forward passes only,
so checkpoint keys never grow an ``_orig_mod.`` prefix.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# 9 train-mode graphs (one per bucket shape, batch dim dynamic after the second
# warm pass) + the same again in eval mode leaves ample slack below 64.
_DYNAMO_CACHE_SIZE = 64


def configure_cuda_backend() -> None:
    """Enable TF32 and cuDNN autotuning. Idempotent; call at trainer entry."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # The aspect-ratio bucket table fixes inputs to a handful of static shapes,
    # so cuDNN's per-shape autotuning amortises within the first epoch.
    torch.backends.cudnn.benchmark = True


def compile_supported() -> bool:
    """CUDA inductor needs triton (the triton-windows wheel on Windows)."""
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, "compile")


def maybe_compile(
    model: nn.Module,
    *,
    enabled: bool,
    cb: Callable[[dict], None] | None = None,
) -> tuple[nn.Module, bool]:
    """Return ``(forward_model, compiled)`` — a compiled wrapper or the model.

    Falls back to eager (with a visible status message) when triton is missing;
    compilation must never be a hard dependency.
    """
    if not enabled:
        return model, False
    if not compile_supported():
        msg = "torch.compile unavailable (triton not installed) — running eager"
        logger.warning(msg)
        if cb is not None:
            cb({"type": "training_progress", "stage": "preparing", "status_text": msg})
        return model, False
    import torch._dynamo as dynamo
    import torch._inductor.config as inductor_config

    dynamo.config.cache_size_limit = _DYNAMO_CACHE_SIZE
    # A compile failure mid-run must degrade to eager, never kill training.
    dynamo.config.suppress_errors = True
    # torch 2.10 inductor hits "CantSplit" generating the mix-order-reduction
    # backward kernel for dynamic-batch ConvNeXt graphs; disable the
    # optimisation rather than the whole backward compile.
    inductor_config.triton.mix_order_reduction = False
    return torch.compile(model), True


def unwrap_compiled(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def prewarm_compile(
    forward_model: nn.Module,
    bucket_counts: dict[tuple[int, int], int],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    memory_format: torch.memory_format | None = None,
    make_inputs: Callable[[int, tuple[int, int]], tuple[torch.Tensor, ...]] | None = None,
    cb: Callable[[dict], None] | None = None,
) -> bool:
    """Capture train-mode graphs for every active bucket shape up front.

    Each shape runs forward+backward at ``batch_size`` then ``batch_size - 1``
    so dynamo specialises H/W statically and marks the batch dim dynamic —
    partial tail batches then reuse the same graph instead of recompiling.
    Moves the multi-minute first-compile out of epoch-0 step timing and behind
    a visible "compiling" stage.

    Returns False when graph capture failed — the caller should fall back to
    the eager model for the run.
    """
    shapes = [b for b, n in bucket_counts.items() if n > 0]
    if not shapes:
        return True

    def _default_inputs(b: int, bucket: tuple[int, int]) -> tuple[torch.Tensor, ...]:
        w, h = bucket
        x = torch.randn(b, 3, h, w, device=device, dtype=dtype)
        if memory_format is not None:
            x = x.contiguous(memory_format=memory_format)
        return (x,)

    build = make_inputs or _default_inputs
    batches = [batch_size] if batch_size <= 1 else [batch_size, batch_size - 1]
    total = len(shapes) * len(batches)
    step = 0
    was_training = forward_model.training
    forward_model.train()
    ok = True
    try:
        for bucket in shapes:
            for b in batches:
                step += 1
                if cb is not None:
                    cb({
                        "type": "training_progress", "stage": "compiling",
                        "status_text": f"Compiling model (shape {step}/{total})",
                        "step": step, "total_steps": total,
                    })
                inputs = build(b, bucket)
                with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                    out = forward_model(*inputs)
                    loss = out.float().sum()
                loss.backward()
                del inputs, out, loss
    except Exception:
        # Inductor/dynamo failures must degrade to eager, never kill training.
        logger.warning("torch.compile pre-warm failed — falling back to eager", exc_info=True)
        if cb is not None:
            cb({
                "type": "training_progress", "stage": "preparing",
                "status_text": "Compilation failed — running eager",
            })
        ok = False
    forward_model.zero_grad(set_to_none=True)
    forward_model.train(was_training)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ok
