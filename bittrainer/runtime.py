"""CUDA backend configuration and torch.compile helpers.

Shared by every trainer entry point. Compilation is *regional*: rather than
wrapping the whole ConvNeXt in a single ``torch.compile`` (one giant graph whose
Inductor compile time grows super-linearly with the inlined block count), each
repeated ConvNeXt block is compiled in place via ``nn.Module.compile``.
Structurally identical blocks share one compiled artifact, so a stage with N
identical blocks pays roughly one block's compile cost instead of N's — the same
trick OneTrainer uses for transformer blocks.

In-place compilation also leaves the eager module as the single source of truth:
``_compiled_call_impl`` is excluded from ``state_dict``/pickling, so optimizer,
EMA, freeze/unfreeze, the backbone hash and checkpoint saves all operate on the
unchanged module and checkpoint keys never grow an ``_orig_mod.`` prefix.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Each compiled block instance caches one graph per active bucket shape (the
# batch dim goes dynamic after the second warm pass). 9 buckets across
# train+eval stays well under 64 per block.
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


def configure_compile_cache() -> None:
    """Point Inductor + Triton at a stable, per-user on-disk cache.

    The Engine retrains many small group classifiers in separate processes, and
    Inductor's default cache lives in a volatile temp dir — so every run cold-
    compiles from scratch. A persistent cache lets run N reuse run 1's kernels,
    turning the multi-minute compile into a near-instant cache hit. Honours a
    pre-set ``TORCHINDUCTOR_CACHE_DIR`` / ``TRITON_CACHE_DIR`` and an optional
    ``BITTRAINER_COMPILE_CACHE`` root override. Idempotent; safe to call before
    every compile.
    """
    root = os.environ.get("BITTRAINER_COMPILE_CACHE")
    base = Path(root) if root else Path.home() / ".cache" / "bittrainer" / "torch_compile"
    inductor = base / "inductor"
    triton = base / "triton"
    for d in (inductor, triton):
        d.mkdir(parents=True, exist_ok=True)
    # setdefault so an explicit user-supplied cache dir always wins.
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor))
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton))


def _compile_targets(model: nn.Module) -> list[nn.Module]:
    """The repeated ConvNeXt blocks to compile individually.

    Walks every submodule exposing a timm ``.stages`` container — covering the
    plain classifier (``model.stages``), the dual-branch model (``crop_branch``
    / ``context_branch``) and the multi-head model (``backbone``) — and collects
    each stage's ``.blocks``. The cheap, shape-specific stem/downsample/head are
    deliberately left eager: that is exactly where the compile cost is *not*.
    """
    targets: list[nn.Module] = []
    seen: set[int] = set()
    for module in model.modules():
        stages = getattr(module, "stages", None)
        if stages is None:
            continue
        for stage in stages:
            blocks = getattr(stage, "blocks", None)
            if blocks is None:
                continue
            for block in blocks:
                if id(block) not in seen:
                    seen.add(id(block))
                    targets.append(block)
    return targets


def compile_regions(model: nn.Module, *, cb: Callable[[dict], None] | None = None) -> int:
    """Compile each repeated ConvNeXt block in place; return the block count.

    Sets up the persistent cache and the dynamo/inductor knobs, then calls
    ``nn.Module.compile`` on every block. Compilation is lazy — the first
    forward at each shape is what actually builds the kernel (see
    ``prewarm_compile``).
    """
    configure_compile_cache()
    import torch._dynamo as dynamo
    import torch._inductor.config as inductor_config

    dynamo.config.cache_size_limit = _DYNAMO_CACHE_SIZE
    # A compile failure mid-run must degrade to eager, never kill training.
    dynamo.config.suppress_errors = True
    # torch 2.10 inductor hits "CantSplit" generating the mix-order-reduction
    # backward kernel for dynamic-batch ConvNeXt graphs; disable the
    # optimisation rather than the whole backward compile.
    inductor_config.triton.mix_order_reduction = False

    targets = _compile_targets(model)
    for block in targets:
        block.compile()
    if cb is not None:
        cb({
            "type": "training_progress", "stage": "preparing",
            "status_text": f"Compiling {len(targets)} blocks regionally",
        })
    return len(targets)


def uncompile_regions(model: nn.Module) -> None:
    """Undo ``compile_regions`` — restore each block's eager forward.

    Used by the pre-warm fallback: if graph capture fails, blocks revert to
    eager instead of retrying a doomed compile on every step. A no-op for blocks
    that were never compiled.
    """
    for block in _compile_targets(model):
        block._compiled_call_impl = None


def maybe_compile(
    model: nn.Module,
    *,
    enabled: bool,
    cb: Callable[[dict], None] | None = None,
) -> tuple[nn.Module, bool]:
    """Return ``(model, compiled)`` — the same module, blocks compiled in place.

    Falls back to eager (with a visible status message) when triton is missing;
    compilation must never be a hard dependency. The returned module is always
    ``model`` itself — regional compilation mutates submodules in place, so
    there is no ``OptimizedModule`` wrapper and no ``_orig_mod`` prefix.
    """
    if not enabled:
        return model, False
    if not compile_supported():
        msg = "torch.compile unavailable (triton not installed) — running eager"
        logger.warning(msg)
        if cb is not None:
            cb({"type": "training_progress", "stage": "preparing", "status_text": msg})
        return model, False
    compile_regions(model, cb=cb)
    return model, True


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
        # Revert the in-place block compilation so the run uses eager forwards
        # instead of retrying a doomed compile every step.
        uncompile_regions(forward_model)
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
