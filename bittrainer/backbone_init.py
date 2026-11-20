"""Backbone initialisation from a Bitcrush Engine backbone spec.

Engine's Backbone Builder resolves where a trainer's backbone weights should
come from and passes the decision down as a plain dict (``backbone_init``):

    {
        "source": "local_active" | "local_candidate" | "random_init",
        "checkpoint_path": str | None,   # safetensors backbone state dict
        "size_alias": str, "convnextv2_size": str, ...  # informational
    }

Every model is created with ``pretrained=False``; then (``apply_backbone_init``):
  * local_* + checkpoint_path -> the checkpoint is loaded into it.
  * random_init               -> nothing is loaded, on purpose.
  * anything else (no spec, no path, a retired source) -> refused: nothing
    trains from random weights unless it says so (Bitcrush ISSUE-1733).

Warm-starting from an existing ``best.pt`` always takes precedence in the
trainers — ``backbone_init`` only governs the fresh-model path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch.nn as nn

logger = logging.getLogger(__name__)

_LOCAL_SOURCES = frozenset({"local_active", "local_candidate"})


def _local_checkpoint(spec: dict | None) -> str | None:
    if not spec:
        return None
    if spec.get("source") not in _LOCAL_SOURCES:
        return None
    path = spec.get("checkpoint_path")
    return str(path) if path else None


def matching_tensors(state: dict, target: dict, origin: str) -> dict:
    """The tensors of *state* that fit *target* by name and shape, in the
    target's dtype. None at all raises: loading nothing would leave the model
    on its random weights while the run reports a start from *origin*."""
    matched = {
        key: value.to(target[key].dtype)
        for key, value in state.items()
        if key in target and target[key].shape == value.shape
    }
    if not matched:
        raise RuntimeError(
            f"{origin} shares no tensors with the target model "
            f"({len(state)} in checkpoint, {len(target)} in model)"
        )
    return matched


def apply_backbone_init(module: nn.Module, spec: dict | None) -> bool:
    """Load a local backbone checkpoint from *spec* into *module*.

    *module* is a timm ConvNeXt V2 model (with or without a classifier head)
    or a bare backbone. Returns True when weights were loaded, False for an
    explicit ``random_init``. A spec that names no local checkpoint and a
    missing file both raise — training silently continuing from random weights
    would be worse than failing.
    """
    if (spec or {}).get("source") == "random_init":
        return False
    checkpoint_path = _local_checkpoint(spec)
    if checkpoint_path is None:
        raise ValueError(
            f"backbone_init {spec!r} names no local backbone file: a fresh model "
            "starts from a local BitVision trunk (source 'local_active' or "
            "'local_candidate' with a checkpoint_path); pass source 'random_init' "
            "to train from random weights on purpose"
        )
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"backbone_init checkpoint not found: {path}")

    from safetensors.torch import load_file

    state = load_file(str(path))
    # Accept both bare backbone keys and "backbone."-prefixed dumps.
    if state and all(key.startswith("backbone.") for key in state):
        state = {key[len("backbone.") :]: value for key, value in state.items()}

    target = module.state_dict()
    matched = matching_tensors(state, target, f"backbone_init checkpoint {path}")
    module.load_state_dict(matched, strict=False)
    logger.info(
        "backbone_init: loaded %d/%d tensors from %s (source=%s)",
        len(matched),
        len(target),
        path,
        spec.get("source") if spec else None,
    )
    return True
