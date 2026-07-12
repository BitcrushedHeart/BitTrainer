"""Backbone initialisation from a Bitcrush Engine backbone spec.

Engine's Backbone Builder resolves where a trainer's backbone weights should
come from and passes the decision down as a plain dict (``backbone_init``):

    {
        "source": "local_active" | "local_candidate"
                  | "temporary_timm_pretrained_fallback" | "random_init",
        "checkpoint_path": str | None,   # safetensors backbone state dict
        "size_alias": str, "convnextv2_size": str, ...  # informational
    }

Routing:
  * local_* + checkpoint_path  -> create the model with ``pretrained=False``
    and load the checkpoint into it (``apply_backbone_init``).
  * random_init                -> ``pretrained=False``, no load.
  * fallback / missing / None  -> legacy behaviour (timm pretrained weights).

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


def wants_timm_pretrained(spec: dict | None, default: bool = True) -> bool:
    """Should model creation download/use timm pretrained weights?"""
    if not spec:
        return default
    if _local_checkpoint(spec):
        return False
    if spec.get("source") == "random_init":
        return False
    return True


def apply_backbone_init(module: nn.Module, spec: dict | None) -> bool:
    """Load a local backbone checkpoint from *spec* into *module*.

    *module* is a timm ConvNeXt V2 model (with or without a classifier head)
    or a bare backbone. Returns True when weights were loaded, False when the
    spec does not point at a local checkpoint. Missing files raise — training
    silently continuing from random weights when the user selected their local
    backbone would be worse than failing.
    """
    checkpoint_path = _local_checkpoint(spec)
    if checkpoint_path is None:
        return False
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"backbone_init checkpoint not found: {path}")

    from safetensors.torch import load_file

    state = load_file(str(path))
    # Accept both bare backbone keys and "backbone."-prefixed dumps.
    if state and all(key.startswith("backbone.") for key in state):
        state = {key[len("backbone.") :]: value for key, value in state.items()}

    target = module.state_dict()
    matched = {
        key: value.to(target[key].dtype)
        for key, value in state.items()
        if key in target and target[key].shape == value.shape
    }
    if not matched:
        raise RuntimeError(
            f"backbone_init checkpoint {path} shares no tensors with the target "
            f"model ({len(state)} in checkpoint, {len(target)} in model)"
        )
    module.load_state_dict(matched, strict=False)
    logger.info(
        "backbone_init: loaded %d/%d tensors from %s (source=%s)",
        len(matched),
        len(target),
        path,
        spec.get("source") if spec else None,
    )
    return True
