"""Central backbone initialization helpers.

BitTrainer owns ConvNeXt-compatible model construction, but callers such as
Bitcrush Engine own backbone resolution. The resolver passes a small init spec
that tells BitTrainer whether to load local safetensors weights, use the
temporary timm fallback, or start from random initialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

_LOCAL_SOURCES = {"local_active", "local_candidate"}
_TIMM_FALLBACK_SOURCE = "temporary_timm_pretrained_fallback"
_RANDOM_SOURCE = "random_init"
_SUPPORTED_SOURCES = _LOCAL_SOURCES | {_TIMM_FALLBACK_SOURCE, _RANDOM_SOURCE}


@dataclass(frozen=True)
class BackboneInitSpec:
    source: str
    checkpoint_path: str | None = None
    architecture: str | None = None
    family_name: str | None = None
    size_alias: str | None = None
    display_size: str | None = None
    convnextv2_size: str | None = None
    release_blocking: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any] | "BackboneInitSpec") -> "BackboneInitSpec":
        if isinstance(raw, BackboneInitSpec):
            return raw
        source = str(raw.get("source") or "")
        if source not in _SUPPORTED_SOURCES:
            known = ", ".join(sorted(_SUPPORTED_SOURCES))
            raise ValueError(f"Unknown backbone init source {source!r}. Expected one of: {known}")
        return cls(
            source=source,
            checkpoint_path=raw.get("checkpoint_path"),
            architecture=raw.get("architecture"),
            family_name=raw.get("family_name"),
            size_alias=raw.get("size_alias"),
            display_size=raw.get("display_size"),
            convnextv2_size=raw.get("convnextv2_size"),
            release_blocking=bool(raw.get("release_blocking", False)),
            metadata=dict(raw.get("metadata") or {}),
        )


def normalize_backbone_init(
    backbone_init: Mapping[str, Any] | BackboneInitSpec | None,
) -> BackboneInitSpec | None:
    if backbone_init is None:
        return None
    return BackboneInitSpec.from_raw(backbone_init)


def uses_timm_pretrained(
    *,
    requested_pretrained: bool,
    backbone_init: Mapping[str, Any] | BackboneInitSpec | None,
) -> bool:
    spec = normalize_backbone_init(backbone_init)
    if spec is None:
        return bool(requested_pretrained)
    if spec.source == _TIMM_FALLBACK_SOURCE:
        return True
    if spec.source in _LOCAL_SOURCES or spec.source == _RANDOM_SOURCE:
        return False
    raise AssertionError(f"Unhandled backbone init source: {spec.source}")


def apply_backbone_init(model: nn.Module, backbone_init: Mapping[str, Any] | BackboneInitSpec | None) -> dict:
    spec = normalize_backbone_init(backbone_init)
    if spec is None or spec.source not in _LOCAL_SOURCES:
        return {"loaded_tensors": 0, "total_tensors": 0}
    if not spec.checkpoint_path:
        raise ValueError(f"Backbone init source {spec.source!r} requires checkpoint_path.")

    checkpoint_path = Path(spec.checkpoint_path)
    if checkpoint_path.suffix != ".safetensors":
        raise ValueError("Local backbone initialization requires a .safetensors checkpoint.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")

    from safetensors.torch import load_file

    state_dict = load_file(str(checkpoint_path))
    target = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        target_tensor = target.get(key)
        if target_tensor is None or target_tensor.shape != tensor.shape:
            continue
        matched[key] = tensor.to(dtype=target_tensor.dtype)

    model.load_state_dict(matched, strict=False)
    return {"loaded_tensors": len(matched), "total_tensors": len(state_dict)}

