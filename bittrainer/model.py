"""ConvNeXt V2 model factory with freezing utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


_MODEL_REGISTRY = {
    "atto": "convnextv2_atto.fcmae_ft_in1k",
    "femto": "convnextv2_femto.fcmae_ft_in1k",
    "pico": "convnextv2_pico.fcmae_ft_in1k",
    "nano": "convnextv2_nano.fcmae_ft_in22k_in1k",
    "tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "base": "convnextv2_base.fcmae_ft_in22k_in1k",
    "large": "convnextv2_large.fcmae_ft_in22k_in1k",
    "huge": "convnextv2_huge.fcmae_ft_in22k_in1k",
}

_STEM_DIM_TO_SIZE: dict[int, str] = {
    40: "atto",
    48: "femto",
    64: "pico",
    80: "nano",
    96: "tiny",
    128: "base",
    192: "large",
    352: "huge",
}


def _infer_model_size(state_dict: dict[str, torch.Tensor]) -> str | None:
    stem_weight = state_dict.get("stem.0.weight")
    if stem_weight is not None:
        return _STEM_DIM_TO_SIZE.get(stem_weight.shape[0])
    return None


def _infer_num_classes(state_dict: dict[str, torch.Tensor]) -> int | None:
    head_weight = state_dict.get("head.fc.weight")
    if head_weight is not None:
        return head_weight.shape[0]
    return None
def create_model(
    *,
    model_size: str = "nano",
    pretrained: bool = True,
    dtype: torch.dtype = torch.float32,
    num_classes: int = 2,
) -> nn.Module:
    """Create a ConvNeXt V2 model with *num_classes* output head."""
    model_name = _MODEL_REGISTRY.get(model_size)
    if model_name is None:
        raise ValueError(f"Unknown model_size '{model_size}'. Valid: {list(_MODEL_REGISTRY.keys())}")
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    if dtype != torch.float32:
        model = model.to(dtype=dtype)
    return model


def get_stages(model: nn.Module) -> list[nn.Module]:
    """Return the 4 ConvNeXt V2 stages (feature extraction blocks)."""
    return list(model.stages)


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classification head."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


def unfreeze_stage(model: nn.Module, stage_index: int) -> None:
    """Unfreeze a single ConvNeXt stage by index (0-3, head-backward = 3 first)."""
    stages = get_stages(model)
    for param in stages[stage_index].parameters():
        param.requires_grad = True


def load_checkpoint(
    path: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    model_size: str = "nano",
    num_classes: int = 2,
) -> nn.Module:
    """Load a saved model checkpoint.

    If the checkpoint contains ``model_size`` or ``num_classes`` metadata
    (saved by the trainer), those values override the caller's arguments
    to prevent architecture mismatches.
    """
    data = torch.load(path, map_location=device, weights_only=True)
    if isinstance(data, dict) and "state_dict" in data:
        state_dict = data["state_dict"]
        ckpt_size = data.get("model_size", model_size)
        ckpt_classes = data.get("num_classes", num_classes)
    else:
        state_dict = data
        ckpt_size = model_size
        ckpt_classes = num_classes

    # Infer architecture from weights when metadata is missing or wrong
    inferred_size = _infer_model_size(state_dict)
    if inferred_size is not None and inferred_size != ckpt_size:
        ckpt_size = inferred_size
    inferred_classes = _infer_num_classes(state_dict)
    if inferred_classes is not None and inferred_classes != ckpt_classes:
        ckpt_classes = inferred_classes

    model = create_model(model_size=ckpt_size, pretrained=False, dtype=dtype, num_classes=ckpt_classes)
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model
