"""ConvNeXt V2 model factory with freezing utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import timm

from bittrainer.backbone_init import apply_backbone_init, uses_timm_pretrained

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


def _infer_head_hidden_size(state_dict: dict[str, torch.Tensor]) -> int | None:
    """Detect a non-linear MLP head (``head_hidden_size``) from saved weights.

    ``NormMlpClassifierHead`` only allocates ``head.pre_logits.fc`` when
    ``hidden_size`` is set; its row count is the hidden dimension. Absent the
    key, the head is a plain linear classifier.
    """
    pre = state_dict.get("head.pre_logits.fc.weight")
    if pre is not None:
        return pre.shape[0]
    return None


def create_model(
    *,
    model_size: str = "nano",
    pretrained: bool = True,
    dtype: torch.dtype = torch.float32,
    num_classes: int = 2,
    head_hidden_size: int | None = None,
    backbone_init: dict | None = None,
) -> nn.Module:
    """Create a ConvNeXt V2 model with *num_classes* output head.

    When *head_hidden_size* is set the classifier head gains a non-linear MLP
    (``Linear -> GELU -> Linear``) ahead of the final layer — used by the
    cached-feature MLP probe as the intermediate rung between a linear probe
    and a full fine-tune.
    """
    model_name = _MODEL_REGISTRY.get(model_size)
    if model_name is None:
        raise ValueError(f"Unknown model_size '{model_size}'. Valid: {list(_MODEL_REGISTRY.keys())}")
    use_pretrained = uses_timm_pretrained(
        requested_pretrained=pretrained,
        backbone_init=backbone_init,
    )
    model = timm.create_model(
        model_name,
        pretrained=use_pretrained,
        num_classes=num_classes,
        head_hidden_size=head_hidden_size,
    )
    apply_backbone_init(model, backbone_init)
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


# ---------------------------------------------------------------------------
# Cached-feature probe support
# ---------------------------------------------------------------------------
#
# The embedding cache stores the pooled feature vector at the point *before*
# the head's pre_logits/fc — i.e. ``flatten(norm(global_pool(features)))``.
# This is computed explicitly rather than via ``forward_head(pre_logits=True)``
# because the latter also runs the MLP (``pre_logits``) when a hidden head is
# present, which the probe must keep trainable.

# Everything from the head's pre_logits onward sits *after* the cached vector,
# so it must NOT contribute to the backbone hash; head.norm and earlier do.
_HEAD_TAIL_PREFIXES = ("head.pre_logits.", "head.fc.")


def pooled_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return the post-norm pooled feature vector ``[B, C]`` — the cache point.

    Deterministic in ``eval()`` (no drop_path, no head dropout). Identical for
    linear and MLP heads, since both share global_pool -> norm -> flatten.
    """
    feat = model.forward_features(x)
    head = model.head
    return head.flatten(head.norm(head.global_pool(feat)))


def head_tail_logits(model: nn.Module, pooled: torch.Tensor) -> torch.Tensor:
    """Map a cached pooled vector through the trainable head tail to logits.

    ``pre_logits`` is Identity for a linear head and the MLP for a hidden head;
    ``drop`` is a no-op in ``eval()`` and the configured dropout in ``train()``.
    """
    head = model.head
    return head.fc(head.drop(head.pre_logits(pooled)))


def head_tail_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Parameters trained by the cached probe: ``head.pre_logits`` + ``head.fc``."""
    params = list(model.head.fc.parameters())
    params += list(model.head.pre_logits.parameters())
    return params


def backbone_feature_hash(model: nn.Module) -> str:
    """Stable hash of every weight that affects the cached pooled vector.

    Covers stem, stages, ``norm_pre`` and ``head.norm`` (the cache point), but
    excludes ``head.pre_logits``/``head.fc`` (which sit after the cache and are
    retrained by the probe). A full fine-tune moves the included weights, so the
    hash changes and the embedding cache rebuilds against the adapted backbone.
    Returned as a 16-char hex digest.
    """
    import xxhash

    h = xxhash.xxh64()
    state = model.state_dict()
    for key in sorted(state):
        if any(key.startswith(p) for p in _HEAD_TAIL_PREFIXES):
            continue
        t = state[key].detach().cpu().contiguous()
        if t.is_floating_point():
            t = t.float()
        h.update(key.encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


_LLRD_DEPTH = {
    "stem": 5,
    "stages.0": 4,
    "stages.1": 3,
    "stages.2": 2,
    "stages.3": 1,
    "head": 0,
}


def build_llrd_param_groups(model: nn.Module, decay: float) -> list[dict]:
    """Group ConvNeXt V2 params by stage depth with decaying lr multipliers.

    Returns a list of param dicts compatible with ``torch.optim.Optimizer``.
    Each group's ``lr`` acts as a multiplier on the optimizer's base ``lr`` —
    for Prodigy_adv this multiplies the adapted step size ``d`` per group, so
    earlier (shallower) stages take smaller steps and the head moves fastest.
    """
    buckets: dict[str, list[nn.Parameter]] = {k: [] for k in _LLRD_DEPTH}
    for name, param in model.named_parameters():
        if name.startswith("stem."):
            bucket = "stem"
        elif name.startswith("stages.0."):
            bucket = "stages.0"
        elif name.startswith("stages.1."):
            bucket = "stages.1"
        elif name.startswith("stages.2."):
            bucket = "stages.2"
        elif name.startswith("stages.3."):
            bucket = "stages.3"
        elif name.startswith("head."):
            bucket = "head"
        else:
            bucket = "head"
        buckets[bucket].append(param)

    groups: list[dict] = []
    for key, params in buckets.items():
        if not params:
            continue
        groups.append({
            "params": params,
            "lr": decay ** _LLRD_DEPTH[key],
            "name": key,
        })
    return groups


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

    head_hidden_size = _infer_head_hidden_size(state_dict)

    model = create_model(
        model_size=ckpt_size, pretrained=False, dtype=dtype,
        num_classes=ckpt_classes, head_hidden_size=head_hidden_size,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model
