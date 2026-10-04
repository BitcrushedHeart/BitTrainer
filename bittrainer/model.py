"""Group/binary model factory with freezing utilities.

ConvNeXt V2 is the historical family; SigLIP2 sizes dispatch to
``bittrainer.siglip_model`` (Bitcrush ISSUE-1106) behind the same functions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm

from bittrainer.backbones import is_siglip, siglip_size_from_dim
from bittrainer.convnextv2_catalog import CONVNEXTV2_PRETRAINED_SOURCES

_MODEL_REGISTRY = {
    size: source.model_name for size, source in CONVNEXTV2_PRETRAINED_SOURCES.items()
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
    # SigLIP2 wrapper (Bitcrush ISSUE-1106): the patch projection's row count is
    # the embedding width, which names the size.
    proj = state_dict.get(_SIGLIP_PROJ_KEY)
    if proj is not None:
        return siglip_size_from_dim(proj.shape[0])
    return None


_SIGLIP_PROJ_KEY = "trunk.embeds.proj.weight"


def _infer_num_classes(state_dict: dict[str, torch.Tensor]) -> int | None:
    head_weight = state_dict.get("head.fc.weight")
    if head_weight is not None:
        return head_weight.shape[0]
    # All-threshold ordinal head (Bitcrush ISSUE-1106): K classes, K-1 logits.
    threshold_weight = state_dict.get("head.fc.threshold_fc.weight")
    if threshold_weight is not None:
        return threshold_weight.shape[0] + 1
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


def _infer_region_head(state_dict: dict[str, torch.Tensor]) -> tuple[int, int] | None:
    """``(num_types, channels)`` of a saved region head, or ``None``.

    Read off the heatmap projection (``[num_types, channels, 1, 1]``), the same
    way the classifier's shape is read off its own weights, so a checkpoint
    whose ``region_head`` metadata is missing still loads (Bitcrush ISSUE-0993).
    """
    weight = state_dict.get("region_head.heatmap.weight")
    if weight is None:
        return None
    return int(weight.shape[0]), int(weight.shape[1])


# Stochastic-depth (drop-path) maxima scaled by trunk size, following the
# ConvNeXt recipe: the shallow atto..nano trunks tolerate only light regularisation
# (0.1), tiny sits mid (0.2) and the deep base/large/huge trunks want the full 0.3.
_DROP_PATH_RATE: dict[str, float] = {
    "atto": 0.1,
    "femto": 0.1,
    "pico": 0.1,
    "nano": 0.1,
    "tiny": 0.2,
    "base": 0.3,
    "large": 0.3,
    "huge": 0.3,
}


def default_drop_path_rate(model_size: str) -> float:
    """Size-scaled stochastic-depth maximum for a ConvNeXt V2 trunk.

    Long-horizon trunk training (the backbone builder) benefits from stochastic
    depth; the value scales with depth so shallow trunks are not over-regularised.
    Unknown sizes fall back to the conservative 0.1.
    """
    return _DROP_PATH_RATE.get(model_size, 0.1)


def create_model(
    *,
    model_size: str = "nano",
    pretrained: bool = True,
    dtype: torch.dtype = torch.float32,
    num_classes: int = 2,
    head_hidden_size: int | None = None,
    drop_path_rate: float | None = None,
) -> nn.Module:
    """Create a ConvNeXt V2 model with *num_classes* output head.

    When *head_hidden_size* is set the classifier head gains a non-linear MLP
    (``Linear -> GELU -> Linear``) ahead of the final layer — used by the
    cached-feature MLP probe as the intermediate rung between a linear probe
    and a full fine-tune.

    ``drop_path_rate`` is the stochastic-depth maximum. ``None`` (the default)
    forwards nothing to timm, so the created model is byte-identical to the
    historical call; a float is passed straight through to timm's ConvNeXt
    ``drop_path_rate`` (parameter-free, so it never changes the state-dict shape).

    SigLIP2 sizes (``"siglip2_so400m"`` / ``"siglip2_base"``, Bitcrush
    ISSUE-1106) build a :class:`~bittrainer.siglip_model.SigLIPClassifier`, a
    wrapper with the same head surface, so every caller stays family-agnostic.
    """
    if is_siglip(model_size):
        from bittrainer.siglip_model import create_siglip_model

        return create_siglip_model(
            model_size=model_size,
            pretrained=pretrained,
            dtype=dtype,
            num_classes=num_classes,
            head_hidden_size=head_hidden_size,
            drop_path_rate=drop_path_rate,
        )
    model_name = _MODEL_REGISTRY.get(model_size)
    if model_name is None:
        raise ValueError(
            f"Unknown model_size '{model_size}'. Valid: {list(_MODEL_REGISTRY.keys())}"
        )
    extra: dict = {}
    if drop_path_rate is not None:
        extra["drop_path_rate"] = drop_path_rate
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        head_hidden_size=head_hidden_size,
        **extra,
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


def prepare_for_finetune(model: nn.Module, finetune_blocks: int = 4) -> None:
    """Set the trainable parameters for the in-loop fine-tune.

    ConvNeXt unfreezes everything (exactly :func:`unfreeze_backbone`). A SigLIP
    wrapper trains its head(s) plus the last ``finetune_blocks`` transformer
    blocks; 0 keeps its trunk frozen (Bitcrush ISSUE-1106).
    """
    from bittrainer.siglip_model import is_siglip_model

    if is_siglip_model(model):
        from bittrainer.siglip_model import prepare_for_finetune as _prepare_siglip

        _prepare_siglip(model, finetune_blocks)
        return
    unfreeze_backbone(model)


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


# The region head (Bitcrush ISSUE-0993) branches off the trunk's stage maps and
# never feeds the pooled vector, so it is not backbone identity either: training
# or attaching one must not invalidate the embedding cache.
_NON_BACKBONE_PREFIXES = (*_HEAD_TAIL_PREFIXES, "region_head.")


def backbone_state_hash(state: dict[str, torch.Tensor]) -> str:
    """:func:`backbone_feature_hash` of a bare ``state_dict`` — lets a saved
    checkpoint be compared with a live model without rebuilding it."""
    import xxhash

    h = xxhash.xxh64()
    for key in sorted(state):
        if any(key.startswith(p) for p in _NON_BACKBONE_PREFIXES):
            continue
        t = state[key].detach().cpu().contiguous()
        if t.is_floating_point():
            t = t.float()
        h.update(key.encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


def backbone_feature_hash(model: nn.Module) -> str:
    """Stable hash of every weight that affects the cached pooled vector.

    Covers stem, stages, ``norm_pre`` and ``head.norm`` (the cache point), but
    excludes ``head.pre_logits``/``head.fc`` (which sit after the cache and are
    retrained by the probe) and the region head (which sits beside it). A full
    fine-tune moves the included weights, so the hash changes and the embedding
    cache rebuilds against the adapted backbone. Returned as a 16-char hex digest.
    """
    return backbone_state_hash(model.state_dict())


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

    A SigLIP wrapper buckets by transformer block instead and keeps only its
    trainable parameters (see ``build_siglip_llrd_param_groups``).
    """
    from bittrainer.siglip_model import is_siglip_model

    if is_siglip_model(model):
        from bittrainer.siglip_model import build_siglip_llrd_param_groups

        return build_siglip_llrd_param_groups(model, decay)
    buckets: dict[str, list[nn.Parameter]] = {k: [] for k in _LLRD_DEPTH}
    for name, param in model.named_parameters():
        # The multi-task backbone builder wraps the trunk as ``backbone.*`` and the
        # heads as ``heads.*``; strip one optional leading ``backbone.`` so the trunk
        # buckets by its true stem/stage depth, and route ``heads.*`` (like the
        # trunk's own ``head.*``) to the fastest depth-0 head bucket. A plain model
        # carries neither prefix, so it buckets exactly as before.
        depth_name = name[len("backbone.") :] if name.startswith("backbone.") else name
        if depth_name.startswith("stem."):
            bucket = "stem"
        elif depth_name.startswith("stages.0."):
            bucket = "stages.0"
        elif depth_name.startswith("stages.1."):
            bucket = "stages.1"
        elif depth_name.startswith("stages.2."):
            bucket = "stages.2"
        elif depth_name.startswith("stages.3."):
            bucket = "stages.3"
        else:
            # head.*, heads.*, norm_pre and anything else train at depth 0.
            bucket = "head"
        buckets[bucket].append(param)

    groups: list[dict] = []
    for key, params in buckets.items():
        if not params:
            continue
        groups.append(
            {
                "params": params,
                "lr": decay ** _LLRD_DEPTH[key],
                "name": key,
            }
        )
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
        cell_masks = data.get("cell_masks")
        grid_rows = int(data.get("grid_rows") or 3)
        grid_cols = int(data.get("grid_cols") or 3)
        class_names = data.get("class_names")
    else:
        state_dict = data
        ckpt_size = model_size
        ckpt_classes = num_classes
        cell_masks = None
        grid_rows = grid_cols = 3
        class_names = None

    # Infer architecture from weights when metadata is missing or wrong
    inferred_size = _infer_model_size(state_dict)
    if inferred_size is not None and inferred_size != ckpt_size:
        ckpt_size = inferred_size
    if _SIGLIP_PROJ_KEY in state_dict and not is_siglip(ckpt_size):
        width = int(state_dict[_SIGLIP_PROJ_KEY].shape[0])
        raise ValueError(
            f"SigLIP checkpoint {path} has an unrecognised trunk width ({width}) "
            f"and no SigLIP model_size (got {ckpt_size!r})"
        )
    inferred_classes = _infer_num_classes(state_dict)
    if inferred_classes is not None and inferred_classes != ckpt_classes:
        ckpt_classes = inferred_classes

    head_hidden_size = _infer_head_hidden_size(state_dict)

    model = create_model(
        model_size=ckpt_size,
        pretrained=False,
        dtype=dtype,
        num_classes=ckpt_classes,
        head_hidden_size=head_hidden_size,
    )
    if cell_masks is not None:
        # Spatial checkpoint: swap in the cell-structured head before loading —
        # the state_dict carries head.fc.cell_fc.* instead of head.fc.*. The
        # module keeps emitting num_classes scores, so callers are unaffected.
        from bittrainer.spatial import install_spatial_head

        install_spatial_head(
            model,
            [list(m) for m in cell_masks],
            grid_rows * grid_cols,
        )
    if "head.fc.threshold_fc.weight" in state_dict:
        # All-threshold ordinal checkpoint (Bitcrush ISSUE-1106): swap in the
        # threshold head; it still emits num_classes log-probabilities.
        from bittrainer.ordinal_threshold import (
            install_ordinal_threshold_head,
            threshold_none_index,
        )

        install_ordinal_threshold_head(
            model, ckpt_classes, threshold_none_index(state_dict, class_names)
        )
    region_shape = _infer_region_head(state_dict)
    if region_shape is not None:
        # Crop-assisted checkpoint: attach the region head before loading so
        # its region_head.* weights have somewhere to land. model.forward is
        # unchanged; only forward_with_regions runs the head.
        from bittrainer.regions import install_region_head

        install_region_head(model, region_shape[0], region_shape[1])
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model
