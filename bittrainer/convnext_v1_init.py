"""Start a ConvNeXt V2 trunk from ConvNeXt V1 weights (Bitcrush ISSUE-1655).

Why: every published ConvNeXt V2 weight (the self-supervised FCMAE ones
included) is CC-BY-NC-4.0. ConvNeXt V1 has weights timm declares Apache-2.0, and
a V1 trunk can become a V2 trunk without changing what it computes.

The two blocks differ in exactly two places:

* **LayerScale.** V1 multiplies each block's output by a learned per-channel
  ``gamma``. That is a scale on the output of the block's last layer (``fc2``),
  so it folds into that layer: ``gamma * (W x + b) == (gamma W) x + gamma b``.
* **GRN.** V2 adds Global Response Normalisation inside the MLP. With its weight
  and bias at zero (where V2 itself starts them) it returns its input unchanged.

Everything else (stem, downsampling, depthwise convolutions, norms) has the same
names and shapes. The converted trunk therefore gives the V1 model's features to
within float rounding, and :func:`build_v1_trunk` checks exactly that on the real
weights before it hands a trunk over. From there the V2 architecture trains as
usual, GRN included.

The licence is taken from timm's own catalogue for the chosen weights and
written into the trunk's metadata; anything not on :data:`PERMISSIVE_LICENCES`
is refused before it is downloaded.
"""

from __future__ import annotations

from pathlib import Path

import torch

PERMISSIVE_LICENCES = frozenset({"apache-2.0", "mit"})

# One Apache-2.0 V1 weight per size BitTrainer trains, preferring the widest
# pretraining available at 224px. ``--v1-weights`` picks another.
V1_WEIGHTS = {
    "atto": "convnext_atto.d2_in1k",
    "femto": "convnext_femto.d1_in1k",
    "pico": "convnext_pico.d1_in1k",
    "nano": "convnext_nano.in12k_ft_in1k",
    "tiny": "convnext_tiny.in12k_ft_in1k",
    "base": "convnext_base.fb_in22k_ft_in1k",
    "large": "convnext_large.fb_in22k_ft_in1k",
}

# Largest feature difference allowed between the V1 model and its conversion,
# relative to the largest feature. Float rounding lands orders below this.
_TOLERANCE = 1e-3


def declared_licence(name: str) -> str:
    """The licence timm's catalogue declares for pretrained weights *name*."""
    from timm.models import get_pretrained_cfg

    config = get_pretrained_cfg(name)
    return str(getattr(config, "license", None) or "unknown").lower()


def convert_v1_state(
    v1_state: dict[str, torch.Tensor], v2_state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """A V2 state dict that computes what *v1_state* computes.

    *v2_state* is the target model's own state dict; it supplies the names and
    shapes the result must have. Raises ``ValueError`` when the V1 weights do
    not fit it (another size, an overlapping-stem variant and so on) rather than
    load part of a trunk.
    """
    converted = {
        key: value.clone()
        for key, value in v1_state.items()
        if not key.endswith(".gamma") and not key.startswith("head.fc.")
    }
    # Fold each block's LayerScale into the layer it scales.
    for key, gamma in v1_state.items():
        if not key.endswith(".gamma"):
            continue
        block = key[: -len("gamma")]
        weight, bias = f"{block}mlp.fc2.weight", f"{block}mlp.fc2.bias"
        if weight not in converted or bias not in converted:
            raise ValueError(f"the V1 weights do not fit: {block} has no final layer")
        # fc2 is Linear (out, hidden) or a 1x1 convolution (out, hidden, 1, 1).
        per_output = gamma.reshape(-1, *([1] * (converted[weight].ndim - 1)))
        converted[weight] = converted[weight] * per_output
        converted[bias] = converted[bias] * gamma
    # GRN at zero passes its input through.
    for key, value in v2_state.items():
        if ".mlp.grn." in key:
            converted[key] = torch.zeros_like(value)

    wanted = {k: v.shape for k, v in v2_state.items() if not k.startswith("head.fc.")}
    missing = wanted.keys() - converted.keys()
    unexpected = converted.keys() - wanted.keys()
    misshapen = [
        k for k in wanted.keys() & converted.keys() if wanted[k] != converted[k].shape
    ]
    if missing or unexpected or misshapen:
        raise ValueError(
            f"the V1 weights do not fit this V2 trunk: {len(missing)} tensors missing, "
            f"{len(unexpected)} unexpected, {len(misshapen)} of another shape"
        )
    return converted


def _load_v1(name: str) -> torch.nn.Module:
    """The published V1 model, without its classifier (downloads the weights)."""
    import timm

    return timm.create_model(name, pretrained=True, num_classes=0).eval()


def build_v1_trunk(size: str, weights: str | None = None) -> tuple[dict, dict]:
    """Convert published V1 weights for a V2 trunk of *size*.

    Returns ``(state, info)``: the V2 backbone state dict and where it came
    from (model id, licence, source URL, measured feature difference). Refuses
    weights whose declared licence is not permissive, and refuses to return a
    trunk that does not reproduce the V1 model's features.
    """
    from bittrainer.model import create_model

    name = weights or V1_WEIGHTS.get(size)
    if name is None:
        known = ", ".join(V1_WEIGHTS)
        raise ValueError(f"no ConvNeXt V1 weights for size {size!r} (have: {known})")
    licence = declared_licence(name)
    if licence not in PERMISSIVE_LICENCES:
        raise ValueError(
            f"{name} is declared {licence}, which is not one of "
            f"{', '.join(sorted(PERMISSIVE_LICENCES))}"
        )

    v1 = _load_v1(name)
    v2 = create_model(
        model_size=size, pretrained=False, num_classes=0, drop_path_rate=0.0
    ).eval()
    state = convert_v1_state(v1.state_dict(), v2.state_dict())
    v2.load_state_dict(state)

    images = torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        reference = v1(images)
        difference = float((v2(images) - reference).abs().max())
        scale = max(1.0, float(reference.abs().max()))
    if difference > _TOLERANCE * scale:
        raise RuntimeError(
            f"the converted trunk does not reproduce {name}: features differ by "
            f"{difference:.3g} (largest feature {scale:.3g})"
        )
    return state, {
        "model_id": f"timm/{name}",
        "license": licence,
        "source_url": f"https://huggingface.co/timm/{name}",
        "max_abs_difference": difference,
    }


def write_v1_trunk(size: str, path: Path, weights: str | None = None) -> dict:
    """Build the trunk and save it where ``backbone_init`` can load it.

    The file is a bare backbone state dict (safetensors) whose metadata names
    the V1 weights it was converted from and their licence. Returns the source
    info plus the ``path`` written.
    """
    from safetensors.torch import save_file

    state, info = build_v1_trunk(size, weights)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.contiguous() for key, value in state.items()},
        str(path),
        metadata={
            "architecture": "convnextv2",
            "convnextv2_size": size,
            "converted_from": info["model_id"],
            "license": info["license"],
            "source_url": info["source_url"],
            "max_abs_difference": f"{info['max_abs_difference']:.3g}",
        },
    )
    return {**info, "path": str(path)}
