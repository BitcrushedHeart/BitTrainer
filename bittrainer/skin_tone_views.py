"""Skin Tone V2 dual-view support (Bitcrush Engine ISSUE-0217, spec §8).

The Engine exports ``skin_tone_normalisation.json`` into the group folder:
a frozen calibration reference plus, per labelled image, the colour
transform (white-balance gains, exposure, saturation, contrast) that
corrects the image toward that reference. Training consumes it as a second
*view* of the same full frame:

- train: the normalised view is a stochastic augmentation (probability
  ``skin_tone_dual_view_prob`` per fetch) applied to the cached original
  tensor — no extra cache entries, no sample duplication;
- validation: two deterministic passes (view off / on) score
  ``original`` / ``normalized`` / ``dual`` macro-F1 separately.

Leakage guard (§8.2): transforms arrive pre-clamped from the Engine, and
entries flagged ``excluded`` (extreme magnitude) are never applied — the
original view always trains.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

# Oklab matrices (Ottosson 2020) — must match the Engine's colour.py.
_M1 = torch.tensor(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=torch.float32,
)
_M2 = torch.tensor(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=torch.float32,
)
_M1_INV = torch.linalg.inv(_M1)
_M2_INV = torch.linalg.inv(_M2)


@dataclass(frozen=True)
class ViewParams:
    wb_gains: tuple[float, float, float]
    exposure_gain: float
    saturation_scale: float
    contrast_scale: float
    magnitude: float
    excluded: bool


class SkinToneViewBank:
    """Per-image normalisation transforms, keyed by absolute path with a
    unique-basename fallback (Engine keys by source path AND copied path;
    the dataset may see either, or a re-rooted copy of one)."""

    def __init__(self, images: dict[str, dict]):
        self._by_path: dict[str, ViewParams] = {}
        by_name: dict[str, ViewParams | None] = {}
        for raw_path, raw in images.items():
            if not isinstance(raw, dict):
                continue
            gains = raw.get("wb_gains") or [1.0, 1.0, 1.0]
            params = ViewParams(
                wb_gains=(float(gains[0]), float(gains[1]), float(gains[2])),
                exposure_gain=float(raw.get("exposure_gain", 1.0)),
                saturation_scale=float(raw.get("saturation_scale", 1.0)),
                contrast_scale=float(raw.get("contrast_scale", 1.0)),
                magnitude=float(raw.get("magnitude", 0.0)),
                excluded=bool(raw.get("excluded", False)),
            )
            key = _norm_path(raw_path)
            self._by_path[key] = params
            name = Path(raw_path).name.lower()
            # None marks an ambiguous basename — fallback disabled for it.
            by_name[name] = None if name in by_name else params
        self._by_name = {k: v for k, v in by_name.items() if v is not None}

    def __len__(self) -> int:
        return len(self._by_path)

    def lookup(self, path: str) -> ViewParams | None:
        params = self._by_path.get(_norm_path(path))
        if params is not None:
            return params
        return self._by_name.get(Path(path).name.lower())


def _norm_path(path: str) -> str:
    return str(Path(path)).replace("\\", "/").lower()


def load_view_bank(manifest_path: str | Path) -> SkinToneViewBank | None:
    """Read the Engine's manifest; None when absent/unreadable (dual-view
    silently disabled — the original view always trains)."""
    try:
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        images = data.get("images", {})
        if not isinstance(images, dict) or not images:
            return None
        return SkinToneViewBank(images)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


def apply_view(tensor: torch.Tensor, params: ViewParams) -> torch.Tensor:
    """Apply the normalisation transform to a CHW uint8 tensor.

    Mirrors the Engine's ``normalise_image`` model: von Kries white balance
    in linear light, uniform exposure gain in gamma space, then saturation /
    contrast in Oklab (contrast pivots on the frame-mean lightness — the
    Engine's mask-local pivot isn't available here; both scales are gentle
    and pre-clamped, so the approximation is bounded).
    """
    if params.excluded:
        return tensor
    x = tensor.to(torch.float32) / 255.0  # CHW, 0..1 gamma sRGB

    gains = torch.tensor(params.wb_gains, dtype=torch.float32).view(3, 1, 1)
    if not torch.allclose(gains, torch.ones_like(gains)):
        x = _linear_to_srgb((_srgb_to_linear(x) * gains).clamp(0.0, 1.0))

    if abs(params.exposure_gain - 1.0) > 1e-3:
        x = (x * params.exposure_gain).clamp(0.0, 1.0)

    sat, ctr = params.saturation_scale, params.contrast_scale
    if abs(sat - 1.0) > 1e-3 or abs(ctr - 1.0) > 1e-3:
        lin = _srgb_to_linear(x)
        flat = lin.permute(1, 2, 0).reshape(-1, 3)  # (N, 3)
        lms = flat @ _M1.T
        lab = torch.sign(lms) * lms.abs().pow(1.0 / 3.0) @ _M2.T
        pivot = lab[:, 0].mean()
        lab[:, 0] = pivot + (lab[:, 0] - pivot) * ctr
        lab[:, 1:] = lab[:, 1:] * sat
        lms = (lab @ _M2_INV.T) ** 3
        lin = (lms @ _M1_INV.T).clamp(0.0, 1.0)
        x = _linear_to_srgb(lin.reshape(tensor.shape[1], tensor.shape[2], 3).permute(2, 0, 1))

    return (x * 255.0).round().clamp(0, 255).to(torch.uint8)


def maybe_apply_view(
    tensor: torch.Tensor,
    path: str,
    bank: SkinToneViewBank | None,
    *,
    probability: float,
    force: bool,
) -> torch.Tensor:
    """Dataset hook: apply the image's normalised view with *probability*
    (train augmentation) or always when *force* (validation view pass)."""
    if bank is None:
        return tensor
    if not force and (probability <= 0.0 or random.random() >= probability):
        return tensor
    params = bank.lookup(path)
    if params is None:
        return tensor
    return apply_view(tensor, params)
