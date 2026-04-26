"""Skin-based illuminant normalisation transform.

Detects skin pixels via YCbCr thresholds, estimates the illuminant colour
cast from their log-chromaticity, and applies per-channel gains to correct
toward a D65 reference.  Returns identity when insufficient skin is detected.

The output format (rGain, gGain, bGain, exposure) is designed to be
identical between this Python implementation and the JavaScript port
in src/renderer/utils/skin-normalise.js.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# ── Calibrated reference constants (tuned via tools/skin_normalise_tuner.py) ─

REF_LOG_RG: float = 0.2321
REF_LOG_BG: float = -0.1441
REF_SKIN_LUM: float = 159.5

# ── Skin detection thresholds (YCbCr, BT.601) ───────────────────────────────

CB_MIN: float = 77.0
CB_MAX: float = 127.0
CR_MIN: float = 133.0
CR_MAX: float = 173.0
Y_MIN: float = 40.0
MIN_SKIN_FRACTION: float = 0.021

# ── Correction limits ────────────────────────────────────────────────────────

GAIN_CLAMP_LO: float = 0.88
GAIN_CLAMP_HI: float = 1.15
EXPOSURE_CLAMP_LO: float = 0.6
EXPOSURE_CLAMP_HI: float = 1.4
EXPOSURE_DAMPEN: float = 0.5
CONFIDENCE_FLOOR: float = 0.03  # skin fraction below which correction is scaled down
CONFIDENCE_FULL: float = 0.10   # skin fraction above which full correction is applied


def _detect_skin(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return (y >= Y_MIN) & (cb >= CB_MIN) & (cb <= CB_MAX) & (cr >= CR_MIN) & (cr <= CR_MAX)


def compute_skin_gains(
    pixels: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute per-channel gains and exposure from skin pixel chromaticity.

    Parameters
    ----------
    pixels : np.ndarray
        (H, W, 3) uint8 RGB image array.

    Returns
    -------
    tuple of (rGain, gGain, bGain, exposure)
        All floats, ready for multiplication.
    """
    r = pixels[:, :, 0].astype(np.float64)
    g = pixels[:, :, 1].astype(np.float64)
    b = pixels[:, :, 2].astype(np.float64)

    skin_mask = _detect_skin(r, g, b)
    skin_frac = skin_mask.sum() / skin_mask.size

    if skin_frac < MIN_SKIN_FRACTION:
        return 1.0, 1.0, 1.0, 1.0

    sr = np.maximum(r[skin_mask], 1.0)
    sg = np.maximum(g[skin_mask], 1.0)
    sb = np.maximum(b[skin_mask], 1.0)

    # If skin pixels span a very wide luminance range, they likely include
    # non-skin surfaces (walls, furniture). Reduce confidence.
    skin_lum_vals = 0.299 * sr + 0.587 * sg + 0.114 * sb
    lum_std = float(np.std(skin_lum_vals))
    if lum_std > 40.0:
        skin_frac *= 0.5  # treat as lower confidence

    # Chromaticity correction
    mean_log_rg = np.mean(np.log(sr) - np.log(sg))
    mean_log_bg = np.mean(np.log(sb) - np.log(sg))

    gain_r = np.exp(-(mean_log_rg - REF_LOG_RG))
    gain_g = 1.0
    gain_b = np.exp(-(mean_log_bg - REF_LOG_BG))

    mean_gain = (gain_r + gain_g + gain_b) / 3.0
    gain_r /= mean_gain
    gain_g /= mean_gain
    gain_b /= mean_gain

    gain_r = float(np.clip(gain_r, GAIN_CLAMP_LO, GAIN_CLAMP_HI))
    gain_g = float(np.clip(gain_g, GAIN_CLAMP_LO, GAIN_CLAMP_HI))
    gain_b = float(np.clip(gain_b, GAIN_CLAMP_LO, GAIN_CLAMP_HI))

    # Scale correction by skin confidence — low skin fraction = less correction
    if skin_frac < CONFIDENCE_FULL:
        confidence = max(0.0, (skin_frac - CONFIDENCE_FLOOR) / (CONFIDENCE_FULL - CONFIDENCE_FLOOR))
        gain_r = 1.0 + (gain_r - 1.0) * confidence
        gain_g = 1.0 + (gain_g - 1.0) * confidence
        gain_b = 1.0 + (gain_b - 1.0) * confidence

    if max(abs(gain_r - 1.0), abs(gain_g - 1.0), abs(gain_b - 1.0)) < 0.03:
        gain_r, gain_g, gain_b = 1.0, 1.0, 1.0

    # Exposure correction
    skin_lum = float(np.mean(0.299 * sr + 0.587 * sg + 0.114 * sb))
    exposure = 1.0
    if skin_lum > 1.0:
        raw_exposure = REF_SKIN_LUM / skin_lum
        exposure = 1.0 + (raw_exposure - 1.0) * EXPOSURE_DAMPEN
        exposure = float(np.clip(exposure, EXPOSURE_CLAMP_LO, EXPOSURE_CLAMP_HI))
        if abs(exposure - 1.0) < 0.03:
            exposure = 1.0

        # Also scale exposure by confidence
        if skin_frac < CONFIDENCE_FULL:
            exposure = 1.0 + (exposure - 1.0) * confidence

    return gain_r, gain_g, gain_b, exposure


class SkinNormalise:
    """PIL-compatible transform for torchvision Compose pipelines.

    Removes illuminant colour cast based on skin pixel chromaticity,
    falling back to grey-world AWB when insufficient skin is detected.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return img

        gr, gg, gb, exposure = compute_skin_gains(arr)

        if gr == 1.0 and gg == 1.0 and gb == 1.0 and exposure == 1.0:
            return img

        result = arr.astype(np.float64)
        result[:, :, 0] *= gr * exposure
        result[:, :, 1] *= gg * exposure
        result[:, :, 2] *= gb * exposure
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def __repr__(self) -> str:
        return "SkinNormalise()"
