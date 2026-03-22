"""Disk-backed resized image cache for iterative training.

Saves resized PIL images as numpy arrays so subsequent training runs skip
both the original file decode and the LANCZOS resize.

Cache layout:
    {cache_dir}/{sha1_of_path}_{bucket_w}x{bucket_h}_c.npy
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _cache_key(source_path: str, bw: int, bh: int) -> str:
    h = hashlib.sha1(source_path.encode()).hexdigest()[:16]
    return f"{h}_{bw}x{bh}_c"


def load_or_resize(
    source_path: str,
    bucket: tuple[int, int],
    cache_dir: Path | None,
    face_bbox: list[int] | None = None,
) -> Image.Image:
    """Load from cache or crop+resize from source.

    When face_bbox is provided, the image is cropped to the bucket's
    aspect ratio using face-aware positioning before resizing.
    """
    bw, bh = bucket

    if cache_dir:
        npy_path = cache_dir / f"{_cache_key(source_path, bw, bh)}.npy"
        try:
            arr = np.load(npy_path)
            return Image.fromarray(arr)
        except (OSError, ValueError):
            pass

    img = Image.open(source_path).convert("RGB")

    # Crop to bucket aspect ratio (face-aware if bbox provided)
    target_ratio = bw / bh
    from bittrainer.face_crop import face_aware_crop
    img = face_aware_crop(img, target_ratio, face_bbox)

    # Resize to exact bucket dimensions
    img = img.resize((bw, bh), Image.LANCZOS)

    if cache_dir:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, np.array(img))
        except OSError:
            logger.warning("Failed to cache resized image for %s", source_path, exc_info=True)

    return img
