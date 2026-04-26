"""Build functions for :class:`bittrainer.smart_cache.SmartCache`.

The build function receives a sample dict and must return a CHW uint8 numpy
array ready to be embedded in a ``.pt`` cache entry. Centralising the default
here keeps the crop/resize/skin-normalise recipe consistent across the concept
and group trainers.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def build_image_tensor(sample: dict) -> np.ndarray:
    """Default build: open, face-aware crop to bucket aspect, LANCZOS resize,
    optional skin-normalise, return CHW uint8.

    Sample fields:
    - ``path`` (str)
    - ``bucket`` (tuple[int, int])
    - ``face_bbox`` (optional [x1, y1, x2, y2])
    - ``skin_normalise`` (bool)
    """
    from bittrainer.face_crop import face_aware_crop

    bw, bh = int(sample["bucket"][0]), int(sample["bucket"][1])
    face_bbox = sample.get("face_bbox")
    skin_normalise = bool(sample.get("skin_normalise", False))

    img = Image.open(sample["path"]).convert("RGB")
    img = face_aware_crop(img, bw / bh, face_bbox)
    img = img.resize((bw, bh), Image.LANCZOS)

    if skin_normalise:
        from bittrainer.skin_normalise import SkinNormalise
        img = SkinNormalise()(img)

    arr = np.asarray(img).transpose(2, 0, 1)  # HWC → CHW
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return np.ascontiguousarray(arr)


def make_build_fn() -> Callable[[dict], np.ndarray]:
    """Factory so callers can compose/wrap the default build if needed."""
    return build_image_tensor
