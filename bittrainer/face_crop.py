from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


class FaceBBoxCache:
    """Disk-cached face bounding boxes, keyed by image path + mtime."""

    def __init__(self, cache_path: Path):
        self._cache_path = cache_path
        self._data: dict[str, tuple[float, list[int]]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if self._cache_path.exists():
            try:
                raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
                self._data = {k: (v[0], v[1]) for k, v in raw.items()}
            except (json.JSONDecodeError, OSError, ValueError):
                logger.warning("Corrupt face bbox cache, rebuilding")
                self._data = {}

    def get(self, path: str) -> list[int] | None:
        """Return cached [x1, y1, x2, y2] if fresh, else None.

        Returns an empty list if no face was detected (cached negative).
        """
        entry = self._data.get(path)
        if entry is None:
            return None
        cached_mtime, bbox = entry
        try:
            current_mtime = os.path.getmtime(path)
        except OSError:
            return None
        if abs(current_mtime - cached_mtime) > 0.01:
            return None
        return bbox

    def put(self, path: str, bbox: list[int]) -> None:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        self._data[path] = (mtime, bbox)
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(self._data, separators=(",", ":")),
                encoding="utf-8",
            )
            self._dirty = False
        except OSError:
            logger.warning("Failed to write face bbox cache", exc_info=True)

    def uncached_paths(self, all_paths: list[str]) -> list[str]:
        """Return paths that have no valid cache entry."""
        return [p for p in all_paths if self.get(p) is None]


def precompute_face_bboxes(
    image_paths: list[str],
    cache: FaceBBoxCache,
    face_model_path: str,
    device: str = "cuda",
    batch_size: int = 32,
) -> None:
    """Run YOLO face detection on uncached images and populate the cache.

    Loads the YOLO model, runs batch inference, stores the union bbox
    of all detected faces per image, then unloads the model.
    """
    import torch

    uncached = cache.uncached_paths(image_paths)
    if not uncached:
        logger.info("All %d images have cached face bboxes", len(image_paths))
        return

    logger.info("Computing face bboxes for %d/%d images", len(uncached), len(image_paths))

    from ultralytics import YOLO
    model = YOLO(face_model_path)
    model.to(device)

    for start in range(0, len(uncached), batch_size):
        batch_paths = uncached[start:start + batch_size]
        try:
            results = model(batch_paths, verbose=False, device=device)
            for path, result in zip(batch_paths, results):
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    x1 = int(xyxy[:, 0].min())
                    y1 = int(xyxy[:, 1].min())
                    x2 = int(xyxy[:, 2].max())
                    y2 = int(xyxy[:, 3].max())
                    cache.put(path, [x1, y1, x2, y2])
                else:
                    cache.put(path, [])
        except (RuntimeError, OSError) as exc:
            logger.warning("Face detection batch failed: %s", exc)
            for p in batch_paths:
                cache.put(p, [])

    cache.flush()

    del model
    torch.cuda.empty_cache()
    logger.info("Face bbox pre-computation complete, model unloaded")


def face_aware_crop(
    img: Image.Image,
    target_ratio: float,
    face_bbox: list[int] | None,
) -> Image.Image:
    """Crop image to target aspect ratio, keeping faces in frame.

    Args:
        img: PIL Image to crop.
        target_ratio: Target width/height ratio.
        face_bbox: [x1, y1, x2, y2] union of all faces, or empty/None for centre crop.

    Returns:
        Cropped PIL Image at the target aspect ratio.
    """
    img_w, img_h = img.size
    img_ratio = img_w / img_h

    if abs(img_ratio - target_ratio) < 0.01:
        return img

    if target_ratio > img_ratio:
        # Crop height (image is too tall)
        new_h = int(img_w / target_ratio)
        new_w = img_w
    else:
        # Crop width (image is too wide)
        new_w = int(img_h * target_ratio)
        new_h = img_h

    if face_bbox and len(face_bbox) == 4:
        fx1, fy1, fx2, fy2 = face_bbox
        face_cx = (fx1 + fx2) / 2
        face_cy = (fy1 + fy2) / 2

        # Position crop centred on face, clamped to image bounds
        left = int(face_cx - new_w / 2)
        top = int(face_cy - new_h / 2)

        # Ensure face bbox fits within crop
        left = min(left, max(0, fx1 - int(new_w * 0.1)))
        top = min(top, max(0, fy1 - int(new_h * 0.1)))

        # Ensure crop doesn't extend beyond face bbox far side
        if left + new_w < fx2:
            left = max(0, fx2 - new_w + int(new_w * 0.1))
        if top + new_h < fy2:
            top = max(0, fy2 - new_h + int(new_h * 0.1))
    else:
        # Centre crop
        left = (img_w - new_w) // 2
        top = (img_h - new_h) // 2

    # Clamp to image bounds
    left = max(0, min(left, img_w - new_w))
    top = max(0, min(top, img_h - new_h))

    return img.crop((left, top, left + new_w, top + new_h))
