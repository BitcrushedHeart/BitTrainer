from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Callable

from PIL import Image

logger = logging.getLogger(__name__)


def select_bbox(
    detections: list[tuple[str, float, tuple[float, float, float, float]]],
    *,
    target_classes: list[str] | None = None,
    selection: str = "union",
) -> list[int]:
    """Pick the crop bbox from a set of (class_name, confidence, xyxy) detections.

    ``target_classes`` filters by detector class name (case-insensitive;
    ``None``/empty keeps everything — the face-crop behaviour). ``selection``
    is ``"union"`` (min/max envelope of all matches, the face-crop behaviour)
    or ``"highest_conf"`` (the single best matching box — the right default
    for a body-part region where a union with a false positive would drag the
    crop off-target). No match returns ``[]`` (cached negative).
    """
    wanted: set[str] | None = None
    if target_classes:
        wanted = {c.strip().lower() for c in target_classes if c and c.strip()}
    matches = [
        d for d in detections
        if wanted is None or d[0].strip().lower() in wanted
    ]
    if not matches:
        return []
    if selection == "highest_conf":
        _, _, box = max(matches, key=lambda d: d[1])
        return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
    x1 = min(d[2][0] for d in matches)
    y1 = min(d[2][1] for d in matches)
    x2 = max(d[2][2] for d in matches)
    y2 = max(d[2][3] for d in matches)
    return [int(x1), int(y1), int(x2), int(y2)]


def region_bbox_cache_name(
    model_path: str,
    target_classes: list[str] | None,
    selection: str,
) -> str:
    """Cache filename for a region-bbox pre-pass, distinct per model/classes/selection.

    The face path keeps its historical ``face_bboxes.json``; region caches are
    side-by-side so switching detector, class filter, or selection mode never
    reads another configuration's boxes.
    """
    stem = Path(model_path).stem or "model"
    classes_csv = ",".join(sorted(c.strip().lower() for c in (target_classes or []) if c))
    classes_key = hashlib.sha1(classes_csv.encode("utf-8")).hexdigest()[:8] if classes_csv else "all"
    return f"region_bboxes_{stem}_{classes_key}_{selection}.json"


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


def precompute_region_bboxes(
    image_paths: list[str],
    cache: FaceBBoxCache,
    model_path: str,
    *,
    target_classes: list[str] | None = None,
    selection: str = "union",
    device: str = "cuda",
    batch_size: int = 32,
    progress_fn: Callable[[int, int], None] | None = None,
) -> None:
    """Run YOLO detection on uncached images and populate the bbox cache.

    Loads the YOLO model, runs batch inference, stores the ``select_bbox``
    result per image (union or highest-confidence, optionally filtered to
    ``target_classes``), then unloads the model. Deduplicates paths before
    processing to avoid redundant work from oversampled training sets.
    """
    import torch

    unique_paths = list(dict.fromkeys(image_paths))
    uncached = cache.uncached_paths(unique_paths)
    if not uncached:
        logger.info("All %d unique images have cached region bboxes", len(unique_paths))
        return

    logger.info("Computing region bboxes for %d/%d unique images", len(uncached), len(unique_paths))

    from ultralytics import YOLO
    model = YOLO(model_path)
    model.to(device)

    total = len(uncached)
    for start in range(0, total, batch_size):
        batch_paths = uncached[start:start + batch_size]
        try:
            results = model(batch_paths, verbose=False, device=device)
            for path, result in zip(batch_paths, results):
                boxes = result.boxes
                detections: list[tuple[str, float, tuple[float, float, float, float]]] = []
                if boxes is not None and len(boxes) > 0:
                    names = result.names or {}
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
                    clses = boxes.cls.cpu().numpy() if boxes.cls is not None else None
                    for i in range(len(xyxy)):
                        cls_name = (
                            str(names.get(int(clses[i]), int(clses[i])))
                            if clses is not None else ""
                        )
                        conf = float(confs[i]) if confs is not None else 0.0
                        detections.append((cls_name, conf, tuple(xyxy[i][:4])))
                cache.put(path, select_bbox(
                    detections, target_classes=target_classes, selection=selection,
                ))
        except (RuntimeError, OSError) as exc:
            logger.warning("Region detection batch failed: %s", exc)
            for p in batch_paths:
                cache.put(p, [])

        done = min(start + batch_size, total)
        if progress_fn and (done % (batch_size * 10) == 0 or done == total):
            progress_fn(done, total)

    cache.flush()

    del model
    torch.cuda.empty_cache()
    logger.info("Region bbox pre-computation complete, model unloaded")


def precompute_face_bboxes(
    image_paths: list[str],
    cache: FaceBBoxCache,
    face_model_path: str,
    device: str = "cuda",
    batch_size: int = 32,
    progress_fn: Callable[[int, int], None] | None = None,
) -> None:
    """Face pre-pass: union bbox of all detections (historical behaviour)."""
    precompute_region_bboxes(
        image_paths, cache, face_model_path,
        target_classes=None, selection="union",
        device=device, batch_size=batch_size, progress_fn=progress_fn,
    )


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
        new_h = max(1, int(img_w / target_ratio))
        new_w = img_w
    else:
        # Crop width (image is too wide)
        new_w = max(1, int(img_h * target_ratio))
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
