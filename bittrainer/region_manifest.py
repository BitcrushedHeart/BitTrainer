"""The Engine's ``regions_manifest.json`` and the box geometry around it
(Bitcrush ISSUE-0993).

Before a crop-assisted group trains, the Engine materialises every labelled
region as a crop file and writes this manifest beside them::

    {
      "version": 1,
      "crop_pad": 0.15,
      "region_types": ["Face"],
      "crops": [
        {"path": ".../.region_crops/train/<crop_hash>.png",
         "source": "<source image path>", "box": [x1, y1, x2, y2],
         "type": 0, "class": "Blue", "split": "train"}
      ],
      "images": [
        {"source": "<source image path>", "paths": ["<source>", "<group copy>"],
         "split": "train",
         "boxes": [[x1, y1, x2, y2, type]], "rejects": [[x1, y1, x2, y2, type]]}
      ]
    }

``crops`` are classification samples (the crop view); the train ones are also
listed in the run's ``extra_paths_train`` so they join the dataset as ordinary
class members. ``images`` carry detector supervision: ``boxes`` are the regions
the user kept, ``rejects`` the proposals they dismissed. An image is listed
under every path the dataset may meet it by (its source path and its
group-folder copy).

Image ``boxes`` / ``rejects`` are normalised to the STORED pixels — the frame
:func:`bittrainer.cache_builders.build_image_tensor` decodes, which does not
apply EXIF orientation. The Engine does that mapping; this module only follows
the pixels through the aspect crop.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

Box = tuple[float, float, float, float]
TypedBox = tuple[float, float, float, float, int]
Rect = tuple[int, int, int, int]

# A box the aspect crop cuts down to less than this share of its own area is
# dropped: what is left no longer shows the thing the user drew around.
MIN_VISIBLE_FRACTION = 0.3

_GENERIC_TYPE = "region"


def norm_path(path: str | os.PathLike) -> str:
    """Path identity for manifest lookups: case- and slash-insensitive on Windows."""
    return os.path.normcase(os.path.normpath(str(path)))


@dataclass(frozen=True)
class RegionCrop:
    path: str
    source: str
    box: Box
    type: int
    class_name: str | None
    split: str


@dataclass(frozen=True)
class RegionImage:
    source: str
    paths: tuple[str, ...]
    split: str
    boxes: tuple[TypedBox, ...]
    rejects: tuple[TypedBox, ...]

    def supervised_types(self) -> set[int]:
        """Region types the user reviewed on this image (kept or dismissed)."""
        return {int(b[4]) for b in self.boxes} | {int(b[4]) for b in self.rejects}


class RegionManifest:
    def __init__(
        self,
        *,
        type_names: list[str],
        crop_pad: float,
        crops: list[RegionCrop],
        images: list[RegionImage],
        path: str = "",
    ) -> None:
        self.type_names = list(type_names) or [_GENERIC_TYPE]
        self.crop_pad = float(crop_pad)
        self.crops = list(crops)
        self.images = list(images)
        self.path = path
        self._crop_by_path = {norm_path(c.path): c for c in self.crops}
        self._image_by_path: dict[str, RegionImage] = {}
        for image in self.images:
            for p in image.paths:
                self._image_by_path[norm_path(p)] = image

    @property
    def num_types(self) -> int:
        return len(self.type_names)

    def crop_for(self, path: str | os.PathLike) -> RegionCrop | None:
        return self._crop_by_path.get(norm_path(path))

    def image_for(self, path: str | os.PathLike) -> RegionImage | None:
        return self._image_by_path.get(norm_path(path))

    def images_in(self, split: str) -> list[RegionImage]:
        return [i for i in self.images if i.split == split]

    def crops_in(self, split: str) -> list[RegionCrop]:
        return [c for c in self.crops if c.split == split]

    def crop_paths_by_class(self, split: str) -> dict[str, list[str]]:
        """``extra_paths``-shaped map of the labelled crop files in *split*."""
        out: dict[str, list[str]] = {}
        for crop in self.crops_in(split):
            if crop.class_name:
                out.setdefault(crop.class_name, []).append(crop.path)
        return out


def _typed_boxes(raw: object, num_types: int) -> tuple[TypedBox, ...]:
    out: list[TypedBox] = []
    for item in raw if isinstance(raw, list) else []:
        try:
            x1, y1, x2, y2 = (float(v) for v in item[:4])
            kind = int(item[4]) if len(item) > 4 else 0
        except (TypeError, ValueError, IndexError):
            continue
        if x2 > x1 and y2 > y1 and 0 <= kind < num_types:
            out.append((x1, y1, x2, y2, kind))
    return tuple(out)


def load_region_manifest(path: str | os.PathLike | None) -> RegionManifest | None:
    """Parse the Engine's manifest; ``None`` when absent or unreadable.

    A missing manifest must never fail a run: the group then trains as an
    ordinary classifier, exactly as it did before it had regions.
    """
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Region manifest unreadable: %s", path)
        return None
    if not isinstance(data, dict):
        return None

    type_names = [str(n) for n in data.get("region_types") or []] or [_GENERIC_TYPE]
    num_types = len(type_names)

    crops: list[RegionCrop] = []
    for raw in data.get("crops") or []:
        try:
            box = tuple(float(v) for v in raw["box"][:4])
            crops.append(
                RegionCrop(
                    path=str(raw["path"]),
                    source=str(raw["source"]),
                    box=box,  # type: ignore[arg-type]
                    type=int(raw.get("type") or 0),
                    class_name=(str(raw["class"]) if raw.get("class") else None),
                    split=str(raw.get("split") or "train"),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    images: list[RegionImage] = []
    for raw in data.get("images") or []:
        try:
            source = str(raw["source"])
        except (KeyError, TypeError):
            continue
        paths = [str(p) for p in raw.get("paths") or [] if p]
        if source not in paths:
            paths.insert(0, source)
        images.append(
            RegionImage(
                source=source,
                paths=tuple(paths),
                split=str(raw.get("split") or "train"),
                boxes=_typed_boxes(raw.get("boxes"), num_types),
                rejects=_typed_boxes(raw.get("rejects"), num_types),
            )
        )

    try:
        crop_pad = float(data.get("crop_pad") or 0.0)
    except (TypeError, ValueError):
        crop_pad = 0.0
    return RegionManifest(
        type_names=type_names, crop_pad=crop_pad, crops=crops, images=images, path=str(path)
    )


def map_box_into_rect(box: Box, img_w: int, img_h: int, rect: Rect) -> Box | None:
    """A source-image box in the frame of the aspect-cropped tensor.

    *rect* is :func:`bittrainer.face_crop.aspect_crop_rect`'s
    ``(left, top, width, height)``. The box is clipped to the crop; ``None``
    when under :data:`MIN_VISIBLE_FRACTION` of it survives.
    """
    left, top, width, height = rect
    x1, y1, x2, y2 = box[0] * img_w, box[1] * img_h, box[2] * img_w, box[3] * img_h
    area = (x2 - x1) * (y2 - y1)
    cx1, cy1 = max(x1, left), max(y1, top)
    cx2, cy2 = min(x2, left + width), min(y2, top + height)
    if area <= 0 or cx2 <= cx1 or cy2 <= cy1:
        return None
    if (cx2 - cx1) * (cy2 - cy1) / area < MIN_VISIBLE_FRACTION:
        return None
    return (
        (cx1 - left) / width,
        (cy1 - top) / height,
        (cx2 - left) / width,
        (cy2 - top) / height,
    )


def map_box_from_rect(box: Box, img_w: int, img_h: int, rect: Rect) -> Box:
    """Inverse of :func:`map_box_into_rect`: a tensor-frame box (a proposal the
    model decoded) back in the source image's normalised frame."""
    left, top, width, height = rect
    return (
        (left + box[0] * width) / img_w,
        (top + box[1] * height) / img_h,
        (left + box[2] * width) / img_w,
        (top + box[3] * height) / img_h,
    )
