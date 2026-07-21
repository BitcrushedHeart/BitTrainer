"""Training dataset with aspect ratio bucketing for ConvNeXt V2."""

from __future__ import annotations

import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import functional as TF

from bittrainer.image_utils import is_supported_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixed aspect ratio buckets (9 named ratios, 512px base ≈ 0.26 MP, 32px-aligned)
#
# Every bucket holds ~512×512 px of area (256k–262k) and every dimension is a
# multiple of 32 so ConvNeXt's stride-32 stem produces clean feature maps.
# Editing these dims changes the SmartCache resolution key — stale tensors from
# a prior table are rebuilt on access (see GroupDataset.__getitem__ shape guard).
# ---------------------------------------------------------------------------

ASPECT_RATIO_BUCKETS: list[tuple[int, int]] = [
    (512, 512),   # 1:1
    (576, 448),   # 4:3
    (448, 576),   # 3:4
    (672, 384),   # 16:9
    (384, 672),   # 9:16
    (736, 352),   # 2:1
    (352, 736),   # 1:2
    (800, 320),   # 21:9
    (320, 800),   # 9:21
]

_BUCKET_RATIOS: list[float] = [w / h for w, h in ASPECT_RATIO_BUCKETS]

DEFAULT_TRAIN_RESOLUTION = 512


def scaled_buckets(train_resolution: int = DEFAULT_TRAIN_RESOLUTION) -> list[tuple[int, int]]:
    """The aspect-bucket table scaled to a per-group training resolution.

    ``train_resolution`` is the square-bucket side; every bucket's dims scale
    by ``train_resolution / 512`` and snap to multiples of 32 (ConvNeXt stem
    stride) so shapes stay kernel-friendly. ``512`` (or ``<=0``/None) returns
    the canonical table unchanged — SmartCache keys embed the bucket dims, so
    the default table keeps every existing cache entry valid, while any other
    resolution produces new dims and therefore fresh cache entries (stale ones
    are reclaimed by the cache GC, and the dataset shape guard rebuilds on
    mismatch).
    """
    if not train_resolution or train_resolution == DEFAULT_TRAIN_RESOLUTION:
        return ASPECT_RATIO_BUCKETS
    scale = train_resolution / DEFAULT_TRAIN_RESOLUTION
    return [
        (max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32))
        for w, h in ASPECT_RATIO_BUCKETS
    ]


def find_nearest_bucket(
    orig_w: int, orig_h: int, train_resolution: int = DEFAULT_TRAIN_RESOLUTION
) -> tuple[int, int]:
    # Aspect selection always runs on the canonical ratios (scaling preserves
    # them up to /32 snapping); only the returned dims scale.
    ratio = orig_w / orig_h
    best_idx = 0
    best_diff = abs(ratio - _BUCKET_RATIOS[0])
    for i, br in enumerate(_BUCKET_RATIOS[1:], 1):
        diff = abs(ratio - br)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    return scaled_buckets(train_resolution)[best_idx]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_heavy_augment_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_skin_normalised_train_transform() -> transforms.Compose:
    from bittrainer.skin_normalise import SkinNormalise
    return transforms.Compose([
        SkinNormalise(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_skin_normalised_val_transform() -> transforms.Compose:
    from bittrainer.skin_normalise import SkinNormalise
    return transforms.Compose([
        SkinNormalise(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_skin_normalised_heavy_augment_transform() -> transforms.Compose:
    from bittrainer.skin_normalise import SkinNormalise
    return transforms.Compose([
        SkinNormalise(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Image dimension cache (persisted to disk, keyed by path + mtime)
# ---------------------------------------------------------------------------

class _DimensionCache:
    """Caches image (width, height) keyed by absolute path + mtime."""

    def __init__(self, cache_path: Path):
        self._cache_path = cache_path
        self._data: dict[str, tuple[float, int, int]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if self._cache_path.exists():
            try:
                raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
                self._data = {k: tuple(v) for k, v in raw.items()}
            except (json.JSONDecodeError, OSError, ValueError):
                logger.warning("Corrupt dimension cache, rebuilding")
                self._data = {}

    def get(self, path: Path | str) -> tuple[int, int] | None:
        key = str(path)
        entry = self._data.get(key)
        if entry is None:
            return None
        cached_mtime, w, h = entry
        try:
            current_mtime = os.path.getmtime(key)
        except OSError:
            return None
        if abs(current_mtime - cached_mtime) > 0.01:
            return None
        return (w, h)

    def put(self, path: Path | str, w: int, h: int) -> None:
        key = str(path)
        try:
            mtime = os.path.getmtime(key)
        except OSError:
            return
        self._data[key] = (mtime, w, h)
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
            logger.warning("Failed to write dimension cache", exc_info=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _list_split_images(folder: Path, label: str, split: str) -> list[Path]:
    d = folder / label / split
    if not d.is_dir():
        return []
    return sorted(f for f in d.iterdir() if f.is_file() and is_supported_image(f))


def _list_split_images_flat(folder: Path, split: str) -> list[Path]:
    d = folder / split
    if not d.is_dir():
        return []
    return sorted(f for f in d.iterdir() if f.is_file() and is_supported_image(f))


class ConceptDataset(Dataset):
    """Dataset for a single concept's training or validation split.

    When a :class:`bittrainer.smart_cache.SmartCache` is provided, images are
    loaded as pre-resized CHW uint8 tensors directly from the cache. Cache
    misses fall back to on-the-fly PIL decode via the build function.

    In ``sourceless=True`` mode the sample list is reconstructed from the
    cache index — the source concept folders do not need to exist on disk.
    """

    def __init__(
        self,
        concept_folder: str | Path,
        split: str = "train",
        *,
        neg_pos_ratio: float = 1.0,
        transform: Any | None = None,
        extra_positive_dirs: list[str] | None = None,
        negative_dirs: list[str] | None = None,
        hard_negative_paths: list[str] | None = None,
        hard_negative_weight: int = 3,
        dim_cache: _DimensionCache | None = None,
        face_bboxes: dict[str, list[int]] | None = None,
        skin_normalise: bool = False,
        cache: Any | None = None,           # SmartCache instance
        sourceless: bool = False,
        concept_name: str = "",
        train_resolution: int = DEFAULT_TRAIN_RESOLUTION,
    ):
        self.concept_folder = Path(concept_folder)
        self.split = split
        self.transform = transform
        self._face_bboxes: dict[str, list[int]] = face_bboxes or {}
        self._skin_normalise = skin_normalise
        self._neg_pos_ratio = neg_pos_ratio
        self._hard_negative_weight = hard_negative_weight
        self._concept_name = concept_name or self.concept_folder.name
        self._cache = cache
        self._sourceless = sourceless
        self._train_resolution = int(train_resolution or DEFAULT_TRAIN_RESOLUTION)

        if sourceless:
            self._init_sourceless()
            return

        # Positives: flat layout first, fall back to legacy
        self._positive_paths = _list_split_images_flat(self.concept_folder, split)
        if not self._positive_paths:
            self._positive_paths = _list_split_images(self.concept_folder, "positive", split)

        for extra_dir in (extra_positive_dirs or []):
            extra_folder = Path(extra_dir)
            extra_paths = _list_split_images_flat(extra_folder, split)
            if not extra_paths:
                extra_paths = _list_split_images(extra_folder, "positive", split)
            self._positive_paths.extend(extra_paths)

        self._all_negative_paths: list[Path] = []
        self._has_cross_concept_negatives = bool(negative_dirs)
        if negative_dirs:
            for neg_dir in negative_dirs:
                neg_folder = Path(neg_dir)
                paths = _list_split_images_flat(neg_folder, split)
                if not paths:
                    paths = _list_split_images(neg_folder, "positive", split)
                self._all_negative_paths.extend(paths)
        else:
            self._all_negative_paths = _list_split_images(self.concept_folder, "negative", split)

        self._hard_negative_paths: list[Path] = [
            Path(p) for p in (hard_negative_paths or []) if Path(p).is_file()
        ]

        self._cache_dir = self.concept_folder / ".resize_cache"
        self._dim_cache = dim_cache or _DimensionCache(self._cache_dir / "dimensions.json")

        self._path_info: dict[str, dict] = {}
        self._precompute_path_info(self._positive_paths, label=1)
        self._precompute_path_info(self._all_negative_paths, label=0)
        self._precompute_path_info(self._hard_negative_paths, label=0)
        self._dim_cache.flush()

        self.samples: list[dict] = []
        self._build_samples()

    def _init_sourceless(self) -> None:
        if self._cache is None:
            raise RuntimeError("sourceless=True requires a SmartCache instance")
        entries = self._cache.iter_sourceless()
        self.samples = [s for s in entries if s.get("split") == self.split]
        self._positive_paths = [Path(s["path"]) for s in self.samples if s["label"] == 1]
        self._all_negative_paths = [Path(s["path"]) for s in self.samples if s["label"] == 0]
        self._hard_negative_paths = []
        self._has_cross_concept_negatives = False

    def _precompute_path_info(self, paths: list[Path], label: int) -> None:
        for p in paths:
            key = str(p)
            if key in self._path_info:
                continue
            dims = self._dim_cache.get(p)
            if dims is None:
                dims = self._read_image_size(p)
                self._dim_cache.put(p, dims[0], dims[1])
            bucket = find_nearest_bucket(dims[0], dims[1], self._train_resolution)
            self._path_info[key] = {
                "path": key,
                "label": label,
                "bucket": bucket,
                "concept_name": self._concept_name,
                "split": self.split,
                "skin_normalise": self._skin_normalise,
                "face_bbox": self._face_bboxes.get(key),
            }

    def _build_samples(self) -> None:
        self.samples = [self._path_info[str(p)] for p in self._positive_paths]

        hard_neg_samples = [self._path_info[str(p)] for p in self._hard_negative_paths]
        for _ in range(self._hard_negative_weight):
            self.samples.extend(hard_neg_samples)

        num_pos = len(self._positive_paths)
        max_neg = int(num_pos * self._neg_pos_ratio) if self._neg_pos_ratio > 0 else len(self._all_negative_paths)
        hard_slots = len(hard_neg_samples) * self._hard_negative_weight
        remaining = max(0, max_neg - hard_slots)

        neg_samples = [self._path_info[str(p)] for p in self._all_negative_paths]
        if len(neg_samples) > remaining:
            neg_samples = random.sample(neg_samples, remaining)

        self.samples.extend(neg_samples)

    def resample_negatives(self) -> None:
        if self._sourceless or not self._has_cross_concept_negatives:
            return
        self._build_samples()

    def set_cache(self, cache: Any) -> None:
        """Attach a SmartCache after construction (used by trainer warm phase)."""
        self._cache = cache

    def refresh_face_bboxes(self, face_bboxes: dict[str, list[int]]) -> None:
        self._face_bboxes = face_bboxes
        for key, info in self._path_info.items():
            info["face_bbox"] = face_bboxes.get(key)

    @staticmethod
    def _read_image_size(path: Path | str) -> tuple[int, int]:
        with Image.open(path) as img:
            return img.size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, tuple[int, int]]:
        sample = self.samples[idx]
        bucket = sample["bucket"]
        bw, bh = int(bucket[0]), int(bucket[1])

        if self._cache is not None:
            result = self._cache.get(sample["path"])
            if result is not None:
                tensor, _ = result
                if tuple(tensor.shape[-2:]) == (bh, bw):
                    return tensor, sample["label"], tuple(bucket)
                # Cached tensor was built under a different aspect-ratio bucket
                # table; its size no longer matches this sample's bucket and
                # would explode the bucket collate. Rebuild from source.
                if self._sourceless:
                    raise RuntimeError(
                        f"Sourceless training: cached tensor for '{sample['path']}' "
                        f"is {tuple(tensor.shape[-2:])}, expected {(bh, bw)}. The "
                        f"cache predates the current bucket table — rebuild it."
                    )
            elif self._sourceless:
                raise RuntimeError(
                    f"Sourceless training: cache miss for '{sample['path']}'. "
                    "The cache is incomplete — rebuild it with sourceless disabled."
                )

        # Fallback: build on-the-fly
        from bittrainer.cache_builders import build_image_tensor
        import numpy as np
        arr = build_image_tensor(sample)
        img_tensor = torch.from_numpy(np.ascontiguousarray(arr))

        if self.transform is not None:
            pil_img = Image.fromarray(arr.transpose(1, 2, 0))
            return self.transform(pil_img), sample["label"], tuple(bucket)

        return img_tensor, sample["label"], tuple(bucket)


# ---------------------------------------------------------------------------
# Bucket batch sampler
# ---------------------------------------------------------------------------

class BucketBatchSampler(Sampler):
    def __init__(
        self,
        dataset: ConceptDataset,
        batch_size: int,
        drop_last: bool = False,
        undersized_policy: str = "keep",
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.undersized_policy = undersized_policy

    def __iter__(self):
        bucket_indices: dict[tuple[int, int], list[int]] = {}
        for i, sample in enumerate(self.dataset.samples):
            bucket = sample["bucket"]
            bucket_indices.setdefault(bucket, []).append(i)

        batches = []
        for bucket, indices in bucket_indices.items():
            random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) < self.batch_size:
                    if self.drop_last or self.undersized_policy == "drop":
                        continue
                    if self.undersized_policy == "duplicate":
                        shortfall = self.batch_size - len(batch)
                        batch = batch + [random.choice(batch) for _ in range(shortfall)]
                batches.append(batch)

        random.shuffle(batches)
        yield from batches

    def __len__(self):
        total = 0
        bucket_indices: dict[tuple[int, int], int] = {}
        for sample in self.dataset.samples:
            bucket = sample["bucket"]
            bucket_indices[bucket] = bucket_indices.get(bucket, 0) + 1
        for count in bucket_indices.values():
            if self.undersized_policy == "drop" or self.drop_last:
                total += count // self.batch_size
            else:
                total += math.ceil(count / self.batch_size)
        return total


def build_bucket_batch_sampler(
    dataset: ConceptDataset,
    batch_size: int,
    drop_last: bool = False,
    undersized_policy: str = "keep",
) -> BucketBatchSampler:
    return BucketBatchSampler(dataset, batch_size, drop_last, undersized_policy)
