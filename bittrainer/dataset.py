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

from bittrainer.image_cache import load_or_resize
from bittrainer.image_utils import is_supported_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixed aspect ratio buckets (9 named ratios, ~0.512 MP, 32px-aligned)
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


def find_nearest_bucket(orig_w: int, orig_h: int) -> tuple[int, int]:
    ratio = orig_w / orig_h
    best_idx = 0
    best_diff = abs(ratio - _BUCKET_RATIOS[0])
    for i, br in enumerate(_BUCKET_RATIOS[1:], 1):
        diff = abs(ratio - br)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    return ASPECT_RATIO_BUCKETS[best_idx]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transform() -> transforms.Compose:
    """Standard training augmentation."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform() -> transforms.Compose:
    """Validation transform — no augmentation."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_heavy_augment_transform() -> transforms.Compose:
    """Heavier augmentation for when negatives < positives."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Skin-normalised transform variants
# ---------------------------------------------------------------------------

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
    """Caches image (width, height) keyed by absolute path + mtime.

    Persisted as a JSON file so subsequent training runs skip all
    PIL.Image.open() calls during dataset init.
    """

    def __init__(self, cache_path: Path):
        self._cache_path = cache_path
        self._data: dict[str, tuple[float, int, int]] = {}  # path -> (mtime, w, h)
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
        """Return cached (w, h) if fresh, else None."""
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
    """List image files in folder/{label}/{split}/."""
    d = folder / label / split
    if not d.is_dir():
        return []
    return sorted(f for f in d.iterdir() if f.is_file() and is_supported_image(f))


class ConceptDataset(Dataset):
    """Dataset for a single concept's training or validation split.

    For training: uses all positives + random negative subset (reshuffled per epoch).
    For validation: uses all images.
    """

    def __init__(
        self,
        concept_folder: str | Path,
        split: str = "train",
        *,
        neg_pos_ratio: float = 1.0,
        transform: Any | None = None,
        extra_positive_dirs: list[str] | None = None,
        dim_cache: _DimensionCache | None = None,
        face_bboxes: dict[str, list[int]] | None = None,
    ):
        self.concept_folder = Path(concept_folder)
        self.split = split
        self.neg_pos_ratio = neg_pos_ratio
        self.transform = transform
        self._face_bboxes: dict[str, list[int]] = face_bboxes or {}

        # Gather all available images
        self._positive_paths = _list_split_images(self.concept_folder, "positive", split)
        self._all_negative_paths = _list_split_images(self.concept_folder, "negative", split)

        for extra_dir in (extra_positive_dirs or []):
            extra_split_dir = Path(extra_dir) / "positive" / split
            if extra_split_dir.is_dir():
                self._positive_paths.extend(
                    p for p in sorted(extra_split_dir.iterdir())
                    if p.is_file() and is_supported_image(p)
                )

        # Disk cache for resized images (survives across training runs)
        self._cache_dir = self.concept_folder / ".resize_cache"

        # Shared dimension cache (avoids 30k+ PIL opens during init)
        self._dim_cache = dim_cache or _DimensionCache(self._cache_dir / "dimensions.json")

        # Pre-compute bucket info for ALL known paths (once)
        self._path_info: dict[str, dict] = {}
        self._precompute_path_info(self._positive_paths, label=1)
        self._precompute_path_info(self._all_negative_paths, label=0)
        self._dim_cache.flush()

        # Build initial sample list
        self.samples: list[dict] = []
        self._build_samples()

    def _precompute_path_info(self, paths: list[Path], label: int) -> None:
        """Read dimensions for a list of paths, using the cache."""
        for p in paths:
            key = str(p)
            if key in self._path_info:
                continue
            dims = self._dim_cache.get(p)
            if dims is None:
                dims = self._read_image_size(p)
                self._dim_cache.put(p, dims[0], dims[1])
            bucket = find_nearest_bucket(dims[0], dims[1])
            self._path_info[key] = {
                "path": key, "label": label,
                "bucket": bucket,
            }

    def _build_samples(self) -> None:
        """Build the sample list: all positives + sampled negatives."""
        self.samples = [self._path_info[str(p)] for p in self._positive_paths]

        if self.split == "val":
            neg_paths = self._all_negative_paths
        else:
            num_neg = min(
                len(self._all_negative_paths),
                max(1, round(len(self._positive_paths) * self.neg_pos_ratio)),
            )
            neg_paths = random.sample(
                self._all_negative_paths,
                min(num_neg, len(self._all_negative_paths)),
            )

        self.samples.extend(self._path_info[str(p)] for p in neg_paths)

    def reshuffle_negatives(self) -> None:
        """Re-randomise the negative subset (call once per epoch)."""
        if self.split != "train":
            return
        self._build_samples()

    @staticmethod
    def _read_image_size(path: Path | str) -> tuple[int, int]:
        """Get image dimensions without fully loading pixel data."""
        with Image.open(path) as img:
            return img.size  # (width, height)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, tuple[int, int]]:
        sample = self.samples[idx]
        bucket = sample["bucket"]
        face_bbox = self._face_bboxes.get(sample["path"])
        img = load_or_resize(sample["path"], bucket, self._cache_dir, face_bbox=face_bbox)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = TF.to_tensor(img)

        return img, sample["label"], bucket


# ---------------------------------------------------------------------------
# Bucket batch sampler
# ---------------------------------------------------------------------------

class BucketBatchSampler(Sampler):
    """Groups samples by bucket so each batch has the same image dimensions.

    ``undersized_policy`` controls what happens when a bucket has fewer
    samples than ``batch_size``:

    * ``"keep"``  – emit the undersized batch as-is (default).
    * ``"drop"``  – discard the bucket entirely.
    * ``"duplicate"`` – duplicate random samples from the same bucket to
      fill the batch to ``batch_size``.
    """

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
        # Group indices by bucket
        bucket_indices: dict[tuple[int, int], list[int]] = {}
        for i, sample in enumerate(self.dataset.samples):
            bucket = sample["bucket"]
            bucket_indices.setdefault(bucket, []).append(i)

        # Shuffle within each bucket, then yield batches
        batches = []
        for bucket, indices in bucket_indices.items():
            random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size:
                    if self.drop_last:
                        continue
                    if self.undersized_policy == "drop":
                        continue
                    if self.undersized_policy == "duplicate":
                        # Pad by duplicating random indices from the same bucket
                        shortfall = self.batch_size - len(batch)
                        batch = batch + [random.choice(batch) for _ in range(shortfall)]
                    # "keep" falls through — emit as-is
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
    """Create a bucket-aware batch sampler for the dataset."""
    return BucketBatchSampler(dataset, batch_size, drop_last, undersized_policy)
