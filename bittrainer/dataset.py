"""Training dataset with aspect ratio bucketing for ConvNeXt V2."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import functional as TF

from bittrainer.image_utils import is_supported_image


# ---------------------------------------------------------------------------
# Target size computation (0.512 MP, multiples of 32)
# ---------------------------------------------------------------------------

_TARGET_PIXELS = 262_144  # 0.512 MP = 512*512


def compute_target_size(orig_w: int, orig_h: int) -> tuple[int, int]:
    """Resize to ~0.512MP preserving aspect ratio, dims rounded to 32."""
    aspect = orig_w / orig_h
    # Solve: w * h = TARGET_PIXELS, w/h = aspect
    h = math.sqrt(_TARGET_PIXELS / aspect)
    w = h * aspect
    w = max(32, round(w / 32) * 32)
    h = max(32, round(h / 32) * 32)
    return int(w), int(h)


# ---------------------------------------------------------------------------
# Aspect ratio buckets (32px increments around 512x512)
# ---------------------------------------------------------------------------

def _generate_buckets() -> list[tuple[int, int]]:
    """Generate aspect ratio buckets at 32px increments, ~0.512MP each."""
    buckets = []
    for w in range(320, 832, 32):
        for h in range(320, 832, 32):
            pixel_count = w * h
            if 0.7 * _TARGET_PIXELS <= pixel_count <= 1.3 * _TARGET_PIXELS:
                buckets.append((w, h))
    return sorted(set(buckets))


ASPECT_RATIO_BUCKETS: list[tuple[int, int]] = _generate_buckets()


def find_nearest_bucket(w: int, h: int) -> tuple[int, int]:
    """Find the bucket closest to the given dimensions."""
    return min(ASPECT_RATIO_BUCKETS, key=lambda b: abs(b[0] - w) + abs(b[1] - h))


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
    ):
        self.concept_folder = Path(concept_folder)
        self.split = split
        self.neg_pos_ratio = neg_pos_ratio
        self.transform = transform

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

        # Build initial sample list
        self.samples: list[dict] = []
        self._build_samples()

    def _build_samples(self) -> None:
        """Build the sample list: all positives + sampled negatives."""
        self.samples = []

        # All positives
        for p in self._positive_paths:
            w, h = self._get_image_size(p)
            tw, th = compute_target_size(w, h)
            bucket = find_nearest_bucket(tw, th)
            self.samples.append({"path": str(p), "label": 1, "target_size": (tw, th), "bucket": bucket})

        # Negatives: for train, sample at ratio; for val, use all
        if self.split == "val":
            neg_paths = self._all_negative_paths
        else:
            num_neg = min(
                len(self._all_negative_paths),
                max(1, round(len(self._positive_paths) * self.neg_pos_ratio)),
            )
            neg_paths = random.sample(self._all_negative_paths, min(num_neg, len(self._all_negative_paths)))

        for p in neg_paths:
            w, h = self._get_image_size(p)
            tw, th = compute_target_size(w, h)
            bucket = find_nearest_bucket(tw, th)
            self.samples.append({"path": str(p), "label": 0, "target_size": (tw, th), "bucket": bucket})

    def reshuffle_negatives(self) -> None:
        """Re-randomise the negative subset (call once per epoch)."""
        if self.split != "train":
            return
        self._build_samples()

    @staticmethod
    def _get_image_size(path: Path | str) -> tuple[int, int]:
        """Get image dimensions without fully loading pixel data."""
        with Image.open(path) as img:
            return img.size  # (width, height)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, tuple[int, int]]:
        sample = self.samples[idx]
        img = Image.open(sample["path"]).convert("RGB")
        tw, th = sample["target_size"]
        img = img.resize((tw, th), Image.LANCZOS)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = TF.to_tensor(img)

        return img, sample["label"], sample["bucket"]


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
