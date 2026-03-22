"""Multi-class dataset with aspect ratio bucketing and class-balanced sampling."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import functional as TF

from bittrainer.dataset import (
    find_nearest_bucket,
    get_skin_normalised_train_transform,
    get_skin_normalised_val_transform,
    get_train_transform,
    get_val_transform,
)
from bittrainer.image_cache import load_or_resize
from bittrainer.image_utils import is_supported_image


def _list_class_images(group_folder: Path, class_name: str, split: str) -> list[Path]:
    d = group_folder / class_name / split
    if not d.is_dir():
        return []
    return sorted(f for f in d.iterdir() if f.is_file() and is_supported_image(f))


class GroupDataset(Dataset):
    """Dataset for multi-class group training/validation.

    Reads from ``group_folder/{class_name}/{split}/`` directories.
    For training: class-balanced sampling — each epoch draws
    ``max_class_count`` images per class, minority classes repeated.
    """

    def __init__(
        self,
        group_folder: str | Path,
        class_names: list[str],
        split: str = "train",
        *,
        transform: Any | None = None,
        multi_label: bool = False,
        face_bboxes: dict[str, list[int]] | None = None,
    ):
        self.group_folder = Path(group_folder)
        self.class_names = class_names
        self.split = split
        self.transform = transform
        self.multi_label = multi_label
        self._face_bboxes: dict[str, list[int]] = face_bboxes or {}

        # Gather images per class (index = class_index)
        self._class_paths: list[list[Path]] = []
        for name in class_names:
            self._class_paths.append(_list_class_images(self.group_folder, name, split))

        # Disk cache for resized images
        self._cache_dir = self.group_folder / ".resize_cache"

        self.samples: list[dict] = []
        self._build_samples()

    def _build_samples(self) -> None:
        self.samples = []

        if self.multi_label:
            self._build_multilabel_samples()
            return

        if self.split == "val":
            for class_idx, paths in enumerate(self._class_paths):
                for p in paths:
                    w, h = self._get_image_size(p)
                    bucket = find_nearest_bucket(w, h)
                    self.samples.append({
                        "path": str(p),
                        "label": class_idx,
                        "bucket": bucket,
                    })
        else:
            max_count = max((len(p) for p in self._class_paths if len(p) > 0), default=0)
            if max_count == 0:
                return

            for class_idx, paths in enumerate(self._class_paths):
                if not paths:
                    continue
                if len(paths) < max_count:
                    expanded = paths * (max_count // len(paths) + 1)
                    random.shuffle(expanded)
                    expanded = expanded[:max_count]
                else:
                    expanded = list(paths)
                    random.shuffle(expanded)

                for p in expanded:
                    w, h = self._get_image_size(p)
                    bucket = find_nearest_bucket(w, h)
                    self.samples.append({
                        "path": str(p),
                        "label": class_idx,
                        "bucket": bucket,
                    })

        random.shuffle(self.samples)

    def _build_multilabel_samples(self) -> None:
        num_classes = len(self.class_names)
        # stem -> {path, class_indices}
        image_map: dict[str, dict] = {}

        for class_idx, paths in enumerate(self._class_paths):
            for p in paths:
                stem = p.stem
                if stem not in image_map:
                    image_map[stem] = {"path": p, "class_indices": set()}
                image_map[stem]["class_indices"].add(class_idx)

        entries = list(image_map.values())

        if self.split == "train" and entries:
            random.shuffle(entries)

        for entry in entries:
            p = entry["path"]
            label = torch.zeros(num_classes, dtype=torch.float32)
            for ci in entry["class_indices"]:
                label[ci] = 1.0

            w, h = self._get_image_size(p)
            bucket = find_nearest_bucket(w, h)
            self.samples.append({
                "path": str(p),
                "label": label,
                "bucket": bucket,
            })

    def reshuffle(self) -> None:
        """Rebuild with fresh class-balanced sampling (call once per epoch)."""
        if self.split == "train":
            self._build_samples()

    @staticmethod
    def _get_image_size(path: Path | str) -> tuple[int, int]:
        with Image.open(path) as img:
            return img.size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int | torch.Tensor, tuple[int, int]]:
        sample = self.samples[idx]
        bucket = sample["bucket"]
        face_bbox = self._face_bboxes.get(sample["path"])
        img = load_or_resize(sample["path"], bucket, self._cache_dir, face_bbox=face_bbox)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = TF.to_tensor(img)

        return img, sample["label"], bucket

    def get_class_counts(self) -> dict[int, int]:
        """Return raw image count per class (before balancing)."""
        return {i: len(paths) for i, paths in enumerate(self._class_paths)}


class GroupBucketBatchSampler(Sampler):
    """Bucket-aware batch sampler for GroupDataset."""

    def __init__(self, dataset: GroupDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        bucket_indices: dict[tuple[int, int], list[int]] = {}
        for i, sample in enumerate(self.dataset.samples):
            bucket = sample["bucket"]
            bucket_indices.setdefault(bucket, []).append(i)

        batches = []
        for indices in bucket_indices.values():
            random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batches.append(indices[start:start + self.batch_size])

        random.shuffle(batches)
        yield from batches

    def __len__(self):
        bucket_counts: dict[tuple[int, int], int] = {}
        for sample in self.dataset.samples:
            b = sample["bucket"]
            bucket_counts[b] = bucket_counts.get(b, 0) + 1
        return sum(math.ceil(c / self.batch_size) for c in bucket_counts.values())


def build_group_bucket_sampler(
    dataset: GroupDataset, batch_size: int
) -> GroupBucketBatchSampler:
    return GroupBucketBatchSampler(dataset, batch_size)
