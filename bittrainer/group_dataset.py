"""Multi-class dataset with aspect ratio bucketing and class-balanced sampling."""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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
from bittrainer.image_utils import is_supported_image


def _list_class_images(group_folder: Path, class_name: str, split: str) -> list[Path]:
    d = group_folder / class_name / split
    if not d.is_dir():
        return []
    return sorted(f for f in d.iterdir() if f.is_file() and is_supported_image(f))


class GroupDataset(Dataset):
    """Dataset for multi-class group training/validation.

    When a :class:`bittrainer.smart_cache.SmartCache` is attached, images are
    loaded as pre-resized CHW uint8 tensors directly from the cache. Cache
    misses fall back to on-the-fly PIL decode via the build function.
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
        skin_normalise: bool = False,
        cache: Any | None = None,           # SmartCache instance
        sourceless: bool = False,
        group_name: str = "",
    ):
        self.group_folder = Path(group_folder)
        self.class_names = class_names
        self.split = split
        self.transform = transform
        self.multi_label = multi_label
        self._face_bboxes: dict[str, list[int]] = face_bboxes or {}
        self._skin_normalise = skin_normalise
        self._cache = cache
        self._sourceless = sourceless
        self._group_name = group_name or self.group_folder.name

        if sourceless:
            self._init_sourceless()
            return

        self._class_paths: list[list[Path]] = []
        for name in class_names:
            self._class_paths.append(_list_class_images(self.group_folder, name, split))

        self._cache_dir = self.group_folder / ".resize_cache"

        self.samples: list[dict] = []
        self._build_samples()

    def _init_sourceless(self) -> None:
        if self._cache is None:
            raise RuntimeError("sourceless=True requires a SmartCache instance")
        entries = self._cache.iter_sourceless()
        self.samples = [s for s in entries if s.get("split") == self.split]
        counts: dict[int, int] = {}
        for s in self.samples:
            counts[s["label"]] = counts.get(s["label"], 0) + 1
        self._class_paths = [[] for _ in self.class_names]

    def set_cache(self, cache: Any) -> None:
        self._cache = cache

    def refresh_face_bboxes(self, face_bboxes: dict[str, list[int]]) -> None:
        self._face_bboxes = face_bboxes
        for s in self.samples:
            s["face_bbox"] = face_bboxes.get(s["path"])

    def _build_samples(self) -> None:
        self.samples = []

        if self.multi_label:
            self._build_multilabel_samples()
            return

        size_cache: dict[str, tuple[int, int]] = {}
        bad_paths: set[str] = set()
        all_unique = {str(p) for paths in self._class_paths for p in paths}
        for ps in all_unique:
            size = self._get_image_size(ps)
            if size is None:
                bad_paths.add(ps)
            else:
                size_cache[ps] = size
        if bad_paths:
            logger.warning("Skipping %d unreadable images", len(bad_paths))

        if self.split == "val":
            for class_idx, paths in enumerate(self._class_paths):
                for p in paths:
                    sp = str(p)
                    if sp in bad_paths:
                        continue
                    bucket = find_nearest_bucket(*size_cache[sp])
                    self.samples.append(self._make_sample(sp, class_idx, bucket))
        else:
            clean_class_paths = [
                [p for p in paths if str(p) not in bad_paths]
                for paths in self._class_paths
            ]
            max_count = max((len(p) for p in clean_class_paths if len(p) > 0), default=0)
            if max_count == 0:
                return

            for class_idx, paths in enumerate(clean_class_paths):
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
                    sp = str(p)
                    bucket = find_nearest_bucket(*size_cache[sp])
                    self.samples.append(self._make_sample(sp, class_idx, bucket))

        random.shuffle(self.samples)

    def _make_sample(self, path: str, label: int | torch.Tensor, bucket: tuple[int, int]) -> dict:
        return {
            "path": path,
            "label": label,
            "bucket": bucket,
            "concept_name": self._group_name + (f"/{self.class_names[label]}" if isinstance(label, int) and 0 <= label < len(self.class_names) else ""),
            "split": self.split,
            "skin_normalise": self._skin_normalise,
            "face_bbox": self._face_bboxes.get(path),
        }

    def _build_multilabel_samples(self) -> None:
        num_classes = len(self.class_names)
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
            size = self._get_image_size(p)
            if size is None:
                continue

            label = torch.zeros(num_classes, dtype=torch.float32)
            for ci in entry["class_indices"]:
                label[ci] = 1.0

            bucket = find_nearest_bucket(*size)
            self.samples.append(self._make_sample(str(p), label, bucket))

    def reshuffle(self) -> None:
        if self.split == "train" and not self._sourceless:
            self._build_samples()

    @staticmethod
    def _get_image_size(path: Path | str) -> tuple[int, int] | None:
        try:
            with Image.open(path) as img:
                return img.size
        except (OSError, SyntaxError):
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int | torch.Tensor, tuple[int, int]]:
        sample = self.samples[idx]
        bucket = sample["bucket"]

        if self._cache is not None:
            result = self._cache.get(sample["path"])
            if result is not None:
                tensor, _ = result
                return tensor, sample["label"], tuple(bucket)
            if self._sourceless:
                raise RuntimeError(
                    f"Sourceless training: cache miss for '{sample['path']}'."
                )

        from bittrainer.cache_builders import build_image_tensor
        import numpy as np
        arr = build_image_tensor(sample)
        img_tensor = torch.from_numpy(np.ascontiguousarray(arr))

        if self.transform is not None:
            pil_img = Image.fromarray(arr.transpose(1, 2, 0))
            return self.transform(pil_img), sample["label"], tuple(bucket)

        return img_tensor, sample["label"], tuple(bucket)

    def get_class_counts(self) -> dict[int, int]:
        if self._sourceless:
            counts: dict[int, int] = {}
            for s in self.samples:
                lbl = s["label"]
                if isinstance(lbl, int):
                    counts[lbl] = counts.get(lbl, 0) + 1
            return counts
        return {i: len(paths) for i, paths in enumerate(self._class_paths)}


class GroupBucketBatchSampler(Sampler):
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
