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


_NONE_CLASS_NAME = "__none__"

# Rare-group ``__none__`` oversample magnitude: the target ``__none__`` count is
# this factor times the (equalised) non-``__none__`` total. Shared with the
# auto-oversample sweep in group_trainer so the probe and the full fine-tune
# agree on what "1.5x" means.
_RARE_GROUP_OVERSAMPLE_FACTOR = 1.5

# Hardcoded global ceiling on minority-class replication during balancing: a class
# is oversampled to at most this multiple of its natural size (still never beyond
# the largest class). Bounds the memorisation that uncapped equalisation to the
# largest class caused on sparse classes. Applies to the baseline equalisation only;
# __none__ rare-group oversampling (above) is layered on top as before.
_MAX_OVERSAMPLE_FACTOR = 4.0


def rare_group_none_target(max_count: int, non_none_class_count: int) -> int:
    """Target ``__none__`` sample count for rare-group oversampling.

    ``ceil(FACTOR * sum_of_non_none_counts)`` where each non-empty
    non-``__none__`` class contributes ``max_count`` after baseline
    equalisation, floored at ``max_count`` so it never *reduces* the count.
    """
    non_none_total = max_count * non_none_class_count
    return max(max_count, math.ceil(_RARE_GROUP_OVERSAMPLE_FACTOR * non_none_total))


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
        oversample_none: bool = False,
        extra_paths: dict[str, list[str]] | None = None,
        natural_sampling: bool = False,
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
        self._oversample_none = oversample_none
        # When True, train samples are taken at their natural class distribution
        # (each image once) instead of replication-equalised to the largest
        # class — used by the "reweight" class-balance mode, where imbalance is
        # handled by class weights in the loss instead of by oversampling.
        self._natural_sampling = natural_sampling

        if sourceless:
            self._init_sourceless()
            return

        # Off-disk paths supplied by the caller (e.g. __none__ samples that
        # the labelling pipeline didn't copy into the group folder). Spliced
        # into the per-class path lists alongside whatever the disk scan
        # finds; deduplicated by absolute string path.
        extra = extra_paths or {}
        self._class_paths: list[list[Path]] = []
        for name in class_names:
            disk_paths = _list_class_images(self.group_folder, name, split)
            extras = extra.get(name, [])
            if extras:
                seen = {str(p) for p in disk_paths}
                for raw in extras:
                    p = Path(raw)
                    if str(p) in seen:
                        continue
                    if not is_supported_image(p):
                        continue
                    if not p.is_file():
                        continue
                    disk_paths.append(p)
                    seen.add(str(p))
            self._class_paths.append(disk_paths)

        self._cache_dir = self.group_folder / ".resize_cache"

        self.samples: list[dict] = []
        self._build_samples()

    def _init_sourceless(self) -> None:
        if self._cache is None:
            raise RuntimeError("sourceless=True requires a SmartCache instance")
        entries = self._cache.iter_sourceless()
        # Base = the exact (equalised) list baked into the cache, kept so the
        # auto-oversample sweep can re-derive __none__ oversampling on the pod.
        self._sourceless_base = [s for s in entries if s.get("split") == self.split]
        self.samples = list(self._sourceless_base)
        if self.split == "train" and self._oversample_none:
            self._apply_sourceless_none_oversample()
        self._class_paths = [[] for _ in self.class_names]

    def _apply_sourceless_none_oversample(self) -> None:
        """Rare-group ``__none__`` oversample for the sourceless (cloud-pod)
        path: duplicates cached ``__none__`` sample dicts up to the same target
        used by :meth:`_apply_rare_group_oversample`. No-op for multi-label or
        when there is no ``__none__`` class/data."""
        try:
            none_idx = self.class_names.index(_NONE_CLASS_NAME)
        except ValueError:
            return
        counts: dict[int, int] = {}
        none_samples: list[dict] = []
        for s in self._sourceless_base:
            lbl = s.get("label")
            if not isinstance(lbl, int):
                return  # multi-label targets: rare-group oversample doesn't apply
            counts[lbl] = counts.get(lbl, 0) + 1
            if lbl == none_idx:
                none_samples.append(s)
        none_count = counts.get(none_idx, 0)
        if none_count == 0:
            return
        max_count = max(counts.values())
        non_none_class_count = sum(1 for k, v in counts.items() if k != none_idx and v)
        if non_none_class_count == 0:
            return
        target = rare_group_none_target(max_count, non_none_class_count)
        extra_needed = target - none_count
        if extra_needed <= 0:
            return
        pool = none_samples * (extra_needed // none_count + 1)
        random.shuffle(pool)
        self.samples.extend(pool[:extra_needed])
        random.shuffle(self.samples)
        logger.info(
            "Rare-group oversample (sourceless): __none__ %d → %d",
            none_count, target,
        )

    def set_cache(self, cache: Any) -> None:
        self._cache = cache

    def refresh_face_bboxes(self, face_bboxes: dict[str, list[int]]) -> None:
        self._face_bboxes = face_bboxes
        for s in self.samples:
            s["face_bbox"] = face_bboxes.get(s["path"])

    def drop_paths_without_bbox(self, bboxes: dict[str, list[int]]) -> int:
        """Remove images with no detected crop region and rebuild samples.

        Region-crop training with ``region_fallback="drop"``: a train image
        where the detector found nothing would otherwise fall back to a
        centre crop of mostly-irrelevant pixels. Filtering happens at the
        per-class path-list level so the exclusion survives ``reshuffle()``'s
        sample rebuild. Returns the number of images dropped.
        """
        if self._sourceless:
            return 0
        dropped = 0
        kept_lists: list[list[Path]] = []
        for paths in self._class_paths:
            kept = [p for p in paths if bboxes.get(str(p))]
            dropped += len(paths) - len(kept)
            kept_lists.append(kept)
        if dropped:
            self._class_paths = kept_lists
            self._build_samples()
        return dropped

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
                if self._natural_sampling:
                    # Natural distribution: every image once, no equalisation.
                    expanded = list(paths)
                    random.shuffle(expanded)
                elif len(paths) < max_count:
                    # Cap replication at _MAX_OVERSAMPLE_FACTOR x the class's natural
                    # size (still <= max_count) to bound memorisation of sparse classes.
                    target = min(max_count, math.ceil(_MAX_OVERSAMPLE_FACTOR * len(paths)))
                    expanded = paths * (target // len(paths) + 1)
                    random.shuffle(expanded)
                    expanded = expanded[:target]
                else:
                    expanded = list(paths)
                    random.shuffle(expanded)

                for p in expanded:
                    sp = str(p)
                    bucket = find_nearest_bucket(*size_cache[sp])
                    self.samples.append(self._make_sample(sp, class_idx, bucket))

            if self._oversample_none:
                self._apply_rare_group_oversample(clean_class_paths, size_cache, max_count)

        random.shuffle(self.samples)

    def _apply_rare_group_oversample(
        self,
        clean_class_paths: list[list[Path]],
        size_cache: dict[str, tuple[int, int]],
        max_count: int,
    ) -> None:
        """Append extra ``__none__`` samples so the rare-group target dominates.

        Target count for ``__none__`` is ``ceil(1.5 * sum_of_non_none_counts)``
        where each non-empty non-``__none__`` class contributes ``max_count``
        after the baseline equalisation. Already-added ``max_count`` __none__
        samples stay in place; only the *extra* needed to reach the target
        are appended here.
        """
        try:
            none_idx = self.class_names.index(_NONE_CLASS_NAME)
        except ValueError:
            return
        if none_idx >= len(clean_class_paths):
            return
        none_paths = clean_class_paths[none_idx]
        if not none_paths:
            return

        non_none_class_count = sum(
            1 for i, p in enumerate(clean_class_paths) if i != none_idx and p
        )
        if non_none_class_count == 0:
            return
        target = rare_group_none_target(max_count, non_none_class_count)
        non_none_total = max_count * non_none_class_count
        extra_needed = target - max_count
        if extra_needed <= 0:
            return

        extra = list(none_paths) * (extra_needed // len(none_paths) + 1)
        random.shuffle(extra)
        extra = extra[:extra_needed]
        for p in extra:
            sp = str(p)
            bucket = find_nearest_bucket(*size_cache[sp])
            self.samples.append(self._make_sample(sp, none_idx, bucket))
        logger.info(
            "Rare-group oversample: __none__ %d → %d (non-none total %d)",
            max_count, target, non_none_total,
        )

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

        try:
            none_idx = self.class_names.index(_NONE_CLASS_NAME)
        except ValueError:
            none_idx = -1

        for entry in entries:
            p = entry["path"]
            size = self._get_image_size(p)
            if size is None:
                continue

            class_indices = set(entry["class_indices"])
            if none_idx >= 0 and none_idx in class_indices and len(class_indices) > 1:
                class_indices.discard(none_idx)

            label = torch.zeros(num_classes, dtype=torch.float32)
            for ci in class_indices:
                label[ci] = 1.0

            bucket = find_nearest_bucket(*size)
            self.samples.append(self._make_sample(str(p), label, bucket))

    def reshuffle(self) -> None:
        if self.split == "train" and not self._sourceless:
            self._build_samples()

    def set_natural_sampling(self, flag: bool) -> None:
        """Switch between natural-distribution and replication-equalised train
        sampling, rebuilding the sample list. No-op for val/sourceless."""
        if self._natural_sampling == flag:
            return
        self._natural_sampling = flag
        if self.split == "train" and not self._sourceless:
            self._build_samples()

    @property
    def oversample_none(self) -> bool:
        return self._oversample_none

    def set_oversample_none(self, flag: bool) -> None:
        """Toggle rare-group ``__none__`` oversampling, rebuilding the sample
        list. No-op for val/sourceless or when unchanged.

        Mirrors :meth:`set_natural_sampling` so the auto-oversample sweep can
        apply its selection to the full fine-tune dataset after the warmup
        probe has decided off-vs-1.5x. Works for sourceless (cloud-pod) datasets
        too, re-deriving from the cached base list."""
        if self._oversample_none == flag:
            return
        self._oversample_none = flag
        if self.split != "train":
            return
        if self._sourceless:
            self.samples = list(self._sourceless_base)
            if flag:
                self._apply_sourceless_none_oversample()
        else:
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
        bw, bh = int(bucket[0]), int(bucket[1])

        if self._cache is not None:
            result = self._cache.get(sample["path"])
            if result is not None:
                tensor, _ = result
                if tuple(tensor.shape[-2:]) == (bh, bw):
                    return tensor, sample["label"], tuple(bucket)
                # Cached tensor was built under a different aspect-ratio bucket
                # table (e.g. a prior training resolution). Its size no longer
                # matches this sample's bucket, so it would explode the bucket
                # collate. Rebuild from source rather than mixing sizes.
                if self._sourceless:
                    raise RuntimeError(
                        f"Sourceless training: cached tensor for '{sample['path']}' "
                        f"is {tuple(tensor.shape[-2:])}, expected {(bh, bw)}. The "
                        f"cache predates the current bucket table — rebuild it."
                    )
            elif self._sourceless:
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
