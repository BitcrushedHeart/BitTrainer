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
    DEFAULT_TRAIN_RESOLUTION,
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


def compute_class_log_priors(counts: dict[int, int], num_classes: int) -> dict[str, float]:
    """Laplace-smoothed log-prior vector keyed by ``str(class_index)``.

    ``+1`` smoothing keeps empty classes finite (``log(1/total)`` rather than
    ``log(0)``), so a class with no samples never poisons the decode-time
    ``log(natural) - log(effective)`` adjustment. Values are natural logs of the
    normalised (smoothed) class probabilities (ISSUE-0490 A).
    """
    smoothed = [float(counts.get(i, 0) or 0) + 1.0 for i in range(num_classes)]
    total = sum(smoothed)
    return {str(i): math.log(smoothed[i] / total) for i in range(num_classes)}


def _list_class_images(group_folder: Path, class_name: str, split: str) -> list[Path]:
    d = group_folder / class_name / split
    if not d.is_dir():
        return []
    return sorted(f for f in d.iterdir() if f.is_file() and is_supported_image(f))


class GroupDataset(Dataset):
    """Dataset for multi-class group training/validation.

    ``samples`` is a STABLE base list — one row per unique (path, class), built
    once at construction and never mutated afterwards. Class balancing (the
    ``_MAX_OVERSAMPLE_FACTOR`` cap, natural sampling, the rare-group ``__none__``
    top-up) and per-epoch shuffling are expressed as an INDEX SCHEDULE over that
    list (:meth:`epoch_indices`, redrawn by :meth:`reshuffle`). Two things depend
    on that split (Bitcrush ISSUE-0859/0860): image sizes are read exactly once
    per instance, and persistent dataloader workers — which hold a copy of the
    dataset pickled at first iteration — keep indexing the same rows the parent's
    schedule refers to, so the loaders can be built once and reused for the run.

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
        train_resolution: int = DEFAULT_TRAIN_RESOLUTION,
        forensic: Any | None = None,        # ForensicAugment instance
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
        # Per-group training resolution (default 512 = the canonical bucket
        # table). Sourceless mode ignores it: samples come from the cache
        # index at whatever resolution they were cached, and there is no
        # source image to rebuild from.
        self._train_resolution = int(train_resolution or DEFAULT_TRAIN_RESOLUTION)
        if sourceless and self._train_resolution != DEFAULT_TRAIN_RESOLUTION:
            logger.warning(
                "GroupDataset(sourceless): train_resolution=%d ignored — cached buckets rule",
                self._train_resolution,
            )
        # When True, train samples are taken at their natural class distribution
        # (each image once) instead of replication-equalised to the largest
        # class — used by the "reweight" class-balance mode, where imbalance is
        # handled by class weights in the loss instead of by oversampling.
        self._natural_sampling = natural_sampling
        # Skin Tone V2 dual-view (skin_tone_views.py): a per-image colour
        # normalisation applied as a stochastic train augmentation, or forced
        # for the validation "normalized" pass. Attached post-init by
        # _prepare_datasets_and_cache; None = feature off.
        self.skin_tone_views = None
        self.skin_tone_view_prob = 0.0
        self.skin_tone_force_view = False
        # Forensic augmentation (blur -> noise -> JPEG) runs HERE, in the
        # DataLoader worker, not on the trainer's main thread (ISSUE-0861).
        # Train split only: val must stay clean, and the SmartCache is filled
        # by cache_builders.build_image_tensor via SmartCache.prepare — never
        # through __getitem__ — so cached tensors stay unaugmented.
        self._forensic = (
            forensic
            if (forensic is not None and split == "train" and forensic.is_active)
            else None
        )

        # ISSUE-0859: (w, h) per unique path, read ONCE. ``reshuffle()`` used to
        # rebuild a LOCAL size cache by re-opening every image (~13.6k PIL opens
        # per epoch on the main thread); sizes never change between epochs.
        self._size_memo: dict[str, tuple[int, int]] = {}
        # ``samples`` is the STABLE base list: each unique (path, class) once,
        # built at construction and never mutated afterwards. Per-epoch
        # replication (oversample cap / rare-group __none__) and shuffling live
        # in ``_epoch_indices``, an index schedule OVER that list. Persistent
        # dataloader workers hold a pickled copy of the dataset taken at first
        # iteration, so mutating ``samples`` per epoch would silently
        # desynchronise the parent's schedule from the workers' rows.
        self.samples: list[dict] = []
        self._class_base_indices: list[list[int]] = []
        self._epoch_indices: list[int] = []

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

        self._build_samples()

    def _init_sourceless(self) -> None:
        if self._cache is None:
            raise RuntimeError("sourceless=True requires a SmartCache instance")
        entries = self._cache.iter_sourceless()
        # Base = the exact (equalised) list baked into the cache, kept so the
        # auto-oversample sweep can re-derive __none__ oversampling on the pod.
        self._sourceless_base = [s for s in entries if s.get("split") == self.split]
        self.samples = list(self._sourceless_base)
        self._class_base_indices = [[] for _ in self.class_names]
        self._build_epoch_indices()
        self._class_paths = [[] for _ in self.class_names]

    def _sourceless_none_extra_indices(self) -> list[int]:
        """Rare-group ``__none__`` oversample for the sourceless (cloud-pod)
        path: the EXTRA base indices that lift cached ``__none__`` rows to the
        same target :meth:`_rare_group_extra_indices` uses. Empty for
        multi-label or when there is no ``__none__`` class/data."""
        try:
            none_idx = self.class_names.index(_NONE_CLASS_NAME)
        except ValueError:
            return []
        counts: dict[int, int] = {}
        none_indices: list[int] = []
        for i, s in enumerate(self.samples):
            lbl = s.get("label")
            if not isinstance(lbl, int):
                return []  # multi-label targets: rare-group oversample doesn't apply
            counts[lbl] = counts.get(lbl, 0) + 1
            if lbl == none_idx:
                none_indices.append(i)
        none_count = counts.get(none_idx, 0)
        if none_count == 0:
            return []
        max_count = max(counts.values())
        non_none_class_count = sum(1 for k, v in counts.items() if k != none_idx and v)
        if non_none_class_count == 0:
            return []
        target = rare_group_none_target(max_count, non_none_class_count)
        extra_needed = target - none_count
        if extra_needed <= 0:
            return []
        pool = none_indices * (extra_needed // none_count + 1)
        random.shuffle(pool)
        logger.info(
            "Rare-group oversample (sourceless): __none__ %d → %d",
            none_count, target,
        )
        return pool[:extra_needed]

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
        per-class path-list level, so every later epoch schedule is drawn from
        the filtered base list. Returns the number of images dropped.
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
        """Full rebuild: the stable base list, then this epoch's schedule."""
        self._build_base_samples()
        self._build_epoch_indices()

    # -- base list (built once; never mutated per epoch) --------------------

    def _build_base_samples(self) -> None:
        self.samples = []
        self._class_base_indices = [[] for _ in self.class_names]

        if self.multi_label:
            self._build_multilabel_samples()
            return

        bad_paths: set[str] = set()
        all_unique = {str(p) for paths in self._class_paths for p in paths}
        for ps in all_unique:
            if self._size_for(ps) is None:
                bad_paths.add(ps)
        if bad_paths:
            logger.warning("Skipping %d unreadable images", len(bad_paths))

        for class_idx, paths in enumerate(self._class_paths):
            for p in paths:
                sp = str(p)
                if sp in bad_paths:
                    continue
                bucket = find_nearest_bucket(*self._size_memo[sp], self._train_resolution)
                self._class_base_indices[class_idx].append(len(self.samples))
                self.samples.append(self._make_sample(sp, class_idx, bucket))

    # -- epoch schedule (redrawn every reshuffle) ---------------------------

    def _build_epoch_indices(self) -> None:
        """Draw this epoch's index schedule over the stable base list.

        Single-label train: per-class replication under ``_MAX_OVERSAMPLE_FACTOR``
        (or natural sampling), the rare-group ``__none__`` top-up, then a global
        shuffle — the exact composition ``_build_samples`` used to bake into
        ``self.samples``, now expressed as base indices with repeats.
        """
        if self._sourceless:
            indices = list(range(len(self.samples)))
            if self.split == "train" and self._oversample_none:
                extra = self._sourceless_none_extra_indices()
                if extra:
                    indices.extend(extra)
                    random.shuffle(indices)
            self._epoch_indices = indices
            return

        if self.multi_label:
            indices = list(range(len(self.samples)))
            if self.split == "train" and indices:
                random.shuffle(indices)
            self._epoch_indices = indices
            return

        if self.split == "val":
            indices = list(range(len(self.samples)))
            random.shuffle(indices)
            self._epoch_indices = indices
            return

        max_count = max(
            (len(b) for b in self._class_base_indices if len(b) > 0), default=0
        )
        if max_count == 0:
            self._epoch_indices = []
            return

        indices = []
        for base in self._class_base_indices:
            if not base:
                continue
            n = len(base)
            if self._natural_sampling:
                # Natural distribution: every image once, no equalisation.
                expanded = list(base)
                random.shuffle(expanded)
            elif n < max_count:
                # Cap replication at _MAX_OVERSAMPLE_FACTOR x the class's natural
                # size (still <= max_count) to bound memorisation of sparse classes.
                target = min(max_count, math.ceil(_MAX_OVERSAMPLE_FACTOR * n))
                expanded = base * (target // n + 1)
                random.shuffle(expanded)
                expanded = expanded[:target]
            else:
                expanded = list(base)
                random.shuffle(expanded)
            indices.extend(expanded)

        if self._oversample_none:
            indices.extend(self._rare_group_extra_indices(max_count))

        random.shuffle(indices)
        self._epoch_indices = indices

    def _rare_group_extra_indices(self, max_count: int) -> list[int]:
        """EXTRA ``__none__`` base indices so the rare-group target dominates.

        Target count for ``__none__`` is ``ceil(1.5 * sum_of_non_none_counts)``
        where each non-empty non-``__none__`` class contributes ``max_count``
        after the baseline equalisation. The ``max_count`` __none__ entries the
        baseline pass already added stay in place; only the *extra* needed to
        reach the target are returned here.
        """
        try:
            none_idx = self.class_names.index(_NONE_CLASS_NAME)
        except ValueError:
            return []
        if none_idx >= len(self._class_base_indices):
            return []
        none_base = self._class_base_indices[none_idx]
        if not none_base:
            return []

        non_none_class_count = sum(
            1 for i, b in enumerate(self._class_base_indices) if i != none_idx and b
        )
        if non_none_class_count == 0:
            return []
        target = rare_group_none_target(max_count, non_none_class_count)
        non_none_total = max_count * non_none_class_count
        extra_needed = target - max_count
        if extra_needed <= 0:
            return []

        extra = list(none_base) * (extra_needed // len(none_base) + 1)
        random.shuffle(extra)
        logger.info(
            "Rare-group oversample: __none__ %d → %d (non-none total %d)",
            max_count, target, non_none_total,
        )
        return extra[:extra_needed]

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

        try:
            none_idx = self.class_names.index(_NONE_CLASS_NAME)
        except ValueError:
            none_idx = -1

        for entry in entries:
            p = entry["path"]
            size = self._size_for(p)
            if size is None:
                continue

            class_indices = set(entry["class_indices"])
            if none_idx >= 0 and none_idx in class_indices and len(class_indices) > 1:
                class_indices.discard(none_idx)

            label = torch.zeros(num_classes, dtype=torch.float32)
            for ci in class_indices:
                label[ci] = 1.0

            bucket = find_nearest_bucket(*size, self._train_resolution)
            self.samples.append(self._make_sample(str(p), label, bucket))

    def reshuffle(self) -> None:
        """Redraw the epoch schedule. The base list (and every image size) is
        untouched, so this never re-opens a file (ISSUE-0859) and the pickled
        dataset copies inside persistent dataloader workers stay valid."""
        if self.split == "train" and not self._sourceless:
            self._build_epoch_indices()

    def epoch_indices(self) -> list[int]:
        """This epoch's base-list indices, WITH replication repeats, in order."""
        return self._epoch_indices

    def epoch_samples(self) -> list[dict]:
        """The epoch schedule expanded to sample dicts (replicated rows repeat).

        This is what the model actually sees in one epoch — the shape the
        pre-``samples``-stabilisation code baked into ``self.samples``.
        """
        return [self.samples[i] for i in self._epoch_indices]

    def set_natural_sampling(self, flag: bool) -> None:
        """Switch between natural-distribution and replication-equalised train
        sampling, redrawing the epoch schedule. No-op for val/sourceless."""
        if self._natural_sampling == flag:
            return
        self._natural_sampling = flag
        if self.split == "train" and not self._sourceless:
            self._build_epoch_indices()

    @property
    def oversample_none(self) -> bool:
        return self._oversample_none

    def set_oversample_none(self, flag: bool) -> None:
        """Toggle rare-group ``__none__`` oversampling, redrawing the epoch
        schedule. No-op for val or when unchanged.

        Mirrors :meth:`set_natural_sampling` so the auto-oversample sweep can
        apply its selection to the full fine-tune dataset after the warmup
        probe has decided off-vs-1.5x. Works for sourceless (cloud-pod) datasets
        too, re-deriving from the cached base list."""
        if self._oversample_none == flag:
            return
        self._oversample_none = flag
        if self.split != "train":
            return
        self._build_epoch_indices()

    def _size_for(self, path: Path | str) -> tuple[int, int] | None:
        """Memoised ``(w, h)``: each unique path is opened at most once per
        dataset instance, so schedule redraws and bbox-driven rebuilds are free."""
        key = str(path)
        if key in self._size_memo:
            return self._size_memo[key]
        size = self._get_image_size(key)
        if size is not None:
            self._size_memo[key] = size
        return size

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
                    tensor = self._maybe_skin_tone_view(tensor, sample)
                    tensor = self._maybe_forensic(tensor)
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

        img_tensor = self._maybe_skin_tone_view(img_tensor, sample)
        img_tensor = self._maybe_forensic(img_tensor)
        return img_tensor, sample["label"], tuple(bucket)

    def _maybe_forensic(self, tensor: torch.Tensor) -> torch.Tensor:
        """Worker-side forensic chain — identity unless an active
        :class:`~bittrainer.forensic_augment.ForensicAugment` is attached."""
        if self._forensic is None:
            return tensor
        return self._forensic(tensor)

    def _maybe_skin_tone_view(self, tensor: torch.Tensor, sample: dict) -> torch.Tensor:
        """Skin Tone V2 dual-view hook — identity unless a view bank is
        attached (see skin_tone_views.maybe_apply_view)."""
        if self.skin_tone_views is None:
            return tensor
        from bittrainer.skin_tone_views import maybe_apply_view

        return maybe_apply_view(
            tensor,
            str(sample.get("source_path") or sample["path"]),
            self.skin_tone_views,
            probability=self.skin_tone_view_prob,
            force=self.skin_tone_force_view,
        )

    def get_class_counts(self) -> dict[int, int]:
        if self._sourceless:
            counts: dict[int, int] = {}
            for s in self.samples:
                lbl = s["label"]
                if isinstance(lbl, int):
                    counts[lbl] = counts.get(lbl, 0) + 1
            return counts
        return {i: len(paths) for i, paths in enumerate(self._class_paths)}

    def get_effective_class_counts(self) -> dict[int, int]:
        """Per-class sample counts AFTER oversample expansion — the model's
        actual per-epoch class exposure.

        Counts the built ``samples`` list (single-label only; multi-label labels
        are tensors and have no single class index). This is the numerator of the
        effective train prior that inference-time prior correction divides out
        (ISSUE-0490 A)."""
        counts: dict[int, int] = {}
        for i in self._epoch_indices:
            lbl = self.samples[i]["label"]
            if isinstance(lbl, int):
                counts[lbl] = counts.get(lbl, 0) + 1
        return counts


class GroupBucketBatchSampler(Sampler):
    """Buckets the dataset's CURRENT epoch schedule into same-shape batches.

    The schedule is a multiset of base-list indices (replicated rows appear more
    than once), re-read on every ``__iter__`` so a ``reshuffle()`` between epochs
    is picked up without rebuilding the sampler or the DataLoader around it.
    Datasets without an ``epoch_indices()`` (the binary/stub datasets) fall back
    to one pass over ``samples``.
    """

    def __init__(self, dataset: GroupDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def _indices(self) -> list[int]:
        fn = getattr(self.dataset, "epoch_indices", None)
        if callable(fn):
            return list(fn())
        return list(range(len(self.dataset.samples)))

    def __iter__(self):
        samples = self.dataset.samples
        bucket_indices: dict[tuple[int, int], list[int]] = {}
        for i in self._indices():
            bucket = samples[i]["bucket"]
            bucket_indices.setdefault(bucket, []).append(i)

        batches = []
        for indices in bucket_indices.values():
            random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batches.append(indices[start:start + self.batch_size])

        random.shuffle(batches)
        yield from batches

    def __len__(self):
        samples = self.dataset.samples
        bucket_counts: dict[tuple[int, int], int] = {}
        for i in self._indices():
            b = samples[i]["bucket"]
            bucket_counts[b] = bucket_counts.get(b, 0) + 1
        return sum(math.ceil(c / self.batch_size) for c in bucket_counts.values())


class EpochIndexSampler(Sampler):
    """Plain (non-bucketed) sampler over the dataset's CURRENT epoch schedule.

    For loaders that used ``DataLoader(dataset, shuffle=True)``: since
    ISSUE-0859 ``samples`` is the de-replicated base list, so ``shuffle=True``
    (which walks ``range(len(dataset))``) would drop the oversample cap / rare-
    group ``__none__`` replication. This walks ``epoch_indices()`` (replication
    included), re-read and re-shuffled on every ``__iter__`` so a loader built
    once per run still sees a fresh order each epoch.
    """

    def __init__(self, dataset: GroupDataset):
        self.dataset = dataset

    def _indices(self) -> list[int]:
        fn = getattr(self.dataset, "epoch_indices", None)
        if callable(fn):
            return list(fn())
        return list(range(len(self.dataset.samples)))

    def __iter__(self):
        indices = self._indices()
        random.shuffle(indices)
        yield from indices

    def __len__(self):
        return len(self._indices())


def build_group_bucket_sampler(
    dataset: GroupDataset, batch_size: int
) -> GroupBucketBatchSampler:
    return GroupBucketBatchSampler(dataset, batch_size)
