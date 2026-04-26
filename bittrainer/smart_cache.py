"""Content-addressed tensor cache with cross-dataset dedup and crash-safe index.

Adapted from OneTrainer's MGDS SmartDiskCache. Key properties:

- **Content hashing** via xxhash64 — only genuinely changed files rebuild.
- **mtime fast path** — unchanged mtime skips hashing entirely.
- **Fast validation** — directory mtime + spot check skips per-file work when
  nothing has changed since last run (sub-second for large datasets).
- **Hash-index dedup** — the same image referenced by multiple concepts is
  cached once; a second reference increments ref_count.
- **Atomic index writes** — cache.json written via tmp+rename with .bak backup.
- **Interruptible** — stop_check callback honoured between file builds.
- **Progress callbacks** — per-stage step/total_steps/ETA/throughput emitted for
  WebSocket consumers, independently of optional tqdm console bars.

On-disk layout::

    {cache_dir}/cache.json              # index, hash_index, last_validated
    {cache_dir}/cache.json.bak
    {cache_dir}/{hash12}_{resolution}_{variation}.pt

Each ``.pt`` stores a dict with ``tensor`` (CHW uint8) plus ``__*`` metadata
fields (modeltype, resolution, bucket, label, split, source_path, source_hash,
skin_normalise, face_bbox, concept_name) enabling sourceless training.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import random
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

try:
    import xxhash
except ImportError as exc:
    raise RuntimeError(
        "SmartCache requires the 'xxhash' package. Install with: pip install xxhash"
    ) from exc

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

logger = logging.getLogger(__name__)

CACHE_VERSION = 2
_HASH_CHUNK = 65536
_FLUSH_INTERVAL_SECONDS = 30.0
_FLUSH_INTERVAL_ITEMS = 250
_PROGRESS_MIN_INTERVAL = 0.25


def _label_to_storage(label) -> Any:
    """Coerce a label into a JSON-serialisable form.

    Scalars (int, float, 0-d tensor) become python ints. Multi-hot tensors
    become lists so multilabel groups can round-trip through cache.json
    without crashing on ``int()``.
    """
    if isinstance(label, torch.Tensor):
        if label.ndim == 0:
            return int(label.item())
        return [float(x) for x in label.tolist()]
    return int(label)


def _label_from_storage(stored) -> Any:
    """Inverse of :func:`_label_to_storage`. List -> float tensor."""
    if isinstance(stored, list):
        return torch.tensor(stored, dtype=torch.float32)
    return int(stored)


def _bbox_sig(bbox) -> str:
    """Short, deterministic signature for a face bbox (or lack thereof)."""
    if not bbox:
        return "none"
    return ",".join(str(int(v)) for v in bbox)


def face_model_signature(face_model_path: str | None) -> str:
    """Signature that changes when the face model binary changes.

    Used to invalidate cache entries baked with a prior face crop model. The
    sample's face_bbox alone isn't sufficient — face detection is run ahead of
    caching, so a newer model may produce the same bboxes on a previously-seen
    image only by coincidence.
    """
    if not face_model_path:
        return "none"
    try:
        mtime = int(os.path.getmtime(face_model_path))
    except OSError:
        return f"{face_model_path}:missing"
    return f"{os.path.basename(face_model_path)}:{mtime}"


class CachingStoppedException(Exception):
    """Raised when stop_check returns True during a cache build."""


def _never_stop() -> bool:
    """Picklable default for stop_check. Module-level so the SmartCache can
    survive pickling when a dataset holding it is shipped to DataLoader
    workers on Windows spawn."""
    return False


def _noop_callback(_msg: dict) -> None:
    """Picklable no-op progress callback."""


class SmartCache:
    """Content-addressed tensor cache shared across datasets.

    Parameters
    ----------
    cache_dir :
        Root directory for the cache. Created if missing.
    modeltype :
        Opaque tag identifying the encoding format (e.g. ``"convnext_v2"``).
        Entries built with a different modeltype are rejected at validation
        time.
    progress_callback :
        Optional ``callback(dict)`` — receives stage/step/total_steps/eta/
        throughput events. Throttled to ~4/s.
    stop_check :
        Optional ``() -> bool`` — polled between per-item builds. Raises
        :class:`CachingStoppedException` when True.
    tqdm_enabled :
        Print OneTrainer-style tqdm bars to stdout. Independent of the
        structured callback.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        modeltype: str = "convnext_v2",
        progress_callback: Callable[[dict], None] | None = None,
        stop_check: Callable[[], bool] | None = None,
        tqdm_enabled: bool = True,
        face_model_sig: str = "none",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self._real_cache_dir = os.path.realpath(self.cache_dir)
        self.modeltype = modeltype
        self.face_model_sig = face_model_sig
        self._progress_cb = progress_callback or _noop_callback
        self._stop_check = stop_check or _never_stop
        self._tqdm_enabled = tqdm_enabled and _tqdm is not None

        self._index_lock = threading.Lock()
        self._last_flush_time = 0.0
        self._last_progress_time = 0.0
        self._cache_index: dict[str, Any] | None = None

    # DataLoader workers on Windows spawn pickle the dataset graph, and the
    # dataset holds a reference to this cache. threading.Lock and any live
    # mp.Event-wrapping callbacks aren't picklable, so strip them on pickle
    # and rebuild safe defaults on the worker side. Workers only call get()
    # (read-only), so they don't need the parent's lock or callbacks.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_index_lock"] = None
        state["_progress_cb"] = _noop_callback
        state["_stop_check"] = _never_stop
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._index_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(
        self,
        samples: list[dict],
        build_fn: Callable[[dict], np.ndarray],
        *,
        num_workers: int = 4,
        stage_label: str = "caching",
    ) -> dict[str, dict]:
        """Ensure every sample has a valid cache entry; build missing ones.

        Each ``sample`` must contain ``path``, ``bucket`` (``(w, h)``), ``label``
        (int), ``split`` (``"train"``/``"val"``), ``concept_name`` (str), and
        may carry ``face_bbox`` and ``skin_normalise``.

        ``build_fn(sample) -> np.ndarray`` must return a CHW uint8 array shaped
        ``(3, bucket_h, bucket_w)``.

        Returns a ``{source_path: entry_dict}`` map for all requested samples.
        """
        self._load_index()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Normalise sample paths (deduplicate by path in case the same image
        # appears in multiple samples — we still only cache it once).
        # Later occurrences of the same path override earlier ones, so the
        # val split wins over train when a file somehow sits in both.
        by_path: dict[str, dict] = {}
        for s in samples:
            p = os.path.normpath(str(s["path"]))
            by_path[p] = {**s, "path": p}

        stats = {"cache_hits": 0, "cache_misses": 0, "cache_dedup_hits": 0}

        # ------------------------------------------------------------------
        # Fast validation path — dir mtimes + spot check
        # ------------------------------------------------------------------
        if self._fast_validate(by_path.keys()):
            n = len(self._cache_index["entries"])
            logger.info("SmartCache: fast validation passed (%d entries)", n)
            self._emit_progress(
                stage="validating",
                step=n,
                total_steps=n,
                status_text=f"Fast-validated {n} cache entries",
                cache_hits=n,
                cache_misses=0,
                cache_dedup_hits=0,
                force=True,
            )
            # Return entries for requested samples
            return {p: self._cache_index["entries"][p] for p in by_path if p in self._cache_index["entries"]}

        # ------------------------------------------------------------------
        # Full validation
        # ------------------------------------------------------------------
        self._cache_index.pop("last_validated", None)
        to_build: list[dict] = []

        total_requested = len(by_path)
        validated = 0
        start = time.monotonic()
        pbar = self._make_pbar(total_requested, desc="validating cache", unit="img")
        for path, sample in by_path.items():
            resolution = _bucket_to_resolution(sample["bucket"])
            entry = self._cache_index["entries"].get(path)
            if entry is not None:
                status = self._validate_entry(path, entry, resolution, sample)
                if status == "valid":
                    # Usage fields (label/split/concept_name) can change without
                    # invalidating the tensor payload. Update them in-place so
                    # iter_sourceless reports the current caller's view.
                    self._refresh_entry_usage(entry, sample)
                    stats["cache_hits"] += 1
                    validated += 1
                    if pbar:
                        pbar.update(1)
                    self._emit_progress(
                        stage="validating",
                        step=validated,
                        total_steps=total_requested,
                        elapsed=time.monotonic() - start,
                        status_text=f"Validating cache ({validated}/{total_requested})",
                        cache_hits=stats["cache_hits"],
                        cache_misses=stats["cache_misses"],
                        cache_dedup_hits=stats["cache_dedup_hits"],
                    )
                    continue
                # Invalid — drop the old entry and rebuild
                with self._index_lock:
                    old_hash = entry.get("hash")
                    if old_hash:
                        self._remove_from_hash_index(old_hash, path)
                    self._cache_index["entries"].pop(path, None)

            to_build.append(sample)
            stats["cache_misses"] += 1
            validated += 1
            if pbar:
                pbar.update(1)
            self._emit_progress(
                stage="validating",
                step=validated,
                total_steps=total_requested,
                elapsed=time.monotonic() - start,
                status_text=f"Validating cache ({validated}/{total_requested})",
                cache_hits=stats["cache_hits"],
                cache_misses=stats["cache_misses"],
                cache_dedup_hits=stats["cache_dedup_hits"],
            )
        if pbar:
            pbar.close()

        # ------------------------------------------------------------------
        # Build missing entries
        # ------------------------------------------------------------------
        if to_build:
            self._build_many(to_build, build_fn, num_workers=num_workers, stage_label=stage_label, stats=stats)

        self._cache_index["last_validated"] = time.time()
        self._save_index()

        total = stats["cache_hits"] + stats["cache_misses"]
        logger.info(
            "SmartCache: %d/%d cached (%d hits, %d misses, %d dedup).",
            stats["cache_hits"] + stats["cache_misses"], total,
            stats["cache_hits"], stats["cache_misses"], stats["cache_dedup_hits"],
        )

        return {p: self._cache_index["entries"][p] for p in by_path if p in self._cache_index["entries"]}

    def get(self, source_path: str) -> tuple[torch.Tensor, dict] | None:
        """Return ``(tensor, metadata)`` for a cached source, or ``None``.

        Tensor is CHW uint8 as written. Metadata excludes the tensor payload.
        """
        self._ensure_index_loaded()
        path = os.path.normpath(source_path)
        entry = self._cache_index["entries"].get(path)
        if entry is None:
            return None
        pt_path = self._pt_path(entry["cache_file"], 0)
        if not os.path.isfile(pt_path):
            return None
        try:
            cached = torch.load(pt_path, weights_only=False, map_location="cpu")
        except (OSError, RuntimeError, EOFError) as exc:
            logger.warning("SmartCache: failed to load %s: %s", pt_path, exc)
            return None
        tensor = cached.get("tensor")
        if tensor is None:
            return None
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        metadata = {k: v for k, v in cached.items() if k != "tensor"}
        return tensor, metadata

    def iter_sourceless(self) -> list[dict]:
        """Rebuild sample dicts from cache.json for training without source files.

        Reads label/split/concept_name from the per-entry fields in cache.json
        (not from the shared .pt payload) so deduped entries each carry their
        own view. Only the bucket/face_bbox/skin_normalise — image-identity
        fields — come from the .pt metadata.
        """
        self._ensure_index_loaded()
        entries = self._cache_index.get("entries", {})
        if not entries:
            raise RuntimeError(
                "Sourceless training enabled but cache is empty. "
                "Build the cache first with sourceless disabled."
            )

        # Cache .pt-metadata lookups per cache_file so dedup entries don't
        # re-open the same file N times.
        meta_by_cache_file: dict[str, dict] = {}

        samples: list[dict] = []
        for path, entry in entries.items():
            if entry.get("modeltype") != self.modeltype:
                raise RuntimeError(
                    f"Cache modeltype mismatch for '{path}': cached as "
                    f"'{entry.get('modeltype')}', requested '{self.modeltype}'. "
                    "Rebuild the cache or change modeltype."
                )
            if entry.get("cache_version", 0) != CACHE_VERSION:
                raise RuntimeError(
                    f"Cache entry for '{path}' was built with an older format "
                    f"(cache_version={entry.get('cache_version')}). Rebuild the cache."
                )
            cache_file = entry["cache_file"]
            pt_path = self._pt_path(cache_file, 0)
            if not os.path.isfile(pt_path):
                raise RuntimeError(
                    f"Sourceless training: cache file '{pt_path}' is missing. Rebuild the cache."
                )

            meta = meta_by_cache_file.get(cache_file)
            if meta is None:
                try:
                    meta = torch.load(pt_path, weights_only=False, map_location="cpu")
                except (OSError, RuntimeError, EOFError) as exc:
                    raise RuntimeError(f"Failed to read cache entry '{pt_path}': {exc}") from exc
                meta_by_cache_file[cache_file] = meta

            bucket = meta.get("__bucket")
            if "label" not in entry or "split" not in entry or bucket is None:
                raise RuntimeError(
                    f"Cache entry '{path}' is missing label/split/bucket — rebuild the cache."
                )

            samples.append({
                "path": path,
                "bucket": tuple(bucket),
                "label": _label_from_storage(entry["label"]),
                "split": str(entry.get("split", "")),
                "concept_name": str(entry.get("concept_name", "")),
                "face_bbox": meta.get("__face_bbox"),
                "skin_normalise": bool(entry.get("skin_normalise", meta.get("__skin_normalise", False))),
                "_sourceless": True,
            })
        return samples

    # ------------------------------------------------------------------
    # GC (static — usable without instantiating)
    # ------------------------------------------------------------------

    @staticmethod
    def gc_preview(
        cache_dir: Path | str,
        *,
        valid_sources: set[str] | None = None,
    ) -> dict:
        """Return counts and bytes of orphaned cache entries without deleting.

        When ``valid_sources`` is provided, entries whose source path is not in
        the set are treated as dead — use this to GC based on DB membership
        (e.g. after a concept is deleted). When omitted, falls back to
        ``os.path.isfile`` — only source files physically removed count as dead.
        """
        cache_dir = Path(cache_dir)
        cache_path = cache_dir / "cache.json"
        if not cache_path.is_file():
            return {
                "dead_entries": 0,
                "dead_entry_bytes": 0,
                "orphan_pt": 0,
                "orphan_pt_bytes": 0,
                "total_entries": 0,
                "total_bytes": 0,
            }

        with open(cache_path, "r") as f:
            index = json.load(f)
        entries = index.get("entries", {})

        normalised_valid = (
            {os.path.normpath(p) for p in valid_sources}
            if valid_sources is not None else None
        )

        def _is_dead(path: str) -> bool:
            if normalised_valid is not None:
                return os.path.normpath(path) not in normalised_valid
            return not os.path.isfile(path)

        dead_paths = {p for p in entries if _is_dead(p)}

        # Count each .pt file at most once, even when several deduped entries
        # point at it. An entry is "live" if any of its entries is not dead.
        live_cache_files: set[str] = set()
        all_cache_files: dict[str, int] = {}  # cache_file -> size
        for path, entry in entries.items():
            cf = entry.get("cache_file", "")
            if not cf:
                continue
            pt_path = cache_dir / f"{cf}_1.pt"
            if cf not in all_cache_files and pt_path.is_file():
                all_cache_files[cf] = pt_path.stat().st_size
            if path not in dead_paths:
                live_cache_files.add(cf)

        total_bytes = sum(all_cache_files.values())
        dead_only_cache_files = set(all_cache_files) - live_cache_files
        dead_entry_bytes = sum(all_cache_files[cf] for cf in dead_only_cache_files)

        referenced_pt_paths = {
            os.path.normpath(str(cache_dir / f"{cf}_1.pt")) for cf in all_cache_files
        }

        orphan_pt = 0
        orphan_pt_bytes = 0
        for scan in os.scandir(cache_dir):
            if scan.name.endswith(".pt") and scan.is_file():
                if os.path.normpath(scan.path) not in referenced_pt_paths:
                    orphan_pt += 1
                    orphan_pt_bytes += scan.stat().st_size

        return {
            "dead_entries": len(dead_paths),
            "dead_entry_bytes": dead_entry_bytes,
            "orphan_pt": orphan_pt,
            "orphan_pt_bytes": orphan_pt_bytes,
            "total_entries": len(entries),
            "total_bytes": total_bytes,
        }

    @staticmethod
    def gc_clean(
        cache_dir: Path | str,
        *,
        valid_sources: set[str] | None = None,
    ) -> dict:
        """Remove orphaned entries and .pt files. Returns a summary dict.

        See :meth:`gc_preview` for the ``valid_sources`` semantics.
        """
        cache_dir = Path(cache_dir)
        cache_path = cache_dir / "cache.json"
        if not cache_path.is_file():
            return {"removed_entries": 0, "removed_pt": 0, "freed_bytes": 0}

        with open(cache_path, "r") as f:
            index = json.load(f)
        entries = index.get("entries", {})
        hash_index = index.get("hash_index", {})

        normalised_valid = (
            {os.path.normpath(p) for p in valid_sources}
            if valid_sources is not None else None
        )

        def _is_dead(path: str) -> bool:
            if normalised_valid is not None:
                return os.path.normpath(path) not in normalised_valid
            return not os.path.isfile(path)

        freed_bytes = 0
        removed_pt = 0
        removed_entries = 0

        dead_paths = [p for p in entries if _is_dead(p)]
        for p in dead_paths:
            entry = entries.pop(p)
            removed_entries += 1
            file_hash = entry.get("hash", "")
            if file_hash in hash_index:
                paths = hash_index[file_hash]
                if p in paths:
                    paths.remove(p)
                if not paths:
                    del hash_index[file_hash]
                    cf = entry.get("cache_file", "")
                    pt_path = cache_dir / f"{cf}_1.pt"
                    if pt_path.is_file():
                        freed_bytes += pt_path.stat().st_size
                        try:
                            pt_path.unlink()
                            removed_pt += 1
                        except OSError:
                            pass

        referenced = set()
        for entry in entries.values():
            cf = entry.get("cache_file", "")
            pt_path = cache_dir / f"{cf}_1.pt"
            referenced.add(os.path.normpath(str(pt_path)))

        for scan in os.scandir(cache_dir):
            if scan.name.endswith(".pt") and scan.is_file():
                if os.path.normpath(scan.path) not in referenced:
                    try:
                        freed_bytes += scan.stat().st_size
                        os.remove(scan.path)
                        removed_pt += 1
                    except OSError:
                        pass

        tmp = cache_path.with_suffix(".json.tmp")
        bak = cache_path.with_suffix(".json.bak")
        with open(tmp, "w") as f:
            json.dump(index, f, indent=2)
        if cache_path.exists():
            try:
                shutil.copy2(cache_path, bak)
            except OSError:
                pass
        os.replace(tmp, cache_path)

        return {
            "removed_entries": removed_entries,
            "removed_pt": removed_pt,
            "freed_bytes": freed_bytes,
        }

    # ------------------------------------------------------------------
    # Index IO
    # ------------------------------------------------------------------

    def _cache_json(self) -> Path:
        return self.cache_dir / "cache.json"

    def _ensure_index_loaded(self) -> None:
        if self._cache_index is None:
            self._load_index()

    def _load_index(self) -> None:
        cache_path = self._cache_json()
        tmp = cache_path.with_suffix(".json.tmp")
        bak = cache_path.with_suffix(".json.bak")

        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    self._cache_index = json.load(f)
                for p in (tmp, bak):
                    if p.exists():
                        try:
                            p.unlink()
                        except OSError:
                            pass
                self._normalise_index()
                return
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("SmartCache: cache.json unreadable (%s) — trying recovery", exc)

        if tmp.exists():
            try:
                with open(tmp, "r") as f:
                    self._cache_index = json.load(f)
                os.replace(tmp, cache_path)
                self._normalise_index()
                return
            except (json.JSONDecodeError, OSError):
                pass

        if bak.exists():
            try:
                with open(bak, "r") as f:
                    self._cache_index = json.load(f)
                os.replace(bak, cache_path)
                self._normalise_index()
                return
            except (json.JSONDecodeError, OSError):
                pass

        self._cache_index = {"version": CACHE_VERSION, "entries": {}, "hash_index": {}}

    def _normalise_index(self) -> None:
        assert self._cache_index is not None
        self._cache_index.setdefault("entries", {})
        self._cache_index.setdefault("hash_index", {})
        self._cache_index.setdefault("version", CACHE_VERSION)

    def _save_index(self, *, compact: bool = False) -> None:
        assert self._cache_index is not None
        cache_path = self._cache_json()
        tmp = cache_path.with_suffix(".json.tmp")
        bak = cache_path.with_suffix(".json.bak")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with self._index_lock:
            with open(tmp, "w") as f:
                json.dump(self._cache_index, f, indent=None if compact else 2)

            if cache_path.exists():
                try:
                    shutil.copy2(cache_path, bak)
                except OSError:
                    pass

            os.replace(tmp, cache_path)

    def _flush_throttled(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._last_flush_time >= _FLUSH_INTERVAL_SECONDS:
            self._save_index(compact=True)
            self._last_flush_time = now

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_entry(self, path: str, entry: dict, resolution: str, sample: dict) -> str:
        if entry.get("modeltype") != self.modeltype:
            return "modeltype_changed"
        if entry.get("resolution") and entry["resolution"] != resolution:
            return "resolution_changed"
        if entry.get("cache_version", 0) != CACHE_VERSION:
            return "version_changed"

        # Build-signature fields: anything that affects the cached tensor bytes.
        if bool(entry.get("skin_normalise", False)) != bool(sample.get("skin_normalise", False)):
            return "build_sig_changed"
        if entry.get("face_bbox_sig", "none") != _bbox_sig(sample.get("face_bbox")):
            return "build_sig_changed"
        if entry.get("face_model_sig", "none") != self.face_model_sig:
            return "build_sig_changed"

        try:
            current_mtime = os.path.getmtime(path)
        except OSError:
            if entry.get("mtime") == 0 and os.path.isfile(self._pt_path(entry["cache_file"], 0)):
                return "valid"
            return "missing"

        if current_mtime == entry.get("mtime") and os.path.isfile(self._pt_path(entry["cache_file"], 0)):
            return "valid"

        file_hash = _hash_file(path)
        if file_hash != entry.get("hash"):
            return "content_changed"

        entry["mtime"] = current_mtime
        if not os.path.isfile(self._pt_path(entry["cache_file"], 0)):
            return "missing_pt"
        return "valid"

    def _refresh_entry_usage(self, entry: dict, sample: dict) -> None:
        """Rewrite label/split/concept_name on an otherwise-valid entry.

        The tensor bytes do not depend on these fields, so we can promote the
        current caller's view without rebuilding. Matters for dedup: two
        concepts sharing a cache_file must each carry their own label/split.
        """
        entry["label"] = _label_to_storage(sample.get("label", 0))
        entry["split"] = str(sample.get("split", ""))
        entry["concept_name"] = str(sample.get("concept_name", ""))

    def _fast_validate(self, requested_paths: Iterable[str]) -> bool:
        last_validated = self._cache_index.get("last_validated")
        if last_validated is None:
            return False
        entries = self._cache_index.get("entries", {})
        if not entries:
            return False

        requested_set = set(requested_paths)
        if not requested_set.issubset(entries.keys()):
            return False

        parent_dirs = {os.path.dirname(p) for p in requested_set}
        for d in parent_dirs:
            try:
                if os.path.getmtime(d) > last_validated:
                    return False
            except OSError:
                return False

        sample_keys = list(requested_set)
        if len(sample_keys) > 100:
            sample_size = min(50, max(10, len(sample_keys) // 20))
            sample_keys = random.sample(sample_keys, sample_size)

        for path in sample_keys:
            entry = entries[path]
            if entry.get("modeltype") != self.modeltype:
                return False
            try:
                current_mtime = os.path.getmtime(path)
            except OSError:
                if entry.get("mtime") != 0:
                    return False
                current_mtime = 0
            if current_mtime != entry.get("mtime"):
                return False
            if not os.path.isfile(self._pt_path(entry["cache_file"], 0)):
                return False

        return True

    # ------------------------------------------------------------------
    # Dedup + build
    # ------------------------------------------------------------------

    def _add_to_hash_index(self, file_hash: str, path: str) -> None:
        paths = self._cache_index["hash_index"].setdefault(file_hash, [])
        if path not in paths:
            paths.append(path)

    def _remove_from_hash_index(self, file_hash: str, path: str) -> None:
        paths = self._cache_index["hash_index"].get(file_hash)
        if not paths:
            return
        if path in paths:
            paths.remove(path)
        if not paths:
            del self._cache_index["hash_index"][file_hash]

    def _try_dedup(self, sample: dict, file_hash: str, resolution: str, mtime: float) -> bool:
        path = sample["path"]
        skin = bool(sample.get("skin_normalise", False))
        bbox_sig = _bbox_sig(sample.get("face_bbox"))
        with self._index_lock:
            candidates = self._cache_index["hash_index"].get(file_hash)
            if not candidates:
                return False
            for existing_path in candidates:
                existing = self._cache_index["entries"].get(existing_path)
                if existing is None:
                    continue
                if existing.get("modeltype") != self.modeltype:
                    continue
                if existing.get("resolution") != resolution:
                    continue
                # Build-signature must match — same tensor bytes.
                if bool(existing.get("skin_normalise", False)) != skin:
                    continue
                if existing.get("face_bbox_sig", "none") != bbox_sig:
                    continue
                if existing.get("face_model_sig", "none") != self.face_model_sig:
                    continue
                cache_file = existing["cache_file"]
                pt_path = Path(self._pt_path(cache_file, 0))
                if not pt_path.is_file():
                    continue
                # Reuse the existing .pt via a new entry pointing at the same cache_file.
                # Usage fields (label/split/concept_name) come from the current caller,
                # not the first writer — each concept keeps its own view.
                self._cache_index["entries"][path] = {
                    "filename": os.path.basename(path),
                    "filepath": path,
                    "hash": file_hash,
                    "mtime": mtime,
                    "modeltype": self.modeltype,
                    "resolution": resolution,
                    "cache_file": cache_file,
                    "cache_version": CACHE_VERSION,
                    "label": _label_to_storage(sample.get("label", 0)),
                    "split": str(sample.get("split", "")),
                    "concept_name": str(sample.get("concept_name", "")),
                    "skin_normalise": skin,
                    "face_bbox_sig": bbox_sig,
                    "face_model_sig": self.face_model_sig,
                }
                self._add_to_hash_index(file_hash, path)
                return True
            return False

    def _pt_path(self, cache_file: str, variation: int) -> str:
        return os.path.join(self._real_cache_dir, f"{cache_file}_{variation + 1}.pt")

    def _make_cache_file(self, file_hash: str, resolution: str) -> str:
        return f"{file_hash[:12]}_{resolution}"

    def _build_one(
        self,
        sample: dict,
        build_fn: Callable[[dict], np.ndarray],
    ) -> tuple[str, str]:
        path = sample["path"]
        resolution = _bucket_to_resolution(sample["bucket"])

        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return path, "missing_source"

        try:
            file_hash = _hash_file(path)
        except OSError as exc:
            return path, f"hash_failed:{exc}"

        # Dedup against existing entries with the same hash + resolution + build signature.
        if self._try_dedup(sample, file_hash, resolution, mtime):
            return path, "dedup"

        try:
            arr = build_fn(sample)
        except Exception as exc:  # build_fn is user-supplied
            return path, f"build_failed:{exc}"

        if not isinstance(arr, np.ndarray) or arr.dtype != np.uint8 or arr.ndim != 3:
            return path, "build_returned_invalid_tensor"

        cache_file = self._make_cache_file(file_hash, resolution)
        skin = bool(sample.get("skin_normalise", False))
        bbox_sig = _bbox_sig(sample.get("face_bbox"))
        # .pt payload carries the tensor plus image-identity metadata. Usage fields
        # (label/split/concept_name) live in cache.json so dedup entries can each
        # carry their own view without rewriting the shared .pt.
        payload = {
            "tensor": arr,
            "__cache_version": CACHE_VERSION,
            "__modeltype": self.modeltype,
            "__resolution": resolution,
            "__bucket": tuple(sample["bucket"]),
            "__source_path": path,
            "__source_mtime": mtime,
            "__source_hash": file_hash,
            "__skin_normalise": skin,
            "__face_bbox": sample.get("face_bbox"),
            "__face_model_sig": self.face_model_sig,
        }

        pt_path = self._pt_path(cache_file, 0)
        tmp_path = f"{pt_path}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, pt_path)
        except (OSError, RuntimeError) as exc:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return path, f"write_failed:{exc}"

        with self._index_lock:
            self._cache_index["entries"][path] = {
                "filename": os.path.basename(path),
                "filepath": path,
                "hash": file_hash,
                "mtime": mtime,
                "modeltype": self.modeltype,
                "resolution": resolution,
                "cache_file": cache_file,
                "cache_version": CACHE_VERSION,
                "label": _label_to_storage(sample.get("label", 0)),
                "split": str(sample.get("split", "")),
                "concept_name": str(sample.get("concept_name", "")),
                "skin_normalise": skin,
                "face_bbox_sig": bbox_sig,
                "face_model_sig": self.face_model_sig,
            }
            self._add_to_hash_index(file_hash, path)

        return path, "built"

    def _build_many(
        self,
        samples: list[dict],
        build_fn: Callable[[dict], np.ndarray],
        *,
        num_workers: int,
        stage_label: str,
        stats: dict,
    ) -> None:
        total = len(samples)
        self._last_flush_time = time.monotonic()
        start = time.monotonic()

        pbar = self._make_pbar(total, desc=stage_label, unit="img")
        done = 0
        failed: list[tuple[str, str]] = []

        if num_workers <= 1:
            for sample in samples:
                path, status = self._build_one(sample, build_fn)
                done += 1
                if status == "built":
                    pass
                elif status == "dedup":
                    stats["cache_dedup_hits"] += 1
                else:
                    failed.append((path, status))
                if pbar:
                    pbar.update(1)
                if done % _FLUSH_INTERVAL_ITEMS == 0:
                    self._flush_throttled()
                self._emit_progress(
                    stage="caching",
                    step=done,
                    total_steps=total,
                    elapsed=time.monotonic() - start,
                    status_text=f"Caching ({done}/{total})",
                    cache_hits=stats["cache_hits"],
                    cache_misses=stats["cache_misses"],
                    cache_dedup_hits=stats["cache_dedup_hits"],
                )
                if self._stop_check():
                    self._save_index()
                    raise CachingStoppedException()
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self._build_one, s, build_fn) for s in samples]
                try:
                    for fut in concurrent.futures.as_completed(futures):
                        path, status = fut.result()
                        done += 1
                        if status == "built":
                            pass
                        elif status == "dedup":
                            stats["cache_dedup_hits"] += 1
                        else:
                            failed.append((path, status))
                        if pbar:
                            pbar.update(1)
                        if done % _FLUSH_INTERVAL_ITEMS == 0:
                            self._flush_throttled()
                        self._emit_progress(
                            stage="caching",
                            step=done,
                            total_steps=total,
                            elapsed=time.monotonic() - start,
                            status_text=f"Caching ({done}/{total})",
                            cache_hits=stats["cache_hits"],
                            cache_misses=stats["cache_misses"],
                            cache_dedup_hits=stats["cache_dedup_hits"],
                        )
                        if self._stop_check():
                            for f in futures:
                                f.cancel()
                            executor.shutdown(wait=False, cancel_futures=True)
                            self._save_index()
                            raise CachingStoppedException()
                except CachingStoppedException:
                    raise

        if pbar:
            pbar.close()

        if failed:
            for p, reason in failed[:10]:
                logger.warning("SmartCache: build failed for %s: %s", p, reason)
            if len(failed) > 10:
                logger.warning("SmartCache: ... and %d more build failures", len(failed) - 10)

        self._save_index()

    # ------------------------------------------------------------------
    # Progress / tqdm plumbing
    # ------------------------------------------------------------------

    def _emit_progress(
        self,
        *,
        stage: str,
        step: int,
        total_steps: int,
        status_text: str,
        cache_hits: int = 0,
        cache_misses: int = 0,
        cache_dedup_hits: int = 0,
        elapsed: float | None = None,
        force: bool = False,
    ) -> None:
        if self._progress_cb is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_progress_time) < _PROGRESS_MIN_INTERVAL and step < total_steps:
            return
        self._last_progress_time = now

        if elapsed is not None and elapsed > 0 and step > 0:
            throughput = step / elapsed
            remaining = max(0, total_steps - step)
            eta_seconds = remaining / throughput if throughput > 0 else None
        else:
            throughput = None
            eta_seconds = None

        try:
            self._progress_cb({
                "type": "training_progress",
                "stage": stage,
                "status_text": status_text,
                "step": step,
                "total_steps": total_steps,
                "eta_seconds": eta_seconds,
                "throughput": throughput,
                "throughput_unit": "img/s",
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "cache_dedup_hits": cache_dedup_hits,
            })
        except Exception as exc:
            logger.warning("SmartCache: progress callback failed: %s", exc)

    def _make_pbar(self, total: int, *, desc: str, unit: str):
        if not self._tqdm_enabled or _tqdm is None or total <= 0:
            return None
        return _tqdm(total=total, desc=desc, unit=unit, smoothing=0.1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bucket_to_resolution(bucket: tuple[int, int] | list[int]) -> str:
    return f"{int(bucket[0])}x{int(bucket[1])}"


def _hash_file(path: str) -> str:
    h = xxhash.xxh64()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
