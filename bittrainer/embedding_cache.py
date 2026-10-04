"""Cached pooled backbone features for the head probe.

Stores one post-norm pooled vector per image, computed once per backbone era.
Layered on top of :class:`bittrainer.smart_cache.SmartCache` (which caches input
image tensors): embeddings are keyed by the backbone-feature hash (directory
namespace) plus the image's content hash (filename). A fine-tuned backbone
produces a different feature hash, so it gets a fresh namespace and never
collides with the pretrained-era cache — re-running the probe after a full
fine-tune rebuilds against the adapted features automatically. Establishing a
new era also prunes the prior one (:meth:`EmbeddingCache.prune_other_eras`, run
from :meth:`ensure`): the superseded hash will never recur, so its vectors are
deleted rather than left to accumulate under the cache root.

The cache point is ``flatten(norm(global_pool(features)))`` (pre pre_logits/fc),
computed by :func:`bittrainer.model.pooled_features` with the backbone in
``eval()`` so the forward is deterministic (no drop_path, no head dropout).

**Packs** (Bitcrush ISSUE-1110, opt-in via ``packed=True``): reading one small
``.npy`` per image costs ~0.5 ms each on NTFS (minutes cold), and a head-only
run read every vector three times (``ensure``'s audit, then each probe
sweep's gather). A packed cache keeps the run's vectors in memory once loaded
and writes one consolidated ``pack-<ns>-<id>.npz`` holding exactly the run's
vectors, so the next run — or the next backbone-comparison candidate — loads
them in one read. The per-image ``.npy`` files are still written and stay the
durable source of truth, so readers that predate packs keep working. A pack
is written under a fresh name and then older packs are deleted best-effort:
it is never replaced in place (on Windows a reader holding a file open makes
``os.replace`` over it fail), it is read fully into memory (never memory-
mapped), and a pack that fails validation is simply ignored.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import random
import re
import shutil
import time
import uuid
import zipfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from bittrainer.gpu_augment import apply_val_transform
from bittrainer.model import pooled_features

logger = logging.getLogger(__name__)

EMBED_CACHE_VERSION = 1

# Era namespace = backbone-feature hash (16-char hex digest from
# model.backbone_feature_hash), optionally suffixed with a preprocessing
# signature digest (see _sig_suffix). Used to recognise our own era directories
# when pruning so we never touch an unrelated sibling under the cache root.
_ERA_DIR_RE = re.compile(r"^[0-9a-f]{16}(-[0-9a-f]{8})?$")

# Consolidated vector packs (see the module docstring). Names sort
# chronologically: ``pack-<time_ns, 20 digits>-<8 hex>.npz``.
PACK_VERSION = 1
_PACK_GLOB = "pack-*.npz"
_PACK_READ_ERRORS = (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile)


def _sig_suffix(preproc_sig: str) -> str:
    """Directory-name suffix folding ``preproc_sig`` into the era identity.

    ``preproc_sig`` is cache IDENTITY, not just provenance: a preprocessing
    change (e.g. a different input resolution) changes every pooled vector's
    VALUES while the dim stays fixed, so vectors built under a different sig
    must never be silently reused. The historical default ``"val_imagenet"``
    maps to the empty suffix so existing caches keep their directory name.
    """
    if preproc_sig == "val_imagenet":
        return ""
    return "-" + hashlib.sha1(preproc_sig.encode("utf-8")).hexdigest()[:8]


class EmbeddingCacheMismatch(RuntimeError):
    """A cached embedding disagrees with a fresh backbone forward (fail loud)."""


def cached_hashes_for_sig(
    cache_dir: str | Path, preproc_sig: str = "val_imagenet"
) -> frozenset[str]:
    """Content hashes holding a pooled vector under any era of *preproc_sig*.

    Era directories are named ``{backbone_hash16}{_sig_suffix(preproc_sig)}``.
    Scanning every era of the requested sig — rather than one known backbone
    hash — lets callers that run before the model is built (negative-pool
    seeding) treat "some era of this preprocessing already embedded this image"
    as a hit. Engine shares one cache root across concepts with ``prune=False``,
    so several eras coexist deliberately; a hit in a superseded era still
    signals an image the pipeline has fully feature-passed before.
    """
    root = Path(cache_dir)
    if not root.is_dir():
        return frozenset()
    suffix = _sig_suffix(preproc_sig)
    hashes: set[str] = set()
    for era in root.iterdir():
        if not era.is_dir() or not _ERA_DIR_RE.match(era.name):
            continue
        if era.name[16:] != suffix:
            continue
        hashes.update(vector.stem for vector in era.glob("*.npy"))
    return frozenset(hashes)


def _content_hash(path: str, smart_cache: Any | None) -> str | None:
    if smart_cache is not None:
        h = smart_cache.content_hash(path)
        if h:
            return h
    from bittrainer.smart_cache import _hash_file

    try:
        return _hash_file(path)
    except OSError:
        return None


class EmbeddingCache:
    """Pooled-feature vectors for one backbone era, on disk under one namespace."""

    def __init__(
        self,
        cache_dir: str | Path,
        backbone_hash: str,
        pooled_dim: int,
        *,
        preproc_sig: str = "val_imagenet",
        packed: bool = False,
    ) -> None:
        self.root = Path(cache_dir) / (backbone_hash + _sig_suffix(preproc_sig))
        self.backbone_hash = backbone_hash
        self.pooled_dim = int(pooled_dim)
        self.preproc_sig = preproc_sig
        self._meta_path = self.root / "meta.json"
        # ensure() -> verify() -> probe gather all hash the same corpus; memoise
        # so only the first pass pays the per-path index lookups.
        self._hash_memo: dict[str, str | None] = {}
        # Sticky CUDA-OOM cap for the embedding forward: once a batch size
        # OOMs, every later batch runs at the reduced size (see
        # _forward_pooled_capped).
        self._forward_cap: int | None = None
        # Packed mode (ISSUE-1110): validated vectors held in memory by content
        # hash (loaded from the newest pack, from per-image files, or built),
        # and a consolidated pack of the run's set written by ensure().
        self.packed = bool(packed)
        self._mem: dict[str, np.ndarray] = {}
        self._pack_loaded = False
        self._pack_hashes: frozenset[str] = frozenset()

    def _hash(self, path: str, smart_cache: Any | None) -> str | None:
        if path in self._hash_memo:
            return self._hash_memo[path]
        h = _content_hash(path, smart_cache)
        self._hash_memo[path] = h
        return h

    def _vec_path(self, content_hash: str) -> Path:
        return self.root / f"{content_hash}.npy"

    def _read_meta(self) -> dict[str, Any]:
        try:
            data = json.loads(self._meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_meta(self, *, complete: bool) -> None:
        """Durably publish the era state before/after vector writes.

        ``complete=False`` is fsynced before building. If the machine dies
        mid-cache, the next run audits every existing vector and selectively
        rebuilds damaged files. The metadata itself is also replaced atomically
        so a torn JSON write cannot make a healthy era look complete.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": EMBED_CACHE_VERSION,
            "backbone_hash": self.backbone_hash,
            "pooled_dim": self.pooled_dim,
            "preproc_sig": self.preproc_sig,
            "complete": bool(complete),
        }
        tmp_path = self._meta_path.with_name(
            f".{self._meta_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self._meta_path)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

    def _valid_vector(self, vector: np.ndarray) -> bool:
        arr = np.asarray(vector)
        return (
            arr.shape == (self.pooled_dim,)
            and np.issubdtype(arr.dtype, np.number)
            and bool(np.isfinite(arr).all())
            and float(np.linalg.norm(arr)) > 1e-12
        )

    def _load_vector_path(self, path: Path) -> np.ndarray | None:
        try:
            vector = np.load(path, allow_pickle=False)
        except (OSError, ValueError, EOFError):
            return None
        return vector if self._valid_vector(vector) else None

    # -- consolidated packs (ISSUE-1110) -----------------------------------
    def _pack_meta(self) -> dict[str, Any]:
        return {
            "pack_version": PACK_VERSION,
            "embed_version": EMBED_CACHE_VERSION,
            "backbone_hash": self.backbone_hash,
            "pooled_dim": self.pooled_dim,
            "preproc_sig": self.preproc_sig,
        }

    def _pack_paths(self) -> list[Path]:
        """Packs in this era, newest first (names sort chronologically)."""
        if not self.root.is_dir():
            return []
        return sorted(self.root.glob(_PACK_GLOB), key=lambda p: p.name, reverse=True)

    def _read_pack(self, path: Path) -> dict[str, np.ndarray] | None:
        """One pack fully in memory, or ``None`` when it fails any check.

        All-or-nothing: a pack from another era, of the wrong shape, or with a
        single non-finite / zero row is rejected whole — the per-image files
        then serve, and the next ``ensure`` writes a good pack.
        """
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"][()]))
                hashes = z["hashes"]
                vectors = z["vectors"]
        except (*_PACK_READ_ERRORS, json.JSONDecodeError, TypeError):
            return None
        if not isinstance(meta, dict) or meta != self._pack_meta():
            return None
        if (
            vectors.ndim != 2
            or vectors.shape != (hashes.shape[0], self.pooled_dim)
            or vectors.dtype != np.float32
            or hashes.ndim != 1
        ):
            return None
        if not bool(np.isfinite(vectors).all()):
            return None
        if vectors.shape[0] and float(np.linalg.norm(vectors, axis=1).min()) <= 1e-12:
            return None
        return {str(h): vectors[i] for i, h in enumerate(hashes)}

    def _load_pack(self) -> dict[str, np.ndarray] | None:
        """The newest valid pack of this era, or ``None``.

        A pack deleted by a concurrent writer between the listing and the read
        (or one this process cannot open) just falls through to the next.
        """
        for path in self._pack_paths():
            rows = self._read_pack(path)
            if rows is not None:
                return rows
            logger.info("EmbeddingCache: ignoring unreadable pack %s", path.name)
        return None

    def _ensure_pack_loaded(self) -> None:
        if not self.packed or self._pack_loaded:
            return
        self._pack_loaded = True
        rows = self._load_pack()
        if rows:
            self._pack_hashes = frozenset(rows)
            for content_hash, vector in rows.items():
                self._mem.setdefault(content_hash, vector)

    def _write_pack(self, hashes: list[str]) -> Path | None:
        """Write the vectors of *hashes* (all held in memory) as a new pack.

        Written to a temp file and renamed to a NEW unique name, so no reader's
        open handle can block it; older packs are then removed best-effort (a
        pack another process is reading stays until a later write). Failures
        only cost the speed-up, never the run.
        """
        if not hashes:
            return None
        self.root.mkdir(parents=True, exist_ok=True)
        name = f"pack-{time.time_ns():020d}-{uuid.uuid4().hex[:8]}.npz"
        final_path = self.root / name
        tmp_path = self.root / f".{name}.{os.getpid()}.tmp"
        try:
            vectors = np.stack([self._mem[h] for h in hashes]).astype(np.float32, copy=False)
            with tmp_path.open("wb") as handle:
                np.savez(
                    handle,
                    meta=np.array(json.dumps(self._pack_meta())),
                    hashes=np.array(hashes),
                    vectors=vectors,
                )
            os.replace(tmp_path, final_path)
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("EmbeddingCache: could not write a vector pack: %s", exc)
            return None
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass
        self._pack_hashes = frozenset(hashes)
        for old in self._pack_paths():
            if old.name >= name:
                continue
            try:
                old.unlink()
            except OSError:
                pass
        return final_path

    def _save_vector(self, content_hash: str, vector: np.ndarray) -> None:
        """Save one vector via temp file + atomic replace (no per-vector fsync).

        Durability comes from the era marker, not from flushing each vector:
        ``ensure`` fsyncs ``complete=False`` before any vector write, and every
        reader (ensure's audit, get_vector, verify) validates shape/finiteness/
        norm on load — so a torn write after power loss is detected and rebuilt
        rather than trusted. Dropping the per-vector fsync removes the dominant
        write cost on NTFS (one fsync per image).
        """
        final_path = self._vec_path(content_hash)
        tmp_path = final_path.with_name(
            f".{final_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with tmp_path.open("wb") as handle:
                np.save(handle, vector, allow_pickle=False)
            os.replace(tmp_path, final_path)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

    def prune_other_eras(self) -> list[str]:
        """Delete sibling era directories for other (now-invalid) backbone hashes.

        A fine-tune moves the backbone weights, producing a new feature hash and
        thus a fresh era namespace; the previous hash will never recur, so its
        stored pooled vectors are dead weight. Without this, every fine-tune
        leaks one full era (a ``.npy`` per image) under the cache root forever.

        Called from :meth:`ensure`, so committing to build the active era also
        reclaims the stale ones. Only directories that are recognisably our own
        eras — a 16-char hex name and/or an embedding-cache ``meta.json`` — are
        removed; any unrelated sibling is left untouched. Best-effort: a removal
        failure (e.g. a concurrent reader holding a handle on Windows) is logged
        and skipped rather than aborting the training run.

        Returns the list of removed era hashes.
        """
        parent = self.root.parent
        if not parent.is_dir():
            return []
        removed: list[str] = []
        for child in parent.iterdir():
            # Compare against the full namespaced name: two preproc sigs of the
            # same backbone hash are mutually pruning (a preprocessing switch
            # invalidates the old vectors just like a weight change does).
            if not child.is_dir() or child.name == self.root.name:
                continue
            if (
                not _ERA_DIR_RE.match(child.name)
                and not (child / "meta.json").is_file()
            ):
                continue
            try:
                shutil.rmtree(child)
                removed.append(child.name)
            except OSError as exc:
                logger.warning(
                    "EmbeddingCache: could not prune stale era %s: %s", child.name, exc
                )
        if removed:
            logger.info(
                "EmbeddingCache: pruned %d stale backbone era(s): %s",
                len(removed),
                ", ".join(removed),
            )
        return removed

    def _load_input_tensor(
        self, sample: dict, smart_cache: Any | None
    ) -> torch.Tensor | None:
        if smart_cache is not None:
            res = smart_cache.get(sample["path"])
            if res is not None:
                tensor, _ = res
                bw, bh = int(sample["bucket"][0]), int(sample["bucket"][1])
                if tuple(tensor.shape[-2:]) == (bh, bw):
                    return tensor
        from bittrainer.cache_builders import build_image_tensor

        try:
            arr = build_image_tensor(sample)
        except (OSError, ValueError):
            return None
        return torch.from_numpy(np.ascontiguousarray(arr))

    def _forward_pooled(
        self,
        batch: torch.Tensor,
        backbone: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
    ) -> np.ndarray:
        if device.type == "cuda":
            # Pinned staging lets the H2D copy overlap the previous batch's
            # compute instead of serialising behind it.
            if not batch.is_pinned():
                batch = batch.pin_memory()
            batch = batch.to(device, non_blocking=True)
        else:
            batch = batch.to(device)
        batch = apply_val_transform(batch, dtype=dtype)
        with (
            torch.no_grad(),
            torch.amp.autocast(
                device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)
            ),
        ):
            vecs = pooled_features(backbone, batch)
        return vecs.float().cpu().numpy()

    def _forward_pooled_capped(
        self,
        tensors: list[torch.Tensor],
        backbone: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
    ) -> np.ndarray:
        """Pooled forward for a same-bucket batch under a CUDA-OOM backoff.

        The no-grad eval forward is activation-bound, so a size that OOMs once
        will OOM every time: on OOM the sub-batch halves and the reduced cap
        sticks for the rest of the run (``self._forward_cap``). Callers can
        therefore request an optimistic batch size without risking the run.
        """
        while True:
            sub = min(len(tensors), self._forward_cap or len(tensors))
            try:
                chunks = [
                    self._forward_pooled(
                        torch.stack(tensors[start : start + sub]), backbone, device, dtype
                    )
                    for start in range(0, len(tensors), sub)
                ]
                return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)
            except torch.cuda.OutOfMemoryError:
                if sub <= 1:
                    raise
                self._forward_cap = max(1, sub // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.warning(
                    "EmbeddingCache: CUDA OOM at embedding batch %d; retrying at %d "
                    "(cap holds for the rest of this run)",
                    sub,
                    self._forward_cap,
                )

    def _save_many(self, pairs: list[tuple[str, np.ndarray]]) -> None:
        for content_hash, vector in pairs:
            self._save_vector(content_hash, vector)

    def ensure(
        self,
        samples: list[dict],
        backbone: nn.Module,
        smart_cache: Any | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int = 256,
        io_workers: int = 8,
        prefetch_batches: int = 2,
        progress_cb: Callable[[int, int], None] | None = None,
        stop_check: Callable[[], bool] | None = None,
        prune: bool = True,
        check_progress_cb: Callable[[int, int], None] | None = None,
    ) -> dict:
        """Build any missing pooled vectors for *samples* under this backbone era.

        The build is pipelined so the GPU never waits on disk: ``io_workers``
        threads keep ``prefetch_batches`` batches of input tensors loading
        ahead of the forward, and vector writes drain through a background
        writer behind it. ``batch_size`` can be optimistic — the forward is
        no-grad eval and backs off (halving, sticky) on CUDA OOM.

        ``prune=False`` skips the sibling-era reclaim — REQUIRED when several
        eras must coexist deliberately (the resolution probe builds one small
        era per candidate resolution and must not delete the production era or
        its own previous candidates; the next real training run prunes losers).

        Packed caches (ISSUE-1110) take every vector the newest valid pack
        holds without touching its per-image file, keep all of the run's
        vectors in memory, and write a new pack when the run's set was not
        already covered. ``check_progress_cb(done, total)`` reports the
        existing-vector check, which reads one file per image not in a pack.

        Returns ``{"built": n, "reused": n, "total": n}``.
        """
        previous_meta = self._read_meta()
        audit_existing = not (
            previous_meta.get("version") == EMBED_CACHE_VERSION
            and previous_meta.get("backbone_hash") == self.backbone_hash
            and previous_meta.get("pooled_dim") == self.pooled_dim
            and previous_meta.get("preproc_sig") == self.preproc_sig
            and previous_meta.get("complete") is True
        )
        # Establishing this era invalidates any other backbone-hash era under the
        # same cache root — reclaim them now so the cache can't accumulate dead
        # vectors for hashes that will never recur.
        if prune:
            self.prune_other_eras()
        backbone.eval()

        by_hash: dict[str, dict] = {}
        for s in samples:
            h = self._hash(s["path"], smart_cache)
            if h is None:
                continue
            by_hash.setdefault(h, s)

        total = len(by_hash)
        self._ensure_pack_loaded()
        missing: list[tuple[str, dict]] = []
        last_check_report = 0.0
        # Vectors validated when the pack was read (or loaded / built by this
        # process) need no per-image read. The rest are read on io_workers
        # threads -- one small file each, ~9x faster in parallel on a cold NTFS
        # cache (measured, ISSUE-1110) -- and consumed in order, so the outcome
        # is the serial audit's.
        skip = {h for h in by_hash if self.packed and h in self._mem}
        to_read = [h for h in by_hash if h not in skip]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, io_workers)) as pool:
            audited = pool.map(self._audit_vector, to_read)
            for checked, (content_hash, sample) in enumerate(by_hash.items(), start=1):
                if check_progress_cb is not None:
                    now = time.monotonic()
                    if checked == total or now - last_check_report >= 0.25:
                        last_check_report = now
                        check_progress_cb(checked, total)
                if content_hash in skip:
                    continue
                exists, vector = next(audited)
                vector_path = self._vec_path(content_hash)
                if not exists:
                    missing.append((content_hash, sample))
                    continue
                # Validate every vector referenced by this run, even when the era
                # marker says complete. The root is shared across concepts: a
                # different concept may have set the era complete after a crash
                # without referencing (and therefore auditing) this file.
                if vector is None:
                    logger.warning(
                        "EmbeddingCache: rebuilding invalid vector left by an "
                        "interrupted cache build: %s",
                        vector_path,
                    )
                    try:
                        vector_path.unlink()
                    except FileNotFoundError:
                        pass
                    missing.append((content_hash, sample))
                elif self.packed:
                    self._mem[content_hash] = np.ascontiguousarray(vector, dtype=np.float32)
        reused = total - len(missing)
        if not missing:
            if audit_existing:
                self._write_meta(complete=True)
            self._maybe_write_run_pack(list(by_hash))
            return {"built": 0, "reused": reused, "total": total}

        # This marker reaches disk before any vector write. A hard reset then
        # causes the next run to audit existing files rather than trusting them.
        self._write_meta(complete=False)

        by_bucket: dict[tuple, list] = defaultdict(list)
        for h, s in missing:
            by_bucket[tuple(s["bucket"])].append((h, s))
        batches: list[list[tuple[str, dict]]] = []
        for items in by_bucket.values():
            for start in range(0, len(items), batch_size):
                batches.append(items[start : start + batch_size])

        built = 0
        save_futures: list[concurrent.futures.Future] = []
        batch_iter = iter(batches)
        with (
            concurrent.futures.ThreadPoolExecutor(max_workers=io_workers) as loader,
            concurrent.futures.ThreadPoolExecutor(max_workers=1) as saver,
        ):
            # Sliding window of batches whose per-image loads are in flight.
            window: deque[list[tuple[str, str, concurrent.futures.Future]]] = deque()

            def _admit() -> None:
                chunk = next(batch_iter, None)
                if chunk is None:
                    return
                window.append(
                    [
                        (h, s["path"], loader.submit(self._load_input_tensor, s, smart_cache))
                        for h, s in chunk
                    ]
                )

            for _ in range(prefetch_batches):
                _admit()
            while window:
                if stop_check is not None and stop_check():
                    for inflight in window:
                        for _h, _p, fut in inflight:
                            fut.cancel()
                    break
                inflight = window.popleft()
                _admit()
                tensors, hashes, sample_paths = [], [], []
                for h, path, fut in inflight:
                    t = fut.result()
                    if t is None:
                        logger.warning(
                            "EmbeddingCache: could not load input for %s", path
                        )
                        continue
                    tensors.append(t)
                    hashes.append(h)
                    sample_paths.append(path)
                if not tensors:
                    continue
                # Store at float32: a lossless capture of the pooled vector the
                # backbone produced (under autocast), so the probe trains on the
                # exact features without an extra fp16 quantisation step.
                vecs = self._forward_pooled_capped(tensors, backbone, device, dtype)
                safe_vecs: list[np.ndarray] = []
                for index, tensor in enumerate(tensors):
                    vector = vecs[index] if index < len(vecs) else None
                    if vector is None or not self._valid_vector(vector):
                        logger.warning(
                            "EmbeddingCache: invalid batched feature for %s; "
                            "retrying that image independently",
                            sample_paths[index],
                        )
                        retry = self._forward_pooled(
                            tensor.unsqueeze(0), backbone, device, dtype
                        )[0]
                        if not self._valid_vector(retry):
                            raise EmbeddingCacheMismatch(
                                "Backbone produced an invalid pooled feature for "
                                f"'{sample_paths[index]}' both in a batch and "
                                "when retried independently; no cache file was written."
                            )
                        vector = retry
                    safe_vecs.append(np.ascontiguousarray(vector, dtype=np.float32))
                if self.packed:
                    for content_hash, vector in zip(hashes, safe_vecs):
                        self._mem[content_hash] = vector
                save_futures.append(
                    saver.submit(self._save_many, list(zip(hashes, safe_vecs)))
                )
                built += len(hashes)
                if progress_cb is not None:
                    progress_cb(built, len(missing))
        # Exiting the pools joined every in-flight load and write. Surface the
        # first write failure BEFORE the completeness check — the era must
        # never be marked complete over vectors that never reached disk.
        for fut in save_futures:
            fut.result()

        if built == len(missing):
            self._write_meta(complete=True)
            self._maybe_write_run_pack(list(by_hash))
        return {"built": built, "reused": reused, "total": total}

    def _maybe_write_run_pack(self, run_hashes: list[str]) -> None:
        """Packed caches: write a pack of exactly the run's set unless the
        newest pack already covers it. Bounded by the current dataset — a
        hash that left the group drops out at the next rewrite."""
        if not self.packed or not run_hashes:
            return
        if self._pack_hashes.issuperset(run_hashes):
            return
        if any(h not in self._mem for h in run_hashes):
            return
        self._write_pack(run_hashes)

    def get_vector(self, path: str, smart_cache: Any | None) -> np.ndarray | None:
        h = self._hash(path, smart_cache)
        if h is None:
            return None
        if self.packed:
            self._ensure_pack_loaded()
            vector = self._mem.get(h)
            if vector is not None:
                return vector
        p = self._vec_path(h)
        if not p.is_file():
            return None
        vector = self._load_vector_path(p)
        if vector is not None and self.packed:
            vector = np.ascontiguousarray(vector, dtype=np.float32)
            self._mem[h] = vector
        return vector

    def vectors_for(
        self,
        samples: list[dict],
        smart_cache: Any | None,
        *,
        progress_cb: Callable[[int, int], None] | None = None,
        io_workers: int = 8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pooled vectors for *samples*, one row per sample (ISSUE-1110).

        Returns ``(matrix, present)``: a float32 ``[len(samples), pooled_dim]``
        array (zero rows where absent) and a bool mask of the rows that have a
        vector. Each distinct image is read once however often it repeats
        (the replication-expanded train schedule repeats rows); vectors come
        from memory / the pack first, then from per-image files on a thread
        pool. ``progress_cb(done, total)`` counts distinct images.
        """
        hashes = [self._hash(s["path"], smart_cache) for s in samples]
        unique: list[str] = list(dict.fromkeys(h for h in hashes if h is not None))
        if self.packed:
            self._ensure_pack_loaded()
        found: dict[str, np.ndarray] = {}
        pending: list[str] = []
        for h in unique:
            vector = self._mem.get(h)
            if vector is not None:
                found[h] = vector
            else:
                pending.append(h)
        total = len(unique)
        done = len(found)
        if progress_cb is not None:
            progress_cb(done, total)
        if pending:
            last_report = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, io_workers)) as pool:
                for h, vector in zip(
                    pending, pool.map(self._load_hash, pending)
                ):
                    done += 1
                    if vector is not None:
                        vector = np.ascontiguousarray(vector, dtype=np.float32)
                        found[h] = vector
                        if self.packed:
                            self._mem[h] = vector
                    now = time.monotonic()
                    if progress_cb is not None and (done == total or now - last_report >= 0.25):
                        last_report = now
                        progress_cb(done, total)
        matrix = np.zeros((len(samples), self.pooled_dim), dtype=np.float32)
        present = np.zeros(len(samples), dtype=bool)
        for row, h in enumerate(hashes):
            vector = found.get(h) if h is not None else None
            if vector is not None:
                matrix[row] = vector
                present[row] = True
        return matrix, present

    def _audit_vector(self, content_hash: str) -> tuple[bool, np.ndarray | None]:
        """``(file exists, validated vector or None)`` for one content hash."""
        path = self._vec_path(content_hash)
        if not path.is_file():
            return False, None
        return True, self._load_vector_path(path)

    def _load_hash(self, content_hash: str) -> np.ndarray | None:
        p = self._vec_path(content_hash)
        if not p.is_file():
            return None
        return self._load_vector_path(p)

    def has_vector(self, path: str, smart_cache: Any | None) -> bool:
        h = self._hash(path, smart_cache)
        if h is None:
            return False
        if self.packed:
            self._ensure_pack_loaded()
            if h in self._mem:
                return True
        return self._vec_path(h).is_file()

    def verify(
        self,
        samples: list[dict],
        backbone: nn.Module,
        smart_cache: Any | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
        sample_n: int = 32,
        rel_tol: float = 0.15,
        rng_seed: int = 0,
    ) -> int:
        """Assert cached vectors match a fresh forward; raise on mismatch.

        Returns the number of vectors checked. Runs before any probe consumes
        the cache — a silently-bad cache would waste hours of probe time, so a
        divergence raises :class:`EmbeddingCacheMismatch` rather than warning.

        Comparison is by **relative L2 error** (``||live - cached|| / ||live||``),
        not element-wise tolerance. Under bf16 the batched build forward and this
        single-image forward use different kernel tilings, so a handful of
        elements can differ by ~1e-2 even when the cache is correct; the
        whole-vector relative error stays a few percent. A genuinely stale cache
        (wrong preprocessing / a changed cache point / corruption) diverges by
        tens of percent or more — well clear of ``rel_tol`` — so this catches
        real staleness without false-positiving on bf16 numerics.
        """
        backbone.eval()
        # Sample candidates progressively instead of filtering the full corpus —
        # a stat() per sample over tens of thousands of files took minutes and
        # only ever fed a 32-vector check.
        rng = random.Random(rng_seed)
        order = list(range(len(samples)))
        rng.shuffle(order)
        picks: list[dict] = []
        for idx in order:
            s = samples[idx]
            if self.has_vector(s["path"], smart_cache):
                picks.append(s)
                if len(picks) >= sample_n:
                    break
        if not picks:
            raise EmbeddingCacheMismatch(
                "No cached embeddings found to verify — the cache is empty or "
                "the backbone hash does not match any stored era."
            )
        checked = 0
        worst = 0.0
        for s in picks:
            t = self._load_input_tensor(s, smart_cache)
            if t is None:
                continue
            live = self._forward_pooled(t.unsqueeze(0), backbone, device, dtype)[0]
            cached_vec = self.get_vector(s["path"], smart_cache)
            if cached_vec is None:
                raise EmbeddingCacheMismatch(
                    f"Cached embedding for '{s['path']}' is unreadable, has the "
                    "wrong shape, or contains zero/non-finite data. An incomplete "
                    "era will be audited and repaired by ensure() on the next run."
                )
            cached_vec = cached_vec.astype(np.float32)
            denom = float(np.linalg.norm(live)) or 1.0
            rel_err = float(np.linalg.norm(live - cached_vec)) / denom
            worst = max(worst, rel_err)
            if rel_err > rel_tol:
                raise EmbeddingCacheMismatch(
                    f"Cached embedding for '{s['path']}' diverges from a live forward "
                    f"(relative L2 error {rel_err:.3f} > {rel_tol}). The cache is stale, "
                    f"was built with different preprocessing, or predates a cache-point "
                    f"change — delete the embedding cache directory to rebuild."
                )
            checked += 1
        if checked == 0:
            raise EmbeddingCacheMismatch(
                "Verification could not load any input tensors"
            )
        logger.info(
            "EmbeddingCache.verify: %d vectors OK (worst relative L2 %.3f)",
            checked,
            worst,
        )
        return checked
