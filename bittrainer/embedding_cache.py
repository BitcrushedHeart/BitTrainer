"""Cached pooled backbone features for the head probe.

Stores one post-norm pooled vector per image, computed once per backbone era.
Layered on top of :class:`bittrainer.smart_cache.SmartCache` (which caches input
image tensors): embeddings are keyed by the backbone-feature hash (directory
namespace) plus the image's content hash (filename). A fine-tuned backbone
produces a different feature hash, so it gets a fresh namespace and never
collides with the pretrained-era cache — re-running the probe after a full
fine-tune rebuilds against the adapted features automatically.

The cache point is ``flatten(norm(global_pool(features)))`` (pre pre_logits/fc),
computed by :func:`bittrainer.model.pooled_features` with the backbone in
``eval()`` so the forward is deterministic (no drop_path, no head dropout).
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from bittrainer.gpu_augment import apply_val_transform
from bittrainer.model import pooled_features

logger = logging.getLogger(__name__)

EMBED_CACHE_VERSION = 1


class EmbeddingCacheMismatch(RuntimeError):
    """A cached embedding disagrees with a fresh backbone forward (fail loud)."""


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
    ) -> None:
        self.root = Path(cache_dir) / backbone_hash
        self.backbone_hash = backbone_hash
        self.pooled_dim = int(pooled_dim)
        self.preproc_sig = preproc_sig
        self._meta_path = self.root / "meta.json"
        # ensure() -> verify() -> probe gather all hash the same corpus; memoise
        # so only the first pass pays the per-path index lookups.
        self._hash_memo: dict[str, str | None] = {}

    def _hash(self, path: str, smart_cache: Any | None) -> str | None:
        if path in self._hash_memo:
            return self._hash_memo[path]
        h = _content_hash(path, smart_cache)
        self._hash_memo[path] = h
        return h

    def _vec_path(self, content_hash: str) -> Path:
        return self.root / f"{content_hash}.npy"

    def _write_meta(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._meta_path.write_text(json.dumps({
            "version": EMBED_CACHE_VERSION,
            "backbone_hash": self.backbone_hash,
            "pooled_dim": self.pooled_dim,
            "preproc_sig": self.preproc_sig,
        }))

    def _load_input_tensor(self, sample: dict, smart_cache: Any | None) -> torch.Tensor | None:
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
        self, batch: torch.Tensor, backbone: nn.Module,
        device: torch.device, dtype: torch.dtype,
    ) -> np.ndarray:
        batch = batch.to(device)
        batch = apply_val_transform(batch, dtype=dtype)
        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)
        ):
            vecs = pooled_features(backbone, batch)
        return vecs.float().cpu().numpy()

    def ensure(
        self,
        samples: list[dict],
        backbone: nn.Module,
        smart_cache: Any | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int = 64,
        progress_cb: Callable[[int, int], None] | None = None,
        stop_check: Callable[[], bool] | None = None,
    ) -> dict:
        """Build any missing pooled vectors for *samples* under this backbone era.

        Returns ``{"built": n, "reused": n, "total": n}``.
        """
        self._write_meta()
        backbone.eval()

        by_hash: dict[str, dict] = {}
        for s in samples:
            h = self._hash(s["path"], smart_cache)
            if h is None:
                continue
            by_hash.setdefault(h, s)

        total = len(by_hash)
        missing = [(h, s) for h, s in by_hash.items() if not self._vec_path(h).is_file()]
        reused = total - len(missing)
        if not missing:
            return {"built": 0, "reused": reused, "total": total}

        by_bucket: dict[tuple, list] = defaultdict(list)
        for h, s in missing:
            by_bucket[tuple(s["bucket"])].append((h, s))

        built = 0
        for items in by_bucket.values():
            for start in range(0, len(items), batch_size):
                if stop_check is not None and stop_check():
                    return {"built": built, "reused": reused, "total": total}
                chunk = items[start:start + batch_size]
                tensors, hashes = [], []
                for h, s in chunk:
                    t = self._load_input_tensor(s, smart_cache)
                    if t is None:
                        logger.warning("EmbeddingCache: could not load input for %s", s["path"])
                        continue
                    tensors.append(t)
                    hashes.append(h)
                if not tensors:
                    continue
                # Store at float32: a lossless capture of the pooled vector the
                # backbone produced (under autocast), so the probe trains on the
                # exact features without an extra fp16 quantisation step.
                vecs = self._forward_pooled(torch.stack(tensors), backbone, device, dtype)
                vecs = np.ascontiguousarray(vecs, dtype=np.float32)
                for h, v in zip(hashes, vecs):
                    np.save(self._vec_path(h), v)
                built += len(hashes)
                if progress_cb is not None:
                    progress_cb(built, len(missing))

        return {"built": built, "reused": reused, "total": total}

    def get_vector(self, path: str, smart_cache: Any | None) -> np.ndarray | None:
        h = self._hash(path, smart_cache)
        if h is None:
            return None
        p = self._vec_path(h)
        if not p.is_file():
            return None
        return np.load(p)

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
            h = self._hash(s["path"], smart_cache)
            if h is not None and self._vec_path(h).is_file():
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
                continue
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
            raise EmbeddingCacheMismatch("Verification could not load any input tensors")
        logger.info("EmbeddingCache.verify: %d vectors OK (worst relative L2 %.3f)", checked, worst)
        return checked
