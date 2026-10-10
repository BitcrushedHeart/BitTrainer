"""Pair manifests, the leakage-free split and cached embeddings for pairwise
reward training (Bitcrush Optimise P8, ISSUE-1133).

Optimise exports its user-confirmed pairs as a manifest::

    {"pairs": [{"chosen": <path>, "rejected": <path>, "caption": <chosen caption>,
                "group_id": <caption group>, "type": "normal"|"offpolicy"|"minor_edits",
                "identities": [<extra identity keys, e.g. content / source hashes>]}, ...],
     ...}

The trunks are frozen, so training works on cached embeddings: one pooled
SigLIP2 image vector per distinct image (keyed by content hash + preprocessing
signature) and one text vector per distinct caption (keyed by text model +
caption). :class:`PairEmbeddingDataset` then hands the head ``(i_chosen,
i_rejected, t)`` tuples, one per pair, so every confirmed pair carries equal
weight whatever its group or dataset type.

:func:`group_aware_split` holds out WHOLE connected components: pairs linked by
a caption group, a shared image, or a shared source identity (an Off-Policy
pair's real photo, a Minor Edits pair's lineage) land on one side only, so
"held-out preference accuracy" is never train-set agreement in disguise. The
split is persisted (:func:`save_split`) so a re-run or a later comparison
evaluates on the same pairs.

:func:`vae_roundtrip_prepare` is the optional VAE round-trip preparation: it
applies a caller-supplied encode/decode to BOTH sides of every pair into
training copies and records the VAE identity and settings; the originals are
never touched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA = 1
SPLIT_SCHEMA = 1
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Manifest                                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PairRecord:
    chosen: str
    rejected: str
    caption: str
    group_id: str
    pair_type: str = "normal"
    #: Extra identity keys that tie pairs together for the split (content
    #: hashes, Off-Policy source hashes, lineage) — never read as paths.
    identities: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "chosen": self.chosen,
            "rejected": self.rejected,
            "caption": self.caption,
            "group_id": self.group_id,
            "type": self.pair_type,
            "identities": list(self.identities),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any], index: int) -> PairRecord:
        for key in ("chosen", "rejected", "caption", "group_id"):
            if key not in data or data[key] in (None, ""):
                raise ValueError(f"pair {index} is missing {key!r}")
        return cls(
            chosen=str(data["chosen"]),
            rejected=str(data["rejected"]),
            caption=str(data["caption"]),
            group_id=str(data["group_id"]),
            pair_type=str(data.get("type") or "normal"),
            identities=tuple(str(x) for x in data.get("identities") or ()),
        )


def load_pair_manifest(path: str | Path) -> list[PairRecord]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs = data.get("pairs") if isinstance(data, dict) else None
    if not isinstance(pairs, list):
        raise ValueError(f"{path}: a pair manifest is an object with a 'pairs' list")
    if not pairs:
        raise ValueError(f"{path}: the manifest has no pairs")
    return [PairRecord.from_json(p, i) for i, p in enumerate(pairs)]


def write_pair_manifest(
    path: str | Path, records: Sequence[PairRecord], extra: dict[str, Any] | None = None
) -> None:
    payload: dict[str, Any] = {"schema": MANIFEST_SCHEMA, **(extra or {})}
    payload["pairs"] = [r.to_json() for r in records]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def per_type_counts(records: Iterable[PairRecord]) -> dict[str, int]:
    return dict(sorted(Counter(r.pair_type for r in records).items()))


# --------------------------------------------------------------------------- #
# Split                                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class SplitResult:
    train: list[int]
    val: list[int]
    components: int
    val_fraction: float
    seed: int

    @property
    def held_out(self) -> bool:
        """True when a non-empty validation set exists that shares nothing with train."""
        return bool(self.val)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _components(records: Sequence[PairRecord]) -> list[list[int]]:
    """Connected components over group id, image path and identity keys."""
    uf = _UnionFind(len(records))
    first_seen: dict[str, int] = {}
    for index, record in enumerate(records):
        keys = [f"group:{record.group_id}", f"image:{record.chosen}", f"image:{record.rejected}"]
        keys.extend(f"identity:{x}" for x in record.identities)
        for key in keys:
            if key in first_seen:
                uf.union(first_seen[key], index)
            else:
                first_seen[key] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(records)):
        groups.setdefault(uf.find(index), []).append(index)
    return [sorted(members) for _, members in sorted(groups.items())]


def _component_key(records: Sequence[PairRecord], members: list[int]) -> str:
    """A content-stable label for a component (its sorted group ids + images)."""
    parts = sorted({records[i].group_id for i in members})
    parts += sorted({records[i].chosen for i in members} | {records[i].rejected for i in members})
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()


def group_aware_split(
    records: Sequence[PairRecord], *, val_fraction: float = 0.2, seed: int = 0
) -> SplitResult:
    """Hold out whole components until ``val_fraction`` of the pairs is reached.

    Deterministic for ``(records, seed)``: components are ordered by a
    content-stable key, shuffled with the seed, and taken greedily. With a
    single component there is nothing to hold out (``held_out`` is False); with
    two or more, at least one component stays on each side.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    components = _components(records)
    if len(components) < 2 or val_fraction == 0.0:
        return SplitResult(
            train=list(range(len(records))),
            val=[],
            components=len(components),
            val_fraction=val_fraction,
            seed=seed,
        )
    order = sorted(components, key=lambda members: _component_key(records, members))
    random.Random(seed).shuffle(order)
    target = val_fraction * len(records)
    val: list[int] = []
    taken = 0
    for members in order:
        if taken >= len(order) - 1:  # keep at least one component for training
            break
        if val and len(val) + len(members) > target and len(val) >= target * 0.5:
            break
        val.extend(members)
        taken += 1
        if len(val) >= target:
            break
    val_set = set(val)
    train = [i for i in range(len(records)) if i not in val_set]
    return SplitResult(
        train=train,
        val=sorted(val),
        components=len(components),
        val_fraction=val_fraction,
        seed=seed,
    )


def _manifest_digest(records: Sequence[PairRecord]) -> str:
    h = hashlib.sha1()
    for r in records:
        h.update(f"{r.chosen}\0{r.rejected}\0{r.caption}\0{r.group_id}\n".encode())
    return h.hexdigest()


def save_split(path: str | Path, split: SplitResult, records: Sequence[PairRecord]) -> None:
    payload = {
        "schema": SPLIT_SCHEMA,
        "seed": split.seed,
        "val_fraction": split.val_fraction,
        "components": split.components,
        "manifest_digest": _manifest_digest(records),
        "pair_count": len(records),
        "train": split.train,
        "val": split.val,
        "train_groups": sorted({records[i].group_id for i in split.train}),
        "val_groups": sorted({records[i].group_id for i in split.val}),
        "train_type_counts": per_type_counts(records[i] for i in split.train),
        "val_type_counts": per_type_counts(records[i] for i in split.val),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split(path: str | Path, records: Sequence[PairRecord]) -> SplitResult:
    """Reload a persisted split; refuses one made for a different manifest."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("manifest_digest") != _manifest_digest(records) or data.get("pair_count") != len(
        records
    ):
        raise ValueError(f"{path}: the persisted split does not match these pairs")
    return SplitResult(
        train=[int(i) for i in data["train"]],
        val=[int(i) for i in data["val"]],
        components=int(data.get("components", 0)),
        val_fraction=float(data.get("val_fraction", 0.0)),
        seed=int(data.get("seed", 0)),
    )


# --------------------------------------------------------------------------- #
# Embeddings                                                                  #
# --------------------------------------------------------------------------- #


def file_content_hash(path: str | Path) -> str:
    """Content identity for the image embedding cache (BLAKE2b hex, 32 chars)."""
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_key(model_id: str, caption: str) -> str:
    return hashlib.sha1(f"{model_id}\0{caption}".encode()).hexdigest()


class EmbeddingStore:
    """``.npy`` per vector under ``<root>/<namespace>/``; ``None`` when absent."""

    def __init__(self, root: str | Path | None, namespace: str) -> None:
        safe = hashlib.sha1(namespace.encode("utf-8")).hexdigest()[:16]
        self.root = Path(root) / safe if root is not None else None
        self.namespace = namespace
        self._memory: dict[str, np.ndarray] = {}

    def _path(self, key: str) -> Path | None:
        return None if self.root is None else self.root / f"{key}.npy"

    def get(self, key: str) -> np.ndarray | None:
        if key in self._memory:
            return self._memory[key]
        path = self._path(key)
        if path is None or not path.is_file():
            return None
        try:
            vector = np.load(path)
        except (OSError, ValueError):
            return None
        self._memory[key] = vector
        return vector

    def put(self, key: str, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float32)
        self._memory[key] = vector
        path = self._path(key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, vector)
        tmp.replace(path)


class ImageEmbedder:
    """Pooled SigLIP2 image embeddings under one fixed preprocessing.

    Square-resizes to ``image_size`` (bicubic), ImageNet-normalises (the wrapper
    converts to SigLIP's 0.5/0.5 inside) and reads the trunk's pooled vector via
    ``forward_features``. ``preproc_sig`` names the recipe so cached vectors are
    never reused across a change of it.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        image_size: int = 256,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.image_size = int(image_size)
        self.device = torch.device(device)
        self.dtype = dtype
        self.model_size = str(getattr(model, "model_size", "unknown"))
        self.mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)

    @property
    def preproc_sig(self) -> str:
        return f"{self.model_size}:square{self.image_size}:bicubic:imagenet-norm:v1"

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        rgb = image.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    @torch.no_grad()
    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        batch = torch.stack([self._to_tensor(im) for im in images])
        batch = ((batch - self.mean) / self.std).to(self.device, self.dtype)
        pooled = self.model.forward_features(batch)
        return pooled.float().cpu().numpy()

    def embed_paths(self, paths: Sequence[str]) -> np.ndarray:
        images = []
        for path in paths:
            with Image.open(path) as im:
                images.append(im.convert("RGB"))
        return self.embed_images(images)


@dataclass
class PairEmbeddings:
    images: dict[str, np.ndarray]
    texts: dict[str, np.ndarray]
    embed_dim: int
    stats: dict[str, int] = field(default_factory=dict)


def embed_records(
    records: Sequence[PairRecord],
    embedder: ImageEmbedder,
    tower: Any,
    *,
    cache_dir: str | Path | None = None,
    batch_size: int = 32,
    progress: Callable[[int, int, str], None] | None = None,
) -> PairEmbeddings:
    """Image + text embeddings for every distinct image and caption in ``records``.

    Images are cached by content hash under the embedder's ``preproc_sig``,
    captions by ``(text model id, caption)``. ``progress(done, total, what)`` is
    called after every batch.
    """
    image_store = EmbeddingStore(cache_dir, f"image:{embedder.preproc_sig}")
    text_store = EmbeddingStore(cache_dir, f"text:{tower.model_id}:{tower.spec.max_length}")
    paths = sorted({p for r in records for p in (r.chosen, r.rejected)})
    captions = sorted({r.caption for r in records})
    stats = {"images_built": 0, "images_reused": 0, "texts_built": 0, "texts_reused": 0}

    images: dict[str, np.ndarray] = {}
    pending: list[tuple[str, str]] = []
    for path in paths:
        key = file_content_hash(path)
        cached = image_store.get(key)
        if cached is not None:
            images[path] = cached
            stats["images_reused"] += 1
        else:
            pending.append((path, key))
    total = len(pending) + len(captions)
    done = 0
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        vectors = embedder.embed_paths([p for p, _ in chunk])
        for (path, key), vector in zip(chunk, vectors, strict=True):
            image_store.put(key, vector)
            images[path] = np.asarray(vector, dtype=np.float32)
            stats["images_built"] += 1
        done += len(chunk)
        if progress:
            progress(done, total, "images")

    texts: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for caption in captions:
        cached = text_store.get(text_key(tower.model_id, caption))
        if cached is not None:
            texts[caption] = cached
            stats["texts_reused"] += 1
        else:
            missing.append(caption)
    for start in range(0, len(missing), batch_size):
        chunk = missing[start : start + batch_size]
        vectors = tower.encode(chunk).cpu().numpy()
        for caption, vector in zip(chunk, vectors, strict=True):
            text_store.put(text_key(tower.model_id, caption), vector)
            texts[caption] = np.asarray(vector, dtype=np.float32)
            stats["texts_built"] += 1
        done += len(chunk)
        if progress:
            progress(done, total, "captions")
    if progress and total:
        progress(total, total, "done")

    dims = {int(v.shape[-1]) for v in images.values()} | {int(v.shape[-1]) for v in texts.values()}
    if len(dims) != 1:
        raise ValueError(f"image and text embeddings disagree on width: {sorted(dims)}")
    return PairEmbeddings(images=images, texts=texts, embed_dim=dims.pop(), stats=stats)


class PairEmbeddingDataset(Dataset):
    """``(i_chosen, i_rejected, t, type_index)`` per pair from cached embeddings."""

    def __init__(self, records: Sequence[PairRecord], embeddings: PairEmbeddings) -> None:
        self.records = list(records)
        self.type_names = sorted({r.pair_type for r in self.records})
        chosen, rejected, text, types = [], [], [], []
        for record in self.records:
            chosen.append(torch.from_numpy(np.asarray(embeddings.images[record.chosen], dtype=np.float32)))
            rejected.append(torch.from_numpy(np.asarray(embeddings.images[record.rejected], dtype=np.float32)))
            text.append(torch.from_numpy(np.asarray(embeddings.texts[record.caption], dtype=np.float32)))
            types.append(self.type_names.index(record.pair_type))
        if self.records:
            self._chosen = torch.stack(chosen)
            self._rejected = torch.stack(rejected)
            self._text = torch.stack(text)
        else:
            self._chosen = self._rejected = self._text = torch.zeros(0, embeddings.embed_dim)
        self._types = torch.tensor(types, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        return self._chosen[index], self._rejected[index], self._text[index], int(self._types[index])

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "chosen": self._chosen,
            "rejected": self._rejected,
            "text": self._text,
            "types": self._types,
        }

    def subset(self, indices: Sequence[int]) -> dict[str, torch.Tensor]:
        idx = torch.tensor(list(indices), dtype=torch.long)
        return {
            "chosen": self._chosen[idx],
            "rejected": self._rejected[idx],
            "text": self._text[idx],
            "types": self._types[idx],
        }


# --------------------------------------------------------------------------- #
# VAE round-trip preparation                                                  #
# --------------------------------------------------------------------------- #


def vae_roundtrip_prepare(
    records: Sequence[PairRecord],
    roundtrip: Callable[[Image.Image], Image.Image],
    out_dir: str | Path,
    *,
    vae: str,
    version: str,
    settings: dict[str, Any] | None = None,
) -> tuple[list[PairRecord], dict[str, Any]]:
    """Apply one VAE encode/decode to BOTH sides of every pair into training copies.

    ``roundtrip`` is the caller's VAE (encode -> decode) on a PIL image; the
    same callable runs on chosen and rejected so the comparison stays fair.
    Copies land under ``<out_dir>/chosen`` and ``<out_dir>/rejected`` as PNG;
    the originals are left exactly as they were. Returns the rewritten records
    and the provenance Optimise stores as ``vae_prep_json``.
    """
    out = Path(out_dir)
    prepared: list[PairRecord] = []
    for index, record in enumerate(records):
        sides = {}
        for side, source in (("chosen", record.chosen), ("rejected", record.rejected)):
            target = out / side / f"pair_{index:05d}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as im:
                processed = roundtrip(im.convert("RGB"))
            processed.save(target, format="PNG")
            sides[side] = str(target)
        prepared.append(
            PairRecord(
                chosen=sides["chosen"],
                rejected=sides["rejected"],
                caption=record.caption,
                group_id=record.group_id,
                pair_type=record.pair_type,
                identities=record.identities,
            )
        )
    meta = {
        "vae": vae,
        "version": version,
        "settings": dict(settings or {}),
        "pairs": len(prepared),
        "output_dir": str(out),
    }
    return prepared, meta
