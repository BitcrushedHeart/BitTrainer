"""Scoped cache export + paranoid handshake for cloud group training.

The training cache (``cache.json`` + ``*.pt``) is **global** — one index for the
whole corpus. Shipping it whole to a pod and training sourceless would train on
the entire corpus, not the selected group. This module builds a *scoped* export
containing only one group's train+val entries plus copies of only their
referenced ``.pt`` files, with hard assertions so a quiet off-by-one can't ship
the wrong data.

Two layers, deliberately split by dependency weight:

* The scoping/handshake core (``build_scoped_export`` / ``verify_scoped_cache``)
  is pure stdlib (json/hashlib/shutil) — no torch — so it imports fast and runs
  on the pod and under unit tests. It takes the authoritative sample list as
  input.
* ``warm_group_cache`` builds that authoritative list by constructing the exact
  same ``GroupDataset`` objects the trainer uses (so the export can never drift
  from what training would read) and runs ``SmartCache.prepare``. It lazily
  imports the heavy bits.

Source of truth for "this group's dataset" is ``GroupDataset.samples`` (disk +
the caller-injected ``extra_paths`` for the never-copied ``__none__`` negatives),
i.e. exactly what ``run_group_training`` caches. The backend additionally passes
``db_expected_paths`` so a disk-vs-DB drift fails loudly at export time.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# Mirrors bittrainer.smart_cache (kept torch-free here on purpose). warm_group_cache
# asserts these against the live SmartCache so a format bump fails loudly.
_CACHE_VERSION = 2
_CACHE_INDEX_FILENAME = "cache.json"


class ScopedExportError(Exception):
    """A scoped export failed a hard correctness assertion. Never ship anyway."""


@dataclass(frozen=True)
class SampleRef:
    path: str  # os.path.normpath, matching the cache.json entry key
    split: str  # "train" | "val"
    label: Any = None
    concept_name: str = ""


@dataclass
class ScopedExportManifest:
    group_name: str
    safe_name: str
    modeltype: str
    num_classes: int
    class_names: list[str]
    expected_entry_count: int
    train_count: int
    val_count: int
    entry_paths: list[str]
    pt_files: list[str]
    cache_json_sha256: str
    pt_sha256: dict[str, str] = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ScopedExportManifest":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in fields})

    def write_json(self, path: str | os.PathLike) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def read_json(cls, path: str | os.PathLike) -> "ScopedExportManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _pt_filename(cache_file: str) -> str:
    """The single ``.pt`` for a cache entry (bucket/variation 0). Mirrors
    ``smart_cache._pt_filename(cache_file, 0)``."""
    return f"{cache_file}_1.pt"


def _sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_global_index(global_cache_dir: str | os.PathLike) -> dict:
    index_path = Path(global_cache_dir) / _CACHE_INDEX_FILENAME
    if not index_path.is_file():
        raise ScopedExportError(f"global cache index not found: {index_path}")
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ScopedExportError(f"global cache index unreadable: {exc}") from exc


def _scoped_index_bytes(entries: dict[str, dict]) -> bytes:
    """Deterministic bytes for a scoped cache.json so its sha is stable across
    the transfer (sender and receiver hash identical bytes)."""
    hash_index: dict[str, list[str]] = {}
    for path, entry in entries.items():
        h = entry.get("hash")
        if h:
            hash_index.setdefault(h, []).append(path)
    for paths in hash_index.values():
        paths.sort()
    index = {
        "version": _CACHE_VERSION,
        "entries": entries,
        "hash_index": hash_index,
        "last_validated": 0.0,
    }
    return json.dumps(index, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_scoped_export(
    *,
    global_cache_dir: str | os.PathLike,
    export_dir: str | os.PathLike,
    samples: Sequence[SampleRef],
    modeltype: str,
    num_classes: int,
    class_names: list[str],
    group_name: str = "",
    safe_name: str = "",
    config: dict | None = None,
    db_expected_paths: set[str] | None = None,
    hash_pt: bool = True,
) -> ScopedExportManifest:
    """Build a scoped cache export under ``export_dir/cache/`` and return its
    manifest. Raises ``ScopedExportError`` on any correctness violation.

    ``samples`` must already be cached in the global cache (run
    :func:`warm_group_cache` first); a missing entry is a hard error.
    """
    export_dir = Path(export_dir)
    cache_out = export_dir / "cache"
    cache_out.mkdir(parents=True, exist_ok=True)

    # Authoritative path -> split (dedup by normpath, which is how cache.json keys).
    auth: dict[str, str] = {}
    for s in samples:
        auth[os.path.normpath(s.path)] = s.split
    auth_paths = set(auth)
    if not auth_paths:
        raise ScopedExportError("no samples to export")

    if db_expected_paths is not None:
        dbset = {os.path.normpath(p) for p in db_expected_paths}
        if dbset != auth_paths:
            only_disk = sorted(auth_paths - dbset)[:5]
            only_db = sorted(dbset - auth_paths)[:5]
            raise ScopedExportError(
                "disk/DB drift: "
                f"{len(auth_paths - dbset)} sample(s) on disk not in DB (e.g. {only_disk}); "
                f"{len(dbset - auth_paths)} in DB not on disk (e.g. {only_db})"
            )

    global_entries = _read_global_index(global_cache_dir).get("entries", {})

    scoped_entries: dict[str, dict] = {}
    not_cached: list[str] = []
    for path in sorted(auth_paths):
        entry = global_entries.get(path)
        if entry is None:
            not_cached.append(path)
            continue
        scoped_entries[path] = entry
    if not_cached:
        raise ScopedExportError(
            f"{len(not_cached)} sample(s) not in the global cache — cache first "
            f"(e.g. {not_cached[:5]})"
        )

    for path, entry in scoped_entries.items():
        if entry.get("modeltype") != modeltype:
            raise ScopedExportError(
                f"modeltype mismatch for {path}: cached {entry.get('modeltype')!r} != {modeltype!r}"
            )
        if entry.get("cache_version", 0) != _CACHE_VERSION:
            raise ScopedExportError(
                f"cache_version mismatch for {path}: {entry.get('cache_version')} != {_CACHE_VERSION}"
            )
        cached_split = entry.get("split")
        if cached_split != auth[path]:
            raise ScopedExportError(
                f"split mismatch for {path}: cache {cached_split!r} != expected {auth[path]!r}"
            )

    pt_files = sorted({_pt_filename(e["cache_file"]) for e in scoped_entries.values()})

    pt_sha: dict[str, str] = {}
    for pt_name in pt_files:
        src = Path(global_cache_dir) / pt_name
        if not src.is_file():
            raise ScopedExportError(f"referenced cache file missing in global cache: {pt_name}")
        shutil.copy2(src, cache_out / pt_name)
        if hash_pt:
            pt_sha[pt_name] = _sha256_file(cache_out / pt_name)

    index_bytes = _scoped_index_bytes(scoped_entries)
    cache_json_path = cache_out / _CACHE_INDEX_FILENAME
    cache_json_path.write_bytes(index_bytes)
    cache_json_sha = hashlib.sha256(index_bytes).hexdigest()

    train_count = sum(1 for e in scoped_entries.values() if e.get("split") == "train")
    val_count = sum(1 for e in scoped_entries.values() if e.get("split") == "val")

    # --- Hard assertions (non-negotiable) ---
    if len(scoped_entries) != len(auth_paths):
        raise ScopedExportError(
            f"entry count {len(scoped_entries)} != expected sample count {len(auth_paths)}"
        )
    if set(scoped_entries) != auth_paths:
        raise ScopedExportError("scoped entries are not exactly the authoritative sample set")
    for pt_name in pt_files:
        if not (cache_out / pt_name).is_file():
            raise ScopedExportError(f"export missing copied cache file: {pt_name}")
    if train_count < 1 or val_count < 1:
        raise ScopedExportError(
            f"eval/train split missing: train={train_count}, val={val_count}"
        )

    manifest = ScopedExportManifest(
        group_name=group_name,
        safe_name=safe_name,
        modeltype=modeltype,
        num_classes=num_classes,
        class_names=list(class_names),
        expected_entry_count=len(scoped_entries),
        train_count=train_count,
        val_count=val_count,
        entry_paths=sorted(scoped_entries),
        pt_files=pt_files,
        cache_json_sha256=cache_json_sha,
        pt_sha256=pt_sha,
        config=dict(config or {}),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    manifest.write_json(export_dir / "manifest.json")
    return manifest


def verify_scoped_cache(
    cache_dir: str | os.PathLike,
    manifest: ScopedExportManifest,
    *,
    check_pt_hashes: bool = True,
) -> tuple[bool, str, str]:
    """Paranoid handshake: verify a received/staged scoped cache against its
    manifest by **count and identity**, not mere presence.

    Returns ``(ok, reason_key, message)``. ``reason_key`` is ``""`` on success
    and ``"DATA_ERROR"`` on any mismatch (kept as a bare string so this module
    stays free of the orchestrator's enum).
    """
    cache_dir = Path(cache_dir)
    cache_json = cache_dir / _CACHE_INDEX_FILENAME
    if not cache_json.is_file():
        return False, "DATA_ERROR", f"cache.json missing in {cache_dir}"
    try:
        index = json.loads(cache_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, "DATA_ERROR", f"cache.json unreadable: {exc}"

    entries = index.get("entries", {})
    if len(entries) != manifest.expected_entry_count:
        return False, "DATA_ERROR", (
            f"entry count {len(entries)} != expected {manifest.expected_entry_count}"
        )

    got = set(entries)
    want = set(manifest.entry_paths)
    if got != want:
        return False, "DATA_ERROR", (
            f"entry-path mismatch: {len(want - got)} missing, {len(got - want)} extra"
        )

    try:
        referenced = sorted({_pt_filename(e["cache_file"]) for e in entries.values()})
    except KeyError:
        return False, "DATA_ERROR", "an entry is missing its cache_file field"
    if referenced != sorted(manifest.pt_files):
        return False, "DATA_ERROR", "referenced .pt set does not match the manifest"

    for pt_name in manifest.pt_files:
        if not (cache_dir / pt_name).is_file():
            return False, "DATA_ERROR", f"missing cache file: {pt_name}"

    train_count = sum(1 for e in entries.values() if e.get("split") == "train")
    val_count = sum(1 for e in entries.values() if e.get("split") == "val")
    if train_count < 1 or val_count < 1:
        return False, "DATA_ERROR", f"split missing: train={train_count}, val={val_count}"

    if check_pt_hashes and manifest.pt_sha256:
        if _sha256_file(cache_json) != manifest.cache_json_sha256:
            return False, "DATA_ERROR", "cache.json sha256 mismatch (corrupt/partial transfer)"
        for pt_name, want_sha in manifest.pt_sha256.items():
            if _sha256_file(cache_dir / pt_name) != want_sha:
                return False, "DATA_ERROR", f"sha256 mismatch for {pt_name} (corrupt/partial transfer)"

    return True, "", "ok"


def warm_group_cache(
    *,
    group_folder: str | os.PathLike,
    class_names: list[str],
    global_cache_dir: str | os.PathLike,
    extra_paths_train: dict[str, list[str]] | None = None,
    extra_paths_val: dict[str, list[str]] | None = None,
    skin_normalise: bool = False,
    face_model_path: str = "",
    multi_label: bool = False,
    oversample_none: bool = False,
    modeltype: str = "convnext_v2",
    group_name: str = "",
    cache_workers: int = 10,
    device: str = "cuda",
    progress_callback=None,
) -> list[SampleRef]:
    """Cache a group's train+val tensors into the global cache and return the
    authoritative sample list.

    Builds the exact ``GroupDataset`` objects ``run_group_training`` uses, so the
    cached set (and therefore the scoped export) can never drift from what
    training would read. Idempotent: already-cached tensors are skipped by
    ``SmartCache.prepare``. This is the CPU-bound preprocessing the
    ``Cache Now`` / ``Cache Locally First`` flows run while the local GPU is
    free.
    """
    from bittrainer.cache_builders import build_image_tensor
    from bittrainer.group_dataset import GroupDataset
    from bittrainer.smart_cache import (
        CACHE_VERSION,
        SmartCache,
        _noop_callback,
        face_model_signature,
    )

    if CACHE_VERSION != _CACHE_VERSION:
        raise ScopedExportError(
            f"cloud_export is built for cache_version {_CACHE_VERSION} but SmartCache "
            f"is {CACHE_VERSION}; update bittrainer.cloud_export"
        )

    cb = progress_callback or _noop_callback
    group_folder = Path(group_folder)
    group_name = group_name or group_folder.name

    cache = SmartCache(
        Path(global_cache_dir),
        modeltype=modeltype,
        progress_callback=cb,
        face_model_sig=face_model_signature(face_model_path or None),
    )

    train_ds = GroupDataset(
        group_folder, class_names, split="train",
        multi_label=multi_label,
        skin_normalise=skin_normalise, group_name=group_name,
        oversample_none=oversample_none,
        extra_paths=extra_paths_train or {},
    )
    val_ds = GroupDataset(
        group_folder, class_names, split="val",
        multi_label=multi_label,
        skin_normalise=skin_normalise, group_name=group_name,
        extra_paths=extra_paths_val or {},
    )

    if face_model_path:
        from bittrainer.face_crop import FaceBBoxCache, precompute_face_bboxes

        face_cache = FaceBBoxCache(group_folder / ".resize_cache" / "face_bboxes.json")
        all_image_paths = [s["path"] for s in train_ds.samples] + [s["path"] for s in val_ds.samples]
        precompute_face_bboxes(all_image_paths, face_cache, face_model_path, device=device)
        face_bboxes = {p: face_cache.get(p) for p in all_image_paths if face_cache.get(p)}
        train_ds.refresh_face_bboxes(face_bboxes)
        val_ds.refresh_face_bboxes(face_bboxes)

    cache.prepare(
        train_ds.samples + val_ds.samples, build_image_tensor,
        num_workers=cache_workers, stage_label="caching",
    )

    refs: list[SampleRef] = []
    for ds, split in ((train_ds, "train"), (val_ds, "val")):
        for s in ds.samples:
            refs.append(
                SampleRef(
                    path=os.path.normpath(s["path"]),
                    split=split,
                    label=s.get("label"),
                    concept_name=s.get("concept_name", ""),
                )
            )
    return refs
