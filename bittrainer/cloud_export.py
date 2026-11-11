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

Private names (Bitcrush ISSUE-1690): an export that leaves the machine may not
hold a local path, a file name, the group's name or a class name. Given a
:class:`PrivateNames`, the export writes ids in their place and hands the maps
back for home use; :func:`scoped_export_strings` lists what an export still
says about its samples, classes and group, for the caller's final check. A
sourceless run never reads a sample's path or name: it keys on the index key,
reads label / split from the index and the bucket from the tensor file, so it
trains on ids exactly as on paths. Only the tensor files whose bytes name a
sample (``__source_path``) are rewritten; that is the one place torch is used.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# Mirrors bittrainer.smart_cache (kept torch-free here on purpose). warm_group_cache
# asserts these against the live SmartCache so a format bump fails loudly.
_CACHE_VERSION = 2
_CACHE_INDEX_FILENAME = "cache.json"
# The trainer treats this class name specially; it is never renamed.
_NONE_CLASS = "__none__"


class ScopedExportError(Exception):
    """A scoped export failed a hard correctness assertion. Never ship anyway."""


@dataclass
class PrivateNames:
    """Ids for everything an export would otherwise name (contract C1).

    Pass one to :func:`build_scoped_export`. The export then says
    ``group_alias`` for the group, ``k`` + the class's place in the group's
    class order for a class (``__none__`` stays ``__none__``) and ``s000000``,
    ``s000001`` ... for the samples in the order of their sorted paths. The
    build fills in the two maps, real -> id, for the caller to keep at home;
    they are never written into the export and are valid once the build
    returned.
    """

    group_alias: str
    class_ids: dict[str, str] = field(default_factory=dict)
    sample_ids: dict[str, str] = field(default_factory=dict)


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


def _label_for_storage(label):
    """JSON-storable label (mirrors smart_cache._label_to_storage without the
    torch import — this module stays stdlib-only for pod-side staging tests)."""
    tolist = getattr(label, "tolist", None)
    if callable(tolist):
        return tolist()
    return label


def _assign_private_ids(
    private: PrivateNames,
    *,
    class_names: list[str],
    paths: list[str],
    group_name: str,
    safe_name: str,
) -> None:
    """Fill ``private``'s maps, refusing any name an id would not hide: a
    missing alias, a class listed twice, or a real name that reads like one
    of the ids (the export would then hold that name after all)."""
    alias = private.group_alias
    if not alias or not isinstance(alias, str) or re.search(r"[\\/]", alias):
        raise ScopedExportError(f"the group alias {alias!r} is empty or holds a path")
    if alias in {group_name, safe_name}:
        raise ScopedExportError("the group alias is the group's own name")
    if len(set(class_names)) != len(class_names):
        raise ScopedExportError("a class is listed twice; class ids would not be invertible")
    class_ids = {
        name: name if name == _NONE_CLASS else f"k{index}"
        for index, name in enumerate(class_names)
    }
    sample_ids = {path: f"s{index:06d}" for index, path in enumerate(sorted(paths))}
    real = {name for name in class_names if name != _NONE_CLASS}
    real |= {name for name in (group_name, safe_name) if name}
    real |= set(paths) | {os.path.basename(path) for path in paths}
    ids = {alias, *class_ids.values(), *sample_ids.values()}
    clash = sorted(real & ids)
    if clash:
        raise ScopedExportError(
            f"cannot hide these names, they read like the ids that replace them: {clash[:5]}"
        )
    private.class_ids = class_ids
    private.sample_ids = sample_ids


def _private_signature(signature: Any) -> Any:
    """A face/region model signature names the model's file: keep whether one
    was used and which (equal stays equal), not its name."""
    if not isinstance(signature, str) or signature == "none":
        return signature
    return "sig-" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


def _private_entry(
    entry: dict, sample_id: str, private: PrivateNames, class_names: list[str]
) -> dict:
    """A cache index entry that names its sample by id. Everything a sourceless
    run reads (label, split, cache file, content hash, build signature) is kept."""
    label = entry.get("label")
    concept = private.group_alias
    if isinstance(label, int) and 0 <= label < len(class_names):
        concept += "/" + private.class_ids[class_names[label]]
    out = dict(entry, filename=sample_id, filepath=sample_id, concept_name=concept)
    if "face_model_sig" in out:
        out["face_model_sig"] = _private_signature(out["face_model_sig"])
    return out


def _write_private_tensor_file(source: Path, target: Path, sample_id: str) -> None:
    """Copy a cached tensor file, naming its sample by id. The tensor and every
    other field are kept as they are."""
    import torch

    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ScopedExportError(f"cache file {source.name} is not a cache payload")
    payload = dict(payload)
    if "__source_path" in payload:
        payload["__source_path"] = sample_id
    if "__face_model_sig" in payload:
        payload["__face_model_sig"] = _private_signature(payload["__face_model_sig"])
    partial = target.with_name(target.name + ".partial")
    torch.save(payload, partial)
    os.replace(partial, target)


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
    private_names: PrivateNames | None = None,
) -> ScopedExportManifest:
    """Build a scoped cache export under ``export_dir/cache/`` and return its
    manifest. Raises ``ScopedExportError`` on any correctness violation.

    ``samples`` must already be cached in the global cache (run
    :func:`warm_group_cache` first); a missing entry is a hard error.

    With ``private_names`` the export carries ids, never names (see
    :class:`PrivateNames`): the index keys, each entry's ``filename`` /
    ``filepath`` / ``concept_name``, the manifest's ``group_name`` /
    ``safe_name`` / ``class_names`` / ``entry_paths`` and the source path
    inside every tensor file, which is therefore written anew rather than
    copied (its sha256 is of the file as written). ``config`` is written as
    given: the caller cleans it. The cache folder must start empty, so no file
    of an earlier export can ride along. Without it, nothing changes.
    """
    export_dir = Path(export_dir)
    cache_out = export_dir / "cache"
    cache_out.mkdir(parents=True, exist_ok=True)
    if private_names is not None and any(cache_out.iterdir()):
        raise ScopedExportError(
            f"the export's cache folder is not empty ({cache_out}); start from an empty "
            "folder, a file left from an earlier export could carry names"
        )

    # Authoritative path -> sample (dedup by normpath, which is how cache.json keys).
    auth: dict[str, SampleRef] = {}
    for s in samples:
        auth[os.path.normpath(s.path)] = s
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
        # Copy: the scoped entry gets this group's label/split stamped below
        # and must never mutate the shared global index.
        scoped_entries[path] = dict(entry)
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
        # The global cache is SHARED across groups but stores a single
        # label/split per path — whichever group warmed it last stamped its
        # view (an off-disk __none__ image is legitimately "train" for one
        # group and "val" for another). The current group's dataset is the
        # authority, and the pod's sourceless loader reads label/split from
        # THIS scoped index — so stamp rather than fail, which also guarantees
        # another group's label can never ship.
        ref = auth[path]
        entry["split"] = ref.split
        if ref.label is not None:
            entry["label"] = _label_for_storage(ref.label)
        if ref.concept_name:
            entry["concept_name"] = ref.concept_name

    pt_files = sorted({_pt_filename(e["cache_file"]) for e in scoped_entries.values()})

    # What the export says: the entries as they are, or keyed and named by id.
    out_entries = scoped_entries
    pt_source_id: dict[str, str] = {}
    if private_names is not None:
        _assign_private_ids(
            private_names, class_names=list(class_names), paths=list(scoped_entries),
            group_name=group_name, safe_name=safe_name,
        )  # fmt: skip
        ids = private_names.sample_ids
        out_entries = {
            ids[path]: _private_entry(entry, ids[path], private_names, list(class_names))
            for path, entry in sorted(scoped_entries.items())
        }
        # A tensor file shared by several samples (same bytes) names the first.
        for sample_id in sorted(out_entries, reverse=True):
            pt_source_id[_pt_filename(out_entries[sample_id]["cache_file"])] = sample_id

    pt_sha: dict[str, str] = {}
    for pt_name in pt_files:
        src = Path(global_cache_dir) / pt_name
        if not src.is_file():
            raise ScopedExportError(f"referenced cache file missing in global cache: {pt_name}")
        if private_names is None:
            shutil.copy2(src, cache_out / pt_name)
        else:
            _write_private_tensor_file(src, cache_out / pt_name, pt_source_id[pt_name])
        if hash_pt:
            pt_sha[pt_name] = _sha256_file(cache_out / pt_name)

    index_bytes = _scoped_index_bytes(out_entries)
    cache_json_path = cache_out / _CACHE_INDEX_FILENAME
    cache_json_path.write_bytes(index_bytes)
    cache_json_sha = hashlib.sha256(index_bytes).hexdigest()

    train_count = sum(1 for e in out_entries.values() if e.get("split") == "train")
    val_count = sum(1 for e in out_entries.values() if e.get("split") == "val")

    # --- Hard assertions (non-negotiable) ---
    if len(scoped_entries) != len(auth_paths):
        raise ScopedExportError(
            f"entry count {len(scoped_entries)} != expected sample count {len(auth_paths)}"
        )
    if set(scoped_entries) != auth_paths:
        raise ScopedExportError("scoped entries are not exactly the authoritative sample set")
    if private_names is not None and set(out_entries) != set(private_names.sample_ids.values()):
        raise ScopedExportError("the export's entries are not exactly the sample ids")
    for pt_name in pt_files:
        if not (cache_out / pt_name).is_file():
            raise ScopedExportError(f"export missing copied cache file: {pt_name}")
    if train_count < 1 or val_count < 1:
        raise ScopedExportError(
            f"eval/train split missing: train={train_count}, val={val_count}"
        )

    if private_names is not None:
        group_name = safe_name = private_names.group_alias
        class_names = [private_names.class_ids[name] for name in class_names]
    manifest = ScopedExportManifest(
        group_name=group_name,
        safe_name=safe_name,
        modeltype=modeltype,
        num_classes=num_classes,
        class_names=list(class_names),
        expected_entry_count=len(out_entries),
        train_count=train_count,
        val_count=val_count,
        entry_paths=sorted(out_entries),
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


def scoped_export_strings(export_dir: str | os.PathLike) -> set[str]:
    """Every string an export holds about its samples, classes and group: the
    index keys, file names and paths, concept and class names, group names and
    each tensor file's source path, plus the last part of every path and each
    part of a concept name. For the caller's final check before anything is
    uploaded. Every tensor file in the cache folder is read, listed in the
    manifest or not."""
    export_dir = Path(export_dir)
    found: set[str] = set()

    def _add(value: Any, *, path: bool = False, concept: bool = False) -> None:
        if not isinstance(value, str) or not value:
            return
        found.add(value)
        if path:
            found.add(re.split(r"[\\/]", value)[-1])
        if concept:
            found.update(part for part in value.split("/") if part)

    manifest_path = export_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _add(manifest.get("group_name"))
        _add(manifest.get("safe_name"))
        for name in manifest.get("class_names") or []:
            _add(name)
        for entry_path in manifest.get("entry_paths") or []:
            _add(entry_path, path=True)
    cache_dir = export_dir / "cache"
    index_path = cache_dir / _CACHE_INDEX_FILENAME
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for key, entry in (index.get("entries") or {}).items():
            _add(key, path=True)
            _add(entry.get("filename"))
            _add(entry.get("filepath"), path=True)
            _add(entry.get("concept_name"), concept=True)
        for paths in (index.get("hash_index") or {}).values():
            for indexed in paths:
                _add(indexed, path=True)
    if cache_dir.is_dir():
        import torch

        for pt_path in sorted(cache_dir.glob("*.pt")):
            payload = torch.load(pt_path, map_location="cpu", weights_only=False)
            if isinstance(payload, dict):
                _add(payload.get("__source_path"), path=True)
    return found


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
        # ``samples`` is the stable, de-replicated base list (ISSUE-0859); the
        # refs are deduped by normpath downstream anyway, and the pod's
        # sourceless loader reads its (path-keyed) entries from the cache index.
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
