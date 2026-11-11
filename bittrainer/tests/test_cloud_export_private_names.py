"""A scoped export that carries ids, never names (Bitcrush ISSUE-1690, contract C1).

A group's export is uploaded to a rented machine. The owner's rule: nothing
uploaded may hold a local path, a file name, the group's name or a class name
(other than the fixed word ``__none__``). ``build_scoped_export`` takes a
``PrivateNames`` and writes ids in their place: ``s000000`` ... for samples (in
the order of their sorted paths), ``k`` + the class's place in the group's
class order for classes, the group's alias for the group. It hands the maps
back for home use only. Without one, the export is byte for byte what it was.

Everything is generated here: images drawn with numpy, a global cache built by
``SmartCache`` on a CPU. No model is trained.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from bittrainer.cloud_export import (
    PrivateNames,
    SampleRef,
    ScopedExportError,
    ScopedExportManifest,
    build_scoped_export,
    scoped_export_strings,
    verify_scoped_cache,
)
from bittrainer.group_dataset import GroupDataset
from bittrainer.smart_cache import SmartCache

GROUP = "Secret Client Group"
SAFE = "secret_client_group"
CLASSES = ["alpha_class", "beta_class", "__none__"]
FACE_SIG = "face_v8.pt:1700000000"
BUCKET = (32, 24)
ALIAS = "g00017"


# --------------------------------------------------------------------------- #
# A global cache of drawn images                                              #
# --------------------------------------------------------------------------- #


def _build(sample: dict) -> np.ndarray:
    width, height = sample["bucket"]
    with Image.open(sample["path"]) as image:
        array = np.asarray(image.convert("RGB").resize((width, height)))
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def _global_cache(tmp_path: Path) -> tuple[Path, list[SampleRef]]:
    """Two train and one val image per class, plus a copy of the first alpha
    image filed under beta: two sample paths, the same bytes, one tensor."""
    rng = np.random.default_rng(3)
    group = tmp_path / "home" / GROUP
    samples: list[dict] = []
    for label, name in enumerate(CLASSES):
        for split, count in (("train", 2), ("val", 1)):
            folder = group / name / split
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                path = folder / f"{name}_{split}_{index}.png"
                Image.fromarray(rng.integers(0, 255, (30, 40, 3), dtype=np.uint8)).save(
                    path
                )
                samples.append({"path": str(path), "label": label, "split": split})
    twin = group / "beta_class" / "train" / "twin_of_alpha.png"
    shutil.copyfile(group / "alpha_class" / "train" / "alpha_class_train_0.png", twin)
    samples.append({"path": str(twin), "label": 1, "split": "train"})
    for sample in samples:
        sample.update(bucket=BUCKET, concept_name=f"{GROUP}/{CLASSES[sample['label']]}")

    global_dir = tmp_path / "global_cache"
    cache = SmartCache(
        global_dir, modeltype="convnext_v2", tqdm_enabled=False, face_model_sig=FACE_SIG
    )
    cache.prepare(samples, _build, num_workers=1)
    refs = [
        SampleRef(
            path=os.path.normpath(s["path"]),
            split=s["split"],
            label=s["label"],
            concept_name=s["concept_name"],
        )
        for s in samples
    ]
    return global_dir, refs


def _export(global_dir, export_dir, refs, private=None, **kw) -> ScopedExportManifest:
    return build_scoped_export(
        global_cache_dir=global_dir,
        export_dir=export_dir,
        samples=refs,
        modeltype="convnext_v2",
        num_classes=len(CLASSES),
        class_names=list(CLASSES),
        group_name=GROUP,
        safe_name=SAFE,
        config={"training_mode": "full", "sourceless": True},
        private_names=private,
        **kw,
    )


def _planted(tmp_path: Path, refs: list[SampleRef]) -> set[str]:
    """Every string the export must not hold: paths, file names, the group's
    names, the classes but ``__none__``, the face model's file name."""
    names = {GROUP, SAFE, "face_v8.pt", str(tmp_path / "home")}
    names |= {c for c in CLASSES if c != "__none__"}
    names |= {ref.path for ref in refs}
    names |= {os.path.basename(ref.path) for ref in refs}
    return names


def _strings_in(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in _strings_in(k) + _strings_in(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _strings_in(v)]
    return []


def _leaks(export: Path, planted: set[str]) -> list[str]:
    """Each place a planted string shows up under *export*: its file names,
    its JSON (parsed and as text), every tensor file (loaded and as bytes)."""
    found = []
    forms = {
        s: (s.encode("utf-8"), json.dumps(s)[1:-1].encode("utf-8")) for s in planted
    }
    for file in sorted(p for p in export.rglob("*") if p.is_file()):
        relative = file.relative_to(export).as_posix()
        data = file.read_bytes()
        strings = [relative]
        if file.suffix == ".json":
            strings += _strings_in(json.loads(data))
        elif file.suffix == ".pt":
            strings += _strings_in(
                torch.load(file, map_location="cpu", weights_only=False)
            )
        for name, encoded in forms.items():
            if any(form in data for form in encoded):
                found.append(f"{relative}: bytes hold {name!r}")
            found += [f"{relative}: {s!r} holds {name!r}" for s in strings if name in s]
    return found


# --------------------------------------------------------------------------- #
# Without private names: exactly what it was                                  #
# --------------------------------------------------------------------------- #

GOLDEN_PATHS = [
    "planted-Secret_Group-alpha-train-0.jpg",
    "planted-Secret_Group-alpha-val-0.jpg",
    "planted-Secret_Group-beta-train-0.jpg",
    "planted-Secret_Group-beta-train-twin.jpg",
    "planted-Secret_Group-none-val-0.jpg",
]
# Written by the export at d954ad3, before private names existed.
GOLDEN_CACHE_JSON = "3edee8c249a0ef1bb766d7c3ca1bdfaff47d1ddf25c979db306ffecddd6e16dd"
GOLDEN_MANIFEST = "885d8542dff61557cc7ada25d39fa8e3e8a0f1822455670c0510cfeebd18344d"
GOLDEN_MANIFEST_TEXT = (
    "abda055fa0014d361634f221df05cd314b7c7f0a57a786c9fb154d231becd4da"
)


def _golden_global_cache(root: Path) -> list[SampleRef]:
    """A hand-written global cache: fixed values, no folders in the paths (the
    same on every OS), and stand-in tensor bytes (a plain export copies them)."""
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        (GOLDEN_PATHS[0], "aaaa00000001_512x512", "aaaa000000010000", "train", 0, "Secret Group/alpha"),
        (GOLDEN_PATHS[1], "aaaa00000002_512x384", "aaaa000000020000", "val", 0, "Secret Group/alpha"),
        (GOLDEN_PATHS[2], "aaaa00000003_384x512", "aaaa000000030000", "train", 1, "Secret Group/beta"),
        (GOLDEN_PATHS[3], "aaaa00000003_384x512", "aaaa000000030000", "train", 1, "Other Group/beta"),
        (GOLDEN_PATHS[4], "aaaa00000004_512x512", "aaaa000000040000", "val", 2, "Other Group/__none__"),
    ]  # fmt: skip
    entries = {}
    hash_index: dict[str, list[str]] = {}
    for number, (path, cache_file, digest, split, label, concept) in enumerate(rows):
        entries[path] = {
            "filename": path,
            "filepath": path,
            "hash": digest,
            "mtime": 1700000000.0 + number,
            "modeltype": "convnext_v2",
            "resolution": cache_file.split("_")[1],
            "cache_file": cache_file,
            "cache_version": 2,
            "label": label,
            "split": "train" if split == "val" else "val",  # stamped over by the export
            "concept_name": concept,
            "skin_normalise": False,
            "face_bbox_sig": "none",
            "face_model_sig": FACE_SIG,
        }
        hash_index.setdefault(digest, []).append(path)
        (root / f"{cache_file}_1.pt").write_bytes(
            f"stand-in tensor {cache_file}".encode()
        )
    index = {
        "version": 2,
        "entries": entries,
        "hash_index": hash_index,
        "last_validated": 1.0,
    }
    (root / "cache.json").write_text(json.dumps(index), encoding="utf-8")
    names = ["alpha", "alpha", "beta", "beta", "__none__"]
    return [
        SampleRef(
            path=path, split=split, label=label, concept_name=f"Secret Group/{name}"
        )
        for (path, _cf, _h, split, label, _c), name in zip(rows, names, strict=True)
    ]


@pytest.mark.parametrize("explicit", [False, True])
def test_a_plain_export_is_byte_for_byte_what_it_was(tmp_path, explicit):
    refs = _golden_global_cache(tmp_path / "global")
    extra = {"private_names": None} if explicit else {}

    build_scoped_export(
        global_cache_dir=tmp_path / "global",
        export_dir=tmp_path / "export",
        samples=refs,
        modeltype="convnext_v2",
        num_classes=3,
        class_names=["alpha", "beta", "__none__"],
        group_name="Secret Group",
        safe_name="secret_group",
        config={"max_epochs": 2, "training_mode": "full"},
        **extra,
    )

    export = tmp_path / "export"
    cache_json = (export / "cache" / "cache.json").read_bytes()
    assert hashlib.sha256(cache_json).hexdigest() == GOLDEN_CACHE_JSON
    text = (export / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(text)
    stamp = manifest.pop("created_at")
    assert (
        hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        == GOLDEN_MANIFEST
    )
    assert (
        hashlib.sha256(text.replace(stamp, "").encode()).hexdigest()
        == GOLDEN_MANIFEST_TEXT
    )
    # The tensor files are copied, not rewritten.
    for name in manifest["pt_files"]:
        assert (export / "cache" / name).read_bytes() == (
            tmp_path / "global" / name
        ).read_bytes()
    assert sorted(p.name for p in (export / "cache").iterdir()) == sorted(
        manifest["pt_files"] + ["cache.json"]
    )


# --------------------------------------------------------------------------- #
# With private names                                                          #
# --------------------------------------------------------------------------- #


def test_a_private_export_holds_no_path_file_name_group_or_class_name(tmp_path):
    global_dir, refs = _global_cache(tmp_path)
    planted = _planted(tmp_path, refs)
    # The global cache does hold them: the search below would find a leak.
    assert _leaks(global_dir, planted)

    _export(global_dir, tmp_path / "export", refs, PrivateNames(group_alias=ALIAS))

    assert _leaks(tmp_path / "export", planted) == []


def test_none_keeps_its_name_and_the_maps_are_complete_and_invertible(tmp_path):
    global_dir, refs = _global_cache(tmp_path)
    private = PrivateNames(group_alias=ALIAS)

    manifest = _export(global_dir, tmp_path / "export", refs, private)

    assert private.class_ids == {
        "alpha_class": "k0",
        "beta_class": "k1",
        "__none__": "__none__",
    }
    paths = sorted({ref.path for ref in refs})
    assert private.sample_ids == {path: f"s{i:06d}" for i, path in enumerate(paths)}
    for ids in (private.class_ids, private.sample_ids):
        assert len(set(ids.values())) == len(ids)  # invertible
    assert manifest.class_names == ["k0", "k1", "__none__"]
    assert manifest.group_name == manifest.safe_name == ALIAS
    assert manifest.entry_paths == sorted(private.sample_ids.values())
    assert manifest.config == {"training_mode": "full", "sourceless": True}  # as given

    entries = json.loads((tmp_path / "export" / "cache" / "cache.json").read_bytes())[
        "entries"
    ]
    for ref in refs:
        sample_id = private.sample_ids[ref.path]
        entry = entries[sample_id]
        assert (entry["label"], entry["split"]) == (ref.label, ref.split)
        assert entry["filename"] == entry["filepath"] == sample_id
        class_id = private.class_ids[CLASSES[ref.label]]
        assert entry["concept_name"] == f"{ALIAS}/{class_id}"
    assert any(e["concept_name"] == f"{ALIAS}/__none__" for e in entries.values())


def test_class_ids_follow_the_group_class_order(tmp_path):
    global_dir, refs = _global_cache(tmp_path)
    private = PrivateNames(group_alias=ALIAS)
    order = ["__none__", "beta_class", "alpha_class"]
    relabelled = [
        SampleRef(r.path, r.split, order.index(CLASSES[r.label]), r.concept_name)
        for r in refs
    ]

    manifest = build_scoped_export(
        global_cache_dir=global_dir, export_dir=tmp_path / "export", samples=relabelled,
        modeltype="convnext_v2", num_classes=3, class_names=order, group_name=GROUP,
        private_names=private,
    )  # fmt: skip

    assert private.class_ids == {
        "__none__": "__none__",
        "beta_class": "k1",
        "alpha_class": "k2",
    }
    assert manifest.class_names == ["__none__", "k1", "k2"]


def test_two_sample_paths_with_the_same_bytes_share_one_rewritten_tensor_file(tmp_path):
    global_dir, refs = _global_cache(tmp_path)
    private = PrivateNames(group_alias=ALIAS)
    manifest = _export(global_dir, tmp_path / "export", refs, private)
    cache_dir = tmp_path / "export" / "cache"
    index = json.loads((cache_dir / "cache.json").read_bytes())

    original = os.path.normpath(
        str(
            tmp_path
            / "home"
            / GROUP
            / "alpha_class"
            / "train"
            / "alpha_class_train_0.png"
        )
    )
    twin = os.path.normpath(
        str(tmp_path / "home" / GROUP / "beta_class" / "train" / "twin_of_alpha.png")
    )
    ids = sorted([private.sample_ids[original], private.sample_ids[twin]])
    first, second = (index["entries"][i] for i in ids)
    assert first["cache_file"] == second["cache_file"]
    assert index["hash_index"][first["hash"]] == ids
    assert {first["label"], second["label"]} == {0, 1}  # each keeps its own label
    assert len(manifest.pt_files) == len(refs) - 1

    pt_name = f"{first['cache_file']}_1.pt"
    written = torch.load(cache_dir / pt_name, map_location="cpu", weights_only=False)
    source = torch.load(global_dir / pt_name, map_location="cpu", weights_only=False)
    assert written["__source_path"] == ids[0]
    assert torch.equal(written["tensor"], source["tensor"])
    assert written["__source_hash"] == source["__source_hash"]
    assert written["__bucket"] == source["__bucket"]
    # The manifest holds the sha of the file as written, not of the original.
    assert (
        manifest.pt_sha256[pt_name]
        == hashlib.sha256((cache_dir / pt_name).read_bytes()).hexdigest()
    )
    assert (
        manifest.pt_sha256[pt_name]
        != hashlib.sha256((global_dir / pt_name).read_bytes()).hexdigest()
    )


def test_verify_scoped_cache_passes_on_a_private_export_and_still_catches_damage(
    tmp_path,
):
    global_dir, refs = _global_cache(tmp_path)
    manifest = _export(
        global_dir, tmp_path / "export", refs, PrivateNames(group_alias=ALIAS)
    )
    cache_dir = tmp_path / "export" / "cache"

    assert verify_scoped_cache(cache_dir, manifest) == (True, "", "ok")
    reread = ScopedExportManifest.read_json(tmp_path / "export" / "manifest.json")
    assert verify_scoped_cache(cache_dir, reread) == (True, "", "ok")

    victim = cache_dir / manifest.pt_files[0]
    victim.write_bytes(victim.read_bytes() + b"x")
    ok, key, message = verify_scoped_cache(cache_dir, manifest)
    assert (ok, key) == (False, "DATA_ERROR") and "sha256" in message


def _sourceless(cache_dir: Path) -> dict[str, list]:
    cache = SmartCache(cache_dir, modeltype="convnext_v2", tqdm_enabled=False)
    rows = {}
    for split in ("train", "val"):
        dataset = GroupDataset(
            cache_dir / "nowhere",
            list(CLASSES),
            split=split,
            cache=cache,
            sourceless=True,
        )
        for index, sample in enumerate(dataset.samples):
            tensor, label, bucket = dataset[index]
            rows[sample["path"]] = [
                split,
                label,
                bucket,
                tensor,
                cache.content_hash(sample["path"]),
            ]
    return rows


def test_a_sourceless_dataset_reads_a_private_export_as_it_reads_a_plain_one(tmp_path):
    global_dir, refs = _global_cache(tmp_path)
    private = PrivateNames(group_alias=ALIAS)
    _export(global_dir, tmp_path / "plain", refs)
    _export(global_dir, tmp_path / "private", refs, private)

    plain = _sourceless(tmp_path / "plain" / "cache")
    hidden = _sourceless(tmp_path / "private" / "cache")

    real = {sample_id: path for path, sample_id in private.sample_ids.items()}
    assert (
        {real[sample_id] for sample_id in hidden}
        == set(plain)
        == {r.path for r in refs}
    )
    for sample_id, (split, label, bucket, tensor, digest) in hidden.items():
        want = plain[real[sample_id]]
        assert (split, label, bucket) == tuple(want[:3])
        assert torch.equal(tensor, want[3])
        # The content hash the feature cache keys on is still there.
        assert digest is not None and digest == want[4]


def test_a_sourceless_dataset_still_refuses_a_missing_tensor_file(tmp_path):
    global_dir, refs = _global_cache(tmp_path)
    manifest = _export(
        global_dir, tmp_path / "export", refs, PrivateNames(group_alias=ALIAS)
    )
    (tmp_path / "export" / "cache" / manifest.pt_files[0]).unlink()

    with pytest.raises(RuntimeError, match="missing"):
        _sourceless(tmp_path / "export" / "cache")


def test_scoped_export_strings_names_everything_on_a_plain_export_and_nothing_on_a_private_one(
    tmp_path,
):
    global_dir, refs = _global_cache(tmp_path)
    planted = _planted(tmp_path, refs) - {"face_v8.pt", str(tmp_path / "home")}
    private = PrivateNames(group_alias=ALIAS)
    _export(global_dir, tmp_path / "plain", refs)
    _export(global_dir, tmp_path / "private", refs, private)

    assert planted <= scoped_export_strings(tmp_path / "plain")

    strings = scoped_export_strings(tmp_path / "private")
    assert not any(name in s for s in strings for name in planted)
    assert set(private.sample_ids.values()) <= strings
    assert {ALIAS, "k0", "k1", "__none__"} <= strings
    pattern = re.compile(rf"^(s\d{{6}}|k\d+|__none__|{ALIAS})(/(k\d+|__none__))?$")
    assert all(pattern.match(s) for s in strings), strings


def test_scoped_export_strings_sees_a_tensor_file_left_from_an_older_export(tmp_path):
    global_dir, refs = _global_cache(tmp_path)
    _export(global_dir, tmp_path / "private", refs, PrivateNames(group_alias=ALIAS))
    stray = next(global_dir.glob("*.pt"))
    shutil.copy2(stray, tmp_path / "private" / "cache" / "left_over_1.pt")

    strings = scoped_export_strings(tmp_path / "private")

    assert {ref.path for ref in refs} & strings


# --------------------------------------------------------------------------- #
# Refusals                                                                     #
# --------------------------------------------------------------------------- #


def test_a_private_export_starts_from_an_empty_cache_folder(tmp_path):
    global_dir, refs = _global_cache(tmp_path)
    _export(global_dir, tmp_path / "export", refs)  # a plain export, names and all

    with pytest.raises(ScopedExportError, match="empty"):
        _export(global_dir, tmp_path / "export", refs, PrivateNames(group_alias=ALIAS))


@pytest.mark.parametrize(
    ("alias", "classes", "match"),
    [
        ("", CLASSES, "alias"),
        ("g/00017", CLASSES, "alias"),
        (GROUP, CLASSES, "alias"),
        (ALIAS, ["k1", "beta_class", "__none__"], "k1"),
        (ALIAS, ["alpha_class", "s000003", "__none__"], "s000003"),
        (ALIAS, ["alpha_class", "alpha_class", "__none__"], "twice"),
    ],
)
def test_a_private_export_refuses_names_it_cannot_hide(tmp_path, alias, classes, match):
    global_dir, refs = _global_cache(tmp_path)

    with pytest.raises(ScopedExportError, match=match):
        build_scoped_export(
            global_cache_dir=global_dir, export_dir=tmp_path / "export", samples=refs,
            modeltype="convnext_v2", num_classes=3, class_names=classes, group_name=GROUP,
            private_names=PrivateNames(group_alias=alias),
        )  # fmt: skip


def test_the_hard_assertions_hold_with_private_names(tmp_path):
    global_dir, refs = _global_cache(tmp_path)
    missing = SampleRef(
        path=os.path.normpath(str(tmp_path / "not_cached.png")), split="train"
    )
    with pytest.raises(ScopedExportError, match="not in the global cache"):
        _export(
            global_dir,
            tmp_path / "a",
            [*refs, missing],
            PrivateNames(group_alias=ALIAS),
        )

    train_only = [r for r in refs if r.split == "train"]
    with pytest.raises(ScopedExportError, match="split missing"):
        _export(global_dir, tmp_path / "b", train_only, PrivateNames(group_alias=ALIAS))

    with pytest.raises(ScopedExportError, match="drift"):
        _export(
            global_dir, tmp_path / "c", refs, PrivateNames(group_alias=ALIAS),
            db_expected_paths={r.path for r in refs[1:]},
        )  # fmt: skip
