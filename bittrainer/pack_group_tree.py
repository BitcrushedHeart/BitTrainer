"""One group's training tree, made from a backbone pack (Bitcrush ISSUE-1721).

On the VM a group model trains from the pack already uploaded for the
backbone, not from an export of Engine's cache. The group trainer reads a
folder tree (``GroupTrainConfig.group_folder``)::

    <tree>/<class id>/train/<content hash>.jpg
    <tree>/<class id>/val/<content hash>.jpg

with ``class_names`` the class ids, in Engine's class order.
:func:`pack_group_tree` builds it from the pack's labels for one ``g`` group:
an image whose answer for the group is held for validation at home (``v`` in
a format 2 pack) goes to ``val``, every other answer to ``train``, and an
image the group has no answer for is left out. The files are links to the
pack's images (a hard link where the file system allows, else a symbolic
link, else a copy), so a tree costs next to nothing on disk.

Only format 2 is read: a format 1 pack does not say which answers are held at
home, and a group trained from it could validate on what it trained on.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from bittrainer.backbone_pack import PackError, pack_records, read_manifest

# Written in every tree built here; a folder without it is never emptied.
TREE_NOTE = "tree.json"
_SPLITS = ("train", "val")


_LINKS = ("hardlink", "symlink", "copy")


def _link(source: Path, target: Path, preferred: str | None) -> str:
    """Put *source* at *target* the cheapest way that works, trying the way
    that worked last time first. Returns the way used."""
    order = _LINKS
    if preferred in _LINKS:
        order = (preferred, *(kind for kind in _LINKS if kind != preferred))
    for kind in order:
        try:
            if kind == "hardlink":
                os.link(source, target)
            elif kind == "symlink":
                os.symlink(source.resolve(), target)
            else:
                shutil.copyfile(source, target)
        except OSError:
            continue
        return kind
    raise PackError(f"cannot place {source.name} in the tree at {target.parent}")


def _clear(tree: Path) -> None:
    """Empty *tree* when this module built it; refuse any other folder."""
    if not tree.exists():
        return
    if not tree.is_dir() or (any(tree.iterdir()) and not (tree / TREE_NOTE).is_file()):
        raise PackError(
            f"{tree} is not a group tree built from a pack ({TREE_NOTE} is missing): "
            "it is left alone. Name an empty or new folder."
        )
    shutil.rmtree(tree)


def pack_group_tree(
    pack,
    group_key: str,
    class_ids: list[str],
    tree=None,
    *,
    folder_names: list[str] | None = None,
    allow_missing: bool = False,
) -> dict:
    """Build the folder tree for group *group_key* (a pack id such as ``g6``)
    from *pack*, and return what it holds.

    *class_ids* are the group's class ids in Engine's order; one folder each,
    with ``train`` and ``val`` inside even when empty. An answer naming a class
    not in *class_ids* is left out and counted (``other_class``).
    *folder_names* (same length and order) names each class's folder instead
    of its id: the trainer's ``class_names`` (the private ``k<index>`` names
    the VM config carries, which are not the pack's ids). *tree*
    defaults to ``<folder holding the pack>/work/groups/<group_key>``. A tree
    built here before is started over; any other non-empty folder is refused.

    Raises :class:`PackError` for a pack that is not format 2, and for images
    missing from the pack folder unless *allow_missing*.

    Returns ``{"tree", "group_key", "classes": {folder: {"train", "val"}},
    "train", "val", "unlabelled", "other_class", "missing_files", "link"}``.
    """
    pack = Path(pack)
    classes = [str(class_id) for class_id in class_ids]
    if len(set(classes)) != len(classes) or not classes:
        raise ValueError(f"class ids must be distinct and not empty: {classes}")
    folders = classes if folder_names is None else [str(n) for n in folder_names]
    if len(folders) != len(classes) or len(set(folders)) != len(folders):
        raise ValueError(
            f"folder names must be distinct, one per class id: {folders} for {classes}"
        )
    folder_of = dict(zip(classes, folders, strict=True))
    manifest = read_manifest(pack)
    if manifest.get("format") != 2:
        raise PackError(
            f"the pack at {pack} is format {manifest.get('format')}, and a group trains "
            "only from a format 2 pack (it marks the answers held for validation at home)"
        )
    records, found = pack_records(pack)
    missing = int(found.get("missing_files", 0))
    if missing and not allow_missing:
        raise PackError(
            f"{missing:,} images are missing from the pack folder (an unfinished "
            "upload?). Finish the upload first."
        )

    tree = (
        Path(tree) if tree is not None else pack.parent / "work" / "groups" / group_key
    )
    _clear(tree)
    for folder in folders:
        for split in _SPLITS:
            (tree / folder / split).mkdir(parents=True, exist_ok=True)

    counts = {folder: {split: 0 for split in _SPLITS} for folder in folders}
    unlabelled = other = 0
    method: str | None = None
    for record in records:
        answer = (record.get("groups") or {}).get(group_key)
        if not answer:
            unlabelled += 1
            continue
        folder = folder_of.get(str(answer))
        if folder is None:
            other += 1
            continue
        split = "val" if group_key in (record.get("held") or ()) else "train"
        source = Path(record["file_paths"][0])
        method = _link(source, tree / folder / split / source.name, method)
        counts[folder][split] += 1

    summary = {
        "tree": str(tree),
        "group_key": group_key,
        "classes": counts,
        "train": sum(c["train"] for c in counts.values()),
        "val": sum(c["val"] for c in counts.values()),
        "unlabelled": unlabelled,
        "other_class": other,
        "missing_files": missing,
        "link": method or "none",
    }
    (tree / TREE_NOTE).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary
