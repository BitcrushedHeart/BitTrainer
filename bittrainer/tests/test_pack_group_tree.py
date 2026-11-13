"""A group's training tree, made from the backbone pack (Bitcrush ISSUE-1721).

On the VM a group trains from the pack already uploaded, not from Engine's
cache: ``pack_group_tree`` links each image the group labels into the folder
tree the group trainer reads, ``<tree>/<class id>/{train,val}/<hash>.jpg``.
Answers held for validation at home (``v`` in a format 2 pack) go to ``val``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bittrainer.backbone_pack import PackError
from bittrainer.pack_group_tree import pack_group_tree

# Engine's class order, which is not the sorted order ("k10" < "k9").
_CLASSES = ["k9", "k10", "__none__"]


def _pack(root: Path, lines: list[dict], *, fmt: int = 2) -> Path:
    pack = root / "pack"
    for line in lines:
        digest = line["h"]
        image = pack / "images" / digest[:2] / f"{digest}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"jpeg {digest}".encode())
    (pack / "labels.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
    )
    held = [line for line in lines if line.get("v")]
    manifest: dict = {"format": fmt, "images": len(lines)}
    if fmt >= 2:
        manifest["home_validation"] = {
            "answers": sum(len(line["v"]) for line in held),
            "images": len(held),
        }
    (pack / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return pack


_LINES = [
    {"h": "aa01", "g": {"g6": "k9"}},
    {"h": "aa02", "g": {"g6": "k10"}, "v": ["g6"]},
    {"h": "bb03", "g": {"g6": "__none__", "g7": "k1"}},
    # Held for another head only: still trains this group.
    {"h": "bb04", "g": {"g6": "k10"}, "b": {"c1": 1}, "v": ["c1"]},
    # No answer for g6: not in the tree.
    {"h": "cc05", "g": {"g7": "k2"}},
    {"h": "cc06", "b": {"c1": 0}},
    # Held for g6 and another head.
    {"h": "dd07", "g": {"g6": "k9", "g7": "k1"}, "v": ["g6", "g7"]},
]


def _files(tree: Path) -> set[str]:
    return {path.relative_to(tree).as_posix() for path in tree.rglob("*.jpg")}


def test_each_labelled_image_is_linked_under_its_class_and_split(tmp_path):
    pack = _pack(tmp_path, _LINES)
    tree = tmp_path / "work" / "g6"

    counts = pack_group_tree(pack, "g6", _CLASSES, tree=tree)

    assert _files(tree) == {
        "k9/train/aa01.jpg",
        "k10/val/aa02.jpg",
        "__none__/train/bb03.jpg",
        "k10/train/bb04.jpg",
        "k9/val/dd07.jpg",
    }
    assert counts["train"] == 3 and counts["val"] == 2
    assert counts["classes"] == {
        "k9": {"train": 1, "val": 1},
        "k10": {"train": 1, "val": 1},
        "__none__": {"train": 1, "val": 0},
    }
    # The image itself, not a copy of something else.
    linked = tree / "k9" / "train" / "aa01.jpg"
    assert linked.read_bytes() == b"jpeg aa01"
    assert counts["link"] in ("hardlink", "symlink", "copy")


def test_an_image_with_no_answer_for_the_group_is_absent(tmp_path):
    pack = _pack(tmp_path, _LINES)
    tree = tmp_path / "tree"

    counts = pack_group_tree(pack, "g6", _CLASSES, tree=tree)

    names = {Path(name).stem for name in _files(tree)}
    assert not names & {"cc05", "cc06"}
    assert counts["unlabelled"] == 2


def test_every_class_has_both_folders_even_when_empty(tmp_path):
    pack = _pack(tmp_path, _LINES)
    tree = tmp_path / "tree"

    pack_group_tree(pack, "g6", _CLASSES, tree=tree)

    for class_id in _CLASSES:
        assert (tree / class_id / "train").is_dir()
        assert (tree / class_id / "val").is_dir()


def test_a_class_the_plan_does_not_have_is_left_out_and_counted(tmp_path):
    pack = _pack(tmp_path, _LINES)
    tree = tmp_path / "tree"

    counts = pack_group_tree(pack, "g6", ["k9", "__none__"], tree=tree)

    assert not (tree / "k10").exists()
    assert counts["other_class"] == 2


def test_building_again_starts_the_tree_over(tmp_path):
    pack = _pack(tmp_path, _LINES)
    tree = tmp_path / "tree"
    pack_group_tree(pack, "g6", _CLASSES, tree=tree)
    stale = tree / "k9" / "train" / "stale.jpg"
    stale.write_bytes(b"old")

    pack_group_tree(pack, "g6", _CLASSES, tree=tree)

    assert not stale.exists()
    assert "k9/train/aa01.jpg" in _files(tree)


def test_a_folder_that_is_not_a_tree_is_never_emptied(tmp_path):
    pack = _pack(tmp_path, _LINES)
    tree = tmp_path / "somebody_elses"
    tree.mkdir()
    (tree / "keep.txt").write_text("mine", encoding="utf-8")

    with pytest.raises(PackError):
        pack_group_tree(pack, "g6", _CLASSES, tree=tree)

    assert (tree / "keep.txt").read_text(encoding="utf-8") == "mine"


def test_the_default_tree_is_beside_the_pack_under_work_groups(tmp_path):
    pack = _pack(tmp_path, _LINES)

    counts = pack_group_tree(pack, "g6", _CLASSES)

    assert Path(counts["tree"]) == tmp_path / "work" / "groups" / "g6"
    assert "k9/train/aa01.jpg" in _files(tmp_path / "work" / "groups" / "g6")


def test_a_pack_that_is_not_format_2_is_refused(tmp_path):
    pack = _pack(
        tmp_path,
        [{k: v for k, v in line.items() if k != "v"} for line in _LINES],
        fmt=1,
    )

    with pytest.raises(PackError, match="format 2"):
        pack_group_tree(pack, "g6", _CLASSES, tree=tmp_path / "tree")

    assert not (tmp_path / "tree").exists()


def test_a_missing_image_is_refused_unless_allowed(tmp_path):
    pack = _pack(tmp_path, _LINES)
    (pack / "images" / "aa" / "aa01.jpg").unlink()

    with pytest.raises(PackError, match="missing"):
        pack_group_tree(pack, "g6", _CLASSES, tree=tmp_path / "tree")

    counts = pack_group_tree(
        pack, "g6", _CLASSES, tree=tmp_path / "tree", allow_missing=True
    )
    assert counts["missing_files"] == 1
    assert "k9/train/aa01.jpg" not in _files(tmp_path / "tree")


def test_class_ids_must_be_distinct(tmp_path):
    pack = _pack(tmp_path, _LINES)

    with pytest.raises(ValueError):
        pack_group_tree(pack, "g6", ["k9", "k9"], tree=tmp_path / "tree")


def test_folders_can_take_the_trainers_class_names(tmp_path):
    """The VM config names classes ``k<index>`` (and ``__none__``), not by
    the pack's ids: the folders must match the config's class_names."""
    pack = _pack(tmp_path, _LINES)
    tree = tmp_path / "tree"

    counts = pack_group_tree(
        pack, "g6", _CLASSES, tree=tree, folder_names=["k0", "k1", "__none__"]
    )

    assert _files(tree) == {
        "k0/train/aa01.jpg",
        "k1/val/aa02.jpg",
        "__none__/train/bb03.jpg",
        "k1/train/bb04.jpg",
        "k0/val/dd07.jpg",
    }
    assert counts["classes"]["k1"] == {"train": 1, "val": 1}
    with pytest.raises(ValueError):
        pack_group_tree(pack, "g6", _CLASSES, tree=tree, folder_names=["k0", "k1"])
