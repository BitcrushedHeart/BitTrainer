"""Read a backbone training pack into the trainer's records (Bitcrush ISSUE-1655).

A pack is what the Suite's ``tools/backbone_pack_export.py`` writes so a backbone
can be trained away from the machine that holds the labels. It carries no names:

    manifest.json   {"format": 1, ...settings and counts...}
    labels.jsonl    one line per image:
                    {"h": "<content hash>",
                     "b": {"c12": 1, "m57": 0},     head id -> 1 (yes) / 0 (no)
                     "g": {"g3": "k17"}}            group id -> class id
    images/<first two characters of h>/<h>.jpg

``c`` heads are concepts, ``m`` heads are classes of a multi-label group (both
train as yes/no heads), ``g`` are single-label groups with ``k`` classes. The
ids stay ids all the way through: they become the head names in the exported
candidate, and the machine that made the pack maps them back.

The manifest may carry ``"geometry_sensitive"``: the ids of groups and yes/no
heads whose answer depends on framing, position or tilt; their images train
without geometric augmentation. The key is optional and the format number does
not change for it.

Format 2 (Bitcrush ISSUE-1718) adds the answers that are validation answers at
home, which must never train:

    manifest.json   "home_validation": {"answers": N, "images": M}, always
                    present in format 2, even when N is 0
    labels.jsonl    "v": ["c12", "g3"]   the ids of this line's ``b`` / ``g``
                                         answers that are marked; left out
                                         when there are none

The number changed on purpose: a trainer that does not know ``v`` refuses the
pack instead of training on the marked answers. A record carries its marks as
``"held"`` (see ``backbone_trainer._split_samples``).
"""

from __future__ import annotations

import json
from pathlib import Path

# The formats this trainer reads; PACK_FORMAT is the newest of them.
PACK_FORMATS = (1, 2)
PACK_FORMAT = 2
_ANSWERS = {1: "positive", 0: "negative"}


class PackError(RuntimeError):
    """The folder is not a pack this trainer can read."""


def read_manifest(pack: Path) -> dict:
    """The pack's manifest; raises :class:`PackError` unless it is a format
    this trainer reads (format 2 must carry its ``home_validation`` counts)."""
    path = Path(pack) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackError(f"no readable manifest.json in {pack}") from exc
    if manifest.get("format") not in PACK_FORMATS:
        readable = " or ".join(str(number) for number in PACK_FORMATS)
        raise PackError(
            f"pack format {manifest.get('format')!r} is not supported "
            f"(this trainer reads format {readable})"
        )
    if manifest["format"] >= 2:
        counts = manifest.get("home_validation")
        if not (
            isinstance(counts, dict)
            and all(
                type(counts.get(key)) is int and counts[key] >= 0
                for key in ("answers", "images")
            )
        ):
            raise PackError(
                f"the format 2 manifest of {pack} has no home_validation counts"
            )
    return manifest


def _held_marks(entry: dict) -> list[str]:
    """A labels line's ``v``: sorted, each once, every one an answer of the line."""
    marks = entry.get("v")
    if marks is None:
        return []
    if not isinstance(marks, list):
        raise PackError(f"the marks of image {entry.get('h')} are not a list")
    answered = set(entry.get("b") or {}) | set(entry.get("g") or {})
    unknown = sorted(
        str(mark) for mark in marks if not isinstance(mark, str) or mark not in answered
    )
    if unknown:
        raise PackError(
            f"image {entry.get('h')} marks {', '.join(unknown)} as validation, "
            "which it has no answer for"
        )
    return sorted(set(marks))


def pack_geometry_sensitive(pack: Path) -> list[str] | None:
    """The manifest's ``geometry_sensitive`` ids, for the training request.

    ``None`` when the manifest has no such key (a pack written before the export
    knew about it). That is not the same as an empty list: an empty list says
    the export looked and found none, and only then may a run say its
    augmentation was geometry-safe."""
    manifest = read_manifest(pack)
    if "geometry_sensitive" not in manifest:
        return None
    names = manifest["geometry_sensitive"] or []
    if not isinstance(names, list):
        raise PackError(f"geometry_sensitive in the manifest of {pack} is not a list")
    return [str(name) for name in names]


def pack_records(pack: Path, *, limit: int | None = None) -> tuple[list[dict], dict]:
    """The pack as ``run_backbone_training`` records, plus what was found.

    ``limit`` keeps the first N label lines (they are sorted by content hash, so
    the slice is an even sample) for a short trial run. An image whose file is
    not in the folder is left out and counted in ``missing_files``: an
    unfinished upload should be noticed, not trained around.

    A format 2 line's marks go on its record as ``"held"`` (no key when it has
    none). The second value then also says how many answers are held, on how
    many of the records returned (``held_answers`` / ``held_images``); a format
    1 pack's counts stay as they were. The trainer validates on the held answers
    when there are any, and otherwise splits by content hash. Read whole (no
    ``limit``), the marks must add up to the manifest's ``home_validation``: a
    labels file that lost them would otherwise train on every one.
    """
    pack = Path(pack)
    manifest = read_manifest(pack)
    try:
        lines = (pack / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PackError(f"no readable labels.jsonl in {pack}") from exc

    records: list[dict] = []
    missing = 0
    taken = 0
    marked_answers = marked_images = 0  # over every line read, file or not
    held_answers = held_images = 0  # over the records returned
    for raw in lines:
        if not raw.strip():
            continue
        if limit is not None and taken >= limit:
            break
        taken += 1
        entry = json.loads(raw)
        digest = entry["h"]
        marks = _held_marks(entry)
        if marks:
            marked_answers += len(marks)
            marked_images += 1
        image = pack / "images" / digest[:2] / f"{digest}.jpg"
        if not image.is_file():
            missing += 1
            continue
        record = {
            "content_hash": digest,
            "file_paths": [str(image)],
            "binary": {
                head: _ANSWERS[int(answer)]
                for head, answer in (entry.get("b") or {}).items()
            },
            "groups": dict(entry.get("g") or {}),
            "splits": {"train": 1},
        }
        if marks:
            record["held"] = marks
            held_answers += len(marks)
            held_images += 1
        records.append(record)
    counts = {"images": len(records), "missing_files": missing}
    home = manifest.get("home_validation") if manifest["format"] >= 2 else None
    if home is not None and limit is None:
        declared = (home["answers"], home["images"])
        if (marked_answers, marked_images) != declared:
            raise PackError(
                f"labels.jsonl marks {marked_answers} answers on {marked_images} "
                f"images, the manifest says {declared[0]} on {declared[1]}"
            )
    if home is not None or marked_answers:
        counts.update(held_answers=held_answers, held_images=held_images)
    return records, counts
