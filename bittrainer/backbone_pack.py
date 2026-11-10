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
"""

from __future__ import annotations

import json
from pathlib import Path

PACK_FORMAT = 1
_ANSWERS = {1: "positive", 0: "negative"}


class PackError(RuntimeError):
    """The folder is not a pack this trainer can read."""


def read_manifest(pack: Path) -> dict:
    """The pack's manifest; raises :class:`PackError` unless it is format 1."""
    path = Path(pack) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackError(f"no readable manifest.json in {pack}") from exc
    if manifest.get("format") != PACK_FORMAT:
        raise PackError(
            f"pack format {manifest.get('format')!r} is not supported "
            f"(this trainer reads format {PACK_FORMAT})"
        )
    return manifest


def pack_geometry_sensitive(pack: Path) -> list[str]:
    """The manifest's ``geometry_sensitive`` ids, for the training request;
    empty if the pack names none."""
    names = read_manifest(pack).get("geometry_sensitive") or []
    if not isinstance(names, list):
        raise PackError(f"geometry_sensitive in the manifest of {pack} is not a list")
    return [str(name) for name in names]


def pack_records(pack: Path, *, limit: int | None = None) -> tuple[list[dict], dict]:
    """The pack as ``run_backbone_training`` records, plus what was found.

    ``limit`` keeps the first N label lines (they are sorted by content hash, so
    the slice is an even sample) for a short trial run. An image whose file is
    not in the folder is left out and counted in ``missing_files``: an
    unfinished upload should be noticed, not trained around.

    No split is assigned: the trainer splits train / validation by content hash.
    """
    pack = Path(pack)
    read_manifest(pack)
    try:
        lines = (pack / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PackError(f"no readable labels.jsonl in {pack}") from exc

    records: list[dict] = []
    missing = 0
    taken = 0
    for raw in lines:
        if not raw.strip():
            continue
        if limit is not None and taken >= limit:
            break
        taken += 1
        entry = json.loads(raw)
        digest = entry["h"]
        image = pack / "images" / digest[:2] / f"{digest}.jpg"
        if not image.is_file():
            missing += 1
            continue
        records.append(
            {
                "content_hash": digest,
                "file_paths": [str(image)],
                "binary": {
                    head: _ANSWERS[int(answer)]
                    for head, answer in (entry.get("b") or {}).items()
                },
                "groups": dict(entry.get("g") or {}),
                "splits": {"train": 1},
            }
        )
    return records, {"images": len(records), "missing_files": missing}
