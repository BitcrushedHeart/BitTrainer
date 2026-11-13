"""Answers marked as validation at home never train (Bitcrush ISSUE-1718).

A backbone head is compared with a concept's or group's own model on Engine's
validation answers, which is only fair if the backbone never trained on them.
Pack format 2 marks those answers (``"v"`` on a labels line); the trainer:

* leaves a marked ("held") answer out of every training sample, and never
  draws the image as an implied negative for that head, which is what the
  soft-negatives policy would otherwise do with an answer that went missing;
* validates on the held answers and nothing else whenever any record has one,
  instead of holding images out by content hash;
* behaves exactly as before when no record has one;
* says which it was (``validation_source``) in ``result.json`` and in the
  candidate's metadata, with the counts.

All data is generated. Runs use the stand-in trunk (``backbone_stub``) except
head-only retraining, which needs a real ``atto`` trunk for its feature cache.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from PIL import Image
from safetensors import safe_open

import bittrainer.backbone_pack as backbone_pack
import bittrainer.backbone_trainer as bb
import bittrainer.generic.tasks.backbone_distill_task as bd
import bittrainer.generic.tasks.backbone_task as bt
import bittrainer.tools.backbone_pack_train as runner
from bittrainer.backbone_pack import PackError, pack_records
from bittrainer.backbone_trainer import run_backbone_training
from bittrainer.tests.backbone_stub import (
    StubTrunk,
    fake_v1_converter,
    use_stub_trunk,
)

_SOFT = {"mode": "soft_implicit_negative", "implicit_negative_value": 0.1}


# -- helpers ------------------------------------------------------------------


def _record(i: int, path, *, binary=None, groups=None, held=None) -> dict:
    record = {
        "content_hash": f"{i:04x}" + "ab" * 30,
        "file_paths": [str(path)],
        "binary": dict(binary or {}),
        "groups": dict(groups or {}),
        "splits": {"train": 1},
    }
    if held:
        record["held"] = sorted(held)
    return record


def _touch(root: Path, name: str) -> Path:
    """A file that exists: ``_build_samples`` only asks that much."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_bytes(b"")
    return path


def _image(root: Path, i: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"img{i}.png"
    Image.new(
        "RGB", (72, 64), color=((i * 37) % 255, (i * 83) % 255, (i * 151) % 255)
    ).save(path)
    return path


def _split(records):
    vocab = bb._Vocab(records)
    samples, _missing = bb._build_samples(records, vocab)
    train, val = bb._split_samples(records, samples, 0.15)
    return vocab, train, val


def _held_pairs(records) -> set[tuple[str, str]]:
    """Every (path, head or group) a record holds back."""
    return {
        (record["file_paths"][0], name)
        for record in records
        for name in record.get("held") or []
    }


def _trained_pairs(planned) -> set[tuple[str, str]]:
    return {
        (sample.path, name)
        for sample in planned
        for name in [*sample.binary, *sample.groups]
    }


def _concept_c_records(root: Path) -> list[dict]:
    """Head ``c``: 5 trained yes, 3 held yes, 4 human no, 40 images it has no
    answer for (they carry ``b`` only). The held images still train ``b``."""
    records = []
    i = 0
    for _ in range(5):
        path = _touch(root, f"p{i}.png")
        records.append(_record(i, path, binary={"c": "positive", "b": "positive"}))
        i += 1
    for _ in range(3):
        path = _touch(root, f"h{i}.png")
        records.append(
            _record(i, path, binary={"c": "positive", "b": "negative"}, held=["c"])
        )
        i += 1
    for _ in range(4):
        path = _touch(root, f"n{i}.png")
        records.append(_record(i, path, binary={"c": "negative"}))
        i += 1
    for k in range(40):
        path = _touch(root, f"u{i}.png")
        answer = "positive" if k % 2 else "negative"
        records.append(_record(i, path, binary={"b": answer}))
        i += 1
    return records


# -- reading a format 2 pack --------------------------------------------------


def _write_pack(root: Path, lines: list[dict], manifest: dict) -> Path:
    pack = root / "pack"
    for line in lines:
        digest = line["h"]
        target = pack / "images" / digest[:2] / f"{digest}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(len(digest), 0, 0)).save(target)
    text = "".join(json.dumps(line) + "\n" for line in lines)
    (pack / "labels.jsonl").write_text(text, encoding="utf-8")
    (pack / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return pack


def _format_2_lines() -> list[dict]:
    return [
        {"h": "a1" + "0" * 62, "b": {"c1": 1, "m7": 0}, "g": {"g3": "k1"}},
        {"h": "b2" + "0" * 62, "b": {"c1": 0}, "v": ["c1"]},
        {
            "h": "c3" + "0" * 62,
            "b": {"c1": 1, "m7": 1},
            "g": {"g3": "k2"},
            "v": ["g3", "m7"],
        },
    ]


def test_a_format_2_pack_puts_its_marks_on_the_records(tmp_path):
    manifest = {"format": 2, "home_validation": {"answers": 3, "images": 2}}
    pack = _write_pack(tmp_path, _format_2_lines(), manifest)

    records, counts = pack_records(pack)

    assert counts == {
        "images": 3,
        "missing_files": 0,
        "held_answers": 3,
        "held_images": 2,
    }
    by_hash = {record["content_hash"][:2]: record for record in records}
    assert "held" not in by_hash["a1"]  # no marks, no key: as a format 1 record
    assert by_hash["b2"]["held"] == ["c1"]
    assert by_hash["c3"]["held"] == ["g3", "m7"]
    # Everything else is exactly what format 1 gives.
    assert by_hash["c3"]["binary"] == {"c1": "positive", "m7": "positive"}
    assert by_hash["c3"]["groups"] == {"g3": "k2"}


def test_a_format_2_pack_with_no_marks_still_reports_zero(tmp_path):
    lines = [{"h": "a1" + "0" * 62, "b": {"c1": 1}}]
    manifest = {"format": 2, "home_validation": {"answers": 0, "images": 0}}
    pack = _write_pack(tmp_path, lines, manifest)

    records, counts = pack_records(pack)

    assert counts["held_answers"] == 0 and counts["held_images"] == 0
    assert "held" not in records[0]


def test_a_trainer_that_only_knows_format_1_refuses_format_2(tmp_path, monkeypatch):
    """The reason the number changed: a trainer that does not know ``v`` must
    refuse the pack rather than train on the marked answers."""
    manifest = {"format": 2, "home_validation": {"answers": 3, "images": 2}}
    pack = _write_pack(tmp_path, _format_2_lines(), manifest)
    monkeypatch.setattr(backbone_pack, "PACK_FORMATS", (1,))

    with pytest.raises(PackError, match="format 2"):
        pack_records(pack)


def test_unknown_formats_and_a_format_2_without_counts_are_refused(tmp_path):
    pack = _write_pack(tmp_path, _format_2_lines(), {"format": 3})
    with pytest.raises(PackError):
        pack_records(pack)

    (pack / "manifest.json").write_text(json.dumps({"format": 2}), encoding="utf-8")
    with pytest.raises(PackError, match="home_validation"):
        pack_records(pack)


def test_a_mark_must_name_an_answer_on_its_own_line(tmp_path):
    lines = [{"h": "a1" + "0" * 62, "b": {"c1": 1}, "v": ["c9"]}]
    manifest = {"format": 2, "home_validation": {"answers": 1, "images": 1}}
    pack = _write_pack(tmp_path, lines, manifest)

    with pytest.raises(PackError, match="c9"):
        pack_records(pack)


def test_marks_that_disagree_with_the_manifest_are_refused(tmp_path):
    """The manifest says marks exist; a labels file that lost them would
    otherwise train on every validation answer without a word."""
    lines = _format_2_lines()
    for line in lines:
        line.pop("v", None)
    manifest = {"format": 2, "home_validation": {"answers": 3, "images": 2}}
    pack = _write_pack(tmp_path, lines, manifest)

    with pytest.raises(PackError, match="manifest"):
        pack_records(pack)
    # A --limit trial reads part of the pack, so the totals cannot be compared.
    records, counts = pack_records(pack, limit=2)
    assert len(records) == 2 and counts["held_answers"] == 0


# -- the split ----------------------------------------------------------------


def _todays_build_and_split(records, vocab, validation_split):
    """``_build_samples`` and ``_split_samples`` as they were before ISSUE-1718,
    copied here as the reference."""
    samples = []
    seen = set()
    for record in records:
        path = next(
            (p for p in record.get("file_paths") or [] if Path(p).is_file()), None
        )
        if path is None:
            continue
        binary = {}
        for concept, label in (record.get("binary") or {}).items():
            lowered = str(label).lower()
            key = bb._safe_module_key(concept)
            if lowered in bb._POSITIVE_LABELS:
                binary[key] = 1.0
            elif lowered in bb._NEGATIVE_LABELS:
                binary[key] = 0.0
        groups = {}
        for group, class_name in (record.get("groups") or {}).items():
            key = bb._safe_module_key(group)
            classes = vocab.groups.get(key)
            if classes and str(class_name) in classes:
                groups[key] = classes.index(str(class_name))
        if not (binary or groups):
            continue
        dedup_key = str(record.get("content_hash") or path)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        samples.append((path, binary, groups))
    hash_by_path = {}
    for record in records:
        for path in record.get("file_paths") or []:
            hash_by_path[path] = record.get("content_hash") or path
    train, val = [], []
    for sample in samples:
        digest = hash_by_path.get(sample[0], sample[0])
        try:
            bucket = int(str(digest)[:8], 16) % 10_000 / 10_000
        except ValueError:
            bucket = (hash(str(digest)) % 10_000) / 10_000
        (val if bucket < validation_split else train).append(sample)
    if not train:
        train, val = val, []
    return train, val


def _as_tuples(samples):
    return [(s.path, s.binary, s.groups) for s in samples]


def test_with_no_held_answer_the_split_is_todays(tmp_path):
    records = []
    for i in range(200):
        path = _touch(tmp_path, f"img{i}.png")
        binary = {}
        if i % 3:
            binary["c"] = "positive" if i % 2 else "negative"
        if i % 5:
            binary["d.e"] = "explicit_known_negative" if i % 4 else "1"
        groups = {"g": ("k1", "k2", "k3")[i % 3]} if i % 7 else {}
        if i % 11 == 0:
            binary["junk"] = "unknown"
        record = _record(i * 7919, path, binary=binary, groups=groups)
        if i % 13 == 0:
            record["content_hash"] = "not-hex-" + str(i)
        records.append(record)
    records.append(dict(records[5]))  # a duplicate, skipped by both
    vocab = bb._Vocab(records)

    for split in (0.15, 0.5):
        samples, _missing = bb._build_samples(records, vocab)
        train, val = bb._split_samples(records, samples, split)
        ref_train, ref_val = _todays_build_and_split(records, vocab, split)

        assert _as_tuples(train) == ref_train
        assert _as_tuples(val) == ref_val
        assert all(not s.held_binary and not s.held_groups for s in samples)
    assert bb.validation_source(records) == "content_hash"


def test_a_held_yes_never_trains_in_any_epoch_or_seed(tmp_path):
    records = _concept_c_records(tmp_path)
    held = _held_pairs(records)
    vocab, train, val = _split(records)
    assert bb.validation_source(records) == "home_split"

    # The training samples carry no held answer at all.
    assert not held & _trained_pairs(train)
    for neg_pos_ratio in (5.0, 0.0):  # 0: uncapped, every unlabelled image drawn
        for policy in (None, _SOFT):
            for seed in range(3):
                for epoch in range(5):
                    planned, stats = bb._plan_epoch_samples(
                        train,
                        vocab,
                        epoch,
                        seed=seed,
                        neg_pos_ratio=neg_pos_ratio,
                        label_policy=policy,
                    )
                    # Not as a label, not as an implied no, not as a replica.
                    assert not held & _trained_pairs(planned)
                    assert stats["c"]["pos_total"] == 5
                    # Five positives: oversampled, from the trained five only.
                    assert stats["c"]["oversample_factor"] > 1
                    if policy and neg_pos_ratio == 0.0:
                        # Every image with no answer for c was drawn: the 40.
                        assert stats["c"]["neg_implicit"] == 40
    # The held images still train the answer that is not held.
    held_paths = {path for path, _name in held}
    trained_b = {s.path for s in train if "b" in s.binary}
    assert held_paths <= trained_b


def test_an_image_holding_any_answer_is_never_an_implied_negative(tmp_path):
    """At home a concept's validation no's include other concepts' validation
    yes images. An image held for ``d`` (or for group ``g``) may be a validation
    no for ``c`` with no ``c`` answer in the pack, so it is never drawn as an
    implied negative for any head; its other answers still train."""
    records = []
    i = 0
    for _ in range(6):
        records.append(
            _record(i, _touch(tmp_path, f"p{i}.png"), binary={"c": "positive"})
        )
        i += 1
    held_paths = set()
    for k in range(6):  # held for d, with an explicit x answer that trains
        path = _touch(tmp_path, f"d{i}.png")
        held_paths.add(str(path))
        records.append(
            _record(
                i,
                path,
                binary={"d": "positive", "x": "positive" if k % 2 else "negative"},
                held=["d"],
            )
        )
        i += 1
    for k in range(6):  # held for the group only
        path = _touch(tmp_path, f"g{i}.png")
        held_paths.add(str(path))
        records.append(
            _record(
                i,
                path,
                binary={"x": "positive" if k % 2 else "negative"},
                groups={"g": ("k1", "k2")[k % 2]},
                held=["g"],
            )
        )
        i += 1
    for k in range(30):  # nothing held: these may be drawn
        path = _touch(tmp_path, f"u{i}.png")
        records.append(
            _record(
                i,
                path,
                binary={"x": "positive" if k % 2 else "negative"},
                groups={"g": ("k1", "k2")[k % 2]},
            )
        )
        i += 1
    vocab, train, _val = _split(records)
    assert set(vocab.concepts) == {"c", "d", "x"}

    for max_prior in (None, 0.5):
        policy = {**_SOFT, "implicit_negative_max_prior": max_prior}
        for seed in range(3):
            for epoch in range(4):
                for ratio in (5.0, 0.0):
                    planned, stats = bb._plan_epoch_samples(
                        train,
                        vocab,
                        epoch,
                        seed=seed,
                        neg_pos_ratio=ratio,
                        label_policy=policy,
                    )
                    implied = {
                        (s.path, head)
                        for s in planned
                        for head, value in s.binary.items()
                        if isinstance(value, bb._ImpliedNegative)
                    }
                    assert implied, "the test must draw some implied negatives"
                    assert not {path for path, _head in implied} & held_paths
                    if ratio == 0.0:
                        # c is unlabelled on 42 images; only the 30 hold nothing.
                        assert stats["c"]["neg_implicit"] == 30
                    # The held images' own explicit answers still train.
                    trained_x = {s.path for s in planned if "x" in s.binary}
                    assert held_paths <= trained_x


def test_marks_are_a_set_whatever_their_order(tmp_path):
    """The export sorts ids as plain strings (``c11`` before ``c2``); the order
    and any repeat mean nothing."""
    lines = [
        {
            "h": "a1" + "0" * 62,
            "b": {"c2": 1, "c11": 0},
            "g": {"g3": "k1"},
            "v": ["g3", "c2", "c11", "c2"],
        }
    ]
    manifest = {"format": 2, "home_validation": {"answers": 3, "images": 1}}
    pack = _write_pack(tmp_path, lines, manifest)

    records, counts = pack_records(pack)

    assert set(records[0]["held"]) == {"c2", "c11", "g3"}
    assert len(records[0]["held"]) == 3
    assert counts["held_answers"] == 3 and counts["held_images"] == 1


def test_a_held_answer_is_left_out_of_the_positive_cap_and_the_prior(tmp_path):
    records = _concept_c_records(tmp_path)
    vocab, train, _val = _split(records)

    planned, stats = bb._plan_epoch_samples(
        train, vocab, 0, positive_cap=2, label_policy=_SOFT
    )

    assert stats["c"]["pos"] == 2 and stats["c"]["pos_total"] == 5
    assert not _held_pairs(records) & _trained_pairs(planned)
    # The prior is the share of the images whose answer for c is known to
    # training: the three held images do not count as "not yes".
    assert vocab.priors["c"] == pytest.approx(5 / (len(train) - 3))


def test_a_held_group_answer_gives_no_group_label_in_training(tmp_path):
    records = [
        _record(
            0,
            _touch(tmp_path, "a.png"),
            binary={"c": "positive"},
            groups={"g": "k1"},
            held=["g"],
        ),
        _record(1, _touch(tmp_path, "b.png"), groups={"g": "k2"}),
        _record(2, _touch(tmp_path, "c.png"), groups={"g": "k1"}),
    ]
    vocab, train, val = _split(records)
    a = str(tmp_path / "a.png")

    assert vocab.groups == {"g": ["k1", "k2"]}  # built from every answer
    (sample,) = [s for s in train if s.path == a]
    assert sample.groups == {} and sample.binary == {"c": 1.0}
    for epoch in range(3):
        planned, _stats = bb._plan_epoch_samples(train, vocab, epoch)
        assert all("g" not in s.groups for s in planned if s.path == a)
    assert _as_tuples(val) == [(a, {}, {"g": 0})]


def test_validation_is_the_held_answers_and_nothing_else(tmp_path):
    records = _concept_c_records(tmp_path)
    records.append(
        _record(
            99,
            _touch(tmp_path, "both.png"),
            binary={"c": "negative", "b": "positive", "x": "positive"},
            groups={"g": "k2"},
            held=["b", "g"],
        )
    )
    records.append(_record(100, _touch(tmp_path, "g1.png"), groups={"g": "k1"}))
    _vocab, train, val = _split(records)
    both = str(tmp_path / "both.png")

    expected = {
        (s["file_paths"][0], name) for s in records for name in s.get("held") or []
    }
    assert _trained_pairs(val) == expected
    assert len(val) == 4  # three held c images and the one with both kinds
    by_path = {s.path: s for s in val}
    assert by_path[both].binary == {"b": 1.0} and by_path[both].groups == {"g": 1}
    # The same image trains its other answers, disjoint from what it validates.
    (trained,) = [s for s in train if s.path == both]
    assert trained.binary == {"c": 0.0, "x": 1.0} and trained.groups == {}
    assert set(trained.held_binary) == {"b"} and set(trained.held_groups) == {"g"}
    # No image is held out by hash: every image with a trainable answer trains.
    assert len(train) == len(records)


def test_an_image_whose_answers_are_all_held_only_validates(tmp_path):
    records = [
        _record(0, _touch(tmp_path, "a.png"), binary={"c": "positive"}, held=["c"]),
        _record(1, _touch(tmp_path, "b.png"), binary={"c": "negative"}),
    ]
    _vocab, train, val = _split(records)

    assert [s.path for s in train] == [str(tmp_path / "b.png")]
    assert _as_tuples(val) == [(str(tmp_path / "a.png"), {"c": 1.0}, {})]


def test_nothing_left_to_train_is_refused_not_swapped(tmp_path):
    """Today a split that leaves training empty swaps the two sides. With held
    answers that would train on all of them, so the run is refused."""
    records = [
        _record(0, _image(tmp_path, 0), binary={"c": "positive"}, held=["c"]),
        _record(1, _image(tmp_path, 1), binary={"c": "negative"}, held=["c"]),
    ]
    _vocab, train, val = _split(records)
    assert train == [] and len(val) == 2

    task = bt.BackboneTask(
        {
            "records": records,
            "candidate_checkpoint_path": str(tmp_path / "c.safetensors"),
            "training_config": {"device": "cpu"},
        }
    )
    with pytest.raises(RuntimeError, match="held"):
        task.make_context(None, None, None, None)


# -- a whole run --------------------------------------------------------------


def _held_records(root: Path, n: int = 16) -> list[dict]:
    """``watermark`` held on images 0-5, ``shot_type`` held on images 3-7."""
    records = []
    for i in range(n):
        held = []
        if i < 6:
            held.append("watermark")
        if 3 <= i < 8:
            held.append("shot_type")
        records.append(
            _record(
                i,
                _image(root, i),
                binary={"watermark": "positive" if i % 2 else "negative"},
                groups={"shot_type": "closeup" if i % 2 else "wide"},
                held=held,
            )
        )
    return records


def _request(root: Path, records, *, run_id="home", config=None) -> dict:
    training_config = {
        "image_size": 48,
        "batch_size": 4,
        "epochs": 2,
        "device": "cpu",
        "use_amp": False,
        "label_policy": dict(_SOFT),
        "neg_pos_ratio": 0.0,
    }
    training_config.update(config or {})
    return {
        "run_id": run_id,
        "convnextv2_size": "atto",
        "candidate_checkpoint_path": str(root / "candidates" / f"{run_id}.safetensors"),
        "records": records,
        "training_config": training_config,
        "backbone_init": {"source": "random_init", "checkpoint_path": None},
    }


def _metadata(path) -> dict:
    with safe_open(str(path), framework="pt") as handle:
        return dict(handle.metadata())


def _spy_plans_and_validation(monkeypatch):
    seen = {"planned": [], "val": []}
    real_plan = bb._plan_epoch_samples
    real_collect = bb._collect_val_scores

    def _plan(*args, **kwargs):
        planned, stats = real_plan(*args, **kwargs)
        seen["planned"].append(planned)
        return planned, stats

    def _collect(*args, **kwargs):
        binary, groups = real_collect(*args, **kwargs)
        seen["val"].append((binary, groups))
        return binary, groups

    monkeypatch.setattr(bb, "_plan_epoch_samples", _plan)
    monkeypatch.setattr(bb, "_collect_val_scores", _collect)
    return seen


def test_a_run_validates_on_held_answers_and_says_so(tmp_path, monkeypatch):
    use_stub_trunk(monkeypatch)
    seen = _spy_plans_and_validation(monkeypatch)
    records = _held_records(tmp_path / "imgs")
    held = _held_pairs(records)
    frames: list[dict] = []

    result = asyncio.run(
        run_backbone_training(
            _request(tmp_path, records), progress_callback=frames.append
        )
    )

    assert seen["planned"] and seen["val"]
    for planned in seen["planned"]:
        assert not held & _trained_pairs(planned)
    for binary, groups in seen["val"]:
        assert len(binary["watermark"][1]) == 6  # the six held answers
        assert len(groups["shot_type"][1]) == 5
    assert result["validation_source"] == "home_split"
    assert result["held_answers"] == 11 and result["held_images"] == 8
    preparing = next(f for f in frames if "train_samples" in f)
    assert preparing["validation_source"] == "home_split"
    assert preparing["val_samples"] == 8
    # Images 3-5 have every answer held: they validate and train nothing.
    assert preparing["train_samples"] == 13
    metadata = _metadata(result["candidate_checkpoint_path"])
    assert metadata["validation_source"] == "home_split"
    assert metadata["held_answers"] == "11" and metadata["held_images"] == "8"


def test_a_run_without_held_answers_says_content_hash(tmp_path, monkeypatch):
    use_stub_trunk(monkeypatch)
    records = _held_records(tmp_path / "imgs")
    for record in records:
        record.pop("held", None)

    result = asyncio.run(run_backbone_training(_request(tmp_path, records)))

    assert result["validation_source"] == "content_hash"
    assert result["held_answers"] == 0 and result["held_images"] == 0
    metadata = _metadata(result["candidate_checkpoint_path"])
    assert metadata["validation_source"] == "content_hash"


# -- the pack runner ----------------------------------------------------------


def _format_2_pack(root: Path) -> tuple[Path, int, int]:
    """The pack runner's own test pack, upgraded to format 2: the first yes/no
    answer of every third line and the group answer of every fourth marked."""
    from bittrainer.tests.test_backbone_pack_train import _make_pack

    pack = _make_pack(root)
    lines = [
        json.loads(raw)
        for raw in (pack / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    answers = images = 0
    for index, line in enumerate(lines):
        marks = []
        if index % 3 == 0 and line.get("b"):
            marks.append(sorted(line["b"])[0])
        if index % 4 == 0 and line.get("g"):
            marks.extend(line["g"])
        if marks:
            line["v"] = sorted(marks)
            answers += len(marks)
            images += 1
    (pack / "labels.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
    )
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(format=2, home_validation={"answers": answers, "images": images})
    (pack / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return pack, answers, images


def test_the_runner_records_the_validation_source(tmp_path, monkeypatch):
    use_stub_trunk(monkeypatch)
    # Only an Apache start trains (Bitcrush ISSUE-1721); faked, no download.
    monkeypatch.setattr(runner, "write_v1_trunk", fake_v1_converter)
    pack, answers, images = _format_2_pack(tmp_path)
    out = tmp_path / "run"
    options = {
        "size": "atto",
        "image_size": 64,
        "epochs": 1,
        "batch_size": 4,
        "device": "cpu",
        "amp": "off",
        "init": "apache",
        "workers": 0,
        "soft_negatives": True,
    }

    summary = runner.train_from_pack(pack, out, **options)

    assert summary["status"] == "finished"
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["validation_source"] == "home_split"
    assert (result["held_answers"], result["held_images"]) == (answers, images)
    metadata = _metadata(out / "backbone_candidate.safetensors")
    assert metadata["validation_source"] == "home_split"
    assert int(metadata["held_answers"]) == answers
    assert int(metadata["held_images"]) == images
    preparing = next(
        json.loads(line)
        for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if "train_samples" in line
    )
    assert "home split" in runner._line(preparing)


# -- distillation -------------------------------------------------------------


def _stub_everywhere(monkeypatch):
    use_stub_trunk(monkeypatch)
    monkeypatch.setattr(bd, "create_model", lambda **_kwargs: StubTrunk())


def test_a_distillation_trains_no_held_answer_and_checks_its_teacher(
    tmp_path, monkeypatch
):
    _stub_everywhere(monkeypatch)
    plain = _held_records(tmp_path / "plain")
    for record in plain:
        record.pop("held", None)
    teacher = asyncio.run(
        run_backbone_training(_request(tmp_path / "t", plain, run_id="teacher"))
    )["candidate_checkpoint_path"]
    seen = _spy_plans_and_validation(monkeypatch)
    records = _held_records(tmp_path / "imgs")
    # An image with no answer at all: a label-free sample the teacher teaches.
    records.append(_record(50, _image(tmp_path / "imgs", 50)))
    held = _held_pairs(records)
    frames: list[dict] = []
    request = _request(
        tmp_path / "s", records, run_id="student", config={"max_steps": 3}
    )
    request["distillation"] = {"teacher_checkpoint_path": teacher}

    result = asyncio.run(
        run_backbone_training(request, progress_callback=frames.append)
    )

    for planned in seen["planned"]:
        assert not held & _trained_pairs(planned)
    assert result["validation_source"] == "home_split"
    # The teacher was not trained on this split: it may have trained on the
    # held answers, and its answers would carry them into the student.
    warnings = [f for f in frames if f.get("stage") == "warning"]
    assert [f["warning"] for f in warnings] == ["teacher_not_home_split"]
    metadata = _metadata(result["candidate_checkpoint_path"])
    used = json.loads(metadata["distillation_json"])
    assert used["teacher_validation_source"] == "content_hash"
    assert metadata["validation_source"] == "home_split"


def test_a_teacher_trained_on_the_home_split_raises_no_warning(tmp_path, monkeypatch):
    _stub_everywhere(monkeypatch)
    teacher = asyncio.run(
        run_backbone_training(
            _request(tmp_path / "t", _held_records(tmp_path / "t"), run_id="teacher")
        )
    )["candidate_checkpoint_path"]
    frames: list[dict] = []
    request = _request(
        tmp_path / "s",
        _held_records(tmp_path / "imgs"),
        run_id="student",
        config={"max_steps": 2},
    )
    request["distillation"] = {"teacher_checkpoint_path": teacher}

    result = asyncio.run(
        run_backbone_training(request, progress_callback=frames.append)
    )

    assert not [f for f in frames if f.get("stage") == "warning"]
    used = json.loads(
        _metadata(result["candidate_checkpoint_path"])["distillation_json"]
    )
    assert used["teacher_validation_source"] == "home_split"


def test_a_distillation_without_held_answers_is_as_before(tmp_path, monkeypatch):
    _stub_everywhere(monkeypatch)
    plain = _held_records(tmp_path / "plain")
    for record in plain:
        record.pop("held", None)
    teacher = asyncio.run(
        run_backbone_training(_request(tmp_path / "t", plain, run_id="teacher"))
    )["candidate_checkpoint_path"]
    frames: list[dict] = []
    request = _request(tmp_path / "s", plain, run_id="student", config={"max_steps": 2})
    request["distillation"] = {"teacher_checkpoint_path": teacher}

    asyncio.run(run_backbone_training(request, progress_callback=frames.append))

    assert not [f for f in frames if f.get("stage") == "warning"]


# -- head-only retraining -----------------------------------------------------


def test_head_only_retraining_uses_the_held_answers_the_same_way(tmp_path, monkeypatch):
    from bittrainer.embedding_cache import EmbeddingCache
    from bittrainer.tests.test_backbone_heads_task import (
        _heads_request,
        _run_heads,
        _train_source,
    )

    source = _train_source(tmp_path / "source")
    request = _heads_request(tmp_path, source, max_steps=4)
    for i, record in enumerate(request["records"]):
        held = (["watermark"] if i < 3 else []) + (["shot_type"] if i == 2 else [])
        if held:
            record["held"] = held
    held = _held_pairs(request["records"])
    seen = _spy_plans_and_validation(monkeypatch)
    cached: list[list[str]] = []
    real_ensure = EmbeddingCache.ensure

    def _ensure(self, samples, *args, **kwargs):
        cached.append([sample["path"] for sample in samples])
        return real_ensure(self, samples, *args, **kwargs)

    monkeypatch.setattr(EmbeddingCache, "ensure", _ensure)

    result = _run_heads(request)

    assert result["validation_source"] == "home_split"
    assert result["held_answers"] == 4 and result["held_images"] == 3
    for planned in seen["planned"]:
        assert not held & _trained_pairs(planned)
    for binary, groups in seen["val"]:
        assert len(binary["watermark"][1]) == 3
        assert len(groups["shot_type"][1]) == 1
    # An image both trains and validates; its vector is built once.
    (paths,) = cached
    assert len(paths) == len(set(paths)) == len(request["records"])
    metadata = _metadata(result["candidate_checkpoint_path"])
    assert metadata["validation_source"] == "home_split"
