"""Every model descends from the Apache ConvNeXt V1 weights (Bitcrush ISSUE-1721).

The owner's rule: everything goes on the Apache weights, so everything has the
same lineage. ``bittrainer.lineage`` reads and extends the ``lineage_json`` a
candidate carries; ``backbone_pack_train`` refuses any other start (exit 2);
a distillation refuses a teacher that does not descend from them; every
candidate written records where it came from, one step per stage.

The Base backbone already training on the VM was started from 5982d6e, before
``lineage_json`` existed. Its candidate records the Apache start only in its
init fields, and must still be accepted as a teacher.

CPU-only, ``atto`` backbone, a couple of dozen generated 64px images; the V1
conversion is faked (no download).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import bittrainer.generic.tasks.backbone_distill_task as bd
import bittrainer.tools.backbone_pack_train as runner
from bittrainer import lineage
from bittrainer.backbone_pack import pack_records
from bittrainer.backbone_trainer import run_backbone_training
from bittrainer.model import create_model
from bittrainer.tests.backbone_stub import StubTrunk, fake_v1_converter, use_stub_trunk
from bittrainer.tests.test_backbone_pack_train import _make_pack, _options

_ATTO_ROOT = "apache-convnext-v1:convnext_atto.d2_in1k"
_BASE_ROOT = "apache-convnext-v1:convnext_base.fb_in22k_ft_in1k"

# What a 5982d6e candidate started with --init apache says about its start.
_LEGACY_APACHE = {
    "convnextv2_size": "base",
    "initialization_source": "convnext_v1_converted",
    "license_provenance": "timm/convnext_base.fb_in22k_ft_in1k",
    "pretrained_model_id": "timm/convnext_base.fb_in22k_ft_in1k",
    "pretrained_license": "apache-2.0",
    "non_commercial_only": "false",
    "external_pretrained_used": "true",
}
# ... and one started with --init timm (the default then).
_LEGACY_TIMM = {
    "convnextv2_size": "base",
    "initialization_source": "timm_pretrained",
    "license_provenance": "timm/convnextv2_base.fcmae_ft_in22k_in1k",
    "pretrained_model_id": "timm/convnextv2_base.fcmae_ft_in22k_in1k",
    "pretrained_license": "CC-BY-NC-4.0",
    "non_commercial_only": "true",
    "external_pretrained_used": "true",
}


@pytest.fixture(autouse=True)
def _no_download(monkeypatch):
    monkeypatch.setattr(runner, "write_v1_trunk", fake_v1_converter)


def _metadata(path) -> dict:
    with safe_open(str(path), framework="pt") as handle:
        return dict(handle.metadata() or {})


def _rewrite_header(source: Path, dest: Path, *, drop=(), **values) -> Path:
    """*source* with its metadata changed: keys in *drop* removed, *values* set."""
    metadata = {k: v for k, v in _metadata(source).items() if k not in drop}
    metadata.update(values)
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_file(load_file(str(source)), str(dest), metadata=metadata)
    return dest


def _lineage(path) -> dict:
    return json.loads(_metadata(path)["lineage_json"])


# -- the lineage module -------------------------------------------------------


def test_the_root_comes_from_lineage_json_as_text_or_as_a_dict():
    record = {"root": _ATTO_ROOT, "steps": []}

    assert lineage.apache_root({"lineage_json": json.dumps(record)}) == _ATTO_ROOT
    assert lineage.apache_root({"lineage_json": record}) == _ATTO_ROOT
    assert lineage.apache_root(record) == _ATTO_ROOT


def test_a_5982d6e_candidate_derives_its_root_from_its_init_fields():
    assert lineage.apache_root(_LEGACY_APACHE) == _BASE_ROOT
    assert lineage.lineage_of(_LEGACY_APACHE) == {
        "root": _BASE_ROOT,
        "steps": [{"kind": "backbone", "size": "base", "from_trunk": None}],
    }
    assert lineage.require_apache(_LEGACY_APACHE, "the teacher") == _BASE_ROOT


def test_a_converted_v1_trunk_is_a_root_with_no_steps():
    trunk = {
        "converted_from": "timm/convnext_nano.in12k_ft_in1k",
        "license": "apache-2.0",
    }

    assert lineage.lineage_of(trunk) == {
        "root": "apache-convnext-v1:convnext_nano.in12k_ft_in1k",
        "steps": [],
    }


@pytest.mark.parametrize(
    ("meta", "said"),
    [
        (_LEGACY_TIMM, "CC-BY-NC-4.0"),
        ({"initialization_source": "random_init"}, "random"),
        ({}, "no lineage"),
        # A 5982d6e student: its own start was Apache, its teacher's is unknown.
        ({**_LEGACY_APACHE, "distilled_from": '{"convnextv2_size":"base"}'}, "teacher"),
        # A recorded lineage is believed over the init fields.
        (
            {
                **_LEGACY_APACHE,
                "lineage_json": '{"root":"timm:convnextv2_base","steps":[]}',
            },
            "timm:convnextv2_base",
        ),
        # An Apache licence on V2 weights is not the V1 conversion.
        ({**_LEGACY_APACHE, "pretrained_model_id": "timm/convnextv2_base.x"}, "V1"),
    ],
)
def test_anything_else_has_no_apache_root_and_says_why(meta, said):
    assert lineage.apache_root(meta) is None
    with pytest.raises(lineage.LineageError) as refused:
        lineage.require_apache(meta, "the teacher")
    assert "the teacher" in str(refused.value)
    assert said in str(refused.value)


def test_extend_appends_one_step_and_keeps_the_root():
    start = {"lineage_json": json.dumps({"root": _ATTO_ROOT, "steps": []})}
    step = {"kind": "backbone", "size": "atto", "from_trunk": None}

    once = lineage.extend(start, step)
    twice = lineage.extend(
        {"lineage_json": once}, {"kind": "group", "size": "atto", "from_trunk": "ab"}
    )

    assert once == {"root": _ATTO_ROOT, "steps": [step]}
    assert twice["root"] == _ATTO_ROOT
    assert [s["kind"] for s in twice["steps"]] == ["backbone", "group"]
    with pytest.raises(lineage.LineageError):
        lineage.extend(_LEGACY_TIMM, step)
    with pytest.raises(ValueError):
        lineage.extend(start, {"kind": "finetune", "size": "atto", "from_trunk": None})


# -- 1. only an Apache start trains --------------------------------------------


def _cli(pack: Path, out: Path, *extra: str) -> list[str]:
    return [
        "--pack", str(pack), "--out", str(out), "--size", "atto",
        "--image-size", "64", "--epochs", "1", "--batch-size", "4",
        "--device", "cpu", "--amp", "off", "--workers", "0", *extra,
    ]  # fmt: skip


@pytest.mark.parametrize("init", ["timm", "random"])
def test_a_timm_or_random_start_is_refused_with_exit_2(tmp_path, init, capsys):
    pack = _make_pack(tmp_path)
    out = tmp_path / "run"

    assert runner.main(_cli(pack, out, "--init", init)) == 2

    assert "Apache" in capsys.readouterr().out
    assert not (out / "result.json").exists()
    assert not (out / "progress.jsonl").exists()


def test_a_local_trunk_without_an_apache_lineage_is_refused_with_exit_2(tmp_path):
    pack = _make_pack(tmp_path)
    trunk = create_model(model_size="atto", pretrained=False, num_classes=0)
    bare = tmp_path / "fcmae.safetensors"
    save_file(dict(trunk.state_dict()), str(bare))
    timm = tmp_path / "timm.safetensors"
    save_file(dict(trunk.state_dict()), str(timm), metadata=_LEGACY_TIMM)

    for path in (bare, timm):
        out = tmp_path / path.stem
        assert runner.main(_cli(pack, out, "--init-checkpoint", str(path))) == 2
        assert not (out / "result.json").exists()


def test_a_local_trunk_with_an_apache_lineage_continues_it(tmp_path):
    pack = _make_pack(tmp_path)
    trunk = fake_v1_converter("atto", tmp_path / "v1.safetensors")["path"]

    request, _ = runner.build_request(
        pack, tmp_path / "run", **_options(init_checkpoint=trunk)
    )

    assert request["backbone_init"]["checkpoint_path"] == trunk
    assert request["lineage_json"] == {
        "root": _ATTO_ROOT,
        "steps": [
            {
                "kind": "backbone",
                "size": "atto",
                "from_trunk": lineage.trunk_hash(trunk),
            }
        ],
    }


def test_the_default_start_is_apache(monkeypatch):
    seen: dict = {}

    def _fake(pack, out, **kw):
        seen.update(kw)
        return {"status": "finished"}

    monkeypatch.setattr(runner, "train_from_pack", _fake)
    assert runner.main(["--pack", "p", "--out", "o"]) == 0
    assert seen["init"] == "apache"


# -- 2 and 3. teachers, and what every candidate records ---------------------


@pytest.fixture(scope="module")
def taught(tmp_path_factory):
    """One Apache-started teacher, trained from a pack through the runner."""
    root = tmp_path_factory.mktemp("lineage")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "write_v1_trunk", fake_v1_converter)
        pack = _make_pack(root)
        out = root / "large"
        summary = runner.train_from_pack(pack, out, **_options(init="apache"))
    assert summary["status"] == "finished"
    return pack, out / "backbone_candidate.safetensors", root


def test_a_backbone_candidate_records_its_root_and_one_step(taught):
    _pack, teacher, _root = taught

    assert _lineage(teacher) == {
        "root": _ATTO_ROOT,
        "steps": [{"kind": "backbone", "size": "atto", "from_trunk": None}],
    }
    result = json.loads((teacher.parent / "result.json").read_text(encoding="utf-8"))
    assert result["lineage_json"] == _lineage(teacher)


def test_an_apache_teacher_and_an_apache_student_distil_and_record_both(taught):
    pack, teacher, root = taught
    out = root / "small"

    summary = runner.train_from_pack(
        pack, out, **_options(init="apache", teacher=str(teacher))
    )

    assert summary["status"] == "finished"
    student = out / "backbone_candidate.safetensors"
    assert _lineage(student) == {
        "root": _ATTO_ROOT,
        "steps": [
            {"kind": "backbone", "size": "atto", "from_trunk": None},
            {
                "kind": "distil",
                "size": "atto",
                "from_trunk": lineage.trunk_hash(teacher),
                "init": _ATTO_ROOT,
            },
        ],
    }
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["lineage_json"] == _lineage(student)


def test_a_5982d6e_teacher_with_an_apache_init_is_accepted(taught):
    """The Base on the VM: Apache init fields, no lineage_json."""
    pack, teacher, root = taught
    legacy = _rewrite_header(
        teacher,
        root / "legacy" / "backbone_candidate.safetensors",
        drop=("lineage_json",),
        initialization_source="convnext_v1_converted",
        pretrained_model_id="timm/convnext_atto.d2_in1k",
        pretrained_license="apache-2.0",
        non_commercial_only="false",
    )
    out = root / "from_legacy"

    summary = runner.train_from_pack(
        pack, out, **_options(init="apache", teacher=str(legacy), max_steps=2)
    )

    assert summary["status"] == "finished"
    record = _lineage(out / "backbone_candidate.safetensors")
    assert record["root"] == _ATTO_ROOT
    assert [step["kind"] for step in record["steps"]] == ["backbone", "distil"]


@pytest.mark.parametrize(
    "header",
    [
        _LEGACY_TIMM,
        {"initialization_source": "random_init"},
        {"lineage_json": '{"root":"timm:convnextv2_atto","steps":[]}'},
    ],
)
def test_a_teacher_without_an_apache_root_is_refused_with_exit_2(
    taught, tmp_path, header, capsys
):
    pack, teacher, _root = taught
    keys = ("lineage_json", *(key for key in _LEGACY_TIMM if key != "convnextv2_size"))
    other = _rewrite_header(teacher, tmp_path / "t.safetensors", drop=keys, **header)
    out = tmp_path / "student"

    assert (
        runner.main(_cli(pack, out, "--init", "apache", "--teacher", str(other))) == 2
    )

    assert "teacher" in capsys.readouterr().out
    assert not (out / "progress.jsonl").exists()


def test_a_student_of_a_teacher_with_no_lineage_claims_none(tmp_path, monkeypatch):
    """Through the task alone (Engine's path, no runner gate): the student
    must not claim a lineage its teacher does not have."""
    use_stub_trunk(monkeypatch)
    monkeypatch.setattr(bd, "create_model", lambda **_kwargs: StubTrunk())
    records, _ = pack_records(_make_pack(tmp_path))

    def _request(name, **more):
        return {
            "run_id": name,
            "convnextv2_size": "atto",
            "candidate_checkpoint_path": str(tmp_path / f"{name}.safetensors"),
            "records": records,
            "training_config": {
                "image_size": 48,
                "batch_size": 4,
                "epochs": 1,
                "device": "cpu",
                "use_amp": False,
                "max_steps": 2,
            },  # fmt: skip
            "backbone_init": {"source": "random_init", "checkpoint_path": None},
            **more,
        }

    teacher = asyncio.run(run_backbone_training(_request("teacher")))
    student = asyncio.run(
        run_backbone_training(
            _request(
                "student",
                distillation={
                    "teacher_checkpoint_path": teacher["candidate_checkpoint_path"]
                },
            )
        )
    )

    assert "lineage_json" not in _metadata(student["candidate_checkpoint_path"])
