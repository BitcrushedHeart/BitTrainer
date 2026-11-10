"""Train the backbone from an exported pack (Bitcrush ISSUE-1655).

The Suite's ``tools/backbone_pack_export.py`` writes a pack: shrunk images named
by content hash and labels keyed by head ids, with no names in it.
``bittrainer.backbone_pack`` reads that format into the trainer's records, and
``bittrainer.tools.backbone_pack_train`` runs the training headless: progress to
a file, a heartbeat inside each epoch, backups, pause / stop / resume, and a
refusal to start over on top of another run's backups.

CPU-only, ``atto`` backbone, a couple of dozen generated 64px images.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors import safe_open

import bittrainer.backbone_trainer as bb
import bittrainer.tools.backbone_pack_train as runner
from bittrainer.backbone_pack import PackError, pack_records, read_manifest
from bittrainer.generic.tasks.backbone_distill_task import BackboneDistillTask
from bittrainer.generic.tasks.backbone_task import BackboneTask
from bittrainer.tools.backbone_proof import make_fixture_dataset

_CONCEPT_IDS = {"striped": "c11", "ringed": "c12"}
_CLASS_IDS = {"side_by_side": "k1", "stacked": "k2", "centre_blob": "k3"}


def _make_pack(root: Path, *, images: int = 24) -> Path:
    """A format-1 pack built from the proof tool's generated images."""
    pack = root / "pack"
    lines = []
    for record in make_fixture_dataset(root / "drawn", count=images, image_size=64):
        digest = record["content_hash"]
        target = pack / "images" / digest[:2] / f"{digest}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(record["file_paths"][0], target)
        line: dict = {"h": digest}
        binary = {
            _CONCEPT_IDS[name]: int(answer == "positive")
            for name, answer in record["binary"].items()
        }
        if binary:
            line["b"] = binary
        if record["groups"]:
            line["g"] = {"g5": _CLASS_IDS[record["groups"]["layout"]]}
        lines.append(json.dumps(line))
    lines.sort()
    (pack / "labels.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (pack / "manifest.json").write_text(
        json.dumps({"format": 1, "images": images, "short_side": 512}), encoding="utf-8"
    )
    shutil.rmtree(root / "drawn")
    return pack


def _options(**overrides) -> dict:
    options = {
        "size": "atto",
        "image_size": 64,
        "epochs": 1,
        "batch_size": 4,
        "device": "cpu",
        "amp": "off",
        "init": "random",
        "workers": 0,
        "progress_every_steps": 1,
    }
    options.update(overrides)
    return options


def _frames(out: Path) -> list[dict]:
    lines = (out / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


# -- reading a pack -----------------------------------------------------------


def test_pack_records_are_what_the_trainer_takes(tmp_path):
    pack = _make_pack(tmp_path)

    records, counts = pack_records(pack)

    assert counts == {"images": 24, "missing_files": 0}
    first = records[0]
    assert Path(first["file_paths"][0]).is_file()
    assert Path(first["file_paths"][0]).name == f"{first['content_hash']}.jpg"
    assert first["splits"] == {"train": 1}
    answers = {answer for record in records for answer in record["binary"].values()}
    assert answers == {"positive", "negative"}
    assert {head for record in records for head in record["binary"]} == {"c11", "c12"}
    assert {
        (head, key) for record in records for head, key in record["groups"].items()
    } == {("g5", "k1"), ("g5", "k2"), ("g5", "k3")}


def test_limit_takes_the_first_records(tmp_path):
    pack = _make_pack(tmp_path)

    limited, counts = pack_records(pack, limit=5)
    everything, _ = pack_records(pack)

    assert counts["images"] == 5
    assert limited == everything[:5]


def test_a_missing_image_is_counted_and_left_out(tmp_path):
    pack = _make_pack(tmp_path)
    victim = next((pack / "images").glob("*/*.jpg"))
    victim.unlink()

    records, counts = pack_records(pack)

    assert counts == {"images": 23, "missing_files": 1}
    assert victim.stem not in {record["content_hash"] for record in records}


def test_only_a_known_pack_format_is_read(tmp_path):
    pack = _make_pack(tmp_path)
    assert read_manifest(pack)["format"] == 1

    (pack / "manifest.json").write_text(json.dumps({"format": 2}), encoding="utf-8")
    with pytest.raises(PackError):
        pack_records(pack)

    (pack / "manifest.json").unlink()
    with pytest.raises(PackError):
        pack_records(pack)


# -- the request --------------------------------------------------------------


def test_options_map_onto_the_training_request(tmp_path):
    pack = _make_pack(tmp_path)
    out = tmp_path / "run"

    request, _counts = runner.build_request(
        pack,
        out,
        **_options(
            size="base",
            init="timm",
            image_size=224,
            finetune_image_size=384,
            finetune_epochs=5,
            epochs=30,
            workers=4,
            channels_last=True,
            amp="auto",
            soft_negatives=True,
            backup_every_steps=1000,
            extra={"neg_pos_ratio": 3, "llrd_decay": 0.75},
        ),
    )

    config = request["training_config"]
    assert request["convnextv2_size"] == "base"
    assert request["candidate_checkpoint_path"] == str(
        out / "backbone_candidate.safetensors"
    )
    assert config["image_size"] == 224 and config["epochs"] == 30
    assert config["finetune_image_size"] == 384 and config["finetune_epochs"] == 5
    assert config["dataloader_workers"] == 4 and config["channels_last"] is True
    assert config["amp_dtype"] == "auto" and config["use_amp"] is True
    assert config["label_policy"]["mode"] == "soft_implicit_negative"
    assert config["backup_dir"] == str(out / "backups")
    assert config["resume_from"] == str(out / "backups")
    assert config["backup_every_steps"] == 1000
    assert config["neg_pos_ratio"] == 3 and config["llrd_decay"] == 0.75
    # Pretrained weights are named, with their licence, in what gets exported.
    assert request["backbone_init"] == {"source": "timm_pretrained"}
    assert request["pretrained_model_id"] == "timm/convnextv2_base.fcmae_ft_in22k_in1k"
    assert request["pretrained_license"] == "CC-BY-NC-4.0"
    assert request["non_commercial_only"] is True


def test_a_local_trunk_or_random_weights_carry_no_outside_licence(tmp_path):
    pack = _make_pack(tmp_path)
    trunk = tmp_path / "fcmae.safetensors"

    local, _ = runner.build_request(
        pack, tmp_path / "a", **_options(init_checkpoint=str(trunk))
    )
    random, _ = runner.build_request(pack, tmp_path / "b", **_options())

    assert local["backbone_init"] == {
        "source": "local_candidate",
        "checkpoint_path": str(trunk),
    }
    assert random["backbone_init"]["source"] == "random_init"
    for request in (local, random):
        assert request["external_pretrained_used"] is False
        assert "pretrained_license" not in request


# -- running ------------------------------------------------------------------


def test_a_run_trains_reports_and_exports(tmp_path):
    pack = _make_pack(tmp_path)
    out = tmp_path / "run"

    summary = runner.train_from_pack(pack, out, **_options())

    assert summary["status"] == "finished"
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["epochs_completed"] == 1
    with safe_open(out / "backbone_candidate.safetensors", framework="pt") as candidate:
        metadata = candidate.metadata()
    assert json.loads(metadata["heads_concepts_json"]) == ["c11", "c12"]
    assert list(json.loads(metadata["heads_groups_json"])) == ["g5"]

    frames = _frames(out)
    steps = [frame for frame in frames if frame.get("stage") == "training_step"]
    assert steps, "no heartbeat inside the epoch"
    assert steps[0]["batch"] == 1 and steps[0]["batches"] >= len(steps)
    assert steps[-1]["images_per_second"] > 0
    assert any(frame.get("stage") == "preparing" for frame in frames)
    # A finished run leaves no backups behind to resume by mistake.
    assert not list((out / "backups").glob("backup_*.pt"))


def test_a_paused_run_resumes_where_it_stopped(tmp_path):
    pack = _make_pack(tmp_path)
    out = tmp_path / "run"

    def _pause_at_first_step(frame, control):
        if frame.get("stage") == "training_step":
            control.pause()

    paused = runner.train_from_pack(
        pack, out, on_frame=_pause_at_first_step, **_options(epochs=2)
    )

    assert paused["status"] == "paused"
    assert list((out / "backups").glob("backup_*.pt"))
    assert not (out / "result.json").exists()

    finished = runner.train_from_pack(pack, out, **_options(epochs=2))

    assert finished["status"] == "finished"
    assert any(frame.get("type") == "training_resumed" for frame in _frames(out))
    assert (
        json.loads((out / "result.json").read_text(encoding="utf-8"))[
            "epochs_completed"
        ]
        == 2
    )


def test_another_runs_backups_are_not_trained_over(tmp_path):
    pack = _make_pack(tmp_path)
    out = tmp_path / "run"

    def _pause_at_first_step(frame, control):
        if frame.get("stage") == "training_step":
            control.pause()

    runner.train_from_pack(
        pack, out, on_frame=_pause_at_first_step, **_options(epochs=2)
    )
    kept = sorted(path.name for path in (out / "backups").glob("backup_*.pt"))

    # Same folder, a different run (more epochs): its backups do not fit.
    refused = runner.train_from_pack(pack, out, **_options(epochs=3))

    assert refused["status"] == "refused"
    assert "--fresh" in refused["reason"]
    assert sorted(path.name for path in (out / "backups").glob("backup_*.pt")) == kept
    assert not (out / "result.json").exists()


def test_fresh_discards_old_backups_before_starting(tmp_path):
    pack = _make_pack(tmp_path)
    backups = tmp_path / "run" / "backups"
    backups.mkdir(parents=True)
    stale = backups / "backup_00000007_0001.pt"
    stale.write_bytes(b"not a real backup")

    request, _ = runner.build_request(pack, tmp_path / "run", fresh=True, **_options())

    assert not stale.exists()
    assert request["training_config"]["resume_from"] == str(backups)


def test_a_pack_with_missing_images_is_refused(tmp_path):
    pack = _make_pack(tmp_path)
    next((pack / "images").glob("*/*.jpg")).unlink()

    refused = runner.train_from_pack(pack, tmp_path / "run", **_options())

    assert refused["status"] == "refused"
    assert "1" in refused["reason"] and "--allow-missing" in refused["reason"]
    assert not (tmp_path / "run" / "progress.jsonl").exists()


# -- controlling a detached run -----------------------------------------------


def test_control_files_pause_and_stop_a_run(tmp_path):
    control = runner.Control(tmp_path)
    control.poll()
    assert not control.pause_event.is_set() and not control.stop_event.is_set()

    (tmp_path / "PAUSE").touch()
    control.poll()
    assert control.pause_event.is_set() and not control.stop_event.is_set()

    (tmp_path / "STOP").touch()
    control.poll()
    assert control.stop_event.is_set()


def test_leftover_control_files_do_not_stop_the_next_run(tmp_path):
    (tmp_path / "PAUSE").touch()
    (tmp_path / "STOP").touch()

    control = runner.Control(tmp_path)
    control.clear_files()
    control.poll()

    assert not (tmp_path / "PAUSE").exists() and not (tmp_path / "STOP").exists()
    assert not control.pause_event.is_set() and not control.stop_event.is_set()


# -- starting weights and teachers --------------------------------------------


def test_an_apache_start_converts_a_v1_trunk_into_the_run_once(tmp_path, monkeypatch):
    pack = _make_pack(tmp_path)
    out = tmp_path / "run"
    calls = []

    def _convert(size, path, weights=None):
        calls.append((size, weights))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"a converted trunk")
        return {
            "model_id": "timm/convnext_atto.d2_in1k",
            "license": "apache-2.0",
            "source_url": "https://huggingface.co/timm/convnext_atto.d2_in1k",
            "max_abs_difference": 1e-6,
            "path": str(path),
        }

    monkeypatch.setattr(runner, "write_v1_trunk", _convert)

    first, _ = runner.build_request(pack, out, **_options(init="apache"))
    resumed, _ = runner.build_request(pack, out, **_options(init="apache"))

    assert calls == [("atto", None)]  # a resumed run does not download it again
    for request in (first, resumed):
        assert request["backbone_init"] == {
            "source": "local_candidate",
            "checkpoint_path": str(out / "init_trunk.safetensors"),
        }
        assert request["initialization_source"] == "convnext_v1_converted"
        assert request["pretrained_model_id"] == "timm/convnext_atto.d2_in1k"
        assert request["pretrained_license"] == "apache-2.0"
        assert request["non_commercial_only"] is False
        assert request["external_pretrained_used"] is True


def test_weights_that_are_refused_refuse_the_run(tmp_path, monkeypatch):
    pack = _make_pack(tmp_path)

    def _refuse(size, path, weights=None):
        raise ValueError("convnext_x is declared cc-by-nc-4.0")

    monkeypatch.setattr(runner, "write_v1_trunk", _refuse)

    refused = runner.train_from_pack(pack, tmp_path / "run", **_options(init="apache"))

    assert refused["status"] == "refused"
    assert "cc-by-nc-4.0" in refused["reason"]


def test_a_teacher_turns_the_run_into_a_distillation(tmp_path):
    pack = _make_pack(tmp_path)
    teacher = tmp_path / "base" / "backbone_candidate.safetensors"

    distilled, _ = runner.build_request(
        pack,
        tmp_path / "nano",
        **_options(teacher=str(teacher), distill={"temperature": 3.0}),
    )
    plain, _ = runner.build_request(pack, tmp_path / "plain", **_options())

    assert distilled["distillation"] == {
        "teacher_checkpoint_path": str(teacher),
        "temperature": 3.0,
    }
    assert "distillation" not in plain


def test_a_teacher_is_refused_when_this_build_cannot_distil(tmp_path, monkeypatch):
    """Without the distillation task the trainer would ignore the teacher and
    train an ordinary run, under a name that says otherwise."""
    pack = _make_pack(tmp_path)
    teacher = tmp_path / "teacher.safetensors"
    teacher.write_bytes(b"x")
    monkeypatch.setattr(runner, "_can_distil", lambda: False)

    refused = runner.train_from_pack(
        pack, tmp_path / "run", **_options(teacher=str(teacher))
    )

    assert refused["status"] == "refused"
    assert "distil" in refused["reason"]
    assert not (tmp_path / "run").exists()


def test_a_missing_teacher_file_is_refused(tmp_path, monkeypatch):
    pack = _make_pack(tmp_path)
    monkeypatch.setattr(runner, "_can_distil", lambda: True)

    refused = runner.train_from_pack(
        pack,
        tmp_path / "run",
        **_options(teacher=str(tmp_path / "nowhere.safetensors")),
    )

    assert refused["status"] == "refused"
    assert "teacher" in refused["reason"]


def test_a_teacher_request_runs_as_the_distillation_task():
    teacher = {"distillation": {"teacher_checkpoint_path": "t.safetensors"}}
    assert bb.backbone_task_class(teacher) is BackboneDistillTask
    assert bb.backbone_task_class({}) is BackboneTask
    assert bb.backbone_task_class({"distillation": {}}) is BackboneTask


def test_backups_are_checked_by_the_task_the_run_will_use(monkeypatch):
    """A distillation's backups carry their own fingerprint. Asked of the plain
    task they would look like another run's, and a paused distillation could
    never be resumed."""
    asked = []

    class _Task:
        def __init__(self, request):
            asked.append(request)

        def make_context(self, *_args):
            return SimpleNamespace(resume_state=None)

        def fingerprint_init(self, context):
            context.resume_state = {"epoch": 1}

    monkeypatch.setattr(bb, "backbone_task_class", lambda request: _Task)
    request = {"distillation": {"teacher_checkpoint_path": "t.safetensors"}}

    assert runner._backups_fit(request) is True
    assert asked == [request]


def test_a_paused_distillation_resumes_through_the_runner(tmp_path):
    """The whole path: train a teacher from the pack, distil it, pause, resume."""
    pack = _make_pack(tmp_path)
    teacher_run = tmp_path / "teacher"
    assert (
        runner.train_from_pack(pack, teacher_run, **_options())["status"] == "finished"
    )
    teacher = str(teacher_run / "backbone_candidate.safetensors")
    out = tmp_path / "student"

    def _pause_at_first_step(frame, control):
        if frame.get("stage") == "training_step":
            control.pause()

    paused = runner.train_from_pack(
        pack, out, on_frame=_pause_at_first_step, **_options(epochs=2, teacher=teacher)
    )
    assert paused["status"] == "paused"

    finished = runner.train_from_pack(pack, out, **_options(epochs=2, teacher=teacher))

    assert finished["status"] == "finished"
    assert any(frame.get("type") == "training_resumed" for frame in _frames(out))
    with safe_open(out / "backbone_candidate.safetensors", framework="pt") as candidate:
        metadata = candidate.metadata()
    assert "distilled_from" in metadata
    assert json.loads(metadata["heads_concepts_json"]) == ["c11", "c12"]


def test_weight_and_teacher_options_reach_the_run(monkeypatch, tmp_path):
    seen: dict = {}

    def _fake(pack, out, **kw):
        seen.update(kw)
        return {"status": "finished"}

    monkeypatch.setattr(runner, "train_from_pack", _fake)
    code = runner.main(
        [
            "--pack", str(tmp_path / "pack"),
            "--out", str(tmp_path / "nano"),
            "--size", "nano",
            "--init", "apache",
            "--v1-weights", "convnext_nano.in12k",
            "--teacher", "base/backbone_candidate.safetensors",
            "--distill-temperature", "3",
            "--distill-feature-weight", "0.5",
        ]
    )  # fmt: skip

    assert code == 0
    assert seen["init"] == "apache" and seen["v1_weights"] == "convnext_nano.in12k"
    assert seen["teacher"] == "base/backbone_candidate.safetensors"
    assert seen["distill"] == {"temperature": 3.0, "feature_weight": 0.5}


# -- the command --------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "code"), [("finished", 0), ("refused", 2), ("paused", 3), ("failed", 1)]
)
def test_exit_code_says_how_the_run_ended(monkeypatch, tmp_path, status, code):
    monkeypatch.setattr(
        runner,
        "train_from_pack",
        lambda pack, out, **kw: {"status": status, "reason": "why"},
    )
    assert (
        runner.main(["--pack", str(tmp_path), "--out", str(tmp_path / "run")]) == code
    )


def test_command_line_options_reach_the_run(monkeypatch, tmp_path):
    seen: dict = {}

    def _fake(pack, out, **kw):
        seen.update(kw, pack=Path(pack), out=Path(out))
        return {"status": "finished"}

    monkeypatch.setattr(runner, "train_from_pack", _fake)
    code = runner.main(
        [
            "--pack", str(tmp_path / "pack"),
            "--out", str(tmp_path / "run"),
            "--size", "base",
            "--image-size", "224",
            "--finetune-image-size", "384",
            "--finetune-epochs", "5",
            "--epochs", "30",
            "--batch-size", "64",
            "--workers", "4",
            "--channels-last",
            "--soft-negatives",
            "--limit", "5000",
            "--set", "neg_pos_ratio=3",
            "--set", "positive_cap_mode=random",
            "--fresh",
        ]
    )  # fmt: skip

    assert code == 0
    assert seen["pack"] == tmp_path / "pack" and seen["out"] == tmp_path / "run"
    assert seen["size"] == "base" and seen["init"] == "timm"
    assert (
        seen["image_size"] == 224 and seen["epochs"] == 30 and seen["batch_size"] == 64
    )
    assert seen["finetune_image_size"] == 384 and seen["finetune_epochs"] == 5
    assert seen["workers"] == 4 and seen["channels_last"] is True
    assert (
        seen["soft_negatives"] is True
        and seen["limit"] == 5000
        and seen["fresh"] is True
    )
    assert seen["extra"] == {"neg_pos_ratio": 3, "positive_cap_mode": "random"}
