"""Train the backbone from an exported pack (Bitcrush ISSUE-1655).

The Suite's ``tools/backbone_pack_export.py`` writes a pack: shrunk images named
by content hash and labels keyed by head ids, with no names in it.
``bittrainer.backbone_pack`` reads that format into the trainer's records, and
``bittrainer.tools.backbone_pack_train`` runs the training headless: progress to
a file, a heartbeat inside each epoch, backups, pause / stop / resume, and a
refusal to start over on top of another run's backups.

CPU-only, ``atto`` backbone, a couple of dozen generated 64px images. Every
run starts from the Apache V1 conversion (Bitcrush ISSUE-1721), faked here so
nothing is downloaded.
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
from bittrainer.tests.backbone_stub import fake_v1_converter, use_stub_trunk
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


@pytest.fixture(autouse=True)
def _no_download(monkeypatch):
    monkeypatch.setattr(runner, "write_v1_trunk", fake_v1_converter)


def _options(**overrides) -> dict:
    options = {
        "size": "atto",
        "image_size": 64,
        "epochs": 1,
        "batch_size": 4,
        "device": "cpu",
        "amp": "off",
        "init": "apache",
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


def test_options_map_onto_the_training_request(tmp_path, monkeypatch):
    pack = _make_pack(tmp_path)
    out = tmp_path / "run"

    def _convert(size, path, weights=None):  # a Base trunk is not built here
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"a converted trunk")
        name = "convnext_base.fb_in22k_ft_in1k"
        return {
            "model_id": f"timm/{name}",
            "license": "apache-2.0",
            "source_url": f"https://huggingface.co/timm/{name}",
            "max_abs_difference": 0.0,
            "path": str(path),
        }

    monkeypatch.setattr(runner, "write_v1_trunk", _convert)

    request, _counts = runner.build_request(
        pack,
        out,
        **_options(
            size="base",
            init="apache",
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
    assert request["backbone_init"]["source"] == "local_candidate"
    assert request["pretrained_model_id"] == "timm/convnext_base.fb_in22k_ft_in1k"
    assert request["pretrained_license"] == "apache-2.0"
    assert request["non_commercial_only"] is False


def test_a_local_trunk_carries_no_outside_licence(tmp_path):
    pack = _make_pack(tmp_path)
    trunk = fake_v1_converter("atto", tmp_path / "v1.safetensors")["path"]

    local, _ = runner.build_request(
        pack, tmp_path / "a", **_options(init_checkpoint=trunk)
    )

    assert local["backbone_init"] == {
        "source": "local_candidate",
        "checkpoint_path": trunk,
    }
    assert local["external_pretrained_used"] is False
    assert "pretrained_license" not in local


# -- running ------------------------------------------------------------------


def test_a_run_trains_reports_and_exports(tmp_path):
    pack = _make_pack(tmp_path)
    out = tmp_path / "run"

    summary = runner.train_from_pack(pack, out, **_options())

    assert summary["status"] == "finished"
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["epochs_completed"] == 1
    # No GradScaler on a CPU: nothing skipped, and no scale to report.
    assert result["skipped_steps"] == 0 and result["grad_scale"] is None
    # The exported epoch's raw validation scores sit beside the candidate.
    assert result["val_scores_file"] == "backbone_candidate.val_scores.npz"
    assert (out / result["val_scores_file"]).is_file()
    with safe_open(out / "backbone_candidate.safetensors", framework="pt") as candidate:
        metadata = candidate.metadata()
    assert json.loads(metadata["heads_concepts_json"]) == ["c11", "c12"]
    assert list(json.loads(metadata["heads_groups_json"])) == ["g5"]
    assert metadata["val_scores_file"] == result["val_scores_file"]

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
    fake_v1_converter("atto", teacher)  # a file with an Apache root

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
    assert seen["size"] == "base" and seen["init"] == "apache"
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


# -- the tail's own batch size and the preflight ---------------------------------


def test_the_tail_batch_size_reaches_the_request_only_when_given(tmp_path):
    pack = _make_pack(tmp_path)

    unset, _ = runner.build_request(pack, tmp_path / "a", **_options())
    given, _ = runner.build_request(
        pack, tmp_path / "b", **_options(finetune_batch_size=16)
    )
    probed, _ = runner.build_request(
        pack, tmp_path / "c", **_options(finetune_batch_size=0)
    )

    # Unset means the trainer's own default: the tail keeps batch_size.
    assert "finetune_batch_size" not in unset["training_config"]
    assert given["training_config"]["finetune_batch_size"] == 16
    assert probed["training_config"]["finetune_batch_size"] == 0


def test_the_tail_batch_size_comes_from_the_command_line(monkeypatch, tmp_path):
    seen: dict = {}

    def _fake(pack, out, **kw):
        seen.update(kw)
        return {"status": "finished"}

    monkeypatch.setattr(runner, "train_from_pack", _fake)
    base = ["--pack", str(tmp_path), "--out", str(tmp_path / "run")]

    assert runner.main(base) == 0
    assert seen["finetune_batch_size"] is None
    assert runner.main([*base, "--finetune-batch-size", "16"]) == 0
    assert seen["finetune_batch_size"] == 16
    assert runner.main([*base, "--finetune-batch-size", "auto"]) == 0
    assert seen["finetune_batch_size"] == 0


def test_a_tail_that_does_not_fit_is_a_refused_run_not_a_failed_one(
    tmp_path, monkeypatch
):
    pack = _make_pack(tmp_path)
    out = tmp_path / "run"
    use_stub_trunk(monkeypatch)  # nothing trains here: no need for a real trunk

    def _dry_step(self, ctx, model, image_size, batch_size):
        if image_size == 96:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB.")
        return None

    monkeypatch.setattr(BackboneTask, "_dry_step", _dry_step)

    refused = runner.train_from_pack(
        pack,
        out,
        **_options(
            epochs=2,
            finetune_image_size=96,
            finetune_epochs=1,
            finetune_batch_size=8,
            extra={"preflight": True},
        ),
    )

    assert refused["status"] == "refused"
    assert runner._EXIT_CODES[refused["status"]] == 2
    reason = refused["reason"]
    assert "tail" in reason and "96px" in reason and "batch size 8" in reason
    assert "--finetune-batch-size" in reason
    # Nothing trained and nothing broke: no step, no backup, no error file.
    frames = _frames(out)
    assert [f["phase"] for f in frames if f.get("stage") == "preflight"] == ["main"]
    assert not any(frame.get("stage") == "training_step" for frame in frames)
    assert not list((out / "backups").glob("backup_*.pt"))
    assert not (out / "error.txt").exists()
    assert not (out / "result.json").exists()


def test_console_lines_show_the_preflight_skipped_steps_and_warnings():
    step = {
        "stage": "training_step", "epoch": 1, "epochs": 2, "batch": 50,
        "batches": 2935, "loss": 0.5, "images_per_second": 61.0,
        "skipped_steps": 16, "grad_scale": 1.0,
    }  # fmt: skip
    assert runner._line(step).endswith("61.0 img/s, 16 skipped, scale 1")
    # bf16, or a CPU: there is no scaler, so nothing is said about one.
    quiet = {**step, "skipped_steps": 0, "grad_scale": None}
    assert runner._line(quiet).endswith("61.0 img/s")
    epoch = {
        "stage": "training", "epoch": 1, "epochs": 2, "loss": 0.5,
        "validation_score": 0.7, "best_score": 0.7, "best_epoch": 1,
        "images_per_second": 61.0, "skipped_steps": 16, "grad_scale": 2.0,
    }  # fmt: skip
    assert runner._line(epoch).endswith("61.0 img/s, 16 skipped, scale 2")

    warning = {"stage": "warning", "status_text": "40 of the 50 steps were skipped"}
    assert runner._line(warning) == "WARNING: 40 of the 50 steps were skipped"
    fits = {"stage": "preflight", "status_text": "Preflight: the tail fits"}
    assert runner._line(fits) == "Preflight: the tail fits"
    resumed = {
        "type": "training_resumed", "epoch": 25, "global_step": 73000,
        "image_size": 384, "batch_size": 16,
    }  # fmt: skip
    assert runner._line(resumed) == "resumed at epoch 26, step 73000 (384px, batch 16)"
    sized = {"type": "autobatch", "phase": "tail", "image_size": 384, "batch_size": 21}
    assert runner._line(sized) == "automatic batch size 21 at 384px (tail)"


# -- the GRN layer -------------------------------------------------------------


def test_the_stock_grn_layer_is_an_option_not_the_default(tmp_path):
    pack = _make_pack(tmp_path)

    default, _ = runner.build_request(pack, tmp_path / "a", **_options())
    stock, _ = runner.build_request(pack, tmp_path / "b", **_options(stock_grn=True))

    # Unset means the trainer's own default, which is the lean layer.
    assert "lean_grn" not in default["training_config"]
    assert stock["training_config"]["lean_grn"] is False


def test_stock_grn_reaches_the_run_from_the_command_line(monkeypatch, tmp_path):
    seen: dict = {}

    def _fake(pack, out, **kw):
        seen.update(kw)
        return {"status": "finished"}

    monkeypatch.setattr(runner, "train_from_pack", _fake)
    base = ["--pack", str(tmp_path), "--out", str(tmp_path / "run")]

    assert runner.main(base) == 0
    assert seen["stock_grn"] is False
    assert runner.main([*base, "--stock-grn"]) == 0
    assert seen["stock_grn"] is True


# -- what a detached run leaves for whoever asks how it is doing ----------------


def test_a_run_keeps_a_status_file_and_stamps_its_frames(tmp_path):
    import bittrainer.tools.backbone_status as status

    pack = _make_pack(tmp_path)
    out = tmp_path / "run"
    seen: list[dict] = []

    def _watch(frame, control):
        if not seen:
            seen.append(json.loads((out / "status.json").read_text(encoding="utf-8")))

    summary = runner.train_from_pack(pack, out, on_frame=_watch, **_options())

    # While it ran: running, with the process, the machine and the plan.
    assert seen[0]["status"] == "running" and seen[0]["pid"] > 0 and seen[0]["host"]
    assert seen[0]["plan"]["epochs"] == 1 and seen[0]["plan"]["image_size"] == 64
    # After: how it ended, and every frame says when it was written.
    assert summary["status"] == "finished"
    ended = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert ended["status"] == "finished" and ended["ended"] >= seen[0]["started"]
    frames = _frames(out)
    assert frames and all(isinstance(frame.get("t"), float) for frame in frames)
    # And the status tool reads the folder.
    read_frames, read_status = status.read_run(out)
    report = status.summarise(read_frames, read_status, now=ended["ended"])
    assert report["state"] == "finished" and report["epoch"] == 1


def test_a_refused_run_says_so_in_its_status_file(tmp_path, monkeypatch):
    from bittrainer.generic.tasks.backbone_task import BackboneRunRefused

    async def _refuse(request, **_kwargs):
        raise BackboneRunRefused("the tail does not fit")

    import bittrainer.backbone_trainer as trainer

    monkeypatch.setattr(trainer, "run_backbone_training", _refuse)
    out = tmp_path / "run"

    summary = runner.train_from_pack(_make_pack(tmp_path), out, **_options())

    assert summary["status"] == "refused"
    ended = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert ended["status"] == "refused" and "does not fit" in ended["reason"]


# -- a finished run is not trained again ----------------------------------------


def _finished_folder(out: Path) -> None:
    out.mkdir(parents=True)
    result = {"validation_score": 0.91, "best_epoch": 5, "epochs_completed": 6}
    (out / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (out / "status.json").write_text(
        json.dumps({"status": "finished", "ended": 1.0}), encoding="utf-8"
    )
    (out / "backbone_candidate.safetensors").write_bytes(b"the finished model")


def test_a_finished_run_is_not_trained_again(tmp_path, monkeypatch):
    """The service that runs this command on the VM starts it at every boot, so
    an interrupted run resumes by itself. A run that had already finished
    started again from epoch 1 (seen on the T4, 2026-11-10) and would have
    written over its own finished model."""
    import bittrainer.backbone_trainer as trainer

    async def _never(request, **_kwargs):
        raise AssertionError("a finished run must not be trained again")

    monkeypatch.setattr(trainer, "run_backbone_training", _never)
    out = tmp_path / "run"
    _finished_folder(out)
    before = {path.name: path.read_bytes() for path in out.iterdir()}

    summary = runner.train_from_pack(_make_pack(tmp_path), out, **_options())

    assert summary["status"] == "finished" and summary["already_finished"] is True
    assert summary["validation_score"] == 0.91
    assert (summary["best_epoch"], summary["epochs_completed"]) == (5, 6)
    # Nothing in the folder was touched: no new status, progress or backups.
    assert {path.name: path.read_bytes() for path in out.iterdir()} == before


def test_fresh_trains_a_finished_run_again_and_forgets_its_old_result(
    tmp_path, monkeypatch
):
    import bittrainer.backbone_trainer as trainer

    out = tmp_path / "run"
    _finished_folder(out)
    seen: list[bool] = []

    async def _train(request, **_kwargs):
        seen.append((out / "result.json").exists())
        return {"paused": True, "backup_path": None}

    monkeypatch.setattr(trainer, "run_backbone_training", _train)

    summary = runner.train_from_pack(_make_pack(tmp_path), out, **_options(fresh=True))

    # The old result was gone before training began: cut short, the new run
    # must be resumed at the next start, not read as finished.
    assert seen == [False]
    assert summary["status"] == "paused"


def test_the_command_says_a_finished_run_was_left_alone(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        runner,
        "train_from_pack",
        lambda pack, out, **kw: {
            "status": "finished",
            "already_finished": True,
            "validation_score": 0.91,
            "best_epoch": 5,
            "epochs_completed": 6,
        },
    )

    code = runner.main(["--pack", str(tmp_path), "--out", str(tmp_path / "run")])

    assert code == 0  # a proper ending: the service does not start it again
    printed = capsys.readouterr().out
    assert "already finished" in printed and "--fresh" in printed
