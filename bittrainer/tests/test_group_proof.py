"""Proof run for the group trainers on generated images (Bitcrush ISSUE-1690).

``bittrainer.tools.group_proof`` draws its own group, trains it head-only, ships
the group's cached tensors through the scoped export, deletes the images and
updates the head-only model from the export alone, the way a cloud run does.
It reports speed, memory, the precision used, the GradScaler, the GRN layer and
the skipped steps. The same command runs at home and on the rented T4.

CPU-only, on the micro stand-in trunk (``group_stub``), a few dozen 64px images.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import bittrainer.tools.group_proof as proof
from bittrainer.generic.tasks.group_task import GroupTask
from bittrainer.tests.group_stub import use_micro_trunk

_OOM = "CUDA out of memory. Tried to allocate 2.00 GiB."


def _run(tmp_path, monkeypatch, **kw):
    use_micro_trunk(monkeypatch)
    options = dict(
        size="atto",
        bucket_size=64,
        images_per_class=6,
        epochs=1,
        batch_size=4,
        device="cpu",
        amp="off",
        compile=False,
        workers=0,
    )
    options.update(kw)
    return proof.run_proof(tmp_path, **options)


def test_the_group_is_drawn_the_same_every_time(tmp_path):
    a = proof.make_group_folder(tmp_path / "a", per_class=3, val_per_class=2, image_size=48)
    b = proof.make_group_folder(tmp_path / "b", per_class=3, val_per_class=2, image_size=48)

    assert a == b == {"train": 12, "val": 8}
    for name in proof.CLASSES:
        for split, count in (("train", 3), ("val", 2)):
            files = sorted((tmp_path / "a" / name / split).iterdir())
            assert len(files) == count
            twin = tmp_path / "b" / name / split / files[0].name
            assert files[0].read_bytes() == twin.read_bytes()
    # Train and validation images differ, and so do the classes.
    first = lambda split, name: (tmp_path / "a" / name / split / f"{name}_00000.jpg")  # noqa: E731
    assert first("train", "stacked").read_bytes() != first("val", "stacked").read_bytes()
    assert first("train", "stacked").read_bytes() != first("train", "centre_blob").read_bytes()


def test_a_proof_run_trains_head_only_exports_and_updates_without_the_images(
    tmp_path, monkeypatch
):
    report = _run(tmp_path, monkeypatch)

    assert report["error"] is None, report["error"]
    checks = report["checks"]
    for name in (
        "head_only_completed",
        "export_verified",
        "sources_removed",
        "update_completed",
        "loss_finite",
        "checkpoint_finite",
    ):
        assert checks[name] is True, name
    assert checks["score_reached"] is None  # no floor was asked for
    assert report["passed"] is True
    assert not (tmp_path / "home" / proof.GROUP).exists()  # the pod never had them
    assert (report["train_images"], report["val_images"]) == (24, 8)

    # What it says about the run: a CPU, so no scaler and timm's GRN.
    assert report["device"] == "cpu" and report["amp_dtype"] == "off"
    assert report["grad_scaler"] is False and report["lean_grn"] is False
    assert report["skipped_steps"] is None
    assert report["fused_back_pass"].startswith("Fused back pass ON")
    assert report["compile_requested"] is False and report["compiled"] is False
    assert len(report["epochs"]) == 1
    assert report["epochs"][0]["images_per_second"] > 0
    assert report["best_images_per_second"] > 0
    assert report["head_only"]["seconds"] >= 0
    # Numbers and settings only: no file is named.
    assert str(tmp_path) not in str(report)


def test_a_paused_update_resumes_from_its_backup_and_finishes(tmp_path, monkeypatch):
    report = _run(tmp_path, monkeypatch, epochs=2, pause_after_steps=3)

    assert report["error"] is None, report["error"]
    assert report["checks"]["paused_with_a_backup"] is True
    assert report["checks"]["resumed"] is True
    assert report["checks"]["update_completed"] is True
    assert report["paused"] is True and report["resumed"] is True
    assert report["passed"] is True
    assert [row["epoch"] for row in report["epochs"]] == [1, 2]


def test_pausing_needs_an_epoch_to_resume_into(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="two epochs"):
        _run(tmp_path, monkeypatch, epochs=1, pause_after_steps=3)


def test_an_oversize_batch_is_refused_before_it_trains_and_leaves_no_backup(
    tmp_path, monkeypatch
):
    def _dry_step(self, ctx, model, batch_size):
        raise RuntimeError(_OOM)

    monkeypatch.setattr(GroupTask, "_dry_step", _dry_step)

    report = _run(tmp_path, monkeypatch, batch_size=64, preflight=True, expect_refusal=True)

    assert "batch size 64" in report["refused"]
    assert report["checks"] == {
        "head_only_completed": True,
        "export_verified": True,
        "sources_removed": True,
        "refused": True,
        "nothing_trained": True,
        "no_backup_left": True,
    }
    assert report["passed"] is True and report["error"] is None
    assert report["epochs"] == []


def test_a_refusal_that_was_not_expected_fails_the_run(tmp_path, monkeypatch):
    def _dry_step(self, ctx, model, batch_size):
        raise RuntimeError(_OOM)

    monkeypatch.setattr(GroupTask, "_dry_step", _dry_step)

    report = _run(tmp_path, monkeypatch, preflight=True)

    assert report["passed"] is False
    assert report["error"].startswith("RunRefused")


def test_an_expected_refusal_that_never_came_fails_the_run(tmp_path, monkeypatch):
    report = _run(tmp_path, monkeypatch, expect_refusal=True)

    assert report["checks"]["refused"] is False
    assert report["checks"]["nothing_trained"] is False
    assert report["passed"] is False


def test_the_settings_reach_the_trainer(tmp_path):
    config = proof.build_config(
        tmp_path / "g", tmp_path / "ck", cache_dir=tmp_path / "c",
        embedding_dir=tmp_path / "e", size="base", bucket_size=384, epochs=3,
        batch_size=0, amp="auto", device="cuda", compile=False, stock_grn=True,
        init="random", workers=4, sourceless=True, preflight=True,
        backup_dir=tmp_path / "b",
    )  # fmt: skip

    assert config.backbone_variant == "base" and config.train_resolution == 384
    assert config.max_epochs == 3 and config.batch_size is None  # automatic
    assert config.dtype == "bfloat16"  # asked for; a T4 turns it into fp16
    assert config.use_compile is False and config.lean_grn is False
    assert config.preflight is True and config.sourceless is True
    assert config.backbone_init == {"source": "random_init", "checkpoint_path": None}
    assert proof.build_config(
        tmp_path / "g", tmp_path / "ck", cache_dir=tmp_path / "c",
        embedding_dir=tmp_path / "e", amp="fp16", trunk="trunks/base.safetensors",
    ).backbone_init == {"source": "local_candidate", "checkpoint_path": "trunks/base.safetensors"}  # fmt: skip
    assert proof.build_config(
        tmp_path / "g", tmp_path / "ck", cache_dir=tmp_path / "c",
        embedding_dir=tmp_path / "e", amp="off", init="timm",
    ).backbone_init is None  # fmt: skip


def test_command_line_options_reach_the_run(monkeypatch, tmp_path):
    seen: dict = {}

    def _fake(root, **kw):
        seen.update(kw, root=Path(root))
        return {"passed": True, "checks": {}}

    monkeypatch.setattr(proof, "run_proof", _fake)
    code = proof.main(
        [
            "--workdir", str(tmp_path),
            "--size", "base",
            "--bucket-size", "512",
            "--images-per-class", "40",
            "--epochs", "3",
            "--batch-size", "8",
            "--device", "cuda",
            "--amp", "fp16",
            "--no-compile",
            "--stock-grn",
            "--init", "timm",
            "--workers", "2",
            "--pause-after-steps", "5",
            "--preflight",
            "--expect-refusal",
            "--min-score", "0.5",
            "--seed", "4",
        ]
    )  # fmt: skip

    assert code == 0
    assert seen == {
        "root": tmp_path,
        "size": "base",
        "bucket_size": 512,
        "images_per_class": 40,
        "epochs": 3,
        "batch_size": 8,
        "device": "cuda",
        "amp": "fp16",
        "compile": False,
        "stock_grn": True,
        "init": "timm",
        "trunk": None,
        "workers": 2,
        "pause_after_steps": 5,
        "preflight": True,
        "expect_refusal": True,
        "min_score": 0.5,
        "seed": 4,
    }

    seen.clear()
    proof.main(["--workdir", str(tmp_path)])
    # Unasked: compiled as Engine runs it, the trainer's own preflight rule.
    assert seen["compile"] is True and seen["preflight"] is None
    assert seen["batch_size"] == 0 and seen["min_score"] is None


def test_a_failed_check_fails_the_command(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        proof,
        "run_proof",
        lambda root, **kw: {"passed": False, "checks": {"update_completed": False}},
    )
    assert proof.main(["--workdir", str(tmp_path)]) == 1
    assert "FAIL" in capsys.readouterr().out

    monkeypatch.setattr(
        proof, "run_proof", lambda root, **kw: {"passed": True, "checks": {}}
    )
    assert proof.main(["--workdir", str(tmp_path)]) == 0
    assert "PASS" in capsys.readouterr().out
