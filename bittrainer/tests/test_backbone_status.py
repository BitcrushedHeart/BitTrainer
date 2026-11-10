"""Where a pack run stands, read from its folder (Bitcrush ISSUE-1655).

The real run takes days on a rented card. The runner now stamps every progress
frame with the time and keeps a ``status.json`` (running, paused, finished,
failed, refused, with the run's plan), and ``backbone_status`` turns the two into
a few lines: which epoch and batch, how fast, the last validation score, skipped
steps, when it last made progress and roughly how long is left. A run whose
process is gone, or that has been silent too long, must not read as "running".

Pure functions over frames: nothing here trains.
"""

from __future__ import annotations

import json

import bittrainer.tools.backbone_status as status

PLAN = {
    "epochs": 10,
    "image_size": 224,
    "batch_size": 64,
    "finetune_image_size": 384,
    "finetune_epochs": 2,
    "finetune_batch_size": 16,
}


def _step(epoch, batch, batches=100, *, t, size=224, bs=64, speed=64.0, skipped=0):
    return {
        "stage": "training_step",
        "epoch": epoch,
        "epochs": PLAN["epochs"],
        "batch": batch,
        "batches": batches,
        "loss": 0.25,
        "images_per_second": speed,
        "image_size": size,
        "batch_size": bs,
        "skipped_steps": skipped,
        "t": t,
    }


def _epoch(epoch, *, t, seconds=100.0, size=224, bs=64, score=0.5, best=0.5, at=1):
    return {
        "stage": "training",
        "epoch": epoch,
        "epochs": PLAN["epochs"],
        "loss": 0.2,
        "validation_score": score,
        "best_score": best,
        "best_epoch": at,
        "train_seconds": seconds,
        "train_images": 6400,
        "images_per_second": 64.0,
        "image_size": size,
        "batch_size": bs,
        "skipped_steps_total": 16,
        "t": t,
    }


def _running(**extra):
    return {"status": "running", "pid": 4242, "started": 0.0, "plan": PLAN, **extra}


def test_a_run_in_the_middle_of_an_epoch_says_where_it_is_and_how_long_is_left():
    frames = [
        _epoch(1, t=100.0),
        _epoch(2, t=200.0),
        {"type": "backup_complete", "global_step": 200, "epoch": 2, "t": 201.0},
        _step(3, 50, t=250.0),
    ]

    summary = status.summarise(frames, _running(), now=260.0, alive=True)

    assert summary["state"] == "running"
    assert (summary["epoch"], summary["epochs"]) == (3, 10)
    assert (summary["batch"], summary["batches"]) == (50, 100)
    assert (summary["image_size"], summary["batch_size"]) == (224, 64)
    assert summary["validation_score"] == 0.5 and summary["best_epoch"] == 1
    assert summary["skipped_steps"] == 16
    assert summary["last_backup_step"] == 200
    assert summary["seconds_since_progress"] == 10.0
    # Half of epoch 3 (50 s), five more 224px epochs (500 s), and two 384px epochs
    # that have not run yet: estimated from the 224px ones by pixel count.
    tail = 100.0 * (384 / 224) ** 2
    assert abs(summary["seconds_left"] - (50 + 500 + 2 * tail)) < 1.0
    assert summary["tail_is_estimated"] is True

    text = "\n".join(status.report_lines(summary))
    assert "running" in text and "epoch 3 of 10" in text and "batch 50 of 100" in text
    assert "estimate" in text


def test_once_a_tail_epoch_has_run_its_own_time_is_used():
    frames = [_epoch(n, t=100.0 * n) for n in range(1, 9)]
    frames.append(_epoch(9, t=1300.0, seconds=400.0, size=384, bs=16))
    frames.append(_step(10, 200, batches=400, t=1500.0, size=384, bs=16, speed=16.0))

    summary = status.summarise(frames, _running(), now=1500.0, alive=True)

    assert (summary["image_size"], summary["batch_size"]) == (384, 16)
    assert abs(summary["seconds_left"] - 200.0) < 1.0  # half of one measured tail epoch
    assert summary["tail_is_estimated"] is False


def test_a_first_epoch_is_timed_from_its_own_speed():
    frames = [_step(1, 25, t=50.0, speed=32.0)]

    summary = status.summarise(frames, _running(), now=50.0, alive=True)

    # 100 batches of 64 at 32 images/s is 200 s an epoch; three quarters of the
    # first is left, then seven more, then the two estimated tail epochs.
    epoch = 100 * 64 / 32.0
    tail = epoch * (384 / 224) ** 2
    assert abs(summary["seconds_left"] - (0.75 * epoch + 7 * epoch + 2 * tail)) < 1.0


def test_a_run_whose_process_is_gone_is_not_called_running():
    summary = status.summarise(
        [_step(3, 50, t=250.0)], _running(), now=300.0, alive=False
    )

    assert summary["state"] == "stopped"
    text = "\n".join(status.report_lines(summary))
    assert "no longer running" in text and "same command" in text


def test_a_run_that_has_been_silent_too_long_says_so():
    summary = status.summarise(
        [_step(3, 50, t=250.0)], _running(), now=250.0 + 45 * 60, alive=True,
        stale_minutes=20,
    )  # fmt: skip

    assert summary["state"] == "silent"
    assert "45 min" in "\n".join(status.report_lines(summary))


def test_endings_are_read_from_the_status_file():
    frames = [_epoch(1, t=100.0)]
    paused = status.summarise(frames, {"status": "paused", "plan": PLAN}, now=500.0)
    failed = status.summarise(
        frames, {"status": "failed", "reason": "OutOfMemoryError: no room", "plan": PLAN},
        now=500.0,
    )  # fmt: skip
    finished = status.summarise(
        [_epoch(n, t=100.0 * n) for n in range(1, 11)],
        {"status": "finished", "plan": PLAN, "validation_score": 0.71, "best_epoch": 9},
        now=2000.0,
    )

    assert paused["state"] == "paused" and paused["seconds_left"] is not None
    assert failed["state"] == "failed"
    assert "OutOfMemoryError" in "\n".join(status.report_lines(failed))
    assert finished["state"] == "finished" and finished["seconds_left"] == 0
    assert "finished" in "\n".join(status.report_lines(finished))


def test_a_run_that_has_not_reported_yet(tmp_path):
    summary = status.summarise([], _running(), now=10.0, alive=True)
    assert summary["state"] == "running" and summary["epoch"] is None
    assert status.report_lines(summary)

    # No status file at all: the folder is not a run (or an older runner wrote it).
    nothing = status.summarise([], {}, now=10.0)
    assert nothing["state"] == "unknown"


def test_skipped_steps_and_warnings_are_shown():
    frames = [
        _epoch(1, t=100.0),
        {"stage": "warning", "status_text": "steps are being skipped", "t": 150.0},
        _step(2, 60, t=160.0, skipped=7),
    ]

    summary = status.summarise(frames, _running(), now=160.0, alive=True)

    assert summary["warnings"] == 1
    assert summary["skipped_steps"] == 16 and summary["skipped_this_epoch"] == 7
    assert "warning" in "\n".join(status.report_lines(summary)).lower()


def test_the_folder_is_read_and_a_cut_off_last_line_is_ignored(tmp_path):
    lines = [json.dumps(_epoch(1, t=100.0)), json.dumps(_step(2, 10, t=120.0))]
    (tmp_path / "progress.jsonl").write_text(
        "\n".join(lines) + '\n{"stage": "training_st', encoding="utf-8"
    )
    (tmp_path / "status.json").write_text(json.dumps(_running()), encoding="utf-8")

    frames, state = status.read_run(tmp_path)

    assert len(frames) == 2 and state["status"] == "running"
    assert status.read_run(tmp_path / "missing") == ([], {})


def test_the_command_prints_the_lines_and_says_how_it_stands(tmp_path, capsys):
    (tmp_path / "progress.jsonl").write_text(
        json.dumps(_epoch(1, t=100.0)) + "\n", encoding="utf-8"
    )
    (tmp_path / "status.json").write_text(
        json.dumps({"status": "paused", "plan": PLAN}), encoding="utf-8"
    )

    assert status.main(["--out", str(tmp_path)]) == 0
    assert "paused" in capsys.readouterr().out

    (tmp_path / "status.json").write_text(
        json.dumps({"status": "failed", "reason": "boom", "plan": PLAN}),
        encoding="utf-8",
    )
    assert status.main(["--out", str(tmp_path)]) == 1
