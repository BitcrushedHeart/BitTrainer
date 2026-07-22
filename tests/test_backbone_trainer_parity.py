"""Tier-1 parity tests for the Backbone Builder trainer (Bitcrush ISSUE-0476).

CPU-only, tiny synthetic datasets, in-process. Covers the machinery ported from
the binary/group trainers: AMP, cosine LR, EMA, backup/pause/resume and early
stopping. Model creation uses ``backbone_init={"source": "random_init"}`` so no
timm weights are downloaded.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import torch
from PIL import Image

import bittrainer.backbone_trainer as bt
from bittrainer.backbone_trainer import run_backbone_training


# --------------------------------------------------------------------------- #
# Synthetic dataset helpers                                                    #
# --------------------------------------------------------------------------- #


def _make_image(path, color) -> None:
    Image.new("RGB", (32, 32), color).save(path)


def _make_request(tmp_path, *, config_extra=None, n_train=4, n_val=4):
    """Two-class group ("color") dataset with deterministic hash-based split.

    content_hash "ff..." lands in train, "00..." lands in val for any
    validation_split in (0, 1).
    """
    records = []
    img_dir = tmp_path / "imgs"
    img_dir.mkdir(exist_ok=True)

    # First 8 hex chars drive the deterministic split:
    #   bucket = int(hash[:8], 16) % 10000 / 10000
    # "00002328" -> 9000/10000 = 0.9 (train); "00000000" -> 0.0 (val).
    def add(idx, digest_prefix, class_name, color):
        p = img_dir / f"{digest_prefix}_{idx}.png"
        _make_image(p, color)
        records.append(
            {
                "content_hash": digest_prefix + f"{idx:056x}",
                "file_paths": [str(p)],
                "binary": {},
                "groups": {"color": class_name},
            }
        )

    for i in range(n_train):
        cls, col = ("a", (255, 0, 0)) if i % 2 == 0 else ("b", (0, 0, 255))
        add(i, "00002328", cls, col)
    for i in range(n_val):
        cls, col = ("a", (255, 0, 0)) if i % 2 == 0 else ("b", (0, 0, 255))
        add(i, "00000000", cls, col)

    candidate = tmp_path / "candidate.safetensors"
    config = {
        "image_size": 32,
        "batch_size": 2,
        "epochs": 2,
        "learning_rate": 1e-3,
        "validation_split": 0.4,
        "device": "cpu",
        "backbone_init": {"source": "random_init"},
    }
    if config_extra:
        config.update(config_extra)
    return {
        "run_id": "test-run",
        "convnextv2_size": "atto",
        "candidate_checkpoint_path": str(candidate),
        "records": records,
        "backbone_init": {"source": "random_init"},
        "training_config": config,
        "heads": {},
    }


def _run(request, **kwargs):
    return asyncio.run(run_backbone_training(request, **kwargs))


# --------------------------------------------------------------------------- #
# Pure helpers                                                                 #
# --------------------------------------------------------------------------- #


def test_amp_settings_default_on_bfloat16():
    enabled, dtype = bt._amp_settings({})
    assert enabled is True
    assert dtype is torch.bfloat16


def test_amp_settings_disabled():
    enabled, dtype = bt._amp_settings({"use_amp": False})
    assert enabled is False


def test_amp_settings_float16():
    _enabled, dtype = bt._amp_settings({"amp_dtype": "float16"})
    assert dtype is torch.float16


def _task_optimizer(tmp_path, **config_extra):
    """Build a BackboneTask and run its create_optimizer hook.

    Scheduler/EMA construction moved off `backbone_trainer` into
    `BackboneTask.create_optimizer` in the GenericTrainer refactor (ISSUE-0542);
    these tests follow it there. `ctx` is only touched on resume, so None is safe
    with `resume_state=None`.
    """
    from bittrainer.generic.tasks.backbone_task import BackboneTask

    task = BackboneTask(_make_request(tmp_path, config_extra=config_extra))
    model = torch.nn.Linear(2, 2)
    optimizer, scheduler, _t_max = task.create_optimizer(None, model, 2, None)
    return task, optimizer, scheduler


def test_scheduler_is_cosine_by_default(tmp_path):
    _task, _opt, sched = _task_optimizer(tmp_path, epochs=5)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_scheduler_off_is_a_constant_multiplier(tmp_path):
    """use_cosine=False no longer returns None: the core steps and serialises the
    scheduler every epoch, so the task hands back a constant-LR LambdaLR."""
    _task, opt, sched = _task_optimizer(tmp_path, epochs=5, use_cosine=False)
    assert isinstance(sched, torch.optim.lr_scheduler.LambdaLR)
    lr0 = opt.param_groups[0]["lr"]
    # Scheduler only — the task builds a Prodigy_adv optimizer, whose .step()
    # needs a real backward pass first. The LR schedule is what is under test.
    sched.step()
    sched.step()
    assert opt.param_groups[0]["lr"] == lr0


def test_cosine_lr_decays_across_steps(tmp_path):
    _task, opt, sched = _task_optimizer(tmp_path, epochs=4)
    lr0 = opt.param_groups[0]["lr"]
    # See test_scheduler_off_is_a_constant_multiplier — scheduler steps only.
    sched.step()
    sched.step()
    assert opt.param_groups[0]["lr"] < lr0


# --------------------------------------------------------------------------- #
# End-to-end behaviour                                                         #
# --------------------------------------------------------------------------- #


def test_training_completes_and_writes_checkpoint(tmp_path):
    result = _run(_make_request(tmp_path))
    assert result["candidate_checkpoint_path"]
    assert (tmp_path / "candidate.safetensors").is_file()
    assert "epochs_completed" in result


def test_ema_updates_are_applied(tmp_path):
    """EMA now lives on the task (ISSUE-0542), so spy on the task's own symbol."""
    import bittrainer.generic.tasks.backbone_task as bbtask

    calls = {"n": 0}
    orig = bbtask.ModelEMA

    class _SpyEMA(orig):  # type: ignore[misc, valid-type]
        def update(self, model):
            calls["n"] += 1
            super().update(model)

    bbtask.ModelEMA = _SpyEMA
    try:
        _run(_make_request(tmp_path, config_extra={"use_ema": True, "max_steps": 3}))
    finally:
        bbtask.ModelEMA = orig
    assert calls["n"] >= 1


def test_ema_is_off_when_disabled(tmp_path):
    task, _opt, _sched = _task_optimizer(tmp_path, use_ema=False)
    assert task.ema is None


def test_backup_complete_emitted(tmp_path):
    # A successful run deletes its (now obsolete) backups on completion, so we
    # assert the periodic backup fired via its progress message rather than by
    # inspecting the pruned directory.
    messages = []
    backup_dir = tmp_path / "backups"
    _run(
        _make_request(
            tmp_path,
            config_extra={
                "backup_dir": str(backup_dir),
                "backup_every_steps": 1,
                "epochs": 1,
                "max_steps": 2,
            },
        ),
        progress_callback=messages.append,
    )
    assert any(m.get("type") == "backup_complete" for m in messages)


def test_resume_emits_training_resumed(tmp_path):
    backup_dir = tmp_path / "backups"
    # First run: produce a backup by pausing MID-RUN. A pause that is already set
    # when run() starts is answered before any training with backup_path=None
    # ("nothing trained, nothing to snapshot"), so it would leave backup_dir empty
    # and this test would never reach the resume it exists to cover.
    _run(
        _make_request(
            tmp_path, config_extra={"backup_dir": str(backup_dir), "backup_every_steps": 1}
        ),
        pause_event=_DelayedStubEvent(after=4),
    )
    assert list(backup_dir.glob("backup_*.pt"))

    messages = []
    _run(
        _make_request(
            tmp_path,
            config_extra={
                "backup_dir": str(backup_dir),
                "resume_from": str(backup_dir),
                "epochs": 2,
            },
        ),
        progress_callback=messages.append,
    )
    assert any(m.get("type") == "training_resumed" for m in messages)


def test_pause_returns_paused(tmp_path):
    pause = _StubEvent()
    pause.set()
    result = _run(
        _make_request(tmp_path, config_extra={"backup_dir": str(tmp_path / "bk")}),
        pause_event=pause,
    )
    assert result.get("paused") is True


def test_pause_before_training_starts_reports_no_backup(tmp_path):
    """A pause during data prep is answered before anything is trained, so there
    is no state to snapshot — `backup_path` is None BY DESIGN. Pinned because
    the Engine relies on it (`job.backup_path = msg[...] or job.backup_path`
    keeps the previous backup rather than clobbering it with None)."""
    backup_dir = tmp_path / "bk_early"
    pause = _StubEvent()
    pause.set()
    result = _run(
        _make_request(tmp_path, config_extra={"backup_dir": str(backup_dir)}),
        pause_event=pause,
    )
    assert result.get("paused") is True
    assert result.get("backup_path") is None
    assert not list(backup_dir.glob("backup_*.pt"))


def test_pause_mid_run_writes_a_resumable_backup(tmp_path):
    """The pause that matters: once the epoch loop is running, the boundary hook
    snapshots state and the returned backup_path points at a real file."""
    backup_dir = tmp_path / "bk_mid"
    result = _run(
        _make_request(
            tmp_path,
            config_extra={"backup_dir": str(backup_dir), "backup_every_steps": 1, "epochs": 3},
        ),
        pause_event=_DelayedStubEvent(after=4),
    )
    assert result.get("paused") is True
    assert result["backup_path"] is not None
    assert Path(result["backup_path"]).is_file()
    assert list(backup_dir.glob("backup_*.pt"))


def test_early_stopping_on_non_improving_validation(tmp_path, monkeypatch):
    # Force validation to never improve -> patience should trip early.
    monkeypatch.setattr(
        bt, "_evaluate", lambda *a, **k: {"group/color": 0.5}
    )
    result = _run(
        _make_request(
            tmp_path,
            config_extra={"epochs": 6, "patience": 1, "use_ema": False},
        )
    )
    assert result["epochs_completed"] < 6


class _StubEvent:
    """Minimal threading.Event-like stub for pause tests."""

    def __init__(self) -> None:
        self._set = False

    def set(self) -> None:
        self._set = True

    def is_set(self) -> bool:
        return self._set


class _DelayedStubEvent(_StubEvent):
    """Fires only after ``after`` polls, so the pause lands INSIDE the epoch loop
    (where there is state worth backing up) rather than during data prep."""

    def __init__(self, *, after: int) -> None:
        super().__init__()
        self._after = int(after)
        self._polls = 0

    def is_set(self) -> bool:
        if self._set:
            return True
        self._polls += 1
        return self._polls > self._after


# --------------------------------------------------------------------------- #
# Graceful stop / stop-now (Bitcrush ISSUE-0554)                               #
# --------------------------------------------------------------------------- #


def test_graceful_stop_finishes_early_and_still_exports(tmp_path):
    """`stop_event` breaks at the next epoch boundary but STILL finalises, so a
    finish-early keeps the best candidate (unlike a pause, which ships nothing)."""
    stop = _StubEvent()
    stop.set()
    messages = []
    result = _run(
        _make_request(tmp_path, config_extra={"epochs": 4}),
        progress_callback=messages.append,
        stop_event=stop,
    )
    assert result.get("paused") is not True
    assert result["candidate_checkpoint_path"]
    assert result["epochs_completed"] < 4
    assert any(m.get("type") == "graceful_stop" for m in messages)


def test_stop_now_breaks_immediately_and_still_exports(tmp_path):
    stop_now = _StubEvent()
    stop_now.set()
    messages = []
    result = _run(
        _make_request(tmp_path, config_extra={"epochs": 4}),
        progress_callback=messages.append,
        stop_now_event=stop_now,
    )
    assert result["candidate_checkpoint_path"]
    assert any(m.get("type") == "stop_now" for m in messages)


def test_max_steps_still_stops_without_an_external_stop_event(tmp_path):
    """The task's own max_steps stop must survive the external-event wiring."""
    result = _run(_make_request(tmp_path, config_extra={"epochs": 4, "max_steps": 2}))
    assert result["epochs_completed"] < 4


def test_head_training_accepts_the_stop_events(tmp_path):
    """Head-only shares the entry-point contract even though it has no loop to
    interrupt — the kwargs must not raise."""
    import inspect

    from bittrainer.backbone_trainer import run_backbone_head_training

    params = inspect.signature(run_backbone_head_training).parameters
    assert "stop_event" in params
    assert "stop_now_event" in params
