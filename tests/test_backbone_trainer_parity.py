"""Tier-1 parity tests for the Backbone Builder trainer (Bitcrush ISSUE-0476).

CPU-only, tiny synthetic datasets, in-process. Covers the machinery ported from
the binary/group trainers: AMP, cosine LR, EMA, backup/pause/resume and early
stopping. Model creation uses ``backbone_init={"source": "random_init"}`` so no
timm weights are downloaded.
"""

from __future__ import annotations

import asyncio

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


def test_build_scheduler_cosine_default_on():
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = bt._build_scheduler(opt, {}, epochs=5)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_build_scheduler_off():
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert bt._build_scheduler(opt, {"use_cosine": False}, epochs=5) is None


def test_cosine_lr_decays_across_steps():
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = bt._build_scheduler(opt, {}, epochs=4)
    lr0 = opt.param_groups[0]["lr"]
    opt.step()
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
    calls = {"n": 0}
    orig = bt.ModelEMA

    class _SpyEMA(orig):  # type: ignore[misc, valid-type]
        def update(self, model):
            calls["n"] += 1
            super().update(model)

    bt.ModelEMA = _SpyEMA
    try:
        _run(_make_request(tmp_path, config_extra={"use_ema": True, "max_steps": 3}))
    finally:
        bt.ModelEMA = orig
    assert calls["n"] >= 1


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
    # First run: produce a backup (do not delete on completion by pausing).
    pause = _StubEvent()
    pause.set()
    _run(
        _make_request(
            tmp_path, config_extra={"backup_dir": str(backup_dir), "backup_every_steps": 1}
        ),
        pause_event=pause,
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
