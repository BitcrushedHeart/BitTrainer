"""fp16 with a GradScaler for the group trainers (Bitcrush ISSUE-1690).

The group full fine-tune autocast in bfloat16 with no GradScaler. A Turing card
(the Azure T4) only emulates bf16, so the precision is now fitted to the card:
bf16 where it is native, fp16 through a GradScaler where it is not. The fused
back pass (the optimizer stepping from hooks inside ``backward()``) cannot work
under a scaler and is turned off, with its reason. The scaler's state and the
steps it skipped ride the backups.

A bf16 card and a CPU must train exactly as before: the same dtype, the same
calls in the same order, the same loop arguments, the same backup.

CPU-only, on a two-layer model or the micro stand-in trunk (``group_stub``).
Tests named ``..._real_gpu`` / ``..._matches_the_card`` skip without CUDA and
are the ones to run on the T4.
"""

from __future__ import annotations

import math
import threading
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import bittrainer.amp as amp
import bittrainer.generic.tasks.group_task as group_task
import bittrainer.group_trainer as gt
from bittrainer.generic.generic_trainer import GenericTrainer
from bittrainer.generic.optimizer import make_optimizer
from bittrainer.generic.tasks.group_task import GroupTask
from bittrainer.tests.group_stub import group_config, make_group, use_micro_trunk

CPU = torch.device("cpu")

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (run on the T4)"
)


# --------------------------------------------------------------------------- #
# Stand-ins                                                                    #
# --------------------------------------------------------------------------- #


class _TinyConv(nn.Module):
    """2-D and 1-D parameters and a real conv backward, nothing more."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        torch.manual_seed(1)
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.head = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x).mean(dim=(2, 3)))


def _cfg(**kw) -> gt.GroupTrainConfig:
    base = dict(
        group_folder=".",
        num_classes=3,
        class_names=["a", "b", "c"],
        device="cpu",
        dtype="float32",
        channels_last=False,
        use_compile=False,
        randaugment_n=0,
        randaugment_m=0,
        random_erasing_p=0.0,
        use_mixup=False,
    )
    base.update(kw)
    return gt.GroupTrainConfig(**base)


def _batches(n: int = 3, batch: int = 2, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randint(0, 255, (batch, 3, 16, 16), dtype=torch.uint8, generator=g),
            torch.randint(0, 3, (batch,), generator=g),
        )
        for _ in range(n)
    ]


def _run_epoch(model, optimizer, config, batches, **kw):
    torch.manual_seed(7)
    return gt._train_one_epoch(model, batches, optimizer, config, CPU, torch.float32, **kw)


def _optimizer(model: nn.Module):
    return make_optimizer(model, llrd=False, wd_exclusions=True)


class _Scaled:
    """What ``scaler.scale(loss)`` hands back: its ``backward`` is recorded."""

    def __init__(self, loss, events):
        self.loss, self.events = loss, events

    def backward(self):
        self.events.append("backward")
        self.loss.backward()


class _RecordingScaler:
    """Plays torch's GradScaler: records how it is driven, and skips the steps
    it is told to the way torch's does when a gradient is not finite (the
    optimizer is left alone and the scale halves)."""

    def __init__(self, events=None, *, skips=(), pause_after=None, pause_event=None):
        self.events = [] if events is None else events
        self.skips = set(skips)
        self.value = 65536.0
        self.steps = 0
        self.applied = 0
        self.loaded = None
        self._skipping = False
        self.pause_after = pause_after
        self.pause_event = pause_event

    def scale(self, loss):
        self.events.append("scale")
        return _Scaled(loss, self.events)

    def unscale_(self, optimizer):
        self.events.append("unscale")

    def step(self, optimizer):
        self.events.append("step")
        self.steps += 1
        self._skipping = self.steps in self.skips
        if not self._skipping:
            optimizer.step()
            self.applied += 1

    def update(self):
        self.events.append("update")
        if self._skipping:
            self.value *= 0.5
        if self.pause_after is not None and self.steps >= self.pause_after:
            self.pause_event.set()

    def get_scale(self):
        return self.value

    def state_dict(self):
        return {"scale": self.value}

    def load_state_dict(self, state):
        self.loaded = dict(state)
        self.value = state["scale"]


def _context(tmp_path, monkeypatch, *, device="cpu", dtype="bfloat16", native=None, **kw):
    """A GroupTask's context on a card that does (``native=True``) or does not
    have bf16. A CPU is left alone: it counts as native without being told."""
    if native is not None:
        monkeypatch.setattr(amp, "native_bf16", lambda _device: native)

    def _make(device, dtype, enabled):
        return _RecordingScaler() if amp.wants_grad_scaler(device, dtype, enabled) else None

    monkeypatch.setattr(group_task, "make_grad_scaler", _make)
    frames: list[dict] = []
    task = GroupTask(group_config(tmp_path, tmp_path / "ck", device=device, dtype=dtype, **kw))
    ctx = task.make_context(frames.append, None, None, None)
    return task, ctx, frames


# --------------------------------------------------------------------------- #
# Choosing the precision                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("device", "asked", "native", "trained", "scaled"),
    [
        ("cpu", "bfloat16", None, torch.bfloat16, False),
        ("cpu", "float32", None, torch.float32, False),
        ("cpu", "float16", None, torch.float16, False),  # no scaler off a GPU
        ("cuda", "bfloat16", True, torch.bfloat16, False),  # an Ampere / Ada card
        ("cuda", "float32", True, torch.float32, False),
        ("cuda", "bfloat16", False, torch.float16, True),  # the T4
        ("cuda", "float32", False, torch.float32, False),
    ],
)
def test_the_precision_follows_the_card(
    tmp_path, monkeypatch, device, asked, native, trained, scaled
):
    task, ctx, frames = _context(
        tmp_path, monkeypatch, device=device, dtype=asked, native=native
    )

    assert ctx.dtype == trained
    assert (task.scaler is not None) is scaled
    precision = [f for f in frames if "grad_scaler" in f]
    if scaled:
        (frame,) = precision
        assert frame["amp_dtype"] == "float16" and frame["grad_scaler"] is True
        assert "GradScaler" in frame["status_text"]
    else:
        # A bf16 card and a CPU say nothing new.
        assert precision == []


def test_on_a_bf16_card_or_a_cpu_the_dtype_is_the_one_it_always_was(
    tmp_path, monkeypatch
):
    for name in ("bfloat16", "float16", "float32"):
        _task, ctx, _frames = _context(tmp_path, monkeypatch, dtype=name)
        assert ctx.dtype == gt._get_dtype(name)
        _task, ctx, _frames = _context(
            tmp_path, monkeypatch, device="cuda", dtype=name, native=True
        )
        assert ctx.dtype == gt._get_dtype(name)


# --------------------------------------------------------------------------- #
# Stepping through the scaler                                                  #
# --------------------------------------------------------------------------- #


def _spy_clip(monkeypatch, events):
    real = gt.clip_gradients

    def _clip(model, max_norm):
        events.append("clip")
        return real(model, max_norm)

    monkeypatch.setattr(gt, "clip_gradients", _clip)


def test_every_step_goes_through_the_scaler_in_order(monkeypatch):
    """scale -> backward, unscale BEFORE the clipper, then step and update:
    the backbone trainer's sequence."""
    events: list[str] = []
    _spy_clip(monkeypatch, events)
    model = _TinyConv()
    optimizer = _optimizer(model)

    _run_epoch(model, optimizer, _cfg(), _batches(3), scaler=_RecordingScaler(events))

    assert events == ["scale", "backward", "unscale", "clip", "step", "update"] * 3
    assert optimizer.param_groups[0]["k"] == 3  # the optimizer really stepped


def test_under_accumulation_the_scaler_steps_once_per_window(monkeypatch):
    events: list[str] = []
    _spy_clip(monkeypatch, events)
    model = _TinyConv()

    _run_epoch(
        model,
        _optimizer(model),
        _cfg(grad_accum_steps=2),
        _batches(4),
        scaler=_RecordingScaler(events),
    )

    window = ["scale", "backward", "scale", "backward", "unscale", "clip", "step", "update"]
    assert events == window * 2


def test_the_fused_back_pass_is_refused_under_a_scaler_with_its_reason(monkeypatch):
    model = _TinyConv()
    optimizer = _optimizer(model)

    fused, reason = gt._fused_back_pass_plan(optimizer, _cfg(), _RecordingScaler())
    assert fused is False and "GradScaler" in reason
    # Without one it is what it always was.
    assert gt._fused_back_pass_plan(optimizer, _cfg()) == (True, "fused back pass")

    registered: list[int] = []
    real = gt._register_fused_hooks
    monkeypatch.setattr(
        gt,
        "_register_fused_hooks",
        lambda *a, **k: registered.append(1) or real(*a, **k),
    )
    messages: list[dict] = []
    _run_epoch(
        model,
        optimizer,
        _cfg(progress_callback=messages.append),
        _batches(2),
        scaler=_RecordingScaler(),
    )
    assert registered == []
    (message,) = messages
    assert "OFF" in message["status_text"] and "GradScaler" in message["status_text"]
    assert optimizer.param_groups[0]["k"] == 2


def test_without_a_scaler_the_loop_makes_the_calls_it_always_made(monkeypatch):
    """The bf16 / CPU path: one zero_grad at the top, then clip, step, zero_grad
    per batch on the standard path; the fused path is pinned by
    test_fused_back_pass.py, unchanged."""
    events: list[str] = []
    _spy_clip(monkeypatch, events)
    model = _TinyConv()
    optimizer = _optimizer(model)
    real_step, real_zero = optimizer.step, optimizer.zero_grad

    def _step(*a, **k):
        events.append("step")
        return real_step(*a, **k)

    def _zero(*a, **k):
        events.append("zero_grad")
        return real_zero(*a, **k)

    optimizer.step, optimizer.zero_grad = _step, _zero

    _run_epoch(model, optimizer, _cfg(fused_back_pass=False), _batches(3))

    assert events == ["zero_grad"] + ["clip", "step", "zero_grad"] * 3


def test_a_bf16_or_cpu_run_calls_the_loop_with_its_old_arguments(tmp_path, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(
        gt, "_train_one_epoch", lambda *a, **k: seen.update(k) or (0.0, {})
    )
    task, ctx, _frames = _context(tmp_path, monkeypatch, device="cpu")
    task.fwd_model = _TinyConv()

    task.train_epoch(ctx, None, None, [], step_callback=None, boundary_hook=None, start_batch=0)

    assert set(seen) == {
        "use_soft_targets",
        "step_callback",
        "stop_now_event",
        "ema",
        "class_weights",
        "mixup_enabled",
        "pause_event",
        "boundary_hook",
        "start_batch",
        "sample_loss_sink",
    }


def test_an_fp16_run_hands_the_loop_its_scaler(tmp_path, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(
        gt, "_train_one_epoch", lambda *a, **k: seen.update(k) or (0.0, {})
    )
    task, ctx, _frames = _context(tmp_path, monkeypatch, device="cuda", native=False)
    task.fwd_model = _TinyConv()

    task.train_epoch(ctx, None, None, [], step_callback=None, boundary_hook=None, start_batch=0)

    assert seen["scaler"] is task.scaler
    assert seen["scaler_sink"] == task._count_skip


# --------------------------------------------------------------------------- #
# Backups and resume                                                           #
# --------------------------------------------------------------------------- #


def test_scaler_state_and_skipped_steps_ride_the_backup(tmp_path):
    ctx = SimpleNamespace(device=CPU)
    task = GroupTask(group_config(tmp_path, tmp_path / "ck"))
    task.scaler = _RecordingScaler()
    task.scaler.value = 1024.0
    task.skipped_steps = 16
    task._scaler_settled = True
    rows = [{"epoch": 1, "steps": 9, "skipped_steps": 16, "late_skipped_steps": 0, "grad_scale": 1024.0}]
    task.scaler_epochs = list(rows)
    task.scaler_warning = "collapsed"

    state = task.collect_extra_state(
        ctx, rng_epoch_start=None, schedule=None, batch_in_epoch=0
    )
    assert state["grad_scaler"] == {"scale": 1024.0}
    assert state["skipped_steps"] == 16 and state["scaler_settled"] is True

    resumed = GroupTask(group_config(tmp_path, tmp_path / "ck"))
    resumed.scaler = _RecordingScaler()
    resumed.restore_resume_extra(ctx, state)
    assert resumed.scaler.loaded == {"scale": 1024.0}
    assert resumed.skipped_steps == 16 and resumed._scaler_settled is True
    assert resumed.scaler_epochs == rows and resumed.scaler_warning == "collapsed"


def test_a_bf16_backup_is_unchanged_and_resumes_in_either_precision(tmp_path):
    ctx = SimpleNamespace(device=CPU)
    plain = GroupTask(group_config(tmp_path, tmp_path / "ck"))
    state = plain.collect_extra_state(
        ctx, rng_epoch_start=None, schedule=None, batch_in_epoch=0
    )
    assert not {
        "grad_scaler", "skipped_steps", "scaler_settled", "scaler_epochs", "scaler_warning"
    } & set(state)  # fmt: skip

    # Resumed in fp16: a fresh scaler, its opening descent still to come.
    fp16 = GroupTask(group_config(tmp_path, tmp_path / "ck"))
    fp16.scaler = _RecordingScaler()
    fp16.restore_resume_extra(ctx, state)
    assert fp16.scaler.loaded is None
    assert fp16.skipped_steps == 0 and fp16._scaler_settled is False

    # An fp16 backup resumed on a bf16 card has nothing to load its scale into.
    plain.restore_resume_extra(ctx, {"grad_scaler": {"scale": 8.0}, "skipped_steps": 3})
    assert plain.scaler is None


def test_skipped_steps_are_counted_and_reported(tmp_path, monkeypatch):
    """Three of the first epoch's six steps are skipped: counted, reported on
    the epoch frames and in the result; the run trains on."""
    use_micro_trunk(monkeypatch)
    make_group(tmp_path / "grp", per_class=4)
    scaler = _RecordingScaler(skips={1, 2, 4})
    monkeypatch.setattr(group_task, "make_grad_scaler", lambda *_a: scaler)
    frames: list[dict] = []

    result = GenericTrainer().run(
        GroupTask(
            group_config(tmp_path / "grp", tmp_path / "ck", batch_size=2, max_epochs=2)
        ),
        progress_callback=frames.append,
    )

    epochs = [f for f in frames if f.get("type") == "epoch_complete"]
    assert [f["skipped_steps"] for f in epochs] == [3, 0]
    assert epochs[-1]["skipped_steps_total"] == 3
    assert epochs[-1]["grad_scale"] == 65536.0 / 8
    assert scaler.steps == 12 and scaler.applied == 9
    assert result["grad_scaler"] is True and result["skipped_steps"] == 3
    assert result["grad_scale"] == 65536.0 / 8
    # Per epoch: its steps, its skips, the ones after the scaler settled (the
    # fourth step; the first two were its opening descent) and the scale.
    assert [(f["steps"], f["late_skipped_steps"]) for f in epochs] == [(6, 1), (6, 0)]
    assert result["scaler_epochs"] == [
        {"epoch": 1, "steps": 6, "skipped_steps": 3, "late_skipped_steps": 1, "grad_scale": 8192.0},
        {"epoch": 2, "steps": 6, "skipped_steps": 0, "late_skipped_steps": 0, "grad_scale": 8192.0},
    ]
    assert result["scaler_collapsed"] is False and result["scaler_warning"] is None
    assert not _warnings(frames)


# --------------------------------------------------------------------------- #
# A collapsing scaler is said out loud                                         #
# --------------------------------------------------------------------------- #


def _warnings(frames):
    return [f for f in frames if f.get("stage") == "warning"]


def _fp16_run(tmp_path, monkeypatch, scaler, *, epochs=1):
    use_micro_trunk(monkeypatch)
    make_group(tmp_path / "grp", per_class=4)
    monkeypatch.setattr(group_task, "make_grad_scaler", lambda *_a: scaler)
    frames: list[dict] = []
    result = GenericTrainer().run(
        GroupTask(
            group_config(tmp_path / "grp", tmp_path / "ck", batch_size=2, max_epochs=epochs)
        ),
        progress_callback=frames.append,
    )
    return result, frames


def test_the_collapse_rule_separates_the_t4_runs():
    """On the T4 (2026-11-10): from random weights the scale fell to 0.0625;
    from a trained trunk, 2 skipped steps and a scale of 16384."""
    rule = group_task.scaler_collapse
    assert rule(epoch=1, steps=11, late_skipped=9, grad_scale=0.0625)
    assert "below 1" in rule(epoch=2, steps=11, late_skipped=0, grad_scale=0.5)
    assert rule(epoch=1, steps=11, late_skipped=0, grad_scale=16384.0) == ""
    # Most of an epoch skipped after the scaler settled, at any scale.
    assert "4 of its 6 steps" in rule(epoch=1, steps=6, late_skipped=4, grad_scale=4096.0)
    # Two late skips are jitter, not a collapse; and half is not most.
    assert rule(epoch=1, steps=3, late_skipped=2, grad_scale=4096.0) == ""
    assert rule(epoch=1, steps=8, late_skipped=4, grad_scale=4096.0) == ""
    assert rule(epoch=1, steps=6, late_skipped=0, grad_scale=None) == ""


def test_a_scale_below_one_raises_a_warning_frame_and_marks_the_result(tmp_path, monkeypatch):
    """The first step lands, then every step is skipped: 4 -> 0.125 within the
    first epoch, and it stays there. The run trains on, and says so each epoch."""
    scaler = _RecordingScaler(skips=set(range(2, 100)))
    scaler.value = 4.0

    result, frames = _fp16_run(tmp_path, monkeypatch, scaler, epochs=2)

    warnings = _warnings(frames)
    assert [w["epoch"] for w in warnings] == [1, 2]  # once an epoch
    first = warnings[0]
    assert first["warning"] == "grad_scaler_collapsed" and first["type"] == "training_progress"
    assert first["grad_scale"] < 1 and "below 1" in first["status_text"]
    epochs = [f for f in frames if f.get("type") == "epoch_complete"]
    assert frames.index(first) < frames.index(epochs[0])
    assert epochs[0]["stage"] == "training"  # the stream goes back to training
    assert result["scaler_collapsed"] is True
    assert "below 1" in result["scaler_warning"]
    assert [row["grad_scale"] < 1 for row in result["scaler_epochs"]] == [True, True]
    assert "checkpoint_path" in result  # it finished; Engine decides what to do


def test_most_of_an_epoch_skipped_after_settling_is_a_collapse(tmp_path, monkeypatch):
    scaler = _RecordingScaler(skips={2, 3, 4, 5})  # 65536 -> 4096: above 1

    result, frames = _fp16_run(tmp_path, monkeypatch, scaler)

    (warning,) = _warnings(frames)
    assert warning["late_skipped_steps"] == 4 and warning["steps"] == 6
    assert "4 of its 6 steps" in warning["status_text"]
    assert result["scaler_collapsed"] is True


def test_an_opening_descent_is_not_a_collapse(tmp_path, monkeypatch):
    """Five of the first epoch's six steps skipped while the scale comes down
    from 65536, then a clean epoch: nothing to warn about."""
    scaler = _RecordingScaler(skips={1, 2, 3, 4, 5})

    result, frames = _fp16_run(tmp_path, monkeypatch, scaler, epochs=2)

    assert not _warnings(frames)
    assert result["scaler_collapsed"] is False
    assert [row["skipped_steps"] for row in result["scaler_epochs"]] == [5, 0]


def test_a_bf16_or_cpu_run_reports_nothing_new(tmp_path, monkeypatch):
    use_micro_trunk(monkeypatch)
    make_group(tmp_path / "grp", per_class=4)
    frames: list[dict] = []

    result = GenericTrainer().run(
        GroupTask(group_config(tmp_path / "grp", tmp_path / "ck", batch_size=2)),
        progress_callback=frames.append,
    )

    new = {"steps", "late_skipped_steps", "skipped_steps", "grad_scale", "warning"}
    for frame in frames:
        if frame.get("stage") in ("training", "warning"):
            assert not new & set(frame), frame
    assert not {"scaler_epochs", "scaler_collapsed", "scaler_warning", "grad_scale"} & set(result)


def test_a_paused_fp16_run_resumes_with_its_scale_and_count(tmp_path, monkeypatch):
    """Paused at the second step (the first was skipped, halving the scale):
    the backup carries the scale and the count, and the resumed run takes both
    up and finishes."""
    use_micro_trunk(monkeypatch)
    make_group(tmp_path / "grp", per_class=4)
    backups = tmp_path / "backups"
    pause = threading.Event()
    first = _RecordingScaler(skips={1}, pause_after=2, pause_event=pause)
    monkeypatch.setattr(group_task, "make_grad_scaler", lambda *_a: first)

    def _config(**kw):
        return group_config(
            tmp_path / "grp",
            tmp_path / "ck",
            batch_size=2,
            max_epochs=2,
            backup_dir=str(backups),
            **kw,
        )

    paused = GenericTrainer().run(GroupTask(_config()), pause_event=pause)
    assert paused["paused"] is True
    assert list(backups.glob("backup_*.pt"))

    second = _RecordingScaler()
    monkeypatch.setattr(group_task, "make_grad_scaler", lambda *_a: second)
    frames: list[dict] = []
    result = GenericTrainer().run(
        GroupTask(_config(resume_from=str(backups))), progress_callback=frames.append
    )

    assert any(f.get("type") == "training_resumed" for f in frames)
    assert second.loaded == {"scale": 32768.0}
    assert result["skipped_steps"] == 1 and result["grad_scale"] == 32768.0
    assert "paused" not in result


# --------------------------------------------------------------------------- #
# Real GPU (skipped at home; run these on the T4)                              #
# --------------------------------------------------------------------------- #


@needs_gpu
def test_fp16_with_a_scaler_trains_a_group_on_a_real_gpu(tmp_path, monkeypatch):
    use_micro_trunk(monkeypatch)
    make_group(tmp_path / "grp", per_class=8)
    frames: list[dict] = []
    config = group_config(
        tmp_path / "grp",
        tmp_path / "ck",
        device="cuda",
        dtype="float16",
        max_epochs=2,
        progress_callback=frames.append,  # the back-pass mode is announced here
    )

    result = GenericTrainer().run(GroupTask(config))

    assert result["amp_dtype"] == "float16" and result["grad_scaler"] is True
    assert result["lean_grn"] is True
    losses = [f["train_loss"] for f in frames if f.get("type") == "epoch_complete"]
    assert losses and all(math.isfinite(loss) for loss in losses)
    assert any("Fused back pass OFF" in f.get("status_text", "") for f in frames)
    assert any(f.get("stage") == "preflight" for f in frames)
    state = torch.load(
        result["checkpoint_path"], map_location="cpu", weights_only=False
    )["state_dict"]
    assert all(
        bool(torch.isfinite(t).all()) for t in state.values() if t.is_floating_point()
    )


@needs_gpu
def test_group_precision_matches_the_card(tmp_path):
    task = GroupTask(
        group_config(tmp_path, tmp_path / "ck", device="cuda", dtype="bfloat16")
    )
    ctx = task.make_context(None, None, None, None)

    native = torch.cuda.get_device_properties(0).major >= 8
    assert ctx.dtype == (torch.bfloat16 if native else torch.float16)
    assert (task.scaler is not None) is (not native)
    assert task.lean_grn is (not native)
