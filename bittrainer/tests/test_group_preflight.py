"""Refuse a group run that cannot fit, before it trains (Bitcrush ISSUE-1690).

Nothing stopped a group run whose training step did not fit the card: the batch
probe read every error as "out of memory" and fell back to batch 8, and the
compile warm-up swallows every exception, running out of memory included, and
carries on uncompiled. Now:

* one real training step at the chosen batch size, on the largest bucket the
  run trains, runs before compile and before the first epoch; a step that runs
  out of memory, or leaves no headroom, refuses the run with ``RunRefused``
  (the backbone trainer's refusal is the same kind) and leaves no backup;
* an error that is not about memory is raised as itself;
* it runs on an fp16 card by default (``preflight`` forces or skips it), never
  without a run context, and changes nothing it touches.

CPU-only, on the micro stand-in trunk (``group_stub``). The test named
``..._real_gpu`` skips without CUDA and is the one to run on the T4.
"""

from __future__ import annotations

import gc
import random

import numpy as np
import pytest
import torch

import bittrainer.generic.tasks.group_task as group_task
import bittrainer.group_trainer as gt
from bittrainer.autobatch import _OPT_STATE_MULT, RunRefused
from bittrainer.generic.generic_trainer import GenericTrainer
from bittrainer.generic.tasks.backbone_task import BackboneRunRefused
from bittrainer.generic.tasks.group_task import GroupTask
from bittrainer.tests.group_stub import (
    group_config,
    make_group,
    micro_convnext,
    stir_grn,
    use_micro_trunk,
)

CPU = torch.device("cpu")
_OOM = "CUDA out of memory. Tried to allocate 2.00 GiB."

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (run on the T4)"
)


def _dry_step_raises(monkeypatch, message=_OOM):
    def _dry_step(self, ctx, model, batch_size):
        raise RuntimeError(message)

    monkeypatch.setattr(GroupTask, "_dry_step", _dry_step)


def _never_reached(monkeypatch) -> list[str]:
    """Record whatever runs after the preflight: compile and the epochs."""
    reached: list[str] = []

    def _compile(model, **_kw):
        reached.append("maybe_compile")
        return model, False

    monkeypatch.setattr(group_task, "maybe_compile", _compile)
    monkeypatch.setattr(
        group_task, "prewarm_compile", lambda *a, **k: reached.append("prewarm") or True
    )
    real = gt._train_one_epoch

    def _epoch(*a, **k):
        if not k.get("dry_run"):
            reached.append("epoch")
        return real(*a, **k)

    monkeypatch.setattr(gt, "_train_one_epoch", _epoch)
    return reached


def _run(tmp_path, monkeypatch, frames=None, **overrides):
    use_micro_trunk(monkeypatch)
    make_group(tmp_path / "grp")
    settings = {
        "preflight": True,
        "use_compile": True,
        "backup_dir": str(tmp_path / "backups"),
        **overrides,
    }
    config = group_config(tmp_path / "grp", tmp_path / "ck", **settings)
    return GenericTrainer().run(
        GroupTask(config), progress_callback=None if frames is None else frames.append
    )


def _built(tmp_path, *, preflight=True, buckets=None):
    """A task as the trainer leaves it just before ``create_optimizer``."""
    frames: list[dict] = []
    task = GroupTask(group_config(tmp_path, tmp_path / "ck", preflight=preflight))
    ctx = task.make_context(frames.append, None, None, None)
    task.bucket_counts = buckets or {(64, 64): 12}
    model = micro_convnext()
    stir_grn(model)
    optimizer = gt._make_optimizer(model, task.config)
    return task, ctx, model, optimizer, frames


# -- refusing -------------------------------------------------------------------


def test_a_step_that_runs_out_of_memory_refuses_before_compile_and_training(
    tmp_path, monkeypatch
):
    _dry_step_raises(monkeypatch)
    reached = _never_reached(monkeypatch)
    frames: list[dict] = []

    with pytest.raises(RunRefused) as refused:
        _run(tmp_path, monkeypatch, frames)

    reason = str(refused.value)
    assert "batch size 4" in reason and "64x64" in reason
    # Worded so a retry that halves the batch on "out of memory" recognises it.
    assert "out of memory" in reason and "batch_size" in reason
    assert _OOM in str(refused.value.__cause__)
    assert reached == []  # no compile, no epoch
    assert not [f for f in frames if f.get("type") == "epoch_complete"]
    assert not list((tmp_path / "backups").glob("backup_*.pt*"))


def test_a_refusal_is_the_kind_the_backbone_trainer_raises():
    assert issubclass(BackboneRunRefused, RunRefused)
    assert issubclass(RunRefused, RuntimeError)


def test_an_error_that_is_not_about_memory_is_raised_as_itself(tmp_path, monkeypatch):
    _dry_step_raises(monkeypatch, "mat1 and mat2 shapes cannot be multiplied")
    reached = _never_reached(monkeypatch)

    with pytest.raises(RuntimeError, match="cannot be multiplied") as raised:
        _run(tmp_path, monkeypatch)

    assert not isinstance(raised.value, RunRefused)
    assert reached == []


def test_a_step_that_leaves_no_headroom_is_refused(tmp_path, monkeypatch):
    task, ctx, model, optimizer, frames = _built(tmp_path)
    room = 10 * 2**30
    peaks = [0.96 * room]
    monkeypatch.setattr(group_task, "vram_room", lambda device: room)
    monkeypatch.setattr(
        GroupTask, "_dry_step", lambda self, ctx, model, batch: peaks[-1]
    )

    with pytest.raises(RunRefused) as refused:
        task._preflight(ctx, model, 4, optimizer)
    assert "9.6 GB" in str(refused.value) and "10.0 GB" in str(refused.value)

    # A fresh optimizer builds its state on its first real step and keeps it:
    # a step that fits on its own, but not beside that state, is refused too.
    to_come = _OPT_STATE_MULT * sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    peaks.append(0.95 * room - to_come / 2)
    with pytest.raises(RunRefused):
        task._preflight(ctx, model, 4, optimizer)

    # A resumed optimizer already holds its state: the same step now fits.
    optimizer.state[next(model.parameters())] = {"restored": True}
    task._preflight(ctx, model, 4, optimizer)
    (fits,) = [f for f in frames if f.get("stage") == "preflight"]
    assert fits["batch_size"] == 4 and fits["bucket"] == [64, 64]
    assert fits["peak_vram_mb"] == round(peaks[-1] / 2**20)


# -- when it runs ---------------------------------------------------------------


def test_when_the_preflight_runs(tmp_path, monkeypatch):
    task, ctx, model, optimizer, _frames = _built(tmp_path, preflight=None)
    ran: list[int] = []
    monkeypatch.setattr(
        GroupTask, "_dry_step", lambda self, ctx, model, batch: ran.append(batch)
    )

    task._preflight(ctx, model, 4, optimizer)  # a CPU, no scaler: off
    assert ran == []
    task.scaler = object()  # an fp16 card: on without being asked
    task._preflight(ctx, model, 4, optimizer)
    assert ran == [4]
    task._preflight(None, model, 4, optimizer)  # no run context: never
    assert ran == [4]
    task.config.preflight = False  # skipped when asked
    task._preflight(ctx, model, 4, optimizer)
    assert ran == [4]
    task.scaler, task.config.preflight = None, True  # forced anywhere
    task._preflight(ctx, model, 4, optimizer)
    assert ran == [4, 4]


def test_a_bf16_or_cpu_run_takes_no_extra_step(tmp_path, monkeypatch):
    """With nothing asked for, a run off an fp16 card never runs the dry step."""
    ran: list[int] = []
    monkeypatch.setattr(
        GroupTask, "_dry_step", lambda self, ctx, model, batch: ran.append(batch)
    )
    _run(tmp_path, monkeypatch, preflight=None, use_compile=False)
    assert ran == []


# -- the step itself --------------------------------------------------------------


def test_the_dry_step_is_one_step_of_the_largest_bucket_and_changes_nothing(
    tmp_path, monkeypatch
):
    task, ctx, model, _optimizer, _frames = _built(
        tmp_path, buckets={(64, 64): 9, (96, 64): 3, (32, 32): 0}
    )
    task.scaler = torch.amp.GradScaler("cpu")
    scaler_state = task.scaler.state_dict()
    seen: list[dict] = []
    real = gt._train_one_epoch

    def _epoch(model_, loader, optimizer_, *a, **k):
        seen.append({"shape": tuple(loader[0][0].shape), **k})
        return real(model_, loader, optimizer_, *a, **k)

    monkeypatch.setattr(gt, "_train_one_epoch", _epoch)
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    model.eval()
    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)
    torch_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    numpy_rng = np.random.get_state()[1].copy()

    assert task._dry_step(ctx, model, 5) is None  # no peak to measure off a GPU

    (call,) = seen
    assert call["shape"] == (5, 3, 64, 96)  # the largest bucket, at the run's batch
    assert call["dry_run"] is True and call["scaler"] is task.scaler
    for key, value in model.state_dict().items():
        assert torch.equal(value, weights[key]), key
    assert all(p.grad is None for p in model.parameters())
    assert model.training is False
    assert task.scaler.state_dict() == scaler_state
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng)


def test_the_dry_step_stops_after_the_backward_pass():
    model = micro_convnext()
    calls: list[str] = []

    class _Optimizer:
        def zero_grad(self):
            calls.append("zero_grad")

        def step(self):  # pragma: no cover - must not run
            calls.append("step")

    images = torch.zeros(2, 3, 32, 32, dtype=torch.uint8)
    gt._train_one_epoch(
        model,
        [(images, torch.tensor([0, 1])), (images, torch.tensor([1, 2]))],
        _Optimizer(),
        group_config(".", ".", progress_callback=lambda m: calls.append("announce")),
        CPU,
        torch.float32,
        dry_run=True,
    )

    assert calls == ["zero_grad"]  # no step, no second batch, nothing announced
    assert any(p.grad is not None for p in model.parameters())  # it did go backward


# -- on a real card ----------------------------------------------------------------


@needs_gpu
def test_the_preflight_refuses_a_step_that_runs_out_of_memory_on_a_real_gpu(
    tmp_path, monkeypatch
):
    """The allocator is capped at half a gigabyte above what it holds, so a
    batch of 512 images at 512x512 cannot fit, on any card. Only running out
    for real is under test, so the headroom rule is switched off."""
    task, ctx, model, optimizer, _frames = _built(tmp_path, buckets={(512, 512): 600})
    ctx.device = torch.device("cuda")
    model = model.cuda()
    optimizer = gt._make_optimizer(model, task.config)
    monkeypatch.setattr(group_task, "_PREFLIGHT_SHARE", 1.0)
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    gc.collect()
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory
    cap = torch.cuda.memory_reserved() + 2**29
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap / total))
    try:
        with pytest.raises(RunRefused) as refused:
            task._preflight(ctx, model, 512, optimizer)
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)

    assert "batch size 512" in str(refused.value)
    for key, value in model.state_dict().items():
        assert torch.equal(value, weights[key]), key
    assert all(p.grad is None for p in model.parameters())
