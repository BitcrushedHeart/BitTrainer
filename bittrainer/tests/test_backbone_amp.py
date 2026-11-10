"""fp16 with a GradScaler for cards without native bf16 (Bitcrush ISSUE-1655).

The Backbone Builder trained in bf16 with no GradScaler. A Turing card (the
Azure T4) only emulates bf16, so the trainer now picks its autocast precision
from the card and steps through a GradScaler whenever that precision is fp16.
The run also reports what it chose and how fast it went, and takes two opt-in
speed knobs (loader workers, channels-last).

CPU-only unless a test says otherwise: ``atto`` backbone and ``use_amp`` off, so
CPU runs stay in fp32 (bf16 autocast on a CPU is several times slower). The two
GPU tests at the bottom skip without CUDA and are the ones to run on the T4.
"""

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file

import bittrainer.amp as amp
import bittrainer.backbone_trainer as bb
import bittrainer.generic.tasks.backbone_task as bt
from bittrainer.backbone_trainer import run_backbone_training
from bittrainer.generic.tasks.backbone_task import BackboneTask

CPU = torch.device("cpu")
CUDA = torch.device("cuda")

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (run on the T4)"
)


def _request(tmp_path, *, epochs=1, max_steps=2, n=8, config=None):
    root = tmp_path / "imgs"
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(n):
        path = root / f"img{i}.png"
        Image.new(
            "RGB", (72, 64), color=((i * 37) % 255, (i * 83) % 255, (i * 151) % 255)
        ).save(path)
        records.append(
            {
                "content_hash": f"{i:02d}" + "ab" * 31,
                "file_paths": [str(path)],
                "binary": {"watermark": "positive" if i % 2 else "negative"},
                "groups": {"shot_type": "closeup" if i % 2 else "wide"},
                "splits": {"train": 1},
            }
        )
    training_config = {
        "image_size": 64,
        "batch_size": 4,
        "epochs": epochs,
        "max_steps": max_steps,
        "validation_split": 0.25,
        "device": "cpu",
        "use_amp": False,
    }
    training_config.update(config or {})
    return {
        "run_id": "run_amp",
        "convnextv2_size": "atto",
        "candidate_checkpoint_path": str(
            tmp_path / "candidates" / "candidate.safetensors"
        ),
        "records": records,
        "training_config": training_config,
        "backbone_init": {"source": "random_init", "checkpoint_path": None},
    }


def _run(request, progress_callback=None):
    return asyncio.run(
        run_backbone_training(request, progress_callback=progress_callback)
    )


def _preparing(frames):
    return next(
        f for f in frames if f.get("stage") == "preparing" and "train_samples" in f
    )


def _epoch_frames(frames):
    return [f for f in frames if f.get("stage") == "training" and "loss" in f]


class _RecordingScaler:
    """Stands in for torch's GradScaler and records the order it is driven in."""

    def __init__(self, events):
        self.events = events
        self.loaded = None

    def scale(self, loss):
        self.events.append("scale")
        return loss

    def unscale_(self, optimizer):
        self.events.append("unscale")

    def step(self, optimizer):
        self.events.append("step")
        optimizer.step()

    def update(self):
        self.events.append("update")

    def get_scale(self):  # read after every update, to count skipped steps
        return 1234.0

    def state_dict(self):
        return {"scale": 1234.0}

    def load_state_dict(self, state):
        self.loaded = dict(state)


# -- choosing the precision -------------------------------------------------


def test_precision_without_a_device_is_unchanged():
    """Callers that never pass a device keep the old answer (bf16, AMP on)."""
    assert bb._amp_settings({}) == (True, torch.bfloat16)
    assert bb._amp_settings({"amp_dtype": "fp16"}) == (True, torch.float16)
    assert bb._amp_settings({"use_amp": False}) == (False, torch.bfloat16)


def test_auto_follows_the_card(monkeypatch):
    monkeypatch.setattr(amp, "native_bf16", lambda device: True)
    assert bb._amp_settings({}, CUDA) == (True, torch.bfloat16)
    assert bb._amp_settings({"amp_dtype": "auto"}, CUDA) == (True, torch.bfloat16)

    monkeypatch.setattr(amp, "native_bf16", lambda device: False)
    assert bb._amp_settings({}, CUDA) == (True, torch.float16)
    assert bb._amp_settings({"amp_dtype": "auto"}, CUDA) == (True, torch.float16)


def test_bf16_asked_for_on_a_card_without_it_becomes_fp16(monkeypatch):
    monkeypatch.setattr(amp, "native_bf16", lambda device: False)
    assert bb._amp_settings({"amp_dtype": "bfloat16"}, CUDA) == (True, torch.float16)


def test_explicit_choices_are_honoured(monkeypatch):
    monkeypatch.setattr(amp, "native_bf16", lambda device: True)
    assert bb._amp_settings({"amp_dtype": "float16"}, CUDA) == (True, torch.float16)
    assert bb._amp_settings({"amp_dtype": "bf16"}, CUDA) == (True, torch.bfloat16)
    enabled, _dtype = bb._amp_settings({"amp_dtype": "float32"}, CUDA)
    assert enabled is False
    enabled, _dtype = bb._amp_settings({"use_amp": False}, CUDA)
    assert enabled is False


def test_a_cpu_counts_as_native_bf16():
    """CPU autocast has always run bf16 here; that must not flip to fp16."""
    assert amp.native_bf16(CPU) is True
    assert bb._amp_settings({}, CPU) == (True, torch.bfloat16)


def test_scaler_only_for_fp16_on_a_gpu():
    assert amp.wants_grad_scaler(CUDA, torch.float16, True) is True
    assert amp.wants_grad_scaler(CUDA, torch.bfloat16, True) is False
    assert amp.wants_grad_scaler(CUDA, torch.float16, False) is False
    assert amp.wants_grad_scaler(CPU, torch.float16, True) is False
    assert amp.make_grad_scaler(CPU, torch.float16, True) is None
    assert amp.make_grad_scaler(CUDA, torch.bfloat16, True) is None


def test_precision_label():
    assert amp.amp_label(True, torch.float16) == "float16"
    assert amp.amp_label(True, torch.bfloat16) == "bfloat16"
    assert amp.amp_label(False, torch.bfloat16) == "off"


# -- stepping through the scaler -------------------------------------------


def test_every_step_goes_through_the_scaler_in_order(tmp_path, monkeypatch):
    """scale -> backward, unscale BEFORE the clipper, then step and update."""
    events: list[str] = []
    scaler = _RecordingScaler(events)
    monkeypatch.setattr(bt, "make_grad_scaler", lambda device, dtype, enabled: scaler)
    real_clip = bt.clip_gradients

    def _clip(model, max_norm):
        events.append("clip")
        real_clip(model, max_norm)

    monkeypatch.setattr(bt, "clip_gradients", _clip)

    _run(_request(tmp_path, epochs=2, max_steps=3))

    step = ["scale", "unscale", "clip", "step", "update"]
    assert len(events) >= 2 * len(step)
    assert events == step * (len(events) // len(step))


def test_scaler_state_rides_the_backup_and_returns_on_resume(tmp_path):
    task = BackboneTask(_request(tmp_path))
    scaler = _RecordingScaler([])
    task._scaler = scaler
    ctx = SimpleNamespace(device=CPU)

    state = task.collect_extra_state(
        ctx, rng_epoch_start=None, schedule=None, batch_in_epoch=0
    )
    assert state["grad_scaler"] == {"scale": 1234.0}

    task.restore_resume_extra(ctx, {"best": {}, "grad_scaler": {"scale": 7.0}})
    assert scaler.loaded == {"scale": 7.0}


def test_backups_from_the_other_precision_still_resume(tmp_path):
    """A bf16 backup has no scaler state; an fp16 backup may resume without one."""
    task = BackboneTask(_request(tmp_path))
    ctx = SimpleNamespace(device=CPU)

    scaler = _RecordingScaler([])
    task._scaler = scaler
    task.restore_resume_extra(ctx, {"best": {}})
    assert scaler.loaded is None

    task._scaler = None
    state = task.collect_extra_state(
        ctx, rng_epoch_start=None, schedule=None, batch_in_epoch=0
    )
    assert state["grad_scaler"] is None
    task.restore_resume_extra(ctx, {"best": {}, "grad_scaler": {"scale": 7.0}})


# -- what the run reports ---------------------------------------------------


def test_run_reports_its_precision_and_speed(tmp_path):
    frames: list[dict] = []
    _run(_request(tmp_path, max_steps=None), frames.append)

    preparing = _preparing(frames)
    assert preparing["device"] == "cpu"
    assert preparing["amp_dtype"] == "off"
    assert preparing["grad_scaler"] is False

    epoch = _epoch_frames(frames)[0]
    assert epoch["train_images"] > 0
    assert epoch["train_seconds"] > 0
    assert epoch["images_per_second"] == pytest.approx(
        epoch["train_images"] / epoch["train_seconds"]
    )


# -- opt-in speed knobs -----------------------------------------------------


def test_loader_workers_follow_the_config(tmp_path):
    task = BackboneTask(_request(tmp_path / "a"))
    plain = task._loader_kwargs(CPU, persistent=True)
    assert plain["num_workers"] == 0
    assert plain["pin_memory"] is False
    assert "persistent_workers" not in plain

    task = BackboneTask(_request(tmp_path / "b", config={"dataloader_workers": 3}))
    val = task._loader_kwargs(CUDA, persistent=True)
    assert val["num_workers"] == 3
    assert val["pin_memory"] is True
    assert val["persistent_workers"] is True
    train = task._loader_kwargs(CUDA, persistent=False)
    assert train["num_workers"] == 3
    assert train["persistent_workers"] is False
    assert train["batch_size"] == 4
    assert train["collate_fn"] is bb._collate


def _stem_weight(model):
    return next(
        p
        for p in model.parameters()
        if p.ndim == 4 and p.shape[1] > 1 and p.shape[2] > 1
    )


def test_channels_last_is_opt_in(tmp_path, monkeypatch):
    captured = []
    real = bt.make_optimizer

    def _spy(model, **kw):
        captured.append(model)
        return real(model, **kw)

    monkeypatch.setattr(bt, "make_optimizer", _spy)

    _run(_request(tmp_path / "a", max_steps=1))
    _run(_request(tmp_path / "b", max_steps=1, config={"channels_last": True}))

    plain, nhwc = captured
    assert not _stem_weight(plain).is_contiguous(memory_format=torch.channels_last)
    assert _stem_weight(nhwc).is_contiguous(memory_format=torch.channels_last)


# -- real GPU (skipped at home while the card is busy; run these on the T4) --


@needs_gpu
def test_fp16_with_a_scaler_trains_on_a_real_gpu(tmp_path):
    frames: list[dict] = []
    request = _request(
        tmp_path,
        epochs=2,
        max_steps=6,
        n=16,
        config={"device": "cuda", "use_amp": True, "amp_dtype": "float16"},
    )
    result = _run(request, frames.append)

    preparing = _preparing(frames)
    assert preparing["amp_dtype"] == "float16"
    assert preparing["grad_scaler"] is True

    losses = [f["loss"] for f in _epoch_frames(frames)]
    assert losses and all(math.isfinite(loss) for loss in losses)
    state = load_file(result["candidate_checkpoint_path"])
    assert all(
        bool(torch.isfinite(t).all()) for t in state.values() if t.is_floating_point()
    )


@needs_gpu
def test_auto_precision_matches_the_card(tmp_path):
    frames: list[dict] = []
    _run(_request(tmp_path, config={"device": "cuda", "use_amp": True}), frames.append)

    native = torch.cuda.get_device_properties(0).major >= 8
    preparing = _preparing(frames)
    assert preparing["amp_dtype"] == ("bfloat16" if native else "float16")
    assert preparing["grad_scaler"] is (not native)
