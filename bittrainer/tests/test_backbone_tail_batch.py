"""The high-resolution tail trains at its own batch size, and a run is checked
before it starts (Bitcrush ISSUE-1655).

``finetune_image_size`` / ``finetune_epochs`` switch the last epochs to a higher
resolution. They used to keep ``batch_size``: a Base run at 224px and batch 64
would have run out of memory at the switch to 384px, a day into training.

* ``finetune_batch_size`` gives the tail its own batch size (unset: as before).
  The loaders, the progress frames and a resume all follow it. It is not part
  of a backup's identity, so a paused run can come back with another one.
* The preflight runs one real training step (forward, loss, backward) at every
  (resolution, batch size) still ahead, before the first epoch, and refuses the
  run when one does not fit. It leaves the model, the optimizer, the scaler, the
  EMA and the RNG streams as they were.

CPU-only unless a test says otherwise, on a stand-in trunk of one small
convolution (``backbone_stub``): 48px and 64px generated images, AMP off, a
whole run in a fraction of a second. The tests at the bottom train a real
``atto`` on a GPU and skip without one; they are the ones to run on the T4.
"""

from __future__ import annotations

import asyncio
import gc
import math
import random
import threading

import numpy as np
import pytest
import torch
from PIL import Image

import bittrainer.backbone_trainer as bb
import bittrainer.generic.tasks.backbone_task as bt
from bittrainer.backbone_trainer import run_backbone_training
from bittrainer.generic.tasks.backbone_task import BackboneRunRefused, BackboneTask
from bittrainer.tests.backbone_stub import use_stub_trunk
from bittrainer.training_state import fingerprint_matches

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (run on the T4)"
)

_OOM = "CUDA out of memory. Tried to allocate 2.00 GiB."


@pytest.fixture(autouse=True)
def _trunk(request, monkeypatch):
    """Everything but the real-GPU tests runs on the stand-in trunk."""
    if "real_gpu" not in request.node.name:
        use_stub_trunk(monkeypatch)


def _request(tmp_path, *, n=12, config=None):
    """Two epochs: one at 48px and batch 4, then a 64px tail at batch 2.

    With 12 images and a quarter held out, 9 train: 3 steps, then 5.
    """
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
        "image_size": 48,
        "finetune_image_size": 64,
        "finetune_epochs": 1,
        "finetune_batch_size": 2,
        "batch_size": 4,
        "epochs": 2,
        "validation_split": 0.25,
        "device": "cpu",
        "use_amp": False,
        "progress_every_steps": 1,
        "oversample_positives": False,
    }
    training_config.update(config or {})
    return {
        "run_id": "run_tail",
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


def _steps(frames, epoch=None):
    return [
        f
        for f in frames
        if f.get("stage") == "training_step" and epoch in (None, f["epoch"])
    ]


def _epochs(frames):
    return [f for f in frames if f.get("stage") == "training" and "loss" in f]


def _spy_loaders(monkeypatch):
    """(kind, image size, batch size) of every DataLoader the task builds."""
    built = []
    real = bt.DataLoader

    def _loader(dataset, **kwargs):
        size = dataset.transform.transforms[0].size
        size = size if isinstance(size, int) else size[0]
        kind = "train" if kwargs.get("shuffle") else "val"
        built.append((kind, size, kwargs["batch_size"]))
        return real(dataset, **kwargs)

    monkeypatch.setattr(bt, "DataLoader", _loader)
    return built


def _built(tmp_path, **config):
    """A task taken by hand to the point where its first epoch would start,
    with the preflight held back so a test can call it itself."""
    frames: list[dict] = []
    task = BackboneTask(_request(tmp_path, config={"preflight": False, **config}))
    ctx = task.make_context(frames.append, None, None, None)
    task.fingerprint_init(ctx)
    model = task.create_model(ctx, None)
    batch = task.resolve_batch_size(ctx, model, None)
    optimizer, _scheduler, _t_max = task.create_optimizer(ctx, model, batch, None)
    task.preflight = True
    return task, ctx, model, optimizer, frames


# -- the tail's own batch size ------------------------------------------------


def test_the_tail_trains_and_validates_at_its_own_batch_size(tmp_path, monkeypatch):
    built = _spy_loaders(monkeypatch)
    emas, t_max = [], []
    real_ema, real_cosine = bt.ModelEMA, bt.CosineAnnealingLR

    def _ema(model, decay):
        emas.append(real_ema(model, decay=decay))
        return emas[-1]

    def _cosine(optimizer, T_max):  # noqa: N803 - torch's own argument name
        t_max.append(T_max)
        return real_cosine(optimizer, T_max=T_max)

    monkeypatch.setattr(bt, "ModelEMA", _ema)
    monkeypatch.setattr(bt, "CosineAnnealingLR", _cosine)
    frames: list[dict] = []
    request = _request(
        tmp_path,
        config={"backup_dir": str(tmp_path / "backups"), "backup_every_steps": 2},
    )

    _run(request, frames.append)

    assert built == [
        ("val", 48, 4),
        ("train", 48, 4),
        ("val", 64, 2),
        ("train", 64, 2),
    ]
    first, tail = _epochs(frames)
    assert (first["image_size"], first["batch_size"]) == (48, 4)
    assert (tail["image_size"], tail["batch_size"]) == (64, 2)
    # The heartbeat counts the tail's own, smaller batches.
    assert {(f["image_size"], f["batch_size"]) for f in _steps(frames, 1)} == {(48, 4)}
    assert {(f["image_size"], f["batch_size"]) for f in _steps(frames, 2)} == {(64, 2)}
    assert _steps(frames, 1)[-1]["batches"] == math.ceil(first["train_images"] / 4)
    assert _steps(frames, 2)[-1]["batches"] == math.ceil(tail["train_images"] / 2)
    assert tail["steps"] - first["steps"] > first["steps"]
    switch = next(f for f in frames if "finetune_batch_size" in f)
    assert (
        switch["finetune_batch_size"] == 2 and "batch size 2" in switch["status_text"]
    )

    # Steps are steps: the backup cadence and the EMA count every one of them,
    # and the schedule moves once per epoch whatever an epoch's step count.
    backed_up = {f["global_step"] for f in frames if f.get("type") == "backup_complete"}
    assert set(range(2, tail["steps"] + 1, 2)) <= backed_up
    assert emas[0].n_updates == tail["steps"] and emas[0].decay == 0.9999
    assert t_max == [2]
    # Off a GPU the preflight is asked for, not assumed.
    assert not [f for f in frames if f.get("stage") == "preflight"]


def test_an_unset_tail_batch_size_is_todays_behaviour(tmp_path, monkeypatch):
    built = _spy_loaders(monkeypatch)
    request = _request(tmp_path)
    del request["training_config"]["finetune_batch_size"]

    assert BackboneTask(request).finetune_batch_size == 4
    _run(request)

    assert [batch for _kind, _size, batch in built] == [4, 4, 4, 4]


def test_max_steps_counts_optimizer_steps_at_either_batch_size(tmp_path):
    frames: list[dict] = []
    _run(_request(tmp_path / "whole"), frames.append)
    first_epoch = _epochs(frames)[0]["steps"]

    capped: list[dict] = []
    _run(
        _request(tmp_path / "capped", config={"max_steps": first_epoch + 2}),
        capped.append,
    )

    tail = _epochs(capped)[-1]
    assert tail["steps"] == first_epoch + 2
    assert tail["train_images"] == 2 * 2  # two tail steps of two images


# -- resuming -----------------------------------------------------------------


def _pause_when(wanted):
    """A callback that sets its pause event on the first frame *wanted* accepts."""
    pause = threading.Event()
    frames: list[dict] = []

    def _callback(frame):
        frames.append(frame)
        if wanted(frame):
            pause.set()

    return pause, _callback, frames


def _resume(request, paused, **config):
    frames: list[dict] = []
    resumed = {
        **request,
        "training_config": {
            **request["training_config"],
            "resume_from": paused["backup_path"],
            **config,
        },
    }
    result = bb._train_backbone(resumed, frames.append, threading.Event())
    return result, frames


def test_a_backup_taken_inside_the_tail_resumes_into_the_tail(tmp_path, monkeypatch):
    request = _request(tmp_path, config={"backup_dir": str(tmp_path / "backups")})
    pause, callback, _frames = _pause_when(
        lambda f: f.get("stage") == "training_step" and f["epoch"] == 2
    )
    paused = bb._train_backbone(request, callback, threading.Event(), pause_event=pause)
    assert paused["paused"] is True and paused["epoch"] == 1

    built = _spy_loaders(monkeypatch)
    result, frames = _resume(request, paused)

    resumed = next(f for f in frames if f.get("type") == "training_resumed")
    assert (resumed["epoch"], resumed["image_size"], resumed["batch_size"]) == (
        1,
        64,
        2,
    )
    assert built == [("val", 64, 2), ("train", 64, 2)]
    assert {f["epoch"] for f in _steps(frames)} == {2}
    assert result["epochs_completed"] == 2


def test_a_backup_taken_just_before_the_tail_resumes_into_it_at_another_batch_size(
    tmp_path, monkeypatch
):
    """The backup at the end of the last low-resolution epoch. Coming back with
    a smaller tail batch (after an out-of-memory, say) must not cost the run:
    the epoch restarts from its beginning, and nothing saved depends on the
    batch size."""
    request = _request(tmp_path, config={"backup_dir": str(tmp_path / "backups")})
    pause, callback, _frames = _pause_when(
        lambda f: f.get("stage") == "training" and "loss" in f and f["epoch"] == 1
    )
    paused = bb._train_backbone(request, callback, threading.Event(), pause_event=pause)
    assert paused["paused"] is True and paused["epoch"] == 1

    built = _spy_loaders(monkeypatch)
    result, frames = _resume(request, paused, finetune_batch_size=3)

    assert not any(f.get("type") == "resume_skipped" for f in frames)
    resumed = next(f for f in frames if f.get("type") == "training_resumed")
    assert (resumed["image_size"], resumed["batch_size"]) == (64, 3)
    assert built == [("val", 64, 3), ("train", 64, 3)]
    assert result["epochs_completed"] == 2 and result["best_epoch"] == 2


def _fingerprint(task):
    ctx = task.make_context(None, None, None, None)
    task.fingerprint_init(ctx)
    return ctx.fingerprint


def test_the_tail_batch_size_is_not_part_of_a_backups_identity(tmp_path):
    two = _fingerprint(BackboneTask(_request(tmp_path)))
    three = _fingerprint(
        BackboneTask(_request(tmp_path, config={"finetune_batch_size": 3}))
    )
    larger = _fingerprint(
        BackboneTask(_request(tmp_path, config={"finetune_image_size": 96}))
    )

    assert fingerprint_matches(two, three) and fingerprint_matches(three, two)
    # The tail's resolution still is: those backups belong to another run.
    assert not fingerprint_matches(two, larger)


# -- the preflight ------------------------------------------------------------


def test_the_phases_still_ahead_are_the_ones_checked(tmp_path):
    task = BackboneTask(_request(tmp_path, config={"epochs": 5, "finetune_epochs": 2}))
    assert task._phases(0) == [("main", 48, 4), ("tail", 64, 2)]
    assert task._phases(2) == [("main", 48, 4), ("tail", 64, 2)]
    # Resumed inside the tail: the low-resolution phase is behind it.
    assert task._phases(3) == [("tail", 64, 2)]
    assert task._phases(5) == []

    plain = BackboneTask(_request(tmp_path, config={"finetune_epochs": 0}))
    assert plain._phases(0) == [("main", 48, 4)]


def test_the_preflight_runs_one_real_step_per_phase_and_leaves_everything_as_it_was(
    tmp_path, monkeypatch
):
    task, ctx, model, optimizer, frames = _built(tmp_path)
    task._scaler = torch.amp.GradScaler("cpu")
    shapes, backwards = [], []
    model.backbone.register_forward_pre_hook(
        lambda _module, args: shapes.append(tuple(args[0].shape))
    )
    real_backward = torch.Tensor.backward

    def _backward(tensor, *args, **kwargs):
        backwards.append(1)
        return real_backward(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", _backward)
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    average = {k: v.clone() for k, v in task.ema.module.state_dict().items()}
    scaler_state = task._scaler.state_dict()
    torch_rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()[1].copy()

    task._preflight(ctx, model, optimizer, 0)

    # A real forward and backward at each size the run will use...
    assert shapes == [(4, 3, 48, 48), (2, 3, 64, 64)]
    assert len(backwards) == 2
    checked = [f for f in frames if f.get("stage") == "preflight"]
    assert [(f["phase"], f["image_size"], f["batch_size"]) for f in checked] == [
        ("main", 48, 4),
        ("tail", 64, 2),
    ]
    # ...and nothing kept of them.
    for key, value in model.state_dict().items():
        assert torch.equal(value, weights[key]), key
    for key, value in task.ema.module.state_dict().items():
        assert torch.equal(value, average[key]), key
    assert task.ema.n_updates == 0
    assert all(p.grad is None for p in model.parameters())
    assert not optimizer.state  # never stepped
    assert task._scaler.state_dict() == scaler_state
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng)
    assert task.step == 0 and task.skipped_steps == 0
    assert model.training is True
    assert (task._train_seconds, task._train_images) == (0.0, 0)


def _dry_step_fails_at(monkeypatch, size, message=_OOM):
    real = BackboneTask._dry_step

    def _dry_step(self, ctx, model, image_size, batch_size):
        if image_size == size:
            raise RuntimeError(message)
        return real(self, ctx, model, image_size, batch_size)

    monkeypatch.setattr(BackboneTask, "_dry_step", _dry_step)


def test_a_tail_that_does_not_fit_refuses_the_run_before_it_trains(
    tmp_path, monkeypatch
):
    _dry_step_fails_at(monkeypatch, 64)
    frames: list[dict] = []
    request = _request(
        tmp_path,
        config={"preflight": True, "backup_dir": str(tmp_path / "backups")},
    )

    with pytest.raises(BackboneRunRefused) as refused:
        _run(request, frames.append)

    reason = str(refused.value)
    assert "tail" in reason and "64px" in reason and "batch size 2" in reason
    assert "finetune_batch_size" in reason
    assert _OOM in str(refused.value.__cause__)
    # The phase that fits was checked first; nothing trained, nothing to resume.
    assert [f["phase"] for f in frames if f.get("stage") == "preflight"] == ["main"]
    assert not _steps(frames)
    assert not list((tmp_path / "backups").glob("backup_*.pt"))


def test_a_main_phase_that_does_not_fit_is_named_too(tmp_path, monkeypatch):
    _dry_step_fails_at(monkeypatch, 48)
    with pytest.raises(BackboneRunRefused) as refused:
        _run(_request(tmp_path, config={"preflight": True}))

    reason = str(refused.value)
    assert "main phase" in reason and "48px" in reason and "batch size 4" in reason
    assert "batch_size (--batch-size)" in reason


def test_an_error_that_is_not_about_memory_is_not_called_a_refusal(
    tmp_path, monkeypatch
):
    _dry_step_fails_at(monkeypatch, 64, "mat1 and mat2 shapes cannot be multiplied")
    with pytest.raises(RuntimeError, match="cannot be multiplied") as raised:
        _run(_request(tmp_path, config={"preflight": True}))
    assert not isinstance(raised.value, BackboneRunRefused)


def test_running_out_of_memory_is_recognised_however_it_is_worded():
    assert bt._out_of_memory(RuntimeError(_OOM))
    # cuDNN failing to get its workspace on a full card says it differently.
    assert bt._out_of_memory(RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED"))
    assert not bt._out_of_memory(RuntimeError("shapes cannot be multiplied"))
    assert not bt._out_of_memory(NotImplementedError("missing the forward function"))


def test_a_step_that_leaves_no_headroom_is_refused(tmp_path, monkeypatch):
    task, ctx, model, optimizer, frames = _built(tmp_path)
    room = 10 * 2**30
    peaks = {48: 0.5 * room, 64: 0.96 * room}
    monkeypatch.setattr(bt, "_vram_room", lambda device: room)
    monkeypatch.setattr(
        BackboneTask, "_dry_step", lambda self, ctx, model, size, batch: peaks[size]
    )

    with pytest.raises(BackboneRunRefused, match="tail") as refused:
        task._preflight(ctx, model, optimizer, 0)
    assert "9.6 GB" in str(refused.value) and "10.0 GB" in str(refused.value)

    # A fresh optimizer builds its state on its first real step and keeps it:
    # a step that fits on its own, but not beside that state, is refused too.
    to_come = bt._OPT_STATE_MULT * sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    peaks[64] = 0.95 * room - to_come / 2
    with pytest.raises(BackboneRunRefused, match="tail"):
        task._preflight(ctx, model, optimizer, 0)

    # A resumed optimizer already holds its state: the same step now fits.
    optimizer.state[next(model.parameters())] = {"restored": True}
    frames.clear()
    task._preflight(ctx, model, optimizer, 0)
    assert [f["phase"] for f in frames if f.get("stage") == "preflight"] == [
        "main",
        "tail",
    ]
    assert frames[-1]["peak_vram_mb"] == round(peaks[64] / 2**20)


def test_the_preflight_is_on_for_a_gpu_and_off_a_gpu_only_when_asked(
    tmp_path, monkeypatch
):
    task, ctx, model, optimizer, frames = _built(tmp_path)
    monkeypatch.setattr(bt, "_vram_room", lambda device: None)
    monkeypatch.setattr(
        BackboneTask, "_dry_step", lambda self, ctx, model, size, batch: None
    )

    task.preflight = None  # not set in the config
    task._preflight(ctx, model, optimizer, 0)
    assert frames == []

    ctx.device = torch.device("cuda")
    task._preflight(ctx, model, optimizer, 0)
    assert len(frames) == 2

    task.preflight = False
    task._preflight(ctx, model, optimizer, 0)
    assert len(frames) == 2


# -- real GPU (skipped at home; run these on the T4) ---------------------------

_FP16 = {"device": "cuda", "use_amp": True, "amp_dtype": "float16"}


@needs_gpu
def test_a_dry_step_measures_memory_and_changes_nothing_on_a_real_gpu(tmp_path):
    task, ctx, model, _optimizer, _frames = _built(tmp_path, **_FP16)
    assert task._scaler is not None
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    scaler_state = task._scaler.state_dict()
    cpu_rng = torch.get_rng_state().clone()
    gpu_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]

    small = task._dry_step(ctx, model, 64, 2)
    large = task._dry_step(ctx, model, 64, 16)

    assert 0 < small < large
    for key, value in model.state_dict().items():
        assert torch.equal(value, weights[key]), key
    assert all(p.grad is None for p in model.parameters())
    assert task._scaler.state_dict() == scaler_state
    assert task.ema.n_updates == 0 and task.step == 0
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert all(
        torch.equal(now, before)
        for now, before in zip(torch.cuda.get_rng_state_all(), gpu_rng)
    )


@needs_gpu
def test_the_preflight_refuses_a_step_that_runs_out_of_memory_on_a_real_gpu(
    tmp_path, monkeypatch
):
    """The allocator is capped at half a gigabyte above what it holds, so the
    48px phase fits and a 384px batch of 192 cannot, on any card. Only running
    out for real is under test, so the headroom rule is switched off."""
    task, ctx, model, optimizer, frames = _built(
        tmp_path, finetune_image_size=384, finetune_batch_size=192, **_FP16
    )
    monkeypatch.setattr(bt, "_PREFLIGHT_SHARE", 1.0)
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    gc.collect()
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory
    cap = torch.cuda.memory_reserved() + 2**29
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap / total))
    try:
        with pytest.raises(BackboneRunRefused, match="tail") as refused:
            task._preflight(ctx, model, optimizer, 0)
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)

    assert "384px" in str(refused.value) and "batch size 192" in str(refused.value)
    assert [f["phase"] for f in frames if f.get("stage") == "preflight"] == ["main"]
    for key, value in model.state_dict().items():
        assert torch.equal(value, weights[key]), key
    assert all(p.grad is None for p in model.parameters())


@needs_gpu
def test_a_tail_run_switches_batch_size_in_fp16_on_a_real_gpu(tmp_path):
    frames: list[dict] = []
    result = _run(_request(tmp_path, n=24, config=_FP16), frames.append)

    # On a GPU the preflight runs without being asked for.
    checked = [f for f in frames if f.get("stage") == "preflight"]
    assert [(f["phase"], f["image_size"], f["batch_size"]) for f in checked] == [
        ("main", 48, 4),
        ("tail", 64, 2),
    ]
    assert all(f["peak_vram_mb"] > 0 for f in checked)
    first, tail = _epochs(frames)
    assert (tail["image_size"], tail["batch_size"]) == (64, 2)
    assert math.isfinite(first["loss"]) and math.isfinite(tail["loss"])
    assert result["epochs_completed"] == 2
