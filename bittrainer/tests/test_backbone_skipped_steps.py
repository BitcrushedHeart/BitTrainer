"""Optimizer steps the fp16 GradScaler skips are counted and reported
(Bitcrush ISSUE-1655).

In fp16 the scaler skips ``optimizer.step()`` whenever a gradient came back
inf / nan, and halves its scale. It starts at 65536, so a fresh run skips its
opening steps while the scale comes down (16 of them on the first T4 run). None
of that was visible, and neither would a run that kept skipping have been.

* Skipped steps and the current scale ride the in-epoch ``training_step``
  frames, the epoch frames and the run's result.
* Skips after the scaler has settled (its first applied step) are the ones that
  matter: more than a small share of an epoch's steps raises a warning frame.
* The opening descent is left alone; the last two tests say why.

CPU-only unless a test says otherwise, on a stand-in trunk of one small
convolution (``backbone_stub``). A scripted stand-in plays the scaler in the
trainer tests; torch's own GradScaler (it runs on a CPU) is used where its
arithmetic is the point. The last test trains a real ``atto`` in fp16 on a GPU
and skips without one.
"""

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import bittrainer.amp as amp
import bittrainer.generic.tasks.backbone_task as bt
from bittrainer.backbone_trainer import run_backbone_training
from bittrainer.generic.tasks.backbone_task import BackboneTask
from bittrainer.tests.backbone_stub import use_stub_trunk

CPU = torch.device("cpu")
FP16_MAX = 65504.0

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (run on the T4)"
)


@pytest.fixture(autouse=True)
def _trunk(request, monkeypatch):
    """Everything but the real-GPU test runs on the stand-in trunk."""
    if "real_gpu" not in request.node.name:
        use_stub_trunk(monkeypatch)


def _request(tmp_path, *, n=20, config=None):
    """20 images, 14 of them training: 7 steps an epoch at batch 2."""
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
        "batch_size": 2,
        "epochs": 1,
        "validation_split": 0.25,
        "device": "cpu",
        "use_amp": False,
        "progress_every_steps": 1,
        "oversample_positives": False,
    }
    training_config.update(config or {})
    return {
        "run_id": "run_skips",
        "convnextv2_size": "atto",
        "candidate_checkpoint_path": str(
            tmp_path / "candidates" / "candidate.safetensors"
        ),
        "records": records,
        "training_config": training_config,
        "backbone_init": {"source": "random_init", "checkpoint_path": None},
    }


class _ScriptedScaler:
    """Skips the steps it is told to, the way torch's GradScaler does when a
    gradient is not finite: the optimizer is left alone and the scale halves."""

    def __init__(self, skips):
        self.skips = skips  # 1-based numbers of the steps to skip
        self.value = 65536.0
        self.calls = 0
        self.applied = 0
        self._skipping = False

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        self.calls += 1
        self._skipping = self.calls in self.skips
        if not self._skipping:
            optimizer.step()
            self.applied += 1

    def update(self):
        if self._skipping:
            self.value *= 0.5

    def get_scale(self):
        return self.value

    def state_dict(self):
        return {"scale": self.value}

    def load_state_dict(self, state):
        self.value = state["scale"]


def _run(tmp_path, monkeypatch, skips, **config):
    scaler = None
    if skips is not None:
        scaler = _ScriptedScaler(skips)
        monkeypatch.setattr(bt, "make_grad_scaler", lambda *_args: scaler)
    frames: list[dict] = []
    result = asyncio.run(
        run_backbone_training(
            _request(tmp_path, config=config), progress_callback=frames.append
        )
    )
    return result, frames, scaler


def _steps(frames):
    return [f for f in frames if f.get("stage") == "training_step"]


def _epochs(frames):
    return [f for f in frames if f.get("stage") == "training" and "loss" in f]


def _warnings(frames):
    return [f for f in frames if f.get("stage") == "warning"]


# -- counting and reporting ---------------------------------------------------


def test_opening_skips_are_counted_and_reported_but_raise_no_warning(
    tmp_path, monkeypatch
):
    result, frames, scaler = _run(tmp_path, monkeypatch, {1, 2, 3})

    steps = _steps(frames)
    assert [f["skipped_steps"] for f in steps[:4]] == [1, 2, 3, 3]
    assert [f["grad_scale"] for f in steps[:4]] == [32768.0, 16384.0, 8192.0, 8192.0]
    (epoch,) = _epochs(frames)
    assert epoch["skipped_steps"] == 3 and epoch["skipped_steps_total"] == 3
    assert epoch["grad_scale"] == 8192.0
    assert result["skipped_steps"] == 3 and result["grad_scale"] == 8192.0
    # A skipped step is still a step of the loop; it just moved no weight.
    assert epoch["steps"] == scaler.calls == scaler.applied + 3
    assert not _warnings(frames)


def test_an_opening_that_runs_into_the_second_epoch_is_still_the_opening(
    tmp_path, monkeypatch
):
    _result, frames, _scaler = _run(tmp_path, monkeypatch, set(range(1, 10)), epochs=2)

    first, second = _epochs(frames)
    assert first["skipped_steps"] == first["steps"]  # the whole first epoch
    assert first["skipped_steps"] + second["skipped_steps"] == 9
    assert second["skipped_steps_total"] == 9
    assert second["grad_scale"] == 65536.0 / 2**9
    assert not _warnings(frames)


@pytest.mark.parametrize("heartbeat", [1, 0])
def test_steps_skipped_after_the_scaler_settled_raise_one_loud_warning(
    tmp_path, monkeypatch, caplog, heartbeat
):
    """Two steps land, then every step is skipped: the run has stalled. The
    warning comes with the heartbeat when there is one, and at the end of the
    epoch either way."""
    with caplog.at_level("WARNING", logger=bt.__name__):
        _result, frames, _scaler = _run(
            tmp_path, monkeypatch, range(3, 10_000), progress_every_steps=heartbeat
        )

    (epoch,) = _epochs(frames)
    assert epoch["skipped_steps"] == epoch["steps"] - 2
    (warning,) = _warnings(frames)
    assert warning["warning"] == "skipped_steps" and warning["epoch"] == 1
    assert warning["skipped_steps"] >= 3
    assert "skipped" in warning["status_text"]
    assert frames.index(warning) < frames.index(epoch)
    if heartbeat:  # raised at the third late skip, not an epoch later
        assert warning["skipped_steps"] == 3 and warning["steps"] == 5
    assert any("skipped" in record.getMessage() for record in caplog.records)


def test_a_settled_scalers_occasional_skip_is_not_a_warning(tmp_path, monkeypatch):
    # One skip while it probes upwards is how a healthy scaler behaves.
    _result, frames, _scaler = _run(tmp_path, monkeypatch, {1, 5})

    (epoch,) = _epochs(frames)
    assert epoch["skipped_steps"] == 2
    assert not _warnings(frames)


def test_without_a_scaler_there_is_nothing_to_skip(tmp_path, monkeypatch):
    result, frames, _scaler = _run(tmp_path, monkeypatch, None)

    (epoch,) = _epochs(frames)
    assert epoch["skipped_steps"] == 0 and epoch["grad_scale"] is None
    assert _steps(frames)[0]["skipped_steps"] == 0
    assert result["skipped_steps"] == 0 and result["grad_scale"] is None


def test_the_counts_ride_the_backup(tmp_path):
    ctx = SimpleNamespace(device=CPU)
    task = BackboneTask(_request(tmp_path / "a"))
    task.skipped_steps = 16
    task._scaler_settled = True

    state = task.collect_extra_state(
        ctx, rng_epoch_start=None, schedule=None, batch_in_epoch=0
    )
    resumed = BackboneTask(_request(tmp_path / "b"))
    resumed.restore_resume_extra(ctx, {"best": {}, **state})

    assert resumed.skipped_steps == 16 and resumed._scaler_settled is True
    # A backup from before the counts existed is from a run already under way.
    older = BackboneTask(_request(tmp_path / "c"))
    older.restore_resume_extra(ctx, {"best": {}})
    assert older.skipped_steps == 0 and older._scaler_settled is True


# -- is the opening descent worth removing? -----------------------------------


def _fp16_steps(scaler, task, gradient, steps):
    """Drive a real GradScaler the way the trainer does. A gradient of
    *gradient* overflows half precision whenever the scaled value passes its
    largest number, which is what a fresh run's opening steps do."""
    weight = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([weight], lr=0.0)
    task._start_skip_count(scaler)
    pattern = []
    for _ in range(steps):
        before = task.skipped_steps
        optimizer.zero_grad()
        scaler.scale((weight * gradient).sum()).backward()
        if float(weight.grad.abs().max()) > FP16_MAX:
            weight.grad.fill_(float("inf"))
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        task._count_skip(scaler)
        pattern.append(task.skipped_steps > before)
    return pattern


def test_the_default_start_costs_a_few_skipped_steps_once_per_run(tmp_path):
    """Coming down from 65536 takes one skipped step per halving: at most 16
    to reach a scale of 1, about 20 seconds of a run of days. The scale rides
    the backups, so a resumed run does not pay it again."""
    task = BackboneTask(_request(tmp_path / "a"))
    scaler = torch.amp.GradScaler("cpu")

    pattern = _fp16_steps(scaler, task, gradient=30_000.0, steps=40)

    assert pattern == [True] * 15 + [False] * 25  # 65536 -> 2, then every step lands
    assert task.skipped_steps == 15 == math.log2(65536 / scaler.get_scale())
    assert task._epoch_late_skipped == 0 and task._scaler_settled is True

    resumed_task = BackboneTask(_request(tmp_path / "b"))
    resumed = torch.amp.GradScaler("cpu")
    resumed.load_state_dict(scaler.state_dict())
    assert not any(_fp16_steps(resumed, resumed_task, gradient=30_000.0, steps=20))


def test_a_start_that_is_too_low_takes_two_thousand_steps_per_doubling(
    tmp_path, monkeypatch
):
    """Why the start is not lowered to skip the descent. The scaler comes down
    in one step per halving but goes up one doubling per 2000 clean steps. A
    start below where a run settles would train for thousands of steps with its
    small gradients rounded away in half precision, and nothing would show it."""
    task = BackboneTask(_request(tmp_path))
    scaler = torch.amp.GradScaler("cpu", init_scale=1.0)

    pattern = _fp16_steps(scaler, task, gradient=1.0, steps=1999)
    assert not any(pattern) and scaler.get_scale() == 1.0

    _fp16_steps(scaler, task, gradient=1.0, steps=1)
    assert scaler.get_scale() == 2.0

    # So the trainer builds its scaler with torch's own start: no setting.
    built = []
    monkeypatch.setattr(
        torch.amp, "GradScaler", lambda *args, **kwargs: built.append((args, kwargs))
    )
    amp.make_grad_scaler(torch.device("cuda"), torch.float16, True)
    assert built == [(("cuda",), {})]


# -- real GPU (skipped at home; run this on the T4) ----------------------------


@needs_gpu
def test_fp16_skips_are_counted_against_the_scale_on_a_real_gpu(tmp_path):
    frames: list[dict] = []
    request = _request(
        tmp_path,
        # 28 steps: room for the whole descent from 65536 and steps after it.
        config={"device": "cuda", "use_amp": True, "amp_dtype": "float16", "epochs": 4},
    )
    result = asyncio.run(
        run_backbone_training(request, progress_callback=frames.append)
    )

    last = _epochs(frames)[-1]
    total = sum(f["skipped_steps"] for f in _epochs(frames))
    assert last["skipped_steps_total"] == total == result["skipped_steps"]
    assert 0 <= total <= 20
    # Far too few steps for the scale to have grown: every halving was a skip.
    assert last["grad_scale"] == 65536.0 / 2**total == result["grad_scale"]
    assert total < last["steps"]  # the run got past its opening and trained
    assert all(math.isfinite(f["loss"]) for f in _epochs(frames))
