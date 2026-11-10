"""Automatic batch size for the backbone trainer (Bitcrush ISSUE-1655).

``--batch-size auto`` chose 8 for ConvNeXt V2 Base at 224px on a 16 GB T4 (2.9
of 16 GB used), and 8 again with a distillation teacher loaded. The cause: the
shared probe calls ``model(inputs)``, the backbone's model is a wrapper with no
``forward``, the ``NotImplementedError`` that raises is a ``RuntimeError``, and
the probe reads every ``RuntimeError`` as "out of memory". No batch size ever
"fit", so it returned its fallback of 8.

On a GPU the backbone now sizes itself with its own training step (precision,
lean GRN, loss and teacher included): batches double for as long as a straight
line through the measurements says the next one fits, then the largest batch
inside the budget is solved for. The tail is sized on its own at its own
resolution, and the preflight then runs a real step at whatever was chosen.

CPU-only unless a test says otherwise: the card is played by a cost function
and the trunk by one small convolution (``backbone_stub``), so nothing is
trained. The last test measures a real ``atto`` on a GPU and skips without one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import bittrainer.backbone_trainer as bb
import bittrainer.generic.tasks.backbone_task as bt
from bittrainer.autobatch import profile_vram_batch_size
from bittrainer.generic.tasks.backbone_task import BackboneRunRefused, BackboneTask
from bittrainer.tests.backbone_stub import use_stub_trunk

GB = 2**30
CUDA = torch.device("cuda")

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (run on the T4)"
)


@pytest.fixture(autouse=True)
def _trunk(request, monkeypatch):
    """Everything but the real-GPU test runs on the stand-in trunk."""
    if "real_gpu" not in request.node.name:
        use_stub_trunk(monkeypatch)


# Measured on the T4, Base with the lean GRN in fp16. At 224px the trainer
# peaks at 11.4 GB at batch 64, 1.1 GB of that the optimizer state and the EMA.
# At 384px the model alone needs 8.3 / 11.9 / 15.4 GB at batch 16 / 24 / 32.
_T4_ROOM = 15.0 * GB
_T4_LATER = 1.14 * GB
_T4_BUDGET = 0.85 * _T4_ROOM - _T4_LATER


def _at_224(batch):
    return (0.36 + 0.155 * batch) * GB


def _at_384(batch):
    return (1.1 + 0.45 * batch) * GB


def _card(cost, *, runs_out_at=None):
    tried: list[int] = []

    def _measure(batch):
        tried.append(batch)
        if runs_out_at is not None and batch >= runs_out_at:
            return None
        return cost(batch)

    return _measure, tried


# -- why it chose 8 -----------------------------------------------------------


def test_why_auto_chose_eight(monkeypatch):
    """The cause, pinned. The shared probe is left as it is for the other
    trainers, whose models do have a ``forward``."""
    heads = bb._MultiTaskHeads(4, SimpleNamespace(concepts=["a"], groups={}))
    model = bb._BackboneWithHeads(torch.nn.Identity(), heads)
    with pytest.raises(NotImplementedError):
        model(torch.zeros(1, 4))
    assert issubclass(NotImplementedError, RuntimeError)

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (15 * GB, 15 * GB))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)

    probe = profile_vram_batch_size(
        model, lambda batch: (torch.zeros(batch, 4),), CUDA, dtype=torch.float32
    )

    # "Out of memory" at a batch of one, on an empty 15 GB card.
    assert probe["max_fitted_rung"] is None and probe["vram_limit"] == 8


# -- the search ---------------------------------------------------------------


def test_base_at_224px_on_a_t4_lands_near_what_fits():
    measure, tried = _card(_at_224)

    found = bt._largest_fitting_batch(measure, _T4_BUDGET, 200_000)

    # Batch 64 was run for real; 128 was never tried, it would not have fitted.
    assert tried == [1, 2, 4, 8, 16, 32, 64]
    assert found["max_fitted_rung"] == 64
    assert 64 <= found["batch_size"] <= 76
    # The largest batch whose whole run (step, optimizer state, EMA) stays
    # inside 85% of the card.
    assert _at_224(found["batch_size"]) <= _T4_BUDGET < _at_224(found["batch_size"] + 1)


def test_the_384px_tail_gets_a_size_of_its_own():
    measure, tried = _card(_at_384)

    found = bt._largest_fitting_batch(measure, _T4_BUDGET, 200_000)

    assert tried == [1, 2, 4, 8, 16]  # 32 needs 15.4 GB: never attempted
    assert 16 <= found["batch_size"] <= 24
    assert (
        found["batch_size"]
        < bt._largest_fitting_batch(_card(_at_224)[0], _T4_BUDGET, 200_000)[
            "batch_size"
        ]
    )


def test_a_step_that_runs_out_of_memory_ends_the_search_at_the_last_that_fitted():
    # The line said 16 would fit; the card disagreed.
    measure, tried = _card(_at_224, runs_out_at=16)

    found = bt._largest_fitting_batch(measure, _T4_BUDGET, 200_000)

    assert tried == [1, 2, 4, 8, 16]
    assert found["batch_size"] == 8 and found["max_fitted_rung"] == 8


def test_no_batch_is_larger_than_the_training_set():
    measure, tried = _card(_at_224)

    found = bt._largest_fitting_batch(measure, _T4_BUDGET, 20)

    assert tried == [1, 2, 4, 8, 16]
    assert found["batch_size"] == 20


def test_when_not_even_one_image_fits_there_is_no_answer():
    measure, tried = _card(_at_384, runs_out_at=1)

    assert bt._largest_fitting_batch(measure, _T4_BUDGET, 200_000) is None
    assert tried == [1]


def test_a_single_image_over_budget_is_still_one_image():
    # It ran, so it is the answer; whether it is safe is the preflight's call.
    found = bt._largest_fitting_batch(_card(_at_384)[0], 1.0 * GB, 200_000)
    assert found["batch_size"] == 1


# -- the task -----------------------------------------------------------------


def _request(tmp_path, **config):
    root = tmp_path / "imgs"
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(12):
        path = root / f"img{i}.png"
        Image.new("RGB", (72, 64), color=((i * 37) % 255, 90, 160)).save(path)
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
        "image_size": 224,
        "finetune_image_size": 384,
        "finetune_epochs": 1,
        "epochs": 3,
        "validation_split": 0.25,
        "device": "cpu",
        "use_amp": False,
        "preflight": False,
    }
    training_config.update(config)
    return {
        "run_id": "run_auto",
        "convnextv2_size": "atto",
        "candidate_checkpoint_path": str(tmp_path / "candidates" / "c.safetensors"),
        "records": records,
        "training_config": training_config,
        "backbone_init": {"source": "random_init", "checkpoint_path": None},
    }


def _sized(tmp_path, monkeypatch, cost=None, **config):
    """Build a task and let it choose its batch sizes on a pretend T4, where a
    training step at (image size, batch) peaks at ``cost(image size, batch)``."""
    frames: list[dict] = []
    tried: list[tuple[int, int]] = []
    task = BackboneTask(_request(tmp_path, **config))
    ctx = task.make_context(frames.append, None, None, None)
    model = task.create_model(ctx, None)
    task.train_samples = task.train_samples * 1000  # a pool, not nine images

    def _dry_step(self, ctx, model, image_size, batch_size):
        tried.append((image_size, batch_size))
        return (cost or _t4)(image_size, batch_size)

    monkeypatch.setattr(bt, "_vram_room", lambda device: _T4_ROOM)
    monkeypatch.setattr(BackboneTask, "_dry_step", _dry_step)
    returned = task.resolve_batch_size(ctx, model, None)
    assert returned == task.batch_size
    return task, tried, [f for f in frames if f.get("type") == "autobatch"]


def _t4(image_size, batch):
    return _at_224(batch) if image_size == 224 else _at_384(batch)


def test_auto_sizes_each_phase_with_real_steps_at_its_own_resolution(
    tmp_path, monkeypatch
):
    task, tried, frames = _sized(tmp_path, monkeypatch, batch_size="auto")

    assert [batch for size, batch in tried if size == 224] == [1, 2, 4, 8, 16, 32, 64]
    assert [batch for size, batch in tried if size == 384] == [1, 2, 4, 8, 16]
    assert 64 <= task.batch_size <= 80
    assert 16 <= task.finetune_batch_size <= 26
    assert [(f["phase"], f["image_size"], f["batch_size"]) for f in frames] == [
        ("main", 224, task.batch_size),
        ("tail", 384, task.finetune_batch_size),
    ]
    # Sized for the whole run: the optimizer's state and the EMA are still to
    # come when the batch is chosen, and they stay for good.
    later = (bt._OPT_STATE_MULT + 1) * sum(
        p.numel() * p.element_size() for p in task.model.parameters()
    )
    for size, batch in ((224, task.batch_size), (384, task.finetune_batch_size)):
        assert (
            _t4(size, batch) + later <= 0.85 * _T4_ROOM < _t4(size, batch + 1) + later
        )


def test_explicit_batch_sizes_are_used_as_given(tmp_path, monkeypatch):
    task, tried, frames = _sized(
        tmp_path, monkeypatch, batch_size=64, finetune_batch_size=16
    )

    assert (task.batch_size, task.finetune_batch_size) == (64, 16)
    assert tried == [] and frames == []


def test_the_tail_alone_can_be_automatic(tmp_path, monkeypatch):
    task, tried, frames = _sized(
        tmp_path, monkeypatch, batch_size=64, finetune_batch_size="auto"
    )

    assert task.batch_size == 64
    assert {size for size, _batch in tried} == {384}
    assert 16 <= task.finetune_batch_size <= 26
    assert [f["phase"] for f in frames] == ["tail"]


def test_an_automatic_main_size_never_carries_over_to_a_set_tail(tmp_path, monkeypatch):
    task, tried, _frames = _sized(
        tmp_path, monkeypatch, batch_size="auto", finetune_batch_size=12
    )

    assert task.finetune_batch_size == 12
    assert {size for size, _batch in tried} == {224}


def test_without_a_tail_only_the_main_size_is_chosen(tmp_path, monkeypatch):
    task, tried, frames = _sized(
        tmp_path, monkeypatch, batch_size="auto", finetune_epochs=0
    )

    assert {size for size, _batch in tried} == {224}
    assert task.finetune_batch_size == task.batch_size
    assert [f["phase"] for f in frames] == ["main"]


def test_a_card_that_cannot_hold_one_image_refuses_the_run(tmp_path, monkeypatch):
    def _out_of_memory(image_size, batch):
        if image_size == 384:
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB.")
        return _at_224(batch)

    with pytest.raises(BackboneRunRefused, match="384px"):
        _sized(tmp_path, monkeypatch, _out_of_memory, batch_size="auto")


def test_an_error_that_is_not_about_memory_is_never_read_as_does_not_fit(
    tmp_path, monkeypatch
):
    """The mistake behind the 8: a broken step must stop the run, not size it."""

    def _broken(image_size, batch):
        raise NotImplementedError("missing the required forward function")

    with pytest.raises(NotImplementedError):
        _sized(tmp_path, monkeypatch, _broken, batch_size="auto")


def test_off_a_gpu_auto_is_the_shared_probes_fixed_answer(tmp_path):
    frames: list[dict] = []
    task = BackboneTask(_request(tmp_path, batch_size="auto"))
    ctx = task.make_context(frames.append, None, None, None)
    model = task.create_model(ctx, None)

    task.resolve_batch_size(ctx, model, None)

    assert task.batch_size == task.finetune_batch_size == len(task.train_samples)
    assert [f["phase"] for f in frames if f.get("type") == "autobatch"] == [
        "main",
        "tail",
    ]


# -- real GPU (skipped at home; run this on the T4) ----------------------------


@needs_gpu
def test_auto_measures_real_fp16_steps_on_a_real_gpu(tmp_path):
    frames: list[dict] = []
    task = BackboneTask(
        _request(
            tmp_path,
            batch_size="auto",
            image_size=48,
            finetune_image_size=64,
            device="cuda",
            use_amp=True,
            amp_dtype="float16",
        )
    )
    ctx = task.make_context(frames.append, None, None, None)
    model = task.create_model(ctx, None)
    weights = {k: v.clone() for k, v in model.state_dict().items()}

    task.resolve_batch_size(ctx, model, None)

    # Nine training images: batches of 1, 2, 4 and 8 were run for real, the
    # cost grew with the batch, and an atto at this size fits any card whole.
    main, tail = [f for f in frames if f.get("type") == "autobatch"]
    assert main["max_fitted_rung"] == 8 and tail["max_fitted_rung"] == 8
    assert main["fit_slope"] > 0 and tail["fit_slope"] > main["fit_slope"]
    assert task.batch_size == task.finetune_batch_size == len(task.train_samples)
    for key, value in model.state_dict().items():
        assert torch.equal(value, weights[key]), key
