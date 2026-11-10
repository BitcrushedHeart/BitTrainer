"""GRN kept in half precision for the group trainers, on fp16 cards only
(Bitcrush ISSUE-1690).

The backbone trainer keeps ConvNeXt V2's GRN layer in the autocast precision
(``bittrainer.lean_grn``): on the T4 in fp16 that was 40% faster and 29% leaner
for Base. The group full fine-tune and the head-only trainer now do the same,
but only when the run autocasts in fp16 on a GPU. Under bf16 autocast at home
the lean form would round differently, so a bf16 card and a CPU keep timm's
layer and train exactly as before.

The switch keeps every weight, every state-dict name, the fp32 output and the
embedding cache's backbone hash, and a saved model loads into a stock one.

CPU-only, on the micro stand-in trunk (``group_stub``). The test named
``..._real_gpu`` skips without CUDA and is the one to run on the T4.
"""

from __future__ import annotations

import copy

import pytest
import torch
from timm.layers.grn import GlobalResponseNorm

import bittrainer.amp as amp
import bittrainer.generic.tasks.group_task as group_task
from bittrainer.ema import ModelEMA
from bittrainer.generic.tasks.group_task import GroupTask
from bittrainer.generic.tasks.head_only_task import HeadOnlyTask
from bittrainer.lean_grn import LeanGlobalResponseNorm, use_lean_grn
from bittrainer.model import backbone_feature_hash, load_checkpoint
from bittrainer.tests.group_stub import (
    group_config,
    grn_layers,
    micro_convnext,
    stir_grn,
    use_micro_trunk,
)

CPU = torch.device("cpu")

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (run on the T4)"
)


def _task(tmp_path, monkeypatch, cls=GroupTask, *, device="cpu", dtype="bfloat16",
          native=None, **kw):
    """A task whose precision was fitted in make_context, with a context that
    builds its models on the CPU whatever card it was fitted for."""
    use_micro_trunk(monkeypatch)
    if native is not None:
        monkeypatch.setattr(amp, "native_bf16", lambda _device: native)
    monkeypatch.setattr(group_task, "make_grad_scaler", lambda *_a: None)
    task = cls(group_config(tmp_path, tmp_path / "ck", device=device, dtype=dtype, **kw))
    ctx = task.make_context(None, None, None, None)
    ctx.device = CPU
    return task, ctx


def _kinds(model) -> set[type]:
    return {type(layer) for layer in grn_layers(model)}


# -- when -----------------------------------------------------------------------


@pytest.mark.parametrize("cls", [GroupTask, HeadOnlyTask])
@pytest.mark.parametrize(
    ("device", "dtype", "native", "asked", "lean"),
    [
        ("cpu", "bfloat16", None, True, False),
        ("cpu", "float16", None, True, False),  # fp16 on a CPU: still timm's
        ("cpu", "float32", None, True, False),
        ("cuda", "bfloat16", True, True, False),  # an Ampere / Ada card
        ("cuda", "float32", False, True, False),
        ("cuda", "bfloat16", False, True, True),  # the T4
        ("cuda", "bfloat16", False, False, False),  # the T4, stock layer asked for
    ],
)
def test_only_an_fp16_run_on_a_gpu_switches(
    tmp_path, monkeypatch, cls, device, dtype, native, asked, lean
):
    task, _ctx = _task(
        tmp_path, monkeypatch, cls, device=device, dtype=dtype, native=native,
        lean_grn=asked,
    )
    assert task.lean_grn is lean


@pytest.mark.parametrize("cls", [GroupTask, HeadOnlyTask])
def test_a_bf16_card_or_a_cpu_keeps_timms_layer(tmp_path, monkeypatch, cls):
    for name, (device, native) in {"cpu": ("cpu", None), "ampere": ("cuda", True)}.items():
        task, ctx = _task(tmp_path / name, monkeypatch, cls, device=device, native=native)
        model = task.create_model(ctx, None)
        assert _kinds(model) == {GlobalResponseNorm}


# -- what it changes, and what it does not --------------------------------------


def test_the_group_model_switches_without_changing_names_weights_or_output(
    tmp_path, monkeypatch
):
    task, ctx = _task(tmp_path, monkeypatch, device="cuda", native=False)
    model = task.create_model(ctx, None)
    stir_grn(model)

    assert _kinds(model) == {LeanGlobalResponseNorm}
    stock = micro_convnext()
    stock.load_state_dict(model.state_dict())  # strict: the very same names
    assert list(model.state_dict()) == list(stock.state_dict())
    assert backbone_feature_hash(model) == backbone_feature_hash(stock)
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(model.eval()(images), stock.eval()(images))


def test_a_saved_lean_model_loads_into_a_stock_model(tmp_path, monkeypatch):
    task, ctx = _task(tmp_path, monkeypatch, device="cuda", native=False)
    model = task.create_model(ctx, None)
    stir_grn(model, seed=3)
    path = tmp_path / "best.pt"
    torch.save({"state_dict": model.state_dict(), "num_classes": 3}, path)

    loaded = load_checkpoint(str(path), num_classes=3)

    assert _kinds(loaded) == {GlobalResponseNorm}
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(loaded.eval()(images), model.eval()(images))


def test_the_ema_copy_stays_lean_and_a_second_switch_changes_nothing(
    tmp_path, monkeypatch
):
    task, ctx = _task(tmp_path, monkeypatch, device="cuda", native=False)
    model = task.create_model(ctx, None)

    assert _kinds(ModelEMA(model, decay=0.9).module) == {LeanGlobalResponseNorm}
    assert use_lean_grn(model) == 0


def test_a_resumed_model_is_switched_too(tmp_path, monkeypatch):
    task, ctx = _task(tmp_path, monkeypatch, device="cuda", native=False)
    source = micro_convnext()
    stir_grn(source, seed=5)

    model = task.create_model(ctx, {"model": source.state_dict()})

    assert _kinds(model) == {LeanGlobalResponseNorm}
    for key, value in source.state_dict().items():
        assert torch.equal(model.state_dict()[key], value), key


def test_the_head_only_trunk_switches_on_an_fp16_card(tmp_path, monkeypatch):
    task, ctx = _task(tmp_path, monkeypatch, HeadOnlyTask, device="cuda", native=False)
    model = task.create_model(ctx, None)

    assert _kinds(model) == {LeanGlobalResponseNorm}
    stock = copy.deepcopy(model)
    for layer in grn_layers(stock):
        layer.__class__ = GlobalResponseNorm
    assert backbone_feature_hash(model) == backbone_feature_hash(stock)


# -- on a real card --------------------------------------------------------------


@needs_gpu
def test_the_lean_group_trunk_gives_the_stock_gradients_in_fp16_on_a_real_gpu():
    grads = {}
    for name in ("stock", "lean"):
        torch.manual_seed(0)
        model = micro_convnext()
        stir_grn(model)
        model = model.cuda().train()
        if name == "lean":
            use_lean_grn(model)
        images = torch.randn(8, 3, 96, 96, generator=torch.Generator().manual_seed(2))
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(images.cuda()).float().square().mean()
        loss.backward()
        grads[name] = torch.cat(
            [p.grad.flatten() for p in model.parameters() if p.grad is not None]
        )
        assert torch.isfinite(grads[name]).all()

    cosine = torch.nn.functional.cosine_similarity(
        grads["lean"].double(), grads["stock"].double(), dim=0
    )
    assert float(cosine) > 0.999
