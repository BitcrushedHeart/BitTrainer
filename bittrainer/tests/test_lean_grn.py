"""A GRN layer that stays in the precision the run trains in (Bitcrush ISSUE-1655).

Under autocast timm's GRN takes its norm in fp32, and the product, the sum and
the result on the widest tensor of every block follow it there. Measured on the
Azure T4 with ConvNeXt V2 Base in fp16: 55 -> 77 images/s and 15.5 -> 11.0 GB at
batch 64, 224px, once only the small per-channel factor is computed in fp32.

In full precision the lean layer is the stock layer, bit for bit. The GPU tests
at the bottom skip without CUDA and are the ones to run on the T4.
"""

from __future__ import annotations

import copy

import pytest
import torch
from PIL import Image
from timm.layers.grn import GlobalResponseNorm

from bittrainer.generic.tasks.backbone_task import BackboneTask
from bittrainer.lean_grn import LeanGlobalResponseNorm, use_lean_grn
from bittrainer.model import create_model

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real GPU (run on the T4)"
)


def _pair(channels_last=True, dim=24):
    """The stock layer and a lean copy of it, with weights that do something
    (GRN starts at zero, where it passes its input through)."""
    torch.manual_seed(0)
    stock = GlobalResponseNorm(dim, channels_last=channels_last)
    torch.nn.init.normal_(stock.weight, std=0.5)
    torch.nn.init.normal_(stock.bias, std=0.5)
    lean = copy.deepcopy(stock)
    lean.__class__ = LeanGlobalResponseNorm
    return stock, lean


def _input(channels_last=True, dim=24):
    torch.manual_seed(1)
    shape = (3, 7, 5, dim) if channels_last else (3, dim, 7, 5)
    return torch.randn(shape) * 3


def _cosine(a, b):
    return float(
        torch.nn.functional.cosine_similarity(
            a.flatten().double(), b.flatten().double(), dim=0
        )
    )


# -- the layer ----------------------------------------------------------------


@pytest.mark.parametrize("channels_last", [True, False])
def test_full_precision_is_the_stock_layer_bit_for_bit(channels_last):
    stock, lean = _pair(channels_last)
    x = _input(channels_last)

    assert torch.equal(lean(x), stock(x))


@pytest.mark.parametrize("channels_last", [True, False])
def test_the_lean_form_computes_what_the_stock_layer_computes(channels_last):
    """Input in another precision than the weights takes the lean form. With a
    float64 input that form can be held against the stock result tightly."""
    stock, lean = _pair(channels_last)
    x = _input(channels_last)

    out = lean(x.double())

    assert out.dtype == torch.float64  # the input's precision, not the weights'
    assert torch.allclose(out.float(), stock(x), rtol=1e-5, atol=1e-5)


def test_a_reduced_precision_input_stays_reduced():
    stock, lean = _pair()
    x = _input()
    with torch.no_grad():
        reference = stock(x)
        out = lean(x.to(torch.bfloat16))

    assert out.dtype == torch.bfloat16
    # bf16 keeps 8 bits: the result is the stock one to within that rounding.
    gap = (out.float() - reference).abs().max()
    assert float(gap) <= 0.03 * float(reference.abs().max())
    assert _cosine(out.float(), reference) > 0.9995


def test_gradients_follow_the_stock_layer():
    stock, lean = _pair()
    x = _input()
    full = x.clone().requires_grad_(True)
    stock(full).square().mean().backward()

    reduced = x.clone().requires_grad_(True)
    lean(reduced.to(torch.bfloat16)).float().square().mean().backward()

    assert _cosine(reduced.grad, full.grad) > 0.999
    assert _cosine(lean.weight.grad, stock.weight.grad) > 0.999
    assert _cosine(lean.bias.grad, stock.bias.grad) > 0.999


def test_a_layer_at_zero_still_passes_its_input_through():
    """A trunk converted from V1 starts with GRN at zero; it must stay exact."""
    lean = LeanGlobalResponseNorm(24)
    x = _input().to(torch.bfloat16)

    assert torch.equal(lean(x), x)


# -- switching a model --------------------------------------------------------


def _atto():
    torch.manual_seed(0)
    model = create_model(
        model_size="atto", pretrained=False, num_classes=0, drop_path_rate=0.0
    ).eval()
    for module in model.modules():
        if isinstance(module, GlobalResponseNorm):
            torch.nn.init.normal_(module.weight, std=0.5)
            torch.nn.init.normal_(module.bias, std=0.5)
    return model


def _grn_layers(model):
    return [m for m in model.modules() if isinstance(m, GlobalResponseNorm)]


def test_a_model_switches_without_changing_weights_or_outputs():
    model = _atto()
    images = torch.randn(2, 3, 64, 64)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        before = model(images)

    switched = use_lean_grn(model)

    assert switched == len(_grn_layers(model)) == 12  # one per block of an atto
    assert all(type(layer) is LeanGlobalResponseNorm for layer in _grn_layers(model))
    after = model.state_dict()
    assert list(after) == list(state)
    assert all(torch.equal(after[key], state[key]) for key in state)
    with torch.no_grad():
        assert torch.equal(model(images), before)


def test_switching_twice_changes_nothing_and_copies_stay_lean():
    model = _atto()
    use_lean_grn(model)

    assert use_lean_grn(model) == 0
    clone = copy.deepcopy(model)  # how the EMA takes its copy
    assert all(type(layer) is LeanGlobalResponseNorm for layer in _grn_layers(clone))


# -- the trainer --------------------------------------------------------------


def _request(tmp_path, config=None):
    root = tmp_path / "imgs"
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(8):
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
        "image_size": 64,
        "batch_size": 4,
        "epochs": 1,
        "max_steps": 1,
        "validation_split": 0.25,
        "device": "cpu",
        "use_amp": False,
    }
    training_config.update(config or {})
    return {
        "run_id": "run_grn",
        "convnextv2_size": "atto",
        "candidate_checkpoint_path": str(tmp_path / "candidates" / "c.safetensors"),
        "records": records,
        "training_config": training_config,
        "backbone_init": {"source": "random_init", "checkpoint_path": None},
    }


def _built(tmp_path, config=None):
    frames: list[dict] = []
    task = BackboneTask(_request(tmp_path, config))
    ctx = task.make_context(frames.append, None, None, None)
    task.prepare_data(ctx)
    model = task.create_model(ctx, None)
    preparing = next(f for f in frames if "train_samples" in f)
    return model, preparing


def test_the_backbone_trainer_uses_the_lean_layer_and_says_so(tmp_path):
    model, preparing = _built(tmp_path)

    layers = _grn_layers(model.backbone)
    assert layers
    assert all(type(layer) is LeanGlobalResponseNorm for layer in layers)
    assert preparing["lean_grn"] is True


def test_the_stock_layer_can_be_asked_for(tmp_path):
    model, preparing = _built(tmp_path, {"lean_grn": False})

    assert all(
        type(layer) is GlobalResponseNorm for layer in _grn_layers(model.backbone)
    )
    assert preparing["lean_grn"] is False


# -- on a real card ------------------------------------------------------------


@needs_gpu
def test_under_autocast_the_stock_layer_leaves_half_precision_and_the_lean_one_stays():
    stock, lean = _pair(dim=64)
    stock, lean = stock.cuda(), lean.cuda()
    x = (_input(dim=64) * 4).cuda()
    with torch.no_grad():
        reference = stock(x)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        low = x.half()
        from_stock, from_lean = stock(low), lean(low)

    assert from_stock.dtype == torch.float32  # what costs the memory and the time
    assert from_lean.dtype == torch.float16
    scale = float(reference.abs().max())
    assert float((from_lean.float() - reference).abs().max()) <= 0.005 * scale
    assert _cosine(from_lean.float(), reference) > 0.99999


@needs_gpu
def test_a_training_step_in_half_precision_gives_the_stock_gradients():
    grads = {}
    for name in ("stock", "lean"):
        model = _atto().cuda().train()
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

    assert grads["stock"].shape == grads["lean"].shape
    assert _cosine(grads["lean"], grads["stock"]) > 0.999
