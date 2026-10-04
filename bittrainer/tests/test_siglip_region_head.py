"""Crop-assist region head on SigLIP patch tokens (Bitcrush ISSUE-1106, Phase A2).

The region head (CenterNet-style, stride 16) was built on ConvNeXt's stage-3 and
stage-4 maps. On SigLIP it reads two transformer blocks' patch tokens instead
(the middle block and the last one), reshaped to NCHW at the patch stride of 16,
so ``RegionHead`` itself, the targets, the loss and the decode are unchanged.
``forward_with_regions`` keeps its signature and dispatches to the model, so
Engine's crop paths call it the same way for both families; the logits stay
bit-identical to ``model(x)``.

Crop-assisted SigLIP groups train in GroupTask: with ``finetune_blocks=0`` the
trunk stays frozen and the class + region heads learn in the loop.

CPU-only, tiny random trunk through the ``_build_trunk`` seam: no download.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

import bittrainer.group_trainer as gt
from bittrainer.generic.optimizer import make_optimizer
from bittrainer.model import create_model, load_checkpoint, prepare_for_finetune
from bittrainer.regions import (
    REGION_STRIDE,
    RegionTargets,
    decode_regions,
    forward_with_regions,
    has_region_head,
    install_region_head,
    region_head_ckpt_meta,
    region_loss,
)
from bittrainer.tests.siglip_fixtures import TINY_DEPTH, TINY_DIM, install_tiny_trunk


@pytest.fixture
def tiny(monkeypatch):
    install_tiny_trunk(monkeypatch)


def _model(num_types: int | None = 1):
    model = create_model(model_size="siglip2_base", pretrained=False, num_classes=3)
    if num_types:
        install_region_head(model, num_types)
    return model.eval()


def test_region_head_reads_two_token_maps_of_trunk_width(tiny):
    model = _model(num_types=2)
    assert has_region_head(model)
    assert model.region_head.in_channels == (TINY_DIM, TINY_DIM)
    meta = region_head_ckpt_meta(model, ["Face", "Hand"])["region_head"]
    assert meta["in_channels"] == [TINY_DIM, TINY_DIM]
    assert meta["stride"] == REGION_STRIDE == 16


def test_install_accepts_explicit_in_channels(tiny):
    model = _model(num_types=None)
    head = install_region_head(model, 1, in_channels=(TINY_DIM, TINY_DIM))
    assert head.in_channels == (TINY_DIM, TINY_DIM)


def test_region_blocks_are_the_middle_and_last(tiny):
    model = _model()
    assert model.region_block_indices() == (TINY_DEPTH // 2 - 1, TINY_DEPTH - 1)


def test_region_feature_maps_are_nchw_at_stride_16(tiny):
    model = _model()
    with torch.no_grad():
        mid, last = model.region_feature_maps(torch.randn(2, 3, 96, 64))
    assert mid.shape == last.shape == (2, TINY_DIM, 6, 4)


def test_forward_with_regions_keeps_the_logits_bit_identical(tiny):
    model = _model(num_types=3)
    x = torch.randn(2, 3, 96, 64)
    with torch.no_grad():
        plain = model(x)
        logits, out = forward_with_regions(model, x)
    assert torch.equal(plain, logits)
    assert out.heatmap.shape == (2, 3, 6, 4)
    assert out.size.shape == out.offset.shape == (2, 2, 6, 4)


def test_untrained_heatmap_starts_near_empty(tiny):
    model = _model()
    with torch.no_grad():
        _, out = forward_with_regions(model, torch.randn(1, 3, 64, 64))
    assert torch.sigmoid(out.heatmap).max().item() < 0.5
    assert decode_regions(out, score_threshold=0.5) == [[]]


def test_forward_without_a_head_raises(tiny):
    with pytest.raises(RuntimeError, match="region head"):
        forward_with_regions(_model(num_types=None), torch.randn(1, 3, 64, 64))


def test_checkpoint_round_trip_restores_the_head(tiny, tmp_path):
    model = _model(num_types=2)
    with torch.no_grad():
        model.region_head.heatmap.bias.fill_(-1.25)
    path = tmp_path / "best.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_size": "siglip2_base",
            "num_classes": 3,
            **region_head_ckpt_meta(model, ["Face", "Hand"]),
        },
        path,
    )
    loaded = load_checkpoint(str(path)).eval()
    assert has_region_head(loaded) and loaded.region_head.in_channels == (
        TINY_DIM,
        TINY_DIM,
    )
    x = torch.randn(1, 3, 64, 48)
    with torch.no_grad():
        want_logits, want = forward_with_regions(model, x)
        got_logits, got = forward_with_regions(loaded, x)
    assert torch.equal(want_logits, got_logits)
    assert torch.equal(want.heatmap, got.heatmap)


def test_frozen_trunk_trains_only_the_heads(tiny):
    model = _model()
    prepare_for_finetune(model, 0)
    assert {
        n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad
    } == {
        "head",
        "region_head",
    }
    model.train()
    targets = RegionTargets(
        boxes=torch.tensor([[0.2, 0.2, 0.6, 0.7]]),
        batch_index=torch.tensor([0]),
        types=torch.tensor([0]),
        supervised=torch.tensor([[True], [False]]),
        cls_mask=torch.tensor([True, True]),
    )
    logits, out = forward_with_regions(model, torch.randn(2, 3, 64, 64))
    loss = torch.nn.functional.cross_entropy(
        logits, torch.tensor([0, 1])
    ) + region_loss(out, targets)
    loss.backward()
    assert model.region_head.heatmap.weight.grad.abs().sum() > 0
    assert model.head.fc.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.trunk.parameters())


def test_train_one_epoch_on_a_crop_assisted_frozen_siglip(tiny):
    """The real loop, one region batch: heads move, the frozen trunk does not."""
    model = _model()
    prepare_for_finetune(model, 0)
    trunk_before = {k: v.clone() for k, v in model.trunk.state_dict().items()}
    head_before = model.region_head.heatmap.weight.clone()
    torch.manual_seed(1)
    images = torch.randint(0, 255, (3, 3, 64, 64), dtype=torch.uint8)
    labels = torch.tensor([0, 1, -100])
    targets = RegionTargets(
        boxes=torch.tensor([[0.2, 0.2, 0.6, 0.7], [0.1, 0.5, 0.5, 0.9]]),
        batch_index=torch.tensor([0, 2]),
        types=torch.tensor([0, 0]),
        supervised=torch.tensor([[True], [False], [True]]),
        cls_mask=labels >= 0,
    )
    config = gt.GroupTrainConfig(
        group_folder=".",
        num_classes=3,
        class_names=["a", "b", "c"],
        backbone_variant="siglip2_base",
        finetune_blocks=0,
        device="cpu",
        dtype="float32",
        channels_last=False,
        use_compile=False,
        randaugment_n=0,
        random_erasing_p=0.0,
        fused_back_pass=False,
    )
    region_losses: list[float] = []
    loss, _ = gt._train_one_epoch(
        model,
        [(images, labels, targets)],
        make_optimizer(model, llrd=True, wd_exclusions=True),
        config,
        torch.device("cpu"),
        torch.float32,
        region_loss_sink=region_losses.append,
    )
    assert loss == loss and region_losses and region_losses[0] > 0
    assert not torch.equal(model.region_head.heatmap.weight, head_before)
    for key, value in model.trunk.state_dict().items():
        assert torch.equal(value, trunk_before[key]), key


# ---------------------------------------------------------------------------
# GroupTask wiring: a crop-assisted SigLIP group gets its region head
# ---------------------------------------------------------------------------


def _png(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)).save(path)
    return path


def _crop_assisted_config(tmp_path, **kw):
    root = tmp_path / "g"
    a = _png(root / "a" / "train" / "a0.png", 1)
    _png(root / "b" / "train" / "b0.png", 2)
    _png(root / "a" / "val" / "a1.png", 3)
    _png(root / "b" / "val" / "b1.png", 4)
    manifest = root / ".region_crops" / "regions_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "crop_pad": 0.15,
                "region_types": ["Face"],
                "crops": [],
                "images": [
                    {
                        "source": str(a),
                        "paths": [str(a)],
                        "split": "train",
                        "boxes": [[0.2, 0.2, 0.8, 0.8, 0]],
                        "rejects": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    base = dict(
        group_folder=str(root),
        num_classes=2,
        class_names=["a", "b"],
        backbone_variant="siglip2_base",
        finetune_blocks=0,
        device="cpu",
        dtype="float32",
        use_cache=False,
        checkpoint_dir=str(tmp_path / "ckpt"),
        backbone_init={"source": "random_init", "checkpoint_path": None},
        region_manifest=str(manifest),
    )
    base.update(kw)
    return gt.GroupTrainConfig(**base)


def test_fresh_crop_assisted_siglip_model_gets_a_region_head(tiny, tmp_path):
    config = _crop_assisted_config(tmp_path)
    assert gt.validate_backbone_config(
        config, mode="full"
    )  # coerced speedups, CPU allowed
    (tmp_path / "ckpt").mkdir()
    model = gt._create_or_warmstart_model(
        config,
        device=torch.device("cpu"),
        dtype=torch.float32,
        head_hidden_size=None,
        checkpoint_dir=tmp_path / "ckpt",
    )
    assert has_region_head(model)
    assert model.region_head.in_channels == (TINY_DIM, TINY_DIM)
