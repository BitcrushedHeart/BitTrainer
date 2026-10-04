"""SigLIP fine-tune wiring: trainable set, LLRD, optimizer identity, autobatch
(Bitcrush ISSUE-1106).

A SigLIP "fine-tune" trains the head plus the LAST ``finetune_blocks`` transformer
blocks (and the norm / MAP pool after them); ``finetune_blocks=0`` keeps the whole
trunk frozen. ConvNeXt keeps its full unfreeze. The optimiser, the resume
fingerprint and the batch-size probe must all see the same trainable set — and
none of it may change for ConvNeXt.

CPU-only, tiny random trunks through the ``_build_trunk`` seam: no download, no
training loop.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import bittrainer.group_trainer as gt
from bittrainer.generic.optimizer import make_optimizer, optimizer_identity
from bittrainer.generic.tasks.group_task import GroupTask
from bittrainer.model import build_llrd_param_groups, create_model, prepare_for_finetune
from bittrainer.siglip_model import trainable_param_count
from bittrainer.tests.siglip_fixtures import TINY_DEPTH, install_tiny_trunk


@pytest.fixture
def tiny(monkeypatch):
    install_tiny_trunk(monkeypatch)


def _siglip(num_classes: int = 3):
    return create_model(
        model_size="siglip2_base", pretrained=False, num_classes=num_classes
    )


def _trainable(model) -> set[str]:
    return {n for n, p in model.named_parameters() if p.requires_grad}


def _config(tmp_path, **kw):
    base = dict(
        group_folder=str(tmp_path),
        num_classes=3,
        class_names=["a", "b", "c"],
        backbone_variant="siglip2_base",
        device="cpu",
        dtype="float32",
        checkpoint_dir=str(tmp_path / "ckpt"),
        backbone_init={"source": "random_init", "checkpoint_path": None},
    )
    base.update(kw)
    return gt.GroupTrainConfig(**base)


# ---------------------------------------------------------------------------
# prepare_for_finetune
# ---------------------------------------------------------------------------


def test_finetune_blocks_default_is_four():
    assert (
        gt.GroupTrainConfig(
            group_folder="x", num_classes=2, class_names=["a", "b"]
        ).finetune_blocks
        == 4
    )


def test_siglip_unfreezes_only_the_last_blocks_and_what_follows(tiny):
    model = _siglip()
    prepare_for_finetune(model, 2)
    names = _trainable(model)
    assert any(n.startswith("trunk.blocks.2.") for n in names)
    assert any(n.startswith("trunk.blocks.3.") for n in names)
    assert not any(
        n.startswith(("trunk.blocks.0.", "trunk.blocks.1.", "trunk.embeds."))
        for n in names
    )
    assert any(n.startswith("trunk.attn_pool.") for n in names)
    assert {"trunk.norm.weight", "head.fc.weight", "head.fc.bias"} <= names


def test_zero_blocks_freezes_the_whole_trunk(tiny):
    model = _siglip()
    prepare_for_finetune(model, 0)
    assert _trainable(model) == {"head.fc.weight", "head.fc.bias"}
    assert model.trunk_frozen()


def test_frozen_trunk_stays_in_eval_mode_while_the_head_trains(tiny):
    model = _siglip()
    prepare_for_finetune(model, 0)
    model.train()
    assert model.head.training and not model.trunk.training
    prepare_for_finetune(model, 1)
    model.train()
    assert model.trunk.training


def test_more_blocks_than_the_depth_unfreezes_the_whole_stack(tiny):
    model = _siglip()
    prepare_for_finetune(model, 99)
    names = _trainable(model)
    for i in range(TINY_DEPTH):
        assert any(n.startswith(f"trunk.blocks.{i}.") for n in names)


def test_convnext_prepare_is_the_full_unfreeze():
    model = create_model(model_size="atto", pretrained=False, num_classes=3)
    for p in model.parameters():
        p.requires_grad_(False)
    prepare_for_finetune(model, 0)
    assert all(p.requires_grad for p in model.parameters())


# ---------------------------------------------------------------------------
# LLRD + optimizer
# ---------------------------------------------------------------------------


def test_siglip_llrd_buckets_trainable_blocks_by_depth(tiny):
    model = _siglip()
    prepare_for_finetune(model, 2)
    groups = build_llrd_param_groups(model, 0.5)
    by_name = {g["name"]: g for g in groups}
    assert set(by_name) == {"siglip.block.2", "siglip.block.3", "head"}
    assert by_name["siglip.block.3"]["lr"] == pytest.approx(0.5)
    assert by_name["siglip.block.2"]["lr"] == pytest.approx(0.25)
    assert by_name["head"]["lr"] == pytest.approx(1.0)
    grouped = [id(p) for g in groups for p in g["params"]]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(p) for p in model.parameters() if p.requires_grad}


def test_full_stack_llrd_puts_the_embedding_deepest(tiny):
    model = _siglip()
    prepare_for_finetune(model, TINY_DEPTH)
    by_name = {g["name"]: g for g in build_llrd_param_groups(model, 0.5)}
    assert by_name["siglip.embeds"]["lr"] == pytest.approx(0.5 ** (TINY_DEPTH + 1))
    assert by_name["siglip.block.0"]["lr"] == pytest.approx(0.5**TINY_DEPTH)


@pytest.mark.parametrize(
    "llrd,wd", [(True, True), (False, True), (False, False), (True, False)]
)
def test_siglip_optimizer_holds_only_trainable_parameters(tiny, llrd, wd):
    model = _siglip()
    prepare_for_finetune(model, 1)
    opt = make_optimizer(model, llrd=llrd, llrd_decay=0.8, wd_exclusions=wd)
    held = {id(p) for g in opt.param_groups for p in g["params"]}
    assert held == {id(p) for p in model.parameters() if p.requires_grad}


def test_convnext_optimizer_still_holds_every_parameter():
    model = create_model(model_size="atto", pretrained=False, num_classes=3)
    for p in model.head.parameters():
        p.requires_grad_(False)
    opt = make_optimizer(model, llrd=True, llrd_decay=0.8, wd_exclusions=True)
    held = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert held == sum(p.numel() for p in model.parameters())


def test_optimizer_identity_suffix_is_opt_in():
    legacy = optimizer_identity(llrd=True, llrd_decay=0.8, wd_exclusions=True)
    assert legacy == "Prodigy_adv+llrd0.8+nd"
    assert (
        optimizer_identity(
            llrd=True, llrd_decay=0.8, wd_exclusions=True, trainable_blocks=None
        )
        == legacy
    )
    assert (
        optimizer_identity(
            llrd=True, llrd_decay=0.8, wd_exclusions=True, trainable_blocks=4
        )
        == "Prodigy_adv+llrd0.8+nd+sb4"
    )
    assert (
        optimizer_identity(
            llrd=False, llrd_decay=0.8, wd_exclusions=False, trainable_blocks=0
        )
        == "Prodigy_adv+sb0"
    )


# ---------------------------------------------------------------------------
# autobatch
# ---------------------------------------------------------------------------


def test_autobatch_budgets_optimizer_state_for_trainable_params_only(tiny):
    from bittrainer.autobatch import determine_batch_size

    model = _siglip()
    prepare_for_finetune(model, 0)
    total = sum(p.numel() for p in model.parameters())
    trainable = trainable_param_count(model)
    cpu = torch.device("cpu")
    legacy = determine_batch_size(model, {(64, 64): 10}, cpu)
    assert legacy["overhead_gb"] == int(2.2 * total * 4) / 1e9
    scoped = determine_batch_size(
        model, {(64, 64): 10}, cpu, trainable_params=trainable
    )
    assert scoped["overhead_gb"] == int(2.2 * trainable * 4) / 1e9
    ema = determine_batch_size(
        model, {(64, 64): 10}, cpu, use_ema=True, trainable_params=trainable
    )
    assert ema["overhead_gb"] == int((2.2 * trainable + total) * 4) / 1e9


# ---------------------------------------------------------------------------
# GroupTask wiring
# ---------------------------------------------------------------------------


def _ctx(tmp_path):
    return SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        cb=lambda _msg: None,
        em=SimpleNamespace(stage=lambda *a, **k: None),
        pause_event=None,
        stop_event=None,
        stop_now_event=None,
        checkpoint_dir=tmp_path,
    )


def test_fingerprint_carries_the_trainable_block_count_for_siglip_only(tmp_path):
    task = GroupTask(_config(tmp_path, finetune_blocks=3))
    ctx = _ctx(tmp_path)
    task.fingerprint_init(ctx)
    assert ctx.fingerprint["optimizer"].endswith("+sb3")

    convnext = GroupTask(_config(tmp_path, backbone_variant="atto"))
    ctx2 = _ctx(tmp_path)
    convnext.fingerprint_init(ctx2)
    assert ctx2.fingerprint["optimizer"] == "Prodigy_adv+llrd0.8+nd"


def test_pre_loop_prepares_the_finetune_set(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "_warmup_head_probe", lambda *a, **k: None)
    task = GroupTask(_config(tmp_path, finetune_blocks=1))
    model = _siglip()
    task.pre_loop(_ctx(tmp_path), model)
    names = _trainable(model)
    assert any(n.startswith("trunk.blocks.3.") for n in names)
    assert not any(n.startswith("trunk.blocks.2.") for n in names)


def test_resume_rebuilds_siglip_and_prepares_the_finetune_set(tiny, tmp_path):
    source = _siglip()
    task = GroupTask(_config(tmp_path, finetune_blocks=0))
    model = task.create_model(_ctx(tmp_path), {"model": source.state_dict()})
    assert model.backbone_family == "siglip2"
    assert _trainable(model) == {"head.fc.weight", "head.fc.bias"}
    for key, value in source.state_dict().items():
        assert torch.equal(model.state_dict()[key], value)


def test_autobatch_gets_the_trainable_count_for_siglip(tiny, tmp_path, monkeypatch):
    import bittrainer.autobatch as autobatch

    seen: dict = {}

    def _fake(model, bucket_counts, device, **kwargs):
        seen.update(kwargs)
        return {"batch_size": 8}

    monkeypatch.setattr(autobatch, "determine_batch_size", _fake)
    config = _config(tmp_path, finetune_blocks=0)
    task = GroupTask(config)
    task.train_ds = SimpleNamespace(oversample_none=config.oversample_none)
    model = _siglip()
    prepare_for_finetune(model, 0)
    assert task.resolve_batch_size(_ctx(tmp_path), model, None) == 8
    assert seen["trainable_params"] == trainable_param_count(model)

    seen.clear()
    convnext_cfg = _config(tmp_path, backbone_variant="atto")
    convnext = GroupTask(convnext_cfg)
    convnext.train_ds = SimpleNamespace(oversample_none=convnext_cfg.oversample_none)
    convnext.resolve_batch_size(
        _ctx(tmp_path), create_model(model_size="atto", pretrained=False), None
    )
    assert "trainable_params" not in seen
