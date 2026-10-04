"""SigLIP configuration guards and cross-family warm starts (Bitcrush ISSUE-1106).

``validate_backbone_config`` is the one place a run's backbone settings are
checked before anything is built: SigLIP refuses what it cannot do (sourceless
cloud runs, a fine-tune on a non-CUDA device, a CrushVision local backbone
init) and switches off what does not apply to a ViT (channels_last, regional
torch.compile, greedy soup), saying so. ConvNeXt configs pass through untouched.

A warm start never crosses families: a SigLIP run next to a ConvNeXt incumbent
(or the other way round) starts fresh instead of half-loading nothing.

CPU-only, tiny random trunks through the ``_build_trunk`` seam: no download.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

import bittrainer.group_trainer as gt
from bittrainer.backbone_init import apply_backbone_init
from bittrainer.model import create_model
from bittrainer.siglip_model import SigLIPClassifier
from bittrainer.tests.siglip_fixtures import install_tiny_trunk

_LOCAL = {
    "source": "local_active",
    "checkpoint_path": "C:/nowhere/backbone.safetensors",
}


def _config(tmp_path, **kw):
    base = dict(
        group_folder=str(tmp_path),
        num_classes=3,
        class_names=["a", "b", "c"],
        backbone_variant="siglip2_so400m",
        device="cuda",
        dtype="float32",
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    base.update(kw)
    return gt.GroupTrainConfig(**base)


# ---------------------------------------------------------------------------
# validate_backbone_config
# ---------------------------------------------------------------------------


def test_convnext_config_is_untouched(tmp_path):
    config = _config(
        tmp_path,
        backbone_variant="nano",
        device="cpu",
        backbone_init=_LOCAL,
        sourceless=True,
    )
    before = dataclasses.asdict(config)
    assert gt.validate_backbone_config(config, mode="full") == []
    assert dataclasses.asdict(config) == before


def test_siglip_switches_off_convnext_only_speedups(tmp_path):
    config = _config(tmp_path)
    notes = gt.validate_backbone_config(config, mode="full")
    assert (config.channels_last, config.use_compile, config.use_greedy_soup) == (
        False,
        False,
        False,
    )
    joined = " ".join(notes)
    for word in ("channels_last", "compile", "soup"):
        assert word in joined


def test_siglip_with_speedups_already_off_reports_nothing(tmp_path):
    config = _config(
        tmp_path, channels_last=False, use_compile=False, use_greedy_soup=False
    )
    assert gt.validate_backbone_config(config, mode="full") == []


def test_siglip_refuses_sourceless_runs(tmp_path):
    with pytest.raises(ValueError, match="sourceless"):
        gt.validate_backbone_config(_config(tmp_path, sourceless=True), mode="full")


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_siglip_finetune_needs_cuda(tmp_path, device):
    with pytest.raises(ValueError, match="CUDA"):
        gt.validate_backbone_config(
            _config(tmp_path, device=device, finetune_blocks=2), mode="full"
        )


def test_frozen_trunk_and_head_only_run_anywhere(tmp_path):
    gt.validate_backbone_config(
        _config(tmp_path, device="cpu", finetune_blocks=0), mode="full"
    )
    gt.validate_backbone_config(
        _config(tmp_path, device="cpu", finetune_blocks=4), mode="head_only"
    )


def test_siglip_refuses_negative_finetune_blocks(tmp_path):
    with pytest.raises(ValueError, match="finetune_blocks"):
        gt.validate_backbone_config(_config(tmp_path, finetune_blocks=-1), mode="full")


def test_siglip_refuses_a_local_backbone_init(tmp_path):
    with pytest.raises(ValueError, match="backbone_init"):
        gt.validate_backbone_config(
            _config(tmp_path, backbone_init=_LOCAL), mode="head_only"
        )


@pytest.mark.parametrize(
    "spec",
    [
        None,
        {"source": "timm_pretrained"},
        {"source": "random_init", "checkpoint_path": None},
    ],
)
def test_siglip_accepts_pretrained_and_random_init(tmp_path, spec):
    gt.validate_backbone_config(_config(tmp_path, backbone_init=spec), mode="full")


def test_crop_assisted_head_only_siglip_is_told_where_the_region_head_trains(tmp_path):
    config = _config(
        tmp_path, device="cpu", region_manifest=str(tmp_path / "regions_manifest.json")
    )
    notes = gt.validate_backbone_config(config, mode="head_only")
    assert any("finetune_blocks=0" in n for n in notes)


def test_apply_backbone_init_refuses_a_siglip_module(monkeypatch):
    install_tiny_trunk(monkeypatch)
    model = create_model(model_size="siglip2_base", pretrained=False, num_classes=2)
    with pytest.raises(ValueError, match="SigLIP"):
        apply_backbone_init(model, _LOCAL)
    assert apply_backbone_init(model, {"source": "timm_pretrained"}) is False


def test_both_group_tasks_validate_before_anything_is_built(tmp_path):
    from bittrainer.generic.tasks.group_task import GroupTask
    from bittrainer.generic.tasks.head_only_task import HeadOnlyTask

    seen: list = []
    full = _config(tmp_path, device="cpu", finetune_blocks=0)
    GroupTask(full).make_context(seen.append, None, None, None)
    assert full.use_compile is False and full.channels_last is False
    assert any("compile" in str(m.get("status_text", "")) for m in seen)

    with pytest.raises(ValueError, match="CUDA"):
        GroupTask(_config(tmp_path, device="cpu", finetune_blocks=2)).make_context(
            None, None, None, None
        )

    head_only = _config(tmp_path, device="cpu", finetune_blocks=2)
    HeadOnlyTask(head_only).make_context(None, None, None, None)
    assert head_only.use_greedy_soup is False

    with pytest.raises(ValueError, match="backbone_init"):
        HeadOnlyTask(
            _config(tmp_path, device="cpu", backbone_init=_LOCAL)
        ).make_context(None, None, None, None)


# ---------------------------------------------------------------------------
# Warm starts never cross families
# ---------------------------------------------------------------------------


def _create(config, tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    return gt._create_or_warmstart_model(
        config,
        device=torch.device("cpu"),
        dtype=torch.float32,
        head_hidden_size=None,
        checkpoint_dir=ckpt_dir,
    )


def _incumbent(tmp_path, model, size):
    (tmp_path / "ckpt").mkdir(exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "model_size": size, "num_classes": 3},
        tmp_path / "ckpt" / "best.pt",
    )


def test_siglip_run_beside_a_convnext_incumbent_starts_fresh(monkeypatch, tmp_path):
    install_tiny_trunk(monkeypatch)
    _incumbent(
        tmp_path,
        create_model(model_size="atto", pretrained=False, num_classes=3),
        "atto",
    )
    config = _config(tmp_path, device="cpu", backbone_init={"source": "random_init"})
    model = _create(config, tmp_path)
    assert isinstance(model, SigLIPClassifier) and model.model_size == "siglip2_so400m"


def test_convnext_run_beside_a_siglip_incumbent_starts_fresh(monkeypatch, tmp_path):
    install_tiny_trunk(monkeypatch)
    _incumbent(
        tmp_path,
        create_model(model_size="siglip2_base", pretrained=False, num_classes=3),
        "siglip2_base",
    )
    config = _config(
        tmp_path,
        backbone_variant="atto",
        device="cpu",
        backbone_init={"source": "random_init"},
    )
    model = _create(config, tmp_path)
    assert not isinstance(model, SigLIPClassifier)
    assert model.stem[0].weight.shape[0] == 40


def test_siglip_warm_start_carries_the_incumbents_weights(monkeypatch, tmp_path):
    install_tiny_trunk(monkeypatch)
    incumbent = create_model(
        model_size="siglip2_so400m", pretrained=False, num_classes=3
    )
    with torch.no_grad():
        incumbent.trunk.blocks[0].attn.qkv.weight.add_(0.5)
        incumbent.head.fc.bias.fill_(0.25)
    _incumbent(tmp_path, incumbent, "siglip2_so400m")
    model = _create(_config(tmp_path, device="cpu"), tmp_path)
    assert torch.equal(
        model.trunk.blocks[0].attn.qkv.weight, incumbent.trunk.blocks[0].attn.qkv.weight
    )
    assert torch.equal(model.head.fc.bias, incumbent.head.fc.bias)


# ---------------------------------------------------------------------------
# Model-load stage names the SigLIP repo
# ---------------------------------------------------------------------------


def test_model_load_stage_checks_the_siglip_repo_cache(monkeypatch, tmp_path):
    import huggingface_hub

    asked: list = []

    def _cache(repo, filename):
        asked.append((repo, filename))
        return "C:/cache/model.safetensors"

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _cache)
    stages: list = []

    class _Em:
        def stage(self, stage, text, **_kw):
            stages.append((stage, text))

    (tmp_path / "ckpt").mkdir()
    gt._emit_model_load_stage(_Em(), _config(tmp_path), tmp_path / "ckpt")
    assert asked == [
        ("timm/naflexvit_so400m_patch16_siglip.v2_webli", "model.safetensors")
    ]
    assert "siglip2_so400m" in stages[-1][1]
