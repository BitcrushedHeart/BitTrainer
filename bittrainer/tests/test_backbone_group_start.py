"""A group model started from a clean trunk and its own backbone head
(Bitcrush ISSUE-1721).

Every group model is retrained from a BitVision trunk that descends from the
Apache ConvNeXt V1 weights, never from its current (CC-BY-NC) model. The
backbone already holds a head for the group, fed the same post-norm pooled
feature as a group model's ``head.fc``, so the group model can start where the
backbone's head is: same answers, in Engine's class order rather than the
backbone's sorted one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from bittrainer import lineage
from bittrainer.backbone_group_start import GroupStartError, start_group_model
from bittrainer.backbone_init import apply_backbone_init
from bittrainer.model import backbone_feature_hash, create_model

# Engine's class order; the backbone sorts them as strings ("k10" < "k9").
_CLASSES = ["k9", "k10", "__none__"]
_ROOT = "apache-convnext-v1:convnext_atto.d2_in1k"
_LINEAGE = {
    "root": _ROOT,
    "steps": [
        {"kind": "backbone", "size": "base", "from_trunk": None},
        {"kind": "distil", "size": "atto", "from_trunk": "0123456789abcdef"},
    ],
}


def _candidate(path: Path, **metadata) -> tuple[Path, torch.nn.Module, dict]:
    """A student candidate: an atto trunk plus heads, as the trainer writes it."""
    torch.manual_seed(0)
    trunk = create_model(
        model_size="atto", pretrained=False, num_classes=0, drop_path_rate=0.0
    ).eval()
    width = trunk.num_features
    groups = {"g6": sorted(_CLASSES), "g7": ["k1", "k2"]}
    state = {k: v.contiguous() for k, v in trunk.state_dict().items()}
    heads = {}
    for group, classes in groups.items():
        heads[f"heads.groups.{group}.weight"] = torch.randn(len(classes), width)
        heads[f"heads.groups.{group}.bias"] = torch.randn(len(classes))
    heads["heads.binary.c1.weight"] = torch.randn(1, width)
    heads["heads.binary.c1.bias"] = torch.randn(1)
    header = {
        "convnextv2_size": "atto",
        "heads_concepts_json": json.dumps(["c1"]),
        "heads_groups_json": json.dumps(groups),
        "lineage_json": json.dumps(_LINEAGE),
    }
    header.update(metadata)
    header = {k: v for k, v in header.items() if v is not None}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({**state, **heads}, str(path), metadata=header)
    return path, trunk, heads


def _backbone_logits(trunk, heads, images) -> torch.Tensor:
    """The backbone's own g6 answers, put in Engine's class order."""
    with torch.no_grad():
        logits = F.linear(
            trunk(images),
            heads["heads.groups.g6.weight"],
            heads["heads.groups.g6.bias"],
        )
    order = [sorted(_CLASSES).index(name) for name in _CLASSES]
    return logits[:, order]


def _images() -> torch.Tensor:
    return torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(1))


def test_the_group_model_answers_as_the_backbone_head_in_engine_order(tmp_path):
    trunk_file, trunk, heads = _candidate(tmp_path / "small.safetensors")

    start = start_group_model(trunk_file, "g6", _CLASSES, "atto")

    model = start.model.eval()
    with torch.no_grad():
        got = model(_images())
    assert got.shape == (2, 3)
    torch.testing.assert_close(got, _backbone_logits(trunk, heads, _images()))
    assert start.head_seeded is True
    assert start.class_names == _CLASSES


def test_the_start_records_its_trunk_and_its_lineage(tmp_path):
    trunk_file, _trunk, _heads = _candidate(tmp_path / "small.safetensors")

    start = start_group_model(trunk_file, "g6", _CLASSES, "atto")

    assert start.start_trunk_hash == lineage.trunk_hash(trunk_file)
    assert start.start_trunk_hash == backbone_feature_hash(start.model)
    assert start.lineage == {
        "root": _ROOT,
        "steps": [
            *_LINEAGE["steps"],
            {"kind": "group", "size": "atto", "from_trunk": start.start_trunk_hash},
        ],
    }


def test_the_pt_carries_lineage_json(tmp_path):
    trunk_file, _trunk, _heads = _candidate(tmp_path / "small.safetensors")
    out = tmp_path / "seed" / "g00006.pt"

    start = start_group_model(trunk_file, "g6", _CLASSES, "atto", out=out)

    saved = torch.load(str(out), map_location="cpu", weights_only=True)
    assert json.loads(saved["lineage_json"]) == start.lineage
    assert saved["class_names"] == _CLASSES
    assert saved["model_size"] == "atto"
    assert saved["start_trunk_hash"] == start.start_trunk_hash
    assert lineage.apache_root(lineage.read_metadata(out)) == _ROOT


def test_a_safetensors_seed_starts_the_group_trainer_through_backbone_init(tmp_path):
    """How the group trainer takes it: ``backbone_init`` local_candidate loads
    every tensor whose key and shape match, the seeded head.fc included."""
    trunk_file, trunk, heads = _candidate(tmp_path / "small.safetensors")
    seed = tmp_path / "seed" / "g00006.safetensors"

    start_group_model(trunk_file, "g6", _CLASSES, "atto", out=seed)

    model = create_model(
        model_size="atto", pretrained=False, num_classes=3, drop_path_rate=0.0
    )
    apply_backbone_init(
        model, {"source": "local_candidate", "checkpoint_path": str(seed)}
    )
    with torch.no_grad():
        got = model.eval()(_images())
    torch.testing.assert_close(got, _backbone_logits(trunk, heads, _images()))
    with safe_open(str(seed), framework="pt") as handle:
        header = dict(handle.metadata())
    assert lineage.apache_root(header) == _ROOT
    assert json.loads(header["class_names_json"]) == _CLASSES


def test_a_5982d6e_trunk_with_an_apache_init_is_a_good_start(tmp_path):
    trunk_file, _trunk, _heads = _candidate(
        tmp_path / "large.safetensors",
        lineage_json=None,
        initialization_source="convnext_v1_converted",
        pretrained_model_id="timm/convnext_atto.d2_in1k",
        pretrained_license="apache-2.0",
        non_commercial_only="false",
    )

    start = start_group_model(trunk_file, "g6", _CLASSES, "atto")

    assert start.lineage["root"] == _ROOT
    assert [s["kind"] for s in start.lineage["steps"]] == ["backbone", "group"]


def test_a_trunk_without_an_apache_root_is_refused(tmp_path):
    trunk_file, _trunk, _heads = _candidate(
        tmp_path / "nc.safetensors",
        lineage_json=None,
        initialization_source="timm_pretrained",
        pretrained_model_id="timm/convnextv2_atto.fcmae_ft_in1k",
        pretrained_license="CC-BY-NC-4.0",
        non_commercial_only="true",
    )

    with pytest.raises(lineage.LineageError, match="CC-BY-NC-4.0"):
        start_group_model(trunk_file, "g6", _CLASSES, "atto")


def test_a_trunk_of_another_size_is_refused(tmp_path):
    trunk_file, _trunk, _heads = _candidate(tmp_path / "small.safetensors")

    with pytest.raises(GroupStartError, match="nano"):
        start_group_model(trunk_file, "g6", _CLASSES, "nano")


def test_a_trunk_with_no_head_for_the_group_is_refused(tmp_path):
    trunk_file, _trunk, _heads = _candidate(tmp_path / "small.safetensors")

    with pytest.raises(GroupStartError, match="g8"):
        start_group_model(trunk_file, "g8", _CLASSES, "atto")


def test_a_head_with_other_classes_is_refused(tmp_path):
    trunk_file, _trunk, _heads = _candidate(tmp_path / "small.safetensors")

    with pytest.raises(GroupStartError, match="k11"):
        start_group_model(trunk_file, "g6", ["k9", "k10", "k11"], "atto")
    with pytest.raises(GroupStartError):
        start_group_model(trunk_file, "g6", ["k9", "k10"], "atto")


def test_a_hidden_mlp_head_keeps_the_trunk_and_starts_its_tail_fresh(tmp_path):
    trunk_file, trunk, _heads = _candidate(tmp_path / "small.safetensors")

    start = start_group_model(trunk_file, "g6", _CLASSES, "atto", head_hidden_size=32)

    assert start.head_seeded is False
    assert "fresh" in start.note
    state = start.model.state_dict()
    for key, value in trunk.state_dict().items():
        torch.testing.assert_close(state[key], value)
    assert state["head.fc.weight"].shape == (3, 32)
    assert start.start_trunk_hash == lineage.trunk_hash(trunk_file)
