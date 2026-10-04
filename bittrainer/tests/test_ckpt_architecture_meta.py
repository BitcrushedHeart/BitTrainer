"""Checkpoints say which architecture they hold (Bitcrush ISSUE-1106).

``_architecture_ckpt_meta`` is spread into every group checkpoint save site. It
is ADDITIVE and only non-empty when the architecture is not the historical one:
a ConvNeXt softmax checkpoint gets ``{}`` and so stays byte-identical, while a
SigLIP checkpoint records ``backbone_family`` and — when its trunk is frozen
(head-only, or ``finetune_blocks=0``) — the ``backbone_hash`` of the saved trunk,
which Engine uses to share one frozen trunk across groups.

The AST guard (modelled on ``test_ckpt_train_resolution``) keeps a future save
site from silently dropping the meta.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

import bittrainer.group_trainer as gt
from bittrainer.generic.task import BestTracker
from bittrainer.generic.tasks.group_task import GroupTask
from bittrainer.model import backbone_state_hash, create_model, prepare_for_finetune
from bittrainer.tests.siglip_fixtures import install_tiny_trunk

_TASKS_DIR = Path(__file__).resolve().parents[1] / "generic" / "tasks"
_SAVE_SITE_FILES = ("group_task.py", "head_only_task.py")


def _config(tmp_path, size="atto", **kw):
    return gt.GroupTrainConfig(
        group_folder=str(tmp_path),
        num_classes=2,
        class_names=["a", "b"],
        backbone_variant=size,
        **kw,
    )


def test_convnext_softmax_meta_is_empty(tmp_path):
    model = create_model(model_size="atto", pretrained=False, num_classes=2)
    assert gt._architecture_ckpt_meta(_config(tmp_path), model) == {}
    # Even with a hash on offer: ConvNeXt checkpoints must not change.
    assert (
        gt._architecture_ckpt_meta(_config(tmp_path), model, backbone_hash="abc") == {}
    )


def test_non_model_objects_are_treated_as_convnext(tmp_path):
    assert gt._architecture_ckpt_meta(_config(tmp_path), nn.Linear(2, 2)) == {}


def test_siglip_meta_names_the_family(monkeypatch, tmp_path):
    install_tiny_trunk(monkeypatch)
    model = create_model(model_size="siglip2_base", pretrained=False, num_classes=2)
    prepare_for_finetune(model, 2)
    meta = gt._architecture_ckpt_meta(_config(tmp_path, "siglip2_base"), model)
    assert meta == {"backbone_family": "siglip2"}


def test_frozen_siglip_trunk_records_the_saved_trunk_hash(monkeypatch, tmp_path):
    install_tiny_trunk(monkeypatch)
    model = create_model(model_size="siglip2_base", pretrained=False, num_classes=2)
    prepare_for_finetune(model, 0)
    state = model.state_dict()
    meta = gt._architecture_ckpt_meta(
        _config(tmp_path, "siglip2_base"), model, state=state
    )
    assert meta == {
        "backbone_family": "siglip2",
        "backbone_hash": backbone_state_hash(state),
    }
    # A precomputed hash (head-only computes it for the embedding cache) wins.
    assert (
        gt._architecture_ckpt_meta(
            _config(tmp_path, "siglip2_base"), model, backbone_hash="feedbeef"
        )["backbone_hash"]
        == "feedbeef"
    )


def test_convnext_candidate_keys_are_unchanged(tmp_path):
    task = GroupTask(_config(tmp_path))
    task.ema = None
    model = create_model(model_size="atto", pretrained=False, num_classes=2)
    task.save_candidate(
        SimpleNamespace(checkpoint_dir=tmp_path),
        model,
        0,
        {"macro_f1": 0.5},
        BestTracker(),
    )
    ckpt = torch.load(tmp_path / "candidate.pt", weights_only=True)
    assert set(ckpt) == {
        "state_dict",
        "num_classes",
        "model_size",
        "class_names",
        "validation_metric",
        "train_resolution",
    }


def test_siglip_candidate_records_family_and_frozen_hash(monkeypatch, tmp_path):
    install_tiny_trunk(monkeypatch)
    task = GroupTask(_config(tmp_path, "siglip2_base", finetune_blocks=0))
    task.ema = None
    model = create_model(model_size="siglip2_base", pretrained=False, num_classes=2)
    prepare_for_finetune(model, 0)
    task.save_candidate(
        SimpleNamespace(checkpoint_dir=tmp_path),
        model,
        0,
        {"macro_f1": 0.5},
        BestTracker(),
    )
    ckpt = torch.load(tmp_path / "candidate.pt", weights_only=True)
    assert ckpt["backbone_family"] == "siglip2"
    assert ckpt["backbone_hash"] == backbone_state_hash(ckpt["state_dict"])
    assert ckpt["model_size"] == "siglip2_base"


# ---------------------------------------------------------------------------
# Structural guard
# ---------------------------------------------------------------------------


def _checkpoint_dict_literals(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        if "state_dict" in keys:
            yield node


def _records_architecture(node: ast.Dict) -> bool:
    for key, value in zip(node.keys, node.values, strict=True):
        if key is None and isinstance(value, ast.Call):
            func = value.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
            if name in ("_architecture_ckpt_meta", "_arch_meta"):
                return True
    return False


def test_every_group_checkpoint_records_the_architecture():
    missing = []
    for name in _SAVE_SITE_FILES:
        tree = ast.parse((_TASKS_DIR / name).read_text(encoding="utf-8"))
        literals = list(_checkpoint_dict_literals(tree))
        assert literals, f"{name}: no checkpoint dict found — guard is stale"
        missing += [
            f"{name}:{n.lineno}" for n in literals if not _records_architecture(n)
        ]
    assert not missing, f"checkpoints without the architecture meta: {missing}"
