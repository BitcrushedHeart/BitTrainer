"""Checkpoints say which architecture they hold (Bitcrush ISSUE-1106).

``_architecture_ckpt_meta`` is spread into every group checkpoint save site. It
is ADDITIVE and only non-empty when the architecture is not the historical one:
a softmax checkpoint gets ``{}`` and so stays byte-identical (the all-threshold
ordinal head's key is pinned in ``test_threshold_loss_wiring``).

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
from bittrainer.model import create_model

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


def test_convnext_softmax_meta_is_empty():
    model = create_model(model_size="atto", pretrained=False, num_classes=2)
    assert gt._architecture_ckpt_meta(model) == {}
    assert gt._architecture_ckpt_meta(nn.Linear(2, 2)) == {}


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
            if name == "_architecture_ckpt_meta":
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
