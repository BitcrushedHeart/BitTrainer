"""Checkpoints record the resolution they were trained at (Bitcrush ISSUE-0992).

Engine served every group/concept checkpoint through the default 512 bucket
table, so a model trained at 768 (Anatomy) or under a batch-train resolution pin
was scored with its subjects at ~2/3 of the trained scale. The fix makes the
checkpoint the source of truth: every save site writes ``train_resolution`` so
serving can bucket at the same scale training did.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from bittrainer.generic.task import BestTracker

_TASKS_DIR = Path(__file__).resolve().parents[1] / "generic" / "tasks"
_SAVE_SITE_FILES = (
    "group_task.py",
    "head_only_task.py",
    "binary_task.py",
    "binary_head_only_task.py",
)


def test_resolution_meta_records_configured_resolution():
    from bittrainer.dataset import resolution_ckpt_meta

    assert resolution_ckpt_meta(768) == {"train_resolution": 768}
    assert resolution_ckpt_meta(512) == {"train_resolution": 512}


def test_resolution_meta_falls_back_to_default_when_unset():
    from bittrainer.dataset import resolution_ckpt_meta

    assert resolution_ckpt_meta(0) == {"train_resolution": 512}
    assert resolution_ckpt_meta(None) == {"train_resolution": 512}


def test_group_candidate_records_train_resolution(tmp_path):
    from bittrainer.generic.tasks.group_task import GroupTask
    from bittrainer.group_trainer import GroupTrainConfig

    config = GroupTrainConfig(
        group_folder=str(tmp_path),
        num_classes=2,
        class_names=["a", "b"],
        train_resolution=768,
    )
    task = GroupTask(config)
    task.ema = None
    ctx = SimpleNamespace(checkpoint_dir=tmp_path)

    task.save_candidate(ctx, nn.Linear(2, 2), 0, {"macro_f1": 0.5}, BestTracker())

    ckpt = torch.load(tmp_path / "candidate.pt", weights_only=False)
    assert ckpt["train_resolution"] == 768


def test_binary_candidate_records_train_resolution(tmp_path):
    from bittrainer.generic.tasks.binary_task import BinaryTask
    from bittrainer.trainer import TrainConfig

    task = BinaryTask(TrainConfig(concept_folder=str(tmp_path), train_resolution=640))
    ctx = SimpleNamespace(checkpoint_dir=tmp_path)

    task.save_candidate(ctx, nn.Linear(2, 2), 0, {"f1": 0.5}, BestTracker())

    ckpt = torch.load(tmp_path / "candidate.pt", weights_only=False)
    assert ckpt["train_resolution"] == 640


def _checkpoint_dict_literals(tree: ast.AST):
    """Every dict literal carrying a ``"state_dict"`` key — i.e. a checkpoint."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        if "state_dict" in keys:
            yield node


def _records_resolution(node: ast.Dict) -> bool:
    for key, value in zip(node.keys, node.values, strict=True):
        if isinstance(key, ast.Constant) and key.value == "train_resolution":
            return True
        # ``**resolution_ckpt_meta(...)`` spread.
        if key is None and isinstance(value, ast.Call):
            func = value.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name == "resolution_ckpt_meta":
                return True
    return False


def test_every_task_checkpoint_records_train_resolution():
    """Structural guard: a new save site cannot silently drop the key again."""
    missing = []
    for name in _SAVE_SITE_FILES:
        path = _TASKS_DIR / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        literals = list(_checkpoint_dict_literals(tree))
        assert literals, f"{name}: no checkpoint dict found — guard is stale"
        missing += [
            f"{name}:{node.lineno}" for node in literals if not _records_resolution(node)
        ]
    assert not missing, f"checkpoints without train_resolution: {missing}"
