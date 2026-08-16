"""Single-concept cached-head explicit-negative strength tuning (ISSUE-0754)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

import bittrainer.generic.tasks.binary_head_only_task as head_task
from bittrainer.trainer import TrainConfig


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(2, 2)


def _config(tmp_path, candidates: list[int]) -> TrainConfig:
    return TrainConfig(
        concept_folder=str(tmp_path),
        hard_negative_weight_candidates=candidates,
    )


def _tensors() -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros(4, 2),
        torch.tensor([1, 1, 0, 0]),
        torch.ones(2, 2),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(3, 2),
        torch.tensor([1, 0, 0]),
    )


def test_weight_sweep_resets_head_and_selects_best_f1(monkeypatch, tmp_path) -> None:
    model = _TinyModel()
    original_weight = model.head.weight.detach().clone()
    starts_clean: list[bool] = []
    train_sizes: list[int] = []
    val_sizes: list[int] = []
    scores = {1: (0.60, 0.70), 2: (0.84, 0.50), 3: (0.80, 0.40), 5: (0.70, 0.30)}

    def _fake_train(model, x_train, _y_train, x_val, _y_val, _config, **_kwargs):
        starts_clean.append(torch.allclose(model.head.weight, original_weight))
        train_sizes.append(len(x_train))
        val_sizes.append(len(x_val))
        strength = (len(x_train) - 4) // 2
        f1, loss = scores[strength]
        model.head.weight.data.fill_(float(strength))
        return {
            "best_epoch": strength,
            "epochs_completed": strength + 1,
            "best_val_f1": f1,
            "best_metrics": {
                "f1": f1,
                "precision": f1 - 0.01,
                "recall": f1 - 0.02,
                "auprc": f1 - 0.03,
                "val_loss": loss,
            },
        }

    monkeypatch.setattr(head_task, "_train_cached_binary_head", _fake_train)
    config = _config(tmp_path, [1, 2, 3, 5])

    result = head_task._train_hard_negative_weight_sweep(
        model,
        *_tensors(),
        config,
        device=torch.device("cpu"),
        cb=lambda _msg: None,
        stop_event=None,
    )

    assert starts_clean == [True, True, True, True]
    assert train_sizes == [6, 8, 10, 14]
    assert val_sizes == [3, 3, 3, 3]
    assert result["best_val_f1"] == 0.84
    assert result["best_epoch"] == 2
    assert config.hard_negative_weight == 2
    assert config.selected_hard_negative_weight == 2
    assert [row["weight"] for row in config.hard_negative_weight_tuning_results] == [
        1,
        2,
        3,
        5,
    ]
    assert config.hard_negative_weight_tuning_results[1]["precision"] == 0.83
    assert config.hard_negative_weight_tuning_elapsed_ms is not None
    assert torch.all(model.head.weight == 2.0)


def test_weight_sweep_tie_prefers_lower_strength(monkeypatch, tmp_path) -> None:
    model = _TinyModel()

    def _fake_train(model, x_train, *_args, **_kwargs):
        strength = len(x_train) - 1
        model.head.weight.data.fill_(float(strength))
        return {
            "best_epoch": 1,
            "epochs_completed": 1,
            "best_val_f1": 0.75,
            "best_metrics": {"f1": 0.75, "val_loss": 0.4},
        }

    monkeypatch.setattr(head_task, "_train_cached_binary_head", _fake_train)
    config = _config(tmp_path, [3, 1])
    x_base = torch.zeros(1, 2)
    y_base = torch.ones(1, dtype=torch.long)
    x_explicit = torch.ones(1, 2)
    y_explicit = torch.zeros(1, dtype=torch.long)
    x_val = torch.zeros(2, 2)
    y_val = torch.tensor([1, 0])

    head_task._train_hard_negative_weight_sweep(
        model,
        x_base,
        y_base,
        x_explicit,
        y_explicit,
        x_val,
        y_val,
        config,
        device=torch.device("cpu"),
        cb=lambda _msg: None,
        stop_event=None,
    )

    assert config.selected_hard_negative_weight == 1
    assert torch.all(model.head.weight == 1.0)


def test_default_candidates_are_a_single_unrepeated_strength(tmp_path) -> None:
    """The sweep is off by default (Bitcrush ISSUE-0773).

    Weight 5 went first (ISSUE-0766, 13 live sweeps). A holdout A/B over 8 seeds
    then showed the surviving [1,2,3] sweep was selecting seed noise: its picks
    moved run to run while the val-score spread between candidates stayed under
    the seed-to-seed variance, and removing it *improved* held-out recall
    (0.624 -> 0.643) at identical precision for a third of the probe cost.

    The machinery stays — a caller can still pass several candidates — but the
    default no longer pays for three head trainings to pick between them.
    """
    config = TrainConfig(concept_folder=str(tmp_path))
    assert config.hard_negative_weight_candidates == [1]


def test_weight_sweep_skips_without_explicit_negatives(monkeypatch, tmp_path) -> None:
    model = _TinyModel()
    calls: list[int] = []

    def _fake_train(_model, x_train, *_args, **_kwargs):
        calls.append(len(x_train))
        return {
            "best_epoch": 1,
            "epochs_completed": 1,
            "best_val_f1": 0.5,
            "best_metrics": {"f1": 0.5, "val_loss": 0.5},
        }

    monkeypatch.setattr(head_task, "_train_cached_binary_head", _fake_train)
    config = _config(tmp_path, [1, 2, 3, 5])
    x_base, y_base, _x_explicit, _y_explicit, x_val, y_val = _tensors()

    result = head_task._train_hard_negative_weight_sweep(
        model,
        x_base,
        y_base,
        torch.empty(0, 2),
        torch.empty(0, dtype=torch.long),
        x_val,
        y_val,
        config,
        device=torch.device("cpu"),
        cb=lambda _msg: None,
        stop_event=None,
    )

    assert result["best_val_f1"] == 0.5
    assert calls == [4]
    assert config.selected_hard_negative_weight is None
    assert config.hard_negative_weight_tuning_results == []
    assert config.hard_negative_weight_tuning_elapsed_ms is None


def test_partition_removes_dataset_repetitions_but_keeps_one_explicit_sample() -> None:
    explicit = {"path": "explicit.png", "label": 0}
    base = [
        {"path": "positive.png", "label": 1},
        {"path": "implied.png", "label": 0},
    ]

    base_samples, explicit_samples = head_task._partition_explicit_negative_samples(
        [*base, explicit, explicit, explicit],
        [Path("explicit.png")],
    )

    assert base_samples == base
    assert explicit_samples == [explicit]


def test_real_head_run_persists_selected_strength_in_checkpoint(tmp_path) -> None:
    rng = np.random.default_rng(75)

    def _images(folder: Path, prefix: str, count: int, red_offset: int) -> list[str]:
        folder.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for index in range(count):
            pixels = rng.integers(0, 100, (64, 64, 3), dtype=np.uint8)
            pixels[..., 0] = np.clip(pixels[..., 0] + red_offset, 0, 255)
            path = folder / f"{prefix}-{index}.png"
            Image.fromarray(pixels).save(path)
            paths.append(str(path))
        return paths

    positives_train = _images(tmp_path / "positive", "train", 4, 120)
    positives_val = _images(tmp_path / "positive", "val", 2, 120)
    implied_train = _images(tmp_path / "implied", "train", 6, 0)
    implied_val = _images(tmp_path / "implied", "val", 3, 0)
    explicit_train = _images(tmp_path / "explicit", "train", 2, 50)
    explicit_val = _images(tmp_path / "explicit", "val", 1, 50)
    config = TrainConfig(
        concept_folder=str(tmp_path / "concept"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        embedding_cache_dir=str(tmp_path / "embeddings"),
        training_mode="head_only",
        model_size="atto",
        device="cpu",
        dtype="float32",
        use_cache=False,
        max_epochs=2,
        patience=2,
        dataloader_workers=0,
        backbone_init={"source": "random_init", "checkpoint_path": None},
        train_positive_paths=positives_train,
        val_positive_paths=positives_val,
        train_negative_paths=implied_train,
        val_negative_paths=implied_val,
        train_hard_negative_paths=explicit_train,
        val_hard_negative_paths=explicit_val,
        hard_negative_weight_candidates=[1, 2],
    )

    result = head_task.bt.run_training(config)

    assert result["selected_hard_negative_weight"] in {1, 2}
    assert [row["weight"] for row in result["hard_negative_weight_tuning_results"]] == [
        1,
        2,
    ]
    checkpoint = torch.load(
        result["checkpoint_path"], map_location="cpu", weights_only=True
    )
    assert (
        checkpoint["selected_hard_negative_weight"]
        == result["selected_hard_negative_weight"]
    )
    assert (
        checkpoint["hard_negative_weight_tuning_results"]
        == result["hard_negative_weight_tuning_results"]
    )
