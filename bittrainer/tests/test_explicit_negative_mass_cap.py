"""Bitcrush ISSUE-0898: bound the explicit-negative block's total loss mass.

The implied pool has been mass-normalised since ISSUE-0773, but explicit
negatives still enter at full weight each, so a concept whose owner has
labelled MORE negatives than positives (Sideboob: 784 vs 238) trains its
verified-negative block at >3x the positives' mass and the head collapses to
"absent" (served recall 3%). Capping that block at ``cap`` x the positive mass
restored served recall to 37% on the harness holdout while leaving a concept
with few explicit negatives (Cleavage, 232 vs 927) untouched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from bittrainer.dataset import KIND_EXPLICIT_NEGATIVE, ConceptDataset
from bittrainer.generic.tasks.binary_head_only_task import (
    _weighted_train_tensors,
    implied_negative_weights,
)
from bittrainer.trainer import TrainConfig


def _image(path: Path, colour: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), colour).save(path)
    return path


def test_config_defaults_to_a_one_to_one_cap() -> None:
    assert TrainConfig(concept_folder="x").explicit_negative_mass_cap == pytest.approx(
        1.0
    )


# --- head-only path: per-row weights -----------------------------------------


def _blocks(n_pos: int, n_implied: int, n_explicit: int):
    x_base = torch.zeros(n_pos + n_implied, 3)
    y_base = torch.cat(
        [torch.ones(n_pos, dtype=torch.long), torch.zeros(n_implied, dtype=torch.long)]
    )
    w_base = implied_negative_weights(y_base, alpha=1.0)
    x_explicit = torch.ones(n_explicit, 3)
    y_explicit = torch.zeros(n_explicit, dtype=torch.long)
    return x_base, y_base, w_base, x_explicit, y_explicit


def test_abundant_explicit_block_is_capped_to_the_positive_mass() -> None:
    x_base, y_base, w_base, x_explicit, y_explicit = _blocks(20, 100, 80)
    _, y, w = _weighted_train_tensors(
        x_base, y_base, x_explicit, y_explicit, 1, w_base, mass_cap=1.0
    )
    explicit_mass = w[len(y_base) :].sum().item()
    assert explicit_mass == pytest.approx(20.0)
    assert w[len(y_base)].item() == pytest.approx(0.25)


def test_scarce_explicit_block_keeps_full_weight() -> None:
    x_base, y_base, w_base, x_explicit, y_explicit = _blocks(20, 100, 5)
    _, y, w = _weighted_train_tensors(
        x_base, y_base, x_explicit, y_explicit, 1, w_base, mass_cap=1.0
    )
    assert torch.equal(w[len(y_base) :], torch.ones(5))


def test_cap_bounds_the_repeated_block_not_each_copy() -> None:
    # 30 explicit x 3 repetitions = 90 rows against 20 positives: capped to 20.
    x_base, y_base, w_base, x_explicit, y_explicit = _blocks(20, 100, 30)
    _, y, w = _weighted_train_tensors(
        x_base, y_base, x_explicit, y_explicit, 3, w_base, mass_cap=1.0
    )
    assert w[len(y_base) :].sum().item() == pytest.approx(20.0)


def test_cap_zero_disables_and_restores_full_weight() -> None:
    x_base, y_base, w_base, x_explicit, y_explicit = _blocks(20, 100, 80)
    _, y, w = _weighted_train_tensors(
        x_base, y_base, x_explicit, y_explicit, 1, w_base, mass_cap=0.0
    )
    assert torch.equal(w[len(y_base) :], torch.ones(80))


# --- full path: ConceptDataset subsamples the block per epoch ------------------


def test_dataset_subsamples_an_abundant_explicit_block(tmp_path: Path) -> None:
    folder = tmp_path / "concept"
    positives = [_image(folder / f"pos{i}.png", (255, 0, 0)) for i in range(4)]
    implied = [
        _image(tmp_path / "other" / f"imp{i}.png", (0, 0, 255)) for i in range(20)
    ]
    explicit = [
        _image(tmp_path / "hard" / f"exp{i}.png", (0, 255, 0)) for i in range(12)
    ]

    ds = ConceptDataset(
        folder,
        split="train",
        positive_paths=[str(p) for p in positives],
        negative_paths=[str(p) for p in implied],
        hard_negative_paths=[str(p) for p in explicit],
        hard_negative_weight=3,
        explicit_negative_mass_cap=1.0,
    )
    explicit_rows = [s for s in ds.samples if s["kind"] == KIND_EXPLICIT_NEGATIVE]
    # 12 x 3 = 36 rows would carry 9x the positives; capped to 4 rows.
    assert len(explicit_rows) == 4
    # Evidence is not lost, only the per-epoch share: every explicit negative
    # is still counted and the draw changes across epochs.
    assert ds.negative_kind_counts()["explicit"] == 12
    draws = set()
    for _ in range(10):
        ds.resample_negatives()
        draws.add(
            tuple(
                sorted(
                    s["path"] for s in ds.samples if s["kind"] == KIND_EXPLICIT_NEGATIVE
                )
            )
        )
    assert len(draws) > 1


def test_dataset_keeps_a_scarce_explicit_block_with_repetition(tmp_path: Path) -> None:
    folder = tmp_path / "concept"
    positives = [_image(folder / f"pos{i}.png", (255, 0, 0)) for i in range(10)]
    implied = [
        _image(tmp_path / "other" / f"imp{i}.png", (0, 0, 255)) for i in range(20)
    ]
    explicit = [
        _image(tmp_path / "hard" / f"exp{i}.png", (0, 255, 0)) for i in range(2)
    ]

    ds = ConceptDataset(
        folder,
        split="train",
        positive_paths=[str(p) for p in positives],
        negative_paths=[str(p) for p in implied],
        hard_negative_paths=[str(p) for p in explicit],
        hard_negative_weight=3,
        explicit_negative_mass_cap=1.0,
    )
    explicit_rows = [s for s in ds.samples if s["kind"] == KIND_EXPLICIT_NEGATIVE]
    assert len(explicit_rows) == 6
