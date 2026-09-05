"""Bitcrush ISSUE-0898: validation metrics split by negative kind.

Explicit negatives are hard near-misses and they sit in the val split, so a
concept's F1 falls as its owner adds them, while a concept with none is scored
only against easy implied negatives. Across 108 live concepts the explicit
neg:pos ratio correlated -0.63 with val F1. The single F1 therefore cannot say
whether a model is confused by near-misses or by the rest of the dataset.
These tests pin the per-kind bookkeeping that makes the two visible.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from bittrainer.dataset import (
    KIND_EXPLICIT_NEGATIVE,
    KIND_IMPLIED_NEGATIVE,
    KIND_POSITIVE,
    BucketBatchSampler,
    ConceptDataset,
)
from bittrainer.trainer import _collate_bucket_batch, _tuned_val_metrics, evaluate
from bittrainer.validation import negative_kind_metrics


def _image(path: Path, colour: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), colour).save(path)
    return path


# --- dataset provenance ---------------------------------------------------


def test_samples_carry_their_negative_kind(tmp_path: Path) -> None:
    folder = tmp_path / "concept"
    pos = _image(folder / "pos.png", (255, 0, 0))
    implied = [
        _image(tmp_path / "other" / f"imp{i}.png", (0, 0, 255)) for i in range(4)
    ]
    explicit = [
        _image(tmp_path / "hard" / f"exp{i}.png", (0, 255, 0)) for i in range(2)
    ]

    ds = ConceptDataset(
        folder,
        split="train",
        positive_paths=[str(pos)],
        negative_paths=[str(p) for p in implied],
        hard_negative_paths=[str(p) for p in explicit],
        hard_negative_weight=3,
    )

    kinds = {s["kind"] for s in ds.samples}
    assert kinds == {KIND_POSITIVE, KIND_EXPLICIT_NEGATIVE, KIND_IMPLIED_NEGATIVE}
    for sample in ds.samples:
        assert (sample["label"] == 1) == (sample["kind"] == KIND_POSITIVE)

    counts = ds.negative_kind_counts()
    # Explicit negatives are counted once each even though repetition puts three
    # copies of every one into the epoch; implied is the sampled quota.
    assert counts == {"explicit": 2, "implied": 3}
    assert sum(s["kind"] == KIND_EXPLICIT_NEGATIVE for s in ds.samples) == 6


def test_negative_kind_counts_with_no_explicit_negatives(tmp_path: Path) -> None:
    folder = tmp_path / "concept"
    pos = _image(folder / "pos.png", (255, 0, 0))
    implied = [
        _image(tmp_path / "other" / f"imp{i}.png", (0, 0, 255)) for i in range(2)
    ]
    ds = ConceptDataset(
        folder,
        split="train",
        positive_paths=[str(pos)],
        negative_paths=[str(p) for p in implied],
    )
    assert ds.negative_kind_counts() == {"explicit": 0, "implied": 2}


# --- per-kind metric arithmetic ------------------------------------------


def test_negative_kind_metrics_split_the_negatives() -> None:
    labels = [1, 1, 1, 0, 0, 0, 0, 0]
    probs = [0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.1, 0.1]
    kinds = (
        [KIND_POSITIVE] * 3 + [KIND_EXPLICIT_NEGATIVE] * 2 + [KIND_IMPLIED_NEGATIVE] * 3
    )

    out = negative_kind_metrics(labels, probs, kinds, threshold=0.5)

    assert out["val_explicit_negative_count"] == 2
    assert out["val_implied_negative_count"] == 3
    # Explicit block: one of two fires (0.7). Positives: 2 of 3 fire.
    assert out["fpr_explicit"] == pytest.approx(0.5)
    assert out["precision_explicit"] == pytest.approx(2 / 3)
    assert out["f1_explicit"] == pytest.approx(
        2 * (2 / 3) * (2 / 3) / ((2 / 3) + (2 / 3))
    )
    # Implied block: one of three fires (0.6).
    assert out["fpr_implied"] == pytest.approx(1 / 3)
    assert out["precision_implied"] == pytest.approx(2 / 3)


def test_negative_kind_metrics_report_none_for_an_absent_kind() -> None:
    labels = [1, 0, 0]
    probs = [0.9, 0.1, 0.8]
    kinds = [KIND_POSITIVE, KIND_IMPLIED_NEGATIVE, KIND_IMPLIED_NEGATIVE]

    out = negative_kind_metrics(labels, probs, kinds, threshold=0.5)

    assert out["val_explicit_negative_count"] == 0
    assert out["f1_explicit"] is None
    assert out["precision_explicit"] is None
    assert out["fpr_explicit"] is None
    assert out["fpr_implied"] == pytest.approx(0.5)


def test_tuned_val_metrics_merge_kind_metrics_when_present() -> None:
    val_result = {
        "labels": [1, 1, 0, 0],
        "probs": [0.9, 0.7, 0.2, 0.8],
        "kinds": [
            KIND_POSITIVE,
            KIND_POSITIVE,
            KIND_IMPLIED_NEGATIVE,
            KIND_EXPLICIT_NEGATIVE,
        ],
    }
    metrics, threshold = _tuned_val_metrics(val_result)
    assert "f1_explicit" in metrics and "f1_implied" in metrics
    assert metrics["val_explicit_negative_count"] == 1
    # Legacy callers without provenance are untouched.
    legacy, _ = _tuned_val_metrics({"labels": [1, 0], "probs": [0.9, 0.1]})
    assert "f1_explicit" not in legacy
    assert 0.0 < threshold <= 1.0


# --- evaluate() keeps provenance aligned with a shuffling sampler ----------


class _KindDataset:
    """The minimum ConceptDataset surface evaluate() and the sampler need."""

    def __init__(self, kinds: list[str]) -> None:
        self.samples = [
            {"kind": kind, "label": int(kind == KIND_POSITIVE), "bucket": (8, 8)}
            for kind in kinds
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        # Positives are bright, negatives dark: the stub model reads brightness.
        value = 255 if sample["label"] == 1 else 0
        return torch.full((3, 8, 8), value, dtype=torch.uint8), sample["label"], (8, 8)


class _BrightnessModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        score = images.float().mean(dim=(1, 2, 3))
        return torch.stack([-score, score], dim=1)


def test_evaluate_returns_kinds_aligned_with_labels_under_shuffle() -> None:
    kinds = (
        [KIND_POSITIVE] * 5 + [KIND_EXPLICIT_NEGATIVE] * 3 + [KIND_IMPLIED_NEGATIVE] * 9
    )
    ds = _KindDataset(kinds)
    loader = DataLoader(
        ds,
        batch_sampler=BucketBatchSampler(ds, batch_size=4),
        collate_fn=_collate_bucket_batch,
        num_workers=0,
    )
    random.seed(7)
    result = evaluate(
        _BrightnessModel(),
        loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        torch.float32,
    )

    assert len(result["kinds"]) == len(result["labels"]) == len(kinds)
    assert sorted(result["kinds"]) == sorted(kinds)
    for kind, label in zip(result["kinds"], result["labels"], strict=True):
        assert (label == 1) == (kind == KIND_POSITIVE)
    # A second pass reshuffles; alignment must survive.
    again = evaluate(
        _BrightnessModel(),
        loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        torch.float32,
    )
    for kind, label in zip(again["kinds"], again["labels"], strict=True):
        assert (label == 1) == (kind == KIND_POSITIVE)


def test_evaluate_without_provenance_reports_no_kinds() -> None:
    x = torch.zeros(6, 3, 8, 8, dtype=torch.uint8)
    y = torch.tensor([1, 0, 1, 0, 1, 0])
    loader = DataLoader(list(zip(x, y, strict=True)), batch_size=3)
    result = evaluate(
        _BrightnessModel(),
        loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        torch.float32,
    )
    assert result["kinds"] is None
