from __future__ import annotations

from pathlib import Path

from PIL import Image

from bittrainer.dataset import ConceptDataset
from bittrainer.generic.tasks.binary_task import BinaryTask
from bittrainer.trainer import TrainConfig, _rebalance_val_negatives


def _image(path: Path, colour: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), colour).save(path)
    return path


def test_explicit_paths_assign_split_without_physical_split_folders(
    tmp_path: Path,
) -> None:
    concept_folder = tmp_path / "flat-concept"
    train_positive = _image(concept_folder / "train-positive.png", (255, 0, 0))
    val_positive = _image(concept_folder / "val-positive.png", (0, 255, 0))
    train_negative = _image(tmp_path / "elsewhere" / "train-negative.png", (0, 0, 255))
    val_negative = _image(tmp_path / "elsewhere" / "val-negative.png", (255, 255, 0))

    train = ConceptDataset(
        concept_folder,
        split="train",
        positive_paths=[str(train_positive)],
        negative_paths=[str(train_negative)],
    )
    val = ConceptDataset(
        concept_folder,
        split="val",
        positive_paths=[str(val_positive)],
        negative_paths=[str(val_negative)],
    )

    assert train._positive_paths == [train_positive]
    assert train._all_negative_paths == [train_negative]
    assert val._positive_paths == [val_positive]
    assert val._all_negative_paths == [val_negative]
    assert not (concept_folder / "train").exists()
    assert not (concept_folder / "val").exists()


def test_explicit_paths_drop_missing_and_duplicate_files(tmp_path: Path) -> None:
    concept_folder = tmp_path / "flat-concept"
    positive = _image(concept_folder / "positive.png", (255, 0, 0))

    dataset = ConceptDataset(
        concept_folder,
        positive_paths=[str(positive), str(positive), str(tmp_path / "missing.png")],
        negative_paths=[],
    )

    assert dataset._positive_paths == [positive]


def test_implied_negative_floor_is_additive_to_every_explicit_negative(
    tmp_path: Path,
) -> None:
    concept_folder = tmp_path / "flat-concept"
    positives = [
        _image(concept_folder / f"positive-{index}.png", (255, index, 0))
        for index in range(2)
    ]
    implied = [
        _image(tmp_path / "implied" / f"negative-{index}.png", (0, index, 255))
        for index in range(10)
    ]
    explicit = [
        _image(tmp_path / "explicit" / f"negative-{index}.png", (index, 0, 0))
        for index in range(2)
    ]

    dataset = ConceptDataset(
        concept_folder,
        positive_paths=[str(path) for path in positives],
        negative_paths=[str(path) for path in implied],
        hard_negative_paths=[str(path) for path in explicit],
        hard_negative_weight=3,
        neg_pos_ratio=1.0,
    )

    sampled_paths = [Path(sample["path"]) for sample in dataset.samples]
    implied_count = sum(path in implied for path in sampled_paths)
    assert implied_count == 6  # 2 positives x the enforced 3:1 implied ratio
    for path in explicit:
        assert sampled_paths.count(path) == 3
    assert len(dataset.samples) == 2 + 6 + 2 * 3


def test_binary_task_routes_each_explicit_split_to_its_dataset(tmp_path: Path) -> None:
    concept_folder = tmp_path / "flat-concept"
    train_positive = _image(concept_folder / "train-positive.png", (255, 0, 0))
    val_positive = _image(concept_folder / "val-positive.png", (0, 255, 0))
    train_negative = _image(tmp_path / "elsewhere" / "train-negative.png", (0, 0, 255))
    val_negative = _image(tmp_path / "elsewhere" / "val-negative.png", (255, 255, 0))

    task = BinaryTask(
        TrainConfig(
            concept_folder=str(concept_folder),
            use_cache=False,
            train_positive_paths=[str(train_positive)],
            val_positive_paths=[str(val_positive)],
            train_negative_paths=[str(train_negative)],
            val_negative_paths=[str(val_negative)],
            train_hard_negative_paths=[],
            val_hard_negative_paths=[],
        )
    )
    context = task.make_context(None, None, None, None)

    task.prepare_data(context)

    assert task.train_ds._positive_paths == [train_positive]
    assert task.train_ds._all_negative_paths == [train_negative]
    assert task.val_ds._positive_paths == [val_positive]
    assert task.val_ds._all_negative_paths == [val_negative]
    assert task.train_ds._neg_pos_ratio == 3.0
    assert task.val_ds._neg_pos_ratio == 3.0


def test_validation_rebalance_preserves_train_implied_floor(tmp_path: Path) -> None:
    train_positives = [
        _image(tmp_path / "train" / f"positive-{index}.png", (255, index, 0))
        for index in range(2)
    ]
    train_implied = [
        _image(tmp_path / "train-implied" / f"negative-{index}.png", (0, index, 255))
        for index in range(8)
    ]
    val_positives = [
        _image(tmp_path / "val" / f"positive-{index}.png", (index, 255, 0))
        for index in range(3)
    ]
    val_implied = [
        _image(tmp_path / "val-implied" / f"negative-{index}.png", (index, 0, 255))
        for index in range(2)
    ]
    train = ConceptDataset(
        tmp_path,
        split="train",
        positive_paths=[str(path) for path in train_positives],
        negative_paths=[str(path) for path in train_implied],
        neg_pos_ratio=2.0,
    )
    val = ConceptDataset(
        tmp_path,
        split="val",
        positive_paths=[str(path) for path in val_positives],
        negative_paths=[str(path) for path in val_implied],
        neg_pos_ratio=2.0,
    )

    _rebalance_val_negatives(train, val)

    # Val wants ceil(3 x 3.0) = 9 but the train pool may only donate down to its
    # own implied floor of ceil(2 x 3.0) = 6, so exactly 2 negatives move over.
    assert len(train._all_negative_paths) == 6
    assert len(val._all_negative_paths) == 4
    assert set(train._all_negative_paths).isdisjoint(val._all_negative_paths)
    assert all(sample["split"] == "val" for sample in val.samples)
