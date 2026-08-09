from __future__ import annotations

from pathlib import Path

from PIL import Image

from bittrainer.dataset import ConceptDataset
from bittrainer.generic.tasks.binary_task import BinaryTask
from bittrainer.trainer import TrainConfig


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
