"""What a backbone candidate records about its own run (Bitcrush ISSUE-1655).

* ``training_actual_json`` in the candidate's metadata holds the values the run
  really used. ``training_config_json`` only holds what was passed in, so a
  default that took effect (the validation split, say) was nowhere on record.
* The winning epoch's raw validation scores (logits and targets for every
  yes/no head and every group) are written beside the candidate as
  ``<candidate>.val_scores.npz``, and named in the metadata and the result.
  Thresholds can then be worked out again later, under any rule, without the
  images or a GPU. A run with no validation images writes no file.

CPU-only, on a stand-in trunk of one small convolution (``backbone_stub``):
48px generated images, AMP off, a whole run in a fraction of a second.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from safetensors import safe_open

import bittrainer.backbone_trainer as bb
import bittrainer.generic.tasks.backbone_task as bt
from bittrainer.backbone_trainer import run_backbone_training
from bittrainer.generic.task import BestTracker
from bittrainer.generic.tasks.backbone_task import load_val_scores, save_val_scores
from bittrainer.tests.backbone_stub import use_stub_trunk


@pytest.fixture(autouse=True)
def _trunk(monkeypatch):
    use_stub_trunk(monkeypatch)


def _request(tmp_path, *, n=16, config=None):
    root = tmp_path / "imgs"
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(n):
        path = root / f"img{i}.png"
        Image.new(
            "RGB", (72, 64), color=((i * 37) % 255, (i * 83) % 255, (i * 151) % 255)
        ).save(path)
        records.append(
            {
                "content_hash": f"{i:02d}" + "ab" * 31,
                "file_paths": [str(path)],
                "binary": {"watermark": "positive" if i % 2 else "negative"},
                "groups": {"shot_type": "closeup" if i % 2 else "wide"},
                "splits": {"train": 1},
            }
        )
    training_config = {
        "image_size": 48,
        "batch_size": 4,
        "epochs": 2,
        "device": "cpu",
        "use_amp": False,
        "oversample_positives": False,
    }
    training_config.update(config or {})
    return {
        "run_id": "run_record",
        "convnextv2_size": "atto",
        "candidate_checkpoint_path": str(
            tmp_path / "candidates" / "candidate.safetensors"
        ),
        "records": records,
        "training_config": training_config,
        "backbone_init": {"source": "random_init", "checkpoint_path": None},
    }


def _metadata(path):
    with safe_open(str(path), framework="pt") as handle:
        return dict(handle.metadata())


def _spy_val_scores(monkeypatch):
    """Every validation pass's raw scores, in the order they were collected."""
    collected = []
    real = bb._collect_val_scores

    def _collect(*args, **kwargs):
        binary, groups = real(*args, **kwargs)
        collected.append({"binary": binary, "groups": groups})
        return binary, groups

    monkeypatch.setattr(bb, "_collect_val_scores", _collect)
    return collected


def _same_scores(a, b):
    for family in ("binary", "groups"):
        assert list(a[family]) == list(b[family])
        for name, (logits, targets) in a[family].items():
            other_logits, other_targets = b[family][name]
            assert logits.dtype == other_logits.dtype
            assert targets.dtype == other_targets.dtype
            assert np.array_equal(logits, other_logits), name
            assert np.array_equal(targets, other_targets), name
    return True


# -- the validation scores file -----------------------------------------------


def test_scores_round_trip_to_the_same_arrays_whatever_the_heads_are_called(tmp_path):
    rng = np.random.default_rng(0)
    odd_names = [
        "plain",
        "Carmen B. Sanchez",
        'said "hi" / back\\slash',
        "tab\there, newline\nthere",
        " spaces at both ends ",
        "naïve — ünïcöde",
        "nul" + chr(0),
        "",
    ]
    scores = {
        "binary": {
            name: (rng.normal(size=3 + i), (rng.random(3 + i) > 0.5).astype(np.float64))
            for i, name in enumerate(odd_names)
        },
        "groups": {
            "shot.type": (rng.normal(size=(5, 3)), rng.integers(0, 3, size=5)),
            "a/b": (rng.normal(size=(2, 7)), rng.integers(0, 7, size=2)),
        },
    }
    path = tmp_path / "candidate.val_scores.npz"

    save_val_scores(path, scores)

    assert _same_scores(load_val_scores(path), scores)
    # Readable with numpy alone, no pickle: names in a list, heads by position.
    with np.load(path, allow_pickle=False) as data:
        assert json.loads(str(data["binary_names_json"])) == odd_names
        assert json.loads(str(data["group_names_json"])) == ["shot.type", "a/b"]
        assert len(data["binary_offsets"]) == len(odd_names) + 1
        assert data["group_1_logits"].shape == (2, 7)


def test_a_file_with_only_one_family_round_trips(tmp_path):
    only_groups = {"binary": {}, "groups": {"g": (np.zeros((2, 2)), np.array([0, 1]))}}
    only_binary = {"binary": {"c": (np.ones(4), np.zeros(4))}, "groups": {}}
    for name, scores in (("g.npz", only_groups), ("b.npz", only_binary)):
        save_val_scores(tmp_path / name, scores)
        assert _same_scores(load_val_scores(tmp_path / name), scores)


# -- what the run records -----------------------------------------------------


def test_the_candidate_says_how_the_run_was_really_done(tmp_path, monkeypatch):
    collected = _spy_val_scores(monkeypatch)
    frames: list[dict] = []
    request = _request(
        tmp_path,
        config={
            "finetune_image_size": 64,
            "finetune_epochs": 1,
            "finetune_batch_size": 2,
        },
    )
    assert "validation_split" not in request["training_config"]

    result = asyncio.run(
        run_backbone_training(request, progress_callback=frames.append)
    )

    candidate = Path(result["candidate_checkpoint_path"])
    metadata = _metadata(candidate)
    actual = json.loads(metadata["training_actual_json"])
    preparing = next(f for f in frames if "train_samples" in f)
    last_epoch = [f for f in frames if f.get("stage") == "training" and "loss" in f][-1]
    assert actual == {
        "validation_split": 0.15,  # the default: nowhere in what was passed in
        "train_images": preparing["train_samples"],
        "val_images": preparing["val_samples"],
        "image_size": 48,
        "batch_size": 4,
        "finetune_image_size": 64,
        "finetune_epochs": 1,
        "finetune_batch_size": 2,
        # The winning epoch was validated at the tail's size: shorter side
        # resized to it, then a centre crop of it.
        "val_resize": 64,
        "val_center_crop": 64,
        "amp_dtype": "off",
        "grad_scaler": False,
        "lean_grn": True,
        "optimizer_steps": last_epoch["steps"],
        "skipped_steps": 0,
        "best_epoch": 2,
    }
    assert actual["val_images"] == 3 and result["best_epoch"] == 2
    # What was passed in is still recorded as it was.
    assert json.loads(metadata["training_config_json"]) == request["training_config"]

    # The winning epoch's scores (the tail's: the second validation pass).
    assert result["val_scores_file"] == "candidate.val_scores.npz"
    assert metadata["val_scores_file"] == "candidate.val_scores.npz"
    saved = load_val_scores(candidate.with_name(result["val_scores_file"]))
    assert len(collected) == 2
    assert _same_scores(saved, collected[1])
    assert list(saved["binary"]) == ["watermark"]
    assert saved["groups"]["shot_type"][0].shape == (3, 2)
    saving = next(f for f in frames if f.get("stage") == "saving")
    assert saving["val_scores_file"] == "candidate.val_scores.npz"


def test_a_single_resolution_run_records_no_tail(tmp_path):
    result = asyncio.run(run_backbone_training(_request(tmp_path)))

    actual = json.loads(
        _metadata(result["candidate_checkpoint_path"])["training_actual_json"]
    )
    assert actual["finetune_image_size"] is None
    assert actual["finetune_batch_size"] is None
    assert actual["val_resize"] == 48 and actual["val_center_crop"] == 48


def test_a_resumed_run_still_writes_the_winning_epochs_scores(tmp_path, monkeypatch):
    """Epoch 1 wins, the run is paused and resumed, epoch 2 does not win. The
    scores written must be epoch 1's, which only the backup still has."""
    collected = _spy_val_scores(monkeypatch)
    selections = [0.9, 0.1]
    real_metrics = bb._backbone_metrics

    def _metrics(binary, groups, **kwargs):
        return {
            **real_metrics(binary, groups, **kwargs),
            "selection": selections.pop(0),
        }

    monkeypatch.setattr(bb, "_backbone_metrics", _metrics)
    request = _request(tmp_path, config={"backup_dir": str(tmp_path / "backups")})
    pause = threading.Event()

    def _pause_after_epoch_one(frame):
        if frame.get("stage") == "training" and "loss" in frame:
            pause.set()

    paused = bb._train_backbone(
        request, _pause_after_epoch_one, threading.Event(), pause_event=pause
    )
    assert paused["paused"] is True and len(collected) == 1

    resumed = {
        **request,
        "training_config": {
            **request["training_config"],
            "resume_from": paused["backup_path"],
        },
    }
    result = bb._train_backbone(resumed, lambda _frame: None, threading.Event())

    assert result["best_epoch"] == 1 and len(collected) == 2
    candidate = Path(result["candidate_checkpoint_path"])
    saved = load_val_scores(candidate.with_name(result["val_scores_file"]))
    assert _same_scores(saved, collected[0])
    assert not np.array_equal(
        saved["binary"]["watermark"][0], collected[1]["binary"]["watermark"][0]
    )
    # The thresholds fitted from those scores came back with them.
    assert "binary_thresholds_json" in _metadata(candidate)


def test_a_run_with_no_validation_images_writes_no_file_and_says_so(tmp_path):
    # A split too small to catch any image (0 itself means "the default").
    request = _request(tmp_path, config={"validation_split": 1e-9, "epochs": 1})
    stale = tmp_path / "candidates" / "candidate.val_scores.npz"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"an earlier run's scores")
    frames: list[dict] = []

    result = asyncio.run(
        run_backbone_training(request, progress_callback=frames.append)
    )

    assert result["val_scores_file"] is None
    assert not stale.exists()  # never an older run's scores beside a new candidate
    assert "val_scores_file" not in _metadata(result["candidate_checkpoint_path"])
    saving = next(f for f in frames if f.get("stage") == "saving")
    assert saving["val_scores_file"] is None
    assert "no validation scores" in saving["status_text"]


def test_head_only_retraining_keeps_its_metadata_as_it_was(tmp_path):
    """It shares ``_candidate_metadata`` but not the trunk loop these facts
    describe, so it does not claim them."""
    from bittrainer.generic.tasks.backbone_heads_task import BackboneHeadsTask

    request = _request(tmp_path)
    request["backbone_init"] = {"source": "timm_pretrained"}
    task = BackboneHeadsTask(request)
    task.vocab = bb._Vocab(request["records"])

    metadata = task._candidate_metadata(0.5, {"selection": 0.5})

    assert "training_actual_json" not in metadata
    assert "val_scores_file" not in metadata
    assert bt.BackboneTask.trainer_name != BackboneHeadsTask.trainer_name

    # Its own loop never counts skipped steps; the epoch frame it inherits
    # still reports, with nothing skipped and its own batch size.
    frames: list[dict] = []
    task._cb = frames.append
    task.epoch_message(None, 0, {}, 0.5, 0.0, BestTracker())
    (epoch,) = frames
    assert epoch["skipped_steps"] == 0 and epoch["grad_scale"] is None
    assert epoch["batch_size"] == 256 and epoch["image_size"] == 48


# -- what a reader needs to judge each head ------------------------------------


def _run_metadata(request):
    result = asyncio.run(run_backbone_training(request))
    return _metadata(Path(result["candidate_checkpoint_path"]))


def test_the_candidate_reports_every_head_from_the_winning_epoch(tmp_path):
    metadata = _run_metadata(_request(tmp_path, config={"validation_split": 0.5}))
    val_images = json.loads(metadata["training_actual_json"])["val_images"]

    report = json.loads(metadata["binary_reports_json"])["watermark"]
    assert report["yes"] + report["no"] == val_images
    assert report["threshold"] == 0.5
    # A handful of validation labels is too few to tune a threshold on: the head
    # keeps 0.5 and the candidate says it was not tuned.
    assert json.loads(metadata["binary_thresholds_tuned_json"]) == {"watermark": False}
    assert json.loads(metadata["binary_thresholds_json"]) == {"watermark": 0.5}

    group = json.loads(metadata["group_reports_json"])["shot_type"]
    assert sorted(group["classes"]) == ["closeup", "wide"]
    assert np.asarray(group["confusion"]).shape == (2, 2)
    assert (
        int(np.asarray(group["confusion"]).sum()) == sum(group["support"]) == val_images
    )
    assert set(json.loads(metadata["group_temperatures_json"])) == {"shot_type"}


def test_how_many_validation_labels_a_threshold_needs_is_a_setting(tmp_path):
    metadata = _run_metadata(
        _request(
            tmp_path, config={"validation_split": 0.5, "calibration_min_labels": 1}
        )
    )

    assert json.loads(metadata["binary_thresholds_tuned_json"]) == {"watermark": True}


@pytest.mark.parametrize(
    ("listed", "hflip", "expected"),
    [
        (None, False, "false"),  # nobody said which heads are geometry-sensitive
        ([], False, "true"),  # looked, none
        (["shot_type"], False, "true"),
        ([], True, "false"),  # mirroring turned back on
    ],
)
def test_the_candidate_says_whether_geometry_was_protected(
    tmp_path, listed, hflip, expected
):
    request = _request(tmp_path, config={"epochs": 1, "hflip": hflip})
    if listed is not None:
        request["geometry_sensitive"] = listed

    assert _run_metadata(request)["geometry_safe_augmentation"] == expected
