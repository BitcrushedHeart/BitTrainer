"""A paused backbone run can come back at another resolution (Bitcrush ISSUE-1721).

Owner, 2026-11-13: BitVision Large started at 224px with a 384px tail; it is to
go on at 384px with a 512px tail, without losing what it has trained.

* The image sizes (main and tail) are not part of a backup's identity: the
  learning-rate schedule runs over epochs, not pixels, and the weights, the
  optimizer, the EMA and the scaler do not depend on the image size. The epoch
  count and the tail's length still are.
* A backup written before this change (its fingerprint carries a
  ``resolution`` key) still matches a run at new sizes.
* When the epoch a run resumes into is at a different size than the backup's
  plan had it, the best score so far restarts: validation scores at two sizes
  are not comparable (the same rule as the switch into the tail), so the
  low-resolution epoch can never be the exported candidate.

CPU-only, on the stand-in trunk the tail tests use.
"""

from __future__ import annotations

import threading

import bittrainer.backbone_trainer as bb
from bittrainer.generic.tasks.backbone_task import BackboneTask
from bittrainer.tests.test_backbone_tail_batch import (  # noqa: F401 (_trunk: autouse)
    _fingerprint,
    _pause_when,
    _request,
    _resume,
    _spy_loaders,
    _trunk,
)
from bittrainer.training_state import fingerprint_matches


def _three_epochs(tmp_path, **config):
    """Two epochs at 48px (batch 4), then a 64px tail epoch (batch 2)."""
    return _request(
        tmp_path,
        config={
            "epochs": 3,
            "finetune_epochs": 1,
            "backup_dir": str(tmp_path / "backups"),
            **config,
        },
    )


def test_image_sizes_are_not_part_of_a_backups_identity(tmp_path):
    base = _fingerprint(BackboneTask(_three_epochs(tmp_path)))
    bigger = _fingerprint(
        BackboneTask(
            _three_epochs(tmp_path, image_size=56, finetune_image_size=80, batch_size=2)
        )
    )
    longer = _fingerprint(BackboneTask(_three_epochs(tmp_path, epochs=4)))

    assert fingerprint_matches(base, bigger) and fingerprint_matches(bigger, base)
    assert not fingerprint_matches(base, longer)  # the epoch count still is


def test_a_backup_written_before_this_change_still_matches(tmp_path):
    current = _fingerprint(BackboneTask(_three_epochs(tmp_path, image_size=56)))
    older = {
        **_fingerprint(BackboneTask(_three_epochs(tmp_path))),
        "resolution": "48->64@1",
    }

    assert fingerprint_matches(older, current)


def test_a_paused_run_resumes_at_a_new_resolution_and_its_best_restarts(
    tmp_path, monkeypatch
):
    request = _three_epochs(tmp_path)
    pause, callback, _frames = _pause_when(
        lambda f: f.get("stage") == "training" and "loss" in f and f["epoch"] == 1
    )
    paused = bb._train_backbone(request, callback, threading.Event(), pause_event=pause)
    assert paused["paused"] is True and paused["epoch"] == 1

    built = _spy_loaders(monkeypatch)
    result, frames = _resume(
        request, paused, image_size=56, finetune_image_size=80, finetune_batch_size=2
    )

    assert not any(f.get("type") == "resume_skipped" for f in frames)
    resumed = next(f for f in frames if f.get("type") == "training_resumed")
    assert (resumed["epoch"], resumed["image_size"]) == (1, 56)
    assert ("train", 56, 4) in built and ("train", 80, 2) in built
    assert ("train", 48, 4) not in built
    changed = [f for f in frames if f.get("resolution_changed")]
    assert len(changed) == 1
    assert (changed[0]["from_size"], changed[0]["to_size"]) == (48, 56)
    assert result["epochs_completed"] == 3
    assert result["best_epoch"] >= 2  # the 48px epoch never wins


def test_a_resume_at_the_same_sizes_keeps_its_best(tmp_path):
    request = _three_epochs(tmp_path)
    pause, callback, _frames = _pause_when(
        lambda f: f.get("stage") == "training" and "loss" in f and f["epoch"] == 1
    )
    paused = bb._train_backbone(request, callback, threading.Event(), pause_event=pause)

    _result, frames = _resume(request, paused)

    assert not [f for f in frames if f.get("resolution_changed")]
    resumed = next(f for f in frames if f.get("type") == "training_resumed")
    assert resumed["image_size"] == 48
