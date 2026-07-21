"""BinaryTask on the GenericTrainer skeleton (Bitcrush ISSUE-0542 Step 5).

Pins the binary concept trainer's externally-observable contract across the
migration of ``run_training`` onto ``GenericTrainer``:

(a) a tiny end-to-end CPU concept run returns the result keys Engine's
    process_manager splats into ``training_complete`` (checkpoint_path + the f1
    fields);
(b) ``TrainConfig`` / ``evaluate`` / ``run_training`` / ``train_one_epoch`` stay
    importable and callable from ``bittrainer.trainer`` (the Engine shim's four
    names);
(c) a worse candidate leaves the incumbent best.pt in place; and
(d) a pause at the epoch-0→1 unfreeze boundary resumes (epoch-restart) and
    completes.

CPU-only, ``atto`` backbone, caching off, ``dataloader_workers=0``.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image

from bittrainer.model import create_model


class _FlagEvent:
    def __init__(self) -> None:
        self._set = False

    def set(self) -> None:
        self._set = True

    def is_set(self) -> bool:
        return self._set


def _seed(n: int = 0) -> None:
    torch.manual_seed(n)
    np.random.seed(n)
    random.seed(n)


def _make_concept(root, *, n_pos, n_neg, n_val=6, seed=0):
    rng = np.random.default_rng(seed)
    for label, n_tr in (("positive", n_pos), ("negative", n_neg)):
        for split, n in (("train", n_tr), ("val", n_val)):
            d = root / label / split
            d.mkdir(parents=True, exist_ok=True)
            for j in range(n):
                Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
                    d / f"{label}_{j}.png"
                )


def _bin_cfg(concept, ckpt, **kw):
    from bittrainer.trainer import TrainConfig

    base = dict(
        concept_folder=str(concept), checkpoint_dir=str(ckpt), max_epochs=2, patience=99,
        model_size="atto", device="cpu", dtype="float32", use_cache=False, from_scratch=True,
        dataloader_workers=0, backbone_init={"source": "random_init", "checkpoint_path": None},
    )
    base.update(kw)
    return TrainConfig(**base)


def test_end_to_end_result_keys_process_manager_reads(tmp_path):
    from bittrainer.trainer import run_training

    _make_concept(tmp_path / "c", n_pos=8, n_neg=8, n_val=4)
    _seed(0)
    res = run_training(_bin_cfg(tmp_path / "c", tmp_path / "ck"))

    # process_manager splats the result into training_complete; these are the
    # keys the Engine side reads back off the wire.
    assert res["checkpoint_path"].endswith("best.pt")
    assert res["epochs_completed"] == 2
    assert isinstance(res["best_val_f1"], float)
    assert "final_val_f1" in res
    assert "optimal_threshold" in res
    assert res["positive_count"] >= 1
    assert "paused" not in res


def test_public_trainer_names_importable_and_callable():
    from bittrainer.trainer import TrainConfig, evaluate, run_training, train_one_epoch

    assert callable(evaluate)
    assert callable(run_training)
    assert callable(train_one_epoch)
    # TrainConfig constructs with just the required field
    cfg = TrainConfig(concept_folder="x")
    assert cfg.max_epochs >= 1


def test_worse_candidate_keeps_incumbent(tmp_path, monkeypatch):
    """A candidate that scores below the incumbent leaves best.pt untouched.

    The candidate vs. incumbent decision is fair-compared on the current val set
    inside ``_binary_compare_promote``; script ``_tuned_val_metrics`` (the shared
    F1-at-tuned-threshold seam both call sites use) so the two epoch candidates
    score low and the incumbent's fair-comparison pass scores high.
    """
    import bittrainer.trainer as bt
    from bittrainer.trainer import run_training

    _make_concept(tmp_path / "c", n_pos=8, n_neg=8, n_val=4)
    ckpt = tmp_path / "ck"
    ckpt.mkdir()

    # Seed a real (loadable) incumbent best.pt.
    incumbent = create_model(model_size="atto", pretrained=False)
    best_pt = ckpt / "best.pt"
    torch.save(
        {"state_dict": incumbent.state_dict(), "num_classes": 2, "model_size": "atto"},
        best_pt,
    )
    before = best_pt.read_bytes()

    _low = {"f1": 0.10, "precision": 0.1, "recall": 0.1, "auprc": 0.1,
            "confusion_matrix": [[1, 0], [0, 1]]}
    _high = {"f1": 0.95, "precision": 0.9, "recall": 0.9, "auprc": 0.9,
             "confusion_matrix": [[2, 0], [0, 2]]}
    calls = {"n": 0}

    def _fake_tuned(val_result):
        calls["n"] += 1
        # first two calls = the two training epochs' candidate; the later call =
        # the incumbent's fair-comparison pass.
        return (dict(_low), 0.5) if calls["n"] <= 2 else (dict(_high), 0.5)

    monkeypatch.setattr(bt, "_tuned_val_metrics", _fake_tuned)

    _seed(0)
    res = run_training(_bin_cfg(tmp_path / "c", ckpt))

    # incumbent kept: best.pt bytes unchanged, its score returned, candidate gone
    assert res["checkpoint_path"].endswith("best.pt")
    assert best_pt.read_bytes() == before
    assert res["best_val_f1"] == 0.95
    assert not (ckpt / "candidate.pt").exists()


def test_pause_resume_across_unfreeze_boundary_completes(tmp_path):
    """Pausing at the epoch-0→1 boundary (the non-gradual unfreeze) and resuming
    replays the reconstruction and completes with no crash."""
    from bittrainer.trainer import run_training

    _make_concept(tmp_path / "c", n_pos=55, n_neg=55, n_val=10)  # >=50 -> non-gradual
    backups = str(tmp_path / "ck" / "backups")

    pause = _FlagEvent()

    def _cb(msg):
        if msg.get("type") == "epoch_complete" and msg.get("epoch") == 1:
            pause.set()

    _seed(0)
    first = run_training(
        _bin_cfg(tmp_path / "c", tmp_path / "ck", max_epochs=3, backup_dir=backups),
        progress_callback=_cb, pause_event=pause,
    )
    assert first["paused"] is True
    assert first["epoch"] == 1

    _seed(0)
    resumed = run_training(
        _bin_cfg(tmp_path / "c", tmp_path / "ck", max_epochs=3,
                 backup_dir=backups, resume_from=backups),
    )
    assert "paused" not in resumed
    assert resumed["checkpoint_path"].endswith("best.pt")
    assert resumed["epochs_completed"] == 3
