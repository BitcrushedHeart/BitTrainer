"""Generic training-lifecycle core (Bitcrush ISSUE-0542 Step 3).

Drives ``GenericTrainer.run`` with a minimal in-memory spy ``TrainingTask`` (a
1-layer ``nn.Linear`` so the backup path has a real ``state_dict`` to serialise,
but no data pipeline) to pin down the lifecycle order and the pause / stop /
best / patience mechanics the core owns — independent of any group specifics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from bittrainer.generic.generic_trainer import GenericTrainer
from bittrainer.generic.task import LoopSpec, TaskContext, TrainingTask
from bittrainer.training_state import BackupCoordinator, make_fingerprint

_DEVICE = torch.device("cpu")


class _FlagEvent:
    def __init__(self) -> None:
        self._set = False

    def set(self) -> None:
        self._set = True

    def is_set(self) -> bool:
        return self._set


class _DummyEmitter:
    """Stand-in for ProgressEmitter — the core only calls ``.stage``."""

    def __init__(self, cb) -> None:
        self.raw = cb

    def stage(self, *_a, **_k) -> None:
        pass


class SpyTask(TrainingTask):
    """Records hook invocation order and scripts the selection scores."""

    trainer_name = "spy"

    def __init__(self, tmp_path, *, scores, max_epochs, patience=99, backup=True) -> None:
        self.tmp_path = tmp_path
        self.scores = scores
        self.max_epochs = max_epochs
        self.patience = patience
        self.backup = backup
        self.calls: list[str] = []
        self.msgs: list[dict] = []
        self._val_loader = None

    # -- setup -------------------------------------------------------------
    def make_context(self, progress_callback, stop_event, stop_now_event, pause_event) -> TaskContext:
        self.calls.append("make_context")
        cb = progress_callback or self.msgs.append
        ckpt = self.tmp_path / "ckpt"
        ckpt.mkdir(parents=True, exist_ok=True)
        return TaskContext(
            device=_DEVICE,
            dtype=torch.float32,
            em=_DummyEmitter(cb),
            cb=cb,
            checkpoint_dir=ckpt,
            stop_event=stop_event,
            stop_now_event=stop_now_event,
            pause_event=pause_event,
        )

    def fingerprint_init(self, ctx) -> None:
        self.calls.append("fingerprint_init")
        ctx.coordinator = BackupCoordinator(
            backup_dir=(self.tmp_path / "backups") if self.backup else None,
            pause_event=ctx.pause_event,
            cb=ctx.cb,
        )
        ctx.fingerprint = make_fingerprint(
            class_names=["a", "b"], num_classes=2, max_epochs=self.max_epochs,
            multi_label=False, ordinal=False, best_model_name="best.pt", model_size="atto",
        )
        ctx.resume_state = None

    def loop_spec(self) -> LoopSpec:
        return LoopSpec(max_epochs=self.max_epochs, patience=self.patience)

    def prepare_data(self, ctx) -> None:
        self.calls.append("prepare_data")

    def create_model(self, ctx, resume_state):
        self.calls.append("create_model")
        return nn.Linear(2, 2)

    def pre_loop(self, ctx, model) -> None:
        self.calls.append("pre_loop")

    def resolve_batch_size(self, ctx, model, resume_state) -> int:
        self.calls.append("resolve_batch_size")
        return 2

    def setup_training(self, ctx, model, resume_state) -> None:
        self.calls.append("setup_training")

    def create_optimizer(self, ctx, model, eff_bs, resume_state):
        self.calls.append("create_optimizer")
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = CosineAnnealingLR(opt, T_max=self.max_epochs)
        return opt, sched, self.max_epochs

    # -- epoch loop --------------------------------------------------------
    def reshuffle(self) -> None:
        self.calls.append("reshuffle")

    def build_loaders(self, ctx, epoch, eff_bs, resume_info):
        self.calls.append("build_loaders")
        self._val_loader = object()
        return [], [[0]], 0

    def train_epoch(self, ctx, model, optimizer, train_loader, *, step_callback, boundary_hook, start_batch):
        self.calls.append("train_epoch")
        return f"trainresult{self._epoch}"

    def on_after_train(self, ctx, model, epoch) -> None:
        self.calls.append("on_after_train")

    def validate(self, ctx, model, epoch, train_result) -> dict:
        self.calls.append("validate")
        self._epoch = epoch
        return {"selected": self.scores[epoch], "macro_f1": self.scores[epoch]}

    def selection_score(self, metrics) -> float:
        return float(metrics["selected"])

    def save_candidate(self, ctx, model, epoch, metrics, best) -> None:
        self.calls.append("save_candidate")
        best.best_val_macro_f1 = metrics["macro_f1"]
        best.best_metrics = dict(metrics)
        best.best_checkpoint_path = f"cand{epoch}"

    def on_epoch_end(self, ctx, model, epoch, metrics, selected_score, best) -> None:
        self.calls.append("on_epoch_end")

    def epoch_message(self, ctx, epoch, metrics, train_result, selected_score, best) -> dict:
        return {"type": "epoch_complete", "epoch": epoch + 1}

    def finalize(self, ctx, model, best, epochs_completed) -> dict:
        self.calls.append("finalize")
        return {
            "epochs_completed": epochs_completed,
            "best_epoch": best.best_epoch + 1,
            "best_val_macro_f1": best.best_val_macro_f1,
            "checkpoint_path": best.best_checkpoint_path,
        }

    # spy bookkeeping
    _epoch = 0


def test_lifecycle_order_two_epochs(tmp_path):
    task = SpyTask(tmp_path, scores=[0.5, 0.7], max_epochs=2)
    result = GenericTrainer().run(task)
    assert task.calls == [
        "make_context", "fingerprint_init",
        "prepare_data", "create_model", "pre_loop",
        "resolve_batch_size", "setup_training", "create_optimizer",
        # epoch 0
        "reshuffle", "build_loaders", "train_epoch", "on_after_train",
        "validate", "save_candidate", "on_epoch_end",
        # epoch 1
        "reshuffle", "build_loaders", "train_epoch", "on_after_train",
        "validate", "save_candidate", "on_epoch_end",
        "finalize",
    ]
    assert result["epochs_completed"] == 2
    assert result["best_epoch"] == 2  # epoch index 1 -> display 2


def test_graceful_stop_at_epoch_boundary_still_finalizes(tmp_path):
    stop = _FlagEvent()
    task = SpyTask(tmp_path, scores=[0.5, 0.7, 0.9], max_epochs=3)

    # Flip the graceful-stop event once epoch 0 completes.
    def _cb(msg):
        task.msgs.append(msg)
        if msg.get("type") == "epoch_complete" and msg.get("epoch") == 1:
            stop.set()

    result = GenericTrainer().run(task, progress_callback=_cb, stop_event=stop)
    assert task.calls.count("train_epoch") == 1  # only epoch 0 trained
    assert "finalize" in task.calls
    # Broke at the top of epoch index 1; epochs_completed = last index + 1 (as before).
    assert result["epochs_completed"] == 2


def test_stop_now_breaks_and_still_finalizes(tmp_path):
    stop_now = _FlagEvent()
    stop_now.set()  # set before the run — first epoch's top-of-loop check breaks
    task = SpyTask(tmp_path, scores=[0.5, 0.7], max_epochs=2)
    result = GenericTrainer().run(task, stop_now_event=stop_now)
    assert task.calls.count("train_epoch") == 0  # broke before training any epoch
    assert "finalize" in task.calls
    # Broke at the top of epoch index 0; epochs_completed = index + 1 (as before).
    assert result["epochs_completed"] == 1


def test_pause_returns_paused_without_finalize(tmp_path):
    pause = _FlagEvent()
    task = SpyTask(tmp_path, scores=[0.5, 0.7, 0.9], max_epochs=3, backup=True)

    def _cb(msg):
        task.msgs.append(msg)
        if msg.get("type") == "epoch_complete" and msg.get("epoch") == 1:
            pause.set()

    result = GenericTrainer().run(task, progress_callback=_cb, pause_event=pause)
    assert result["paused"] is True
    assert result["backup_path"] and "backup_" in result["backup_path"]
    assert "finalize" not in task.calls  # a pause never finalizes/promotes
    assert result["epoch"] == 1  # paused entering epoch index 1


def test_best_and_patience_early_stop(tmp_path):
    # improve, improve, plateau, plateau -> patience 2 stops after epoch index 3.
    task = SpyTask(tmp_path, scores=[0.5, 0.6, 0.55, 0.56, 0.9], max_epochs=5, patience=2)
    result = GenericTrainer().run(task)
    assert result["epochs_completed"] == 4  # ran epochs 0..3
    assert result["best_epoch"] == 2  # best was epoch index 1 -> display 2
    assert abs(result["best_val_macro_f1"] - 0.6) < 1e-9
    assert task.calls.count("train_epoch") == 4
