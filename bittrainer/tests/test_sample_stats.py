"""Per-sample loss / probability export (Bitcrush ISSUE-0850).

The trainer already computes per-example CE under ``no_grad`` inside
``_train_one_epoch`` and immediately reduces it to per-CLASS aggregates. These
tests pin the new seam that keeps the per-IMAGE values: a ``sample_loss_sink``
callback on the train loop, a recorder that keys them by path per epoch, and a
val loader whose batch order is materialised so val probabilities can be mapped
back to paths.
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bittrainer.collate import _collate_bucket_batch
from bittrainer.group_trainer import GroupTrainConfig, _train_one_epoch
from bittrainer.sample_stats import SampleStatsRecorder
from bittrainer.training_state import _FixedBatchSampler

_DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class TestRecorder:
    def test_train_losses_keyed_by_path_per_epoch(self):
        rec = SampleStatsRecorder(class_names=["a", "b"])
        rec.record_train(0, ["p1", "p2"], [0, 1], [0.5, 1.5])
        rec.record_train(1, ["p2", "p1"], [1, 0], [0.25, 0.75])
        rows = {r["path"]: r for r in rec.rows()}
        assert rows["p1"]["train_loss"] == [0.5, 0.75]
        assert rows["p2"]["train_loss"] == [1.5, 0.25]
        assert rows["p1"]["label"] == 0
        assert rows["p1"]["split"] == "train"

    def test_duplicate_path_in_one_epoch_is_averaged(self):
        # Oversampling replicates a path within an epoch: one loss per epoch,
        # averaged, never a list that grows with replication.
        rec = SampleStatsRecorder(class_names=["a"])
        rec.record_train(0, ["p", "p"], [0, 0], [1.0, 3.0])
        (row,) = rec.rows()
        assert row["train_loss"] == [2.0]

    def test_missing_epoch_is_none_not_skipped(self):
        # A path absent from an epoch (mixup batch, dropped sample) keeps its
        # column position so the trajectory stays aligned with epoch index.
        rec = SampleStatsRecorder(class_names=["a"])
        rec.record_train(0, ["p"], [0], [1.0])
        rec.record_train(2, ["p"], [0], [0.5])
        (row,) = rec.rows()
        assert row["train_loss"] == [1.0, None, 0.5]

    def test_val_probs_and_loss_recorded_per_epoch(self):
        rec = SampleStatsRecorder(class_names=["a", "b"])
        rec.record_val(0, ["v1"], [1], torch.tensor([[0.2, 0.8]]))
        rec.record_val(1, ["v1"], [1], torch.tensor([[0.6, 0.4]]))
        (row,) = rec.rows()
        assert row["split"] == "val"
        assert row["val_probs"][0] == [0.2, 0.8]
        assert row["val_probs"][1] == [0.6, 0.4]
        # loss = -log p(label)
        assert abs(row["val_loss"][0] - (-torch.log(torch.tensor(0.8))).item()) < 1e-4
        assert abs(row["val_loss"][1] - (-torch.log(torch.tensor(0.4))).item()) < 1e-4

    def test_write_produces_versioned_json(self, tmp_path):
        rec = SampleStatsRecorder(class_names=["a", "b"])
        rec.record_train(0, ["p"], [1], [0.3])
        rec.record_val(0, ["v"], [0], torch.tensor([[0.9, 0.1]]))
        out = rec.write(tmp_path / "sample_stats.json")
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert data["class_names"] == ["a", "b"]
        assert data["epochs"] == 1
        by_path = {r["path"]: r for r in data["samples"]}
        assert by_path["p"]["split"] == "train"
        assert by_path["v"]["split"] == "val"
        assert by_path["v"]["val_probs"] == [[0.9, 0.1]]

    def test_write_is_idempotent_overwrite(self, tmp_path):
        rec = SampleStatsRecorder(class_names=["a"])
        rec.record_train(0, ["p"], [0], [0.3])
        rec.write(tmp_path / "s.json")
        rec.record_train(1, ["p"], [0], [0.2])
        rec.write(tmp_path / "s.json")
        data = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
        assert data["epochs"] == 2
        assert data["samples"][0]["train_loss"] == [0.3, 0.2]


# ---------------------------------------------------------------------------
# Train-loop seam
# ---------------------------------------------------------------------------


class _TinyDS(torch.utils.data.Dataset):
    def __init__(self, n: int, num_classes: int):
        torch.manual_seed(0)
        self.x = torch.randn(n, 3, 8, 8)
        self.y = torch.randint(0, num_classes, (n,))
        self.samples = [
            {"path": f"img{i}.png", "label": int(self.y[i]), "bucket": (8, 8)} for i in range(n)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i]), (8, 8)


class _TinyModel(nn.Module):
    """Input-invariant logits (a learned bias): the loop's own augmentation and
    normalisation run before the forward, so an exact per-example reference is
    only possible when the output does not depend on pixels. What matters here
    is ORDER and LABEL alignment of the recorded losses, which this pins."""

    def __init__(self, num_classes: int):
        super().__init__()
        torch.manual_seed(1)
        self.bias = nn.Parameter(torch.randn(num_classes))

    def forward(self, x):
        return self.bias.unsqueeze(0).expand(x.shape[0], -1) + 0.0 * x.flatten(1).sum(1, keepdim=True)


def _cfg(**kw) -> GroupTrainConfig:
    base = dict(
        group_folder="/tmp/grp",
        num_classes=3,
        class_names=["a", "b", "c"],
        randaugment_n=0,
        random_erasing_p=0.0,
        aug_noise_p=0.0,
        aug_blur_p=0.0,
        aug_jpeg_p=0.0,
        use_mixup=False,
        channels_last=False,
    )
    base.update(kw)
    return GroupTrainConfig(**base)


class TestTrainLoopSink:
    def test_sink_receives_absolute_batch_index_and_per_example_losses(self):
        ds = _TinyDS(n=7, num_classes=3)
        schedule = [[0, 1, 2], [3, 4], [5, 6]]
        loader = DataLoader(
            ds, batch_sampler=_FixedBatchSampler(schedule), collate_fn=_collate_bucket_batch
        )
        model = _TinyModel(3)
        opt = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0: weights frozen
        config = _cfg()
        calls: list[tuple[int, torch.Tensor]] = []

        def sink(batch_index: int, losses: torch.Tensor) -> None:
            calls.append((batch_index, losses.clone()))

        _train_one_epoch(
            model, loader, opt, config, _DEVICE, torch.float32, sample_loss_sink=sink
        )

        assert [c[0] for c in calls] == [0, 1, 2]
        assert [c[1].numel() for c in calls] == [3, 2, 2]
        # lr=0 and no augmentation: the recorded losses equal a fresh forward.
        with torch.no_grad():
            ref = nn.functional.cross_entropy(
                model.bias.unsqueeze(0).expand(3, -1), ds.y[[0, 1, 2]], reduction="none"
            )
        assert torch.allclose(calls[0][1], ref, atol=1e-5)
        ref_last = nn.functional.cross_entropy(
            model.bias.unsqueeze(0).expand(2, -1), ds.y[[5, 6]], reduction="none"
        )
        assert torch.allclose(calls[2][1], ref_last.detach(), atol=1e-5)

    def test_sink_index_is_offset_on_mid_epoch_resume(self):
        ds = _TinyDS(n=6, num_classes=3)
        # Resume at batch 2 of a 3-batch schedule: only the tail is loaded, but
        # the sink must see the ABSOLUTE position so it maps to schedule[2].
        loader = DataLoader(
            ds, batch_sampler=_FixedBatchSampler([[4, 5]]), collate_fn=_collate_bucket_batch
        )
        model = _TinyModel(3)
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        seen: list[int] = []
        _train_one_epoch(
            model,
            loader,
            opt,
            _cfg(),
            _DEVICE,
            torch.float32,
            start_batch=2,
            sample_loss_sink=lambda i, _l: seen.append(i),
        )
        assert seen == [2]

    def test_no_sink_is_a_no_op(self):
        ds = _TinyDS(n=4, num_classes=3)
        loader = DataLoader(
            ds, batch_sampler=_FixedBatchSampler([[0, 1], [2, 3]]), collate_fn=_collate_bucket_batch
        )
        model = _TinyModel(3)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        loss, per_class = _train_one_epoch(model, loader, opt, _cfg(), _DEVICE, torch.float32)
        assert loss > 0
        assert per_class


# ---------------------------------------------------------------------------
# Task wiring: schedule → paths, pinned val order
# ---------------------------------------------------------------------------


class TestTaskWiring:
    def _task(self):
        from bittrainer.generic.tasks.group_task import GroupTask

        config = _cfg()
        task = GroupTask(config)
        task.train_ds = _TinyDS(n=6, num_classes=3)
        task.val_ds = _TinyDS(n=5, num_classes=3)
        return task

    def test_train_sink_maps_batch_index_to_schedule_paths(self):
        task = self._task()
        task._train_schedule = [[3, 0], [5, 1, 2]]
        task._current_epoch = 4
        task.sample_stats = SampleStatsRecorder(class_names=["a", "b", "c"])
        sink = task._make_sample_loss_sink()
        sink(1, torch.tensor([0.1, 0.2, 0.3]))
        rows = {r["path"]: r for r in task.sample_stats.rows()}
        assert set(rows) == {"img5.png", "img1.png", "img2.png"}
        assert rows["img5.png"]["train_loss"][4] == 0.1
        assert rows["img2.png"]["train_loss"][4] == 0.3
        assert rows["img5.png"]["label"] == task.train_ds.samples[5]["label"]

    def test_val_loader_order_is_materialised_and_stable(self):
        from bittrainer.generic.task import ResumeInfo, TaskContext
        from bittrainer.progress import ProgressEmitter
        from bittrainer.smart_cache import _noop_callback

        task = self._task()
        ctx = TaskContext(
            device=_DEVICE,
            dtype=torch.float32,
            em=ProgressEmitter(_noop_callback),
            cb=_noop_callback,
            checkpoint_dir=None,
            stop_event=None,
            stop_now_event=None,
            pause_event=None,
        )
        task.config.dataloader_workers = 0
        task.build_loaders(ctx, 0, 2, ResumeInfo(mid_resume=False))
        assert isinstance(task._val_sampler, _FixedBatchSampler)
        first = [list(b) for b in task._val_sampler]
        second = [list(b) for b in task._val_sampler]
        assert first == second
        assert sorted(i for b in first for i in b) == list(range(5))
        assert task._val_schedule == first

    def test_record_val_epoch_uses_val_schedule_order(self):
        task = self._task()
        task._val_schedule = [[2, 0], [4]]
        task.sample_stats = SampleStatsRecorder(class_names=["a", "b", "c"])
        logits = torch.tensor([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]])
        labels = torch.tensor([task.val_ds.samples[i]["label"] for i in (2, 0, 4)])
        task._record_val_epoch(3, logits, labels)
        rows = {r["path"]: r for r in task.sample_stats.rows()}
        assert set(rows) == {"img2.png", "img0.png", "img4.png"}
        assert rows["img0.png"]["val_probs"][3][1] > 0.98
        assert rows["img4.png"]["split"] == "val"

    def test_record_val_epoch_refuses_misaligned_labels(self):
        import pytest

        task = self._task()
        task._val_schedule = [[0, 1]]
        task.sample_stats = SampleStatsRecorder(class_names=["a", "b", "c"])
        wrong = torch.tensor([(task.val_ds.samples[0]["label"] + 1) % 3, 0])
        with pytest.raises(RuntimeError):
            task._record_val_epoch(0, torch.zeros(2, 3), wrong)
