"""Stable base sample list + persistent loaders (Bitcrush ISSUE-0859 / ISSUE-0860).

Two measured defects on a live 13.6k-image run:

* ``GroupDataset.reshuffle()`` re-opened every unique image with PIL to rebuild a
  local size cache (~200 s/epoch on the main thread, GPU idle).
* ``GroupTask.build_loaders`` built brand-new train/val ``DataLoader``s every
  epoch, so ``persistent_workers=True`` never persisted — 12 worker spawns per
  epoch on Windows.

The fix makes ``samples`` a STABLE base list (one row per unique (path, class),
never mutated after construction) and expresses per-epoch replication + shuffle
as an index schedule over it, so the loaders (and their pickled worker copies of
the dataset) can be built once and reused.

CPU-only: no CUDA is touched anywhere in this file.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
from collections import Counter

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from bittrainer.group_dataset import GroupDataset, build_group_bucket_sampler
from bittrainer.training_state import SCHEDULE_INDEXING, _FixedBatchSampler


def _build_group(root, counts, split="train"):
    for ci, (cname, n) in enumerate(counts.items()):
        d = root / cname / split
        d.mkdir(parents=True, exist_ok=True)
        for j in range(n):
            rng = np.random.default_rng(ci * 1000 + j)
            Image.fromarray(
                rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
            ).save(d / f"img_{ci}_{j}.png")


# ---------------------------------------------------------------------------
# ISSUE-0859: sizes are read once; reshuffle never re-opens an image
# ---------------------------------------------------------------------------


class TestSizeMemo:
    def test_reshuffle_never_reopens_images(self, tmp_path, monkeypatch):
        root = tmp_path / "g"
        _build_group(root, {"a": 3, "b": 9})
        random.seed(0)
        ds = GroupDataset(root, ["a", "b"], split="train", group_name="g")

        calls: list[str] = []
        real = GroupDataset._get_image_size

        def _spy(path):
            calls.append(str(path))
            return real(path)

        monkeypatch.setattr(GroupDataset, "_get_image_size", staticmethod(_spy))

        before = list(ds.epoch_indices())
        orders = [before]
        for _ in range(3):
            ds.reshuffle()
            orders.append(list(ds.epoch_indices()))

        assert calls == [], f"reshuffle re-opened {len(calls)} images"
        # Still actually reshuffling: at least one redraw differs from the first.
        assert any(o != before for o in orders[1:])

    def test_construction_opens_each_unique_path_once(self, tmp_path, monkeypatch):
        root = tmp_path / "g"
        _build_group(root, {"a": 2, "b": 5})
        calls: list[str] = []
        real = GroupDataset._get_image_size

        def _spy(path):
            calls.append(str(path))
            return real(path)

        monkeypatch.setattr(GroupDataset, "_get_image_size", staticmethod(_spy))
        random.seed(0)
        GroupDataset(root, ["a", "b"], split="train", group_name="g")
        assert len(calls) == 7
        assert len(set(calls)) == 7

    def test_toggles_do_not_reopen_images(self, tmp_path, monkeypatch):
        root = tmp_path / "g"
        _build_group(root, {"a": 3, "b": 9, "__none__": 2})
        random.seed(0)
        ds = GroupDataset(root, ["a", "b", "__none__"], split="train", group_name="g")

        calls: list[str] = []
        monkeypatch.setattr(
            GroupDataset, "_get_image_size", staticmethod(lambda p: calls.append(p))
        )
        ds.set_natural_sampling(True)
        ds.set_natural_sampling(False)
        ds.set_oversample_none(True)
        ds.set_oversample_none(False)
        assert calls == []


# ---------------------------------------------------------------------------
# Stable base list + epoch schedule
# ---------------------------------------------------------------------------


class TestStableBaseList:
    def test_samples_are_unique_rows_and_never_mutated(self, tmp_path):
        root = tmp_path / "g"
        _build_group(root, {"a": 3, "b": 9})
        random.seed(0)
        ds = GroupDataset(root, ["a", "b"], split="train", group_name="g")

        base = list(ds.samples)
        identity = [id(s) for s in ds.samples]
        # One row per (path, class), no replication baked in.
        assert len(base) == 12
        assert len({(s["path"], s["label"]) for s in base}) == 12

        for _ in range(3):
            ds.reshuffle()
            assert ds.samples is not None
            assert [id(s) for s in ds.samples] == identity
            assert list(ds.samples) == base

    def test_len_is_unique_rows_schedule_carries_replication(self, tmp_path):
        root = tmp_path / "g"
        _build_group(root, {"a": 3, "b": 9})
        random.seed(0)
        ds = GroupDataset(root, ["a", "b"], split="train", group_name="g")
        assert len(ds) == 12
        # a equalises up to b (3 -> 9, within the 4x cap), b keeps its 9.
        assert len(ds.epoch_indices()) == 18

    def test_effective_counts_track_the_epoch_schedule(self, tmp_path):
        root = tmp_path / "g"
        _build_group(root, {"a": 2, "b": 20})
        random.seed(0)
        ds = GroupDataset(root, ["a", "b"], split="train", group_name="g")
        # Oversample cap behaviour is unchanged: ceil(4 * 2) = 8, not 20.
        assert ds.get_effective_class_counts() == {0: 8, 1: 20}
        # ...and it stays that way after a reshuffle.
        ds.reshuffle()
        assert ds.get_effective_class_counts() == {0: 8, 1: 20}
        # Natural counts still come from the raw disk lists.
        assert ds.get_class_counts() == {0: 2, 1: 20}

    def test_epoch_samples_expand_the_schedule(self, tmp_path):
        root = tmp_path / "g"
        _build_group(root, {"a": 2, "b": 20})
        random.seed(0)
        ds = GroupDataset(root, ["a", "b"], split="train", group_name="g")
        rows = ds.epoch_samples()
        assert len(rows) == 28
        assert dict(Counter(r["label"] for r in rows)) == {0: 8, 1: 20}

    def test_bucket_sampler_covers_the_epoch_schedule(self, tmp_path):
        root = tmp_path / "g"
        _build_group(root, {"a": 2, "b": 20})
        random.seed(0)
        ds = GroupDataset(root, ["a", "b"], split="train", group_name="g")
        sampler = build_group_bucket_sampler(ds, batch_size=4)
        flat = [i for b in sampler for i in b]
        assert sorted(flat) == sorted(ds.epoch_indices())
        assert len(sampler) == len(list(iter(sampler)))
        # A second iteration reflects a fresh reshuffle, not the stale list.
        ds.reshuffle()
        flat2 = [i for b in sampler for i in b]
        assert sorted(flat2) == sorted(ds.epoch_indices())

    def test_val_split_schedule_covers_every_row_once(self, tmp_path):
        root = tmp_path / "g"
        _build_group(root, {"a": 3, "b": 4}, split="val")
        random.seed(0)
        ds = GroupDataset(root, ["a", "b"], split="val", group_name="g")
        assert sorted(ds.epoch_indices()) == list(range(7))
        # val never reshuffles
        before = list(ds.epoch_indices())
        ds.reshuffle()
        assert list(ds.epoch_indices()) == before


# ---------------------------------------------------------------------------
# ISSUE-0860: loaders built once, schedule swapped per epoch
# ---------------------------------------------------------------------------


def _task(tmp_path, workers=0):
    from bittrainer.generic.task import TaskContext
    from bittrainer.generic.tasks.group_task import GroupTask
    from bittrainer.group_trainer import GroupTrainConfig
    from bittrainer.progress import ProgressEmitter
    from bittrainer.smart_cache import _noop_callback

    root = tmp_path / "g"
    _build_group(root, {"a": 3, "b": 9})
    _build_group(root, {"a": 2, "b": 2}, split="val")
    config = GroupTrainConfig(
        group_folder=str(root),
        num_classes=2,
        class_names=["a", "b"],
        dataloader_workers=workers,
        channels_last=False,
    )
    task = GroupTask(config)
    random.seed(0)
    task.train_ds = GroupDataset(root, ["a", "b"], split="train", group_name="g")
    task.val_ds = GroupDataset(root, ["a", "b"], split="val", group_name="g")
    ctx = TaskContext(
        device=torch.device("cpu"),
        dtype=torch.float32,
        em=ProgressEmitter(_noop_callback),
        cb=_noop_callback,
        checkpoint_dir=None,
        stop_event=None,
        stop_now_event=None,
        pause_event=None,
    )
    return task, ctx


class TestLoaderReuse:
    def test_same_loader_objects_across_epochs(self, tmp_path):
        from bittrainer.generic.task import ResumeInfo

        task, ctx = _task(tmp_path)
        l0, s0, b0 = task.build_loaders(ctx, 0, 4, ResumeInfo(mid_resume=False))
        val0 = task._val_loader
        task.reshuffle()
        l1, s1, b1 = task.build_loaders(ctx, 1, 4, ResumeInfo(mid_resume=False))

        assert l1 is l0, "train DataLoader rebuilt — persistent workers would respawn"
        assert task._val_loader is val0, "val DataLoader rebuilt every epoch"
        assert b0 == 0 and b1 == 0
        # The schedule the loader runs is the NEW epoch's, not epoch 0's.
        assert [list(b) for b in l1.batch_sampler] == s1
        assert s1 != s0

    def test_mid_resume_slices_the_stored_schedule_on_the_same_loader(self, tmp_path):
        from bittrainer.generic.task import ResumeInfo

        task, ctx = _task(tmp_path)
        loader, schedule, _ = task.build_loaders(ctx, 0, 4, ResumeInfo(mid_resume=False))
        stored = [list(b) for b in schedule]

        task.reshuffle()
        loader2, sched2, start_batch = task.build_loaders(
            ctx,
            1,
            4,
            ResumeInfo(
                mid_resume=True,
                resume_schedule=stored,
                resume_batch_in_epoch=2,
                resume_rng_now=None,
            ),
        )
        assert loader2 is loader
        assert sched2 == stored
        assert start_batch == 2
        assert [list(b) for b in loader2.batch_sampler] == stored[2:]

    def test_fixed_batch_sampler_schedule_is_settable(self):
        s = _FixedBatchSampler([[0, 1], [2]])
        assert len(s) == 2
        s.set_batches([[3], [4], [5]])
        assert [list(b) for b in s] == [[3], [4], [5]]
        assert len(s) == 3


class TestSampleStatsMapping:
    def test_batch_paths_match_what_the_loader_yields_after_reshuffle(self, tmp_path):
        """The sink maps schedule[batch] -> samples[i]; with num_workers=0 we can
        compare that against the labels the loader actually delivers."""
        from bittrainer.generic.task import ResumeInfo
        from bittrainer.sample_stats import SampleStatsRecorder

        task, ctx = _task(tmp_path)
        task.sample_stats = SampleStatsRecorder(class_names=["a", "b"])
        task.build_loaders(ctx, 0, 4, ResumeInfo(mid_resume=False))
        task.reshuffle()
        loader, schedule, _ = task.build_loaders(ctx, 1, 4, ResumeInfo(mid_resume=False))

        sink = task._make_sample_loss_sink()
        base = task.train_ds.samples
        for bi, (_imgs, labels) in enumerate(loader):
            expected = [int(base[i]["label"]) for i in schedule[bi]]
            assert expected == [int(x) for x in labels], f"batch {bi} mis-mapped"
            sink(bi, torch.arange(labels.numel(), dtype=torch.float32))

        recorded = {r["path"] for r in task.sample_stats.rows()}
        assert recorded == {base[i]["path"] for b in schedule for i in b}


# ---------------------------------------------------------------------------
# Windows/spawn evidence: the SAME workers serve consecutive epochs
# ---------------------------------------------------------------------------


class _PidDataset(torch.utils.data.Dataset):
    """Module-level (picklable) stub: every item reports the serving PID."""

    def __len__(self) -> int:
        return 32

    def __getitem__(self, idx: int):
        return torch.tensor([idx, os.getpid()], dtype=torch.long)


@pytest.mark.skipif(
    "spawn" not in mp.get_all_start_methods(), reason="spawn start method unavailable"
)
def test_persistent_workers_survive_between_epochs():
    sampler = _FixedBatchSampler([[0, 1, 2, 3], [4, 5, 6, 7]])
    loader = DataLoader(
        _PidDataset(),
        batch_sampler=sampler,
        num_workers=2,
        persistent_workers=True,
    )
    first = {int(row[1]) for batch in loader for row in batch}
    # A new epoch with a DIFFERENT schedule through the SAME loader.
    sampler.set_batches([[8, 9, 10, 11], [12, 13, 14, 15]])
    second = {int(row[1]) for batch in loader for row in batch}

    assert first, "no worker PIDs observed"
    assert os.getpid() not in first, "items were served in-process, not by workers"
    assert first == second, f"workers respawned: {first} -> {second}"
    del loader


# ---------------------------------------------------------------------------
# Backup schema guard: an old-style schedule must not drive a mid-epoch resume
# ---------------------------------------------------------------------------


class TestScheduleIndexingGuard:
    def test_collect_epoch_state_stamps_the_marker(self):
        from bittrainer.training_state import collect_epoch_state

        model = torch.nn.Linear(2, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        state = collect_epoch_state(
            fingerprint={}, trainer="group", epoch=1, global_step=3, eff_bs=2,
            scheduler_t_max=3, model=model, optimizer=opt, scheduler=sched,
            best={}, batch_in_epoch=4,
        )
        assert state["schedule_indexing"] == SCHEDULE_INDEXING

    def test_old_backup_falls_back_to_epoch_restart(self):
        from bittrainer.training_state import resume_schedule_from_state

        old = {"batch_in_epoch": 5, "batch_schedule": [[0, 1], [2, 3]]}
        assert resume_schedule_from_state(old, bs_changed=False) == (None, 0)

        new = {
            "batch_in_epoch": 5,
            "batch_schedule": [[0, 1], [2, 3]],
            "schedule_indexing": SCHEDULE_INDEXING,
        }
        assert resume_schedule_from_state(new, bs_changed=False) == (
            [[0, 1], [2, 3]],
            5,
        )
        # A changed batch size still discards the schedule.
        assert resume_schedule_from_state(new, bs_changed=True) == (None, 0)


# ---------------------------------------------------------------------------
# Epoch-loop contract (stub model; no real training run)
# ---------------------------------------------------------------------------


class _BiasOnlyModel(torch.nn.Module):
    """Input-invariant logits — pins ORDER/shape handling, not learning."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        return self.bias.unsqueeze(0).expand(x.shape[0], -1) + 0.0 * x.flatten(1).sum(
            1, keepdim=True
        )


class TestEpochLoopContract:
    def test_two_epochs_through_one_loader(self, tmp_path):
        """The order GenericTrainer.run uses: reshuffle -> build_loaders ->
        train_epoch. The loader is created once; each epoch still runs exactly
        its own schedule's batches."""
        import bittrainer.group_trainer as gt
        from bittrainer.generic.task import ResumeInfo

        task, ctx = _task(tmp_path)
        task.config.num_classes = 2
        task.config.randaugment_n = 0
        task.config.random_erasing_p = 0.0
        task.config.aug_noise_p = 0.0
        task.config.aug_blur_p = 0.0
        task.config.aug_jpeg_p = 0.0
        task.config.use_mixup = False
        model = _BiasOnlyModel(2)
        opt = torch.optim.SGD(model.parameters(), lr=0.0)

        loaders = []
        for epoch in range(2):
            task.reshuffle()
            loader, schedule, start_batch = task.build_loaders(
                ctx, epoch, 4, ResumeInfo(mid_resume=False)
            )
            loaders.append(loader)
            assert start_batch == 0
            assert len(loader) == len(schedule)
            steps = []
            gt._train_one_epoch(
                model,
                loader,
                opt,
                task.config,
                torch.device("cpu"),
                torch.float32,
                step_callback=lambda s, t, _l: steps.append((s, t)),
            )
            assert steps and steps[-1][0] == len(schedule)

        assert loaders[0] is loaders[1]
