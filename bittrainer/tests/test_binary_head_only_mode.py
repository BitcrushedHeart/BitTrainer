"""Binary cached-feature head-only mode (Backbeat ISSUE-0700)."""

from __future__ import annotations

from bittrainer.trainer import TrainConfig


def test_train_config_keeps_full_mode_for_direct_bittrainer_callers(tmp_path) -> None:
    config = TrainConfig(concept_folder=str(tmp_path))
    assert config.training_mode == "full"


def test_frozen_backbone_mode_never_runs_epoch_unfreeze(monkeypatch, tmp_path) -> None:
    from bittrainer.generic.tasks.binary_task import BinaryTask
    import bittrainer.generic.tasks.binary_task as binary_task_module

    config = TrainConfig(concept_folder=str(tmp_path), training_mode="frozen_backbone")
    task = BinaryTask(config)
    calls: list[str] = []
    monkeypatch.setattr(
        binary_task_module,
        "unfreeze_backbone",
        lambda _model: calls.append("backbone"),
    )
    monkeypatch.setattr(
        binary_task_module,
        "unfreeze_stage",
        lambda _model, stage: calls.append(f"stage-{stage}"),
    )

    for epoch in range(1, 6):
        result = task.on_epoch_start(
            None,
            object(),
            epoch,
            optimizer=object(),
            scheduler=object(),
            scheduler_t_max=10,
            start_epoch=0,
        )
        assert result is None

    assert calls == []


def test_head_only_dispatches_to_cached_feature_task(monkeypatch, tmp_path) -> None:
    from bittrainer.generic.generic_trainer import GenericTrainer
    from bittrainer.generic.tasks.binary_head_only_task import BinaryHeadOnlyTask
    from bittrainer.trainer import run_training

    seen = []

    def _run(_self, task, **_kwargs):
        seen.append(task)
        return {"mode": "head_only"}

    monkeypatch.setattr(GenericTrainer, "run", _run)

    result = run_training(
        TrainConfig(concept_folder=str(tmp_path), training_mode="head_only")
    )

    assert result == {"mode": "head_only"}
    assert len(seen) == 1
    assert isinstance(seen[0], BinaryHeadOnlyTask)


def test_head_only_samples_negative_quota_before_dataset_dimension_scan() -> None:
    from bittrainer.generic.tasks.binary_head_only_task import _sample_negative_pool

    pool = [f"negative-{index}.png" for index in range(100)]

    sampled = _sample_negative_pool(
        pool,
        positive_count=12,
        hard_negative_count=3,
        hard_negative_weight=3,
        neg_pos_ratio=1.0,
    )

    # Twelve total negative slots minus three hard negatives repeated 3x.
    assert len(sampled) == 3
    assert set(sampled) <= set(pool)


def test_head_only_builds_then_reuses_cached_features(tmp_path) -> None:
    import numpy as np
    from PIL import Image

    from bittrainer.trainer import run_training

    root = tmp_path / "concept"
    rng = np.random.default_rng(7)
    for label, offset in (("positive", 120), ("negative", 0)):
        for split, count in (("train", 6), ("val", 4)):
            folder = root / label / split
            folder.mkdir(parents=True)
            for index in range(count):
                pixels = rng.integers(0, 100, (64, 64, 3), dtype=np.uint8)
                pixels[..., 0] = np.clip(pixels[..., 0] + offset, 0, 255)
                Image.fromarray(pixels).save(folder / f"{index}.png")

    config = TrainConfig(
        concept_folder=str(root),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        embedding_cache_dir=str(tmp_path / "embeddings"),
        training_mode="head_only",
        model_size="atto",
        device="cpu",
        dtype="float32",
        use_cache=False,
        from_scratch=False,
        max_epochs=8,
        patience=3,
        dataloader_workers=0,
        backbone_init={"source": "random_init", "checkpoint_path": None},
    )

    first = run_training(config)
    second = run_training(config)

    assert first["mode"] == "head_only"
    assert first["embedding_cache_stats"]["built"] == 20
    assert second["embedding_cache_stats"] == {"built": 0, "reused": 20, "total": 20}
    assert first["backbone_hash"] == second["backbone_hash"]
    assert second["checkpoint_path"].endswith("best.pt")
