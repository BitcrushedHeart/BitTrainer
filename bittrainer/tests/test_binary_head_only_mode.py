"""Binary head-only mode (Bitcrush ISSUE-0692)."""

from __future__ import annotations

from bittrainer.generic.tasks.binary_task import BinaryTask
from bittrainer.trainer import TrainConfig


def test_train_config_keeps_full_mode_for_direct_bittrainer_callers(tmp_path) -> None:
    config = TrainConfig(concept_folder=str(tmp_path))
    assert config.training_mode == "full"


def test_head_only_mode_never_runs_epoch_unfreeze(monkeypatch, tmp_path) -> None:
    import bittrainer.generic.tasks.binary_task as binary_task_module

    config = TrainConfig(concept_folder=str(tmp_path), training_mode="head_only")
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
