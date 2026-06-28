from __future__ import annotations

import json

import torch


class TinyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = torch.nn.Linear(1, 1)

    def forward(self, x):  # pragma: no cover - not used by this contract test
        return self.stem(x)


def test_run_backbone_training_saves_safetensors_candidate_with_metadata(
    monkeypatch,
    tmp_path,
):
    from safetensors import safe_open

    import bittrainer.backbone_trainer as backbone_trainer

    calls: list[dict] = []
    events: list[dict] = []
    candidate = tmp_path / "candidate_unit.safetensors"
    init_spec = {"source": "random_init"}

    def fake_create_model(**kwargs):
        calls.append(kwargs)
        return TinyBackbone()

    monkeypatch.setattr(backbone_trainer, "create_model", fake_create_model)

    result = backbone_trainer.run_backbone_training(
        {
            "run_id": "run_unit",
            "family_name": "CrushVision",
            "architecture": "convnextv2_compatible",
            "size_alias": "pro",
            "display_size": "Pro",
            "convnextv2_size": "base",
            "candidate_checkpoint_path": str(candidate),
            "dataset_snapshot_id": "sha256:dataset",
            "content_hash_index_id": "sha256:index",
            "validation_score": 0.87,
            "validation_metrics": {"global_score": 0.87},
            "heads": {"classifier": {"supported": True, "instantiated": True}},
            "training_config": {"epochs": 1},
            "backbone_init": init_spec,
        },
        progress_callback=lambda msg: events.append(msg),
    )

    assert calls == [
        {
            "model_size": "base",
            "pretrained": True,
            "num_classes": 0,
            "backbone_init": init_spec,
        }
    ]
    assert result["candidate_checkpoint_path"] == str(candidate)
    assert result["validation_score"] == 0.87
    assert candidate.exists()
    with safe_open(str(candidate), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    assert metadata["family_name"] == "CrushVision"
    assert metadata["status"] == "candidate"
    assert metadata["release_blocking"] == "false"
    assert json.loads(metadata["validation_metrics_json"])["global_score"] == 0.87
    assert [event["type"] for event in events] == [
        "training_started",
        "training_progress",
        "training_complete",
    ]
