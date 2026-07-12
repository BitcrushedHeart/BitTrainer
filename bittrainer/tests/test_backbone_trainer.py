"""run_backbone_training: end-to-end micro run on synthetic data (CPU).

Pins the request/response contract Engine's BackboneTrainingManager relies on
(Bitcrush ISSUE-0342): async entry point, thread-safe progress forwarding,
candidate safetensors with Engine-readable metadata, and the result keys the
manager writes to BackboneTrainingRun.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from PIL import Image

from bittrainer.backbone_trainer import run_backbone_training


def _make_images(root, names):
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, name in enumerate(names):
        path = root / f"{name}.png"
        Image.new("RGB", (72, 64), color=((i * 37) % 255, (i * 83) % 255, (i * 151) % 255)).save(
            path
        )
        paths.append(str(path))
    return paths


def _request(tmp_path):
    paths = _make_images(tmp_path / "imgs", [f"img{i}" for i in range(8)])
    records = []
    for i, path in enumerate(paths):
        records.append(
            {
                "content_hash": f"{i:02d}" + "ab" * 31,
                "file_paths": [path],
                "binary": {"watermark": "positive" if i % 2 else "negative"},
                "groups": {"shot_type": "closeup" if i % 2 else "wide"},
                "splits": {"train": 1},
            }
        )
    candidate = tmp_path / "candidates" / "candidate_test.safetensors"
    return {
        "run_id": "run_test",
        "family_name": "bitcrush_backbone",
        "architecture": "convnextv2",
        "size_alias": "lite",
        "display_size": "Lite",
        "display_label": "Lite",
        "convnextv2_size": "atto",
        "candidate_checkpoint_path": str(candidate),
        "records": records,
        "dataset_audit": {"unique_images": len(records)},
        "dataset_snapshot_id": "sha256:snap",
        "content_hash_index_id": "sha256:index",
        "heads": {"classifier": {"supported": True, "instantiated": True, "trained": True}},
        "training_config": {
            "image_size": 64,
            "batch_size": 4,
            "epochs": 1,
            "max_steps": 4,
            "learning_rate": 1e-4,
            "validation_split": 0.25,
            "device": "cpu",
        },
        "backbone_init": {"source": "random_init", "checkpoint_path": None},
        "license_provenance": "locally_trained",
        "external_pretrained_used": False,
        "temporary_timm_fallback_used": False,
        "release_blocking": False,
    }


@pytest.fixture(scope="module")
def run_result(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("backbone_run")
    request = _request(tmp_path)
    events = []

    async def _progress(msg):
        events.append(msg)

    result = asyncio.run(run_backbone_training(request, progress_callback=_progress))
    return request, result, events


def test_result_contract(run_result):
    request, result, _ = run_result
    assert result["candidate_checkpoint_path"] == request["candidate_checkpoint_path"]
    assert isinstance(result["validation_score"], float)
    assert isinstance(result["validation_metrics"], dict)
    assert result["heads"] == request["heads"]
    assert result["release_blocking"] is False


def test_progress_events_forwarded(run_result):
    _, _, events = run_result
    assert events, "no progress events forwarded"
    assert all(isinstance(e, dict) for e in events)
    assert any(e.get("type") == "training_progress" for e in events)


def test_candidate_checkpoint_written_with_metadata(run_result):
    from safetensors import safe_open

    request, result, _ = run_result
    path = result["candidate_checkpoint_path"]
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata()
        keys = list(f.keys())
    assert keys, "checkpoint holds no tensors"
    # head.norm (pre-classifier norm) is part of the feature extractor and
    # ships with the backbone; an actual classifier layer must not.
    assert not any(k.startswith("head.fc") for k in keys), "classifier layer leaked into backbone"
    assert metadata["family_name"] == "bitcrush_backbone"
    assert metadata["size_alias"] == "lite"
    assert metadata["convnextv2_size"] == "atto"
    assert metadata["status"] == "candidate"
    assert metadata["training_run_id"] == "run_test"
    assert metadata["dataset_snapshot_id"] == "sha256:snap"
    assert metadata["content_hash_index_id"] == "sha256:index"
    assert metadata["license_provenance"] == "locally_trained"
    assert metadata["external_pretrained_used"] == "false"
    assert metadata["temporary_timm_fallback_used"] == "false"
    assert metadata["release_blocking"] == "false"
    assert json.loads(metadata["validation_metrics_json"]) == result["validation_metrics"]


def test_local_backbone_init_is_honoured(tmp_path):
    """A local_active spec must load the given checkpoint into the new backbone."""
    import torch
    from safetensors.torch import load_file, save_file

    from bittrainer.model import create_model

    donor = create_model(model_size="atto", pretrained=False, num_classes=0)
    active = tmp_path / "active.safetensors"
    save_file(donor.state_dict(), str(active))

    request = _request(tmp_path)
    request["backbone_init"] = {"source": "local_active", "checkpoint_path": str(active)}
    request["training_config"]["max_steps"] = 0  # init-only: no optimiser step

    result = asyncio.run(run_backbone_training(request, progress_callback=None))
    saved = load_file(result["candidate_checkpoint_path"])
    donor_state = donor.state_dict()
    assert torch.equal(saved["stem.0.weight"], donor_state["stem.0.weight"])
