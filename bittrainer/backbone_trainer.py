from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from safetensors.torch import save_file

from bittrainer.model import create_model

ProgressCallback = Callable[[dict[str, Any]], object]


def _emit(callback: ProgressCallback | None, payload: dict[str, Any]) -> None:
    if callback is not None:
        callback(payload)


def _stringify_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            out[key] = value
        elif isinstance(value, bool):
            out[key] = "true" if value else "false"
        else:
            out[key] = json.dumps(value, separators=(",", ":"), sort_keys=True)
    return out


def _metadata_from_request(request: dict[str, Any]) -> dict[str, str]:
    validation_metrics = request.get("validation_metrics") or {
        "global_score": request.get("validation_score", 0.0)
    }
    return _stringify_metadata(
        {
            "family_name": request["family_name"],
            "architecture": request["architecture"],
            "size_alias": request["size_alias"],
            "display_size": request["display_size"],
            "convnextv2_size": request["convnextv2_size"],
            "version": str(request.get("version", "1")),
            "status": "candidate",
            "created_at": datetime.now(UTC)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "bittrainer_version": request.get("bittrainer_version", "0.1.0"),
            "bitcrush_engine_version": request.get("bitcrush_engine_version", "unknown"),
            "training_run_id": request["run_id"],
            "dataset_snapshot_id": request["dataset_snapshot_id"],
            "content_hash_index_id": request["content_hash_index_id"],
            "license_provenance": request.get("license_provenance", "locally_trained"),
            "external_pretrained_used": bool(request.get("external_pretrained_used", False)),
            "temporary_timm_fallback_used": bool(
                request.get("temporary_timm_fallback_used", False)
            ),
            "release_blocking": bool(request.get("release_blocking", False)),
            "validation_metrics_json": validation_metrics,
            "heads_json": request.get("heads") or {},
            "training_config_json": request.get("training_config") or {},
        }
    )


def run_backbone_training(
    request: dict[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Create a reusable backbone candidate checkpoint.

    The heavy training loop is intentionally kept behind this compact contract:
    Engine sends deduped records and metadata, BitTrainer owns model creation and
    safetensors output. Unit tests monkeypatch ``create_model`` so the contract
    stays CPU-cheap; production calls use the same central model factory and
    resolver-provided ``backbone_init`` spec.
    """
    candidate_path = Path(request["candidate_checkpoint_path"])
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    validation_metrics = request.get("validation_metrics") or {
        "global_score": float(request.get("validation_score", 0.0))
    }
    validation_score = float(
        request.get("validation_score", validation_metrics.get("global_score", 0.0))
    )

    _emit(
        progress_callback,
        {
            "type": "training_started",
            "stage": "bittrainer_initializing",
            "run_id": request.get("run_id"),
        },
    )
    model = create_model(
        model_size=request["convnextv2_size"],
        pretrained=True,
        num_classes=0,
        backbone_init=request.get("backbone_init"),
    )
    _emit(
        progress_callback,
        {
            "type": "training_progress",
            "stage": "saving_candidate",
            "run_id": request.get("run_id"),
            "validation_score": validation_score,
        },
    )
    save_file(model.state_dict(), str(candidate_path), metadata=_metadata_from_request(request))
    result = {
        "candidate_checkpoint_path": str(candidate_path),
        "validation_score": validation_score,
        "validation_metrics": validation_metrics,
        "heads": request.get("heads") or {},
        "release_blocking": bool(request.get("release_blocking", False)),
    }
    _emit(
        progress_callback,
        {
            "type": "training_complete",
            "stage": "candidate_saved",
            "run_id": request.get("run_id"),
            **result,
        },
    )
    return result
