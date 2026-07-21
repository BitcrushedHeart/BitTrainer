"""BackboneTask on the GenericTrainer skeleton (Bitcrush ISSUE-0542 Step 6).

Pins the three intended behaviour changes of migrating ``_train_backbone`` onto
:class:`~bittrainer.generic.generic_trainer.GenericTrainer`:

1. the optimizer is ``Prodigy_adv`` (Kourkoutas-beta) built via the shared
   ``make_optimizer`` factory, not AdamW;
2. the exported candidate carries the best-epoch multi-task head tensors
   (``heads.*``) alongside the BARE backbone keys, with metadata that lets a
   consumer rebuild the heads, and legacy bare-only candidates still load; and
3. ``_build_samples`` de-duplicates records sharing a ``content_hash`` and the
   "preparing" progress payload reports ``unique_images``.

CPU-only, ``atto`` backbone, ``dataloader_workers=0``.
"""

from __future__ import annotations

import asyncio

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file

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


def _request(tmp_path, *, epochs=1, max_steps=8, n=8):
    paths = _make_images(tmp_path / "imgs", [f"img{i}" for i in range(n)])
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
        "convnextv2_size": "atto",
        "candidate_checkpoint_path": str(candidate),
        "records": records,
        "dataset_snapshot_id": "sha256:snap",
        "content_hash_index_id": "sha256:index",
        "heads": {"classifier": {"supported": True, "instantiated": True, "trained": True}},
        "training_config": {
            "image_size": 64,
            "batch_size": 4,
            "epochs": epochs,
            "max_steps": max_steps,
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


def _run(request, progress_callback=None):
    return asyncio.run(run_backbone_training(request, progress_callback=progress_callback))


def test_optimizer_is_prodigy_with_kourkoutas(tmp_path, monkeypatch):
    """The migrated task builds Prodigy_adv (Kourkoutas-beta), not AdamW."""
    import bittrainer.generic.tasks.backbone_task as bt

    captured = {}
    real = bt.make_optimizer

    def _spy(model, **kw):
        opt = real(model, **kw)
        captured["opt"] = opt
        return opt

    monkeypatch.setattr(bt, "make_optimizer", _spy)

    _run(_request(tmp_path))
    opt = captured["opt"]
    assert type(opt).__name__ == "Prodigy_adv"
    # Kourkoutas-beta keeps its per-layer helper on the optimizer instance.
    assert getattr(opt, "kourkoutas_helper", None) is not None


def test_candidate_has_bare_backbone_and_heads(tmp_path):
    """Exported candidate: BARE backbone keys + heads.* tensors + head metadata."""
    request = _request(tmp_path)
    result = _run(request)
    with safe_open(result["candidate_checkpoint_path"], framework="pt") as f:
        metadata = f.metadata()
        keys = list(f.keys())

    backbone_keys = [k for k in keys if not k.startswith("heads.")]
    head_keys = [k for k in keys if k.startswith("heads.")]
    assert backbone_keys, "no backbone tensors"
    assert head_keys, "no head tensors exported"
    # Backbone keys stay BARE (no 'backbone.' prefix) so apply_backbone_init's
    # unprefix branch is never tripped by the mixed key space.
    assert not any(k.startswith("backbone.") for k in keys)
    assert metadata["heads_state_present"] == "1"
    # Per-head metadata is enough to rebuild the head shapes.
    import json

    groups = json.loads(metadata["heads_groups_json"])
    concepts = json.loads(metadata["heads_concepts_json"])
    assert "shot_type" in groups and len(groups["shot_type"]) >= 2
    assert "watermark" in concepts
    assert int(metadata["backbone_feature_dim"]) > 0
    # A group head's out-features == its class count.
    gw = load_file(result["candidate_checkpoint_path"])["heads.groups.shot_type.weight"]
    assert gw.shape[0] == len(groups["shot_type"])


def test_apply_backbone_init_loads_new_format(tmp_path):
    """apply_backbone_init loads the NEW mixed-key candidate into a fresh atto."""
    from bittrainer.backbone_init import apply_backbone_init
    from bittrainer.model import create_model

    result = _run(_request(tmp_path))
    fresh = create_model(model_size="atto", pretrained=False, num_classes=0)
    loaded = apply_backbone_init(
        fresh, {"source": "local_active", "checkpoint_path": result["candidate_checkpoint_path"]}
    )
    assert loaded is True
    # Every bare backbone tensor in the candidate matched the target.
    saved = load_file(result["candidate_checkpoint_path"])
    target = fresh.state_dict()
    for key in saved:
        if key.startswith("heads."):
            continue
        assert key in target and target[key].shape == saved[key].shape


def test_apply_backbone_init_loads_legacy_bare_only(tmp_path):
    """A LEGACY bare-only safetensors still loads through apply_backbone_init."""
    from bittrainer.backbone_init import apply_backbone_init
    from bittrainer.model import create_model

    donor = create_model(model_size="atto", pretrained=False, num_classes=0)
    legacy = tmp_path / "legacy.safetensors"
    save_file(donor.state_dict(), str(legacy))

    fresh = create_model(model_size="atto", pretrained=False, num_classes=0)
    loaded = apply_backbone_init(
        fresh, {"source": "local_active", "checkpoint_path": str(legacy)}
    )
    assert loaded is True
    assert torch.equal(fresh.state_dict()["stem.0.weight"], donor.state_dict()["stem.0.weight"])


def test_exported_heads_are_from_best_epoch(tmp_path, monkeypatch):
    """Heads exported come from the BEST epoch, not the last.

    Force epoch 1 to score higher than epoch 2 via a val-metric spy and assert
    the exported heads equal the epoch-1 snapshot (save_candidate ran once)."""
    import bittrainer.backbone_trainer as bb
    import bittrainer.generic.tasks.backbone_task as bt

    scores = iter([{"binary/watermark": 0.9}, {"binary/watermark": 0.1}])

    def _fake_evaluate(*_a, **_k):
        return next(scores)

    monkeypatch.setattr(bb, "_evaluate", _fake_evaluate)

    snapshots = []
    real_save = bt.BackboneTask.save_candidate

    def _spy_save(self, ctx, model, epoch, metrics, best):
        real_save(self, ctx, model, epoch, metrics, best)
        snapshots.append({k: v.clone() for k, v in (self.best_heads_state or {}).items()})

    monkeypatch.setattr(bt.BackboneTask, "save_candidate", _spy_save)

    request = _request(tmp_path, epochs=2, max_steps=1000, n=12)
    result = _run(request)

    # Only epoch 1 improved -> exactly one candidate snapshot, from epoch 1.
    assert len(snapshots) == 1
    assert result["best_epoch"] == 1
    saved = load_file(result["candidate_checkpoint_path"])
    for key, tensor in snapshots[0].items():
        assert torch.equal(saved[f"heads.{key}"], tensor)


def test_build_samples_dedup_and_unique_images(tmp_path):
    """Two records sharing a content_hash collapse to one sample; the preparing
    payload reports unique_images."""
    from bittrainer.backbone_trainer import _Vocab, _build_samples

    paths = _make_images(tmp_path / "imgs", ["a", "b"])
    shared = "cc" + "ab" * 31
    records = [
        {"content_hash": shared, "file_paths": [paths[0]], "binary": {"watermark": "positive"}},
        {"content_hash": shared, "file_paths": [paths[1]], "binary": {"watermark": "negative"}},
    ]
    vocab = _Vocab(records)
    samples, missing = _build_samples(records, vocab)
    assert len(samples) == 1
    assert missing == 0

    # End-to-end: unique_images rides the preparing progress payload.
    request = _request(tmp_path, n=8)
    dup = dict(request["records"][0])
    dup = {**dup, "file_paths": request["records"][1]["file_paths"]}
    request["records"].append(dup)  # duplicate content_hash of record 0
    events = []

    async def _cb(msg):
        events.append(msg)

    asyncio.run(run_backbone_training(request, progress_callback=_cb))
    preparing = [e for e in events if e.get("stage") == "preparing" and "unique_images" in e]
    assert preparing, "no preparing payload with unique_images"
    # 9 records, one a duplicate -> 8 unique images.
    assert preparing[0]["unique_images"] == 8
    assert "unique images" in preparing[0]["status_text"]


def test_result_contract_keys(tmp_path):
    """The result keys Engine's BackboneTrainingManager reads stay intact."""
    request = _request(tmp_path)
    result = _run(request)
    assert result["candidate_checkpoint_path"] == request["candidate_checkpoint_path"]
    assert isinstance(result["validation_score"], float)
    assert isinstance(result["validation_metrics"], dict)
    assert result["heads"] == request["heads"]
    assert result["release_blocking"] is False
    assert isinstance(result["epochs_completed"], int)
    assert isinstance(result["best_epoch"], int)
