"""Region head in checkpoints, warm starts and the task wiring (Bitcrush ISSUE-0993).

The region head's weights ride in the ordinary ``state_dict``, so the things to
pin are the edges: a checkpoint without one still loads exactly as before, a
checkpoint with one comes back with the head installed, the embedding-cache
identity does not move when only the head changes, and head-only training never
ships a head that was not trained on the trunk it sits on.

CPU-only, random-init ``atto`` trunk; no training loop is run.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import bittrainer.group_trainer as gt
from bittrainer.generic.task import BestTracker
from bittrainer.generic.tasks.group_task import GroupTask
from bittrainer.generic.tasks.head_only_task import settle_head_only_region_head
from bittrainer.model import (
    _infer_num_classes,
    backbone_feature_hash,
    backbone_state_hash,
    create_model,
    load_checkpoint,
)
from bittrainer.regions import (
    forward_with_regions,
    has_region_head,
    install_region_head,
    region_head_ckpt_meta,
)
from bittrainer.training_state import make_fingerprint

_TASKS_DIR = Path(__file__).resolve().parents[1] / "generic" / "tasks"


def _model(num_types: int | None = 1, seed: int = 0):
    torch.manual_seed(seed)
    model = create_model(model_size="atto", pretrained=False, num_classes=3)
    if num_types:
        install_region_head(model, num_types)
    return model.eval()


def _save(path, model, **meta):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": 3,
            "model_size": "atto",
            "class_names": ["a", "b", "c"],
            **meta,
        },
        path,
    )
    return str(path)


def _png(path, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)).save(path)
    return path


def _group_with_manifest(tmp_path, *, drawn: bool = True):
    """A two-class group folder plus a regions manifest; returns the config."""
    root = tmp_path / "g"
    a = _png(root / "a" / "train" / "a0.png", 1)
    _png(root / "b" / "train" / "b0.png", 2)
    _png(root / "a" / "val" / "a1.png", 3)
    _png(root / "b" / "val" / "b1.png", 4)
    crop = _png(root / ".region_crops" / "train" / "c0.png", 5)
    loose = _png(tmp_path / "pool" / "loose.png", 6)
    manifest = root / ".region_crops" / "regions_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "crop_pad": 0.15,
                "region_types": ["Face"],
                "crops": [
                    {
                        "path": str(crop),
                        "source": str(a),
                        "box": [0.2, 0.2, 0.8, 0.8],
                        "type": 0,
                        "class": "a",
                        "split": "train",
                    }
                ],
                "images": (
                    [
                        {
                            "source": str(a),
                            "paths": [str(a)],
                            "split": "train",
                            "boxes": [[0.2, 0.2, 0.8, 0.8, 0]],
                            "rejects": [],
                        },
                        {
                            "source": str(loose),
                            "paths": [str(loose)],
                            "split": "train",
                            "boxes": [[0.1, 0.1, 0.5, 0.5, 0]],
                            "rejects": [],
                        },
                    ]
                    if drawn
                    else []
                ),
            }
        ),
        encoding="utf-8",
    )
    return gt.GroupTrainConfig(
        group_folder=str(root),
        num_classes=2,
        class_names=["a", "b"],
        backbone_variant="atto",
        device="cpu",
        dtype="float32",
        use_cache=False,
        use_compile=False,
        checkpoint_dir=str(tmp_path / "ckpt"),
        backbone_init={"source": "random_init", "checkpoint_path": None},
        extra_paths_train={"a": [str(crop)]},
        region_manifest=str(manifest),
    )


# ---------------------------------------------------------------------------
# model.py
# ---------------------------------------------------------------------------


class TestBackboneHash:
    def test_installing_or_training_the_head_keeps_the_hash(self):
        """The embedding cache is keyed by this hash; the region head sits
        beside the pooled-feature path, never on it."""
        plain = _model(num_types=None)
        before = backbone_feature_hash(plain)
        install_region_head(plain, 2)
        assert backbone_feature_hash(plain) == before
        with torch.no_grad():
            plain.region_head.heatmap.weight.add_(1.0)
        assert backbone_feature_hash(plain) == before

    def test_state_hash_matches_the_model_hash(self):
        model = _model()
        assert backbone_state_hash(model.state_dict()) == backbone_feature_hash(model)

    def test_trunk_change_still_moves_the_hash(self):
        model = _model()
        before = backbone_feature_hash(model)
        with torch.no_grad():
            next(model.stages.parameters()).add_(0.1)
        assert backbone_feature_hash(model) != before


def test_infer_num_classes_ignores_the_region_head():
    # Five region types on a three-class classifier.
    assert _infer_num_classes(_model(num_types=5).state_dict()) == 3


class TestLoadCheckpoint:
    def test_region_checkpoint_comes_back_with_its_head(self, tmp_path):
        model = _model(num_types=2)
        path = _save(tmp_path / "best.pt", model, **region_head_ckpt_meta(model, ["Face", "Hand"]))
        loaded = load_checkpoint(path)
        assert has_region_head(loaded) and loaded.region_head.num_types == 2
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            want_logits, want = forward_with_regions(model, x)
            got_logits, got = forward_with_regions(loaded.eval(), x)
        assert torch.equal(want_logits, got_logits)
        assert torch.equal(want.heatmap, got.heatmap)

    def test_head_is_rebuilt_from_the_weights_when_the_meta_is_missing(self, tmp_path):
        model = _model(num_types=2)
        loaded = load_checkpoint(_save(tmp_path / "best.pt", model))
        assert has_region_head(loaded) and loaded.region_head.num_types == 2

    def test_legacy_checkpoint_round_trips_without_a_head(self, tmp_path):
        model = _model(num_types=None)
        loaded = load_checkpoint(_save(tmp_path / "best.pt", model)).eval()
        assert not has_region_head(loaded)
        assert set(loaded.state_dict()) == set(model.state_dict())
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            assert torch.equal(model(x), loaded(x))

    def test_bare_state_dict_checkpoint_still_loads(self, tmp_path):
        model = _model(num_types=None)
        path = tmp_path / "bare.pt"
        torch.save(model.state_dict(), path)
        loaded = load_checkpoint(str(path), model_size="atto", num_classes=3)
        assert not has_region_head(loaded)


# ---------------------------------------------------------------------------
# Run wiring
# ---------------------------------------------------------------------------


class TestResolveManifest:
    def test_resolves_when_supported(self, tmp_path):
        config = _group_with_manifest(tmp_path)
        manifest = gt.resolve_region_manifest(config)
        assert manifest is not None and manifest.type_names == ["Face"]

    def test_no_manifest_configured(self, tmp_path):
        config = _group_with_manifest(tmp_path)
        config.region_manifest = ""
        assert gt.resolve_region_manifest(config) is None

    @pytest.mark.parametrize(
        "override",
        [{"sourceless": True}, {"cell_masks": [[0], [1]]}, {"flip_class_map": [1, 0]}],
    )
    def test_unsupported_combinations_train_without_regions(self, tmp_path, override):
        config = _group_with_manifest(tmp_path)
        for key, value in override.items():
            setattr(config, key, value)
        assert gt.resolve_region_manifest(config) is None


class TestModelCreation:
    def _create(self, config, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir(exist_ok=True)
        return gt._create_or_warmstart_model(
            config,
            device=torch.device("cpu"),
            dtype=torch.float32,
            head_hidden_size=None,
            checkpoint_dir=ckpt_dir,
        )

    def test_fresh_model_gets_a_region_head(self, tmp_path):
        model = self._create(_group_with_manifest(tmp_path), tmp_path)
        assert has_region_head(model) and model.region_head.num_types == 1

    def test_no_head_without_drawn_images(self, tmp_path):
        model = self._create(_group_with_manifest(tmp_path, drawn=False), tmp_path)
        assert not has_region_head(model)

    def test_no_head_for_an_ordinary_group(self, tmp_path):
        config = _group_with_manifest(tmp_path)
        config.region_manifest = ""
        assert not has_region_head(self._create(config, tmp_path))

    def test_warm_start_carries_the_incumbents_head(self, tmp_path):
        config = _group_with_manifest(tmp_path)
        (tmp_path / "ckpt").mkdir()
        torch.manual_seed(7)
        incumbent = create_model(model_size="atto", pretrained=False, num_classes=2)
        install_region_head(incumbent, 1)
        with torch.no_grad():
            incumbent.region_head.heatmap.bias.fill_(0.37)
        torch.save(
            {"state_dict": incumbent.state_dict(), "model_size": "atto", "num_classes": 2},
            tmp_path / "ckpt" / "best.pt",
        )
        model = self._create(config, tmp_path)
        assert model.region_head.heatmap.bias.tolist() == pytest.approx([0.37])

    def test_warm_start_from_a_headless_incumbent_starts_the_head_fresh(self, tmp_path):
        config = _group_with_manifest(tmp_path)
        (tmp_path / "ckpt").mkdir()
        torch.manual_seed(7)
        incumbent = create_model(model_size="atto", pretrained=False, num_classes=2)
        torch.save(
            {"state_dict": incumbent.state_dict(), "model_size": "atto", "num_classes": 2},
            tmp_path / "ckpt" / "best.pt",
        )
        model = self._create(config, tmp_path)
        assert has_region_head(model)
        assert torch.equal(model.stem[0].weight, incumbent.stem[0].weight)


class TestPrepareDatasets:
    def test_only_the_train_set_is_region_aware(self, tmp_path):
        config = _group_with_manifest(tmp_path)
        train_ds, val_ds, _cache, _buckets = gt._prepare_datasets_and_cache(
            config, cb=lambda _msg: None, stop_event=None
        )
        assert len(train_ds[0]) == 4
        assert len(val_ds[0]) == 3
        assert any(s.get("detection_only") for s in train_ds.samples)
        assert any(s.get("view") == "crop" for s in train_ds.samples)

    def test_ordinary_group_is_untouched(self, tmp_path):
        config = _group_with_manifest(tmp_path)
        config.region_manifest = ""
        train_ds, _val, _cache, _buckets = gt._prepare_datasets_and_cache(
            config, cb=lambda _msg: None, stop_event=None
        )
        assert len(train_ds[0]) == 3


class TestGroupTask:
    def _task(self, tmp_path):
        config = _group_with_manifest(tmp_path)
        task = GroupTask(config)
        task.ema = None
        task.region_manifest = gt.resolve_region_manifest(config)
        return task

    def test_candidate_records_the_region_head(self, tmp_path):
        task = self._task(tmp_path)
        torch.manual_seed(0)
        model = create_model(model_size="atto", pretrained=False, num_classes=2)
        install_region_head(model, 1)
        ctx = SimpleNamespace(checkpoint_dir=tmp_path)
        task.save_candidate(ctx, model, 0, {"macro_f1": 0.5}, BestTracker())
        ckpt = torch.load(tmp_path / "candidate.pt", weights_only=True)
        assert ckpt["region_head"]["types"] == ["Face"]
        assert has_region_head(load_checkpoint(str(tmp_path / "candidate.pt")))

    def test_candidate_without_a_head_records_nothing(self, tmp_path):
        task = self._task(tmp_path)
        model = create_model(model_size="atto", pretrained=False, num_classes=2)
        ctx = SimpleNamespace(checkpoint_dir=tmp_path)
        task.save_candidate(ctx, model, 0, {"macro_f1": 0.5}, BestTracker())
        assert "region_head" not in torch.load(tmp_path / "candidate.pt", weights_only=True)

    def test_sample_stats_skip_crop_and_detection_only_rows(self, tmp_path):
        """sample_stats.json feeds Engine's label-issue ranking, which is about
        labelled IMAGES: crop files and unlabelled drawn images are not rows it
        can act on."""
        task = self._task(tmp_path)
        task.train_ds, _val, _cache, _buckets = gt._prepare_datasets_and_cache(
            task.config, cb=lambda _msg: None, stop_event=None
        )
        samples = task.train_ds.samples
        task._train_schedule = [list(range(len(samples)))]
        task._current_epoch = 0
        recorded: dict = {}

        def _record(epoch, paths, labels, losses):
            recorded.update(paths=paths, labels=labels, losses=losses)

        task.sample_stats.record_train = _record
        task._make_sample_loss_sink()(0, torch.arange(len(samples), dtype=torch.float32))
        kept = [
            i
            for i, s in enumerate(samples)
            if not s.get("detection_only") and s.get("view") != "crop"
        ]
        assert 0 < len(kept) < len(samples)
        assert recorded["paths"] == [samples[i]["path"] for i in kept]
        assert recorded["losses"] == [float(i) for i in kept]
        assert all(label >= 0 for label in recorded["labels"])


def test_fingerprint_folds_in_the_region_types():
    base = dict(
        class_names=["a", "b"],
        num_classes=2,
        max_epochs=5,
        multi_label=False,
        ordinal=False,
        best_model_name="best.pt",
        model_size="atto",
    )
    assert "region_types" not in make_fingerprint(**base)
    assert make_fingerprint(**base, region_types=["Face"])["region_types"] == ["Face"]


# ---------------------------------------------------------------------------
# Head-only: never ship a head the trunk was not trained with
# ---------------------------------------------------------------------------


class TestHeadOnlyRegionHead:
    def test_headless_model_reports_none(self, tmp_path):
        model = _model(num_types=None)
        assert settle_head_only_region_head(model, tmp_path / "best.pt") == "none"

    def test_matching_trunk_keeps_the_incumbents_head(self, tmp_path):
        incumbent = _model(num_types=1)
        path = _save(tmp_path / "best.pt", incumbent)
        model = _model(num_types=1)  # same seed: the same trunk and head
        assert settle_head_only_region_head(model, path) == "kept"
        assert has_region_head(model)

    def test_different_trunk_drops_the_head(self, tmp_path):
        path = _save(tmp_path / "best.pt", _model(num_types=1, seed=0))
        model = _model(num_types=1, seed=1)
        assert settle_head_only_region_head(model, path) == "needs_full_train"
        assert not has_region_head(model)
        assert not any(k.startswith("region_head.") for k in model.state_dict())

    def test_headless_incumbent_drops_the_fresh_head(self, tmp_path):
        path = _save(tmp_path / "best.pt", _model(num_types=None))
        model = _model(num_types=1)
        assert settle_head_only_region_head(model, path) == "needs_full_train"
        assert not has_region_head(model)

    def test_no_incumbent_drops_the_fresh_head(self, tmp_path):
        model = _model(num_types=1)
        assert settle_head_only_region_head(model, tmp_path / "missing.pt") == "needs_full_train"
        assert not has_region_head(model)


# ---------------------------------------------------------------------------
# Structural guard: every group checkpoint literal records the region head
# ---------------------------------------------------------------------------


def _checkpoint_dict_literals(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        if "state_dict" in keys:
            yield node


def _records_region_head(node: ast.Dict) -> bool:
    for key, value in zip(node.keys, node.values, strict=True):
        if key is None and isinstance(value, ast.Call):
            func = value.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in ("region_head_ckpt_meta", "_region_meta"):
                return True
    return False


@pytest.mark.parametrize("name", ["group_task.py", "head_only_task.py"])
def test_every_group_checkpoint_records_the_region_head(name):
    tree = ast.parse((_TASKS_DIR / name).read_text(encoding="utf-8"))
    literals = list(_checkpoint_dict_literals(tree))
    assert literals, f"{name}: no checkpoint dict found — guard is stale"
    missing = [f"{name}:{n.lineno}" for n in literals if not _records_region_head(n)]
    assert not missing, f"checkpoints without the region head meta: {missing}"
