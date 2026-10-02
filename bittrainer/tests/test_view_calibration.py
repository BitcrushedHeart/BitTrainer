"""Per-view calibration, view selection and proposer threshold (Bitcrush ISSUE-0993).

A crop-assisted checkpoint is served two ways: on the whole image and on the
crops its own region head proposes. The two views are calibrated separately at
finalisation, a margin decides when a crop may overrule the whole image, and
the proposer's score threshold is tuned against the drawn validation boxes.

CPU-only. The pure tuning functions run on synthetic logits; one integration
test drives a random-init ``atto`` checkpoint through the real finaliser.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

import bittrainer.group_trainer as gt
from bittrainer.group_dataset import GroupDataset
from bittrainer.model import create_model
from bittrainer.region_manifest import load_region_manifest
from bittrainer.regions import RegionProposal, install_region_head, region_head_ckpt_meta
from bittrainer.view_calibration import (
    DEFAULT_CROP_MARGIN,
    DEFAULT_CROP_TRIGGER,
    MIN_VIEW_VAL_CROPS,
    box_iou,
    combine_exclusive,
    combine_multilabel,
    finalize_region_views,
    fit_crop_calibration,
    match_regions,
    ordinal_radius,
    tune_score_threshold,
    tune_view_selection,
    view_confidence,
)


def _config(**kw) -> gt.GroupTrainConfig:
    base = dict(group_folder=".", num_classes=3, class_names=["a", "b", "c"])
    base.update(kw)
    return gt.GroupTrainConfig(**base)


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_plain_groups_use_the_predicted_class_probability(self):
        assert view_confidence([0.1, 0.7, 0.2], 1, ordinal=False) == pytest.approx(0.7)

    def test_ordinal_groups_sum_the_neighbourhood(self):
        """An ordinal head spreads mass over adjacent classes on purpose, so
        one bin's probability understates how sure it is."""
        probs = [0.05, 0.25, 0.4, 0.25, 0.05]
        assert view_confidence(probs, 2, ordinal=True, radius=1) == pytest.approx(0.9)
        assert view_confidence(probs, 0, ordinal=True, radius=1) == pytest.approx(0.3)

    def test_ordinal_neighbourhood_skips_the_none_class(self):
        probs = [0.3, 0.4, 0.3]  # index 0 is __none__
        assert view_confidence(probs, 1, ordinal=True, none_index=0, radius=1) == pytest.approx(0.7)

    @pytest.mark.parametrize(
        ("sigma", "classes", "radius"),
        [(None, 10, 2), (0.05, 10, 1), (0.3, 10, 2), (1.0, 10, 3), (1.0, 2, 2), (0.3, 1, 1)],
    )
    def test_ordinal_radius_follows_the_label_softness(self, sigma, classes, radius):
        assert ordinal_radius(sigma, classes) == radius


# ---------------------------------------------------------------------------
# Crop-view calibration
# ---------------------------------------------------------------------------


def _overconfident(n: int, num_classes: int, *, accuracy: float, scale: float, seed: int = 0):
    """Logits that are right ``accuracy`` of the time but always very sure."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(1, num_classes, size=n)
    preds = labels.copy()
    wrong = rng.random(n) > accuracy
    preds[wrong] = 1 + (labels[wrong] - 1 + 1) % (num_classes - 1)
    logits = np.zeros((n, num_classes), dtype=np.float32)
    logits[np.arange(n), preds] = scale
    return torch.from_numpy(logits), torch.from_numpy(labels)


class TestCropCalibration:
    FULL = {"temperature": 1.15, "class_logit_bias": [0.05, 0.0, 0.0]}

    def test_too_few_crops_fall_back_to_the_full_view(self):
        config = _config(class_names=["__none__", "a", "b"])
        logits, labels = _overconfident(MIN_VIEW_VAL_CROPS - 1, 3, accuracy=0.8, scale=6.0)
        out = fit_crop_calibration(logits, labels, config, 0, full=self.FULL)
        assert out["fallback"] is True and out["n_val"] == MIN_VIEW_VAL_CROPS - 1
        assert out["temperature"] == pytest.approx(1.15)
        assert out["class_logit_bias"] == pytest.approx([0.05, 0.0, 0.0])

    def test_no_crops_at_all_fall_back(self):
        config = _config(class_names=["__none__", "a", "b"])
        out = fit_crop_calibration(None, None, config, 0, full=self.FULL)
        assert out["fallback"] is True and out["n_val"] == 0

    def test_overconfident_crop_view_gets_its_own_temperature(self):
        config = _config(class_names=["__none__", "a", "b"])
        logits, labels = _overconfident(200, 3, accuracy=0.7, scale=6.0)
        out = fit_crop_calibration(logits, labels, config, 0, full=self.FULL)
        assert out["fallback"] is False and out["n_val"] == 200
        assert out["temperature"] > 1.15
        assert len(out["class_logit_bias"]) == 3

    def test_multi_label_fits_per_class_thresholds(self):
        config = _config(multi_label=True)
        rng = np.random.default_rng(0)
        labels = (rng.random((120, 3)) > 0.5).astype(np.float32)
        # Class 0 fires high for positives, class 1 only weakly, class 2 at 0.5.
        probs = np.stack(
            [
                np.where(labels[:, 0] > 0, 0.9, 0.6),
                np.where(labels[:, 1] > 0, 0.3, 0.1),
                np.where(labels[:, 2] > 0, 0.7, 0.3),
            ],
            axis=1,
        )
        logits = torch.from_numpy(np.log(probs / (1 - probs)).astype(np.float32))
        out = fit_crop_calibration(
            logits, torch.from_numpy(labels), config, -1, full={"per_class_thresholds": [0.5] * 3}
        )
        assert out["fallback"] is False
        thresholds = out["per_class_thresholds"]
        assert 0.6 <= thresholds[0] <= 0.9
        assert 0.1 <= thresholds[1] <= 0.3

    def test_multi_label_falls_back_to_the_full_thresholds(self):
        config = _config(multi_label=True)
        out = fit_crop_calibration(
            torch.zeros(4, 3),
            torch.zeros(4, 3),
            config,
            -1,
            full={"per_class_thresholds": [0.4, 0.5, 0.6]},
        )
        assert out["fallback"] is True
        assert out["per_class_thresholds"] == pytest.approx([0.4, 0.5, 0.6])


# ---------------------------------------------------------------------------
# Combining the views
# ---------------------------------------------------------------------------


class TestCombineExclusive:
    def test_more_confident_crop_wins(self):
        full = [0.5, 0.3, 0.2]
        crops = [[0.1, 0.8, 0.1], [0.2, 0.6, 0.2]]
        out = combine_exclusive(full, crops, margin=0.0)
        assert out.view == "crop" and out.crop_index == 0
        assert out.predicted == 1 and out.probs == pytest.approx(crops[0])

    def test_margin_keeps_the_full_view(self):
        out = combine_exclusive([0.5, 0.3, 0.2], [[0.1, 0.55, 0.35]], margin=0.1)
        assert out.view == "full" and out.crop_index is None and out.predicted == 0

    def test_equal_confidence_goes_to_the_crop_at_zero_margin(self):
        out = combine_exclusive([0.6, 0.4], [[0.4, 0.6]], margin=0.0)
        assert out.view == "crop" and out.predicted == 1

    def test_no_crops_is_the_full_view(self):
        out = combine_exclusive([0.2, 0.8], [], margin=0.0)
        assert out.view == "full" and out.predicted == 1

    def test_a_crop_that_says_none_never_overrules(self):
        """Regions never carry ``__none__``: a crop that looks like nothing is
        a bad crop, not evidence that the image is empty."""
        out = combine_exclusive([0.2, 0.5, 0.3], [[0.95, 0.03, 0.02]], margin=0.0, none_index=0)
        assert out.view == "full" and out.predicted == 1

    def test_trigger_skips_crops_for_a_sure_full_view(self):
        out = combine_exclusive([0.92, 0.08], [[0.01, 0.99]], margin=0.0, trigger=0.9)
        assert out.view == "full"


def test_combine_multilabel_takes_the_per_class_best_view():
    out = combine_multilabel(
        [0.2, 0.7, 0.1],
        [[0.9, 0.1, 0.1], [0.1, 0.1, 0.45]],
        full_thresholds=[0.5, 0.5, 0.5],
        crop_thresholds=[0.6, 0.6, 0.4],
    )
    assert out.predicted_classes == [0, 1, 2]
    assert out.views == ["crop", "full", "crop"]
    assert out.probs == pytest.approx([0.9, 0.7, 0.45])


class TestViewSelection:
    def _data(self, n, *, crop_right: bool, seed=0):
        """Full view: a coin flip at 0.55 confidence. Crop view: confident, and
        right or wrong on every image."""
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, 3, size=n)
        full = np.full((n, 3), 0.225, dtype=np.float64)
        guess = np.where(rng.random(n) < 0.5, labels, (labels + 1) % 3)
        full[np.arange(n), guess] = 0.55
        crops = []
        for i in range(n):
            target = labels[i] if crop_right else (labels[i] + 2) % 3
            row = np.full(3, 0.05)
            row[target] = 0.9
            crops.append(row[None, :])
        return full, crops, labels

    def test_helpful_crops_are_let_through(self):
        full, crops, labels = self._data(120, crop_right=True)
        selection, metrics = tune_view_selection(full, crops, labels, _config(), -1)
        assert selection["tuned"] is True
        assert selection["crop_margin"] <= 0.3
        assert metrics["combined"]["macro_f1"] > metrics["full"]["macro_f1"] + 0.3

    def test_harmful_crops_are_margined_out(self):
        full, crops, labels = self._data(120, crop_right=False)
        selection, metrics = tune_view_selection(full, crops, labels, _config(), -1)
        assert selection["crop_margin"] > 0.3  # 0.9 never clears 0.55 + margin
        assert metrics["combined"]["macro_f1"] == pytest.approx(metrics["full"]["macro_f1"])

    def test_too_few_images_keep_the_defaults(self):
        full, crops, labels = self._data(MIN_VIEW_VAL_CROPS - 1, crop_right=True)
        selection, _metrics = tune_view_selection(full, crops, labels, _config(), -1)
        assert selection["tuned"] is False
        assert selection["crop_margin"] == DEFAULT_CROP_MARGIN
        assert selection["crop_trigger"] == DEFAULT_CROP_TRIGGER

    def test_trigger_drops_when_sure_full_views_never_need_a_crop(self):
        rng = np.random.default_rng(1)
        n = 120
        labels = rng.integers(0, 3, size=n)
        full = np.full((n, 3), 0.02)
        full[np.arange(n), labels] = 0.96  # the full view is sure and right
        crops = [full[i][None, :].copy() for i in range(n)]
        selection, _metrics = tune_view_selection(full, crops, labels, _config(), -1)
        assert selection["crop_trigger"] < 0.96

    def test_selection_records_how_confidence_is_measured(self):
        full, crops, labels = self._data(60, crop_right=True)
        plain, _ = tune_view_selection(full, crops, labels, _config(), -1)
        assert plain["confidence"] == "top1"
        ordinal, _ = tune_view_selection(
            full, crops, labels, _config(ordinal=True, ordinal_sigma=0.3), -1
        )
        assert ordinal["confidence"] == "ordinal_neighbourhood" and ordinal["radius"] == 2


# ---------------------------------------------------------------------------
# Proposer threshold
# ---------------------------------------------------------------------------


def _p(score, box, kind=0):
    return RegionProposal(type=kind, score=score, box=box)


class TestDetectorTuning:
    def test_iou(self):
        assert box_iou((0, 0, 1, 1), (0, 0, 1, 1)) == pytest.approx(1.0)
        assert box_iou((0, 0, 0.5, 1), (0.5, 0, 1, 1)) == 0.0
        assert box_iou((0, 0, 0.5, 1), (0.25, 0, 0.75, 1)) == pytest.approx(1 / 3)

    def test_match_counts_hits_misses_and_extras(self):
        gt_boxes = [(0.1, 0.1, 0.4, 0.4, 0), (0.6, 0.6, 0.9, 0.9, 0)]
        proposals = [
            _p(0.9, (0.1, 0.1, 0.4, 0.4)),
            _p(0.8, (0.11, 0.1, 0.4, 0.41)),  # a second box on the same region
            _p(0.7, (0.0, 0.6, 0.2, 0.9)),  # nothing there
        ]
        assert match_regions(proposals, gt_boxes) == (1, 2, 1)

    def test_a_box_of_the_wrong_type_does_not_match(self):
        assert match_regions([_p(0.9, (0.1, 0.1, 0.4, 0.4), kind=1)], [(0.1, 0.1, 0.4, 0.4, 0)]) == (
            0,
            1,
            1,
        )

    def test_threshold_lands_between_real_and_spurious_scores(self):
        images = []
        for _ in range(20):
            images.append(
                {
                    "proposals": [
                        _p(0.8, (0.1, 0.1, 0.4, 0.4)),
                        _p(0.25, (0.6, 0.6, 0.9, 0.9)),
                    ],
                    "boxes": [(0.1, 0.1, 0.4, 0.4, 0)],
                    "supervised": {0},
                }
            )
        threshold, metrics = tune_score_threshold(images)
        assert 0.25 < threshold <= 0.8
        assert metrics["tuned"] is True and metrics["f1"] == pytest.approx(1.0)
        assert metrics["n_val_boxes"] == 20

    def test_unsupervised_types_are_not_scored(self):
        """A proposal of a type the user never reviewed on that image is
        unknown, not a false positive."""
        images = [
            {
                "proposals": [_p(0.9, (0.1, 0.1, 0.4, 0.4)), _p(0.9, (0.5, 0.5, 0.9, 0.9), kind=1)],
                "boxes": [(0.1, 0.1, 0.4, 0.4, 0)],
                "supervised": {0},
            }
        ]
        _threshold, metrics = tune_score_threshold(images)
        assert metrics["precision"] == pytest.approx(1.0)

    def test_no_validation_boxes_keeps_the_default(self):
        threshold, metrics = tune_score_threshold(
            [{"proposals": [_p(0.9, (0, 0, 1, 1))], "boxes": [], "supervised": set()}]
        )
        assert threshold == pytest.approx(0.3) and metrics["tuned"] is False


# ---------------------------------------------------------------------------
# The finaliser, end to end on a random-init checkpoint
# ---------------------------------------------------------------------------


def _png(path, size=(64, 64), seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)).save(path)
    return path


@pytest.fixture
def region_run(tmp_path):
    root = tmp_path / "g"
    names = ["__none__", "a", "b"]
    crops, images = [], []
    seed = 0
    for cls in ("a", "b"):
        for j in range(3):
            seed += 1
            img = _png(root / cls / "val" / f"{cls}{j}.png", seed=seed)
            crop = _png(root / ".region_crops" / "val" / f"{cls}{j}c.png", size=(40, 40), seed=seed + 50)
            crops.append(
                {
                    "path": str(crop),
                    "source": str(img),
                    "box": [0.2, 0.2, 0.8, 0.8],
                    "type": 0,
                    "class": cls,
                    "split": "val",
                }
            )
            images.append(
                {
                    "source": str(img),
                    "paths": [str(img)],
                    "split": "val",
                    "boxes": [[0.2, 0.2, 0.8, 0.8, 0]],
                    "rejects": [],
                }
            )
        _png(root / cls / "train" / f"{cls}t.png", seed=seed + 100)
    manifest_path = root / ".region_crops" / "regions_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "crop_pad": 0.15,
                "region_types": ["Face"],
                "crops": crops,
                "images": images,
            }
        ),
        encoding="utf-8",
    )
    config = gt.GroupTrainConfig(
        group_folder=str(root),
        num_classes=3,
        class_names=names,
        backbone_variant="atto",
        device="cpu",
        dtype="float32",
        channels_last=False,
        use_cache=False,
        region_manifest=str(manifest_path),
    )
    val_ds = GroupDataset(root, names, split="val", group_name="g")
    return config, load_region_manifest(manifest_path), val_ds


def _checkpoint(path, *, region: bool, **meta):
    torch.manual_seed(0)
    model = create_model(model_size="atto", pretrained=False, num_classes=3)
    if region:
        install_region_head(model, 1)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": 3,
            "model_size": "atto",
            "class_names": ["__none__", "a", "b"],
            "temperature": 1.3,
            "class_logit_bias": [0.1, 0.0, 0.0],
            **region_head_ckpt_meta(model, ["Face"]),
            **meta,
        },
        path,
    )
    return str(path)


class TestFinalizer:
    def test_persists_the_region_keys_and_leaves_the_full_view_alone(self, tmp_path, region_run):
        config, manifest, val_ds = region_run
        path = _checkpoint(tmp_path / "best.pt", region=True)
        summary = finalize_region_views(
            config,
            checkpoint_path=path,
            manifest=manifest,
            val_ds=val_ds,
            device=torch.device("cpu"),
            dtype=torch.float32,
            batch_size=4,
        )
        assert summary["status"] == "ok"
        ckpt = torch.load(path, weights_only=True)
        # Old Engine builds read these two: they are still the full view.
        assert ckpt["temperature"] == pytest.approx(1.3)
        assert ckpt["class_logit_bias"] == pytest.approx([0.1, 0.0, 0.0])
        decode = ckpt["region_decode"]
        assert 0.0 < decode["score_threshold"] < 1.0
        assert decode["max_regions"] >= 1 and decode["crop_pad"] == pytest.approx(0.15)
        calibration = ckpt["view_calibration"]
        assert calibration["full"]["temperature"] == pytest.approx(1.3)
        # Six validation crops are far under the minimum: the crop view borrows.
        assert calibration["crop"]["fallback"] is True
        assert calibration["crop"]["temperature"] == pytest.approx(1.3)
        selection = ckpt["view_selection"]
        assert selection["crop_margin"] == DEFAULT_CROP_MARGIN and selection["tuned"] is False
        metrics = ckpt["view_metrics"]
        assert metrics["n_val_crops"] == 6 and metrics["n_val_images_with_crops"] == 6
        assert metrics["detector"]["n_val_boxes"] == 6
        assert summary["view_metrics"] == metrics

    def test_checkpoint_without_a_region_head_is_left_untouched(self, tmp_path, region_run):
        config, manifest, val_ds = region_run
        path = _checkpoint(tmp_path / "best.pt", region=False)
        before = (tmp_path / "best.pt").read_bytes()
        summary = finalize_region_views(
            config,
            checkpoint_path=path,
            manifest=manifest,
            val_ds=val_ds,
            device=torch.device("cpu"),
            dtype=torch.float32,
            batch_size=4,
        )
        assert summary == {"status": "no_region_head"}
        assert (tmp_path / "best.pt").read_bytes() == before

    def test_missing_checkpoint_reports_instead_of_raising(self, tmp_path, region_run):
        config, manifest, val_ds = region_run
        summary = finalize_region_views(
            config,
            checkpoint_path=str(tmp_path / "missing.pt"),
            manifest=manifest,
            val_ds=val_ds,
            device=torch.device("cpu"),
            dtype=torch.float32,
            batch_size=4,
        )
        assert summary["status"] == "error"


# ---------------------------------------------------------------------------
# Task wiring: the finaliser runs after promotion, for region runs only
# ---------------------------------------------------------------------------


class TestTaskWiring:
    def _ctx(self, tmp_path):
        from types import SimpleNamespace

        return SimpleNamespace(
            device=torch.device("cpu"), dtype=torch.float32, cb=lambda _m: None,
            checkpoint_dir=tmp_path,
        )

    def test_group_task_reports_the_region_finalisation(self, tmp_path, region_run, monkeypatch):
        import bittrainer.generic.tasks.group_task as group_task

        config, manifest, val_ds = region_run
        seen = {}

        def _fake(cfg, **kwargs):
            seen.update(kwargs)
            return {"status": "ok", "view_metrics": {"n_val_crops": 6}}

        monkeypatch.setattr(group_task, "finalize_region_views", _fake)
        task = group_task.GroupTask(config)
        task.region_manifest, task.val_ds, task.eff_bs = manifest, val_ds, 8
        result = {"checkpoint_path": "best.pt"}
        task._finalize_regions(self._ctx(tmp_path), result)
        assert result["region_training"]["status"] == "ok"
        assert seen["checkpoint_path"] == "best.pt" and seen["batch_size"] == 8
        assert seen["manifest"] is manifest and seen["val_ds"] is val_ds

    def test_ordinary_group_never_runs_it(self, tmp_path, region_run, monkeypatch):
        import bittrainer.generic.tasks.group_task as group_task

        config, _manifest, val_ds = region_run

        def _boom(*a, **k):
            raise AssertionError("ordinary group ran the region finaliser")

        monkeypatch.setattr(group_task, "finalize_region_views", _boom)
        task = group_task.GroupTask(config)
        task.val_ds = val_ds
        result = {"checkpoint_path": "best.pt"}
        task._finalize_regions(self._ctx(tmp_path), result)
        assert "region_training" not in result

    def test_head_only_task_reports_it_too(self, tmp_path, region_run, monkeypatch):
        import bittrainer.generic.tasks.head_only_task as head_only_task

        config, manifest, val_ds = region_run
        monkeypatch.setattr(
            head_only_task, "finalize_region_views", lambda cfg, **kw: {"status": "no_region_head"}
        )
        task = head_only_task.HeadOnlyTask(config)
        task.region_manifest, task.val_ds = manifest, val_ds
        result = {"checkpoint_path": "best.pt"}
        task._finalize_regions(self._ctx(tmp_path), result)
        assert result["region_training"] == {"status": "no_region_head"}
