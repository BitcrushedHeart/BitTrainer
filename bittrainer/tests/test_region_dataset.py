"""Region-aware group dataset (Bitcrush ISSUE-0993).

The Engine writes ``regions_manifest.json`` beside the materialised crop files.
The dataset reads it to (a) tag crop files as the crop view of their source
image, (b) attach each drawn image's boxes to its full-image sample, mapped
through the same aspect crop the pixels go through, and (c) add drawn images
that carry no usable class label as detection-only samples.

CPU-only; images are tiny synthetic PNGs.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch
from PIL import Image

from bittrainer.face_crop import aspect_crop_rect, face_aware_crop
from bittrainer.group_dataset import GroupDataset, labelled_samples
from bittrainer.region_manifest import (
    RegionManifest,
    load_region_manifest,
    map_box_from_rect,
    map_box_into_rect,
)


def _png(path, size=(64, 64), seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)).save(path)
    return path


def _manifest(tmp_path, *, crops=(), images=(), types=("Face",), crop_pad=0.15):
    path = tmp_path / "g" / ".region_crops" / "regions_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "crop_pad": crop_pad,
                "region_types": list(types),
                "crops": list(crops),
                "images": list(images),
            }
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# aspect_crop_rect: the crop geometry, split out of face_aware_crop
# ---------------------------------------------------------------------------


class TestAspectCropRect:
    @pytest.mark.parametrize(
        ("size", "ratio", "bbox"),
        [
            ((200, 100), 736 / 352, None),
            ((100, 400), 320 / 800, None),
            ((640, 480), 1.0, None),
            ((640, 480), 1.0, [400, 100, 600, 300]),
            ((300, 900), 448 / 576, [10, 700, 120, 880]),
            ((512, 512), 1.0, [0, 0, 10, 10]),
        ],
    )
    def test_rect_is_the_crop_face_aware_crop_takes(self, size, ratio, bbox):
        rng = np.random.default_rng(3)
        img = Image.fromarray(
            rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
        )
        left, top, w, h = aspect_crop_rect(size[0], size[1], ratio, bbox)
        expected = img.crop((left, top, left + w, top + h))
        actual = face_aware_crop(img, ratio, bbox)
        assert actual.size == expected.size
        assert np.array_equal(np.asarray(actual), np.asarray(expected))

    def test_matching_aspect_is_the_whole_image(self):
        assert aspect_crop_rect(512, 512, 1.0, None) == (0, 0, 512, 512)

    def test_wide_image_into_a_wider_bucket_trims_height(self):
        # 200x100 into 736x352: ratio 2.09 > 2.0, so rows are cut, centred.
        assert aspect_crop_rect(200, 100, 736 / 352, None) == (0, 2, 200, 95)


class TestBoxMapping:
    def test_into_rect_rescales_to_the_crop(self):
        box = map_box_into_rect((0.25, 0.2, 0.75, 0.8), 200, 100, (0, 2, 200, 95))
        assert box == pytest.approx((0.25, 18 / 95, 0.75, 78 / 95))

    def test_box_cut_by_the_crop_is_clipped(self):
        # Rows 75..325 of a 100x400 image survive; the box spans rows 40..200.
        box = map_box_into_rect((0.0, 0.1, 1.0, 0.5), 100, 400, (0, 75, 100, 250))
        assert box == pytest.approx((0.0, 0.0, 1.0, 0.5))

    def test_box_outside_the_crop_is_dropped(self):
        assert map_box_into_rect((0.0, 0.0, 1.0, 0.1), 100, 400, (0, 75, 100, 250)) is None

    def test_box_mostly_outside_the_crop_is_dropped(self):
        # Rows 0..100: only 25 of 100 rows are inside the crop.
        assert map_box_into_rect((0.0, 0.0, 1.0, 0.25), 100, 400, (0, 75, 100, 250)) is None

    def test_from_rect_inverts_into_rect(self):
        rect = (10, 20, 300, 150)
        box = (0.2, 0.3, 0.6, 0.7)
        inside = map_box_into_rect(box, 400, 200, rect)
        assert map_box_from_rect(inside, 400, 200, rect) == pytest.approx(box)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestManifest:
    def test_missing_or_broken_manifest_is_none(self, tmp_path):
        assert load_region_manifest("") is None
        assert load_region_manifest(tmp_path / "nope.json") is None
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        assert load_region_manifest(bad) is None

    def test_loads_types_crops_and_images(self, tmp_path):
        path = _manifest(
            tmp_path,
            types=("Face", "Hand"),
            crops=[
                {
                    "path": str(tmp_path / "c1.png"),
                    "source": str(tmp_path / "src.png"),
                    "box": [0.1, 0.1, 0.5, 0.5],
                    "type": 1,
                    "class": "blue",
                    "split": "val",
                }
            ],
            images=[
                {
                    "source": str(tmp_path / "src.png"),
                    "paths": [str(tmp_path / "src.png"), str(tmp_path / "copy.png")],
                    "split": "val",
                    "boxes": [[0.1, 0.1, 0.5, 0.5, 1]],
                    "rejects": [[0.6, 0.6, 0.9, 0.9, 0]],
                }
            ],
        )
        manifest = load_region_manifest(path)
        assert isinstance(manifest, RegionManifest)
        assert manifest.type_names == ["Face", "Hand"]
        assert manifest.num_types == 2
        assert manifest.crop_pad == pytest.approx(0.15)
        crop = manifest.crop_for(str(tmp_path / "c1.png"))
        assert crop.class_name == "blue" and crop.type == 1 and crop.split == "val"
        # Either path of the image finds its entry, whatever the slash style.
        entry = manifest.image_for(str(tmp_path / "copy.png").replace("\\", "/"))
        assert entry is not None and entry.boxes == ((0.1, 0.1, 0.5, 0.5, 1),)
        assert entry.rejects == ((0.6, 0.6, 0.9, 0.9, 0),)
        assert manifest.crop_paths_by_class("val") == {"blue": [str(tmp_path / "c1.png")]}
        assert manifest.crop_paths_by_class("train") == {}

    def test_untyped_manifest_gets_one_generic_type(self, tmp_path):
        manifest = load_region_manifest(_manifest(tmp_path, types=()))
        assert manifest.num_types == 1


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@pytest.fixture
def group(tmp_path):
    """Two classes; ``a/train/wide.png`` is a 200x100 image with one drawn box."""
    root = tmp_path / "g"
    wide = _png(root / "a" / "train" / "wide.png", size=(200, 100), seed=1)
    plain = _png(root / "a" / "train" / "plain.png", seed=2)
    other = _png(root / "b" / "train" / "other.png", seed=3)
    crop = _png(root / ".region_crops" / "train" / "aaaa.png", size=(40, 30), seed=4)
    loose = _png(tmp_path / "pool" / "unlabelled.png", size=(80, 80), seed=5)
    val_img = _png(tmp_path / "pool" / "val_drawn.png", size=(80, 80), seed=6)
    manifest_path = _manifest(
        tmp_path,
        crops=[
            {
                "path": str(crop),
                "source": str(wide),
                "box": [0.25, 0.2, 0.75, 0.8],
                "type": 0,
                "class": "b",
                "split": "train",
            }
        ],
        images=[
            {
                "source": str(tmp_path / "pool" / "wide_source.png"),
                "paths": [str(tmp_path / "pool" / "wide_source.png"), str(wide)],
                "split": "train",
                "boxes": [[0.25, 0.2, 0.75, 0.8, 0]],
                "rejects": [],
            },
            {
                "source": str(loose),
                "paths": [str(loose)],
                "split": "train",
                "boxes": [[0.1, 0.1, 0.6, 0.6, 0]],
                "rejects": [[0.7, 0.7, 0.9, 0.9, 0]],
            },
            {
                "source": str(val_img),
                "paths": [str(val_img)],
                "split": "val",
                "boxes": [],
                "rejects": [[0.2, 0.2, 0.4, 0.4, 0]],
            },
        ],
    )
    return {
        "root": root,
        "wide": wide,
        "plain": plain,
        "other": other,
        "crop": crop,
        "loose": loose,
        "val_img": val_img,
        "manifest": load_region_manifest(manifest_path),
    }


def _dataset(group, **kwargs):
    random.seed(0)
    kwargs.setdefault("extra_paths", {"b": [str(group["crop"])]})
    return GroupDataset(
        group["root"],
        ["a", "b"],
        split=kwargs.pop("split", "train"),
        group_name="g",
        region_manifest=group["manifest"],
        **kwargs,
    )


def _by_path(ds, path):
    (index,) = [i for i, s in enumerate(ds.samples) if s["path"] == str(path)]
    return index


class TestRegionDataset:
    def test_plain_dataset_still_yields_triples(self, group):
        ds = GroupDataset(group["root"], ["a", "b"], split="train", group_name="g")
        assert len(ds[0]) == 3

    def test_region_crops_folder_is_not_scanned(self, group):
        """Crop files live under the group folder but only train when listed."""
        ds = GroupDataset(group["root"], ["a", "b"], split="train", group_name="g")
        assert str(group["crop"]) not in {s["path"] for s in ds.samples}

    def test_crop_file_is_tagged_with_its_view_and_source(self, group):
        ds = _dataset(group)
        sample = ds.samples[_by_path(ds, group["crop"])]
        assert sample["view"] == "crop"
        assert sample["source_path"] == str(group["wide"])
        assert sample["label"] == 1
        assert ds.samples[_by_path(ds, group["plain"])]["view"] == "full"

    def test_drawn_image_carries_boxes_in_the_tensor_frame(self, group):
        ds = _dataset(group)
        tensor, label, bucket, info = ds[_by_path(ds, group["wide"])]
        assert bucket == (736, 352) and tuple(tensor.shape[-2:]) == (352, 736)
        assert label == 0 and info["cls"] is True
        assert info["supervised"].tolist() == [True]
        assert info["types"].tolist() == [0]
        # 200x100 -> rows 2..97 survive the aspect crop (see TestAspectCropRect).
        assert info["boxes"][0].tolist() == pytest.approx([0.25, 18 / 95, 0.75, 78 / 95])

    def test_undrawn_and_crop_samples_are_not_detector_supervised(self, group):
        ds = _dataset(group)
        for path in (group["plain"], group["crop"]):
            info = ds[_by_path(ds, path)][3]
            assert info["supervised"].tolist() == [False]
            assert info["boxes"].shape == (0, 4)
            assert info["cls"] is True

    def test_unlabelled_drawn_image_is_a_detection_only_sample(self, group):
        ds = _dataset(group)
        index = _by_path(ds, group["loose"])
        sample = ds.samples[index]
        assert sample["detection_only"] is True and sample["label"] == -100
        _tensor, label, _bucket, info = ds[index]
        assert label == -100 and info["cls"] is False
        assert info["supervised"].tolist() == [True]
        assert info["boxes"].shape == (1, 4)  # the reject is not a positive
        assert index in ds.epoch_indices()

    def test_detection_only_rows_stay_out_of_class_accounting(self, group):
        ds = _dataset(group)
        assert ds.get_class_counts() == {0: 2, 1: 2}
        assert set(ds.get_effective_class_counts()) == {0, 1}
        labelled = labelled_samples(ds.samples)
        assert str(group["loose"]) not in {s["path"] for s in labelled}
        assert len(labelled) == len(ds.samples) - 1

    def test_split_filters_the_manifest(self, group):
        train = _dataset(group)
        assert str(group["val_img"]) not in {s["path"] for s in train.samples}
        val = _dataset(group, split="val", extra_paths={})
        paths = {s["path"] for s in val.samples}
        assert str(group["val_img"]) in paths and str(group["loose"]) not in paths

    def test_rejects_alone_supervise_an_empty_image(self, group):
        val = _dataset(group, split="val", extra_paths={})
        info = val[_by_path(val, group["val_img"])][3]
        assert info["supervised"].tolist() == [True]
        assert info["boxes"].shape == (0, 4)

    def test_scan_disk_false_reads_only_the_listed_paths(self, group):
        ds = _dataset(group, scan_disk=False)
        assert {s["path"] for s in labelled_samples(ds.samples)} == {str(group["crop"])}

    def test_multi_label_detection_only_has_an_empty_target(self, group):
        ds = _dataset(group, multi_label=True)
        index = _by_path(ds, group["loose"])
        _tensor, label, _bucket, info = ds[index]
        assert torch.is_tensor(label) and label.tolist() == [0.0, 0.0]
        assert info["cls"] is False
        assert index in ds.epoch_indices()

    def test_resample_keeps_the_region_annotations(self, group):
        ds = _dataset(group)
        ds.reshuffle()
        ds.set_oversample_none(True)
        assert ds.samples[_by_path(ds, group["crop"])]["view"] == "crop"
        assert _by_path(ds, group["loose"]) in ds.epoch_indices()


def test_crop_takes_its_source_images_skin_tone_view(group):
    """A crop of an image is colour-normalised with that image's transform:
    the Skin Tone dual-view manifest is keyed by the SOURCE image."""
    from bittrainer.skin_tone_views import SkinToneViewBank

    ds = _dataset(group)
    index = _by_path(ds, group["crop"])
    plain = ds[index][0].clone()
    ds.skin_tone_views = SkinToneViewBank(
        {str(group["wide"]): {"wb_gains": [1.0, 1.0, 1.0], "exposure_gain": 1.5}}
    )
    ds.skin_tone_force_view = True
    brightened = ds[index][0]
    assert brightened.float().mean().item() > plain.float().mean().item() + 5.0
    # An image the bank does not know keeps its original view.
    other = _by_path(ds, group["other"])
    ds.skin_tone_force_view = False
    untouched = ds[other][0].clone()
    ds.skin_tone_force_view = True
    assert torch.equal(ds[other][0], untouched)
