"""Region targets through the collate, the augmentation and the train loop
(Bitcrush ISSUE-0993).

A batch from a crop-assisted group carries a third element, the
:class:`~bittrainer.regions.RegionTargets`. Everything that moves pixels has to
move the boxes with them (or leave the pixels alone), and the classification
loss has to skip rows that are there for the detector only.

CPU-only, stub loaders, random-init ``atto`` trunk.
"""

from __future__ import annotations

import pytest
import torch

import bittrainer.gpu_augment as gpu_augment
import bittrainer.group_trainer as gt
from bittrainer.collate import _collate_bucket_batch, _collate_multilabel_batch
from bittrainer.generic.optimizer import make_optimizer
from bittrainer.model import create_model
from bittrainer.regions import RegionTargets, install_region_head, region_hflip_batch


def _info(boxes=(), types=(), supervised=(False,), cls=True):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "types": torch.tensor(types, dtype=torch.long),
        "supervised": torch.tensor(supervised, dtype=torch.bool),
        "cls": cls,
    }


def _image(value=0):
    return torch.full((3, 32, 32), value, dtype=torch.uint8)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


class TestCollate:
    def test_plain_items_stay_a_two_tuple(self):
        batch = [(_image(), 0, (32, 32)), (_image(), 1, (32, 32))]
        out = _collate_bucket_batch(batch)
        assert len(out) == 2
        assert out[1].tolist() == [0, 1]

    def test_region_items_add_the_targets(self):
        batch = [
            (_image(), 1, (32, 32), _info([[0.1, 0.2, 0.3, 0.4]], [0], [True])),
            (_image(), -100, (32, 32), _info(supervised=[True], cls=False)),
            (_image(), 0, (32, 32), _info()),
        ]
        images, labels, targets = _collate_bucket_batch(batch)
        assert images.shape == (3, 3, 32, 32)
        assert labels.tolist() == [1, -100, 0]
        assert isinstance(targets, RegionTargets)
        assert targets.batch_index.tolist() == [0]
        assert targets.supervised.tolist() == [[True], [True], [False]]
        assert targets.cls_mask.tolist() == [True, False, True]
        assert targets.detector_rows.tolist() == [True, True, False]

    def test_multilabel_collate_adds_the_targets_too(self):
        batch = [
            (_image(), torch.tensor([1.0, 0.0]), (32, 32), _info([[0.1, 0.2, 0.3, 0.4]], [0], [True])),
            (_image(), torch.tensor([0.0, 0.0]), (32, 32), _info(supervised=[True], cls=False)),
        ]
        images, labels, targets = _collate_multilabel_batch(batch)
        assert labels.shape == (2, 2)
        assert targets.cls_mask.tolist() == [True, False]
        plain = _collate_multilabel_batch([(_image(), torch.tensor([1.0, 0.0]), (32, 32))])
        assert len(plain) == 2


# ---------------------------------------------------------------------------
# Flip
# ---------------------------------------------------------------------------


def _targets(boxes, batch_index, batch, supervised=None):
    return RegionTargets(
        boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        batch_index=torch.tensor(batch_index, dtype=torch.long),
        types=torch.zeros(len(batch_index), dtype=torch.long),
        supervised=(
            torch.ones(batch, 1, dtype=torch.bool) if supervised is None else supervised
        ),
        cls_mask=torch.ones(batch, dtype=torch.bool),
    )


class TestRegionHflip:
    def _asym(self, n):
        batch = torch.zeros(n, 3, 8, 8, dtype=torch.uint8)
        batch[:, :, :, :4] = 200
        return batch

    def test_flip_mirrors_images_and_boxes(self):
        targets = _targets([[0.1, 0.2, 0.3, 0.6], [0.5, 0.0, 1.0, 1.0]], [0, 1], batch=2)
        images, flipped = region_hflip_batch(self._asym(2), targets, p=1.0)
        assert (images[:, :, :, 4:] == 200).all() and (images[:, :, :, :4] == 0).all()
        assert flipped.boxes[0].tolist() == pytest.approx([0.7, 0.2, 0.9, 0.6])
        assert flipped.boxes[1].tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0])
        # The caller's targets are not edited in place.
        assert targets.boxes[0].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.6])

    def test_p0_is_identity(self):
        targets = _targets([[0.1, 0.2, 0.3, 0.6]], [0], batch=1)
        images, same = region_hflip_batch(self._asym(1), targets, p=0.0)
        assert (images[:, :, :, :4] == 200).all()
        assert same.boxes.tolist() == targets.boxes.tolist()

    def test_only_flipped_rows_have_their_boxes_mirrored(self):
        torch.manual_seed(3)
        n = 64
        boxes = [[0.1, 0.2, 0.3, 0.6]] * n
        images, flipped = region_hflip_batch(self._asym(n), _targets(boxes, list(range(n)), n))
        was_flipped = images[:, 0, 0, 0] == 0
        assert 0 < int(was_flipped.sum()) < n
        for row in range(n):
            expected = [0.7, 0.2, 0.9, 0.6] if was_flipped[row] else [0.1, 0.2, 0.3, 0.6]
            assert flipped.boxes[row].tolist() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# RandAugment: rows with boxes are photometric-only
# ---------------------------------------------------------------------------


class TestPhotometricRows:
    def test_flagged_rows_get_the_photometric_policy(self, monkeypatch):
        used: list[bool] = []

        def _fake(num_ops, magnitude, photometric_only=False):
            def _apply(img):
                used.append(photometric_only)
                return img

            return _apply

        monkeypatch.setattr(gpu_augment, "_get_randaugment", _fake)
        batch = torch.zeros(4, 3, 8, 8, dtype=torch.uint8)
        rows = torch.tensor([True, False, True, False])
        gpu_augment.gpu_randaugment(batch, 2, 9, photometric_rows=rows)
        assert used == [True, False, True, False]

    def test_global_photometric_only_still_covers_every_row(self, monkeypatch):
        used: list[bool] = []

        def _fake(num_ops, magnitude, photometric_only=False):
            def _apply(img):
                used.append(photometric_only)
                return img

            return _apply

        monkeypatch.setattr(gpu_augment, "_get_randaugment", _fake)
        batch = torch.zeros(2, 3, 8, 8, dtype=torch.uint8)
        gpu_augment.gpu_randaugment(
            batch, 2, 9, photometric_only=True, photometric_rows=torch.tensor([False, False])
        )
        assert used == [True, True]

    def test_apply_train_augment_threads_the_rows_through(self, monkeypatch):
        seen = {}

        def _spy(batch, num_ops, magnitude, *, photometric_only=False, photometric_rows=None):
            seen["rows"] = photometric_rows
            return batch

        monkeypatch.setattr(gpu_augment, "gpu_randaugment", _spy)
        rows = torch.tensor([True, False])
        gpu_augment.apply_train_augment(
            torch.zeros(2, 3, 8, 8, dtype=torch.uint8),
            randaugment_n=2,
            randaugment_m=9,
            photometric_rows=rows,
        )
        assert seen["rows"] is rows


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------


def _config(**kw) -> gt.GroupTrainConfig:
    base = dict(
        group_folder=".",
        num_classes=3,
        class_names=["a", "b", "c"],
        device="cpu",
        dtype="float32",
        channels_last=False,
        use_compile=False,
        randaugment_n=0,
        randaugment_m=0,
        random_erasing_p=0.0,
        fused_back_pass=False,
    )
    base.update(kw)
    return gt.GroupTrainConfig(**base)


def _region_model(num_types=1):
    torch.manual_seed(0)
    model = create_model(model_size="atto", pretrained=False, num_classes=3)
    install_region_head(model, num_types)
    return model


def _region_batch(labels=(0, 1, -100), multi_label=False):
    torch.manual_seed(1)
    n = len(labels)
    images = torch.randint(0, 255, (n, 3, 64, 64), dtype=torch.uint8)
    if multi_label:
        label_t = torch.tensor(labels, dtype=torch.float32)
        cls = torch.tensor([bool(sum(row)) for row in labels])
    else:
        label_t = torch.tensor(labels, dtype=torch.long)
        cls = label_t >= 0
    supervised = torch.zeros(n, 1, dtype=torch.bool)
    supervised[0] = True
    supervised[n - 1] = True
    targets = RegionTargets(
        boxes=torch.tensor([[0.2, 0.2, 0.6, 0.7], [0.1, 0.5, 0.5, 0.9]]),
        batch_index=torch.tensor([0, n - 1]),
        types=torch.tensor([0, 0]),
        supervised=supervised,
        cls_mask=cls,
    )
    return images, label_t, targets


def _run(model, batches, config, **kw):
    return gt._train_one_epoch(
        model,
        batches,
        make_optimizer(model),
        config,
        torch.device("cpu"),
        torch.float32,
        **kw,
    )


class TestTrainLoop:
    def test_region_batches_train_the_region_head(self):
        model = _region_model()
        before = {k: v.clone() for k, v in model.region_head.state_dict().items()}
        region_losses: list[float] = []
        loss, per_class = _run(
            model,
            [_region_batch(), _region_batch()],
            _config(),
            region_loss_sink=region_losses.append,
        )
        after = model.region_head.state_dict()
        assert any(not torch.equal(before[k], after[k]) for k in before)
        assert len(region_losses) == 1 and region_losses[0] > 0
        assert loss == loss and loss > 0  # finite classification loss
        # The detection-only row (label -100) is not a class.
        assert set(per_class) <= {"0", "1", "2"}

    def test_reported_loss_is_classification_only(self, monkeypatch):
        """train_loss stays comparable with val_loss: the detector's loss is
        optimised but reported through its own sink."""
        model = _region_model()
        monkeypatch.setattr(gt, "region_loss", lambda out, targets: out.heatmap.sum() * 0.0 + 50.0)
        sunk: list[float] = []
        loss, _ = _run(model, [_region_batch()], _config(), region_loss_sink=sunk.append)
        assert sunk == [pytest.approx(50.0)]
        assert loss < 10.0

    def test_soft_targets_skip_detection_only_rows(self):
        model = _region_model()
        loss, _ = _run(
            model, [_region_batch()], _config(label_smoothing=0.1), use_soft_targets=True
        )
        assert loss == loss

    def test_batch_of_only_detection_rows_still_steps(self):
        model = _region_model()
        loss, per_class = _run(model, [_region_batch(labels=(-100, -100))], _config())
        assert loss == 0.0 and per_class == {}

    def test_sample_loss_sink_gets_one_value_per_row(self):
        model = _region_model()
        seen: list[tuple[int, int]] = []
        _run(
            model,
            [_region_batch()],
            _config(),
            sample_loss_sink=lambda i, losses: seen.append((i, losses.numel())),
        )
        assert seen == [(0, 3)]

    def test_multi_label_region_batch(self):
        model = _region_model()
        batch = _region_batch(labels=([1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]), multi_label=True)
        loss, _ = _run(model, [batch], _config(multi_label=True))
        assert loss == loss and loss > 0

    def test_plain_multi_label_batch_trains(self):
        """Regression: the per-class hard-label telemetry ran on multi-hot
        targets and raised, so no multi-label full fine-tune got past batch 1."""
        model = create_model(model_size="atto", pretrained=False, num_classes=3)
        torch.manual_seed(0)
        batch = (
            torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]),
        )
        loss, per_class = _run(model, [batch], _config(multi_label=True))
        assert loss > 0 and per_class == {}

    def test_region_batches_are_never_mixed(self, monkeypatch):
        import bittrainer.mixing as mixing

        def _boom(*a, **k):
            raise AssertionError("MixUp would detach the boxes from the pixels")

        monkeypatch.setattr(mixing, "apply_mixing", _boom)
        model = _region_model()
        _run(model, [_region_batch()], _config(mixup_prob=1.0), mixup_enabled=True)

    def test_region_batches_flip_boxes_not_blindly(self, monkeypatch):
        seen = {}
        real = gpu_augment.apply_train_augment

        def _spy(batch, *a, **k):
            seen.update(k)
            return real(batch, *a, **k)

        monkeypatch.setattr(gpu_augment, "apply_train_augment", _spy)
        flips: list[int] = []
        real_flip = gt.region_hflip_batch

        def _flip(images, targets, **k):
            flips.append(images.shape[0])
            return real_flip(images, targets, **k)

        monkeypatch.setattr(gt, "region_hflip_batch", _flip)
        model = _region_model()
        _run(model, [_region_batch()], _config())
        assert flips == [3]
        assert seen["hflip"] is False
        assert seen["photometric_rows"].tolist() == [True, False, True]

    def test_plain_batches_never_touch_the_region_path(self, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("plain batch took the region forward")

        monkeypatch.setattr(gt, "forward_with_regions", _boom)
        model = create_model(model_size="atto", pretrained=False, num_classes=3)
        torch.manual_seed(0)
        batches = [
            (torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8), torch.tensor([0, 2]))
        ]
        sunk: list[float] = []
        loss, _ = _run(model, batches, _config(), region_loss_sink=sunk.append)
        assert loss > 0 and sunk == []

    def test_region_loss_weight_scales_the_detector_gradient(self):
        def _head_delta(weight):
            model = _region_model()
            before = model.region_head.heatmap.bias.detach().clone()
            _run(model, [_region_batch()], _config(region_loss_weight=weight))
            return float((model.region_head.heatmap.bias.detach() - before).abs().sum())

        assert _head_delta(0.0) == 0.0
        assert _head_delta(1.0) > 0.0


# ---------------------------------------------------------------------------
# Dataset -> loader -> loop, in one process
# ---------------------------------------------------------------------------


def _region_group(tmp_path):
    import json

    import numpy as np
    from PIL import Image

    from bittrainer.group_dataset import GroupDataset
    from bittrainer.region_manifest import load_region_manifest

    def _png(path, seed):
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        Image.fromarray(rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)).save(path)
        return path

    root = tmp_path / "g"
    drawn = _png(root / "a" / "train" / "a0.png", 1)
    _png(root / "b" / "train" / "b0.png", 2)
    _png(root / "c" / "train" / "c0.png", 3)
    crop = _png(root / ".region_crops" / "train" / "crop0.png", 4)
    loose = _png(tmp_path / "pool" / "loose.png", 5)
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
                        "source": str(drawn),
                        "box": [0.2, 0.2, 0.8, 0.8],
                        "type": 0,
                        "class": "b",
                        "split": "train",
                    }
                ],
                "images": [
                    {
                        "source": str(drawn),
                        "paths": [str(drawn)],
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
                ],
            }
        ),
        encoding="utf-8",
    )
    return GroupDataset(
        root,
        ["a", "b", "c"],
        split="train",
        group_name="g",
        extra_paths={"b": [str(crop)]},
        region_manifest=load_region_manifest(manifest),
        train_resolution=64,
        # Each image once: the default replication would repeat the thin classes.
        natural_sampling=True,
    )


def test_region_dataset_survives_pickling(tmp_path):
    """DataLoader workers get a pickled copy of the dataset (Windows spawn)."""
    import pickle

    ds = _region_group(tmp_path)
    clone = pickle.loads(pickle.dumps(ds))
    assert len(clone[0]) == 4
    assert [s.get("view") for s in clone.samples] == [s.get("view") for s in ds.samples]


def test_real_dataset_batches_run_through_the_loop(tmp_path):
    from torch.utils.data import DataLoader

    from bittrainer.group_dataset import build_group_bucket_sampler

    ds = _region_group(tmp_path)
    loader = DataLoader(
        ds,
        batch_sampler=build_group_bucket_sampler(ds, batch_size=8),
        collate_fn=_collate_bucket_batch,
        num_workers=0,
    )
    batches = list(loader)
    assert len(batches) == 1
    images, labels, targets = batches[0]
    assert images.shape[0] == 5  # three class images, one crop, one detection-only
    assert sorted(labels.tolist()) == [-100, 0, 1, 1, 2]
    assert int(targets.detector_rows.sum()) == 2 and int((~targets.cls_mask).sum()) == 1

    model = _region_model()
    sunk: list[float] = []
    loss, per_class = _run(model, batches, _config(), region_loss_sink=sunk.append)
    assert loss > 0 and sunk and sunk[0] > 0
    assert set(per_class) == {"0", "1", "2"}
