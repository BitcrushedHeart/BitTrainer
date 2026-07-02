"""Tests for spatial-grid group support: mirror maps, flip-aware augmentation,
and the cell-structured head + Bernoulli-likelihood class decode."""

import torch
import torch.nn.functional as F

from bittrainer.spatial import (
    SpatialCellFC,
    build_hflip_class_map,
    install_spatial_head,
    mirror_cell_mask,
    spatial_hflip_batch,
)


# 3x3 grid cell indices:  0 1 2 / 3 4 5 / 6 7 8
LEFT_COL = [0, 3, 6]
RIGHT_COL = [2, 5, 8]
CENTRE_COL = [1, 4, 7]
FULL = list(range(9))
L_SHAPE = [0, 3, 6, 7, 8]
REVERSE_L = [2, 5, 6, 7, 8]


class TestMirrorCellMask:
    def test_left_column_mirrors_to_right(self):
        assert mirror_cell_mask(LEFT_COL, 3, 3) == RIGHT_COL

    def test_centre_column_is_self_mirror(self):
        assert mirror_cell_mask(CENTRE_COL, 3, 3) == CENTRE_COL

    def test_full_frame_is_self_mirror(self):
        assert mirror_cell_mask(FULL, 3, 3) == FULL

    def test_l_mirrors_to_reverse_l(self):
        assert mirror_cell_mask(L_SHAPE, 3, 3) == REVERSE_L

    def test_empty_mask_is_self_mirror(self):
        assert mirror_cell_mask([], 3, 3) == []

    def test_non_square_grid(self):
        # 2x4 grid: 0 1 2 3 / 4 5 6 7 — cell 0 mirrors to 3, 4 to 7.
        assert mirror_cell_mask([0, 4], 2, 4) == [3, 7]


class TestBuildHflipClassMap:
    def test_maps_mirror_pairs_and_self(self):
        masks = [[], LEFT_COL, RIGHT_COL, CENTRE_COL]
        assert build_hflip_class_map(masks, 3, 3) == [0, 2, 1, 3]

    def test_missing_mirror_is_minus_one(self):
        # Mirror of [0, 1] is [1, 2], which is not a class.
        masks = [[0, 1], CENTRE_COL]
        assert build_hflip_class_map(masks, 3, 3) == [-1, 1]

    def test_extra_compositions_close_under_mirror(self):
        masks = [L_SHAPE, REVERSE_L]
        assert build_hflip_class_map(masks, 3, 3) == [1, 0]


class TestSpatialHflipBatch:
    def _asym_batch(self, n: int) -> torch.Tensor:
        # Left half bright, right half dark — orientation is detectable.
        batch = torch.zeros(n, 3, 8, 8, dtype=torch.uint8)
        batch[:, :, :, :4] = 200
        return batch

    def test_p1_flips_images_and_remaps_labels(self):
        flip_map = torch.tensor([0, 2, 1, 3])
        images = self._asym_batch(4)
        labels = torch.tensor([1, 2, 3, 0])
        out_images, out_labels = spatial_hflip_batch(images, labels, flip_map, p=1.0)
        assert out_labels.tolist() == [2, 1, 3, 0]
        # Every image flipped: bright half now on the right.
        assert (out_images[:, :, :, 4:] == 200).all()
        assert (out_images[:, :, :, :4] == 0).all()

    def test_p0_is_identity(self):
        flip_map = torch.tensor([0, 2, 1, 3])
        images = self._asym_batch(3)
        labels = torch.tensor([1, 2, 0])
        out_images, out_labels = spatial_hflip_batch(images, labels, flip_map, p=0.0)
        assert out_labels.tolist() == [1, 2, 0]
        assert (out_images[:, :, :, :4] == 200).all()

    def test_unmirrorable_labels_never_flip(self):
        flip_map = torch.tensor([-1, 1])
        images = self._asym_batch(2)
        labels = torch.tensor([0, 1])
        out_images, out_labels = spatial_hflip_batch(images, labels, flip_map, p=1.0)
        # Label 0 has no mirror class: image and label unchanged.
        assert out_labels[0].item() == 0
        assert (out_images[0, :, :, :4] == 200).all()
        # Label 1 is its own mirror: flipped, label unchanged.
        assert out_labels[1].item() == 1
        assert (out_images[1, :, :, 4:] == 200).all()


class TestSpatialCellFC:
    def test_decode_matches_bernoulli_likelihood(self):
        masks = [[], LEFT_COL, CENTRE_COL, FULL]
        head = SpatialCellFC(16, masks, 9)
        x = torch.randn(5, 16)
        scores = head(x)
        assert scores.shape == (5, 4)
        z = head.cell_fc(x).float()
        for k, mask in enumerate(masks):
            expected = sum(
                (F.logsigmoid(z[:, c]) if c in mask else F.logsigmoid(-z[:, c]))
                for c in range(9)
            )
            assert torch.allclose(scores[:, k], expected, atol=1e-5)

    def test_confident_cells_pick_matching_class(self):
        masks = [[], LEFT_COL, RIGHT_COL, CENTRE_COL]
        head = SpatialCellFC(4, masks, 9)
        # Force cell logits directly: left column strongly on, rest strongly off.
        z = torch.full((1, 9), -20.0)
        z[0, LEFT_COL] = 20.0
        scores = head.decode_cells(z)
        assert scores.argmax(dim=1).item() == 1
        # All cells off → __none__ (empty mask) wins.
        z_none = torch.full((1, 9), -20.0)
        assert head.decode_cells(z_none).argmax(dim=1).item() == 0

    def test_weight_property_exposes_cell_fc(self):
        head = SpatialCellFC(8, [[], [0]], 9)
        assert head.weight.dtype == head.cell_fc.weight.dtype
        assert head.weight.shape == (9, 8)

    def test_gradients_flow_through_decode(self):
        head = SpatialCellFC(8, [[], LEFT_COL, RIGHT_COL], 9)
        x = torch.randn(4, 8)
        loss = F.cross_entropy(head(x), torch.tensor([0, 1, 2, 1]))
        loss.backward()
        assert head.cell_fc.weight.grad is not None
        assert head.cell_fc.weight.grad.abs().sum() > 0


class TestInstallAndCheckpointRoundTrip:
    def _small_model(self, num_classes: int):
        from bittrainer.model import create_model

        return create_model(
            model_size="atto", pretrained=False, num_classes=num_classes,
        )

    def test_install_replaces_fc_and_forward_shape(self):
        masks = [[], LEFT_COL, RIGHT_COL, CENTRE_COL, FULL]
        model = self._small_model(num_classes=len(masks))
        install_spatial_head(model, masks, 9)
        assert isinstance(model.head.fc, SpatialCellFC)
        out = model(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 5)

    def test_load_checkpoint_reconstructs_spatial_head(self, tmp_path):
        masks = [[], LEFT_COL, RIGHT_COL, CENTRE_COL]
        model = self._small_model(num_classes=len(masks))
        install_spatial_head(model, masks, 9)
        model.eval()
        path = tmp_path / "spatial.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "num_classes": len(masks),
                "model_size": "atto",
                "class_names": ["__none__", "Left", "Right", "Centre"],
                "cell_masks": masks,
                "grid_rows": 3,
                "grid_cols": 3,
            },
            path,
        )
        from bittrainer.model import load_checkpoint

        loaded = load_checkpoint(str(path), device="cpu")
        assert isinstance(loaded.head.fc, SpatialCellFC)
        loaded.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            assert torch.allclose(model(x), loaded(x), atol=1e-5)

    def test_load_checkpoint_plain_linear_unaffected(self, tmp_path):
        model = self._small_model(num_classes=3)
        model.eval()
        path = tmp_path / "plain.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "num_classes": 3,
                "model_size": "atto",
                "class_names": ["a", "b", "c"],
            },
            path,
        )
        from bittrainer.model import load_checkpoint

        loaded = load_checkpoint(str(path), device="cpu")
        assert isinstance(loaded.head.fc, torch.nn.Linear)
        loaded.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            assert torch.allclose(model(x), loaded(x), atol=1e-5)

    def test_warmstart_from_linear_incumbent_keeps_backbone(self, tmp_path):
        """A pre-spatial (linear-head) best.pt warm-starts the backbone while
        the spatial head starts clean — no shape errors."""
        from bittrainer.group_trainer import GroupTrainConfig, _create_or_warmstart_model

        masks = [[], LEFT_COL, RIGHT_COL]
        old = self._small_model(num_classes=3)
        torch.save(
            {"state_dict": old.state_dict(), "num_classes": 3, "model_size": "atto"},
            tmp_path / "best.pt",
        )
        config = GroupTrainConfig(
            group_folder=str(tmp_path),
            num_classes=3,
            class_names=["__none__", "Left", "Right"],
            backbone_variant="atto",
            best_model_name="best.pt",
            cell_masks=masks,
            grid_rows=3,
            grid_cols=3,
        )
        model = _create_or_warmstart_model(
            config, device=torch.device("cpu"), dtype=torch.float32,
            head_hidden_size=None, checkpoint_dir=tmp_path,
        )
        assert isinstance(model.head.fc, SpatialCellFC)
        # Backbone carried over from the incumbent.
        assert torch.equal(
            model.state_dict()["stem.0.weight"], old.state_dict()["stem.0.weight"]
        )


class TestPhotometricRandAugment:
    def test_geometric_ops_removed(self):
        from bittrainer.gpu_augment import _get_randaugment

        ra = _get_randaugment(2, 9, photometric_only=True)
        space = ra._AUGMENTATION_SPACE
        for op in ("ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"):
            assert op not in space
        assert "Brightness" in space
        # The unfiltered variant keeps its full op set (no class-level bleed).
        full = _get_randaugment(2, 9, photometric_only=False)
        assert "TranslateX" in full._AUGMENTATION_SPACE

    def test_photometric_randaugment_runs(self):
        from bittrainer.gpu_augment import gpu_randaugment

        batch = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
        out = gpu_randaugment(batch, 2, 9, photometric_only=True)
        assert out.shape == batch.shape
        assert out.dtype == torch.uint8


class TestApplyTrainAugmentHflipGate:
    def test_hflip_false_preserves_orientation(self):
        from bittrainer.gpu_augment import apply_train_augment

        batch = torch.zeros(8, 3, 8, 8, dtype=torch.uint8)
        batch[:, :, :, :4] = 220
        out = apply_train_augment(batch, hflip=False)
        # Colour jitter moves values but never mirrors: the bright half must
        # still be on the left for every sample.
        left = out[:, :, :, :4].mean(dim=(1, 2, 3))
        right = out[:, :, :, 4:].mean(dim=(1, 2, 3))
        assert (left > right).all()


class TestSpatialCkptMeta:
    def test_meta_present_only_for_spatial_config(self):
        from bittrainer.group_trainer import GroupTrainConfig, _spatial_ckpt_meta

        base = dict(group_folder="g", num_classes=2, class_names=["a", "b"])
        assert _spatial_ckpt_meta(GroupTrainConfig(**base)) == {}
        spatial = GroupTrainConfig(
            **base, cell_masks=[[], [0]], grid_rows=3, grid_cols=3,
        )
        meta = _spatial_ckpt_meta(spatial)
        assert meta["cell_masks"] == [[], [0]]
        assert meta["grid_rows"] == 3
        assert meta["grid_cols"] == 3
