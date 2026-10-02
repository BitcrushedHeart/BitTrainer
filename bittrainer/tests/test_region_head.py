"""Region head: a CenterNet-style proposer on the classifier trunk (Bitcrush ISSUE-0993).

Crop-assisted groups train a detection head beside the classifier so the model
can later propose the crops the user used to draw by hand. These tests pin the
pieces that have to be exact: the classifier forward is untouched, targets land
on the right cells, the loss only sees images the user actually drew on, and a
decode recovers the boxes the targets were built from.

CPU-only, random-init ``atto`` trunk: no weights are downloaded.
"""

from __future__ import annotations

import pytest
import torch

from bittrainer.model import create_model
from bittrainer.regions import (
    REGION_STRIDE,
    RegionHead,
    RegionOutput,
    RegionTargets,
    build_region_maps,
    collate_region_targets,
    decode_regions,
    forward_with_regions,
    has_region_head,
    install_region_head,
    region_head_ckpt_meta,
    region_loss,
)


def _model(num_types: int = 1):
    torch.manual_seed(0)
    model = create_model(model_size="atto", pretrained=False, num_classes=3)
    install_region_head(model, num_types)
    return model.eval()


def _targets(boxes, types, *, batch: int, num_types: int, supervised=None, cls=None):
    """Hand-built targets: ``boxes`` is ``[(batch_index, x1, y1, x2, y2)]``."""
    if supervised is None:
        supervised = torch.ones(batch, num_types, dtype=torch.bool)
    if cls is None:
        cls = torch.ones(batch, dtype=torch.bool)
    return RegionTargets(
        boxes=torch.tensor([b[1:] for b in boxes], dtype=torch.float32).reshape(-1, 4),
        batch_index=torch.tensor([b[0] for b in boxes], dtype=torch.long),
        types=torch.tensor(types, dtype=torch.long),
        supervised=supervised,
        cls_mask=cls,
    )


class TestInstall:
    def test_model_without_head_reports_none(self):
        model = create_model(model_size="atto", pretrained=False, num_classes=3)
        assert not has_region_head(model)
        assert region_head_ckpt_meta(model, ["Face"]) == {}

    def test_install_adds_region_head_state_keys(self):
        model = _model(num_types=2)
        assert has_region_head(model)
        keys = [k for k in model.state_dict() if k.startswith("region_head.")]
        assert keys, "region head weights must ride in the model state_dict"

    def test_head_is_small(self):
        model = _model()
        params = sum(p.numel() for p in model.region_head.parameters())
        assert params < 600_000

    def test_ckpt_meta_describes_the_head(self):
        model = _model(num_types=2)
        meta = region_head_ckpt_meta(model, ["Face", "Hand"])["region_head"]
        assert meta["num_types"] == 2
        assert meta["types"] == ["Face", "Hand"]
        assert meta["stride"] == REGION_STRIDE
        assert meta["in_channels"] == [160, 320]

    def test_heatmap_starts_near_empty(self):
        """CenterNet prior: an untrained head must not fire everywhere."""
        model = _model()
        with torch.no_grad():
            _, out = forward_with_regions(model, torch.randn(1, 3, 64, 64))
        assert torch.sigmoid(out.heatmap).max().item() < 0.5


class TestForward:
    def test_classifier_logits_are_untouched(self):
        model = _model()
        x = torch.randn(2, 3, 96, 64)
        with torch.no_grad():
            plain = model(x)
            logits, _ = forward_with_regions(model, x)
        assert torch.equal(plain, logits)

    def test_output_is_stride_16(self):
        model = _model(num_types=3)
        with torch.no_grad():
            _, out = forward_with_regions(model, torch.randn(2, 3, 96, 64))
        assert isinstance(out, RegionOutput)
        assert out.heatmap.shape == (2, 3, 6, 4)
        assert out.size.shape == (2, 2, 6, 4)
        assert out.offset.shape == (2, 2, 6, 4)

    def test_forward_without_head_raises(self):
        model = create_model(model_size="atto", pretrained=False, num_classes=3)
        with pytest.raises(RuntimeError, match="region head"):
            forward_with_regions(model, torch.randn(1, 3, 64, 64))

    def test_standalone_head_accepts_feature_maps(self):
        head = RegionHead((8, 16), num_types=2, channels=16)
        out = head(torch.randn(1, 8, 4, 6), torch.randn(1, 16, 2, 3))
        assert out.heatmap.shape == (1, 2, 4, 6)


class TestTargets:
    def test_peak_lands_on_the_box_centre_cell(self):
        targets = _targets([(0, 0.25, 0.25, 0.75, 0.75)], [0], batch=1, num_types=1)
        maps = build_region_maps(targets, 8, 8)
        assert maps.heat.shape == (1, 1, 8, 8)
        assert maps.heat[0, 0, 4, 4].item() == pytest.approx(1.0)
        assert maps.heat.max().item() == pytest.approx(1.0)
        # A Gaussian, not a single hot cell.
        assert 0.0 < maps.heat[0, 0, 4, 3].item() < 1.0

    def test_size_and_offset_are_in_cells(self):
        # Centre (0.3, 0.55) on a 10x10 map -> cell (3, 5), offset (0.0, 0.5).
        targets = _targets([(0, 0.1, 0.4, 0.5, 0.7)], [0], batch=1, num_types=1)
        maps = build_region_maps(targets, 10, 10)
        assert maps.cells.tolist() == [[0, 5, 3]]
        assert maps.size[0].tolist() == pytest.approx([4.0, 3.0])
        assert maps.offset[0].tolist() == pytest.approx([0.0, 0.5], abs=1e-5)

    def test_each_type_gets_its_own_channel(self):
        targets = _targets(
            [(0, 0.0, 0.0, 0.5, 0.5), (0, 0.5, 0.5, 1.0, 1.0)], [0, 1], batch=1, num_types=2
        )
        maps = build_region_maps(targets, 8, 8)
        assert maps.heat[0, 0, 2, 2].item() == pytest.approx(1.0)
        assert maps.heat[0, 1, 6, 6].item() == pytest.approx(1.0)
        assert maps.heat[0, 0, 6, 6].item() < 0.5

    def test_collate_stacks_per_image_info(self):
        infos = [
            {
                "boxes": torch.tensor([[0.1, 0.1, 0.4, 0.4]]),
                "types": torch.tensor([0]),
                "supervised": torch.tensor([True]),
                "cls": True,
            },
            {
                "boxes": torch.zeros(0, 4),
                "types": torch.zeros(0, dtype=torch.long),
                "supervised": torch.tensor([False]),
                "cls": False,
            },
        ]
        targets = collate_region_targets(infos)
        assert targets.boxes.shape == (1, 4)
        assert targets.batch_index.tolist() == [0]
        assert targets.supervised.tolist() == [[True], [False]]
        assert targets.cls_mask.tolist() == [True, False]

    def test_collate_of_plain_items_is_none(self):
        assert collate_region_targets([None, None]) is None


class TestLoss:
    def test_unsupervised_images_contribute_nothing(self):
        """An image nobody drew on is unknown to the detector, not a negative."""
        torch.manual_seed(1)
        out = RegionOutput(
            heatmap=torch.randn(2, 1, 4, 4, requires_grad=True),
            size=torch.randn(2, 2, 4, 4, requires_grad=True),
            offset=torch.randn(2, 2, 4, 4, requires_grad=True),
        )
        targets = _targets(
            [(0, 0.25, 0.25, 0.75, 0.75)],
            [0],
            batch=2,
            num_types=1,
            supervised=torch.tensor([[True], [False]]),
        )
        loss = region_loss(out, targets)
        loss.backward()
        assert out.heatmap.grad[0].abs().sum().item() > 0
        assert out.heatmap.grad[1].abs().sum().item() == 0
        assert out.size.grad[1].abs().sum().item() == 0

    def test_no_supervision_is_a_differentiable_zero(self):
        out = RegionOutput(
            heatmap=torch.randn(1, 1, 4, 4, requires_grad=True),
            size=torch.randn(1, 2, 4, 4, requires_grad=True),
            offset=torch.randn(1, 2, 4, 4, requires_grad=True),
        )
        targets = _targets([], [], batch=1, num_types=1, supervised=torch.tensor([[False]]))
        loss = region_loss(out, targets)
        assert loss.item() == 0.0
        loss.backward()  # must not raise

    def test_a_drawn_image_with_no_boxes_is_all_negative(self):
        """Dismissing every proposal says "nothing here" - the heatmap is pushed down."""
        heat = torch.zeros(1, 1, 4, 4, requires_grad=True)
        out = RegionOutput(heatmap=heat, size=torch.zeros(1, 2, 4, 4), offset=torch.zeros(1, 2, 4, 4))
        targets = _targets([], [], batch=1, num_types=1, supervised=torch.tensor([[True]]))
        loss = region_loss(out, targets)
        loss.backward()
        assert loss.item() > 0
        assert (heat.grad > 0).all(), "gradient descent must lower every cell"

    def test_perfect_prediction_scores_lower_than_a_wrong_one(self):
        targets = _targets([(0, 0.25, 0.25, 0.75, 0.75)], [0], batch=1, num_types=1)
        maps = build_region_maps(targets, 8, 8)
        good_heat = torch.full((1, 1, 8, 8), -8.0)
        good_heat[0, 0, 4, 4] = 8.0
        size = torch.zeros(1, 2, 8, 8)
        size[0, :, 4, 4] = maps.size[0]
        offset = torch.zeros(1, 2, 8, 8)
        offset[0, :, 4, 4] = maps.offset[0]
        good = region_loss(RegionOutput(good_heat, size, offset), targets)
        bad_heat = torch.full((1, 1, 8, 8), -8.0)
        bad_heat[0, 0, 0, 0] = 8.0
        bad = region_loss(RegionOutput(bad_heat, size, offset), targets)
        assert good.item() < bad.item()
        assert good.item() < 0.05


class TestDecode:
    def _perfect_output(self, targets: RegionTargets, h: int, w: int) -> RegionOutput:
        maps = build_region_maps(targets, h, w)
        heat = torch.full_like(maps.heat, -8.0)
        size = torch.zeros(maps.heat.shape[0], 2, h, w)
        offset = torch.zeros(maps.heat.shape[0], 2, h, w)
        for i, (b, y, x) in enumerate(maps.cells.tolist()):
            heat[b, targets.types[i], y, x] = 8.0
            size[b, :, y, x] = maps.size[i]
            offset[b, :, y, x] = maps.offset[i]
        return RegionOutput(heat, size, offset)

    def test_decode_recovers_the_target_boxes(self):
        targets = _targets(
            [(0, 0.10, 0.20, 0.40, 0.60), (0, 0.55, 0.50, 0.95, 0.90), (1, 0.3, 0.3, 0.7, 0.7)],
            [0, 1, 0],
            batch=2,
            num_types=2,
        )
        out = self._perfect_output(targets, 16, 16)
        proposals = decode_regions(out, score_threshold=0.5, max_regions=8)
        assert [len(p) for p in proposals] == [2, 1]
        first = sorted(proposals[0], key=lambda p: p.type)
        assert first[0].type == 0 and first[1].type == 1
        assert first[0].box == pytest.approx((0.10, 0.20, 0.40, 0.60), abs=1e-4)
        assert first[1].box == pytest.approx((0.55, 0.50, 0.95, 0.90), abs=1e-4)
        assert proposals[1][0].box == pytest.approx((0.3, 0.3, 0.7, 0.7), abs=1e-4)
        assert all(p.score > 0.99 for row in proposals for p in row)

    def test_threshold_and_cap_limit_the_proposals(self):
        heat = torch.full((1, 1, 8, 8), -8.0)
        heat[0, 0, 1, 1] = 2.0  # 0.88
        heat[0, 0, 5, 5] = 0.0  # 0.50
        heat[0, 0, 6, 2] = -1.0  # 0.27
        out = RegionOutput(heat, torch.ones(1, 2, 8, 8), torch.zeros(1, 2, 8, 8))
        assert len(decode_regions(out, score_threshold=0.2, max_regions=8)[0]) == 3
        assert len(decode_regions(out, score_threshold=0.4, max_regions=8)[0]) == 2
        capped = decode_regions(out, score_threshold=0.2, max_regions=1)[0]
        assert len(capped) == 1 and capped[0].score == pytest.approx(0.8808, abs=1e-3)

    def test_only_local_maxima_become_proposals(self):
        """A 3x3 max-pool stands in for NMS: one box per blob, not nine."""
        heat = torch.full((1, 1, 8, 8), -8.0)
        heat[0, 0, 3:6, 3:6] = 1.0
        heat[0, 0, 4, 4] = 3.0
        out = RegionOutput(heat, torch.ones(1, 2, 8, 8), torch.zeros(1, 2, 8, 8))
        assert len(decode_regions(out, score_threshold=0.5, max_regions=8)[0]) == 1

    def test_boxes_are_clipped_to_the_frame(self):
        heat = torch.full((1, 1, 4, 4), -8.0)
        heat[0, 0, 0, 0] = 8.0
        out = RegionOutput(heat, torch.full((1, 2, 4, 4), 6.0), torch.zeros(1, 2, 4, 4))
        (proposal,) = decode_regions(out, score_threshold=0.5, max_regions=4)[0]
        assert min(proposal.box) >= 0.0 and max(proposal.box) <= 1.0


def test_head_learns_a_single_box():
    """End to end on one synthetic image: the loss is trainable, not just defined."""
    torch.manual_seed(0)
    head = RegionHead((8, 16), num_types=1, channels=16)
    f3 = torch.zeros(1, 8, 8, 8)
    f3[0, :, 2:5, 3:7] = 1.0
    f4 = torch.zeros(1, 16, 4, 4)
    targets = _targets([(0, 3 / 8, 2 / 8, 7 / 8, 5 / 8)], [0], batch=1, num_types=1)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
    first = None
    for _ in range(150):
        loss = region_loss(head(f3, f4), targets)
        first = loss.item() if first is None else first
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < first * 0.2
    with torch.no_grad():
        (row,) = decode_regions(head(f3, f4), score_threshold=0.3, max_regions=4)
    assert row, "trained head must propose the box it was taught"
    assert row[0].box == pytest.approx((3 / 8, 2 / 8, 7 / 8, 5 / 8), abs=0.08)
