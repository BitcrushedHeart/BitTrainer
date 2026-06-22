"""Tests for multi-head losses, the model, and multi-head metrics."""

from __future__ import annotations

import torch
from bittrainer.group_validation import compute_multihead_metrics
from bittrainer.multihead_losses import (
    BandConsistencyLoss,
    BandOrdinalSoftLabelLoss,
    VolumeSoftLabelLoss,
)
from bittrainer.multihead_model import MultiHeadConvNeXt
from bittrainer.multihead_ordinal import size_to_band_index, size_to_volume, volume_ranks

SIZE_CLASSES = ["__none__", "32C", "34C", "34D", "34DD"]


def test_volume_soft_label_ignore_index():
    volumes = size_to_volume(SIZE_CLASSES)
    loss = VolumeSoftLabelLoss(volumes, ignore_index=-1)
    logits = torch.randn(4, len(SIZE_CLASSES))
    # All ignored -> zero loss, no NaN.
    targets = torch.full((4,), -1, dtype=torch.long)
    assert loss(logits, targets).item() == 0.0
    # Some real targets -> positive finite loss.
    targets = torch.tensor([1, 2, 3, -1])
    val = loss(logits, targets)
    assert torch.isfinite(val) and val.item() > 0.0


def test_band_ordinal_soft_label():
    loss = BandOrdinalSoftLabelLoss(num_bands=3, ignore_index=-1)
    logits = torch.randn(3, 3)
    targets = torch.tensor([0, 1, -1])
    val = loss(logits, targets)
    assert torch.isfinite(val) and val.item() > 0.0


def test_band_consistency_zero_when_consistent():
    # 2 bands, 4 sizes: sizes 0,1 -> band 0; sizes 2,3 -> band 1.
    size_to_band = [0, 0, 1, 1]
    loss = BandConsistencyLoss(size_to_band, num_bands=2, weight=1.0)
    # Size head puts all mass on size 0 (band 0); band head puts all mass on band 0.
    big = 20.0
    size_logits = torch.tensor([[big, -big, -big, -big]])
    band_logits = torch.tensor([[big, -big]])
    consistent = loss(band_logits, size_logits)
    # Band head disagrees (predicts band 1) -> larger loss.
    band_logits_bad = torch.tensor([[-big, big]])
    inconsistent = loss(band_logits_bad, size_logits)
    assert consistent.item() < inconsistent.item()


def test_two_head_model_forward_and_roundtrip(tmp_path):
    model = MultiHeadConvNeXt(
        backbone_variant="atto",
        n_bands=3,
        n_sizes=len(SIZE_CLASSES),
        band_classes=["32", "34"],
        size_classes=SIZE_CLASSES,
        pretrained=False,
    )
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out["band"].shape == (2, 3)
    assert out["size"].shape == (2, len(SIZE_CLASSES))

    path = tmp_path / "model.pt"
    model.save_checkpoint(str(path), metadata={"epoch": 1})
    restored = MultiHeadConvNeXt.from_checkpoint(str(path))
    restored.eval()
    out2 = restored(x)
    assert torch.allclose(out["size"], out2["size"], atol=1e-5)
    assert restored.size_classes == SIZE_CLASSES


def test_multihead_metrics_perfect():
    band_labels = [0, 1, 1, 0]
    size_ranks = [0, 1, 2, 3]  # every rank present so macro-F1 isn't dragged by empty classes
    m = compute_multihead_metrics(
        band_labels=band_labels,
        band_preds=band_labels,
        num_bands=2,
        size_volume_labels=size_ranks,
        size_volume_preds=size_ranks,
        num_size_ranks=4,
        none_index=-1,
    )
    assert m["band"]["f1"] == 1.0
    assert m["size"]["f1"] == 1.0
    assert m["multi_head"]["f1"] == 1.0


def test_multihead_sister_confusion_is_zero_size_error():
    # True/pred differ only by sister-size (same volume rank) -> size head perfect.
    classes = ["__none__", "32D", "34C", "34DD"]
    ranks = volume_ranks(classes)  # 32D and 34C share a rank
    # Map "labelled 34C, predicted 32D": both share the same volume rank.
    r_34c = ranks[classes.index("34C")]
    r_32d = ranks[classes.index("32D")]
    assert r_34c == r_32d
    # But their bands differ, so the band head is what separates them.
    bands = size_to_band_index(classes, ["32", "34"])
    assert bands[classes.index("34C")] != bands[classes.index("32D")]
