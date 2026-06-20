"""Unit tests for the backbone feature hash that keys the embedding cache.

The hash must cover every weight that affects the cached pooled vector (stem,
stages, norm_pre, head.norm) and exclude everything after it (head.pre_logits,
head.fc), so a full fine-tune invalidates the cache while a retrained head alone
does not.
"""

from __future__ import annotations

import torch

from bittrainer.model import (
    backbone_feature_hash,
    create_model,
    _infer_head_hidden_size,
)


def _model(num_classes=4, head_hidden_size=None):
    return create_model(
        model_size="nano", pretrained=False,
        num_classes=num_classes, head_hidden_size=head_hidden_size,
    ).eval()


def test_hash_is_deterministic():
    m = _model()
    assert backbone_feature_hash(m) == backbone_feature_hash(m)
    assert len(backbone_feature_hash(m)) == 16


def test_head_fc_does_not_affect_hash():
    m = _model()
    h0 = backbone_feature_hash(m)
    with torch.no_grad():
        m.head.fc.weight.add_(1.0)
        m.head.fc.bias.add_(1.0)
    assert backbone_feature_hash(m) == h0


def test_head_norm_changes_hash():
    m = _model()
    h0 = backbone_feature_hash(m)
    with torch.no_grad():
        m.head.norm.weight.add_(0.5)
    assert backbone_feature_hash(m) != h0


def test_stage_weight_changes_hash():
    m = _model()
    h0 = backbone_feature_hash(m)
    with torch.no_grad():
        next(m.stages.parameters()).add_(0.1)
    assert backbone_feature_hash(m) != h0


def test_mlp_pre_logits_excluded_from_hash():
    # Two MLP-head models that share backbone+norm but differ only in pre_logits
    # must hash identically (pre_logits sits after the cache point).
    m1 = _model(head_hidden_size=256)
    m2 = _model(head_hidden_size=256)
    m2.load_state_dict(m1.state_dict())
    with torch.no_grad():
        m2.head.pre_logits.fc.weight.add_(1.0)
    assert backbone_feature_hash(m1) == backbone_feature_hash(m2)


def test_infer_head_hidden_size():
    assert _infer_head_hidden_size(_model().state_dict()) is None
    assert _infer_head_hidden_size(_model(head_hidden_size=384).state_dict()) == 384
