"""Soft implicit negatives for the backbone label plan (Bitcrush ISSUE-0545).

The Engine ``label_policy`` knob becomes real: under
``mode == "soft_implicit_negative"`` an unlabelled (image, head) pair may enter
the epoch plan as a soft negative (target ``implicit_negative_value``), but
only into the headroom the per-head neg:pos cap leaves after every explicit
negative is seated. ``masked_unknown`` (and no policy at all) is the legacy
behaviour: unknowns never train.

Validation never sees implicit negatives — they are a training prior, not
ground truth.
"""

from __future__ import annotations

import asyncio

from bittrainer.backbone_trainer import (
    _Sample,
    _plan_epoch_samples,
    run_backbone_training,
)
from bittrainer.tests.test_backbone_sampling import _plan_counts, _samples, _vocab

_SOFT = {"mode": "soft_implicit_negative", "implicit_negative_value": 0.1}


def _mixed_samples():
    # Head "c": 10 pos, 4 explicit neg. 50 images unlabelled for "c" (they
    # carry only head "b" labels so they survive plan assembly regardless).
    samples = _samples(10, 4)
    samples += [_Sample(f"u{i}.png", {"b": 1.0}, {}) for i in range(50)]
    samples += [_Sample(f"ub{i}.png", {"b": 0.0}, {}) for i in range(50)]
    return samples


def test_masked_unknown_is_default():
    samples = _mixed_samples()
    for policy in (None, {"mode": "masked_unknown", "implicit_negative_value": 0.1}):
        planned, stats = _plan_epoch_samples(
            samples, _vocab(concepts=("c", "b")), epoch=0, label_policy=policy
        )
        assert stats["c"]["neg_implicit"] == 0
        assert all(s.binary.get("c") in (None, 0.0, 1.0) for s in planned)


def test_implicit_negatives_fill_headroom_only():
    """10 pos -> ratio tightens to 1:1 -> cap 10; 4 explicit negatives seat
    first, 6 implicit fill the rest at the soft value."""
    samples = _mixed_samples()
    planned, stats = _plan_epoch_samples(
        samples, _vocab(concepts=("c", "b")), epoch=0, label_policy=_SOFT
    )
    assert stats["c"]["neg_explicit"] == 4
    assert stats["c"]["neg_implicit"] == 6
    soft = [s for s in planned if s.binary.get("c") == 0.1]
    hard = [s for s in planned if s.binary.get("c") == 0.0]
    assert len(soft) == 6 and len(hard) == 4


def test_no_headroom_no_implicit():
    """Explicit negatives already fill the cap -> zero implicit drawn."""
    samples = _samples(10, 40)
    samples += [_Sample(f"u{i}.png", {"b": 1.0}, {}) for i in range(50)]
    samples += [_Sample(f"ub{i}.png", {"b": 0.0}, {}) for i in range(50)]
    planned, stats = _plan_epoch_samples(
        samples, _vocab(concepts=("c", "b")), epoch=0, label_policy=_SOFT
    )
    assert stats["c"]["neg_implicit"] == 0
    _, neg = _plan_counts(planned)
    assert neg == 10  # 1:1 cap, all explicit


def test_implicit_value_flows_to_targets():
    samples = _mixed_samples()
    policy = {"mode": "soft_implicit_negative", "implicit_negative_value": 0.25}
    planned, stats = _plan_epoch_samples(
        samples, _vocab(concepts=("c", "b")), epoch=0, label_policy=policy
    )
    assert stats["c"]["neg_implicit"] == 6
    assert sum(1 for s in planned if s.binary.get("c") == 0.25) == 6


def test_e2e_soft_policy_trains_and_val_stays_clean(tmp_path, monkeypatch):
    """End-to-end run with the soft policy: a soft target reaches the BCE loss
    and the val loader's labels stay the original hard dicts."""
    import bittrainer.backbone_trainer as bb
    import bittrainer.generic.tasks.backbone_task as bt
    from bittrainer.tests.test_backbone_generic import _request

    request = _request(tmp_path, epochs=1, max_steps=50, n=12)
    # Make watermark rare so unlabelled images exist for it: only two records
    # keep a (positive) watermark label, the rest carry just the group label —
    # with no explicit negatives the 1:1 tiny-head cap is pure implicit headroom.
    for i, record in enumerate(request["records"]):
        if i not in (1, 3):
            record["binary"] = {}
    request["training_config"]["label_policy"] = dict(_SOFT)
    request["training_config"]["validation_split"] = 0.35

    seen_targets: list[float] = []
    real_loss = bb._batch_loss

    def _spy_loss(features, heads, binary_labels, group_labels, device, **kw):
        for labels in binary_labels:
            seen_targets.extend(labels.values())
        return real_loss(features, heads, binary_labels, group_labels, device, **kw)

    monkeypatch.setattr(bb, "_batch_loss", _spy_loss)

    val_label_sets: list[dict] = []
    real_build = bt.BackboneTask.build_loaders

    def _spy_build(self, ctx, epoch, eff_bs, resume_info):
        out = real_build(self, ctx, epoch, eff_bs, resume_info)
        val_label_sets.extend(s.binary for s in self.val_samples)
        return out

    monkeypatch.setattr(bt.BackboneTask, "build_loaders", _spy_build)

    result = asyncio.run(run_backbone_training(request))
    assert result["candidate_checkpoint_path"]
    assert 0.1 in seen_targets, "no soft implicit-negative target reached the loss"
    for labels in val_label_sets:
        assert all(v in (0.0, 1.0) for v in labels.values())
