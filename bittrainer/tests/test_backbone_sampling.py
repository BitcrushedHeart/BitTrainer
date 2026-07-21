"""Per-epoch backbone label plan (Bitcrush ISSUE-0545/0546).

Pins the sampling layer that keeps extreme neg:pos ratios from collapsing a
head to always-negative:

1. per-head neg:pos cap (default 5:1) with a fresh negative draw each epoch,
   auto-tightened toward 1:1 for tiny heads;
2. guaranteed positive presence — tiny heads' positives are replicated
   (bounded), replicas carrying ONLY that head's positive label;
3. residual BCE ``pos_weight`` per head computed from the post-cap ratio;
4. per-head positive cap preferring label-dense images;
5. the cap operates at the (head, image) label level — an image dropped from
   one head's plan still trains its other heads and groups.

Pure unit tests over ``_plan_epoch_samples`` — no images, no training run.
"""

from __future__ import annotations

from bittrainer.backbone_trainer import (
    _Sample,
    _Vocab,
    _effective_neg_ratio,
    _head_pos_weights,
    _plan_epoch_samples,
)


def _vocab(concepts=("c",), groups=None):
    records = []
    for concept in concepts:
        records.append({"binary": {concept: "positive"}})
        records.append({"binary": {concept: "negative"}})
    for group, classes in (groups or {}).items():
        for cls in classes:
            records.append({"groups": {group: cls}})
    return _Vocab(records)


def _samples(n_pos, n_neg, concept="c", start=0):
    out = []
    for i in range(n_pos):
        out.append(_Sample(f"p{start + i}.png", {concept: 1.0}, {}))
    for i in range(n_neg):
        out.append(_Sample(f"n{start + i}.png", {concept: 0.0}, {}))
    return out


def _plan_counts(planned, concept="c"):
    pos = sum(1 for s in planned if s.binary.get(concept) == 1.0)
    neg = sum(1 for s in planned if s.binary.get(concept) == 0.0)
    return pos, neg


def test_effective_neg_ratio_tightens_for_tiny_heads():
    assert _effective_neg_ratio(10, 5.0) == 1.0
    assert _effective_neg_ratio(20, 5.0) == 2.0
    assert _effective_neg_ratio(30, 5.0) == 3.0
    assert _effective_neg_ratio(100, 5.0) == 5.0
    # <=0 means uncapped, regardless of positives.
    assert _effective_neg_ratio(10, 0.0) == 0.0
    assert _effective_neg_ratio(10, -1.0) == 0.0


def test_neg_cap_five_to_one():
    """100 pos / 1000 neg at the default ratio keeps exactly 500 negatives."""
    samples = _samples(100, 1000)
    planned, stats = _plan_epoch_samples(samples, _vocab(), epoch=0)
    _, neg = _plan_counts(planned)
    assert neg == 500
    assert stats["c"]["neg_explicit"] == 500
    assert stats["c"]["effective_ratio"] == 5.0


def test_uncapped_when_ratio_nonpositive():
    samples = _samples(10, 500)
    planned, _stats = _plan_epoch_samples(samples, _vocab(), epoch=0, neg_pos_ratio=0.0)
    _, neg = _plan_counts(planned)
    assert neg == 500


def test_tiny_head_caps_at_one_to_one():
    """10 positives tighten the 5:1 default to 1:1 -> 10 negatives survive."""
    samples = _samples(10, 5000)
    planned, stats = _plan_epoch_samples(samples, _vocab(), epoch=0)
    _, neg = _plan_counts(planned)
    assert neg == 10
    assert stats["c"]["effective_ratio"] == 1.0


def test_cap_is_per_head_not_per_image():
    """An image dropped from head A's plan keeps its other labels."""
    samples = _samples(2, 0, concept="a")
    # 100 images negative-for-a AND positive-for-b, plus b negatives.
    for i in range(100):
        samples.append(_Sample(f"x{i}.png", {"a": 0.0, "b": 1.0}, {"g": i % 2}))
    for i in range(50):
        samples.append(_Sample(f"y{i}.png", {"b": 0.0}, {}))
    vocab = _vocab(concepts=("a", "b"), groups={"g": ["u", "v"]})
    planned, _stats = _plan_epoch_samples(samples, vocab, epoch=0, min_positive_threshold=0)
    # Head a: 2 pos -> ratio 1.0 -> only 2 of the 100 negatives keep an "a" label.
    a_pos, a_neg = _plan_counts(planned, "a")
    assert (a_pos, a_neg) == (2, 2)
    # But every x image still trains head b and its group.
    b_pos, _b_neg = _plan_counts(planned, "b")
    assert b_pos == 100
    assert sum(1 for s in planned if "g" in s.groups) == 100


def test_resample_is_deterministic_and_varies_by_epoch():
    samples = _samples(10, 200)
    kw = {"neg_pos_ratio": 5.0, "seed": 7}
    plan_a, _ = _plan_epoch_samples(samples, _vocab(), epoch=0, **kw)
    plan_b, _ = _plan_epoch_samples(samples, _vocab(), epoch=0, **kw)
    plan_c, _ = _plan_epoch_samples(samples, _vocab(), epoch=1, **kw)
    paths = lambda plan: sorted(s.path for s in plan if s.binary.get("c") == 0.0)  # noqa: E731
    assert paths(plan_a) == paths(plan_b)
    assert paths(plan_a) != paths(plan_c)


def test_tiny_head_positive_replication_bounded():
    """10 pos under threshold 30 -> factor 3; replicas carry ONLY the head's
    positive label and no groups."""
    samples = _samples(10, 10)
    samples[0].groups = {"g": 0}  # a positive that also carries a group label
    vocab = _vocab(groups={"g": ["u", "v"]})
    planned, stats = _plan_epoch_samples(samples, vocab, epoch=0)
    pos, _neg = _plan_counts(planned)
    assert stats["c"]["oversample_factor"] == 3
    assert pos == 30  # 10 originals + 2 extra copies each
    replicas = [s for s in planned if s.binary.get("c") == 1.0 and not s.groups]
    # Only the base sample with a group label keeps it; its replicas do not.
    assert sum(1 for s in planned if s.groups) == 1
    assert all(set(s.binary) == {"c"} for s in replicas)


def test_oversample_factor_capped():
    """2 positives would need 15x to reach 30 -> capped at max factor 4."""
    samples = _samples(2, 2)
    _planned, stats = _plan_epoch_samples(samples, _vocab(), epoch=0)
    assert stats["c"]["oversample_factor"] == 4


def test_no_oversample_at_or_above_threshold():
    samples = _samples(30, 30)
    _planned, stats = _plan_epoch_samples(samples, _vocab(), epoch=0)
    assert stats["c"]["oversample_factor"] == 1


def test_pos_weight_residual_and_clamp():
    # 40 pos / 200 selected negs, factor 1 -> w = 5.0.
    stats = {"a": {"pos": 40, "neg_explicit": 200, "neg_implicit": 0, "oversample_factor": 1}}
    assert _head_pos_weights(stats)["a"] == 5.0
    # Replication counts toward positive occurrences: 10*3 pos vs 30 neg -> 1.0.
    stats = {"a": {"pos": 10, "neg_explicit": 30, "neg_implicit": 0, "oversample_factor": 3}}
    assert _head_pos_weights(stats)["a"] == 1.0
    # Clamped to 10 and floored at 1; zero-pos heads get 1.0.
    stats = {"a": {"pos": 1, "neg_explicit": 500, "neg_implicit": 0, "oversample_factor": 1}}
    assert _head_pos_weights(stats)["a"] == 10.0
    stats = {"a": {"pos": 0, "neg_explicit": 500, "neg_implicit": 0, "oversample_factor": 1}}
    assert _head_pos_weights(stats)["a"] == 1.0


def test_positive_cap_prefers_label_dense():
    """Over the cap, images carrying more labels survive."""
    dense = [_Sample(f"d{i}.png", {"c": 1.0, "b": 1.0}, {"g": 0}) for i in range(5)]
    sparse = [_Sample(f"s{i}.png", {"c": 1.0}, {}) for i in range(5)]
    fill = _samples(0, 10) + [_Sample(f"bn{i}.png", {"b": 0.0}, {}) for i in range(10)]
    vocab = _vocab(concepts=("c", "b"), groups={"g": ["u", "v"]})
    planned, stats = _plan_epoch_samples(
        dense + sparse + fill, vocab, epoch=0, positive_cap=5, min_positive_threshold=0
    )
    kept = [s.path for s in planned if s.binary.get("c") == 1.0]
    assert len(kept) == 5
    assert all(p.startswith("d") for p in kept)
    assert stats["c"]["pos"] == 5
    assert stats["c"]["pos_total"] == 10


def test_positive_cap_zero_is_uncapped():
    samples = _samples(50, 50)
    _planned, stats = _plan_epoch_samples(
        samples, _vocab(), epoch=0, positive_cap=0, min_positive_threshold=0
    )
    assert stats["c"]["pos"] == 50


def test_groups_never_capped():
    """Group labels appear in every epoch view regardless of binary caps."""
    samples = [_Sample(f"g{i}.png", {"c": 0.0}, {"g": i % 2}) for i in range(100)]
    samples += _samples(2, 0, start=1000)
    planned, _stats = _plan_epoch_samples(samples, _vocab(groups={"g": ["u", "v"]}), epoch=0)
    assert sum(1 for s in planned if "g" in s.groups) == 100
