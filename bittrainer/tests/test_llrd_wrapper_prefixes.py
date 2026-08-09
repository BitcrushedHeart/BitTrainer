"""LLRD bucketing must survive the ``_BackboneWithHeads`` wrapper (round A4).

``build_llrd_param_groups`` buckets by ``stem.`` / ``stages.N.`` / ``head.``
name prefixes, which the backbone wrapper's ``backbone.*`` / ``heads.*``
prefixes defeat — historically every parameter collapsed into the ``head``
bucket, which is why ``BackboneTask.create_optimizer`` used a flat group. The
fix strips one optional leading ``backbone.`` before matching and routes
``heads.*`` to depth 0, enabling LLRD for the backbone task; the fingerprint's
optimizer identity distinguishes the new layout so stale flat backups cleanly
mismatch instead of crashing on optimizer state.
"""

from __future__ import annotations

import asyncio

import pytest

import bittrainer.backbone_trainer as bb
from bittrainer.model import build_llrd_param_groups, create_model
from bittrainer.tests.test_backbone_generic import _request


def _wrapper():
    backbone = create_model(model_size="atto", pretrained=False, num_classes=0)
    vocab = bb._Vocab(
        [
            {"binary": {"watermark": "positive"}},
            {"binary": {"watermark": "negative"}},
            {"groups": {"shot_type": "closeup"}},
            {"groups": {"shot_type": "wide"}},
        ]
    )
    heads = bb._MultiTaskHeads(backbone.num_features, vocab)
    return bb._BackboneWithHeads(backbone, heads), backbone, heads


def test_wrapper_prefixes_bucket_by_true_depth():
    model, backbone, heads = _wrapper()
    groups = build_llrd_param_groups(model, 0.8)
    by_name = {g["name"].split("/")[0]: g for g in groups}

    # The historical bug: everything fell through to the "head" bucket. The
    # fixed bucketing must populate the stem and every stage.
    for expected in ("stem", "stages.0", "stages.1", "stages.2", "stages.3", "head"):
        assert expected in by_name, f"bucket {expected!r} missing: {sorted(by_name)}"

    def _params_for(base_name: str) -> set[int]:
        return {
            id(p)
            for g in groups
            if g["name"].split("/")[0] == base_name
            for p in g["params"]
        }

    stem_ids = {id(p) for p in backbone.stem.parameters()}
    assert _params_for("stem") == stem_ids
    stage3_ids = {id(p) for p in backbone.stages[3].parameters()}
    assert _params_for("stages.3") == stage3_ids
    # Multi-task heads train at depth 0 (fastest), alongside the trunk's head.norm.
    head_ids = _params_for("head")
    for p in heads.parameters():
        assert id(p) in head_ids

    assert by_name["stem"]["lr"] == pytest.approx(0.8**5)
    assert by_name["head"]["lr"] == pytest.approx(1.0)

    # Nothing lost, nothing duplicated.
    all_ids = [id(p) for g in groups for p in g["params"]]
    assert len(all_ids) == len(set(all_ids)) == len(list(model.parameters()))


def test_plain_model_bucketing_is_unchanged():
    model = create_model(model_size="atto", pretrained=False, num_classes=3)
    groups = build_llrd_param_groups(model, 0.8)
    for g in groups:
        base = g["name"].split("/")[0]
        for p in g["params"]:
            names = [n for n, q in model.named_parameters() if q is p]
            assert names, "unknown param in group"
            name = names[0]
            if base == "head":
                assert not name.startswith(("stem.", "stages."))
            else:
                assert name.startswith(base + ".")


def test_optimizer_identity_strings():
    from bittrainer.generic.optimizer import optimizer_identity

    legacy = optimizer_identity(llrd=False, llrd_decay=0.8, wd_exclusions=False)
    assert legacy == "Prodigy_adv"  # the historical fingerprint value, exactly
    llrd = optimizer_identity(llrd=True, llrd_decay=0.8, wd_exclusions=False)
    both = optimizer_identity(llrd=True, llrd_decay=0.8, wd_exclusions=True)
    nd = optimizer_identity(llrd=False, llrd_decay=0.8, wd_exclusions=True)
    assert len({legacy, llrd, both, nd}) == 4
    # The decay VALUE is part of the layout identity too.
    assert llrd != optimizer_identity(llrd=True, llrd_decay=0.9, wd_exclusions=False)


def test_backbone_fingerprint_carries_optimizer_identity():
    vocab = bb._Vocab([{"binary": {"c": "positive"}}, {"binary": {"c": "negative"}}])
    fp = bb._backbone_fingerprint(vocab, "atto", 3, optimizer_identity="XYZ")
    assert fp["optimizer"] == "XYZ"
    # Default stays the historical value so untouched callers are unchanged.
    assert bb._backbone_fingerprint(vocab, "atto", 3)["optimizer"] == "Prodigy_adv"


def test_backbone_task_enables_llrd_and_new_identity(tmp_path, monkeypatch):
    """End-to-end: the task builds an LLRD optimizer by default and stamps the
    non-legacy identity into the run fingerprint."""
    import bittrainer.generic.tasks.backbone_task as bt

    captured: dict = {}
    real_make = bt.make_optimizer

    def _spy_make(model, **kw):
        captured["make_kwargs"] = dict(kw)
        opt = real_make(model, **kw)
        captured["param_groups"] = len(opt.param_groups)
        return opt

    monkeypatch.setattr(bt, "make_optimizer", _spy_make)

    real_fp = bb._backbone_fingerprint

    def _spy_fp(*a, **kw):
        fp = real_fp(*a, **kw)
        captured["fingerprint"] = fp
        return fp

    monkeypatch.setattr(bb, "_backbone_fingerprint", _spy_fp)

    asyncio.run(
        bb.run_backbone_training(_request(tmp_path, epochs=1, max_steps=4, n=8))
    )
    assert captured["make_kwargs"].get("llrd") is True
    assert captured["make_kwargs"].get("wd_exclusions") is True
    assert captured["param_groups"] > 1
    assert captured["fingerprint"]["optimizer"] != "Prodigy_adv"
    assert "llrd" in captured["fingerprint"]["optimizer"]
