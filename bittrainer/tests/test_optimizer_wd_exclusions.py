"""Weight-decay exclusions in the shared optimizer factory (round B7).

Norm scales, biases and GRN gains (all 1-D in ConvNeXtV2) must not be decayed:
``make_optimizer(..., wd_exclusions=True)`` places every 1-D parameter in a
``weight_decay=0.0`` group, with and without LLRD. The factory DEFAULT stays
the legacy single flat group (existing callers bit-identical); trainers opt in
via config — ``GroupTrainConfig.wd_exclusions`` defaults True. Because the
param-group layout changes, the run fingerprint gains an optimizer identity
(``make_fingerprint(optimizer_identity=...)``) so old backups cleanly mismatch
instead of crashing into ``optimizer.load_state_dict``.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from bittrainer.generic.optimizer import make_optimizer
from bittrainer.model import create_model


def _tiny_model() -> nn.Module:
    return create_model(model_size="atto", pretrained=False, num_classes=3)


def _group_wd(group: dict, default: float) -> float:
    return float(group.get("weight_decay", default))


def test_bare_factory_call_stays_single_flat_group():
    opt = make_optimizer(_tiny_model())
    assert len(opt.param_groups) == 1  # legacy layout: existing callers untouched


def test_flat_exclusions_split_one_d_params_out_of_decay():
    model = _tiny_model()
    opt = make_optimizer(model, wd_exclusions=True)
    assert len(opt.param_groups) >= 2
    default_wd = opt.defaults["weight_decay"]
    seen = 0
    for group in opt.param_groups:
        wd = _group_wd(group, default_wd)
        for p in group["params"]:
            seen += 1
            if p.ndim < 2:
                assert wd == 0.0, "1-D param (norm/bias/GRN) left in a decayed group"
            else:
                assert wd == pytest.approx(0.01), (
                    "matrix param lost its canonical decay"
                )
    assert seen == len(list(model.parameters()))  # nothing dropped
    all_ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(all_ids) == len(set(all_ids))  # nothing duplicated


def test_llrd_plus_exclusions_keep_depth_multipliers():
    model = _tiny_model()
    opt = make_optimizer(model, llrd=True, llrd_decay=0.8, wd_exclusions=True)
    default_wd = opt.defaults["weight_decay"]
    stem_lrs = set()
    for group in opt.param_groups:
        wd = _group_wd(group, default_wd)
        dims = {p.ndim < 2 for p in group["params"]}
        assert len(dims) == 1, "decay and no-decay params mixed in one group"
        if dims == {True}:
            assert wd == 0.0
        name = str(group.get("name", ""))
        if name.split("/")[0] == "stem":
            stem_lrs.add(round(float(group["lr"]), 6))
    # Both stem subgroups (decay + no-decay) share the depth-5 multiplier.
    assert stem_lrs == {round(0.8**5, 6)}


def test_group_config_field_and_passthrough():
    from bittrainer.group_trainer import GroupTrainConfig, _make_optimizer

    config = GroupTrainConfig(
        group_folder=".", num_classes=3, class_names=["a", "b", "c"]
    )
    assert config.wd_exclusions is True  # the justified group default: ON
    model = _tiny_model()
    opt = _make_optimizer(model, config)
    default_wd = opt.defaults["weight_decay"]
    no_decay = [
        g
        for g in opt.param_groups
        if _group_wd(g, default_wd) == 0.0 and all(p.ndim < 2 for p in g["params"])
    ]
    assert no_decay, "group trainer did not pass wd_exclusions through to the factory"

    legacy = GroupTrainConfig(
        group_folder=".",
        num_classes=3,
        class_names=["a", "b", "c"],
        llrd=False,
        wd_exclusions=False,
    )
    assert len(_make_optimizer(_tiny_model(), legacy).param_groups) == 1


def test_make_fingerprint_optional_optimizer_identity():
    from bittrainer.training_state import make_fingerprint

    base_kwargs = dict(
        class_names=["a", "b"],
        num_classes=2,
        max_epochs=3,
        multi_label=False,
        ordinal=False,
        best_model_name="best.pt",
        model_size="atto",
    )
    plain = make_fingerprint(**base_kwargs)
    assert "optimizer" not in plain  # untouched callers keep the legacy shape
    stamped = make_fingerprint(**base_kwargs, optimizer_identity="Prodigy_adv+nd")
    assert stamped["optimizer"] == "Prodigy_adv+nd"
    # An old backup (no key) must NOT satisfy a stamped fingerprint.
    from bittrainer.training_state import fingerprint_matches

    assert not fingerprint_matches(plain, stamped)
    assert fingerprint_matches(stamped, stamped)


def test_group_task_stamps_optimizer_identity_into_fingerprint(monkeypatch, tmp_path):
    """The GroupTask fingerprint must carry the identity so a pre-round backup
    (different param-group layout) is skipped rather than loaded."""
    import bittrainer.generic.tasks.group_task as gtask
    from bittrainer.group_trainer import GroupTrainConfig

    captured: dict = {}
    real_init = gtask.init_backup

    def _spy(config, pause_event, cb, **kw):
        captured.update(kw)
        return real_init(config, pause_event, cb, **kw)

    monkeypatch.setattr(gtask, "init_backup", _spy)

    config = GroupTrainConfig(
        group_folder=str(tmp_path),
        num_classes=2,
        class_names=["a", "b"],
        checkpoint_dir=str(tmp_path / "ck"),
        device="cpu",
        dtype="float32",
        use_compile=False,
        channels_last=False,
    )
    task = gtask.GroupTask(config)
    ctx = task.make_context(lambda _m: None, None, None, None)
    task.fingerprint_init(ctx)
    identity = captured.get("optimizer_identity")
    assert identity, "GroupTask.fingerprint_init did not pass optimizer_identity"
    assert identity != "Prodigy_adv"  # llrd=True + wd_exclusions=True by default
    assert ctx.fingerprint.get("optimizer") == identity
