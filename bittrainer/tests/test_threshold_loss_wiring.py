"""All-threshold head wiring through the group trainer (Bitcrush ISSUE-1106, Phase B).

``GroupTrainConfig.ordinal_head = "threshold"`` (ordinal groups only) must reach
every place the head matters — the loss zoo (incl. the MixUp path), the model
build and warm start, the resume path, the resume fingerprint, the shipped
decode (argmax, no cut-points), checkpoints and the run result — while the
default ``"softmax"`` leaves every one of them exactly as it was.

CPU-only, random-init ConvNeXt atto; no download, no full run.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import bittrainer.generic.evaluation as ev
import bittrainer.group_trainer as gt
from bittrainer.generic.optimizer import make_optimizer
from bittrainer.generic.task import BestTracker
from bittrainer.generic.tasks.group_task import GroupTask
from bittrainer.model import create_model
from bittrainer.ordinal_threshold import (
    has_threshold_head,
    install_ordinal_threshold_head,
    ordinal_threshold_loss,
    threshold_logits_from_scores,
)
from bittrainer.soft_labels import _build_soft_targets, _soft_ce_loss
from bittrainer.training_state import make_fingerprint

_NAMES = ["s0", "s1", "s2", "s3", "__none__"]


def _config(tmp_path=".", **kw):
    base = dict(
        group_folder=str(tmp_path),
        num_classes=5,
        class_names=list(_NAMES),
        backbone_variant="atto",
        ordinal=True,
        ordinal_head="threshold",
        ordinal_sigma=1.0,
        device="cpu",
        dtype="float32",
        channels_last=False,
        use_compile=False,
        randaugment_n=0,
        random_erasing_p=0.0,
        fused_back_pass=False,
        checkpoint_dir=str(tmp_path) + "/ckpt" if tmp_path != "." else None,
    )
    base.update(kw)
    return gt.GroupTrainConfig(**base)


def _scores(seed=0, n=6):
    torch.manual_seed(seed)
    from bittrainer.ordinal_threshold import OrdinalThresholdFC

    fc = OrdinalThresholdFC(4, 5, none_index=4)
    return fc.decode(torch.randn(n, 4) * 2)


def test_default_ordinal_head_is_softmax():
    assert (
        gt.GroupTrainConfig(
            group_folder="x", num_classes=2, class_names=["a", "b"]
        ).ordinal_head
        == "softmax"
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def test_threshold_branch_is_bce_against_the_soft_targets():
    config = _config()
    labels = torch.tensor([0, 1, 3, 4, 2, 0])
    scores = _scores()
    loss_fn = gt.build_group_loss_fn(
        config, use_soft_targets=True, none_index=4, device=torch.device("cpu")
    )
    soft = _build_soft_targets(labels, 5, ordinal=True, ordinal_sigma=1.0, none_index=4)
    want = ordinal_threshold_loss(
        threshold_logits_from_scores(scores, 4), soft, none_index=4
    )
    torch.testing.assert_close(loss_fn(scores, labels), want)


def test_the_sigma_sweep_still_shapes_the_threshold_loss():
    labels = torch.tensor([0, 1, 3, 2])
    scores = _scores(n=4)
    cpu = torch.device("cpu")
    sharp = gt.build_group_loss_fn(
        _config(ordinal_sigma=0.3), use_soft_targets=True, none_index=4, device=cpu
    )
    soft = gt.build_group_loss_fn(
        _config(ordinal_sigma=2.0), use_soft_targets=True, none_index=4, device=cpu
    )
    assert not torch.isclose(sharp(scores, labels), soft(scores, labels))


def test_softmax_ordinal_branch_is_unchanged():
    config = _config(ordinal_head="softmax")
    labels = torch.tensor([0, 1, 3, 4])
    logits = torch.randn(4, 5)
    loss_fn = gt.build_group_loss_fn(
        config, use_soft_targets=True, none_index=4, device=torch.device("cpu")
    )
    soft = _build_soft_targets(labels, 5, ordinal=True, ordinal_sigma=1.0, none_index=4)
    assert torch.equal(
        loss_fn(logits, labels), _soft_ce_loss(torch.log_softmax(logits, 1), soft)
    )


def test_threshold_head_needs_ordinal():
    """ordinal_head is ignored for a non-ordinal group: plain soft-CE as before."""
    config = _config(ordinal=False)
    labels = torch.tensor([0, 1])
    logits = torch.randn(2, 5)
    loss_fn = gt.build_group_loss_fn(
        config, use_soft_targets=True, none_index=4, device=torch.device("cpu")
    )
    soft = _build_soft_targets(labels, 5, label_smoothing=0.1, none_index=4)
    assert torch.equal(
        loss_fn(logits, labels), _soft_ce_loss(torch.log_softmax(logits, 1), soft)
    )


def test_soft_target_loss_dispatches_by_head():
    soft = torch.softmax(torch.randn(3, 5), 1)
    logits = torch.randn(3, 5)
    weights = torch.rand(5) + 0.5
    softmax_cfg = _config(ordinal_head="softmax")
    assert torch.equal(
        gt._soft_target_loss(
            logits,
            soft,
            softmax_cfg,
            none_index=4,
            class_weights=weights,
            focal_gamma=0.5,
        ),
        _soft_ce_loss(
            torch.log_softmax(logits.float(), 1),
            soft,
            class_weights=weights,
            focal_gamma=0.5,
        ),
    )
    scores = _scores(n=3)
    torch.testing.assert_close(
        gt._soft_target_loss(
            scores, soft, _config(), none_index=4, class_weights=weights
        ),
        ordinal_threshold_loss(
            threshold_logits_from_scores(scores, 4),
            soft,
            none_index=4,
            class_weights=weights,
        ),
    )


def test_mixup_batches_go_through_the_soft_target_loss(monkeypatch):
    seen: list = []
    real = gt._soft_target_loss

    def _spy(logits, soft, config, **kw):
        seen.append(config.ordinal_head)
        return real(logits, soft, config, **kw)

    monkeypatch.setattr(gt, "_soft_target_loss", _spy)
    torch.manual_seed(0)
    model = create_model(model_size="atto", pretrained=False, num_classes=5)
    install_ordinal_threshold_head(model, 5, none_index=4)
    batch = (
        torch.randint(0, 255, (4, 3, 64, 64), dtype=torch.uint8),
        torch.tensor([0, 1, 2, 4]),
    )
    loss, _ = gt._train_one_epoch(
        model,
        [batch],
        make_optimizer(model),
        _config(mixup_prob=1.0),
        torch.device("cpu"),
        torch.float32,
        use_soft_targets=True,
        mixup_enabled=True,
    )
    assert seen == ["threshold"] and loss == loss


# ---------------------------------------------------------------------------
# Model build, resume, fingerprint
# ---------------------------------------------------------------------------


def _create(config, tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    return gt._create_or_warmstart_model(
        config,
        device=torch.device("cpu"),
        dtype=torch.float32,
        head_hidden_size=None,
        checkpoint_dir=ckpt,
    )


def test_fresh_model_gets_the_threshold_head(tmp_path):
    model = _create(
        _config(tmp_path, backbone_init={"source": "random_init"}), tmp_path
    )
    assert has_threshold_head(model) and model.head.fc.none_index == 4


@pytest.mark.parametrize("override", [{"ordinal_head": "softmax"}, {"ordinal": False}])
def test_no_threshold_head_otherwise(tmp_path, override):
    config = _config(tmp_path, backbone_init={"source": "random_init"}, **override)
    assert not has_threshold_head(_create(config, tmp_path))


def test_mlp_probe_head_composes_with_the_threshold_head(tmp_path):
    (tmp_path / "ckpt").mkdir()
    model = gt._create_or_warmstart_model(
        _config(tmp_path, backbone_init={"source": "random_init"}),
        device=torch.device("cpu"),
        dtype=torch.float32,
        head_hidden_size=16,
        checkpoint_dir=tmp_path / "ckpt",
    )
    assert has_threshold_head(model) and model.head.fc.in_features == 16


def test_resume_reinstalls_the_threshold_head(tmp_path):
    torch.manual_seed(0)
    source = create_model(model_size="atto", pretrained=False, num_classes=5)
    install_ordinal_threshold_head(source, 5, none_index=4)
    task = GroupTask(_config(tmp_path))
    ctx = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32)
    model = task.create_model(ctx, {"model": source.state_dict()})
    assert has_threshold_head(model)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(source.eval()(x), model.eval()(x))


def test_fingerprint_key_is_additive():
    base = dict(
        class_names=_NAMES,
        num_classes=5,
        max_epochs=3,
        multi_label=False,
        ordinal=True,
        best_model_name="best.pt",
        model_size="atto",
    )
    assert make_fingerprint(**base) == make_fingerprint(**base, ordinal_head=None)
    assert "ordinal_head" not in make_fingerprint(**base)
    assert (
        make_fingerprint(**base, ordinal_head="threshold")["ordinal_head"]
        == "threshold"
    )


def test_group_task_fingerprint_names_the_threshold_head(tmp_path):
    def _fp(config):
        ctx = SimpleNamespace(
            pause_event=None,
            cb=lambda _m: None,
            em=SimpleNamespace(stage=lambda *a, **k: None),
        )
        GroupTask(config).fingerprint_init(ctx)
        return ctx.fingerprint

    assert _fp(_config(tmp_path))["ordinal_head"] == "threshold"
    assert "ordinal_head" not in _fp(_config(tmp_path, ordinal_head="softmax"))
    assert "ordinal_head" not in _fp(_config(tmp_path, ordinal=False))


# ---------------------------------------------------------------------------
# Config guard
# ---------------------------------------------------------------------------


def test_softmax_needs_no_validation():
    assert gt.validate_ordinal_head(_config(ordinal_head="softmax")) == []


@pytest.mark.parametrize(
    "override,match",
    [
        ({"multi_label": True}, "multi_label"),
        ({"cell_masks": [[0], [1], [2], [3], []]}, "spatial"),
        ({"use_focal": True}, "focal"),
        ({"ordinal_head": "cumulative"}, "ordinal_head"),
    ],
)
def test_threshold_refusals(override, match):
    with pytest.raises(ValueError, match=match):
        gt.validate_ordinal_head(_config(**override))


def test_threshold_on_a_non_ordinal_group_is_ignored_with_a_note():
    notes = gt.validate_ordinal_head(_config(ordinal=False))
    assert notes and "ignored" in notes[0]


def test_group_task_refuses_threshold_multi_label(tmp_path):
    with pytest.raises(ValueError, match="multi_label"):
        GroupTask(_config(tmp_path, multi_label=True)).make_context(
            None, None, None, None
        )


# ---------------------------------------------------------------------------
# Decode, checkpoints, result
# ---------------------------------------------------------------------------


def test_finalise_decode_is_threshold_argmax_without_cut_points(monkeypatch):
    monkeypatch.setattr(
        ev, "find_ordinal_cut_points", lambda *a, **k: pytest.fail("no cut-point fit")
    )
    scores = _scores(n=8)
    labels = scores.argmax(1)
    cuts, metrics = ev._finalise_ordinal_decode(scores, labels, _config(), 4)
    assert cuts is None
    assert metrics["ordinal_decode"] == "threshold_argmax"
    assert "ordinal_cut_points" not in metrics
    cm = torch.tensor(metrics["confusion_matrix"])
    assert (
        int(cm.sum() - cm.diagonal().sum()) == 0
    )  # argmax decode reproduces the labels
    shipped = ev._shipped_decode_metrics(scores, labels, _config(), 4)
    assert shipped["ordinal_decode"] == "threshold_argmax"


def test_explicit_head_flag_overrides_the_config(monkeypatch):
    """Finalisation decides by the SHIPPED checkpoint: a kept softmax incumbent
    keeps its cut-point decode even when the run asked for the threshold head."""
    called: list = []
    monkeypatch.setattr(
        ev, "find_ordinal_cut_points", lambda *a, **k: called.append(1) or None
    )
    scores = _scores(n=8)
    ev._finalise_ordinal_decode(
        scores, scores.argmax(1), _config(), 4, threshold_head=False
    )
    assert called


def test_threshold_checkpoint_records_the_head(tmp_path):
    torch.manual_seed(0)
    model = create_model(model_size="atto", pretrained=False, num_classes=5)
    install_ordinal_threshold_head(model, 5, none_index=4)
    assert gt._architecture_ckpt_meta(_config(), model) == {"ordinal_head": "threshold"}
    task = GroupTask(_config(tmp_path))
    task.ema = None
    task.save_candidate(
        SimpleNamespace(checkpoint_dir=tmp_path),
        model,
        0,
        {"macro_f1": 0.5},
        BestTracker(),
    )
    assert (
        torch.load(tmp_path / "candidate.pt", weights_only=True)["ordinal_head"]
        == "threshold"
    )


def test_finalize_result_names_the_threshold_head(tmp_path):
    torch.manual_seed(0)
    model = create_model(model_size="atto", pretrained=False, num_classes=5)
    install_ordinal_threshold_head(model, 5, none_index=4)
    config = _config(tmp_path)
    candidate = tmp_path / "candidate.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": 5,
            "model_size": "atto",
            "class_names": list(_NAMES),
            **gt._architecture_ckpt_meta(config, model),
        },
        candidate,
    )
    torch.manual_seed(1)
    val = [
        (
            torch.randint(0, 255, (5, 3, 64, 64), dtype=torch.uint8),
            torch.tensor([0, 1, 2, 3, 4]),
        )
    ]
    result = gt._compare_promote_finalize(
        config,
        candidate_path=str(candidate),
        best_metrics={"macro_f1": 0.2, "qwk": 0.1, "val_loss": 1.0},
        candidate_macro_f1=0.2,
        candidate_qwk=0.1,
        best_epoch_display=1,
        epochs_completed=1,
        val_loader=val,
        device=torch.device("cpu"),
        dtype=torch.float32,
        checkpoint_dir=tmp_path,
        class_counts={i: 2 for i in range(5)},
        total_raw=10,
    )
    assert result["ordinal_head"] == "threshold"
    assert result["ordinal_decode"] == "threshold_argmax"
    assert result["ordinal_cut_points"] is None
    shipped = torch.load(result["checkpoint_path"], weights_only=True)
    assert (
        shipped["ordinal_head"] == "threshold" and "ordinal_cut_points" not in shipped
    )
    assert shipped["val_probs"].shape == (5, 5)


def test_softmax_result_has_no_ordinal_head_key(tmp_path):
    torch.manual_seed(0)
    model = create_model(model_size="atto", pretrained=False, num_classes=5)
    config = _config(tmp_path, ordinal_head="softmax")
    candidate = tmp_path / "candidate.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": 5,
            "model_size": "atto",
            "class_names": list(_NAMES),
        },
        candidate,
    )
    val = [
        (
            torch.randint(0, 255, (5, 3, 64, 64), dtype=torch.uint8),
            torch.tensor([0, 1, 2, 3, 4]),
        )
    ]
    result = gt._compare_promote_finalize(
        config,
        candidate_path=str(candidate),
        best_metrics={"macro_f1": 0.2, "qwk": 0.1, "val_loss": 1.0},
        candidate_macro_f1=0.2,
        candidate_qwk=0.1,
        best_epoch_display=1,
        epochs_completed=1,
        val_loader=val,
        device=torch.device("cpu"),
        dtype=torch.float32,
        checkpoint_dir=tmp_path,
        class_counts={i: 2 for i in range(5)},
        total_raw=10,
    )
    assert "ordinal_head" not in result
    assert F is not None  # keep the functional import used
