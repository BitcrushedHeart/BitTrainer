"""Distilling a trained backbone into a smaller one (Bitcrush ISSUE-1655).

A student run is an ordinary backbone run with ``request["distillation"]``
naming a teacher candidate. The frozen teacher sees the student's batch and
adds two terms to the loss: its head outputs on every image for every head both
sides share, and its pooled feature through a trainable projector.

CPU only: ``atto`` trunks, 64 px generated images, a few optimizer steps, AMP
off, random init (nothing is ever downloaded). The teacher is made in-process
by one tiny ordinary run, shared by the whole module.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import bittrainer.backbone_trainer as bb
import bittrainer.generic.tasks.backbone_distill_task as bd
import bittrainer.generic.tasks.backbone_task as bt
from bittrainer.backbone_trainer import run_backbone_training
from bittrainer.generic.tasks.backbone_distill_task import BackboneDistillTask
from bittrainer.generic.tasks.backbone_task import BackboneTask
from bittrainer.training_state import fingerprint_matches

CPU = torch.device("cpu")


# -- request builders ----------------------------------------------------------


def _teacher_labels(i):
    """The teacher's layout: two yes/no heads and two two-class groups."""
    binary = {
        "watermark": "positive" if i % 2 else "negative",
        "blurry": "positive" if i % 3 == 0 else "negative",
    }
    groups = {
        "shot_type": "closeup" if i % 2 else "wide",
        "lighting": "day" if i % 4 < 2 else "night",
    }
    return binary, groups


def _student_labels(i):
    """Same as the teacher: every head is shared."""
    return _teacher_labels(i)


def _mixed_labels(i):
    """``blurry`` renamed to ``sharpness``; ``lighting`` gains a class.

    Shared with the teacher: ``watermark`` and ``shot_type``. Student only:
    ``sharpness`` and the three-class ``lighting``. Teacher only: ``blurry``
    and its two-class ``lighting``.
    """
    binary = {
        "watermark": "positive" if i % 2 else "negative",
        "sharpness": "positive" if i % 3 == 0 else "negative",
    }
    groups = {
        "shot_type": "closeup" if i % 2 else "wide",
        "lighting": ("day", "dusk", "night")[i % 3],
    }
    return binary, groups


def _foreign_labels(i):
    """Nothing in common with the teacher."""
    return (
        {"cat": "positive" if i % 2 else "negative"},
        {"season": "summer" if i % 2 else "winter"},
    )


def _records(root: Path, n: int, labels, *, unlabelled: int = 0):
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(n + unlabelled):
        path = root / f"img{i}.png"
        Image.new(
            "RGB", (72, 64), color=((i * 37) % 255, (i * 83) % 255, (i * 151) % 255)
        ).save(path)
        if i < n:
            binary, groups = labels(i)
            content_hash = f"{i:02d}" + "ab" * 31
        else:
            binary, groups = {}, {}
            content_hash = f"{i:02d}" + "cd" * 31
        records.append(
            {
                "content_hash": content_hash,
                "file_paths": [str(path)],
                "binary": binary,
                "groups": groups,
                "splits": {"train": 1},
            }
        )
    return records


def _request(
    root: Path,
    records,
    *,
    run_id: str,
    max_steps=2,
    config=None,
    distillation=None,
):
    training_config = {
        "image_size": 64,
        "batch_size": 4,
        "epochs": 1,
        "max_steps": max_steps,
        "validation_split": 0.25,
        "device": "cpu",
        "use_amp": False,
    }
    training_config.update(config or {})
    request = {
        "run_id": run_id,
        "convnextv2_size": "atto",
        "candidate_checkpoint_path": str(root / "candidates" / f"{run_id}.safetensors"),
        "records": records,
        "training_config": training_config,
        "backbone_init": {"source": "random_init", "checkpoint_path": None},
    }
    if distillation is not None:
        request["distillation"] = distillation
    return request


def _run(request, progress_callback=None):
    return asyncio.run(
        run_backbone_training(request, progress_callback=progress_callback)
    )


def _preparing(frames):
    return next(
        f for f in frames if f.get("stage") == "preparing" and "train_samples" in f
    )


def _epoch_frame(frames):
    return [f for f in frames if f.get("stage") == "training" and "loss" in f][-1]


def _metadata(path):
    with safe_open(str(path), framework="pt") as handle:
        return dict(handle.metadata())


# -- shared fixtures -----------------------------------------------------------


@pytest.fixture(scope="module")
def teacher_path(tmp_path_factory):
    """A real teacher: the candidate of one tiny ordinary backbone run."""
    root = tmp_path_factory.mktemp("teacher")
    request = _request(
        root, _records(root / "imgs", 8, _teacher_labels), run_id="teacher_run"
    )
    return _run(request)["candidate_checkpoint_path"]


@pytest.fixture(scope="module")
def mixed_run(tmp_path_factory, teacher_path):
    """One distillation run with partly matching heads, watched from inside."""
    root = tmp_path_factory.mktemp("mixed")
    seen = {"frames": [], "head_calls": [], "teacher_before": load_file(teacher_path)}
    with pytest.MonkeyPatch.context() as mp:
        real_load = bd.load_teacher

        def _spy_load(path, device):
            teacher = real_load(path, device)
            seen["teacher"] = teacher
            return teacher

        real_head = bd.head_distillation_loss

        def _spy_head(student_logits, teacher_logits, match, temperature):
            value = real_head(student_logits, teacher_logits, match, temperature)
            seen["head_calls"].append(
                {
                    "width": int(student_logits.shape[1]),
                    "concepts": tuple(match.concepts),
                    "groups": tuple(match.groups),
                    "value": float(value.detach()),
                }
            )
            return value

        real_optimizer = bt.make_optimizer

        def _spy_optimizer(model, **kwargs):
            optimizer = real_optimizer(model, **kwargs)
            seen["optimizer"] = optimizer
            seen["model"] = model
            return optimizer

        mp.setattr(bd, "load_teacher", _spy_load)
        mp.setattr(bd, "head_distillation_loss", _spy_head)
        mp.setattr(bt, "make_optimizer", _spy_optimizer)
        request = _request(
            root,
            _records(root / "imgs", 8, _mixed_labels),
            run_id="student_run",
            max_steps=3,
            distillation={"teacher_checkpoint_path": teacher_path},
        )
        seen["result"] = _run(request, seen["frames"].append)
    return seen


# -- (a) the loss maths, against straightforward references --------------------


def test_soft_yes_no_loss_matches_the_reference():
    student = torch.tensor([[0.5, -1.0, 2.0], [1.5, 0.0, -0.3]])
    teacher = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 0.2, 3.0]])
    t = 2.0
    p = 1.0 / (1.0 + torch.exp(-teacher / t))
    q = 1.0 / (1.0 + torch.exp(-student / t))
    reference = (-(p * torch.log(q) + (1 - p) * torch.log(1 - q))).mean() * t * t

    torch.testing.assert_close(bd.soft_binary_loss(student, teacher, t), reference)


def test_soft_group_loss_is_kl_times_t_squared():
    student = torch.tensor([[0.3, -1.2, 2.0, 0.0], [1.0, 1.0, -0.5, 0.2]])
    teacher = torch.tensor([[1.1, 0.4, -0.7, 0.9], [-2.0, 0.3, 0.8, 1.5]])
    t = 3.0
    p_teacher = torch.softmax(teacher / t, dim=1)
    p_student = torch.softmax(student / t, dim=1)
    kl = (p_teacher * (torch.log(p_teacher) - torch.log(p_student))).sum(dim=1)
    reference = kl.mean() * t * t

    torch.testing.assert_close(bd.soft_group_loss(student, teacher, t), reference)
    # Identical distributions cost nothing.
    assert float(bd.soft_group_loss(teacher, teacher, t)) == pytest.approx(
        0.0, abs=1e-6
    )


def test_feature_loss_is_cosine_distance_and_ignores_scale():
    projected = torch.tensor([[1.0, 2.0, 0.0], [0.5, -1.0, 2.0]])
    teacher = torch.tensor([[2.0, 1.0, 1.0], [-0.5, -1.0, 1.0]])
    cosine = (projected * teacher).sum(1) / (
        projected.norm(dim=1) * teacher.norm(dim=1)
    )
    reference = (1.0 - cosine).mean()

    torch.testing.assert_close(
        bd.feature_distillation_loss(projected, teacher), reference
    )
    torch.testing.assert_close(
        bd.feature_distillation_loss(projected * 7.0, teacher * 0.1), reference
    )


def test_feature_projector_bridges_different_widths():
    """A small student and a wide teacher: the projector maps 4 -> 6."""
    projector = nn.Linear(4, 6)
    student = torch.randn(3, 4, generator=torch.Generator().manual_seed(0))
    teacher = torch.randn(3, 6, generator=torch.Generator().manual_seed(1))

    loss = bd.feature_distillation_loss(projector(student), teacher)
    loss.backward()

    assert projector.weight.grad is not None
    assert float(projector.weight.grad.abs().sum()) > 0


def test_head_term_averages_each_family_then_the_families():
    match = bd.match_heads(
        ["a", "b"],
        {"g1": ["x", "y", "z"], "g2": ["p", "q"]},
        ["a", "b"],
        {"g1": ["x", "y", "z"], "g2": ["p", "q"]},
    )
    assert match.width == 7
    generator = torch.Generator().manual_seed(2)
    student = torch.randn(4, 7, generator=generator)
    teacher = torch.randn(4, 7, generator=generator)
    t = 2.0
    yes_no = bd.soft_binary_loss(student[:, :2], teacher[:, :2], t)
    groups = (
        bd.soft_group_loss(student[:, 2:5], teacher[:, 2:5], t)
        + bd.soft_group_loss(student[:, 5:7], teacher[:, 5:7], t)
    ) / 2

    total = bd.head_distillation_loss(student, teacher, match, t)

    torch.testing.assert_close(total, (yes_no + groups) / 2)


def test_head_term_does_not_grow_with_the_number_of_heads():
    one = bd.match_heads(["a"], {}, ["a"], {})
    many = bd.match_heads(["a", "b", "c", "d"], {}, ["a", "b", "c", "d"], {})
    student = torch.tensor([[0.4], [-1.0], [2.0]])
    teacher = torch.tensor([[1.0], [0.5], [-0.3]])

    single = bd.head_distillation_loss(student, teacher, one, 2.0)
    repeated = bd.head_distillation_loss(
        student.repeat(1, 4), teacher.repeat(1, 4), many, 2.0
    )

    torch.testing.assert_close(single, repeated)
    nothing = bd.match_heads(["a"], {}, ["z"], {})
    assert (
        bd.head_distillation_loss(student[:, :0], teacher[:, :0], nothing, 2.0) is None
    )


def _heads(concepts, groups, dim=8, seed=0):
    torch.manual_seed(seed)
    return bb._MultiTaskHeads(dim, SimpleNamespace(concepts=concepts, groups=groups))


def test_shared_logits_are_one_matrix_product_in_the_shared_layout():
    heads = _heads(["a", "b", "c"], {"g": ["x", "y", "z"], "h": ["p", "q"]})
    match = bd.match_heads(
        ["a", "b", "c"],
        {"g": ["x", "y", "z"], "h": ["p", "q"]},
        ["a", "c"],
        {"g": ["x", "y", "z"], "h": ["q", "p"]},
    )
    calls = []
    for head in heads.groups.values():
        head.register_forward_hook(lambda *_args: calls.append(1))
    features = torch.randn(5, 8, generator=torch.Generator().manual_seed(3))

    logits = bd.shared_head_logits(features, heads, match)

    assert calls == []
    expected = torch.cat(
        [
            F.linear(features, *heads.binary_rows(["a"])),
            F.linear(features, *heads.binary_rows(["c"])),
            F.linear(features, heads.groups["g"].weight, heads.groups["g"].bias),
        ],
        dim=1,
    )
    torch.testing.assert_close(logits, expected)


# -- (d) an image with no label still learns from the teacher ------------------


def test_rows_without_a_label_get_a_teacher_gradient():
    heads = _heads(["a", "b"], {"g": ["x", "y", "z"]})
    match = bd.match_heads(
        ["a", "b"], {"g": ["x", "y", "z"]}, ["a", "b"], {"g": ["x", "y", "z"]}
    )
    features = torch.randn(3, 8, generator=torch.Generator().manual_seed(4))
    features.requires_grad_()
    binary = [{"a": 1.0}, {}, {}]
    groups = [{"g": 0}, {}, {}]

    supervised = bb._batch_loss(features, heads, binary, groups, CPU)
    (supervised_grad,) = torch.autograd.grad(supervised, features)
    assert float(supervised_grad[1:].abs().sum()) == 0.0

    teacher_logits = torch.randn(
        3, match.width, generator=torch.Generator().manual_seed(5)
    )
    distilled = bd.head_distillation_loss(
        bd.shared_head_logits(features, heads, match), teacher_logits, match, 2.0
    )
    (distilled_grad,) = torch.autograd.grad(distilled, features)
    assert bool((distilled_grad[1:].abs().sum(dim=1) > 0).all())


def test_unlabelled_images_train_in_batches_of_their_own(
    tmp_path, teacher_path, monkeypatch
):
    """Batch size 1: each unlabelled image is a batch with no label at all.

    The plain loop skips such a batch; a distillation run must step on it.
    """
    outcomes = []
    real_batch_loss = bb._batch_loss

    def _spy_batch_loss(*args, **kwargs):
        value = real_batch_loss(*args, **kwargs)
        outcomes.append(value is not None)
        return value

    head_calls = []
    real_head = bd.head_distillation_loss

    def _spy_head(*args, **kwargs):
        head_calls.append(1)
        return real_head(*args, **kwargs)

    monkeypatch.setattr(bb, "_batch_loss", _spy_batch_loss)
    monkeypatch.setattr(bd, "head_distillation_loss", _spy_head)
    frames = []
    request = _request(
        tmp_path,
        _records(tmp_path / "imgs", 8, _student_labels, unlabelled=3),
        run_id="student_unlabelled",
        max_steps=None,
        config={"batch_size": 1, "oversample_positives": False},
        distillation={"teacher_checkpoint_path": teacher_path},
    )

    _run(request, frames.append)

    assert _preparing(frames)["distillation"]["unlabelled_images"] == 3
    assert outcomes.count(False) == 3
    steps = _epoch_frame(frames)["steps"]
    assert steps == len(head_calls) == len(outcomes)


# -- (b) the teacher is frozen and untouched -----------------------------------


def test_teacher_weights_are_unchanged_and_carry_no_gradient(mixed_run):
    teacher = mixed_run["teacher"].model
    before = mixed_run["teacher_before"]

    assert not teacher.training
    for name, value in teacher.backbone.state_dict().items():
        assert torch.equal(value, before[name]), name
    for name, value in teacher.heads.state_dict().items():
        assert torch.equal(value, before[f"heads.{name}"]), name
    for parameter in teacher.parameters():
        assert parameter.requires_grad is False
        assert parameter.grad is None

    teacher_ids = {id(p) for p in teacher.parameters()}
    optimized = {
        id(p) for group in mixed_run["optimizer"].param_groups for p in group["params"]
    }
    assert not teacher_ids & optimized
    assert not teacher_ids & {id(p) for p in mixed_run["model"].parameters()}


# -- (c) a distillation run exports an ordinary candidate ----------------------


def test_distilled_candidate_has_the_student_layout(mixed_run):
    path = mixed_run["result"]["candidate_checkpoint_path"]
    state = load_file(path)
    metadata = _metadata(path)

    assert not any("projector" in key or "teacher" in key for key in state)
    assert not any(key.startswith("backbone.") for key in state)
    assert "heads.binary.sharpness.weight" in state
    assert "heads.binary.blurry.weight" not in state
    assert state["heads.groups.lighting.weight"].shape[0] == 3
    assert json.loads(metadata["heads_concepts_json"]) == ["sharpness", "watermark"]
    assert json.loads(metadata["heads_groups_json"])["lighting"] == [
        "day",
        "dusk",
        "night",
    ]

    distilled_from = json.loads(metadata["distilled_from"])
    assert distilled_from["training_run_id"] == "teacher_run"
    assert distilled_from["convnextv2_size"] == "atto"
    used = json.loads(metadata["distillation_json"])
    assert used["logit_weight"] == 1.0
    assert used["feature_weight"] == 1.0
    assert used["temperature"] == 2.0
    assert used["supervised_weight"] == 1.0


def test_distilled_candidate_works_as_a_trunk_and_as_a_teacher(mixed_run):
    from bittrainer.backbone_init import apply_backbone_init
    from bittrainer.model import create_model

    path = mixed_run["result"]["candidate_checkpoint_path"]
    trunk = create_model(model_size="atto", pretrained=False, num_classes=0)
    assert apply_backbone_init(
        trunk, {"source": "local_active", "checkpoint_path": path}
    )

    again = bd.load_teacher(path, CPU)
    assert again.concepts == ["sharpness", "watermark"]
    assert again.groups["lighting"] == ["day", "dusk", "night"]


def test_head_output_term_was_used_on_every_step(mixed_run):
    calls = mixed_run["head_calls"]
    steps = _epoch_frame(mixed_run["frames"])["steps"]

    assert steps > 0
    assert len(calls) == steps
    assert all(math.isfinite(call["value"]) and call["value"] > 0 for call in calls)
    terms = _epoch_frame(mixed_run["frames"])["loss_terms"]
    assert set(terms) >= {"head", "feature", "supervised"}


# -- (e) only heads both sides agree on are distilled --------------------------


def test_renamed_and_reshaped_heads_are_not_distilled_but_counted(mixed_run):
    calls = mixed_run["head_calls"]
    # watermark (one column) + shot_type (two classes); never sharpness/lighting.
    assert {call["width"] for call in calls} == {3}
    assert calls[0]["concepts"] == ("watermark",)
    assert calls[0]["groups"] == (("shot_type", 2),)

    preparing = _preparing(mixed_run["frames"])
    assert preparing["shared_heads"] == 2
    assert preparing["student_only_heads"] == 2
    assert preparing["teacher_only_heads"] == 2
    assert preparing["distillation"]["mismatched_groups"] == ["lighting"]


def test_head_matching_rules():
    match = bd.match_heads(
        ["sharpness", "watermark"],
        {"lighting": ["day", "dusk", "night"], "shot_type": ["closeup", "wide"]},
        ["blurry", "watermark"],
        {"lighting": ["day", "night"], "shot_type": ["closeup", "wide"]},
    )
    assert match.concepts == ("watermark",)
    assert match.groups == (("shot_type", 2),)
    assert match.shared == 2
    assert sorted(match.student_only) == ["binary/sharpness", "group/lighting"]
    assert sorted(match.teacher_only) == ["binary/blurry", "group/lighting"]
    assert match.mismatched_groups == ("lighting",)


# -- (f) nothing to teach refuses to start -------------------------------------


def test_nothing_shared_and_no_feature_term_refuses(tmp_path, teacher_path):
    request = _request(
        tmp_path,
        _records(tmp_path / "imgs", 8, _foreign_labels),
        run_id="student_refused",
        distillation={"teacher_checkpoint_path": teacher_path, "feature_weight": 0.0},
    )
    with pytest.raises(RuntimeError, match="shares no heads"):
        _run(request)


def test_both_distillation_weights_zero_refuses(tmp_path, teacher_path):
    request = _request(
        tmp_path,
        _records(tmp_path / "imgs", 8, _student_labels),
        run_id="student_zero",
        distillation={
            "teacher_checkpoint_path": teacher_path,
            "feature_weight": 0.0,
            "logit_weight": 0.0,
        },
    )
    with pytest.raises(RuntimeError, match="both 0"):
        _run(request)


def test_a_teacher_without_heads_or_with_missing_tensors_is_refused(
    tmp_path, teacher_path
):
    from bittrainer.model import create_model

    donor = create_model(model_size="atto", pretrained=False, num_classes=0)
    bare = tmp_path / "bare.safetensors"
    save_file(donor.state_dict(), str(bare), metadata={"convnextv2_size": "atto"})
    with pytest.raises(RuntimeError, match="no head tensors"):
        bd.load_teacher(bare, CPU)

    state = load_file(teacher_path)
    state.pop(next(key for key in state if not key.startswith("heads.")))
    broken = tmp_path / "broken.safetensors"
    save_file(state, str(broken), metadata=_metadata(teacher_path))
    with pytest.raises(RuntimeError, match="does not load completely"):
        bd.load_teacher(broken, CPU)

    with pytest.raises(FileNotFoundError):
        bd.load_teacher(tmp_path / "missing.safetensors", CPU)


# -- (g) the request decides which task runs -----------------------------------


def test_only_a_teacher_path_switches_on_distillation(
    tmp_path, teacher_path, monkeypatch
):
    from bittrainer.generic import generic_trainer

    seen = []

    def _fake_run(self, task, **_kwargs):
        seen.append(task)
        return {}

    monkeypatch.setattr(generic_trainer.GenericTrainer, "run", _fake_run)
    base = _request(
        tmp_path, _records(tmp_path / "imgs", 4, _student_labels), run_id="dispatch"
    )
    variants = [
        base,
        {**base, "distillation": None},
        {**base, "distillation": {}},
        {**base, "distillation": {"teacher_checkpoint_path": teacher_path}},
    ]
    for request in variants:
        bb._train_backbone(request, lambda _frame: None, threading.Event())

    assert [type(task) for task in seen] == [
        BackboneTask,
        BackboneTask,
        BackboneTask,
        BackboneDistillTask,
    ]
    distill = seen[-1]
    assert distill.logit_weight == 1.0
    assert distill.feature_weight == 1.0
    assert distill.temperature == 2.0
    assert distill.supervised_weight == 1.0


# -- (h) the projector rides the backups, never the candidate -------------------


def test_projector_is_backed_up_resumed_and_never_exported(
    tmp_path, teacher_path, monkeypatch
):
    request = _request(
        tmp_path,
        _records(tmp_path / "imgs", 8, _student_labels),
        run_id="student_pause",
        max_steps=4,
        config={"backup_dir": str(tmp_path / "backups"), "progress_every_steps": 1},
        distillation={"teacher_checkpoint_path": teacher_path},
    )
    pause = threading.Event()

    def _pause_after_first_step(frame):
        if frame.get("stage") == "training_step":
            pause.set()

    paused = bb._train_backbone(
        request, _pause_after_first_step, threading.Event(), pause_event=pause
    )
    assert paused["paused"] is True
    backup = torch.load(paused["backup_path"], map_location="cpu", weights_only=False)
    assert "projector.weight" in backup["model"]
    assert all(
        key.split(".")[0] in {"backbone", "heads", "projector"}
        for key in backup["model"]
    )
    assert backup["fingerprint"]["distillation"]["teacher"] == Path(teacher_path).name

    resumed_projector = {}
    real_optimizer = bt.make_optimizer

    def _spy_optimizer(model, **kwargs):
        resumed_projector["weight"] = model.projector.weight.detach().clone()
        return real_optimizer(model, **kwargs)

    monkeypatch.setattr(bt, "make_optimizer", _spy_optimizer)
    frames = []
    resumed = {
        **request,
        "training_config": {
            **request["training_config"],
            "resume_from": paused["backup_path"],
        },
    }
    result = bb._train_backbone(resumed, frames.append, threading.Event())

    assert any(f.get("type") == "training_resumed" for f in frames)
    assert not any(f.get("type") == "resume_skipped" for f in frames)
    assert torch.equal(resumed_projector["weight"], backup["model"]["projector.weight"])
    exported = load_file(result["candidate_checkpoint_path"])
    assert not any("projector" in key for key in exported)


# -- (i) plain and distillation backups never cross ----------------------------


def _fingerprint(task):
    ctx = task.make_context(None, None, None, None)
    task.fingerprint_init(ctx)
    return ctx.fingerprint


def test_plain_and_distillation_fingerprints_differ(tmp_path, teacher_path):
    request = _request(
        tmp_path, _records(tmp_path / "imgs", 8, _student_labels), run_id="fingerprints"
    )
    teacher = {"teacher_checkpoint_path": teacher_path}
    plain = _fingerprint(BackboneTask(request))
    distilled = _fingerprint(BackboneDistillTask({**request, "distillation": teacher}))
    hotter = _fingerprint(
        BackboneDistillTask(
            {**request, "distillation": {**teacher, "temperature": 4.0}}
        )
    )

    # A distillation run never resumes a plain backup, nor the reverse.
    assert not fingerprint_matches(plain, distilled)
    assert not fingerprint_matches(distilled, plain)
    # Changing the recipe starts fresh too.
    assert not fingerprint_matches(distilled, hotter)
    assert fingerprint_matches(distilled, distilled)


# -- the GRN layer -------------------------------------------------------------


def test_teacher_and_student_both_run_the_lean_grn_layer(mixed_run):
    from timm.layers.grn import GlobalResponseNorm

    from bittrainer.lean_grn import LeanGlobalResponseNorm

    for model in (mixed_run["teacher"].model, mixed_run["model"]):
        layers = [m for m in model.modules() if isinstance(m, GlobalResponseNorm)]
        assert layers
        assert all(type(layer) is LeanGlobalResponseNorm for layer in layers)
