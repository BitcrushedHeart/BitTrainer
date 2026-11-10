"""Distil a trained backbone into a smaller one (Bitcrush ISSUE-1655).

The owner trains one large teacher (ConvNeXt V2 Base) on a rented GPU, then
distils it into smaller students (nano, pico, femto, atto ...) on the same data.
A student run is an ordinary backbone run with a teacher attached: when
``request["distillation"]`` names a teacher, ``_train_backbone`` runs
:class:`BackboneDistillTask` instead of ``BackboneTask``::

    request["distillation"] = {
        "teacher_checkpoint_path": str,  # a candidate written by BackboneTask.finalize
        "logit_weight": 1.0,             # head-output term
        "feature_weight": 1.0,           # pooled-feature term
        "temperature": 2.0,              # softening for the head-output term
        "supervised_weight": 1.0,        # the ordinary labelled loss
    }

Every step the frozen teacher sees the student's own augmented batch, and the
loss becomes::

    supervised_weight * labelled loss          (when the batch has any label)
    + logit_weight    * head-output term       (every image, every shared head)
    + feature_weight  * pooled-feature term    (every image)

The head-output term is the point: labels cover a few (image, head) pairs, the
teacher answers for all of them. For the same reason, images whose records
carry no usable label join the training split (never validation) with empty
labels, so a pool's unlabelled images still teach the student.

Everything else is the parent's: the sampling plan, the fp16 GradScaler and its
count of skipped steps, clipping, EMA, channels-last, the heartbeat, max_steps,
cancel, pause / backup / resume, the tail and its batch size, validation on real
labels and the candidate export. The parent's preflight and automatic batch size
run this task's own step, so the teacher's forward pass is measured too.

The candidate keeps the student's ordinary layout, so it can be a trunk or a
teacher in its turn. The feature projector trains with the student and rides the
backups (it is part of the model's state dict) but is never exported: it only
means something next to this one teacher.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

import bittrainer.backbone_trainer as bb
from bittrainer.generic.optimizer import clip_gradients
from bittrainer.generic.task import TaskContext
from bittrainer.generic.tasks.backbone_task import BackboneTask
from bittrainer.lean_grn import use_lean_grn
from bittrainer.model import create_model

logger = logging.getLogger(__name__)

# The resume fingerprint's ``best_model_name`` for a distillation run. The
# fingerprint check only compares the keys of the CURRENT run, so an extra
# "distillation" key alone would stop a distillation run resuming a plain
# backup but not the reverse; a different value for a key every backbone run
# carries stops both.
_DISTILLED_MODEL_NAME = "backbone_candidate_distilled"


# --------------------------------------------------------------------------- #
# Losses                                                                      #
# --------------------------------------------------------------------------- #


def soft_binary_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Yes/no heads: BCE of ``student / T`` against ``sigmoid(teacher / T)``.

    Averaged over every (image, head) entry, so the family's scale does not
    depend on how many heads it has. Scaled by ``T**2`` like the group term:
    the gradient of either soft loss with respect to a student logit shrinks
    as ``1 / T**2`` (Hinton et al.), so without the factor a hotter temperature
    would quietly weaken distillation, and leaving it off one family only would
    tilt the balance between yes/no heads and groups by ``T**2``.
    """
    t = float(temperature)
    targets = torch.sigmoid(teacher_logits.float() / t)
    loss = F.binary_cross_entropy_with_logits(student_logits.float() / t, targets)
    return loss * (t * t)


def soft_group_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """One single-label group: ``KL(softmax(teacher / T) || softmax(student / T))``.

    Summed over the classes, averaged over the images, times ``T**2``.
    """
    t = float(temperature)
    log_student = F.log_softmax(student_logits.float() / t, dim=-1)
    log_teacher = F.log_softmax(teacher_logits.float() / t, dim=-1)
    loss = F.kl_div(log_student, log_teacher, reduction="batchmean", log_target=True)
    return loss * (t * t)


def feature_distillation_loss(
    projected: torch.Tensor, teacher_features: torch.Tensor
) -> torch.Tensor:
    """Cosine distance between the projected student feature and the teacher's.

    Both vectors are compared by direction only. The pooled feature's length is
    set by each trunk's own final norm and differs between sizes; the
    direction is what the heads read. Cosine distance is the squared distance
    between the two L2-normalised vectors, halved, so it is the same signal as
    MSE on normalised vectors, but it stays in [0, 2] whatever the teacher's
    width: an MSE averaged over the dimensions would shrink with a wider
    teacher and change what ``feature_weight`` means.
    """
    cosine = F.cosine_similarity(
        projected.float(), teacher_features.float(), dim=-1, eps=1e-6
    )
    return (1.0 - cosine).mean()


# --------------------------------------------------------------------------- #
# Which heads the teacher can teach                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HeadMatch:
    """The heads teacher and student agree on, in the student's order.

    The shared logits are laid out as one row per image: a column per shared
    yes/no head, then each shared group's classes side by side.
    """

    concepts: tuple[str, ...]
    groups: tuple[tuple[str, int], ...]
    student_only: tuple[str, ...]
    teacher_only: tuple[str, ...]
    mismatched_groups: tuple[str, ...]

    @property
    def shared(self) -> int:
        return len(self.concepts) + len(self.groups)

    @property
    def width(self) -> int:
        return len(self.concepts) + sum(n for _name, n in self.groups)

    @property
    def group_spans(self) -> list[tuple[int, int]]:
        spans = []
        start = len(self.concepts)
        for _name, n in self.groups:
            spans.append((start, start + n))
            start += n
        return spans


def match_heads(
    student_concepts, student_groups: dict, teacher_concepts, teacher_groups: dict
) -> HeadMatch:
    """Pair the student's heads with the teacher's.

    A yes/no head is shared when both sides have the same name. A group is
    shared only when the name AND the ordered class list are identical: a class
    added, dropped or reordered changes what each output column means, so the
    teacher's answer would be read against the wrong classes. Unshared student
    heads still learn from their labels; unshared teacher heads are ignored.
    """
    teacher_concept_set = set(teacher_concepts)
    concepts = tuple(c for c in student_concepts if c in teacher_concept_set)
    groups: list[tuple[str, int]] = []
    mismatched: list[str] = []
    for name, classes in student_groups.items():
        teacher_classes = teacher_groups.get(name)
        if teacher_classes is None:
            continue
        if list(teacher_classes) == list(classes):
            groups.append((name, len(classes)))
        else:
            mismatched.append(name)
    shared_concepts = set(concepts)
    shared_groups = {name for name, _n in groups}
    student_only = [f"binary/{c}" for c in student_concepts if c not in shared_concepts]
    student_only += [f"group/{g}" for g in student_groups if g not in shared_groups]
    teacher_only = [f"binary/{c}" for c in teacher_concepts if c not in shared_concepts]
    teacher_only += [f"group/{g}" for g in teacher_groups if g not in shared_groups]
    return HeadMatch(
        concepts=concepts,
        groups=tuple(groups),
        student_only=tuple(student_only),
        teacher_only=tuple(teacher_only),
        mismatched_groups=tuple(mismatched),
    )


def shared_head_logits(
    features: torch.Tensor, heads: bb._MultiTaskHeads, match: HeadMatch
) -> torch.Tensor:
    """Logits of every shared head from ONE matrix product, in the shared layout.

    With hundreds of concepts a call per head would be hundreds of tiny kernels
    per step; stacking the heads' rows keeps it to one.
    """
    weight, bias = heads.binary_rows(match.concepts)
    groups = [heads.groups[name] for name, _n in match.groups]
    weight = torch.cat([weight, *[m.weight for m in groups]])
    bias = torch.cat([bias, *[m.bias for m in groups]])
    return F.linear(features, weight, bias).float()


def head_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    match: HeadMatch,
    temperature: float,
) -> torch.Tensor | None:
    """The head-output term over the shared layout, or None with nothing shared.

    Each family is averaged on its own (yes/no over every image-head entry,
    groups over their groups), then the families present are averaged with
    equal weight, as the validation ``selection`` score treats them. The term
    therefore keeps its scale as heads are added, and a few groups are not
    drowned by hundreds of concepts.
    """
    families: list[torch.Tensor] = []
    k = len(match.concepts)
    if k:
        families.append(
            soft_binary_loss(student_logits[:, :k], teacher_logits[:, :k], temperature)
        )
    if match.groups:
        per_group = [
            soft_group_loss(
                student_logits[:, start:stop],
                teacher_logits[:, start:stop],
                temperature,
            )
            for start, stop in match.group_spans
        ]
        families.append(torch.stack(per_group).mean())
    if not families:
        return None
    return torch.stack(families).mean()


# --------------------------------------------------------------------------- #
# The teacher                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class Teacher:
    path: Path
    model: bb._BackboneWithHeads
    concepts: list[str]
    groups: dict[str, list[str]]
    feature_dim: int
    metadata: dict[str, str]


def load_teacher(path, device) -> Teacher:
    """Rebuild a frozen teacher from a backbone candidate file.

    The candidate holds bare trunk keys, ``heads.*`` tensors, and metadata
    naming the trunk size and the head layout. Both halves must load exactly:
    a partly loaded teacher would teach from random weights without a word.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Distillation teacher checkpoint not found: {path}")
    with safe_open(str(path), framework="pt") as handle:
        metadata = dict(handle.metadata() or {})
    state = load_file(str(path))
    head_state = {
        key[len("heads.") :]: value
        for key, value in state.items()
        if key.startswith("heads.")
    }
    if not head_state:
        raise RuntimeError(
            f"Distillation teacher {path.name} has no head tensors: it is a bare "
            "trunk, and a teacher needs its heads to answer for the student. Use a "
            "candidate written by a full backbone run."
        )
    trunk_state = {k: v for k, v in state.items() if not k.startswith("heads.")}
    size = metadata.get("convnextv2_size")
    if not size:
        raise RuntimeError(
            f"Distillation teacher {path.name} does not record its convnextv2_size."
        )
    try:
        concepts = [
            str(c) for c in json.loads(metadata.get("heads_concepts_json") or "[]")
        ]
        groups = {
            str(name): [str(c) for c in classes]
            for name, classes in json.loads(
                metadata.get("heads_groups_json") or "{}"
            ).items()
        }
    except (TypeError, ValueError, AttributeError) as exc:
        raise RuntimeError(
            f"Distillation teacher {path.name} has an unreadable head layout: {exc}"
        ) from exc

    trunk = create_model(
        model_size=size, pretrained=False, num_classes=0, drop_path_rate=0.0
    )
    heads = bb._MultiTaskHeads(
        trunk.num_features, SimpleNamespace(concepts=concepts, groups=groups)
    )
    try:
        trunk.load_state_dict(trunk_state, strict=True)
        heads.load_state_dict(head_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Distillation teacher {path.name} does not load completely into a "
            f"ConvNeXt V2 {size} trunk with its recorded heads: {exc}"
        ) from exc
    model = bb._BackboneWithHeads(trunk, heads).to(device)
    model.eval()
    model.requires_grad_(False)
    logger.info(
        "Distillation teacher %s: %s trunk, %d yes/no heads, %d groups",
        path.name,
        size,
        len(concepts),
        len(groups),
    )
    return Teacher(path, model, concepts, groups, int(trunk.num_features), metadata)


class _StudentWithProjector(bb._BackboneWithHeads):
    """The student plus the feature projector, as one module.

    Holding the projector inside the model puts it in the optimizer, the EMA and
    the backup's model state with no extra plumbing. The export reads only
    ``backbone`` and ``heads``, so the projector never reaches the candidate.
    """

    def __init__(self, backbone: nn.Module, heads: nn.Module, projector: nn.Module):
        super().__init__(backbone, heads)
        self.projector = projector


def _number(spec: dict, key: str, default: float) -> float:
    value = spec.get(key)
    return float(default if value is None else value)


# --------------------------------------------------------------------------- #
# The task                                                                    #
# --------------------------------------------------------------------------- #


class BackboneDistillTask(BackboneTask):
    """A backbone run that also learns from a frozen teacher's answers."""

    def __init__(
        self,
        request: dict,
        *,
        cancel_event: threading.Event | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        super().__init__(request, cancel_event=cancel_event, stop_event=stop_event)
        spec = dict(request.get("distillation") or {})
        teacher_path = spec.get("teacher_checkpoint_path")
        if not teacher_path:
            raise ValueError(
                "Distillation needs distillation.teacher_checkpoint_path to name "
                "the teacher's candidate file."
            )
        self.teacher_path = Path(teacher_path)
        self.logit_weight = _number(spec, "logit_weight", 1.0)
        self.feature_weight = _number(spec, "feature_weight", 1.0)
        self.temperature = _number(spec, "temperature", 2.0)
        self.supervised_weight = _number(spec, "supervised_weight", 1.0)
        weights = {
            "logit_weight": self.logit_weight,
            "feature_weight": self.feature_weight,
            "supervised_weight": self.supervised_weight,
        }
        negative = [name for name, value in weights.items() if value < 0]
        if negative:
            raise ValueError(f"Distillation weights cannot be negative: {negative}")
        if self.temperature <= 0:
            raise ValueError("Distillation temperature must be above 0.")

        self.teacher: Teacher | None = None
        self.head_match: HeadMatch | None = None
        self._uses_heads = False
        self._unlabelled: list = []
        # The head report waits for the parent's preparing frame (see _emit).
        self._pending_report: dict | None = None
        self._last_terms: dict[str, float] = {}

    def _recipe(self) -> dict:
        return {
            "logit_weight": self.logit_weight,
            "feature_weight": self.feature_weight,
            "temperature": self.temperature,
            "supervised_weight": self.supervised_weight,
        }

    # -- one-time setup ----------------------------------------------------
    def make_context(
        self, progress_callback, stop_event, stop_now_event, pause_event
    ) -> TaskContext:
        ctx = super().make_context(
            progress_callback, stop_event, stop_now_event, pause_event
        )
        self.teacher = load_teacher(self.teacher_path, ctx.device)
        match = match_heads(
            self.vocab.concepts,
            self.vocab.groups,
            self.teacher.concepts,
            self.teacher.groups,
        )
        self.head_match = match
        self._uses_heads = match.shared > 0 and self.logit_weight > 0
        if not self._uses_heads and self.feature_weight <= 0:
            if match.shared == 0:
                raise RuntimeError(
                    f"Distillation refused: the student shares no heads with teacher "
                    f"{self.teacher_path.name} (same yes/no name, or same group with "
                    "the same classes) and feature_weight is 0, so the teacher has "
                    "nothing to teach."
                )
            raise RuntimeError(
                "Distillation refused: logit_weight and feature_weight are both 0, "
                "so the teacher would take no part in training."
            )
        if self.lean_grn:  # the teacher's forward pass is a third of each step
            use_lean_grn(self.teacher.model)
        if self.channels_last:
            self.teacher.model.to(memory_format=torch.channels_last)

        self._unlabelled = self._unlabelled_samples(
            list(self.request.get("records") or [])
        )
        self._pending_report = {
            "shared_heads": match.shared,
            "student_only_heads": len(match.student_only),
            "teacher_only_heads": len(match.teacher_only),
            "distillation": {
                "teacher": self.teacher_path.name,
                "teacher_size": self.teacher.metadata.get("convnextv2_size"),
                "shared_concepts": len(match.concepts),
                "shared_groups": len(match.groups),
                "mismatched_groups": list(match.mismatched_groups),
                "unlabelled_images": len(self._unlabelled),
                **self._recipe(),
            },
        }
        return ctx

    def _unlabelled_samples(self, records: list[dict]) -> list:
        """Images with a file but no usable label, as label-free training samples.

        A record counts as labelled by the same rule ``_build_samples`` uses; an
        image already present through a labelled record is not added twice.
        """
        usable = bb._POSITIVE_LABELS | bb._NEGATIVE_LABELS
        taken = {s.path for s in self.train_samples + self.val_samples}
        unlabelled_records = []
        for record in records:
            has_binary = any(
                str(label).lower() in usable
                for label in (record.get("binary") or {}).values()
            )
            has_group = any(
                str(class_name)
                in (self.vocab.groups.get(bb._safe_module_key(group)) or ())
                for group, class_name in (record.get("groups") or {}).items()
            )
            if has_binary or has_group:
                if record.get("content_hash"):
                    taken.add(str(record["content_hash"]))
            else:
                unlabelled_records.append(record)
        samples = []
        for record in unlabelled_records:
            path = next(
                (p for p in record.get("file_paths") or [] if Path(p).is_file()), None
            )
            if path is None:
                continue
            key = str(record.get("content_hash") or path)
            if key in taken or path in taken:
                continue
            taken.add(key)
            samples.append(bb._Sample(path, {}, {}))
        return samples

    def fingerprint_init(self, ctx: TaskContext) -> None:
        # The parent builds the coordinator and the plain fingerprint, and would
        # load a backup against that plain fingerprint. Hold resume_from back so
        # the one load happens against the distillation fingerprint instead.
        resume_from = self.config.pop("resume_from", None)
        try:
            super().fingerprint_init(ctx)
        finally:
            if resume_from is not None:
                self.config["resume_from"] = resume_from
        ctx.fingerprint = {
            **ctx.fingerprint,
            "best_model_name": _DISTILLED_MODEL_NAME,
            # The file name, not the full path, so moving the folder (a rented
            # machine, a new drive) does not orphan the backups.
            "distillation": {"teacher": self.teacher_path.name, **self._recipe()},
        }
        ctx.resume_state = (
            ctx.coordinator.load_resume(ctx.fingerprint, resume_from=resume_from)
            if resume_from
            else None
        )

    def _emit(self, stage: str, status_text: str, **extra) -> None:
        # The head counts ride the parent's own preparing frame, and the
        # per-term losses its epoch frame, so a reader of either needs nothing new.
        if stage == "preparing" and self._pending_report is not None:
            report, self._pending_report = self._pending_report, None
            status_text = (
                f"{status_text}; distilling from {self.teacher_path.name} "
                f"({report['shared_heads']} shared heads)"
            )
            extra = {**extra, **report}
        elif stage == "training" and "loss" in extra and self._last_terms:
            extra = {**extra, "loss_terms": dict(self._last_terms)}
        super()._emit(stage, status_text, **extra)

    def _plan_epoch(self, epoch: int):
        # The labelled plan is exactly a plain run's; label-free images ride on
        # top and learn from the teacher alone.
        planned, stats = super()._plan_epoch(epoch)
        return planned + self._unlabelled, stats

    def create_model(self, ctx: TaskContext, resume_state: dict | None):
        # Build without the resume state: the parent's module has no projector
        # and could not take a backup that carries one.
        model = super().create_model(ctx, None)
        if self.feature_weight > 0:
            projector = nn.Linear(self.feature_dim, self.teacher.feature_dim)
            model = _StudentWithProjector(
                model.backbone, model.heads, projector.to(ctx.device)
            )
        if resume_state is not None:
            model.load_state_dict(resume_state["model"])
        self.model = model
        return model

    # -- per-epoch ---------------------------------------------------------
    def _step_loss(
        self,
        model,
        features: torch.Tensor,
        teacher_features: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        binary_labels,
        group_labels,
        device,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        terms: dict[str, torch.Tensor] = {}
        if self.supervised_weight > 0:
            supervised = bb._batch_loss(
                features,
                model.heads,
                binary_labels,
                group_labels,
                device,
                pos_weight=self._pos_weight,
                group_label_smoothing=self.group_label_smoothing,
                reduction=self.loss_reduction,
            )
            if supervised is not None:
                terms["supervised"] = supervised
        if teacher_logits is not None:
            head = head_distillation_loss(
                shared_head_logits(features, model.heads, self.head_match),
                teacher_logits,
                self.head_match,
                self.temperature,
            )
            if head is not None:
                terms["head"] = head
        if self.feature_weight > 0:
            terms["feature"] = feature_distillation_loss(
                model.projector(features), teacher_features
            )
        if not terms:
            return None, terms
        weights = {
            "supervised": self.supervised_weight,
            "head": self.logit_weight,
            "feature": self.feature_weight,
        }
        loss = sum(weights[name] * value for name, value in terms.items())
        return loss, terms

    def train_epoch(
        self,
        ctx: TaskContext,
        model,
        optimizer,
        train_loader,
        *,
        step_callback,
        boundary_hook,
        start_batch: int,
    ):
        # The parent's loop with the teacher's forward and two more loss terms.
        # Keep the scaler / clip / EMA / heartbeat order in step with the parent.
        device = ctx.device
        scaler = self._scaler
        teacher = self.teacher.model
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        images_seen = 0
        # Per-term sums stay on the device until the epoch ends: a float() per
        # term per step would add a device sync to every step.
        term_sums: dict[str, torch.Tensor] = {}
        term_counts: dict[str, int] = {}
        total_batches = len(train_loader)
        started = time.perf_counter()
        self._start_skip_count(scaler)
        for images, binary_labels, group_labels in train_loader:
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise bb.BackboneTrainingCancelled
            if self.max_steps is not None and self.step >= int(self.max_steps):
                self.steps_stop_event.set()
                break
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device, non_blocking=True)
            if self.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.amp.autocast(
                device_type=device.type, dtype=self.amp_dtype, enabled=self.amp_enabled
            ):
                # Same batch, same augmentation, same precision as the student.
                with torch.no_grad():
                    teacher_features = teacher.backbone(images).float()
                    teacher_logits = (
                        shared_head_logits(
                            teacher_features, teacher.heads, self.head_match
                        )
                        if self._uses_heads
                        else None
                    )
                features = model.backbone(images).float()
                loss, terms = self._step_loss(
                    model,
                    features,
                    teacher_features,
                    teacher_logits,
                    binary_labels,
                    group_labels,
                    device,
                )
            if loss is None:
                continue
            clip = self.clip_grad_norm and self.clip_grad_norm > 0
            if self._dry_run:
                # A step for its memory only (the parent's _dry_step): the
                # teacher's forward is in it, and nothing is updated.
                (scaler.scale(loss) if scaler is not None else loss).backward()
                break
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if clip:
                    clip_gradients(model, self.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                self._count_skip(scaler)
            else:
                loss.backward()
                if clip:
                    clip_gradients(model, self.clip_grad_norm)
                optimizer.step()
            if self.ema is not None:
                self.ema.update(model)
            self.step += 1
            epoch_loss += float(loss.detach())
            for name, value in terms.items():
                detached = value.detach()
                term_sums[name] = (
                    term_sums[name] + detached if name in term_sums else detached
                )
                term_counts[name] = term_counts.get(name, 0) + 1
            epoch_batches += 1
            images_seen += int(images.shape[0])
            if (
                step_callback is not None
                and epoch_batches % self.progress_every_steps == 0
            ):
                step_callback(
                    epoch_batches,
                    total_batches,
                    epoch_loss / epoch_batches,
                    images_seen,
                    time.perf_counter() - started,
                )
            if boundary_hook(epoch_batches) == "stop":  # pause backed up mid-epoch
                break
        self._train_seconds = time.perf_counter() - started
        self._train_images = images_seen
        self._last_terms = {
            name: float(total) / term_counts[name] for name, total in term_sums.items()
        }
        return epoch_loss / max(epoch_batches, 1)

    # -- finalisation ------------------------------------------------------
    def _candidate_metadata(self, validation_score, validation_metrics) -> dict:
        metadata = super()._candidate_metadata(validation_score, validation_metrics)
        teacher_metadata = self.teacher.metadata if self.teacher is not None else {}
        distilled_from = {
            key: teacher_metadata[key]
            for key in ("training_run_id", "convnextv2_size")
            if teacher_metadata.get(key)
        }
        distilled_from["teacher_checkpoint"] = self.teacher_path.name
        metadata["distilled_from"] = distilled_from
        used = dict(self._recipe())
        if self.head_match is not None:
            used.update(
                shared_heads=self.head_match.shared,
                student_only_heads=len(self.head_match.student_only),
                teacher_only_heads=len(self.head_match.teacher_only),
            )
        metadata["distillation_json"] = used
        return metadata
