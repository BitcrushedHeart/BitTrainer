"""Pre-training head-probe sweeps (Bitcrush ISSUE-0542).

Extracted verbatim from ``bittrainer.group_trainer``: the auto soft-label and
auto ``__none__``-oversample sweeps that run on cached features before the full
fine-tune, plus the RNG-snapshot / resolved-config plumbing the backup/resume
path shares. ``group_trainer`` re-imports every name from here (the FT loop reads
``_resolved_snapshot`` / ``_apply_resolved`` / ``_auto_oversample_enabled`` too),
keeping the objects identical.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn

from bittrainer.group_dataset import rare_group_none_target
from bittrainer.head_probe import (
    prepare_head_probe_tensors,
    train_head_probe,
    train_head_probe_from_tensors,
)
from bittrainer.selection import _metric_score, _score_metric_label

if TYPE_CHECKING:
    from bittrainer.embedding_cache import EmbeddingCache
    from bittrainer.group_trainer import GroupTrainConfig

logger = logging.getLogger(__name__)

_ORDINAL_SIGMA_CANDIDATES = [round(i / 10, 3) for i in range(11)]
_LABEL_SMOOTHING_CANDIDATES = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
# __none__ oversample sweep candidates: (label, oversample_none flag).
_OVERSAMPLE_NONE_CANDIDATES = [("off", False), ("1.5x", True)]


def _auto_softness_kind(config: GroupTrainConfig) -> str | None:
    if not config.auto_label_softness or config.multi_label:
        return None
    return "ordinal_sigma" if config.ordinal else "label_smoothing"


def _auto_softness_candidates(kind: str) -> list[float]:
    return _ORDINAL_SIGMA_CANDIDATES if kind == "ordinal_sigma" else _LABEL_SMOOTHING_CANDIDATES


def _capture_rng_state(device: torch.device) -> dict:
    # Delegate to training_state so the soft-label/oversample sweeps snapshot the
    # SAME generators (incl. python ``random``, which the samplers shuffle with)
    # that the backup/resume path relies on — one source of truth, no drift.
    from bittrainer.training_state import capture_rng_states

    return capture_rng_states(device)


def _restore_rng_state(state: dict, device: torch.device) -> None:
    from bittrainer.training_state import restore_rng_states

    restore_rng_states(state, device)


def _resolved_snapshot(config: GroupTrainConfig) -> dict:
    """Sweep/probe outcomes the pre-loop phase writes onto the config.

    The soft-label + __none__-oversample sweeps mutate these before the fine-tune
    loop; a resume skips those sweeps, so the backup must carry the RESOLVED
    values and re-apply them (via :func:`_apply_resolved`) so the loss, soft
    targets and dataset composition match the interrupted run exactly. Everything
    else (``use_soft``, ``mixup_enabled``, ``swa_start_epoch``, ``balance_mode``)
    is recomputed deterministically from these fields, so it needn't be stored.
    """
    return {
        "oversample_none": bool(config.oversample_none),
        "label_smoothing": float(config.label_smoothing),
        "ordinal_sigma": float(config.ordinal_sigma),
        "class_balance_mode": config.class_balance_mode,
    }


def _apply_resolved(config: GroupTrainConfig, resolved: dict) -> None:
    for key in ("oversample_none", "label_smoothing", "ordinal_sigma", "class_balance_mode"):
        if key in resolved:
            setattr(config, key, resolved[key])


def _apply_softness(config: GroupTrainConfig, kind: str, value: float) -> None:
    if kind == "ordinal_sigma":
        config.ordinal_sigma = float(value)
    else:
        config.label_smoothing = float(value)


def _softness_status_label(kind: str) -> str:
    return "ordinal softness" if kind == "ordinal_sigma" else "label smoothing"


def _softness_candidate_better(candidate: dict, incumbent: dict | None) -> bool:
    if incumbent is None:
        return True
    cand_score = float(candidate.get("score") or 0.0)
    inc_score = float(incumbent.get("score") or 0.0)
    if cand_score != inc_score:
        return cand_score > inc_score
    cand_loss = candidate.get("val_loss")
    inc_loss = incumbent.get("val_loss")
    if cand_loss is not None and inc_loss is not None and float(cand_loss) != float(inc_loss):
        return float(cand_loss) < float(inc_loss)
    return float(candidate["value"]) < float(incumbent["value"])


def _run_auto_softness_probe(
    model: nn.Module,
    config: GroupTrainConfig,
    embed_cache: EmbeddingCache,
    smart_cache: object | None,
    train_samples: list[dict],
    val_samples: list[dict],
    *,
    device: torch.device,
    none_index: int,
    cb: Callable[[dict], None],
    stop_event: object | None,
) -> dict:
    kind = _auto_softness_kind(config)
    if kind is None:
        return train_head_probe(
            model, embed_cache, smart_cache,
            train_samples, val_samples, config,
            device=device, none_index=none_index,
            cb=cb, stop_event=stop_event,
        )

    x_train, y_train, x_val, y_val = prepare_head_probe_tensors(
        embed_cache, smart_cache, train_samples, val_samples, config, cb=cb,
    )
    original_head_state = copy.deepcopy(model.head.state_dict())
    original_rng_state = _capture_rng_state(device)
    original_sigma = config.ordinal_sigma
    original_smoothing = config.label_smoothing
    candidates = _auto_softness_candidates(kind)
    label = _softness_status_label(kind)
    score_metric = _score_metric_label(config)

    best_row: dict | None = None
    best_probe: dict | None = None
    best_head_state: dict | None = None
    matrix: list[dict] = []
    sweep_start = time.monotonic()

    for idx, value in enumerate(candidates, start=1):
        if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
            break
        model.head.load_state_dict(original_head_state)
        _restore_rng_state(original_rng_state, device)
        _apply_softness(config, kind, value)
        cb({
            "type": "training_progress",
            "stage": "soft_label_tuning",
            "status_text": f"Testing {label} {value:g} ({idx}/{len(candidates)})",
            "step": idx,
            "total_steps": len(candidates),
            "softness_kind": kind,
            "softness_value": value,
            "soft_label_tuning_metric": score_metric,
        })
        candidate_start = time.monotonic()
        probe = train_head_probe_from_tensors(
            model, x_train, y_train, x_val, y_val, config,
            device=device, none_index=none_index, cb=cb,
            stop_event=stop_event,
            progress_stage="soft_label_tuning",
            progress_prefix=f"{label.capitalize()} {value:g}",
            score_metric=score_metric,
        )
        score = _metric_score(probe, config)
        row = {
            "kind": kind,
            "value": float(value),
            "score": score,
            "macro_f1": probe.get("macro_f1"),
            "qwk": probe.get("qwk"),
            "none_f1": probe.get("none_f1"),
            "none_recall": probe.get("none_recall"),
            "none_precision": probe.get("none_precision"),
            "none_false_positive_rate": probe.get("none_false_positive_rate"),
            "val_loss": probe.get("val_loss"),
            "best_epoch": probe.get("best_epoch"),
            "epochs_completed": probe.get("epochs_completed"),
            "elapsed_ms": int(round((time.monotonic() - candidate_start) * 1000)),
        }
        matrix.append(row)
        cb({
            "type": "training_progress",
            "stage": "soft_label_tuning",
            "status_text": (
                f"Tested {label} {value:g}: {score_metric} {score:.3f}, macro F1 {(row['macro_f1'] or 0.0):.3f}"
                + (f", QWK {row['qwk']:.3f}" if row.get("qwk") is not None else "")
            ),
            "step": idx,
            "total_steps": len(candidates),
            "softness_kind": kind,
            "softness_value": value,
            "val_macro_f1": row["macro_f1"],
            "val_qwk": row.get("qwk"),
            "val_none_f1": row.get("none_f1"),
            "val_none_recall": row.get("none_recall"),
            "selected_validation_score": score,
        })
        if _softness_candidate_better(row, best_row):
            best_row = row
            best_probe = probe
            best_head_state = copy.deepcopy(model.head.state_dict())

    config.ordinal_sigma = original_sigma
    config.label_smoothing = original_smoothing

    if best_row is None or best_probe is None or best_head_state is None:
        model.head.load_state_dict(original_head_state)
        return {"best_epoch": 0, "epochs_completed": 0}

    model.head.load_state_dict(best_head_state)
    _apply_softness(config, kind, float(best_row["value"]))
    config.selected_softness_kind = kind
    config.selected_softness_value = float(best_row["value"])
    config.soft_label_tuning_metric = score_metric
    config.soft_label_tuning_results = matrix
    config.soft_label_tuning_elapsed_ms = int(round((time.monotonic() - sweep_start) * 1000))
    cb({
        "type": "training_progress",
        "stage": "soft_label_tuning",
        "status_text": f"Selected {label} {best_row['value']:g} by {score_metric}",
        "step": len(candidates),
        "total_steps": len(candidates),
        "softness_kind": kind,
        "softness_value": best_row["value"],
        "best_val_macro_f1": best_row.get("macro_f1"),
        "best_val_qwk": best_row.get("qwk"),
        "best_val_none_f1": best_row.get("none_f1"),
        "best_val_none_recall": best_row.get("none_recall"),
        "selected_validation_score": best_row.get("score"),
        "soft_label_tuning_metric": score_metric,
    })
    return best_probe


def _auto_oversample_enabled(config: GroupTrainConfig, none_index: int) -> bool:
    """The auto ``__none__`` oversample sweep applies only to single-label
    groups that actually have a ``__none__`` class to oversample."""
    return bool(config.auto_oversample_none) and not config.multi_label and none_index >= 0


def _build_oversampled_tensors(
    x_train: torch.Tensor, y_train: torch.Tensor, none_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append extra ``__none__`` feature rows up to the rare-group target.

    Mirrors ``GroupDataset._apply_rare_group_oversample`` at the cached-feature
    level: the ``__none__`` images are already embedded, so reaching the 1.5x
    target only duplicates existing rows â€” no re-embed. Returns the inputs
    unchanged when there is nothing to oversample.
    """
    if none_index < 0 or y_train.numel() == 0:
        return x_train, y_train
    counts = torch.bincount(y_train, minlength=none_index + 1)
    none_count = int(counts[none_index].item())
    if none_count == 0:
        return x_train, y_train
    max_count = int(counts.max().item())
    non_none_class_count = int((counts > 0).sum().item()) - 1  # exclude __none__
    if non_none_class_count <= 0:
        return x_train, y_train
    target = rare_group_none_target(max_count, non_none_class_count)
    extra_needed = target - none_count
    if extra_needed <= 0:
        return x_train, y_train
    none_rows = x_train[y_train == none_index]
    reps = extra_needed // none_count + 1
    extra_x = none_rows.repeat(reps, *([1] * (none_rows.dim() - 1)))[:extra_needed]
    extra_y = y_train.new_full((extra_needed,), none_index)
    return torch.cat([x_train, extra_x], dim=0), torch.cat([y_train, extra_y], dim=0)


def _oversample_candidate_better(candidate: dict, incumbent: dict | None) -> bool:
    if incumbent is None:
        return True
    cand_score = float(candidate.get("score") or 0.0)
    inc_score = float(incumbent.get("score") or 0.0)
    if cand_score != inc_score:
        return cand_score > inc_score
    cand_loss = candidate.get("val_loss")
    inc_loss = incumbent.get("val_loss")
    if cand_loss is not None and inc_loss is not None and float(cand_loss) != float(inc_loss):
        return float(cand_loss) < float(inc_loss)
    # Tie: prefer *less* oversampling (the simpler, cheaper dataset).
    return bool(incumbent.get("oversample")) and not bool(candidate.get("oversample"))


def _run_auto_oversample_probe(
    model: nn.Module,
    config: GroupTrainConfig,
    embed_cache: EmbeddingCache,
    smart_cache: object | None,
    train_samples: list[dict],
    val_samples: list[dict],
    *,
    device: torch.device,
    none_index: int,
    cb: Callable[[dict], None],
    stop_event: object | None,
) -> dict:
    """Sweep ``__none__`` oversampling (off vs 1.5x) on cached features.

    Runs after the soft-label sweep, building on the soft-label-selected head.
    Trains a head probe for each candidate, selects the better by validation
    score, and writes the choice to ``config.oversample_none`` (+ tuning
    metadata). Returns the winning probe dict, or ``{}`` when the sweep is
    skipped (multi-label, no ``__none__`` class, or oversampling adds nothing).
    """
    if not _auto_oversample_enabled(config, none_index):
        return {}

    x_train, y_train, x_val, y_val = prepare_head_probe_tensors(
        embed_cache, smart_cache, train_samples, val_samples, config, cb=cb,
    )
    x_train_os, y_train_os = _build_oversampled_tensors(x_train, y_train, none_index)
    if x_train_os.shape[0] == x_train.shape[0]:
        # No extra __none__ rows were added: the two candidates are identical,
        # so a sweep is meaningless. Leave config.oversample_none untouched.
        return {}

    tensors_by_flag = {False: (x_train, y_train), True: (x_train_os, y_train_os)}
    original_head_state = copy.deepcopy(model.head.state_dict())
    original_rng_state = _capture_rng_state(device)
    original_oversample = config.oversample_none
    score_metric = _score_metric_label(config)
    total = len(_OVERSAMPLE_NONE_CANDIDATES)

    best_row: dict | None = None
    best_probe: dict | None = None
    best_head_state: dict | None = None
    matrix: list[dict] = []
    sweep_start = time.monotonic()

    for idx, (clabel, flag) in enumerate(_OVERSAMPLE_NONE_CANDIDATES, start=1):
        if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
            break
        model.head.load_state_dict(original_head_state)
        _restore_rng_state(original_rng_state, device)
        xt, yt = tensors_by_flag[flag]
        cb({
            "type": "training_progress",
            "stage": "oversample_tuning",
            "status_text": f"Testing __none__ oversample {clabel} ({idx}/{total})",
            "step": idx,
            "total_steps": total,
            "oversample_none": flag,
            "oversample_tuning_metric": score_metric,
        })
        candidate_start = time.monotonic()
        probe = train_head_probe_from_tensors(
            model, xt, yt, x_val, y_val, config,
            device=device, none_index=none_index, cb=cb,
            stop_event=stop_event,
            progress_stage="oversample_tuning",
            progress_prefix=f"__none__ oversample {clabel}",
            score_metric=score_metric,
        )
        score = _metric_score(probe, config)
        row = {
            "label": clabel,
            "oversample": bool(flag),
            "score": score,
            "macro_f1": probe.get("macro_f1"),
            "qwk": probe.get("qwk"),
            "none_f1": probe.get("none_f1"),
            "none_recall": probe.get("none_recall"),
            "none_precision": probe.get("none_precision"),
            "none_false_positive_rate": probe.get("none_false_positive_rate"),
            "val_loss": probe.get("val_loss"),
            "best_epoch": probe.get("best_epoch"),
            "epochs_completed": probe.get("epochs_completed"),
            "train_samples": int(xt.shape[0]),
            "elapsed_ms": int(round((time.monotonic() - candidate_start) * 1000)),
        }
        matrix.append(row)
        cb({
            "type": "training_progress",
            "stage": "oversample_tuning",
            "status_text": (
                f"Tested __none__ oversample {clabel}: {score_metric} {score:.3f}, "
                f"macro F1 {(row['macro_f1'] or 0.0):.3f}"
                + (f", none F1 {row['none_f1']:.3f}" if row.get("none_f1") is not None else "")
            ),
            "step": idx,
            "total_steps": total,
            "oversample_none": flag,
            "val_macro_f1": row["macro_f1"],
            "val_qwk": row.get("qwk"),
            "val_none_f1": row.get("none_f1"),
            "val_none_recall": row.get("none_recall"),
            "selected_validation_score": score,
        })
        if _oversample_candidate_better(row, best_row):
            best_row = row
            best_probe = probe
            best_head_state = copy.deepcopy(model.head.state_dict())

    config.oversample_none = original_oversample

    if best_row is None or best_probe is None or best_head_state is None:
        model.head.load_state_dict(original_head_state)
        return {}

    model.head.load_state_dict(best_head_state)
    config.oversample_none = bool(best_row["oversample"])
    config.selected_oversample_none = bool(best_row["oversample"])
    config.oversample_tuning_metric = score_metric
    config.oversample_tuning_results = matrix
    config.oversample_tuning_elapsed_ms = int(round((time.monotonic() - sweep_start) * 1000))
    cb({
        "type": "training_progress",
        "stage": "oversample_tuning",
        "status_text": f"Selected __none__ oversample {best_row['label']} by {score_metric}",
        "step": total,
        "total_steps": total,
        "oversample_none": best_row["oversample"],
        "best_val_macro_f1": best_row.get("macro_f1"),
        "best_val_qwk": best_row.get("qwk"),
        "best_val_none_f1": best_row.get("none_f1"),
        "best_val_none_recall": best_row.get("none_recall"),
        "selected_validation_score": best_row.get("score"),
        "oversample_tuning_metric": score_metric,
    })
    return best_probe
