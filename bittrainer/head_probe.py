"""Shared cached-feature head probe.

Trains the model's head tail (``head.pre_logits`` + ``head.fc``) on cached pooled
features as a pure linear (or MLP) probe — to convergence, early-stopped on the
validation metric. The backbone is never touched. Used by both
``run_head_only_training`` (terminal scouting) and ``run_group_training``
(warmup before the full fine-tune), so the two cannot diverge.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from bittrainer.group_validation import (
    compute_multiclass_metrics,
    compute_multilabel_metrics,
    compute_none_metrics,
    compute_ordinal_metrics,
    find_per_class_thresholds,
)
from bittrainer.model import head_tail_logits, head_tail_parameters

logger = logging.getLogger(__name__)

_PROBE_BATCH = 256
_PROBE_LR = 1e-3


def _gather(
    samples: list[dict],
    embed_cache: Any,
    smart_cache: Any | None,
    *,
    multi_label: bool,
    progress_cb: Callable[[int], None] | None = None,
    progress_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    feats: list[torch.Tensor] = []
    labels: list[Any] = []
    for i, s in enumerate(samples):
        if progress_cb is not None:
            progress_cb(progress_offset + i + 1)
        v = embed_cache.get_vector(s["path"], smart_cache)
        if v is None:
            continue
        feats.append(torch.from_numpy(v.astype(np.float32)))
        lbl = s["label"]
        if multi_label:
            labels.append(lbl if torch.is_tensor(lbl) else torch.as_tensor(lbl, dtype=torch.float32))
        else:
            labels.append(int(lbl))
    if not feats:
        raise RuntimeError("Head probe: no cached embeddings available for these samples")
    x = torch.stack(feats)
    y = torch.stack(labels).float() if multi_label else torch.tensor(labels, dtype=torch.long)
    return x, y


def _evaluate_probe(
    model: torch.nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    config: Any,
    device: torch.device,
    none_index: int,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    score_metric: str = "primary",
) -> tuple[dict, float]:
    with torch.no_grad():
        logits = head_tail_logits(model, x_val.to(device)).float()
        val_loss = float(loss_fn(logits, y_val.to(device)).item())
    logits = logits.cpu()

    if config.multi_label:
        probs = torch.sigmoid(logits).numpy()
        labels = y_val.int().numpy()
        if config.per_class_thresholds_enabled:
            thresholds = find_per_class_thresholds(probs, labels)
        else:
            thresholds = np.full(config.num_classes, 0.5, dtype=np.float64)
        metrics = compute_multilabel_metrics(
            labels, predictions=None, num_classes=config.num_classes,
            thresholds=thresholds, probs=probs,
        )
        score = metrics["macro_f1"]
    else:
        preds = logits.argmax(dim=1).numpy()
        labels = y_val.numpy()
        metrics = compute_multiclass_metrics(list(labels), list(preds), config.num_classes)
        if none_index >= 0:
            metrics.update(compute_none_metrics(
                list(labels), list(preds), config.num_classes, none_index=none_index,
            ))
        if config.ordinal:
            metrics.update(compute_ordinal_metrics(
                list(labels), list(preds), config.num_classes, none_index=none_index,
            ))
            if score_metric == "guarded_qwk":
                score = metrics["qwk"] + 0.10 * (metrics.get("none_f1") or 0.0)
            else:
                score = metrics["macro_f1"] if score_metric == "macro_f1" else metrics["qwk"]
        else:
            score = metrics["macro_f1"] + 0.10 * (metrics.get("none_f1") or 0.0) if score_metric == "guarded_macro_f1" else metrics["macro_f1"]

    metrics["val_loss"] = val_loss
    return metrics, score


def prepare_head_probe_tensors(
    embed_cache: Any,
    smart_cache: Any | None,
    train_samples: list[dict],
    val_samples: list[dict],
    config: Any,
    *,
    cb: Callable[[dict], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load cached feature tensors once for one or more probe runs."""
    _cb = cb or (lambda _msg: None)
    total_vectors = len(train_samples) + len(val_samples)
    _last_load_report = [0.0]

    def _load_progress(done: int) -> None:
        now = time.monotonic()
        if done < total_vectors and now - _last_load_report[0] < 0.25:
            return
        _last_load_report[0] = now
        _cb({
            "type": "training_progress", "stage": "embedding_build",
            "status_text": f"Loading cached features ({done}/{total_vectors})",
            "step": done, "total_steps": total_vectors,
        })

    x_train, y_train = _gather(
        train_samples, embed_cache, smart_cache, multi_label=config.multi_label,
        progress_cb=_load_progress,
    )
    x_val, y_val = _gather(
        val_samples, embed_cache, smart_cache, multi_label=config.multi_label,
        progress_cb=_load_progress, progress_offset=len(train_samples),
    )
    return x_train, y_train, x_val, y_val


def train_head_probe_from_tensors(
    model: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    config: Any,
    *,
    device: torch.device,
    none_index: int,
    cb: Callable[[dict], None] | None = None,
    stop_event: Any | None = None,
    progress_stage: str = "training",
    progress_prefix: str = "Head probe",
    score_metric: str = "primary",
) -> dict:
    """Train the head tail on preloaded cached features."""
    from bittrainer.group_trainer import build_group_loss_fn

    _cb = cb or (lambda _msg: None)
    use_soft = (
        config.ordinal
        or bool(config.soft_aliases)
        or bool(config.class_similarity_centroids)
        or (not config.multi_label and config.label_smoothing > 0)
    )

    for p in model.parameters():
        p.requires_grad_(False)
    tail_dtype = model.head.fc.weight.dtype
    model.head.fc.float()
    model.head.pre_logits.float()
    tail_params = head_tail_parameters(model)
    for p in tail_params:
        p.requires_grad_(True)
    model.eval()

    loss_fn = build_group_loss_fn(
        config, use_soft_targets=use_soft, none_index=none_index, device=device,
    )
    optimizer = torch.optim.AdamW(tail_params, lr=_PROBE_LR, weight_decay=config.head_weight_decay)

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=min(_PROBE_BATCH, len(x_train)), shuffle=True, drop_last=False,
    )

    best_score = -1.0
    best_epoch = 0
    best_metrics: dict = {}
    best_head_state = copy.deepcopy(model.head.state_dict())
    patience = 0
    epoch = -1

    for epoch in range(config.head_max_epochs):
        if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
            break

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = head_tail_logits(model, xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

        metrics, score = _evaluate_probe(
            model, x_val, y_val, config, device, none_index, loss_fn,
            score_metric=score_metric,
        )
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics
            best_head_state = copy.deepcopy(model.head.state_dict())
            patience = 0
        else:
            patience += 1

        metric_label = "macroF1" if score_metric == "macro_f1" else ("qwk" if config.ordinal else "macroF1")
        probe_msg = {
            "type": "training_progress",
            "stage": progress_stage,
            "status_text": (
                f"{progress_prefix} epoch {epoch + 1}/{config.head_max_epochs} "
                f"(val {metric_label} {score:.3f})"
            ),
            "epoch": epoch + 1,
            "max_epochs": config.head_max_epochs,
            "val_macro_f1": metrics.get("macro_f1"),
            "per_class_f1": metrics.get("per_class_f1", {}),
            "best_val_macro_f1": best_metrics.get("macro_f1"),
            "best_epoch": best_epoch + 1,
        }
        if config.ordinal:
            probe_msg["val_qwk"] = metrics.get("qwk")
            probe_msg["best_val_qwk"] = best_metrics.get("qwk")
        _cb(probe_msg)

        if patience >= config.head_patience:
            logger.info("Head probe early-stopping at epoch %d (patience=%d)", epoch + 1, config.head_patience)
            break

    model.head.load_state_dict(best_head_state)
    model.head.to(tail_dtype)

    result = {
        "macro_f1": best_metrics.get("macro_f1", 0.0),
        "macro_precision": best_metrics.get("macro_precision", 0.0),
        "macro_recall": best_metrics.get("macro_recall", 0.0),
        "per_class_f1": best_metrics.get("per_class_f1", {}),
        "per_class_precision": best_metrics.get("per_class_precision", {}),
        "per_class_recall": best_metrics.get("per_class_recall", {}),
        "none_precision": best_metrics.get("none_precision"),
        "none_recall": best_metrics.get("none_recall"),
        "none_f1": best_metrics.get("none_f1"),
        "none_false_positive_rate": best_metrics.get("none_false_positive_rate"),
        "none_support": best_metrics.get("none_support"),
        "val_loss": best_metrics.get("val_loss"),
        "best_epoch": best_epoch + 1,
        "epochs_completed": epoch + 1,
        "thresholds": best_metrics.get("thresholds") if config.multi_label else None,
    }
    if config.ordinal:
        result["qwk"] = best_metrics.get("qwk")
        result["ordinal_mae"] = best_metrics.get("ordinal_mae")
        result["adjacent_accuracy"] = best_metrics.get("adjacent_accuracy")
    if config.multi_label:
        result["hamming_loss"] = best_metrics.get("hamming_loss")
        result["exact_match_ratio"] = best_metrics.get("exact_match_ratio")
    else:
        result["confusion_matrix"] = best_metrics.get("confusion_matrix", [])
        result["balanced_accuracy"] = best_metrics.get("balanced_accuracy")
    return result


def train_head_probe(
    model: torch.nn.Module,
    embed_cache: Any,
    smart_cache: Any | None,
    train_samples: list[dict],
    val_samples: list[dict],
    config: Any,
    *,
    device: torch.device,
    none_index: int,
    cb: Callable[[dict], None] | None = None,
    stop_event: Any | None = None,
) -> dict:
    """Train the head tail on cached features; leave *model* holding the best weights.

    Returns a metrics dict (macro_f1, per-class scores, ordinal/multi-label
    extras, tuned thresholds, best_epoch, epochs_completed, val_loss).
    """
    from bittrainer.group_trainer import build_group_loss_fn

    _cb = cb or (lambda _msg: None)
    use_soft = (
        config.ordinal
        or bool(config.soft_aliases)
        or bool(config.class_similarity_centroids)
        or (not config.multi_label and config.label_smoothing > 0)
    )

    # Loading tens of thousands of small .npy vectors takes minutes on NTFS —
    # without per-step frames this was the longest silent phase of a run.
    total_vectors = len(train_samples) + len(val_samples)
    _last_load_report = [0.0]

    def _load_progress(done: int) -> None:
        now = time.monotonic()
        if done < total_vectors and now - _last_load_report[0] < 0.25:
            return
        _last_load_report[0] = now
        _cb({
            "type": "training_progress", "stage": "embedding_build",
            "status_text": f"Loading cached features ({done}/{total_vectors})",
            "step": done, "total_steps": total_vectors,
        })

    x_train, y_train = _gather(
        train_samples, embed_cache, smart_cache, multi_label=config.multi_label,
        progress_cb=_load_progress,
    )
    x_val, y_val = _gather(
        val_samples, embed_cache, smart_cache, multi_label=config.multi_label,
        progress_cb=_load_progress, progress_offset=len(train_samples),
    )

    # Freeze everything except the head tail the probe trains. The probe runs in
    # float32 (the cached features are float32, and a tiny head trains fine/stably
    # in full precision), so the trainable tail is cast to float32 regardless of
    # the model's compute dtype (bf16/fp16) — otherwise head.fc(float32 input)
    # hits "mat1 and mat2 must have same dtype". The model dtype is restored after
    # training so the full model stays uniform for eval/checkpoint.
    for p in model.parameters():
        p.requires_grad_(False)
    tail_dtype = model.head.fc.weight.dtype
    model.head.fc.float()
    model.head.pre_logits.float()
    tail_params = head_tail_parameters(model)
    for p in tail_params:
        p.requires_grad_(True)
    model.eval()  # head dropout defaults to p=0 -> deterministic; backbone unused

    loss_fn = build_group_loss_fn(
        config, use_soft_targets=use_soft, none_index=none_index, device=device,
    )
    # Probe loss runs in fp32 on cached features (tiny compute, no autocast).
    optimizer = torch.optim.AdamW(tail_params, lr=_PROBE_LR, weight_decay=config.head_weight_decay)

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=min(_PROBE_BATCH, len(x_train)), shuffle=True, drop_last=False,
    )

    best_score = -1.0
    best_epoch = 0
    best_metrics: dict = {}
    best_head_state = copy.deepcopy(model.head.state_dict())
    patience = 0

    for epoch in range(config.head_max_epochs):
        if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
            break

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = head_tail_logits(model, xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

        metrics, score = _evaluate_probe(model, x_val, y_val, config, device, none_index, loss_fn)
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics
            best_head_state = copy.deepcopy(model.head.state_dict())
            patience = 0
        else:
            patience += 1

        probe_msg = {
            "type": "training_progress",
            "stage": "training",
            "status_text": (
                f"Head probe epoch {epoch + 1}/{config.head_max_epochs} "
                f"(val {'qwk' if config.ordinal else 'macroF1'} {score:.3f})"
            ),
            "epoch": epoch + 1,
            "max_epochs": config.head_max_epochs,
            "val_macro_f1": metrics.get("macro_f1"),
            "per_class_f1": metrics.get("per_class_f1", {}),
            "best_val_macro_f1": best_metrics.get("macro_f1"),
            "best_epoch": best_epoch + 1,
        }
        if config.ordinal:
            probe_msg["val_qwk"] = metrics.get("qwk")
            probe_msg["best_val_qwk"] = best_metrics.get("qwk")
        _cb(probe_msg)

        if patience >= config.head_patience:
            logger.info("Head probe early-stopping at epoch %d (patience=%d)", epoch + 1, config.head_patience)
            break

    # Restore the best head weights, then cast the head back to the model's
    # compute dtype so the full model is uniform again for eval/checkpoint.
    model.head.load_state_dict(best_head_state)
    model.head.to(tail_dtype)

    result = {
        "macro_f1": best_metrics.get("macro_f1", 0.0),
        "macro_precision": best_metrics.get("macro_precision", 0.0),
        "macro_recall": best_metrics.get("macro_recall", 0.0),
        "per_class_f1": best_metrics.get("per_class_f1", {}),
        "per_class_precision": best_metrics.get("per_class_precision", {}),
        "per_class_recall": best_metrics.get("per_class_recall", {}),
        "none_precision": best_metrics.get("none_precision"),
        "none_recall": best_metrics.get("none_recall"),
        "none_f1": best_metrics.get("none_f1"),
        "none_false_positive_rate": best_metrics.get("none_false_positive_rate"),
        "none_support": best_metrics.get("none_support"),
        "val_loss": best_metrics.get("val_loss"),
        "best_epoch": best_epoch + 1,
        "epochs_completed": epoch + 1,
        "thresholds": best_metrics.get("thresholds") if config.multi_label else None,
    }
    if config.ordinal:
        result["qwk"] = best_metrics.get("qwk")
        result["ordinal_mae"] = best_metrics.get("ordinal_mae")
        result["adjacent_accuracy"] = best_metrics.get("adjacent_accuracy")
    if config.multi_label:
        result["hamming_loss"] = best_metrics.get("hamming_loss")
        result["exact_match_ratio"] = best_metrics.get("exact_match_ratio")
    else:
        result["confusion_matrix"] = best_metrics.get("confusion_matrix", [])
        result["balanced_accuracy"] = best_metrics.get("balanced_accuracy")
    return result
