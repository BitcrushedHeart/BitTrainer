"""Function 1: train_head_only â€” train the classifier head on cached features.

Trains the head to convergence on cached backbone features (the backbone itself is
never touched), then resolves the result through the same promote-if-better path as
full fine-tune: a head-only model that beats the current one becomes the group's
deployed model; a worse one leaves the incumbent in place. Cheap by design â€” the
embedding cache is built once per backbone era, so re-running (or running after a
full fine-tune adapts the backbone) reuses or rebuilds vectors automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from bittrainer.embedding_cache import EmbeddingCache
from bittrainer.group_dataset import build_group_bucket_sampler
from bittrainer.group_trainer import (
    GroupTrainConfig,
    _collate_bucket_batch,
    _collate_multilabel_batch,
    _compare_promote_finalize,
    _create_or_warmstart_model,
    _evaluate,
    _get_dtype,
    _metric_score,
    _prepare_datasets_and_cache,
    _primary_validation_metric,
    _resolve_none_index,
    _run_auto_oversample_probe,
    _run_auto_softness_probe,
    _spatial_ckpt_meta,
)
from bittrainer.model import backbone_feature_hash

logger = logging.getLogger(__name__)

_THIN_CLASS_THRESHOLD = 20


def _thin_class_warnings(class_counts: dict[int, int], class_names: list[str]) -> list[dict]:
    warnings: list[dict] = []
    for idx, count in class_counts.items():
        if 0 < count < _THIN_CLASS_THRESHOLD:
            name = class_names[idx] if 0 <= idx < len(class_names) else str(idx)
            warnings.append({"class_index": idx, "class_name": name, "count": count})
    return warnings


def run_head_only_training(
    config: GroupTrainConfig,
    *,
    progress_callback: Callable[[dict], None] | None = None,
    stop_event: Any | None = None,
    stop_now_event: Any | None = None,
    pause_event: Any | None = None,
) -> dict:
    """Train a cached-feature head probe and report per-class scores. Terminal.

    ``pause_event`` (Bitcrush ISSUE-0405) is accepted for signature uniformity
    with the full trainers; head-only training has no backup/resume (out of
    scope), so a set pause_event simply behaves like ``stop_now`` — the probe
    finishes early and returns its partial result.
    """
    from bittrainer.progress import ProgressEmitter
    from bittrainer.runtime import configure_cuda_backend
    from bittrainer.smart_cache import _noop_callback

    em = ProgressEmitter(progress_callback or config.progress_callback or _noop_callback)
    cb = em.raw
    device = torch.device(config.device)
    dtype = _get_dtype(config.dtype)
    configure_cuda_backend()
    group_folder = Path(config.group_folder)
    checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else group_folder / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _stop() -> bool:
        return bool(
            (stop_event is not None and stop_event.is_set())
            or (stop_now_event is not None and stop_now_event.is_set())
            or (pause_event is not None and pause_event.is_set())
        )

    train_ds, val_ds, smart_cache, _bucket_counts = _prepare_datasets_and_cache(
        config, cb=cb, stop_event=stop_event,
    )

    head_hidden = config.probe_mlp_hidden if config.probe_head == "mlp" else None
    cb({"type": "training_progress", "stage": "preparing", "status_text": "Loading model"})
    model = _create_or_warmstart_model(
        config, device=device, dtype=dtype,
        head_hidden_size=head_hidden, checkpoint_dir=checkpoint_dir,
    )

    backbone_hash = backbone_feature_hash(model)
    pooled_dim = int(getattr(model, "num_features", 0))
    embed_root = config.embedding_cache_dir or str(group_folder / ".embedding_cache")
    embed_cache = EmbeddingCache(embed_root, backbone_hash, pooled_dim)

    all_samples = train_ds.samples + val_ds.samples

    def _build_progress(done: int, total: int) -> None:
        cb({
            "type": "training_progress", "stage": "embedding_build",
            "status_text": f"Building feature cache ({done}/{total})",
            "step": done, "total_steps": total,
        })

    cb({"type": "training_progress", "stage": "embedding_build",
        "status_text": f"Caching backbone features (era {backbone_hash})"})
    stats = embed_cache.ensure(
        all_samples, model, smart_cache, device=device, dtype=dtype,
        batch_size=config.batch_size or 64, progress_cb=_build_progress, stop_check=_stop,
    )
    logger.info("EmbeddingCache: %s", stats)

    if _stop():
        cb({"type": "training_cancelled", "stage": "caching",
            "status_text": "Cancelled before probe"})
        return {"cancelled": True, "mode": "head_only"}

    # Mandatory: fail loud on a stale/bad cache before any probe consumes it.
    checked = embed_cache.verify(all_samples, model, smart_cache, device=device, dtype=dtype)
    logger.info("EmbeddingCache.verify: %d vectors matched live forward", checked)

    class_counts = train_ds.get_class_counts()
    total_raw = sum(class_counts.values())
    thin = _thin_class_warnings(class_counts, config.class_names)
    if thin:
        names = ", ".join(f"{w['class_name']} ({w['count']})" for w in thin)
        cb({"type": "training_progress", "stage": "training",
            "status_text": f"Thin classes (probe may overfit): {names}"})

    cb({"type": "training_progress", "stage": "training",
        "status_text": f"Training head probe ({config.probe_head})"})
    none_index = _resolve_none_index(config.class_names)
    probe = _run_auto_softness_probe(
        model, config, embed_cache, smart_cache,
        train_ds.samples, val_ds.samples,
        device=device, none_index=none_index,
        cb=cb, stop_event=stop_event,
    )
    # Second pre-training sweep: __none__ oversample off vs 1.5x. When it runs
    # it leaves the head in the selected state and supersedes the softness probe
    # as the candidate to evaluate.
    oversample_probe = _run_auto_oversample_probe(
        model, config, embed_cache, smart_cache,
        train_ds.samples, val_ds.samples,
        device=device, none_index=none_index,
        cb=cb, stop_event=stop_event,
    )
    if oversample_probe:
        probe = oversample_probe

    # Evaluate the trained candidate (frozen backbone + warm head) on the val
    # images, so it competes with the incumbent on identical _evaluate footing,
    # then run the same promote-if-better path as full fine-tune. A one-shot val
    # pass (num_workers=0 â€” small, single-process, no spawn).
    collate_fn = _collate_multilabel_batch if config.multi_label else _collate_bucket_batch
    val_bs = config.batch_size or 32
    val_loader = DataLoader(
        val_ds,
        batch_sampler=build_group_bucket_sampler(val_ds, batch_size=val_bs),
        collate_fn=collate_fn, num_workers=0,
    )
    model.eval()
    candidate_metrics = _evaluate(
        model, val_loader, config.num_classes, device, dtype,
        multi_label=config.multi_label, ordinal=config.ordinal,
        none_index=_resolve_none_index(config.class_names),
    )
    candidate_macro_f1 = candidate_metrics.get("macro_f1", 0.0)
    candidate_qwk = candidate_metrics.get("qwk", -1.0)

    candidate_path = checkpoint_dir / "candidate.pt"
    ckpt_meta: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "num_classes": config.num_classes,
        "model_size": config.backbone_variant,
        "class_names": list(config.class_names),
        "validation_metric": _primary_validation_metric(config),
        **_spatial_ckpt_meta(config),
    }
    if head_hidden is not None:
        ckpt_meta["head_hidden_size"] = head_hidden
    if config.multi_label:
        ckpt_meta["multi_label"] = True
    torch.save(ckpt_meta, candidate_path)

    result = _compare_promote_finalize(
        config,
        candidate_path=str(candidate_path),
        best_metrics=candidate_metrics,
        candidate_macro_f1=candidate_macro_f1,
        candidate_qwk=candidate_qwk,
        best_epoch_display=probe["best_epoch"],
        epochs_completed=probe["epochs_completed"],
        val_loader=val_loader,
        device=device, dtype=dtype,
        checkpoint_dir=checkpoint_dir,
        class_counts=class_counts,
        effective_class_counts=train_ds.get_effective_class_counts(),
        total_raw=total_raw,
        cb=cb,
    )
    result["mode"] = "head_only"
    result["probe_head"] = config.probe_head
    result["thin_class_warnings"] = thin
    result["backbone_hash"] = backbone_hash
    result["embedding_cache_stats"] = stats

    promoted = result.get("promotion_reason") not in (None, "incumbent_wins")
    primary_metric = _primary_validation_metric(config)
    final_score = result.get("selected_validation_score")
    if final_score is None:
        final_score = _metric_score({
            "macro_f1": result.get("best_val_macro_f1"),
            "qwk": result.get("best_val_qwk"),
            "none_f1": result.get("final_val_none_f1"),
        }, config)
    cb({
        "type": "epoch_complete", "stage": "training",
        "status_text": (
            f"Head training complete (val {primary_metric} {final_score:.3f}) â€” "
            f"{'deployed' if promoted else 'kept existing model'}"
        ),
        "epoch": probe["epochs_completed"], "max_epochs": config.head_max_epochs,
        "val_macro_f1": result.get("best_val_macro_f1"),
        "val_qwk": result.get("qwk"),
        "validation_metric": primary_metric,
        "per_class_f1": result.get("per_class_f1", {}),
        "best_val_macro_f1": result.get("best_val_macro_f1"),
        "best_val_qwk": result.get("best_val_qwk"),
        "best_epoch": probe["best_epoch"],
    })
    return result


