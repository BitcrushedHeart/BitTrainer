"""Function: run_oft_training — OFTv2-style incremental orthogonal fine-tune.

The "Orthogonal / Fast Incremental" rung between head-only training and a full
fine-tune. It reuses the full-FT machinery (dataset prep, head warmup, the
shared train/eval loop, and the promote-if-better finaliser) but, instead of
unfreezing the whole backbone, it freezes the backbone and learns a small
block-diagonal *orthogonal* rotation per Linear layer (see :mod:`bittrainer.oft`).

Flow:

1. Prepare datasets + caches (shared with full FT).
2. **Head warmup** — converge the classifier head on cached backbone features.
   Head training is cheap, so it is kept as OFT's warmup phase: it removes the
   random-head feature-distortion risk before any backbone-affecting update, and
   reuses the embedding cache.
3. Wrap the backbone's Linear layers with OFT (base frozen, head + OFT
   generators trainable) and fine-tune.
4. On each validation improvement, save the candidate as a **merged full-weight**
   ``state_dict`` (OFT collapsed into ``W' = R @ W``), so the artefact is a plain
   ConvNeXt checkpoint indistinguishable in format from a full fine-tune.
5. Resolve through the same ``_compare_promote_finalize`` gate — a candidate is
   promoted only if it beats the incumbent on the identical validation set.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import torch
from adv_optm import Prodigy_adv
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from bittrainer.group_dataset import build_group_bucket_sampler
from bittrainer.group_trainer import (
    GroupTrainConfig,
    _collate_bucket_batch,
    _collate_multilabel_batch,
    _compare_promote_finalize,
    _create_or_warmstart_model,
    _emit_model_load_stage,
    _evaluate,
    _get_dtype,
    _metric_score,
    _prepare_datasets_and_cache,
    _primary_validation_metric,
    _resolve_class_balance,
    _resolve_none_index,
    _train_one_epoch,
    _warmup_head_probe,
    _SELECTION_MIN_DELTA,
    _effective_number_weights,
)
from bittrainer.oft import (
    VALID_BACKENDS,
    merged_state_dict,
    oft_parameters,
    wrap_backbone_with_oft,
)

logger = logging.getLogger(__name__)


def _make_oft_optimizer(params: list[torch.nn.Parameter]) -> Prodigy_adv:
    """Prodigy_adv over the OFT generators + head only (backbone is frozen).

    Mirrors ``_make_optimizer`` but flat (no LLRD): OFT generators do not map
    onto the ConvNeXt stage-depth buckets, and there are far fewer of them.
    """
    return Prodigy_adv(
        params, lr=1.0, d_coef=0.9,
        weight_decay=0.01, betas=(0.9, 0.999),
        kourkoutas_beta=True, k_warmup_steps=50,
        cautious_wd=True,
    )


def run_oft_training(
    config: GroupTrainConfig,
    *,
    progress_callback: Callable[[dict], None] | None = None,
    stop_event: object | None = None,
    stop_now_event: object | None = None,
) -> dict:
    """Run an OFTv2-style orthogonal fine-tune. Terminal; returns the result dict."""
    from bittrainer.progress import ProgressEmitter, Stage
    from bittrainer.runtime import configure_cuda_backend
    from bittrainer.smart_cache import _noop_callback

    if config.oft_backend not in VALID_BACKENDS:
        raise ValueError(
            f"Unknown oft_backend '{config.oft_backend}'. Valid: {VALID_BACKENDS}"
        )

    em = ProgressEmitter(progress_callback or config.progress_callback or _noop_callback)
    cb = em.raw
    device = torch.device(config.device)
    dtype = _get_dtype(config.dtype)
    configure_cuda_backend()
    group_folder = Path(config.group_folder)
    checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else group_folder / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    use_soft = config.ordinal or bool(config.soft_aliases) or (
        not config.multi_label and config.label_smoothing > 0
    )

    em.stage(Stage.scanning, "Scanning dataset")
    train_ds, val_ds, smart_cache, bucket_counts = _prepare_datasets_and_cache(
        config, cb=cb, stop_event=stop_event,
    )

    head_hidden_size = config.probe_mlp_hidden if config.probe_head == "mlp" else None
    _emit_model_load_stage(em, config, checkpoint_dir)
    model = _create_or_warmstart_model(
        config, device=device, dtype=dtype,
        head_hidden_size=head_hidden_size, checkpoint_dir=checkpoint_dir,
    )

    # --- OFT warmup phase: converge the head on cached features (cheap). ---
    _warmup_head_probe(
        model, config, train_ds, val_ds, smart_cache,
        device=device, dtype=dtype, cb=cb,
        stop_event=stop_event, stop_now_event=stop_now_event,
    )
    if (stop_event is not None and stop_event.is_set()) or (
        stop_now_event is not None and stop_now_event.is_set()
    ):
        cb({"type": "training_cancelled", "stage": "warmup",
            "status_text": "Cancelled during OFT head warmup"})
        return {"cancelled": True, "mode": "oft"}

    # --- Wrap backbone with OFT (base frozen, head + OFT generators trainable). ---
    cb({"type": "training_progress", "stage": "preparing",
        "status_text": f"Applying orthogonal adapters ({config.oft_backend})"})
    n_wrapped = wrap_backbone_with_oft(
        model,
        blocks=config.oft_blocks,
        backend=config.oft_backend,
        clipped_norm=config.oft_clipped_norm,
        neumann_terms=config.oft_neumann_terms,
        cans_iters=config.oft_cans_iters,
        dora=config.oft_dora,
    )
    trainable = oft_parameters(model)
    n_params = sum(p.numel() for p in trainable)
    cb({"type": "training_progress", "stage": "preparing",
        "status_text": (
            f"OFT: {n_wrapped} layers, {n_params / 1e6:.2f}M trainable params "
            f"(backend {config.oft_backend}, clip {config.oft_clipped_norm})"
        )})

    eff_bs = int(config.batch_size) if config.batch_size else 32
    cb({"type": "autobatch", "batch_size": eff_bs,
        "manual_override": config.batch_size is not None})

    class_counts = train_ds.get_class_counts()
    total_raw = sum(class_counts.values())

    balance_mode = _resolve_class_balance(config, class_counts)
    class_weights: torch.Tensor | None = None
    if not config.multi_label and balance_mode == "reweight":
        train_ds.set_natural_sampling(True)
        class_weights = _effective_number_weights(
            class_counts, config.num_classes, config.class_balance_beta, device,
        )

    optimizer = _make_oft_optimizer(trainable)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    best_val_macro_f1 = -1.0
    best_val_qwk = -1.0
    best_validation_score = -1.0
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path = None
    best_metrics: dict = {}
    epoch = 0

    for epoch in range(config.max_epochs):
        if stop_now_event is not None and stop_now_event.is_set():
            cb({"type": "stop_now", "epoch": epoch, "max_epochs": config.max_epochs})
            break
        if stop_event is not None and stop_event.is_set():
            cb({"type": "graceful_stop", "epoch": epoch, "max_epochs": config.max_epochs})
            break

        train_ds.reshuffle()
        collate_fn = _collate_multilabel_batch if config.multi_label else _collate_bucket_batch
        train_loader = DataLoader(
            train_ds, batch_sampler=build_group_bucket_sampler(train_ds, batch_size=eff_bs),
            collate_fn=collate_fn, num_workers=6, pin_memory=True,
            persistent_workers=True, prefetch_factor=4,
        )
        val_loader = DataLoader(
            val_ds, batch_sampler=build_group_bucket_sampler(val_ds, batch_size=eff_bs),
            collate_fn=collate_fn, num_workers=6, pin_memory=True,
            persistent_workers=True, prefetch_factor=4,
        )

        def _on_step(step: int, total_steps: int, avg_loss: float) -> None:
            cb({
                "type": "training_progress", "stage": "training",
                "status_text": f"OFT training (epoch {epoch + 1}/{config.max_epochs}, step {step}/{total_steps})",
                "epoch": epoch + 1, "max_epochs": config.max_epochs,
                "step": step, "total_steps": total_steps,
                "batch_size": eff_bs, "train_loss": round(avg_loss, 4),
                "validation_metric": _primary_validation_metric(config),
                "best_validation_score": best_validation_score if best_validation_score >= 0 else None,
            })

        train_loss = _train_one_epoch(
            model, train_loader, optimizer, config, device, dtype,
            use_soft_targets=use_soft, step_callback=_on_step,
            stop_now_event=stop_now_event, ema=None,
            class_weights=class_weights, mixup_enabled=False,
        )
        scheduler.step()

        em.stage(Stage.validating, f"Validating (epoch {epoch + 1}/{config.max_epochs})",
                 epoch=epoch + 1, max_epochs=config.max_epochs)
        model.eval()
        val_metrics = _evaluate(
            model, val_loader, config.num_classes, device, dtype,
            multi_label=config.multi_label, ordinal=config.ordinal,
            none_index=_resolve_none_index(config.class_names),
            channels_last=config.channels_last,
        )
        val_metrics["train_loss"] = train_loss
        val_macro_f1 = val_metrics["macro_f1"]
        val_qwk = val_metrics.get("qwk", 0.0)
        selected_score = _metric_score(val_metrics, config)

        improved = selected_score > best_validation_score + _SELECTION_MIN_DELTA
        if improved:
            best_val_macro_f1 = val_macro_f1
            best_val_qwk = val_qwk
            best_validation_score = selected_score
            best_epoch = epoch
            patience_counter = 0
            best_metrics = val_metrics.copy()

            # Persist the MERGED full-weight state_dict, not the OFT adapter, so
            # the candidate is a plain ConvNeXt checkpoint (format-identical to a
            # full fine-tune) and competes/loads identically.
            ckpt_meta = {
                "state_dict": merged_state_dict(model),
                "num_classes": config.num_classes,
                "model_size": config.backbone_variant,
                "class_names": list(config.class_names),
                "validation_metric": _primary_validation_metric(config),
                "training_mode": "oft",
                "oft_backend": config.oft_backend,
            }
            if head_hidden_size is not None:
                ckpt_meta["head_hidden_size"] = head_hidden_size
            if config.multi_label:
                ckpt_meta["multi_label"] = True
            ckpt_path = checkpoint_dir / "candidate.pt"
            torch.save(ckpt_meta, ckpt_path)
            best_checkpoint_path = str(ckpt_path)
        else:
            patience_counter += 1

        cb({
            "type": "epoch_complete", "stage": "training",
            "status_text": f"OFT epoch {epoch + 1}/{config.max_epochs} (val macro F1 {val_macro_f1:.3f})",
            "epoch": epoch + 1, "max_epochs": config.max_epochs,
            "train_loss": train_loss, "val_loss": val_metrics["val_loss"],
            "val_macro_f1": val_macro_f1,
            "per_class_f1": val_metrics.get("per_class_f1", {}),
            "val_none_precision": val_metrics.get("none_precision"),
            "val_none_recall": val_metrics.get("none_recall"),
            "val_none_f1": val_metrics.get("none_f1"),
            "best_val_macro_f1": best_val_macro_f1,
            "selected_validation_score": selected_score,
            "best_validation_score": best_validation_score,
            "validation_metric": _primary_validation_metric(config),
            "best_epoch": best_epoch + 1,
            **({"val_qwk": val_qwk, "best_val_qwk": best_val_qwk} if config.ordinal else {}),
        })

        if patience_counter >= config.patience:
            logger.info("OFT early stopping at epoch %d (patience=%d)", epoch + 1, config.patience)
            break

    result = _compare_promote_finalize(
        config,
        candidate_path=best_checkpoint_path,
        best_metrics=best_metrics,
        candidate_macro_f1=best_val_macro_f1,
        candidate_qwk=best_val_qwk,
        best_epoch_display=best_epoch + 1,
        epochs_completed=epoch + 1,
        val_loader=val_loader,
        device=device, dtype=dtype,
        checkpoint_dir=checkpoint_dir,
        class_counts=train_ds.get_class_counts(),
        total_raw=total_raw,
        cb=cb,
    )
    result["mode"] = "oft"
    result["oft_backend"] = config.oft_backend
    result["oft_layers_wrapped"] = n_wrapped
    return result
