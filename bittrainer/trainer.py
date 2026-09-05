"""Training loop for ConvNeXt V2 binary classifiers."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from adv_optm import Prodigy_adv
from torch.utils.data import DataLoader

from bittrainer.dataset import MAX_BINARY_NEG_POS_RATIO, ConceptDataset
from bittrainer.backbone_init import apply_backbone_init, wants_timm_pretrained
from bittrainer.ema import ModelEMA
from bittrainer.generic.optimizer import make_optimizer
from bittrainer.model import create_model
from bittrainer.validation import (
    compute_metrics,
    find_optimal_threshold,
    negative_kind_metrics,
)

logger = logging.getLogger(__name__)


def _stop_event_is_set(event) -> bool:
    """Picklable stop-check. Module-level so the SmartCache holding it can
    survive pickling when datasets ship to DataLoader workers on Windows spawn."""
    return event is not None and event.is_set()


_NUM_STAGES = 4  # ConvNeXt V2 has 4 stages


@dataclass
class TrainConfig:
    concept_folder: str
    max_epochs: int = 50
    patience: int = 3
    # Adaptive implied-negative ceiling. Binary datasets scale from 3:1 for
    # tiny concepts to this 5:1 default at 50 positives; explicit/hard
    # negatives are additive and keep their repetition weight. Lowered from
    # 10:1 in ISSUE-0773 — see MAX_BINARY_NEG_POS_RATIO for the measurement.
    neg_pos_ratio: float = MAX_BINARY_NEG_POS_RATIO
    # Per-concept training resolution: scales the aspect-bucket table
    # (512 = the canonical ~512px buckets; see bittrainer.dataset.scaled_buckets).
    # SmartCache keys embed bucket dims, so a change simply builds fresh entries.
    train_resolution: int = 512
    model_size: str = "nano"
    # ``head_only`` dispatches to the cached-feature binary probe.
    # ``frozen_backbone`` retains the legacy image loop with a frozen trunk.
    # ``full`` preserves the historical epoch-0 warmup followed by unfreezing.
    training_mode: str = "full"
    device: str = "cuda"
    dtype: str = "bfloat16"
    from_scratch: bool = False
    # Bitcrush Engine backbone spec (see bittrainer.backbone_init) — governs
    # where fresh-model backbone weights come from. None = timm pretrained.
    backbone_init: dict | None = None
    extra_positive_dirs: list[str] = field(default_factory=list)
    negative_dirs: list[str] = field(default_factory=list)
    # Explicit per-file split contract. ``None`` preserves legacy directory
    # scanning; an explicit empty list means the split intentionally has no
    # files. Engine uses these for flat mapped datasets.
    train_positive_paths: list[str] | None = None
    val_positive_paths: list[str] | None = None
    train_negative_paths: list[str] | None = None
    val_negative_paths: list[str] | None = None
    train_hard_negative_paths: list[str] | None = None
    val_hard_negative_paths: list[str] | None = None
    hard_negative_paths: list[str] = field(default_factory=list)
    # Explicit-negative repetition for the FULL-image path (ConceptDataset
    # repeats those samples this many times per epoch). The cached head-only
    # path overrides it from hard_negative_weight_candidates below, which now
    # defaults to no repetition — so the two paths diverge here until Bitcrush
    # ISSUE-0774 closes the gap, and this 3 stays unmeasured rather than
    # silently changed.
    hard_negative_weight: int = 3
    # Cached single-concept head training can cheaply try several repetition
    # strengths because every candidate reuses the same pooled features. An
    # empty/single-value list keeps fixed-weight behaviour.
    #
    # Narrowed to [1] in Bitcrush ISSUE-0773. Weight 5 had already gone in
    # ISSUE-0766; a holdout A/B over 8 seeds then showed the surviving [1,2,3]
    # sweep was selecting seed noise — its picks moved run to run while the
    # val-score spread between candidates stayed below the seed-to-seed
    # variance, and dropping it *improved* held-out recall (0.624 -> 0.643) at
    # identical precision, for a third of the probe cost. Repeating hard
    # lookalike negatives pushes the boundary into the positive cluster, and
    # tuned-threshold F1 cannot see that clearly enough to reject it.
    hard_negative_weight_candidates: list[int] = field(default_factory=lambda: [1])
    selected_hard_negative_weight: int | None = None
    hard_negative_weight_tuning_results: list[dict] = field(default_factory=list)
    hard_negative_weight_tuning_elapsed_ms: int | None = None
    # Total loss mass the implied-negative pool carries, as a multiple of the
    # positives' mass (Bitcrush ISSUE-0773). Implied negatives are other
    # concepts' positives — unlabelled for this concept — so at full weight a
    # 10:1 pool taught the model to suppress the concept (24.7% served recall
    # on a live concept; 74% of positives fell below the served threshold).
    # 1.0 keeps their coverage while removing their dominance; 0 disables the
    # normalisation and restores the historical full-weight behaviour.
    implied_negative_mass_alpha: float = 1.0
    # Bitcrush ISSUE-0898: bound the explicit-negative block's TOTAL loss mass
    # to this multiple of the positives' mass. Explicit negatives are verified,
    # so a scarce block keeps full weight per row; only an abundance (more
    # explicit negatives than positives, e.g. Sideboob 784 vs 238) is bounded,
    # which is what collapsed served recall to 3% on the harness holdout. The
    # head-only path scales per-row weights; the image path subsamples the
    # block per epoch. <= 0 disables.
    explicit_negative_mass_cap: float = 1.0
    # Validation implied-negative ratio, pinned independently of the training
    # ratio so metrics stay comparable across runs. None follows neg_pos_ratio.
    val_neg_pos_ratio: float | None = None
    label_smoothing: float = 0.1
    best_model_name: str = "best.pt"
    checkpoint_dir: str | None = None
    skin_normalise: bool = False
    face_model_path: str = ""
    cache_dir: str | None = None
    use_cache: bool = True
    cache_workers: int = 10
    # Pooled-feature cache used by genuine head-only training. None keeps the
    # cache beside the concept; Engine supplies a shared content-addressed root.
    embedding_cache_dir: str | None = None
    # Cache-build forward only (no-grad eval; EmbeddingCache backs off on OOM).
    embedding_batch_size: int = 256
    head_weight_decay: float = 0.02
    sourceless: bool = False
    concept_name: str = ""
    modeltype: str = "convnext_v2"
    progress_callback: Callable[[dict], None] | None = None
    # Layer-wise learning rate decay
    llrd: bool = True
    llrd_decay: float = 0.8
    # Exponential moving average of weights. Off by default: at 1k-10k-image
    # dataset sizes the configured decay never engages (effective decay is
    # (1+n)/(warmup+n), which only nears 0.9999 after ~90k steps), and the
    # full-model GPU copy adds VRAM pressure for negligible gain.
    use_ema: bool = False
    ema_decay: float = 0.9999
    # RandAugment + RandomErasing (DeiT/ConvNeXt official fine-tune recipe)
    randaugment_n: int = 2
    randaugment_m: int = 9
    random_erasing_p: float = 0.25
    # --- Backup / Pause / Resume (Bitcrush ISSUE-0405) ---
    # backup_dir=None => NO backups written and NO resume attempted (legacy).
    # backup_every_steps=0 => epoch-boundary backups only. resume_from points at
    # a backup dir/file. dataloader_workers replaces the hardcoded num_workers=4.
    backup_dir: str | None = None
    backup_every_steps: int = 500
    resume_from: str | None = None
    dataloader_workers: int = 4


def _fresh_binary_model(config: "TrainConfig", *, dtype: torch.dtype) -> nn.Module:
    model = create_model(
        model_size=config.model_size,
        pretrained=wants_timm_pretrained(config.backbone_init),
        dtype=dtype,
    )
    apply_backbone_init(model, config.backbone_init)
    return model


def _make_optimizer(model: nn.Module, config: "TrainConfig") -> Prodigy_adv:
    """Delegate to the shared factory (Bitcrush ISSUE-0542); signature kept so
    the binary loop's call sites are unchanged."""
    return make_optimizer(model, llrd=config.llrd, llrd_decay=config.llrd_decay)


def _get_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _unwrap_state_dict(data: dict | object) -> dict:
    if isinstance(data, dict) and "state_dict" in data:
        return data["state_dict"]
    return data


def _collate_bucket_batch(batch):
    """Collate a bucket batch — all images should share the same dimensions.

    Includes a center-crop safety net for the rare case where dimensions
    differ (e.g. edge-case bucket assignment changes between runs).
    """
    from torchvision.transforms import functional as TF

    images = [item[0] for item in batch]
    target_h, target_w = images[0].shape[1], images[0].shape[2]
    for i in range(1, len(images)):
        if images[i].shape[1] != target_h or images[i].shape[2] != target_w:
            images[i] = TF.center_crop(images[i], [target_h, target_w])
    images = torch.stack(images)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    *,
    step_callback: Callable[[int, int, float], None] | None = None,
    stop_now_event: object | None = None,
    ema: ModelEMA | None = None,
    randaugment_n: int = 0,
    randaugment_m: int = 0,
    random_erasing_p: float = 0.0,
    boundary_hook: Callable[[int], str | None] | None = None,
) -> float:
    from bittrainer.gpu_augment import apply_train_augment

    model.train()
    total_loss = 0.0
    num_batches = 0
    total_steps = len(dataloader)
    _last_report = time.monotonic()

    optimizer.zero_grad()
    for images, labels in dataloader:
        if stop_now_event is not None and stop_now_event.is_set():
            break
        images = images.to(device, non_blocking=True)
        images = apply_train_augment(
            images,
            dtype=dtype,
            randaugment_n=randaugment_n,
            randaugment_m=randaugment_m,
            random_erasing_p=random_erasing_p,
        )
        labels = labels.to(device)

        with torch.amp.autocast(
            device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

        # Backup/pause boundary (every batch — the binary trainer has no grad
        # accumulation). "stop" => a pause was requested and backed up; break.
        boundary_signal = (
            boundary_hook(num_batches) if boundary_hook is not None else None
        )

        if step_callback is not None:
            now = time.monotonic()
            if now - _last_report >= 2.0 or num_batches == total_steps:
                _last_report = now
                step_callback(num_batches, total_steps, total_loss / num_batches)

        if boundary_signal == "stop":
            break

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Evaluate on validation set. Returns loss, predictions, and labels."""
    from bittrainer.gpu_augment import apply_val_transform

    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    num_batches = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        images = apply_val_transform(images, dtype=dtype)
        labels = labels.to(device)

        with torch.amp.autocast(
            device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits.float(), dim=1)[:, 1]  # P(positive)
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        total_loss += loss.item()
        num_batches += 1

    return {
        "val_loss": total_loss / max(num_batches, 1),
        "probs": all_probs,
        "labels": all_labels,
        "kinds": _sample_kinds_in_loader_order(dataloader, len(all_labels)),
    }


def _sample_kinds_in_loader_order(
    dataloader: DataLoader, expected: int
) -> list[str] | None:
    """Per-prediction provenance for the pass that just ran, or None.

    ``BucketBatchSampler`` records the shuffled batch order it fed the loader
    (``last_epoch_batches``); mapping those indices onto ``dataset.samples``
    gives each prediction its ``kind``. Loaders without that sampler, or a
    length mismatch, report no provenance rather than a misaligned one.
    """
    batches = getattr(
        getattr(dataloader, "batch_sampler", None), "last_epoch_batches", None
    )
    samples = getattr(getattr(dataloader, "dataset", None), "samples", None)
    if not batches or samples is None:
        return None
    kinds = [samples[i].get("kind") for batch in batches for i in batch]
    if len(kinds) != expected or any(k is None for k in kinds):
        return None
    return kinds


def _tuned_val_metrics(val_result: dict) -> tuple[dict, float]:
    """Validation metrics at the F1-optimal threshold, plus that threshold.

    Inference ships ``find_optimal_threshold`` (not 0.5), so selecting and
    promoting checkpoints on F1@0.5 picks a model that is best at a boundary we
    never serve. Evaluating at the tuned threshold aligns checkpoint choice with
    the decision rule actually used at inference. The single-scalar threshold is
    fit on the same val set already used for the shipped threshold, so this adds
    no optimism beyond what the served metric already carries.
    """
    threshold = find_optimal_threshold(val_result["labels"], val_result["probs"])
    metrics = compute_metrics(
        val_result["labels"], val_result["probs"], threshold=threshold
    )
    kinds = val_result.get("kinds")
    if kinds:
        metrics.update(
            negative_kind_metrics(
                val_result["labels"], val_result["probs"], kinds, threshold=threshold
            )
        )
    return metrics, threshold


def _rebalance_val_negatives(train_ds: ConceptDataset, val_ds: ConceptDataset) -> None:
    """Ensure the val set has enough negatives for meaningful evaluation.

    Target: the validation dataset's implied-negative ratio (minimum 3:1).
    Donation never takes the training pool below its own implied-negative quota.
    """
    val_pos = len(val_ds._positive_paths)
    val_neg = len(val_ds._all_negative_paths)
    target = max(5, math.ceil(val_pos * val_ds._neg_pos_ratio))

    if val_neg >= target:
        return

    needed = target - val_neg
    train_floor = math.ceil(len(train_ds._positive_paths) * train_ds._neg_pos_ratio)
    max_donate = max(0, len(train_ds._all_negative_paths) - train_floor)
    to_donate = min(needed, max_donate, len(train_ds._all_negative_paths))

    if to_donate <= 0:
        return

    donated = train_ds._all_negative_paths[:to_donate]
    train_ds._all_negative_paths = train_ds._all_negative_paths[to_donate:]
    val_ds._all_negative_paths = val_ds._all_negative_paths + donated

    # Ensure val_ds has bucket info for donated paths (they were precomputed by train_ds)
    val_ds._path_info.update(
        {
            str(p): {**train_ds._path_info[str(p)], "split": val_ds.split}
            for p in donated
            if str(p) in train_ds._path_info
        }
    )

    train_ds._build_samples()
    val_ds._build_samples()

    logger.info(
        "Rebalanced val set: donated %d negatives from train → val (val now %d neg / %d pos)",
        to_donate,
        len(val_ds._all_negative_paths),
        val_pos,
    )


def run_training(
    config: TrainConfig,
    *,
    progress_callback: Callable[[dict], None] | None = None,
    stop_event: object | None = None,
    stop_now_event: object | None = None,
    pause_event: object | None = None,
) -> dict:
    """Run the full training loop. Returns a result dict with metrics and checkpoint path.

    stop_event signals a graceful stop that takes effect at the next epoch
    boundary. stop_now_event additionally interrupts the current epoch's
    training loop mid-batch; validation and the final fair-comparison block
    still run on the partial-epoch state.

    pause_event (Bitcrush ISSUE-0405) requests a resumable pause: the training
    state is backed up and the loop returns ``{"paused": True, ...}`` without
    running the fair-comparison / promotion block. Combined with
    ``config.backup_dir`` / ``config.resume_from`` a resume rebuilds the model,
    replays the gradual-unfreeze reconstruction, and **restarts the interrupted
    epoch** (mid-epoch snapshot, epoch-restart resume — the per-epoch scheduler
    keeps it consistent).
    """
    if config.training_mode == "head_only":
        from bittrainer.binary_head_only_trainer import run_binary_head_only_training

        return run_binary_head_only_training(
            config,
            progress_callback=progress_callback,
            stop_event=stop_event,
            stop_now_event=stop_now_event,
            pause_event=pause_event,
        )

    from bittrainer.generic.generic_trainer import GenericTrainer
    from bittrainer.generic.tasks.binary_task import BinaryTask

    return GenericTrainer().run(
        BinaryTask(config),
        progress_callback=progress_callback,
        stop_event=stop_event,
        stop_now_event=stop_now_event,
        pause_event=pause_event,
    )


def _binary_compare_promote(
    config,
    *,
    best_checkpoint_path,
    existing_best,
    model,
    val_loader,
    criterion,
    device,
    dtype,
    best_val_f1,
    best_metrics,
    best_epoch,
    epochs_completed,
    num_positives,
    train_ds,
) -> dict:
    """Promote-if-better vs the incumbent and build the binary result dict.

    Extracted verbatim from ``run_training`` so the trainer's exception-wrapped
    body ends in a single call (Bitcrush ISSUE-0405)."""
    optimal_threshold = 0.5
    if best_checkpoint_path:
        if existing_best.exists():
            # Re-evaluate the old best.pt on the current validation set
            try:
                old_data = torch.load(
                    str(existing_best),
                    map_location=device,
                    weights_only=True,
                )
                old_sd = (
                    old_data["state_dict"]
                    if isinstance(old_data, dict) and "state_dict" in old_data
                    else old_data
                )
                old_size = (
                    old_data.get("model_size", config.model_size)
                    if isinstance(old_data, dict)
                    else config.model_size
                )
                old_model = create_model(
                    model_size=old_size, pretrained=False, dtype=dtype
                ).to(device)
                old_model.load_state_dict(old_sd)
                old_val_result = evaluate(
                    old_model,
                    val_loader,
                    criterion,
                    device,
                    dtype,
                )
                # Compare old vs new at the tuned threshold (consistent with the
                # per-epoch selection metric and with what inference serves).
                old_metrics, old_threshold = _tuned_val_metrics(old_val_result)
                old_f1 = old_metrics.get("f1", 0.0)
                del old_model  # free GPU memory

                if old_f1 > best_val_f1:
                    # Old model is better — keep existing best.pt, discard candidate
                    logger.info(
                        "Old checkpoint F1 %.4f > new F1 %.4f — keeping old",
                        old_f1,
                        best_val_f1,
                    )
                    candidate = Path(best_checkpoint_path)
                    if candidate.exists():
                        candidate.unlink()
                    best_checkpoint_path = str(existing_best)
                    best_val_f1 = old_f1
                    best_metrics = old_metrics
                    optimal_threshold = old_threshold
                else:
                    # New model wins — promote candidate to best.pt
                    logger.info(
                        "New checkpoint F1 %.4f >= old F1 %.4f — promoting new",
                        best_val_f1,
                        old_f1,
                    )
                    candidate = Path(best_checkpoint_path)
                    candidate.replace(existing_best)
                    best_checkpoint_path = str(existing_best)
                    model.load_state_dict(
                        _unwrap_state_dict(
                            torch.load(
                                best_checkpoint_path,
                                map_location=device,
                                weights_only=True,
                            ),
                        )
                    )
                    val_result = evaluate(
                        model,
                        val_loader,
                        criterion,
                        device,
                        dtype,
                    )
                    optimal_threshold = find_optimal_threshold(
                        val_result["labels"],
                        val_result["probs"],
                    )
            except Exception:
                # Old checkpoint incompatible (e.g. architecture change) — new wins
                logger.warning(
                    "Failed to re-evaluate old checkpoint, keeping new",
                    exc_info=True,
                )
                candidate = Path(best_checkpoint_path)
                candidate.replace(existing_best)
                best_checkpoint_path = str(existing_best)
                model.load_state_dict(
                    _unwrap_state_dict(
                        torch.load(
                            best_checkpoint_path,
                            map_location=device,
                            weights_only=True,
                        ),
                    ),
                )
                val_result = evaluate(
                    model,
                    val_loader,
                    criterion,
                    device,
                    dtype,
                )
                optimal_threshold = find_optimal_threshold(
                    val_result["labels"],
                    val_result["probs"],
                )
        else:
            # No existing best.pt — promote candidate directly
            candidate = Path(best_checkpoint_path)
            candidate.replace(existing_best)
            best_checkpoint_path = str(existing_best)
            model.load_state_dict(
                _unwrap_state_dict(
                    torch.load(
                        best_checkpoint_path, map_location=device, weights_only=True
                    ),
                )
            )
            val_result = evaluate(model, val_loader, criterion, device, dtype)
            optimal_threshold = find_optimal_threshold(
                val_result["labels"],
                val_result["probs"],
            )

    return {
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch + 1,
        "best_val_f1": best_val_f1,
        "final_val_f1": best_metrics.get("f1"),
        "final_val_precision": best_metrics.get("precision"),
        "final_val_recall": best_metrics.get("recall"),
        "final_val_auprc": best_metrics.get("auprc"),
        "final_val_loss": best_metrics.get("val_loss"),
        "optimal_threshold": optimal_threshold,
        "checkpoint_path": best_checkpoint_path,
        "positive_count": num_positives,
        "negative_count": len(train_ds._all_negative_paths),
        "confusion_matrix": best_metrics.get("confusion_matrix"),
        # Bitcrush ISSUE-0898: negatives by kind. ``negative_count`` above is
        # the whole implied pool available, not what was trained on.
        **_negative_kind_result(train_ds, best_metrics),
    }


def _negative_kind_result(train_ds, best_metrics: dict) -> dict:
    counts = getattr(train_ds, "negative_kind_counts", lambda: {})()
    return {
        "explicit_negative_count": counts.get("explicit"),
        "implied_negative_count": counts.get("implied"),
        "val_explicit_negative_count": best_metrics.get("val_explicit_negative_count"),
        "val_implied_negative_count": best_metrics.get("val_implied_negative_count"),
        "final_val_f1_explicit": best_metrics.get("f1_explicit"),
        "final_val_f1_implied": best_metrics.get("f1_implied"),
        "final_val_fpr_explicit": best_metrics.get("fpr_explicit"),
        "final_val_fpr_implied": best_metrics.get("fpr_implied"),
    }
