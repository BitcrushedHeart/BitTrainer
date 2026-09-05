"""Binary head-only task backed by a persistent pooled-feature cache."""

from __future__ import annotations

import copy
import logging
import math
import random
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import bittrainer.trainer as bt
from bittrainer.dataset import (
    DEFAULT_TRAIN_RESOLUTION,
    build_bucket_batch_sampler,
    effective_binary_neg_pos_ratio,
)
from bittrainer.embedding_cache import EmbeddingCache, cached_hashes_for_sig
from bittrainer.generic.task import BestTracker, LoopSpec, TaskContext
from bittrainer.generic.tasks.binary_task import BinaryTask
from bittrainer.head_probe import _gather
from bittrainer.model import (
    backbone_feature_hash,
    head_tail_logits,
    head_tail_parameters,
)
from bittrainer.training_state import (
    BackupCoordinator,
    capture_rng_states,
    restore_rng_states,
)

logger = logging.getLogger(__name__)

_PROBE_BATCH_SIZE = 256
_PROBE_LEARNING_RATE = 1e-3


def _embedding_preproc_sig(train_resolution: int) -> str:
    if not train_resolution or train_resolution == DEFAULT_TRAIN_RESOLUTION:
        return "val_imagenet"
    return f"val_imagenet@{int(train_resolution)}"


def _cached_negative_predicate(
    config: bt.TrainConfig,
) -> Callable[[str], bool] | None:
    """Build ``path -> already feature-passed`` for negative-pool seeding.

    Binary concepts sharing a pinned backbone also share the pooled-vector
    cache, so an implied negative whose content hash already has a vector under
    the current preproc-sig eras trains for free. Path→hash resolution is a
    read-only SmartCache index lookup; a path the smart cache has never seen
    counts as a miss (no file hashing at sampling time). Returns ``None`` when
    there is nothing to prefer, keeping the sampler purely random.
    """
    if not (config.use_cache and config.cache_dir and config.embedding_cache_dir):
        return None
    cached_hashes = cached_hashes_for_sig(
        config.embedding_cache_dir, _embedding_preproc_sig(config.train_resolution)
    )
    if not cached_hashes:
        return None
    from bittrainer.smart_cache import SmartCache

    index = SmartCache(config.cache_dir, modeltype=config.modeltype)

    def is_cached(path: str) -> bool:
        content_hash = index.content_hash(path)
        return content_hash is not None and content_hash in cached_hashes

    return is_cached


def _sample_negative_pool(
    paths: list[str],
    *,
    positive_count: int,
    neg_pos_ratio: float,
    ratio_positive_count: int | None = None,
    reserve: int = 0,
    is_cached: Callable[[str], bool] | None = None,
) -> list[str]:
    """Select implied negatives before expensive dimension indexing.

    The implied pool independently receives at least three images per positive.
    Explicit negatives live in a separate additive pool and are not subtracted.
    ``reserve`` keeps enough train candidates available for a validation
    shortfall to be donated without weakening the train ratio.

    With ``is_cached``, the quota fills from already-feature-passed candidates
    first (random within that tier) and tops up from uncached images only on
    shortfall, so retrains reuse the transferable embedding cache instead of
    paying backbone forwards for fresh random negatives.
    """
    ratio = effective_binary_neg_pos_ratio(
        neg_pos_ratio,
        positive_count=(
            positive_count if ratio_positive_count is None else ratio_positive_count
        ),
    )
    quota = math.ceil(positive_count * ratio) + max(0, int(reserve))
    if len(paths) <= quota:
        return list(paths)
    if is_cached is None:
        return random.sample(paths, quota)
    hits: list[str] = []
    misses: list[str] = []
    for path in paths:
        (hits if is_cached(path) else misses).append(path)
    if len(hits) >= quota:
        return random.sample(hits, quota)
    return hits + random.sample(misses, quota - len(hits))


def implied_negative_weights(y: torch.Tensor, *, alpha: float) -> torch.Tensor:
    """Per-sample loss weights that bound the implied pool's total influence.

    ``y`` holds the base training block only — positives and implied negatives,
    with explicit negatives already partitioned out — so every zero label here
    is an implied negative.

    Implied negatives are other concepts' positives (``deps._binary_file_splits``),
    never assessed for *this* concept. They are unlabelled, not verified negative,
    and on a people-heavy dataset they are densely contaminated with genuine
    positives. At full weight a 10:1 pool therefore teaches the model that images
    resembling the rest of the dataset are negative, which is why affected
    concepts only fired on out-of-domain images. Scaling the pool so it carries
    ``alpha`` times the positives' total loss mass keeps its coverage while
    removing its dominance — the standard treatment for unlabelled data.

    ``alpha <= 0`` disables normalisation and restores full weight.
    """
    weights = torch.ones_like(y, dtype=torch.float32)
    if alpha <= 0:
        return weights
    positives = int((y == 1).sum())
    implied = int((y == 0).sum())
    if positives and implied:
        weights[y == 0] = alpha * positives / implied
    return weights


def _weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None,
    smoothing: float,
) -> torch.Tensor:
    """Cross entropy under per-sample weights, normalised by total weight.

    Normalising by summed weight rather than sample count keeps the effective
    step size unchanged: down-weighting the implied pool must alter the
    *balance* of the gradient, not shrink the whole update.
    """
    if weights is None:
        return nn.functional.cross_entropy(logits, labels, label_smoothing=smoothing)
    losses = nn.functional.cross_entropy(
        logits, labels, label_smoothing=smoothing, reduction="none"
    )
    total = weights.sum()
    if float(total) <= 0.0:
        return losses.mean()
    return (losses * weights).sum() / total


def _train_cached_binary_head(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    config: bt.TrainConfig,
    *,
    device: torch.device,
    cb,
    stop_event,
    w_train: torch.Tensor | None = None,
    progress_stage: str = "training",
    progress_prefix: str = "Head probe",
) -> dict:
    """Train only the classifier tail and select on tuned-threshold binary F1."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tail_dtype = model.head.fc.weight.dtype
    model.head.fc.float()
    model.head.pre_logits.float()
    tail_parameters = head_tail_parameters(model)
    for parameter in tail_parameters:
        parameter.requires_grad_(True)
    model.eval()

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        tail_parameters,
        lr=_PROBE_LEARNING_RATE,
        weight_decay=config.head_weight_decay,
    )
    # Weights ride the TensorDataset so shuffling keeps each sample with its own
    # weight; an all-ones tensor reproduces the historical unweighted loss.
    if w_train is None:
        w_train = torch.ones_like(y_train, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(x_train, y_train, w_train),
        batch_size=min(_PROBE_BATCH_SIZE, len(x_train)),
        shuffle=True,
        drop_last=False,
    )

    best_score = -1.0
    best_epoch = 0
    best_metrics: dict = {}
    best_head_state = copy.deepcopy(model.head.state_dict())
    patience = 0
    epoch = -1

    for epoch in range(config.max_epochs):
        if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
            break

        model.head.train()
        for features, labels, weights in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = head_tail_logits(model, features)
            loss = _weighted_cross_entropy(
                logits, labels, weights.to(device), config.label_smoothing
            )
            loss.backward()
            optimizer.step()

        model.head.eval()
        with torch.no_grad():
            val_logits = head_tail_logits(model, x_val.to(device)).float()
            # Validation loss stays unweighted: it measures the model, not the
            # run's training-time emphasis, so it remains comparable.
            val_loss = float(criterion(val_logits, y_val.to(device)).item())
        val_probs = torch.softmax(val_logits, dim=1)[:, 1]
        val_result = {
            "val_loss": val_loss,
            "probs": val_probs.cpu().tolist(),
            "labels": y_val.cpu().tolist(),
        }
        metrics, _threshold = bt._tuned_val_metrics(val_result)
        metrics["val_loss"] = val_loss
        score = float(metrics.get("f1", 0.0))

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics
            best_head_state = copy.deepcopy(model.head.state_dict())
            patience = 0
        else:
            patience += 1

        cb(
            {
                "type": "training_progress",
                "stage": progress_stage,
                "status_text": (
                    f"{progress_prefix} epoch {epoch + 1}/{config.max_epochs} (val F1 {score:.3f})"
                ),
                "epoch": epoch + 1,
                "max_epochs": config.max_epochs,
                "val_f1": score,
                "val_precision": metrics.get("precision", 0.0),
                "val_recall": metrics.get("recall", 0.0),
                "best_val_f1": best_score,
                "best_epoch": best_epoch + 1,
            }
        )

        if patience >= config.patience:
            logger.info(
                "Binary head probe early-stopping at epoch %d (patience=%d)",
                epoch + 1,
                config.patience,
            )
            break

    model.head.load_state_dict(best_head_state)
    model.head.to(tail_dtype)
    return {
        "best_epoch": best_epoch + 1,
        "epochs_completed": epoch + 1,
        "best_val_f1": best_score,
        "best_metrics": best_metrics,
    }


def _normalise_hard_negative_weight_candidates(config: bt.TrainConfig) -> list[int]:
    candidates: list[int] = []
    for raw in config.hard_negative_weight_candidates or []:
        try:
            weight = int(raw)
        except (TypeError, ValueError):
            continue
        if weight >= 1 and weight not in candidates:
            candidates.append(weight)
    if not candidates:
        candidates.append(max(1, int(config.hard_negative_weight)))
    return candidates


def explicit_block_row_weight(
    *, positives: int, explicit_rows: int, mass_cap: float
) -> float:
    """Per-row weight that bounds the explicit block at ``mass_cap`` x positives.

    Bitcrush ISSUE-0898. A scarce block (rows <= cap x positives) keeps full
    weight; an abundant one is scaled so its total mass equals the cap.
    ``mass_cap <= 0`` disables the bound.
    """
    if mass_cap <= 0 or positives <= 0 or explicit_rows <= 0:
        return 1.0
    return min(1.0, mass_cap * positives / explicit_rows)


def _weighted_train_tensors(
    x_base: torch.Tensor,
    y_base: torch.Tensor,
    x_explicit: torch.Tensor,
    y_explicit: torch.Tensor,
    weight: int,
    w_base: torch.Tensor | None = None,
    mass_cap: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Append the explicit-negative block, repeated ``weight`` times.

    Explicit negatives are user-verified, so each keeps full weight until the
    repeated block would carry more than ``mass_cap`` x the positives' mass
    (ISSUE-0898); the implied pool inside ``w_base`` is normalised separately.
    """
    if len(x_explicit) == 0:
        return x_base, y_base, w_base
    repeated_x = torch.cat([x_explicit] * weight, dim=0)
    repeated_y = torch.cat([y_explicit] * weight, dim=0)
    w_out = None
    if w_base is not None:
        row_weight = explicit_block_row_weight(
            positives=int((y_base == 1).sum()),
            explicit_rows=len(repeated_y),
            mass_cap=mass_cap,
        )
        w_out = torch.cat(
            (w_base, torch.full((len(repeated_y),), row_weight, dtype=w_base.dtype)),
            dim=0,
        )
    return (
        torch.cat((x_base, repeated_x), dim=0),
        torch.cat((y_base, repeated_y), dim=0),
        w_out,
    )


def _hard_negative_candidate_better(candidate: dict, incumbent: dict | None) -> bool:
    if incumbent is None:
        return True
    candidate_f1 = float(candidate.get("f1") or 0.0)
    incumbent_f1 = float(incumbent.get("f1") or 0.0)
    if candidate_f1 != incumbent_f1:
        return candidate_f1 > incumbent_f1
    candidate_loss = candidate.get("val_loss")
    incumbent_loss = incumbent.get("val_loss")
    if (
        candidate_loss is not None
        and incumbent_loss is not None
        and float(candidate_loss) != float(incumbent_loss)
    ):
        return float(candidate_loss) < float(incumbent_loss)
    return int(candidate["weight"]) < int(incumbent["weight"])


def _train_hard_negative_weight_sweep(
    model: nn.Module,
    x_base: torch.Tensor,
    y_base: torch.Tensor,
    x_explicit: torch.Tensor,
    y_explicit: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    config: bt.TrainConfig,
    *,
    device: torch.device,
    cb,
    stop_event,
    w_base: torch.Tensor | None = None,
) -> dict:
    """Select explicit-negative repetition strength by tuned-threshold val F1.

    Every candidate starts from the same head and RNG state. Validation tensors
    are never repeated, so increasing the training strength cannot inflate its
    own selection metric by changing validation support.
    """
    candidates = _normalise_hard_negative_weight_candidates(config)
    if len(x_explicit) == 0 or len(candidates) == 1:
        weight = max(1, int(config.hard_negative_weight))
        if len(x_explicit) > 0:
            weight = candidates[0]
            config.hard_negative_weight = weight
        x_train, y_train, w_train = _weighted_train_tensors(
            x_base,
            y_base,
            x_explicit,
            y_explicit,
            weight,
            w_base,
            mass_cap=config.explicit_negative_mass_cap,
        )
        return _train_cached_binary_head(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            config,
            device=device,
            cb=cb,
            stop_event=stop_event,
            w_train=w_train,
        )

    original_head_state = copy.deepcopy(model.head.state_dict())
    original_rng_state = capture_rng_states(device)
    best_row: dict | None = None
    best_probe: dict | None = None
    best_head_state: dict | None = None
    matrix: list[dict] = []
    sweep_start = time.monotonic()

    for index, weight in enumerate(candidates, start=1):
        if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
            break
        model.head.load_state_dict(original_head_state)
        restore_rng_states(original_rng_state, device)
        x_train, y_train, w_train = _weighted_train_tensors(
            x_base,
            y_base,
            x_explicit,
            y_explicit,
            weight,
            w_base,
            mass_cap=config.explicit_negative_mass_cap,
        )
        cb(
            {
                "type": "training_progress",
                "stage": "explicit_negative_tuning",
                "status_text": (
                    f"Testing explicit-negative strength {weight} ({index}/{len(candidates)})"
                ),
                "step": index,
                "total_steps": len(candidates),
                "hard_negative_weight": weight,
            }
        )
        candidate_start = time.monotonic()
        probe = _train_cached_binary_head(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            config,
            device=device,
            cb=cb,
            stop_event=stop_event,
            w_train=w_train,
            progress_stage="explicit_negative_tuning",
            progress_prefix=f"Explicit-negative strength {weight}",
        )
        if int(probe.get("epochs_completed") or 0) <= 0:
            break
        metrics = probe.get("best_metrics") or {}
        row = {
            "weight": weight,
            "f1": float(probe.get("best_val_f1") or metrics.get("f1") or 0.0),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "auprc": metrics.get("auprc"),
            "val_loss": metrics.get("val_loss"),
            "best_epoch": probe.get("best_epoch"),
            "epochs_completed": probe.get("epochs_completed"),
            "elapsed_ms": int(round((time.monotonic() - candidate_start) * 1000)),
        }
        matrix.append(row)
        cb(
            {
                "type": "training_progress",
                "stage": "explicit_negative_tuning",
                "status_text": (
                    f"Tested explicit-negative strength {weight}: F1 {row['f1']:.3f}"
                ),
                "step": index,
                "total_steps": len(candidates),
                "hard_negative_weight": weight,
                "val_f1": row["f1"],
                "val_precision": row["precision"],
                "val_recall": row["recall"],
            }
        )
        if _hard_negative_candidate_better(row, best_row):
            best_row = row
            best_probe = probe
            best_head_state = copy.deepcopy(model.head.state_dict())

    if best_row is None or best_probe is None or best_head_state is None:
        model.head.load_state_dict(original_head_state)
        return {"best_epoch": 0, "epochs_completed": 0, "best_val_f1": -1.0}

    model.head.load_state_dict(best_head_state)
    selected = int(best_row["weight"])
    config.hard_negative_weight = selected
    config.selected_hard_negative_weight = selected
    config.hard_negative_weight_tuning_results = matrix
    config.hard_negative_weight_tuning_elapsed_ms = int(
        round((time.monotonic() - sweep_start) * 1000)
    )
    cb(
        {
            "type": "training_progress",
            "stage": "explicit_negative_tuning",
            "status_text": (
                f"Selected explicit-negative strength {selected} by validation F1"
            ),
            "step": len(matrix),
            "total_steps": len(candidates),
            "hard_negative_weight": selected,
            "best_val_f1": best_row["f1"],
        }
    )
    return best_probe


def _partition_explicit_negative_samples(
    samples: list[dict], hard_negative_paths: list[Path]
) -> tuple[list[dict], list[dict]]:
    explicit_paths = {str(path) for path in hard_negative_paths}
    if not explicit_paths:
        return samples, []
    base: list[dict] = []
    explicit_by_path: dict[str, dict] = {}
    for sample in samples:
        path = str(sample["path"])
        if path in explicit_paths:
            explicit_by_path.setdefault(path, sample)
        else:
            base.append(sample)
    return base, list(explicit_by_path.values())


class BinaryHeadOnlyTask(BinaryTask):
    """Run binary training without an image/backbone epoch loop."""

    trainer_name = "binary_head_only"

    def __init__(self, config: bt.TrainConfig) -> None:
        super().__init__(config)
        self.embedding_cache_stats: dict = {}
        self.backbone_hash = ""
        self.probe: dict = {}
        self.candidate_metrics: dict = {}
        self.candidate_path: Path | None = None
        self.criterion: nn.Module | None = None
        self._cancelled = False

    def _stop(self, ctx: TaskContext) -> bool:
        return bool(
            (ctx.stop_event is not None and ctx.stop_event.is_set())
            or (ctx.stop_now_event is not None and ctx.stop_now_event.is_set())
            or (ctx.pause_event is not None and ctx.pause_event.is_set())
        )

    def fingerprint_init(self, ctx: TaskContext) -> None:
        # The cached probe is cheap to restart and has no optimizer-loop backup.
        ctx.coordinator = BackupCoordinator(backup_dir=None)
        ctx.fingerprint = None
        ctx.resume_state = None

    def loop_spec(self) -> LoopSpec:
        return LoopSpec(max_epochs=0, patience=0)

    def prepare_data(self, ctx: TaskContext) -> None:
        """Build a fixed probe set without indexing every resampling candidate."""
        config = self.config
        original_train = config.train_negative_paths
        original_val = config.val_negative_paths
        try:
            ratio_positive_count = (
                len(config.train_positive_paths) + len(config.val_positive_paths)
                if config.train_positive_paths is not None
                and config.val_positive_paths is not None
                else None
            )
            ratio = effective_binary_neg_pos_ratio(
                config.neg_pos_ratio,
                positive_count=ratio_positive_count,
            )
            # Validation prevalence is pinned independently of the train ratio so
            # F1/AUPRC stay comparable across runs (ISSUE-0755: raw F1 is not
            # comparable across neg:pos ratios).
            val_ratio = (
                ratio
                if config.val_neg_pos_ratio is None
                else float(config.val_neg_pos_ratio)
            )
            is_cached = _cached_negative_predicate(config)
            val_shortfall = 0
            if original_val is not None and config.val_positive_paths is not None:
                val_quota = math.ceil(len(config.val_positive_paths) * val_ratio)
                val_shortfall = max(0, val_quota - len(original_val))
            if original_train is not None and config.train_positive_paths is not None:
                config.train_negative_paths = _sample_negative_pool(
                    original_train,
                    positive_count=len(config.train_positive_paths),
                    neg_pos_ratio=ratio,
                    ratio_positive_count=ratio_positive_count,
                    reserve=val_shortfall,
                    is_cached=is_cached,
                )
            if original_val is not None and config.val_positive_paths is not None:
                config.val_negative_paths = _sample_negative_pool(
                    original_val,
                    positive_count=len(config.val_positive_paths),
                    neg_pos_ratio=val_ratio,
                    # A pinned val ratio is a ceiling in its own right, so the
                    # concept-wide positive count (which drives the adaptive
                    # train ratio) must not be fed in alongside it.
                    ratio_positive_count=(
                        ratio_positive_count
                        if config.val_neg_pos_ratio is None
                        else None
                    ),
                    is_cached=is_cached,
                )
            super().prepare_data(ctx)
        finally:
            config.train_negative_paths = original_train
            config.val_negative_paths = original_val

    def pre_loop(self, ctx: TaskContext, model) -> None:
        config = self.config
        all_samples = self.train_ds.samples + self.val_ds.samples
        self.backbone_hash = backbone_feature_hash(model)
        cache_root = config.embedding_cache_dir or str(
            Path(config.concept_folder) / ".embedding_cache"
        )
        embedding_cache = EmbeddingCache(
            cache_root,
            self.backbone_hash,
            int(getattr(model, "num_features", 0)),
            preproc_sig=_embedding_preproc_sig(config.train_resolution),
        )

        def _build_progress(done: int, total: int) -> None:
            ctx.cb(
                {
                    "type": "training_progress",
                    "stage": "embedding_build",
                    "status_text": f"Building feature cache ({done}/{total})",
                    "step": done,
                    "total_steps": total,
                }
            )

        ctx.cb(
            {
                "type": "training_progress",
                "stage": "embedding_build",
                "status_text": f"Caching backbone features (era {self.backbone_hash})",
            }
        )
        self.embedding_cache_stats = embedding_cache.ensure(
            all_samples,
            model,
            self.smart_cache,
            device=ctx.device,
            dtype=ctx.dtype,
            batch_size=config.embedding_batch_size,
            progress_cb=_build_progress,
            stop_check=lambda: self._stop(ctx),
            # Engine uses one shared content-addressed root. Different backbone
            # eras intentionally coexist so binary concepts can reuse common
            # pretrained vectors without deleting fine-tuned concept eras.
            prune=False,
        )
        if self._stop(ctx):
            self._cancelled = True
            return

        embedding_cache.verify(
            all_samples,
            model,
            self.smart_cache,
            device=ctx.device,
            dtype=ctx.dtype,
        )
        load_started = time.monotonic()
        base_samples, explicit_samples = _partition_explicit_negative_samples(
            self.train_ds.samples, self.train_ds._hard_negative_paths
        )
        x_train, y_train = _gather(
            base_samples,
            embedding_cache,
            self.smart_cache,
            multi_label=False,
        )
        if explicit_samples:
            x_explicit, y_explicit = _gather(
                explicit_samples,
                embedding_cache,
                self.smart_cache,
                multi_label=False,
            )
        else:
            x_explicit = x_train.new_empty((0, *x_train.shape[1:]))
            y_explicit = y_train.new_empty((0,))
        x_val, y_val = _gather(
            self.val_ds.samples,
            embedding_cache,
            self.smart_cache,
            multi_label=False,
        )
        ctx.cb(
            {
                "type": "training_progress",
                "stage": "training",
                "status_text": "Training binary head on cached features",
                "cached_features": len(x_train) + len(x_explicit) + len(x_val),
                "feature_load_seconds": round(time.monotonic() - load_started, 3),
            }
        )
        # x_train/y_train are the base block (positives + implied negatives);
        # explicit negatives were partitioned out above and keep full weight.
        w_train = implied_negative_weights(
            y_train, alpha=config.implied_negative_mass_alpha
        )
        self.probe = _train_hard_negative_weight_sweep(
            model,
            x_train,
            y_train,
            x_explicit,
            y_explicit,
            x_val,
            y_val,
            config,
            device=ctx.device,
            cb=ctx.cb,
            stop_event=ctx.stop_event,
            w_base=w_train,
        )
        if self._stop(ctx) or self.probe["epochs_completed"] <= 0:
            self._cancelled = True
            return

        # One full-image pass keeps candidate/incumbent comparison identical to
        # ordinary binary training; the backbone is not repeated per head epoch.
        val_loader = DataLoader(
            self.val_ds,
            # Capped at 64: this full-image backbone pass has no OOM backoff
            # (unlike EmbeddingCache.ensure, where the larger default is safe).
            batch_sampler=build_bucket_batch_sampler(
                self.val_ds, batch_size=min(config.embedding_batch_size, 64)
            ),
            collate_fn=bt._collate_bucket_batch,
            num_workers=0,
        )
        self._val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        val_result = bt.evaluate(
            model, val_loader, self.criterion, ctx.device, ctx.dtype
        )
        self.candidate_metrics, _threshold = bt._tuned_val_metrics(val_result)
        self.candidate_metrics["val_loss"] = val_result["val_loss"]

        self.candidate_path = ctx.checkpoint_dir / "candidate.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "num_classes": 2,
                "model_size": config.model_size,
                "training_mode": "head_only",
                "selected_hard_negative_weight": config.selected_hard_negative_weight,
                "hard_negative_weight_tuning_results": (
                    config.hard_negative_weight_tuning_results
                ),
            },
            self.candidate_path,
        )

    def resolve_batch_size(self, ctx, model, resume_state) -> int:
        return 0

    def create_optimizer(self, ctx, model, eff_bs, resume_state):
        return None, None, 0

    def build_loaders(self, ctx, epoch, eff_bs, resume_info):  # pragma: no cover
        raise AssertionError("binary head-only has no image epoch loop")

    def train_epoch(
        self,
        ctx,
        model,
        optimizer,
        train_loader,
        *,
        step_callback,
        boundary_hook,
        start_batch,
    ):  # pragma: no cover
        raise AssertionError("binary head-only has no image epoch loop")

    def finalize(self, ctx, model, best: BestTracker, epochs_completed: int) -> dict:
        if self._cancelled:
            return {"cancelled": True, "mode": "head_only"}

        candidate_f1 = float(self.candidate_metrics.get("f1", 0.0))
        result = bt._binary_compare_promote(
            self.config,
            best_checkpoint_path=str(self.candidate_path),
            existing_best=ctx.checkpoint_dir / self.config.best_model_name,
            model=model,
            val_loader=self._val_loader,
            criterion=self.criterion,
            device=ctx.device,
            dtype=ctx.dtype,
            best_val_f1=candidate_f1,
            best_metrics=self.candidate_metrics,
            best_epoch=int(self.probe["best_epoch"]) - 1,
            epochs_completed=int(self.probe["epochs_completed"]),
            num_positives=self.num_positives,
            train_ds=self.train_ds,
        )
        result["mode"] = "head_only"
        result["backbone_hash"] = self.backbone_hash
        result["embedding_cache_stats"] = self.embedding_cache_stats
        result["selected_hard_negative_weight"] = (
            self.config.selected_hard_negative_weight
        )
        result["hard_negative_weight_tuning_results"] = (
            self.config.hard_negative_weight_tuning_results
        )
        result["hard_negative_weight_tuning_elapsed_ms"] = (
            self.config.hard_negative_weight_tuning_elapsed_ms
        )
        return result
