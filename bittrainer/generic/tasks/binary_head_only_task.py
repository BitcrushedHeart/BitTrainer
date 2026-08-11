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
from bittrainer.training_state import BackupCoordinator

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
    ratio = effective_binary_neg_pos_ratio(neg_pos_ratio)
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
    loader = DataLoader(
        TensorDataset(x_train, y_train),
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
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = head_tail_logits(model, features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.head.eval()
        with torch.no_grad():
            val_logits = head_tail_logits(model, x_val.to(device)).float()
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
                "stage": "training",
                "status_text": (
                    f"Head probe epoch {epoch + 1}/{config.max_epochs} "
                    f"(val F1 {score:.3f})"
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
            ratio = effective_binary_neg_pos_ratio(config.neg_pos_ratio)
            is_cached = _cached_negative_predicate(config)
            val_shortfall = 0
            if original_val is not None and config.val_positive_paths is not None:
                val_quota = math.ceil(len(config.val_positive_paths) * ratio)
                val_shortfall = max(0, val_quota - len(original_val))
            if original_train is not None and config.train_positive_paths is not None:
                config.train_negative_paths = _sample_negative_pool(
                    original_train,
                    positive_count=len(config.train_positive_paths),
                    neg_pos_ratio=ratio,
                    reserve=val_shortfall,
                    is_cached=is_cached,
                )
            if original_val is not None and config.val_positive_paths is not None:
                config.val_negative_paths = _sample_negative_pool(
                    original_val,
                    positive_count=len(config.val_positive_paths),
                    neg_pos_ratio=ratio,
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
        x_train, y_train = _gather(
            self.train_ds.samples,
            embedding_cache,
            self.smart_cache,
            multi_label=False,
        )
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
                "cached_features": len(x_train) + len(x_val),
                "feature_load_seconds": round(time.monotonic() - load_started, 3),
            }
        )
        self.probe = _train_cached_binary_head(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            config,
            device=ctx.device,
            cb=ctx.cb,
            stop_event=ctx.stop_event,
        )
        if self._stop(ctx) or self.probe["epochs_completed"] <= 0:
            self._cancelled = True
            return

        # One full-image pass keeps candidate/incumbent comparison identical to
        # ordinary binary training; the backbone is not repeated per head epoch.
        val_loader = DataLoader(
            self.val_ds,
            batch_sampler=build_bucket_batch_sampler(
                self.val_ds, batch_size=config.embedding_batch_size
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
        return result
