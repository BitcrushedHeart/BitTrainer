"""Backbone head-only retraining on cached pooled features.

``BackboneHeadsTask`` retrains the multi-task concept/group heads against a
FROZEN existing backbone checkpoint: one embedding pass per backbone era
(memoised in :class:`~bittrainer.embedding_cache.EmbeddingCache`, namespaced by
backbone-feature hash x preprocessing signature), then head epochs over cached
vectors — seconds per epoch instead of a full backbone forward per image. This
makes "retrain all heads after a labelling round" the cheap default and a full
backbone era the exception.

Subclasses :class:`BackboneTask` so the record/vocab assembly, the per-epoch
sampling plan (neg:pos cap, tiny-head oversample, pos_weight, label policy —
ISSUE-0545/0546) and the selection/early-stop shape are shared; only the
model/build/loss plumbing swaps images for vectors. The exported candidate
keeps the 0542 convention — BARE backbone keys (byte-copied from the source
checkpoint) + fresh ``heads.*`` tensors — so ``apply_backbone_init`` and every
existing consumer reads it unchanged.

No backup/resume (runs are minutes; always fresh) and no EMA on the heads.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Dataset

import bittrainer.backbone_trainer as bb
from bittrainer.backbone_init import apply_backbone_init
from bittrainer.generic.optimizer import make_optimizer
from bittrainer.generic.task import BestTracker, TaskContext
from bittrainer.generic.tasks.backbone_task import BackboneTask
from bittrainer.model import create_model
from bittrainer.training_state import BackupCoordinator

logger = logging.getLogger(__name__)


class _VectorDataset(Dataset):
    """(pooled vector, binary labels, group labels) triples for one epoch plan.

    Samples whose vector is missing from the cache (unhashable / unreadable
    file) are silently excluded — they were already warned about at cache-build
    time.
    """

    def __init__(self, samples: list, vectors: dict):
        self.samples = [s for s in samples if s.path in vectors]
        self.vectors = vectors

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        vector = torch.from_numpy(self.vectors[sample.path]).float()
        return vector, sample.binary, sample.groups


class BackboneHeadsTask(BackboneTask):
    """Head-only backbone training: frozen trunk, cached features, fresh heads."""

    trainer_name = "backbone_heads"

    def __init__(self, request: dict, *, cancel_event: threading.Event | None = None) -> None:
        super().__init__(request, cancel_event=cancel_event)
        c = self.config
        # Vectors are cheap — a much larger batch than the image default.
        self.batch_size = int(c.get("head_batch_size") or 256)
        # Heads converge in minutes; EMA/backup/resume stay off (minimal surface).
        self.use_ema = False
        # No resolution tail on cached vectors — image_size only fixes the
        # embedding preprocessing signature.
        self.finetune_image_size = 0
        self.finetune_epochs = 0
        self.embedding_cache_dir = c.get("embedding_cache_dir")
        self.backbone_hash = ""
        self.embedding_cache_stats: dict = {}
        self._vectors: dict[str, object] = {}
        self._identity = nn.Identity()
        spec = request.get("backbone_init") or {}
        if not spec.get("checkpoint_path"):
            raise RuntimeError(
                "Backbone head-only training requires backbone_init.checkpoint_path "
                "(an existing backbone checkpoint to freeze)."
            )

    # -- one-time setup ----------------------------------------------------
    def fingerprint_init(self, ctx: TaskContext) -> None:
        # Never backs up or resumes: inert coordinator, always a fresh run.
        ctx.coordinator = BackupCoordinator(backup_dir=None)
        ctx.fingerprint = None
        ctx.resume_state = None

    def create_model(self, ctx: TaskContext, resume_state: dict | None):
        spec = self.request.get("backbone_init")
        backbone = create_model(model_size=self.model_size, pretrained=False, num_classes=0)
        if not apply_backbone_init(backbone, spec):
            raise RuntimeError(
                f"Could not load the source backbone checkpoint {spec.get('checkpoint_path')!r}"
            )
        backbone = backbone.to(ctx.device).eval()
        backbone.requires_grad_(False)
        self.feature_dim = backbone.num_features
        heads = bb._MultiTaskHeads(backbone.num_features, self.vocab).to(ctx.device)
        model = bb._BackboneWithHeads(backbone, heads).to(ctx.device)
        self.model = model
        return model

    def _cache_sample(self, sample) -> dict:
        return {
            "path": sample.path,
            "bucket": (self.image_size, self.image_size),
            "skin_normalise": False,
            "face_bbox": None,
        }

    def pre_loop(self, ctx: TaskContext, model) -> None:
        from bittrainer.embedding_cache import EmbeddingCache
        from bittrainer.model import backbone_feature_hash

        self.backbone_hash = backbone_feature_hash(model.backbone)
        cache_dir = self.embedding_cache_dir or str(
            Path(self.request["candidate_checkpoint_path"]).parent / ".embedding_cache"
        )
        cache = EmbeddingCache(
            cache_dir,
            self.backbone_hash,
            self.feature_dim,
            # Identity, not provenance: head vectors are built square at
            # image_size, distinct from any bucketed cache of the same trunk.
            preproc_sig=f"val_imagenet@{self.image_size}sq",
        )
        all_samples = self.train_samples + self.val_samples
        cache_samples = [self._cache_sample(s) for s in all_samples]
        self._emit(
            "embedding_build",
            f"Caching backbone features for {len(cache_samples)} images "
            f"(era {self.backbone_hash})",
        )

        def _progress(done: int, total: int) -> None:
            self._emit(
                "embedding_build", f"Building feature cache ({done}/{total})",
                step=done, total_steps=total,
            )

        self.embedding_cache_stats = cache.ensure(
            cache_samples, model.backbone, None,
            device=ctx.device, dtype=self.amp_dtype, batch_size=64,
            progress_cb=_progress,
            stop_check=lambda: self.cancel_event is not None and self.cancel_event.is_set(),
        )
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise bb.BackboneTrainingCancelled
        cache.verify(cache_samples, model.backbone, None, device=ctx.device, dtype=self.amp_dtype)

        dropped = 0
        for sample in all_samples:
            vector = cache.get_vector(sample.path, None)
            if vector is None:
                dropped += 1
                continue
            self._vectors[sample.path] = vector
        if dropped:
            logger.warning(
                "BackboneHeadsTask: %d sample(s) have no cached vector and are skipped", dropped
            )
        if not any(s.path in self._vectors for s in self.train_samples):
            raise RuntimeError("No cached feature vectors for any training sample.")

    def create_optimizer(self, ctx: TaskContext, model, eff_bs: int, resume_state: dict | None):
        # Only the heads train; the frozen trunk never enters the optimizer.
        optimizer = make_optimizer(model.heads)
        t_max = max(1, self.epochs)
        scheduler = (
            CosineAnnealingLR(optimizer, T_max=t_max)
            if self.use_cosine
            else LambdaLR(optimizer, lambda _epoch: 1.0)
        )
        self.ema = None
        return optimizer, scheduler, t_max

    def _eval_modules(self):
        # Validation runs on cached vectors — the "backbone" is a pass-through.
        return self._identity, self.model.heads

    # -- per-epoch ---------------------------------------------------------
    def build_loaders(self, ctx: TaskContext, epoch: int, eff_bs: int, resume_info):
        loader_kwargs = {"batch_size": self.batch_size, "collate_fn": bb._collate, "num_workers": 0}
        if self._val_loader is None:
            self._val_loader = DataLoader(
                _VectorDataset(self.val_samples, self._vectors), shuffle=False, **loader_kwargs,
            )
        epoch_samples, plan_stats = self._plan_epoch(epoch)
        self._pos_weight = bb._head_pos_weights(plan_stats) if self.use_pos_weight else None
        train_loader = DataLoader(
            _VectorDataset(epoch_samples, self._vectors), shuffle=True, **loader_kwargs,
        )
        return train_loader, None, 0

    def train_epoch(self, ctx: TaskContext, model, optimizer, train_loader, *, step_callback, boundary_hook, start_batch: int):
        device = ctx.device
        model.heads.train()
        epoch_loss = 0.0
        epoch_batches = 0
        for vectors, binary_labels, group_labels in train_loader:
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise bb.BackboneTrainingCancelled
            if self.max_steps is not None and self.step >= int(self.max_steps):
                self.steps_stop_event.set()
                break
            optimizer.zero_grad(set_to_none=True)
            loss = bb._batch_loss(
                vectors.to(device).float(), model.heads, binary_labels, group_labels, device,
                pos_weight=self._pos_weight,
            )
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            self.step += 1
            epoch_loss += float(loss.detach())
            epoch_batches += 1
            if boundary_hook(epoch_batches) == "stop":
                break
        return epoch_loss / max(epoch_batches, 1)

    # -- finalisation ------------------------------------------------------
    def finalize(self, ctx: TaskContext, model, best: BestTracker, epochs_completed: int) -> dict:
        self._emit("validating", "Validating retrained heads")
        if self.val_samples and self.best_heads_state is not None:
            heads_state = self.best_heads_state
            validation_metrics = self.best_metrics
            validation_score = self.best_score
        else:
            heads_state = {
                k: v.detach().cpu().clone() for k, v in self.model.heads.state_dict().items()
            }
            validation_metrics = self.validation_metrics
            validation_score = self.validation_score

        from safetensors.torch import load_file, save_file

        # Trunk tensors are byte-copied from the source checkpoint (never
        # retrained here); any heads.* the source carried are replaced.
        source_path = str((self.request.get("backbone_init") or {}).get("checkpoint_path"))
        state = {
            key: value
            for key, value in load_file(source_path).items()
            if not key.startswith("heads.")
        }
        for key, value in (heads_state or {}).items():
            state[f"heads.{key}"] = value.detach().to("cpu")

        candidate_path = Path(self.request["candidate_checkpoint_path"])
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self._candidate_metadata(validation_score, validation_metrics)
        metadata["head_only_retrain"] = "1"
        metadata["source_backbone_checkpoint"] = source_path
        save_file(
            state,
            str(candidate_path),
            metadata={
                key: bb._stringify(value) for key, value in metadata.items() if value is not None
            },
        )
        self._emit(
            "saving", "Retrained-heads candidate checkpoint written",
            candidate_checkpoint_path=str(candidate_path), validation_score=validation_score,
        )
        return {
            "candidate_checkpoint_path": str(candidate_path),
            "validation_score": float(validation_score),
            "validation_metrics": validation_metrics,
            "heads": self.request.get("heads") or {},
            "release_blocking": bool(self.request.get("release_blocking")),
            "epochs_completed": int(epochs_completed),
            "best_epoch": int(best.best_epoch + 1) if self.val_samples else int(epochs_completed),
            "mode": "backbone_head_only",
            "backbone_hash": self.backbone_hash,
            "embedding_cache_stats": self.embedding_cache_stats,
        }
