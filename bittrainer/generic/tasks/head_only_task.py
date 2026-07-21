"""Head-only (cached-feature probe) training as a :class:`TrainingTask`
(Bitcrush ISSUE-0542 Step 4).

Head-only is not an epoch-loop trainer: it freezes the backbone, builds+verifies
an :class:`EmbeddingCache` keyed by the backbone-feature hash, runs the
softness + __none__-oversample probes on cached features, evaluates the trained
candidate on the val images, and resolves it through the same
``_compare_promote_finalize`` path as full fine-tune. It therefore rides the
generic skeleton with an EMPTY epoch loop (``LoopSpec(max_epochs=0)``): all the
real work happens in :meth:`pre_loop`, and :meth:`finalize` does the candidate
save + promotion.

Head-only has no backup / resume (out of scope): the coordinator is inert
(``backup_dir=None``, no ``pause_event``), so the generic pause / resume envelope
never fires. Pause / stop are honoured internally exactly as before — a set
event makes the run cancel (behaves like ``stop_now``) and return the
``{"cancelled": True, "mode": "head_only"}`` partial result.

The group helpers are reached through the ``gt`` module alias so their existing
test / monkeypatch seams keep firing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

import bittrainer.group_trainer as gt
from bittrainer.embedding_cache import EmbeddingCache
from bittrainer.generic.task import BestTracker, LoopSpec, TaskContext, TrainingTask
from bittrainer.group_dataset import build_group_bucket_sampler
from bittrainer.model import backbone_feature_hash
from bittrainer.training_state import BackupCoordinator

logger = logging.getLogger(__name__)

_THIN_CLASS_THRESHOLD = 20


def _thin_class_warnings(class_counts: dict[int, int], class_names: list[str]) -> list[dict]:
    warnings: list[dict] = []
    for idx, count in class_counts.items():
        if 0 < count < _THIN_CLASS_THRESHOLD:
            name = class_names[idx] if 0 <= idx < len(class_names) else str(idx)
            warnings.append({"class_index": idx, "class_name": name, "count": count})
    return warnings


class HeadOnlyTask(TrainingTask):
    """Drives ``GenericTrainer`` for a cached-feature head probe (no epoch loop)."""

    trainer_name = "head_only"

    def __init__(self, config: gt.GroupTrainConfig) -> None:
        self.config = config
        # Populated across the lifecycle hooks.
        self.train_ds = None
        self.val_ds = None
        self.smart_cache = None
        self.head_hidden: int | None = None
        self.class_counts: dict = {}
        self.total_raw = 0
        self.thin: list[dict] = []
        self.backbone_hash: str = ""
        self.stats: Any = None
        self.probe: dict | None = None
        self.candidate_metrics: dict = {}
        self.candidate_macro_f1 = 0.0
        self.candidate_qwk = -1.0
        self.candidate_path: Path | None = None
        self._val_loader = None
        self._cancelled = False

    def _stop(self, ctx: TaskContext) -> bool:
        """A stop / stop-now / pause all cancel head-only (no backup/resume)."""
        return bool(
            (ctx.stop_event is not None and ctx.stop_event.is_set())
            or (ctx.stop_now_event is not None and ctx.stop_now_event.is_set())
            or (ctx.pause_event is not None and ctx.pause_event.is_set())
        )

    # -- one-time setup ----------------------------------------------------
    def make_context(self, progress_callback, stop_event, stop_now_event, pause_event) -> TaskContext:
        from bittrainer.progress import ProgressEmitter
        from bittrainer.runtime import configure_cuda_backend
        from bittrainer.smart_cache import _noop_callback

        config = self.config
        em = ProgressEmitter(progress_callback or config.progress_callback or _noop_callback)
        device = torch.device(config.device)
        dtype = gt._get_dtype(config.dtype)
        configure_cuda_backend()
        group_folder = Path(config.group_folder)
        checkpoint_dir = (
            Path(config.checkpoint_dir) if config.checkpoint_dir else group_folder / "checkpoints"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return TaskContext(
            device=device, dtype=dtype, em=em, cb=em.raw, checkpoint_dir=checkpoint_dir,
            stop_event=stop_event, stop_now_event=stop_now_event, pause_event=pause_event,
        )

    def fingerprint_init(self, ctx: TaskContext) -> None:
        # Head-only never backs up or resumes: an inert coordinator (no backup
        # dir, no pause event) means the generic pause/resume envelope stays off
        # and the run is always fresh. Cancellation is handled internally.
        ctx.coordinator = BackupCoordinator(backup_dir=None)
        ctx.fingerprint = None
        ctx.resume_state = None

    def loop_spec(self) -> LoopSpec:
        return LoopSpec(max_epochs=0, patience=0)

    def prepare_data(self, ctx: TaskContext) -> None:
        self.train_ds, self.val_ds, self.smart_cache, _bucket_counts = (
            gt._prepare_datasets_and_cache(self.config, cb=ctx.cb, stop_event=ctx.stop_event)
        )

    def create_model(self, ctx: TaskContext, resume_state: dict | None):
        config = self.config
        self.head_hidden = config.probe_mlp_hidden if config.probe_head == "mlp" else None
        ctx.cb({"type": "training_progress", "stage": "preparing", "status_text": "Loading model"})
        return gt._create_or_warmstart_model(
            config, device=ctx.device, dtype=ctx.dtype,
            head_hidden_size=self.head_hidden, checkpoint_dir=ctx.checkpoint_dir,
        )

    def pre_loop(self, ctx: TaskContext, model) -> None:
        config = self.config
        device, dtype = ctx.device, ctx.dtype
        cb = ctx.cb
        group_folder = Path(config.group_folder)

        self.backbone_hash = backbone_feature_hash(model)
        pooled_dim = int(getattr(model, "num_features", 0))
        embed_root = config.embedding_cache_dir or str(group_folder / ".embedding_cache")
        embed_cache = EmbeddingCache(embed_root, self.backbone_hash, pooled_dim)

        all_samples = self.train_ds.samples + self.val_ds.samples

        def _build_progress(done: int, total: int) -> None:
            cb({
                "type": "training_progress", "stage": "embedding_build",
                "status_text": f"Building feature cache ({done}/{total})",
                "step": done, "total_steps": total,
            })

        cb({"type": "training_progress", "stage": "embedding_build",
            "status_text": f"Caching backbone features (era {self.backbone_hash})"})
        self.stats = embed_cache.ensure(
            all_samples, model, self.smart_cache, device=device, dtype=dtype,
            batch_size=config.batch_size or 64, progress_cb=_build_progress,
            stop_check=lambda: self._stop(ctx),
        )
        logger.info("EmbeddingCache: %s", self.stats)

        if self._stop(ctx):
            cb({"type": "training_cancelled", "stage": "caching",
                "status_text": "Cancelled before probe"})
            self._cancelled = True
            return

        # Mandatory: fail loud on a stale/bad cache before any probe consumes it.
        checked = embed_cache.verify(all_samples, model, self.smart_cache, device=device, dtype=dtype)
        logger.info("EmbeddingCache.verify: %d vectors matched live forward", checked)

        self.class_counts = self.train_ds.get_class_counts()
        self.total_raw = sum(self.class_counts.values())
        self.thin = _thin_class_warnings(self.class_counts, config.class_names)
        if self.thin:
            names = ", ".join(f"{w['class_name']} ({w['count']})" for w in self.thin)
            cb({"type": "training_progress", "stage": "training",
                "status_text": f"Thin classes (probe may overfit): {names}"})

        cb({"type": "training_progress", "stage": "training",
            "status_text": f"Training head probe ({config.probe_head})"})
        none_index = gt._resolve_none_index(config.class_names)
        probe = gt._run_auto_softness_probe(
            model, config, embed_cache, self.smart_cache,
            self.train_ds.samples, self.val_ds.samples,
            device=device, none_index=none_index, cb=cb, stop_event=ctx.stop_event,
        )
        # Second pre-training sweep: __none__ oversample off vs 1.5x. When it runs
        # it leaves the head in the selected state and supersedes the softness probe
        # as the candidate to evaluate.
        oversample_probe = gt._run_auto_oversample_probe(
            model, config, embed_cache, self.smart_cache,
            self.train_ds.samples, self.val_ds.samples,
            device=device, none_index=none_index, cb=cb, stop_event=ctx.stop_event,
        )
        if oversample_probe:
            probe = oversample_probe
        self.probe = probe

        # Evaluate the trained candidate (frozen backbone + warm head) on the val
        # images, so it competes with the incumbent on identical _evaluate footing.
        # A one-shot val pass (num_workers=0 — small, single-process, no spawn).
        collate_fn = gt._collate_multilabel_batch if config.multi_label else gt._collate_bucket_batch
        val_bs = config.batch_size or 32
        val_loader = DataLoader(
            self.val_ds,
            batch_sampler=build_group_bucket_sampler(self.val_ds, batch_size=val_bs),
            collate_fn=collate_fn, num_workers=0,
        )
        self._val_loader = val_loader
        model.eval()
        self.candidate_metrics = gt._evaluate(
            model, val_loader, config.num_classes, device, dtype,
            multi_label=config.multi_label, ordinal=config.ordinal,
            none_index=none_index,
        )
        self.candidate_macro_f1 = self.candidate_metrics.get("macro_f1", 0.0)
        self.candidate_qwk = self.candidate_metrics.get("qwk", -1.0)

        candidate_path = ctx.checkpoint_dir / "candidate.pt"
        ckpt_meta: dict[str, Any] = {
            "state_dict": model.state_dict(),
            "num_classes": config.num_classes,
            "model_size": config.backbone_variant,
            "class_names": list(config.class_names),
            "validation_metric": gt._primary_validation_metric(config),
            **gt._spatial_ckpt_meta(config),
        }
        if self.head_hidden is not None:
            ckpt_meta["head_hidden_size"] = self.head_hidden
        if config.multi_label:
            ckpt_meta["multi_label"] = True
        torch.save(ckpt_meta, candidate_path)
        self.candidate_path = candidate_path

    # Head-only has no epoch loop / optimizer — the following hooks are inert.
    def resolve_batch_size(self, ctx: TaskContext, model, resume_state: dict | None) -> int:
        return int(self.config.batch_size or 0)

    def create_optimizer(self, ctx: TaskContext, model, eff_bs: int, resume_state: dict | None):
        # No epoch loop runs (max_epochs=0) and backups are disabled, so the core
        # never touches these — placeholders keep the 3-tuple contract.
        return None, None, 0

    def build_loaders(self, ctx: TaskContext, epoch, eff_bs, resume_info):  # pragma: no cover
        raise AssertionError("head-only has no epoch loop")

    def train_epoch(self, ctx, model, optimizer, train_loader, *, step_callback, boundary_hook, start_batch):  # pragma: no cover
        raise AssertionError("head-only has no epoch loop")

    def validate(self, ctx, model, epoch, train_result) -> dict:  # pragma: no cover
        raise AssertionError("head-only has no epoch loop")

    def selection_score(self, metrics: dict) -> float:  # pragma: no cover
        return 0.0

    def save_candidate(self, ctx, model, epoch, metrics, best) -> None:  # pragma: no cover
        raise AssertionError("head-only has no epoch loop")

    # -- finalisation ------------------------------------------------------
    def finalize(self, ctx: TaskContext, model, best: BestTracker, epochs_completed: int) -> dict:
        if self._cancelled:
            return {"cancelled": True, "mode": "head_only"}

        config = self.config
        cb = ctx.cb
        result = gt._compare_promote_finalize(
            config,
            candidate_path=str(self.candidate_path),
            best_metrics=self.candidate_metrics,
            candidate_macro_f1=self.candidate_macro_f1,
            candidate_qwk=self.candidate_qwk,
            best_epoch_display=self.probe["best_epoch"],
            epochs_completed=self.probe["epochs_completed"],
            val_loader=self._val_loader,
            device=ctx.device, dtype=ctx.dtype,
            checkpoint_dir=ctx.checkpoint_dir,
            class_counts=self.class_counts,
            effective_class_counts=self.train_ds.get_effective_class_counts(),
            total_raw=self.total_raw,
            cb=cb,
        )
        result["mode"] = "head_only"
        result["probe_head"] = config.probe_head
        result["thin_class_warnings"] = self.thin
        result["backbone_hash"] = self.backbone_hash
        result["embedding_cache_stats"] = self.stats

        promoted = result.get("promotion_reason") not in (None, "incumbent_wins")
        primary_metric = gt._primary_validation_metric(config)
        final_score = result.get("selected_validation_score")
        if final_score is None:
            final_score = gt._metric_score({
                "macro_f1": result.get("best_val_macro_f1"),
                "qwk": result.get("best_val_qwk"),
                "none_f1": result.get("final_val_none_f1"),
            }, config)
        cb({
            "type": "epoch_complete", "stage": "training",
            "status_text": (
                f"Head training complete (val {primary_metric} {final_score:.3f}) — "
                f"{'deployed' if promoted else 'kept existing model'}"
            ),
            "epoch": self.probe["epochs_completed"], "max_epochs": config.head_max_epochs,
            "val_macro_f1": result.get("best_val_macro_f1"),
            "val_qwk": result.get("qwk"),
            "validation_metric": primary_metric,
            "per_class_f1": result.get("per_class_f1", {}),
            "best_val_macro_f1": result.get("best_val_macro_f1"),
            "best_val_qwk": result.get("best_val_qwk"),
            "best_epoch": self.probe["best_epoch"],
        })
        return result
