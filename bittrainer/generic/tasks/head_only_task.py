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

import copy
import logging
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

import bittrainer.group_trainer as gt
from bittrainer.cached_eval import CachedValScorer, val_inputs_are_cache_inputs
from bittrainer.dataset import resolution_ckpt_meta
from bittrainer.embedding_cache import EmbeddingCache
from bittrainer.generic.task import BestTracker, LoopSpec, TaskContext, TrainingTask
from bittrainer.group_dataset import build_group_bucket_sampler, labelled_samples
from bittrainer.head_probe import prepare_head_probe_tensors
from bittrainer.model import backbone_feature_hash, backbone_state_hash
from bittrainer.ordinal_threshold import (
    ORDINAL_HEADS,
    SOFTMAX_HEAD,
    THRESHOLD_HEAD,
    install_ordinal_threshold_head,
)
from bittrainer.regions import REGION_HEAD_PREFIX, has_region_head, region_head_ckpt_meta
from bittrainer.training_state import (
    BackupCoordinator,
    capture_rng_states,
    restore_rng_states,
)
from bittrainer.view_calibration import finalize_region_views

logger = logging.getLogger(__name__)

_THIN_CLASS_THRESHOLD = 20

#: How the head search names each head in its progress lines (Engine's words).
HEAD_LABELS = {SOFTMAX_HEAD: "Standard", THRESHOLD_HEAD: "Ordered"}

#: Config fields the head-probe sweeps resolve. Each searched head tunes them
#: afresh from the same starting values; the winner's choices stick.
_SWEEP_FIELDS = (
    "ordinal_sigma",
    "label_smoothing",
    "oversample_none",
    "selected_softness_kind",
    "selected_softness_value",
    "soft_label_tuning_metric",
    "soft_label_tuning_results",
    "soft_label_tuning_elapsed_ms",
    "selected_oversample_none",
    "oversample_tuning_metric",
    "oversample_tuning_results",
    "oversample_tuning_elapsed_ms",
)


def head_search_enabled(config: gt.GroupTrainConfig) -> bool:
    """True when the run searches softmax vs threshold (Bitcrush ISSUE-1496):
    asked for, and the threshold head can train this group."""
    return (
        bool(config.ordinal_head_search)
        and bool(config.ordinal)
        and not config.multi_label
        and not config.cell_masks
        and not config.use_focal
    )


def pick_head(arms: list[dict]) -> dict:
    """The searched head with the higher validation score; softmax (the
    default head) on a tie."""
    best: dict | None = None
    for arm in arms:
        if (
            best is None
            or arm["score"] > best["score"]
            or (arm["score"] == best["score"] and arm["head"] == SOFTMAX_HEAD)
        ):
            best = arm
    assert best is not None
    return best


def _sweep_state(config: gt.GroupTrainConfig) -> dict:
    return {name: copy.copy(getattr(config, name)) for name in _SWEEP_FIELDS}


def _apply_sweep_state(config: gt.GroupTrainConfig, state: dict) -> None:
    for name, value in state.items():
        setattr(config, name, copy.copy(value))


def settle_head_only_region_head(model: torch.nn.Module, incumbent_path: str | Path) -> str:
    """Decide whether a head-only candidate may ship a region head.

    Head-only training never trains the region head, so the only head worth
    shipping is the incumbent's own, and only while it still sits on the trunk
    it was trained with (Bitcrush ISSUE-0993). Returns:

    * ``"none"`` — the model has no region head (an ordinary group);
    * ``"kept"`` — the incumbent's head was warm-started onto the same trunk;
    * ``"needs_full_train"`` — no usable incumbent head; the head is removed
      from *model* so a random or mismatched proposer is never deployed.
    """
    if not has_region_head(model):
        return "none"
    try:
        data = torch.load(str(incumbent_path), map_location="cpu", weights_only=True)
        state = data["state_dict"] if isinstance(data, dict) and "state_dict" in data else data
        own = model.state_dict()
        head_keys = [k for k in state if k.startswith(REGION_HEAD_PREFIX)]
        same_head = bool(head_keys) and all(
            k in own and own[k].shape == state[k].shape for k in head_keys
        )
        if same_head and backbone_state_hash(state) == backbone_feature_hash(model):
            return "kept"
    except (OSError, RuntimeError, KeyError, EOFError, AttributeError, TypeError):
        logger.info("Head-only: no usable incumbent region head at %s", incumbent_path)
    del model.region_head
    return "needs_full_train"


def _val_pass_progress(cb: Callable[[dict], None], label: str) -> Callable[[int, int], None]:
    """Rate-limited per-batch frames for a full val pass (ISSUE-1110: job
    1593's pass emitted none for 20+ minutes and the UI looked frozen)."""
    last = [0.0]

    def _report(done: int, total: int) -> None:
        now = time.monotonic()
        if done < total and now - last[0] < 0.5:
            return
        last[0] = now
        cb({"type": "training_progress", "stage": "evaluating",
            "status_text": f"{label} ({done}/{total} batches)",
            "step": done, "total_steps": total})

    return _report


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

    def __init__(self, config: gt.GroupTrainConfig, *, cached_val_eval: bool = True) -> None:
        self.config = config
        # ISSUE-1110: score the candidate (and a same-trunk incumbent) on the
        # cached val vectors instead of running the frozen trunk over every val
        # image. ``False`` forces the full passes (tests compare the two).
        self.cached_val_eval = bool(cached_val_eval)
        self.cached_val = None
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
        # Softmax vs threshold head search (Bitcrush ISSUE-1496).
        self.search_heads = head_search_enabled(config)
        self.head_search: list[dict] = []
        # Crop-assisted groups (Bitcrush ISSUE-0993).
        self.region_manifest = None
        self.region_proposer = "none"

    def _region_meta(self, model) -> dict:
        """Checkpoint metadata for the model's region head; empty without one."""
        names = self.region_manifest.type_names if self.region_manifest is not None else None
        return region_head_ckpt_meta(model, names)

    def _finalize_regions(self, ctx: TaskContext, result: dict) -> None:
        """Crop-assisted groups: re-fit the crop view and the proposer threshold
        on whichever checkpoint promotion left in place. A no-op otherwise."""
        if self.region_manifest is None or not isinstance(result, dict):
            return
        if result.get("promotion_reason") == "incumbent_wins":
            # Bitcrush ISSUE-1501: a kept incumbent's checkpoint is never rewritten.
            result["region_training"] = {"status": "incumbent_kept"}
            return
        result["region_training"] = finalize_region_views(
            self.config,
            checkpoint_path=result.get("checkpoint_path"),
            manifest=self.region_manifest,
            val_ds=self.val_ds,
            device=ctx.device,
            dtype=ctx.dtype,
            batch_size=int(self.config.batch_size or 32),
            cb=ctx.cb,
        )

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
        # Refuse impossible backbone settings up front (no-op for ConvNeXt).
        notes = gt.validate_backbone_config(config, mode="head_only")
        notes += gt.validate_ordinal_head(config)
        em = ProgressEmitter(progress_callback or config.progress_callback or _noop_callback)
        for note in notes:
            em.raw({"type": "training_progress", "stage": "preparing", "status_text": note})
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
        self.region_manifest = gt.resolve_region_manifest(self.config)

    def create_model(self, ctx: TaskContext, resume_state: dict | None):
        config = self.config
        self.head_hidden = config.probe_mlp_hidden if config.probe_head == "mlp" else None
        if self.search_heads:
            # Both searched tails start fresh on the kept trunk: the softmax one
            # is built here, the threshold one is installed by the search.
            config.ordinal_head = SOFTMAX_HEAD
            config.fresh_head = True
        ctx.cb({"type": "training_progress", "stage": "preparing", "status_text": "Loading model"})
        model = gt._create_or_warmstart_model(
            config, device=ctx.device, dtype=ctx.dtype,
            head_hidden_size=self.head_hidden, checkpoint_dir=ctx.checkpoint_dir,
        )
        # A probe trains the classifier head only. Keep the incumbent's region
        # head when the trunk under it is unchanged; otherwise ship none.
        self.region_proposer = settle_head_only_region_head(
            model, ctx.checkpoint_dir / config.best_model_name
        )
        if self.region_proposer == "needs_full_train":
            ctx.cb({
                "type": "training_progress", "stage": "preparing",
                "status_text": "Region proposer needs a full train — head-only keeps none",
            })
        return model

    def pre_loop(self, ctx: TaskContext, model) -> None:
        config = self.config
        device, dtype = ctx.device, ctx.dtype
        cb = ctx.cb
        group_folder = Path(config.group_folder)

        self.backbone_hash = backbone_feature_hash(model)
        pooled_dim = int(getattr(model, "num_features", 0))
        embed_root = config.embedding_cache_dir or str(group_folder / ".embedding_cache")
        # Packed (ISSUE-1110): the run's vectors stay in memory after ensure()
        # and one consolidated pack serves the next run / comparison candidate.
        embed_cache = EmbeddingCache(
            embed_root, self.backbone_hash, pooled_dim,
            preproc_sig=gt._group_embedding_preproc_sig(config),
            packed=True,
        )

        # Detection-only rows (crop-assisted groups) carry no class for a probe.
        all_samples = labelled_samples(self.train_ds.samples) + self.val_ds.samples

        def _build_progress(done: int, total: int) -> None:
            cb({
                "type": "training_progress", "stage": "embedding_build",
                "status_text": f"Building feature cache ({done}/{total})",
                "step": done, "total_steps": total,
            })

        def _check_progress(done: int, total: int) -> None:
            cb({
                "type": "training_progress", "stage": "embedding_build",
                "status_text": f"Checking feature cache ({done}/{total})",
                "step": done, "total_steps": total,
            })

        cb({"type": "training_progress", "stage": "embedding_build",
            "status_text": f"Caching backbone features (era {self.backbone_hash})"})
        self.stats = embed_cache.ensure(
            all_samples, model, self.smart_cache, device=device, dtype=dtype,
            batch_size=config.embedding_batch_size, progress_cb=_build_progress,
            stop_check=lambda: self._stop(ctx),
            prune=bool(config.embedding_cache_prune),
            check_progress_cb=_check_progress,
        )
        logger.info("EmbeddingCache: %s", self.stats)

        if self._stop(ctx):
            cb({"type": "training_cancelled", "stage": "caching",
                "status_text": "Cancelled before probe"})
            self._cancelled = True
            return

        # Mandatory: fail loud on a stale/bad cache before any probe consumes it.
        cb({"type": "training_progress", "stage": "embedding_build",
            "status_text": "Verifying feature cache against a live forward"})
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
        # Epoch-shaped (replication-expanded) train rows: ``samples`` is the
        # stable, de-replicated base list since ISSUE-0859.
        train_epoch_samples = labelled_samples(self.train_ds.epoch_samples())
        # Load the cached features ONCE for both sweeps (ISSUE-1110: each sweep
        # used to re-read every vector from its own .npy file).
        tensors = prepare_head_probe_tensors(
            embed_cache, self.smart_cache, train_epoch_samples, self.val_ds.samples,
            config, cb=cb,
        )
        collate_fn = gt._collate_multilabel_batch if config.multi_label else gt._collate_bucket_batch
        val_bs = config.batch_size or 32
        self._val_loader = DataLoader(
            self.val_ds,
            batch_sampler=build_group_bucket_sampler(self.val_ds, batch_size=val_bs),
            collate_fn=collate_fn, num_workers=0,
        )
        if self.search_heads:
            probe = self._search_heads(ctx, model, embed_cache, tensors,
                                       train_epoch_samples, none_index)
            if probe is None:
                cb({"type": "training_cancelled", "stage": "training",
                    "status_text": "Cancelled during the head search"})
                self._cancelled = True
                return
        else:
            probe = self._train_head(ctx, model, embed_cache, tensors,
                                     train_epoch_samples, none_index)
            self.candidate_metrics = self._score_head(ctx, model, tensors[2], none_index)
        self.probe = probe
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
            **resolution_ckpt_meta(config.train_resolution),
            **self._region_meta(model),
            # The trunk is frozen: SigLIP records the embedding-cache hash it
            # already computed; ConvNeXt gets nothing (byte-identical).
            **gt._architecture_ckpt_meta(config, model, backbone_hash=self.backbone_hash),
        }
        if self.head_hidden is not None:
            ckpt_meta["head_hidden_size"] = self.head_hidden
        if config.multi_label:
            ckpt_meta["multi_label"] = True
        torch.save(ckpt_meta, candidate_path)
        self.candidate_path = candidate_path

    def _train_head(self, ctx: TaskContext, model, embed_cache, tensors,
                    train_samples: list[dict], none_index: int) -> dict:
        """The softness sweep, then the __none__ oversample sweep, on the head
        ``model`` carries now. Returns the probe dict of the head left in place."""
        config = self.config
        probe = gt._run_auto_softness_probe(
            model, config, embed_cache, self.smart_cache,
            train_samples, self.val_ds.samples,
            device=ctx.device, none_index=none_index, cb=ctx.cb, stop_event=ctx.stop_event,
            tensors=tensors,
        )
        # Second pre-training sweep: __none__ oversample off vs 1.5x. When it runs
        # it leaves the head in the selected state and supersedes the softness probe
        # as the candidate to evaluate.
        oversample_probe = gt._run_auto_oversample_probe(
            model, config, embed_cache, self.smart_cache,
            train_samples, self.val_ds.samples,
            device=ctx.device, none_index=none_index, cb=ctx.cb, stop_event=ctx.stop_event,
            tensors=tensors,
        )
        return oversample_probe or probe

    def _score_head(self, ctx: TaskContext, model, x_val: torch.Tensor, none_index: int) -> dict:
        """Evaluate the trained head (frozen backbone) on the val set, so it
        competes with the incumbent on identical _evaluate footing."""
        config = self.config
        model.eval()
        if (
            self.cached_val is None
            and self.cached_val_eval
            and x_val.shape[0] == len(self.val_ds.samples)
            and val_inputs_are_cache_inputs(self.val_ds)
        ):
            # Every val image has its cached vector (the gather skips none), and
            # the val set feeds the model exactly the tensors the cache embedded:
            # the head on those vectors IS the full pass (ISSUE-1110).
            self.cached_val = CachedValScorer(
                model, x_val, self.backbone_hash, multi_label=config.multi_label,
            )
        if self.cached_val is not None:
            ctx.cb({"type": "training_progress", "stage": "evaluating",
                    "status_text": "Scoring the trained head on cached validation features"})
            return self.cached_val.evaluate(
                model, self._val_loader, config.num_classes, ctx.device, ctx.dtype,
                ordinal=config.ordinal, none_index=none_index,
            )
        return gt._evaluate(
            model, self._val_loader, config.num_classes, ctx.device, ctx.dtype,
            multi_label=config.multi_label, ordinal=config.ordinal,
            none_index=none_index,
            progress=_val_pass_progress(ctx.cb, "Scoring the trained head on validation images"),
        )

    def _search_heads(self, ctx: TaskContext, model, embed_cache, tensors,
                      train_samples: list[dict], none_index: int) -> dict | None:
        """Train a softmax and a threshold tail and keep the better (ISSUE-1496).

        Both start from the same fresh tail, RNG state and sweep settings; each
        runs the full softness + oversample sweeps and is scored like the
        candidate always is. The winner's head, sweep choices and
        ``ordinal_head`` are left on the model and config, and its metrics
        become the candidate's. ``None`` when a stop / pause arrived.
        """
        config = self.config
        cb = ctx.cb
        base_pre_logits = copy.deepcopy(model.head.pre_logits)
        base_fc = copy.deepcopy(model.head.fc)
        start_sweep = _sweep_state(config)
        start_rng = capture_rng_states(ctx.device)
        primary = gt._primary_validation_metric(config)
        arms: list[dict] = []
        for idx, head in enumerate(ORDINAL_HEADS, start=1):
            if self._stop(ctx):
                return None
            _apply_sweep_state(config, start_sweep)
            restore_rng_states(start_rng, ctx.device)
            config.ordinal_head = head
            model.head.pre_logits = copy.deepcopy(base_pre_logits)
            model.head.fc = copy.deepcopy(base_fc)
            if head == THRESHOLD_HEAD:
                install_ordinal_threshold_head(model, config.num_classes, none_index)
            label = HEAD_LABELS[head]
            cb({"type": "training_progress", "stage": "head_search",
                "status_text": f"Training the {label} head ({idx}/{len(ORDINAL_HEADS)})",
                "step": idx, "total_steps": len(ORDINAL_HEADS), "ordinal_head": head})
            probe = self._train_head(ctx, model, embed_cache, tensors, train_samples, none_index)
            if self._stop(ctx):
                return None
            metrics = self._score_head(ctx, model, tensors[2], none_index)
            score = float(gt._metric_score(metrics, config))
            arms.append({
                "head": head, "score": score, "metrics": metrics, "probe": probe,
                "pre_logits": model.head.pre_logits, "fc": model.head.fc,
                "sweep": _sweep_state(config),
            })
            cb({"type": "training_progress", "stage": "head_search",
                "status_text": f"{label} head: val {primary} {score:.3f}",
                "step": idx, "total_steps": len(ORDINAL_HEADS), "ordinal_head": head,
                "selected_validation_score": score})

        winner = pick_head(arms)
        model.head.pre_logits = winner["pre_logits"]
        model.head.fc = winner["fc"]
        _apply_sweep_state(config, winner["sweep"])
        config.ordinal_head = winner["head"]
        self.head_search = [
            {
                "head": arm["head"],
                "score": arm["score"],
                "qwk": arm["metrics"].get("qwk"),
                "macro_f1": arm["metrics"].get("macro_f1"),
                "selected": arm is winner,
            }
            for arm in arms
        ]
        self.candidate_metrics = winner["metrics"]
        cb({"type": "training_progress", "stage": "head_search",
            "status_text": f"Kept the {HEAD_LABELS[winner['head']]} head "
                           f"(val {primary} {winner['score']:.3f})",
            "step": len(ORDINAL_HEADS), "total_steps": len(ORDINAL_HEADS),
            "ordinal_head": winner["head"], "selected_validation_score": winner["score"]})
        return winner["probe"]

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
            cached_val=self.cached_val,
        )
        result["mode"] = "head_only"
        result["probe_head"] = config.probe_head
        result["thin_class_warnings"] = self.thin
        result["backbone_hash"] = self.backbone_hash
        result["embedding_cache_stats"] = self.stats
        if self.region_proposer != "none":
            result["region_proposer"] = self.region_proposer
        self._finalize_regions(ctx, result)

        promoted = result.get("promotion_reason") not in (None, "incumbent_wins")
        if self.head_search:
            result["head_search"] = self.head_search
            if promoted:
                # The searched winner shipped: name it either way, so Engine
                # adopts a softmax winner over a threshold group too.
                result["ordinal_head"] = config.ordinal_head
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
