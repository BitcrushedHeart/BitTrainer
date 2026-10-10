"""Pairwise reward training as a :class:`TrainingTask` (Bitcrush Optimise P8).

Lifecycle on the generic skeleton:

* ``prepare_data`` loads the pair manifest, resolves the SigLIP2 image trunk and
  the matching text tower (falling back to ``siglip2_base`` together), checks
  their embedding widths agree, embeds every image and caption once through the
  cache, then loads or computes the group + source-identity aware split and
  persists it. The trunks are then dropped — only vectors stay in memory.
* ``create_model`` builds the :class:`~bittrainer.reward_head.RewardHead`.
* each epoch trains the head with ``softplus(-(r_c - r_r))`` over a shuffled
  pass of the train pairs (one item per pair: equal per-pair sampling), then
  ``validate`` scores the HELD-OUT pairs — preference accuracy, mean margin and
  per-type accuracy. With a single connected component there is no held-out
  set; the metrics are then reported as ``None`` and selection falls back to
  the train loss, never to train agreement dressed up as validation.
* ``finalize`` fits the fixed 0-100 normalisation mapping over the rewards of
  every pair (both sides) under the best head and writes it into the
  checkpoint next to ``text_model``, ``reward_head_arch``,
  ``embed_preproc_sig``, ``val_pref_acc`` and the per-type / split provenance.

There is no backup/resume (an inert coordinator): a set ``pause_event`` is
treated like ``stop_now``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from bittrainer.generic.task import BestTracker, LoopSpec, TaskContext, TrainingTask
from bittrainer.pair_dataset import (
    ImageEmbedder,
    PairEmbeddingDataset,
    PairRecord,
    SplitResult,
    embed_records,
    group_aware_split,
    load_pair_manifest,
    load_split,
    per_type_counts,
    save_split,
)
from bittrainer.progress import ProgressEmitter, Stage
from bittrainer.reward_head import (
    RewardHead,
    fit_norm_mapping,
    pairwise_ranking_loss,
    preference_accuracy,
    preference_margin,
)
from bittrainer.training_state import BackupCoordinator

logger = logging.getLogger(__name__)

TRAINER_NAME = "pairwise_reward"
CHECKPOINT_NAME = "reward_best.pt"
SPLIT_NAME = "split.json"
CHECKPOINT_SCHEMA = 1

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def _is_set(event: Any | None) -> bool:
    return event is not None and bool(getattr(event, "is_set", lambda: False)())


class PairwiseRewardTask(TrainingTask):
    trainer_name = TRAINER_NAME

    def __init__(self, config) -> None:
        self.config = config
        self.records: list[PairRecord] = []
        self.split: SplitResult | None = None
        self.split_path: Path | None = None
        self.dataset: PairEmbeddingDataset | None = None
        self.train_tensors: dict[str, torch.Tensor] = {}
        self.val_tensors: dict[str, torch.Tensor] = {}
        self.embed_dim = 0
        self.image_model_size = config.image_model_size
        self.text_model = ""
        self.fell_back = False
        self.embed_preproc_sig = ""
        self.cache_stats: dict[str, int] = {}
        self.checkpoint_path: Path | None = None
        self.last_train: dict[str, Any] = {}
        self.last_metrics: dict[str, Any] = {}
        self.epochs_trained = 0

    # -- one-time setup ----------------------------------------------------
    def make_context(self, progress_callback, stop_event, stop_now_event, pause_event) -> TaskContext:
        cb = progress_callback or (lambda _msg: None)
        em = ProgressEmitter(cb)
        requested = str(self.config.device)
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; reward training runs on CPU")
            requested = "cpu"
        device = torch.device(requested)
        dtype = _DTYPES.get(str(self.config.dtype), torch.float32)
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        # No backup/resume: a pause behaves like stop_now (see module docstring).
        effective_stop_now = _AnyEvent(stop_now_event, pause_event)
        return TaskContext(
            device=device,
            dtype=dtype,
            em=em,
            cb=em.raw,
            checkpoint_dir=out,
            stop_event=stop_event,
            stop_now_event=effective_stop_now,
            pause_event=None,
        )

    def fingerprint_init(self, ctx: TaskContext) -> None:
        ctx.coordinator = BackupCoordinator(backup_dir=None, pause_event=None, backup_every_steps=0, cb=ctx.cb)
        ctx.fingerprint = {"trainer": TRAINER_NAME}
        ctx.resume_state = None

    def loop_spec(self) -> LoopSpec:
        return LoopSpec(
            max_epochs=int(self.config.max_epochs),
            patience=int(self.config.patience),
            selection_min_delta=1e-6,
        )

    def prepare_data(self, ctx: TaskContext) -> None:
        cfg = self.config
        ctx.em.stage(Stage.scanning, "Reading the pair manifest")
        self.records = load_pair_manifest(cfg.manifest_path)

        ctx.em.stage(Stage.loading_model, "Loading the SigLIP2 image trunk and text tower")
        from bittrainer.model import create_model
        from bittrainer.siglip_text import check_embedding_parity, resolve_reward_towers

        image_size, tower = resolve_reward_towers(cfg.image_model_size, fallback=cfg.fallback_to_base)
        self.image_model_size = image_size
        self.text_model = tower.model_id
        self.fell_back = tower.fell_back
        trunk = create_model(model_size=image_size, pretrained=True, num_classes=0, dtype=ctx.dtype)
        trunk = trunk.to(ctx.device).eval()
        tower.to(ctx.device)
        check_embedding_parity(trunk, tower)
        embedder = ImageEmbedder(trunk, image_size=cfg.image_size, device=ctx.device, dtype=ctx.dtype)
        self.embed_preproc_sig = embedder.preproc_sig

        ctx.em.stage(Stage.embedding_build, "Embedding images and captions")

        def progress(done: int, total: int, what: str) -> None:
            ctx.em.step(done, total, f"Embedding {what} ({done}/{total})")

        embeddings = embed_records(
            self.records,
            embedder,
            tower,
            cache_dir=cfg.embedding_cache_dir,
            batch_size=int(cfg.embed_batch_size),
            progress=progress,
        )
        self.cache_stats = dict(embeddings.stats)
        self.embed_dim = embeddings.embed_dim
        # The trunks have done their job; keep only the vectors.
        del trunk, tower, embedder
        if ctx.device.type == "cuda":  # pragma: no cover - GPU only
            torch.cuda.empty_cache()

        ctx.em.stage(Stage.preparing, "Splitting pairs by group and source identity")
        existing = Path(cfg.split_path) if cfg.split_path else None
        if existing is not None and existing.is_file():
            self.split = load_split(existing, self.records)
            self.split_path = existing
        else:
            self.split = group_aware_split(self.records, val_fraction=float(cfg.val_fraction), seed=int(cfg.seed))
            self.split_path = existing if existing is not None else Path(cfg.output_dir) / SPLIT_NAME
            save_split(self.split_path, self.split, self.records)

        self.dataset = PairEmbeddingDataset(self.records, embeddings)
        self.train_tensors = self.dataset.subset(self.split.train)
        self.val_tensors = self.dataset.subset(self.split.val)

    def create_model(self, ctx: TaskContext, resume_state):
        torch.manual_seed(int(self.config.seed))
        head = RewardHead(self.embed_dim, int(self.config.hidden_dim), dropout=float(self.config.dropout))
        return head.to(ctx.device)

    def resolve_batch_size(self, ctx: TaskContext, model, resume_state) -> int:
        return max(1, min(int(self.config.batch_size), len(self.split.train)))

    def create_optimizer(self, ctx: TaskContext, model, eff_bs: int, resume_state):
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(self.config.learning_rate), weight_decay=float(self.config.weight_decay)
        )
        t_max = max(1, int(self.config.max_epochs))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        return optimizer, scheduler, t_max

    # -- per-epoch ---------------------------------------------------------
    def build_loaders(self, ctx: TaskContext, epoch: int, eff_bs: int, resume_info):
        generator = torch.Generator().manual_seed(int(self.config.seed) * 1000 + epoch)
        dataset = TensorDataset(
            self.train_tensors["chosen"], self.train_tensors["rejected"], self.train_tensors["text"]
        )
        loader = DataLoader(dataset, batch_size=eff_bs, shuffle=True, drop_last=False, generator=generator)
        return loader, None, 0

    def train_epoch(self, ctx: TaskContext, model, optimizer, train_loader, *, step_callback, boundary_hook, start_batch):
        model.train()
        total_loss, batches = 0.0, 0
        chosen_scores, rejected_scores = [], []
        for num, (i_c, i_r, t) in enumerate(train_loader, start=1):
            if _is_set(ctx.stop_now_event):
                break
            i_c, i_r, t = i_c.to(ctx.device), i_r.to(ctx.device), t.to(ctx.device)
            optimizer.zero_grad(set_to_none=True)
            r_c, r_r = model(i_c, t), model(i_r, t)
            loss = pairwise_ranking_loss(r_c, r_r)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
            chosen_scores.append(r_c.detach())
            rejected_scores.append(r_r.detach())
            if boundary_hook is not None:
                boundary_hook(num)
        if chosen_scores:
            r_c, r_r = torch.cat(chosen_scores), torch.cat(rejected_scores)
            train_acc = preference_accuracy(r_c, r_r)
        else:
            train_acc = None
        if batches:
            self.epochs_trained += 1
        self.last_train = {
            "train_loss": total_loss / batches if batches else None,
            "train_pref_acc": train_acc,
            "batches": batches,
        }
        return self.last_train

    @torch.no_grad()
    def _score(self, model, tensors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        model.eval()
        if tensors["chosen"].numel() == 0:
            empty = torch.zeros(0, device=ctx_device(model))
            return empty, empty
        i_c = tensors["chosen"].to(ctx_device(model))
        i_r = tensors["rejected"].to(ctx_device(model))
        t = tensors["text"].to(ctx_device(model))
        return model(i_c, t), model(i_r, t)

    def validate(self, ctx: TaskContext, model, epoch: int, train_result) -> dict:
        held_out = bool(self.split and self.split.held_out)
        metrics: dict[str, Any] = {
            "held_out": held_out,
            "val_pref_acc": None,
            "val_margin": None,
            "val_loss": None,
            "per_type": {},
            **(train_result or {}),
        }
        if held_out:
            r_c, r_r = self._score(model, self.val_tensors)
            metrics["val_pref_acc"] = preference_accuracy(r_c, r_r)
            metrics["val_margin"] = preference_margin(r_c, r_r)
            metrics["val_loss"] = float(pairwise_ranking_loss(r_c, r_r).item())
            types = self.val_tensors["types"]
            for index, name in enumerate(self.dataset.type_names):
                mask = (types == index).to(r_c.device)
                if int(mask.sum()) == 0:
                    continue
                metrics["per_type"][name] = {
                    "pairs": int(mask.sum()),
                    "pref_acc": preference_accuracy(r_c[mask], r_r[mask]),
                    "margin": preference_margin(r_c[mask], r_r[mask]),
                }
        self.last_metrics = metrics
        return metrics

    def selection_score(self, metrics: dict) -> float:
        if metrics.get("held_out") and metrics.get("val_pref_acc") is not None:
            margin = metrics.get("val_margin") or 0.0
            return float(metrics["val_pref_acc"]) + 1e-3 * float(torch.tanh(torch.tensor(margin)))
        loss = metrics.get("train_loss")
        return -float(loss) if loss is not None else -1e9

    def _checkpoint_payload(self, model, epoch: int, metrics: dict, norm_mapping: dict | None) -> dict:
        return {
            "schema": CHECKPOINT_SCHEMA,
            "trainer": TRAINER_NAME,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "epoch": int(epoch),
            "image_model": self.image_model_size,
            "requested_image_model": self.config.image_model_size,
            "fell_back": self.fell_back,
            "text_model": self.text_model,
            "embed_preproc_sig": self.embed_preproc_sig,
            "image_size": int(self.config.image_size),
            "reward_head_arch": model.arch,
            "head_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "held_out": bool(metrics.get("held_out")),
            "val_pref_acc": metrics.get("val_pref_acc"),
            "val_margin": metrics.get("val_margin"),
            "train_pref_acc": metrics.get("train_pref_acc"),
            "per_type": metrics.get("per_type", {}),
            "type_counts": per_type_counts(self.records),
            "split_path": str(self.split_path) if self.split_path else None,
            "split": {
                "train_pairs": len(self.split.train) if self.split else 0,
                "val_pairs": len(self.split.val) if self.split else 0,
                "components": self.split.components if self.split else 0,
                "seed": int(self.config.seed),
                "val_fraction": float(self.config.val_fraction),
            },
            "norm_mapping": norm_mapping,
            "vae_prep": self.config.vae_prep,
            "manifest_path": str(self.config.manifest_path),
        }

    def save_candidate(self, ctx: TaskContext, model, epoch: int, metrics: dict, best: BestTracker) -> None:
        ctx.em.stage(Stage.saving, f"Saving epoch {epoch + 1} as the best reward head")
        path = Path(ctx.checkpoint_dir) / CHECKPOINT_NAME
        torch.save(self._checkpoint_payload(model, epoch, metrics, None), path)
        self.checkpoint_path = path
        best.best_checkpoint_path = str(path)
        best.best_metrics = dict(metrics)
        acc = metrics.get("val_pref_acc")
        best.best_val_macro_f1 = float(acc) if acc is not None else -1.0

    def epoch_message(self, ctx: TaskContext, epoch: int, metrics: dict, train_result, selected_score: float, best: BestTracker):
        return {
            "type": "epoch_complete",
            "epoch": epoch + 1,
            "max_epochs": int(self.config.max_epochs),
            "train_loss": metrics.get("train_loss"),
            "train_pref_acc": metrics.get("train_pref_acc"),
            "val_pref_acc": metrics.get("val_pref_acc"),
            "val_margin": metrics.get("val_margin"),
            "held_out": metrics.get("held_out"),
            "best_epoch": best.best_epoch + 1,
            "selection_score": selected_score,
        }

    # -- finalisation ------------------------------------------------------
    def finalize(self, ctx: TaskContext, model, best: BestTracker, epochs_completed: int) -> dict:
        stopped = _is_set(ctx.stop_event) or _is_set(ctx.stop_now_event)
        base = {
            "mode": TRAINER_NAME,
            "epochs_completed": int(self.epochs_trained),
            "stopped": bool(stopped),
            "pair_count": len(self.records),
            "type_counts": per_type_counts(self.records),
            "image_model": self.image_model_size,
            "requested_image_model": self.config.image_model_size,
            "fell_back": self.fell_back,
            "text_model": self.text_model,
            "embed_preproc_sig": self.embed_preproc_sig,
            "embedding_cache_stats": self.cache_stats,
            "split_path": str(self.split_path) if self.split_path else None,
            "split": {
                "train_pairs": len(self.split.train) if self.split else 0,
                "val_pairs": len(self.split.val) if self.split else 0,
                "components": self.split.components if self.split else 0,
            },
            "held_out": bool(self.split and self.split.held_out),
        }
        if self.checkpoint_path is None or not self.checkpoint_path.is_file():
            return {
                **base,
                "checkpoint_path": None,
                "val_pref_acc": None,
                "val_margin": None,
                "train_pref_acc": self.last_train.get("train_pref_acc"),
                "per_type": {},
                "norm_mapping": None,
            }
        ctx.em.stage(Stage.comparing, "Fitting the fixed score mapping on the best head")
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        head = RewardHead.from_arch(ckpt["reward_head_arch"])
        head.load_state_dict(ckpt["head_state_dict"])
        head = head.to(ctx.device).eval()
        everything = self.dataset.tensors()
        r_c, r_r = self._score(head, everything)
        pool = torch.cat([r_c, r_r]).cpu().tolist()
        mapping = fit_norm_mapping(pool)
        metrics = dict(best.best_metrics)
        payload = self._checkpoint_payload(head, int(ckpt["epoch"]), metrics, mapping)
        payload["created_at"] = ckpt["created_at"]
        torch.save(payload, self.checkpoint_path)
        ctx.em.stage(Stage.saving, "Reward model saved")
        return {
            **base,
            "checkpoint_path": str(self.checkpoint_path),
            "best_epoch": int(ckpt["epoch"]) + 1,
            "val_pref_acc": metrics.get("val_pref_acc"),
            "val_margin": metrics.get("val_margin"),
            "train_pref_acc": metrics.get("train_pref_acc"),
            "per_type": metrics.get("per_type", {}),
            "norm_mapping": mapping,
            "reward_head_arch": head.arch,
            "vae_prep": self.config.vae_prep,
        }


class _AnyEvent:
    """``is_set()`` when any of the wrapped events is set (None entries ignored)."""

    def __init__(self, *events: Any | None) -> None:
        self.events = [e for e in events if e is not None]

    def is_set(self) -> bool:
        return any(_is_set(e) for e in self.events)


def ctx_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        return torch.device("cpu")
