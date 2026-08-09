"""FCMAE self-supervised pretraining as a :class:`TrainingTask`.

Drives the shared :class:`~bittrainer.generic.generic_trainer.GenericTrainer`
for the dense-masked FCMAE pretraining path. The masking / decoder / loss /
dataset helpers and the async entry point live in ``bittrainer.fcmae_trainer``
and are reached through the ``ft`` module alias (house pattern), so the epoch
loop here stays thin: build the encoder+decoder wrapper, run one masked epoch,
score a held-out slice with deterministic masks, and export the BARE trunk.

Design summary (see ``fcmae_trainer`` for the full rationale):

* **Dense-masked approximation** — masked patches are zeroed at the input and
  re-zeroed after the stem and every stage; the decoder sees only the mask token
  at masked positions. All multiplies out-of-place (autograd intact).
* **AdamW + linear warmup + cosine**, base lr scaled by ``eff_batch / 256``,
  norms/biases/mask-token excluded from weight decay. The scheduler steps ONCE
  PER EPOCH (owned by the core). Prodigy_adv is deliberately NOT used.
* **"Epochs" are step chunks** (D5): a chunk is one scheduler period and, when
  ``steps_per_epoch > 0``, consumes the first N entries of the freshly shuffled
  train list. Resume is epoch-restart (``build_loaders`` ignores ``resume_info``).
* **No EMA** — MAE-family pretraining fine-tunes the raw weights.
* **Export is bare-trunk only** (D6): the decoder is never serialised.
"""

from __future__ import annotations

import logging
import random
import threading
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import bittrainer.backbone_trainer as bb
import bittrainer.fcmae_trainer as ft
from bittrainer.generic.task import (
    BestTracker,
    LoopSpec,
    ResumeInfo,
    TaskContext,
    TrainingTask,
)
from bittrainer.model import create_model
from bittrainer.training_state import (
    BackupCoordinator,
    capture_rng_states,
    loader_kwargs,
    restore_optimizer_state,
)

logger = logging.getLogger(__name__)


class FcmaeTask(TrainingTask):
    """Drives ``GenericTrainer`` for FCMAE self-supervised trunk pretraining."""

    trainer_name = "fcmae"

    def __init__(
        self,
        request: dict,
        *,
        cancel_event: threading.Event | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.request = request
        self.config = dict(request.get("training_config") or {})
        # Cancellation (raise FcmaeTrainingCancelled) is distinct from the core's
        # stop_event; max_steps rides the core's stop_event via steps_stop_event.
        self.cancel_event = cancel_event
        self.steps_stop_event = (
            stop_event if stop_event is not None else threading.Event()
        )

        c = self.config
        self.image_size = int(c.get("image_size") or 224)
        self.batch_size = int(c.get("batch_size") or 32)
        self.accumulation_steps = max(1, int(c.get("accumulation_steps") or 1))
        self.epochs = int(c.get("epochs") or 20)
        self.steps_per_epoch = int(c.get("steps_per_epoch") or 0)
        self.max_steps = c.get("max_steps")
        self.mask_ratio = float(c.get("mask_ratio", 0.6))
        self.decoder_dim = int(c.get("decoder_dim") or 512)
        self.norm_pix_loss = bool(c.get("norm_pix_loss", True))
        # BASE lr per effective batch 256 (scaled in create_optimizer).
        self.learning_rate = float(c.get("learning_rate") or 1.5e-4)
        self.weight_decay = float(c.get("weight_decay", 0.05))
        warmup = c.get("warmup_epochs")
        self.warmup_epochs = (
            int(warmup) if warmup is not None else max(1, round(0.05 * self.epochs))
        )
        self.val_fraction = float(c.get("validation_split", 0.02))
        self.val_max_images = int(c.get("val_max_images") or 2000)
        self.dataloader_workers = int(c.get("dataloader_workers") or 4)
        self.patience = int(c.get("patience") or c.get("early_stopping_patience") or 0)
        self.model_size = request.get("convnextv2_size") or "nano"
        self.patch_size = ft._PATCH_SIZE
        self.amp_enabled, self.amp_dtype = bb._amp_settings(c)

        # Populated across the lifecycle hooks.
        self.all_paths: list[str] = []
        self.train_paths: list[str] = []
        self.val_paths: list[str] = []
        self.content_hashes: dict[str, str] = {}
        self.dataset_fingerprint = ""
        self.dataset_count = 0
        self.model = None
        self.step = 0
        self.seq = 1
        self._cb = None
        self._val_loader: DataLoader | None = None
        self._last_lr = 0.0
        self.best_encoder_state: dict | None = None
        self.best_val_loss: float | None = None

        # max_steps <= 0 means "no optimiser step" — stop before the first epoch.
        if self.max_steps is not None and int(self.max_steps) <= 0:
            self.steps_stop_event.set()

    # -- helpers -----------------------------------------------------------
    def _emit(self, stage: str, status_text: str, **extra) -> None:
        self.seq += 1
        self._cb(
            {
                "type": "training_progress",
                "stage": stage,
                "status_text": status_text,
                "run_id": self.request.get("run_id"),
                "seq": self.seq,
                **extra,
            }
        )

    @property
    def _grid(self) -> tuple[int, int]:
        side = self.image_size // self.patch_size
        return side, side

    # -- one-time setup ----------------------------------------------------
    def make_context(
        self, progress_callback, stop_event, stop_now_event, pause_event
    ) -> TaskContext:
        from bittrainer.smart_cache import _noop_callback

        cb = progress_callback or _noop_callback
        self._cb = cb
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"FCMAE image_size must be a multiple of {self.patch_size}; got {self.image_size}"
            )
        requested_device = self.config.get("device")
        device = torch.device(
            requested_device
            if requested_device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Enumerate here (not in prepare_data) because loop_spec — called before
        # prepare_data — needs to know whether a val split exists.
        self.content_hashes = dict(self.request.get("content_hashes") or {})
        paths = ft._gather_images(self.request)
        if not paths:
            raise RuntimeError(
                "FCMAE pretraining found no readable images; supply 'images' and/or 'image_roots'."
            )
        self.all_paths = paths
        self.dataset_count = len(paths)
        self.dataset_fingerprint = ft._dataset_fingerprint(paths, self.content_hashes)
        self.train_paths, self.val_paths = ft._split_paths(
            paths, self.content_hashes, self.val_fraction, self.val_max_images
        )

        class _Emitter:
            """FCMAE emits raw frames only; ``.stage`` is a no-op the core calls
            once per epoch (validating) which this trainer surfaces via
            ``epoch_message`` instead."""

            def __init__(self, raw) -> None:
                self.raw = raw

            def stage(self, *_a, **_k) -> None:
                pass

        checkpoint_dir = Path(self.request["candidate_checkpoint_path"]).parent
        return TaskContext(
            device=device,
            dtype=self.amp_dtype,
            em=_Emitter(cb),
            cb=cb,
            checkpoint_dir=checkpoint_dir,
            stop_event=stop_event,
            stop_now_event=stop_now_event,
            pause_event=pause_event,
        )

    def fingerprint_init(self, ctx: TaskContext) -> None:
        c = self.config
        coordinator = BackupCoordinator(
            backup_dir=c.get("backup_dir"),
            backup_every_steps=int(c.get("backup_every_steps") or 0),
            pause_event=ctx.pause_event,
            cb=ctx.cb,
        )
        fingerprint = ft._fcmae_fingerprint(
            self.model_size,
            self.epochs,
            mask_ratio=self.mask_ratio,
            image_size=self.image_size,
            dataset_fingerprint=self.dataset_fingerprint,
            dataset_count=self.dataset_count,
            steps_per_epoch=self.steps_per_epoch,
        )
        resume_from = c.get("resume_from")
        resume_state = (
            coordinator.load_resume(fingerprint, resume_from=resume_from)
            if resume_from
            else None
        )
        ctx.coordinator = coordinator
        ctx.fingerprint = fingerprint
        ctx.resume_state = resume_state

    def loop_spec(self) -> LoopSpec:
        # Patience only bites with a val split; disabled/no-val runs get an
        # unreachable patience so they run every chunk.
        eff_patience = (
            self.patience
            if (self.patience > 0 and self.val_paths)
            else (self.epochs + 1)
        )
        return LoopSpec(
            max_epochs=self.epochs, patience=eff_patience, selection_min_delta=0.0
        )

    def prepare_data(self, ctx: TaskContext) -> None:
        self._emit(
            "preparing",
            f"Preparing FCMAE pretraining on {self.dataset_count} images "
            f"({len(self.train_paths)} train / {len(self.val_paths)} val)",
            train_images=len(self.train_paths),
            val_images=len(self.val_paths),
            dataset_fingerprint=self.dataset_fingerprint,
            image_size=self.image_size,
            mask_ratio=self.mask_ratio,
        )

    def create_model(self, ctx: TaskContext, resume_state: dict | None):
        # Always pretrained=False — the whole point is clean local provenance.
        encoder = create_model(
            model_size=self.model_size, pretrained=False, num_classes=0
        )
        decoder = ft._FcmaeDecoder(
            encoder.num_features, self.decoder_dim, self.patch_size
        )
        model = ft._FcmaeModel(encoder, decoder, self.patch_size).to(ctx.device)
        if resume_state is not None:
            model.load_state_dict(resume_state["model"])
        self.model = model
        return model

    def resolve_batch_size(
        self, ctx: TaskContext, model, resume_state: dict | None
    ) -> int:
        return self.batch_size

    def create_optimizer(
        self, ctx: TaskContext, model, eff_bs: int, resume_state: dict | None
    ):
        eff_batch = self.batch_size * self.accumulation_steps
        lr = self.learning_rate * eff_batch / 256
        optimizer = torch.optim.AdamW(
            ft._param_groups(model, self.weight_decay), lr=lr, betas=(0.9, 0.95)
        )
        t_max = max(1, self.epochs)
        scheduler = LambdaLR(
            optimizer, ft.warmup_cosine_lambda(self.warmup_epochs, self.epochs)
        )
        if resume_state is not None:
            restore_optimizer_state(resume_state, optimizer, scheduler, ctx.device)
            self.step = int(resume_state.get("global_step", 0))
        self._last_lr = optimizer.param_groups[0]["lr"]
        return optimizer, scheduler, t_max

    def restore_resume_extra(self, ctx: TaskContext, resume_state: dict) -> None:
        self.best_encoder_state = resume_state.get("best_encoder_state")
        self.best_val_loss = resume_state.get("best_val_loss")

    def resumed_message(
        self, ctx: TaskContext, best: BestTracker, global_step: int, start_epoch: int
    ) -> dict:
        return {
            "type": "training_resumed",
            "run_id": self.request.get("run_id"),
            "resumed_from": str(self.config.get("resume_from")),
            "epoch": start_epoch,
            "global_step": global_step,
            "best_score": best.best_validation_score,
        }

    # -- per-epoch ---------------------------------------------------------
    def reshuffle(self) -> None:
        random.shuffle(self.train_paths)

    def build_loaders(
        self, ctx: TaskContext, epoch: int, eff_bs: int, resume_info: ResumeInfo
    ):
        kwargs = loader_kwargs(self.dataloader_workers)
        # Val loader built once, deterministic order (shuffle=False).
        if self._val_loader is None and self.val_paths:
            self._val_loader = DataLoader(
                ft._FcmaeDataset(self.val_paths, ft._val_transform(self.image_size)),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=ft._collate_drop_none,
                **kwargs,
            )
        if self.steps_per_epoch > 0:
            limit = self.steps_per_epoch * self.batch_size * self.accumulation_steps
            epoch_paths = self.train_paths[:limit]
        else:
            epoch_paths = self.train_paths
        train_loader = DataLoader(
            ft._FcmaeDataset(epoch_paths, ft._train_transform(self.image_size)),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=ft._collate_drop_none,
            **kwargs,
        )
        # Epoch-restart resume: no schedule replay, no partial start.
        return train_loader, None, 0

    def _autocast(self, device):
        return torch.amp.autocast(
            device_type=device.type, dtype=self.amp_dtype, enabled=self.amp_enabled
        )

    def train_epoch(
        self,
        ctx: TaskContext,
        model,
        optimizer,
        train_loader,
        *,
        step_callback,
        boundary_hook,
        start_batch: int,
    ):
        device = ctx.device
        gh, gw = self._grid
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        micro = 0
        broke = False
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise ft.FcmaeTrainingCancelled
            if self.max_steps is not None and self.step >= int(self.max_steps):
                self.steps_stop_event.set()
                broke = True
                break
            if batch is None:
                continue
            images, _paths = batch
            images = images.to(device)
            grid_mask = ft.make_random_masks(
                images.shape[0], gh, gw, self.mask_ratio
            ).to(device)
            with self._autocast(device):
                pred = model.forward_masked(images, grid_mask)
            loss = ft.fcmae_loss(
                pred,
                images,
                grid_mask,
                patch_size=self.patch_size,
                norm_pix=self.norm_pix_loss,
            )
            (loss / self.accumulation_steps).backward()
            micro += 1
            epoch_loss += float(loss.detach())
            epoch_batches += 1
            if micro % self.accumulation_steps == 0:
                optimizer.step()
                self._last_lr = optimizer.param_groups[0]["lr"]
                optimizer.zero_grad(set_to_none=True)
                self.step += 1
                if boundary_hook(epoch_batches) == "stop":  # pause backed up mid-epoch
                    broke = True
                    break
        # Flush a trailing partial accumulation group (unless we broke to stop/cap).
        if not broke and micro % self.accumulation_steps != 0:
            optimizer.step()
            self._last_lr = optimizer.param_groups[0]["lr"]
            optimizer.zero_grad(set_to_none=True)
            self.step += 1
            boundary_hook(epoch_batches)
        return epoch_loss / max(epoch_batches, 1)

    @torch.no_grad()
    def validate(self, ctx: TaskContext, model, epoch: int, train_result) -> dict:
        if not self.val_paths or self._val_loader is None:
            return {}
        device = ctx.device
        gh, gw = self._grid
        model.eval()
        total = 0.0
        count = 0
        for batch in self._val_loader:
            if batch is None:
                continue
            images, paths = batch
            images = images.to(device)
            grid_mask = torch.cat(
                [
                    ft.make_random_masks(
                        1, gh, gw, self.mask_ratio, ft.val_mask_generator(p)
                    )
                    for p in paths
                ],
                dim=0,
            ).to(device)
            with self._autocast(device):
                pred = model.forward_masked(images, grid_mask)
            loss = ft.fcmae_loss(
                pred,
                images,
                grid_mask,
                patch_size=self.patch_size,
                norm_pix=self.norm_pix_loss,
            )
            total += float(loss) * images.shape[0]
            count += images.shape[0]
        model.train()
        return {"val_loss": total / count} if count else {}

    def selection_score(self, metrics: dict) -> float:
        # The core maximises; a loss-based score is negated.
        return -float(metrics["val_loss"]) if metrics else 0.0

    def save_candidate(
        self, ctx: TaskContext, model, epoch: int, metrics: dict, best: BestTracker
    ) -> None:
        # No val signal: never snapshot a "best" — finalize serialises the final
        # encoder weights instead.
        if not self.val_paths or not metrics:
            return
        # clone() is load-bearing: on CPU ``.cpu()`` returns the SAME tensor, so a
        # bare snapshot would alias the live weights and drift as later epochs
        # update them — the "best" would silently become the last. clone() freezes it.
        self.best_encoder_state = {
            k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()
        }
        self.best_val_loss = float(metrics["val_loss"])
        best.best_metrics = dict(metrics)

    def epoch_message(
        self,
        ctx: TaskContext,
        epoch: int,
        metrics: dict,
        train_result,
        selected_score: float,
        best: BestTracker,
    ) -> None:
        self._emit(
            "training",
            f"Epoch {epoch + 1}/{self.epochs}",
            epoch=epoch + 1,
            epochs=self.epochs,
            steps=self.step,
            loss=train_result,
            val_loss=(metrics.get("val_loss") if metrics else None),
            lr=self._last_lr,
            best_val_loss=self.best_val_loss,
        )
        return None

    def collect_extra_state(
        self, ctx: TaskContext, *, rng_epoch_start, schedule, batch_in_epoch: int
    ) -> dict:
        return {
            "rng_now": capture_rng_states(ctx.device),
            "rng_epoch_start": rng_epoch_start,
            "best_encoder_state": self.best_encoder_state,
            "best_val_loss": self.best_val_loss,
        }

    # -- finalisation ------------------------------------------------------
    def finalize(
        self, ctx: TaskContext, model, best: BestTracker, epochs_completed: int
    ) -> dict:
        self._emit("saving", "Exporting FCMAE pretrained trunk")
        if self.val_paths and self.best_encoder_state is not None:
            encoder_state = self.best_encoder_state
            val_loss = self.best_val_loss
            best_epoch = best.best_epoch + 1
        else:
            # No val signal (or no improved chunk): serialise the final trunk.
            encoder_state = {
                k: v.detach().cpu() for k, v in model.encoder.state_dict().items()
            }
            val_loss = self.best_val_loss
            best_epoch = epochs_completed

        candidate_path = Path(self.request["candidate_checkpoint_path"])
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self._candidate_metadata(val_loss, epochs_completed)
        from safetensors.torch import save_file

        # BARE trunk keys only (stem.*, stages.*, norm_pre.*, head.norm.*) — no
        # decoder.*, no backbone. prefix — so apply_backbone_init consumes it
        # unchanged (its bare-key branch).
        state = {key: value.detach().cpu() for key, value in encoder_state.items()}
        save_file(
            state,
            str(candidate_path),
            metadata={
                k: bb._stringify(v) for k, v in metadata.items() if v is not None
            },
        )
        self._emit(
            "saving",
            "FCMAE pretrained trunk written",
            candidate_checkpoint_path=str(candidate_path),
            val_loss=val_loss,
        )
        return {
            "candidate_checkpoint_path": str(candidate_path),
            "val_loss": (float(val_loss) if val_loss is not None else None),
            "steps_completed": int(self.step),
            "epochs_completed": int(epochs_completed),
            "dataset_image_count": int(self.dataset_count),
            "best_epoch": int(best_epoch),
        }

    def _candidate_metadata(self, val_loss, epochs_completed) -> dict:
        """FCMAE provenance metadata (D6) — all values stringified, None dropped."""
        return {
            "pretrain_method": "fcmae",
            "mask_ratio": self.mask_ratio,
            "patch_size": str(self.patch_size),
            "image_size": self.image_size,
            "steps_completed": self.step,
            "epochs_completed": epochs_completed,
            "dataset_image_count": self.dataset_count,
            "dataset_fingerprint": self.dataset_fingerprint,
            "val_loss": val_loss,
            "license_provenance": self.request.get("license_provenance")
            or "locally_trained",
            "external_pretrained_used": bool(
                self.request.get("external_pretrained_used")
            ),
            "release_blocking": bool(self.request.get("release_blocking")),
            "family_name": self.request.get("family_name"),
            "architecture": self.request.get("architecture"),
            "size_alias": self.request.get("size_alias"),
            "display_size": self.request.get("display_size"),
            "convnextv2_size": self.request.get("convnextv2_size"),
            "training_run_id": self.request.get("run_id"),
            "version": "1",
            "status": "candidate",
            "created_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "training_config_json": self.config,
        }
