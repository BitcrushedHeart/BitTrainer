"""Training loop for ConvNeXt V2 binary classifiers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from adv_optm import Prodigy_adv
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from bittrainer.dataset import (
    ConceptDataset,
    _DimensionCache,
    build_bucket_batch_sampler,
)
from bittrainer.backbone_init import apply_backbone_init, wants_timm_pretrained
from bittrainer.ema import ModelEMA
from bittrainer.model import (
    build_llrd_param_groups,
    create_model,
    freeze_backbone,
    load_checkpoint,
    unfreeze_backbone,
    unfreeze_stage,
)
from bittrainer.validation import compute_metrics, find_optimal_threshold

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
    neg_pos_ratio: float = 1.0
    model_size: str = "nano"
    device: str = "cuda"
    dtype: str = "bfloat16"
    from_scratch: bool = False
    # Bitcrush Engine backbone spec (see bittrainer.backbone_init) — governs
    # where fresh-model backbone weights come from. None = timm pretrained.
    backbone_init: dict | None = None
    extra_positive_dirs: list[str] = field(default_factory=list)
    negative_dirs: list[str] = field(default_factory=list)
    hard_negative_paths: list[str] = field(default_factory=list)
    hard_negative_weight: int = 3
    label_smoothing: float = 0.1
    best_model_name: str = "best.pt"
    checkpoint_dir: str | None = None
    skin_normalise: bool = False
    face_model_path: str = ""
    cache_dir: str | None = None
    use_cache: bool = True
    cache_workers: int = 10
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
    if config.llrd:
        params = build_llrd_param_groups(model, config.llrd_decay)
    else:
        params = model.parameters()
    return Prodigy_adv(
        params, lr=1.0, d_coef=0.9,
        weight_decay=0.01, betas=(0.9, 0.999),
        kourkoutas_beta=True, k_warmup_steps=50,
        cautious_wd=True,
    )


def _get_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


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
            images, dtype=dtype,
            randaugment_n=randaugment_n,
            randaugment_m=randaugment_m,
            random_erasing_p=random_erasing_p,
        )
        labels = labels.to(device)

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
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
        boundary_signal = boundary_hook(num_batches) if boundary_hook is not None else None

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

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
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
    }


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
    metrics = compute_metrics(val_result["labels"], val_result["probs"], threshold=threshold)
    return metrics, threshold


def _rebalance_val_negatives(train_ds: ConceptDataset, val_ds: ConceptDataset) -> None:
    """Ensure the val set has enough negatives for meaningful evaluation.

    Target: at least as many negatives as positives in val.
    Cap: never take more than 40% of total negatives (training still needs them).
    """
    val_pos = len(val_ds._positive_paths)
    val_neg = len(val_ds._all_negative_paths)
    target = max(5, val_pos)

    if val_neg >= target:
        return

    needed = target - val_neg
    total_neg = len(train_ds._all_negative_paths) + val_neg
    max_donate = max(0, int(total_neg * 0.4) - val_neg)
    to_donate = min(needed, max_donate, len(train_ds._all_negative_paths))

    if to_donate <= 0:
        return

    donated = train_ds._all_negative_paths[:to_donate]
    train_ds._all_negative_paths = train_ds._all_negative_paths[to_donate:]
    val_ds._all_negative_paths = val_ds._all_negative_paths + donated

    # Ensure val_ds has bucket info for donated paths (they were precomputed by train_ds)
    val_ds._path_info.update(
        {str(p): train_ds._path_info[str(p)] for p in donated if str(p) in train_ds._path_info}
    )

    train_ds._build_samples()
    val_ds._build_samples()

    logger.info(
        "Rebalanced val set: donated %d negatives from train → val (val now %d neg / %d pos)",
        to_donate, len(val_ds._all_negative_paths), val_pos,
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
    from bittrainer.runtime import configure_cuda_backend
    from bittrainer.smart_cache import _noop_callback, _never_stop
    from bittrainer.training_state import (
        BackupCoordinator,
        backup_on_exception,
        capture_optimizer_aux_state,
        make_fingerprint,
        prime_optimizer_after_resume,
        restore_optimizer_aux_state,
        sanitize_for_backup,
    )
    cb = progress_callback or config.progress_callback or _noop_callback
    device = torch.device(config.device)
    dtype = _get_dtype(config.dtype)
    configure_cuda_backend()
    concept_folder = Path(config.concept_folder)
    checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else concept_folder / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    coordinator = BackupCoordinator(
        backup_dir=config.backup_dir, pause_event=pause_event,
        backup_every_steps=config.backup_every_steps, cb=cb,
    )
    fingerprint = make_fingerprint(
        class_names=["negative", "positive"], num_classes=2,
        max_epochs=config.max_epochs, multi_label=False, ordinal=False,
        best_model_name=config.best_model_name, model_size=config.model_size,
    )
    resume_state = (
        coordinator.load_resume(fingerprint, resume_from=config.resume_from)
        if config.resume_from else None
    )

    def _paused_result(cur_epoch: int, gstep: int, backup_path) -> dict:
        bp = str(backup_path) if backup_path else None
        cb({"type": "training_paused", "epoch": cur_epoch, "global_step": gstep, "backup_path": bp})
        return {"paused": True, "backup_path": bp, "epoch": cur_epoch, "global_step": gstep}

    concept_name = config.concept_name or concept_folder.name

    # --- SmartCache setup ---
    smart_cache = None
    if config.use_cache:
        from bittrainer.smart_cache import SmartCache, face_model_signature
        cache_root = Path(config.cache_dir) if config.cache_dir else (concept_folder / ".smart_cache")
        smart_cache = SmartCache(
            cache_root,
            modeltype=config.modeltype,
            progress_callback=cb,
            stop_check=partial(_stop_event_is_set, stop_event),
            face_model_sig=face_model_signature(config.face_model_path or None),
        )

    # --- Sourceless path: reconstruct samples from cache, skip dataset indexing ---
    if config.sourceless:
        if smart_cache is None:
            raise RuntimeError("sourceless=True requires use_cache=True and a cache_dir")
        cb({
            "type": "training_progress", "stage": "validating",
            "status_text": "Loading sourceless samples from cache",
            "step": 0, "total_steps": 0,
        })
        train_ds = ConceptDataset(
            concept_folder, split="train", cache=smart_cache,
            sourceless=True, concept_name=concept_name,
        )
        val_ds = ConceptDataset(
            concept_folder, split="val", cache=smart_cache,
            sourceless=True, concept_name=concept_name,
        )
        num_positives = len(train_ds._positive_paths)
        face_bboxes: dict[str, list[int]] = {}
    else:
        cache_dir = concept_folder / ".resize_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dim_cache = _DimensionCache(cache_dir / "dimensions.json")

        train_ds = ConceptDataset(
            concept_folder, split="train",
            neg_pos_ratio=config.neg_pos_ratio,
            extra_positive_dirs=config.extra_positive_dirs,
            negative_dirs=config.negative_dirs,
            hard_negative_paths=config.hard_negative_paths,
            hard_negative_weight=config.hard_negative_weight,
            dim_cache=dim_cache,
            skin_normalise=config.skin_normalise,
            concept_name=concept_name,
        )
        val_ds = ConceptDataset(
            concept_folder, split="val",
            extra_positive_dirs=config.extra_positive_dirs,
            negative_dirs=config.negative_dirs,
            hard_negative_paths=config.hard_negative_paths,
            hard_negative_weight=1,
            dim_cache=dim_cache,
            skin_normalise=config.skin_normalise,
            concept_name=concept_name,
        )

        _rebalance_val_negatives(train_ds, val_ds)

        num_positives = len(train_ds._positive_paths)

        # --- Face-aware cropping pre-computation ---
        face_bboxes = {}
        if config.face_model_path:
            from bittrainer.face_crop import FaceBBoxCache, precompute_face_bboxes
            face_cache = FaceBBoxCache(cache_dir / "face_bboxes.json")
            all_image_paths = [s["path"] for s in train_ds.samples] + [s["path"] for s in val_ds.samples]

            def _face_progress(done: int, total: int) -> None:
                cb({
                    "type": "training_progress", "stage": "face_detection",
                    "status_text": f"Detecting faces ({done}/{total})",
                    "step": done, "total_steps": total,
                })

            precompute_face_bboxes(
                all_image_paths, face_cache, config.face_model_path,
                device=config.device,
                progress_fn=_face_progress,
            )
            for p in all_image_paths:
                bbox = face_cache.get(p)
                if bbox:
                    face_bboxes[p] = bbox
            train_ds.refresh_face_bboxes(face_bboxes)
            val_ds.refresh_face_bboxes(face_bboxes)

        # --- Warm SmartCache (validate + build missing) ---
        if smart_cache is not None:
            from bittrainer.cache_builders import build_image_tensor
            from bittrainer.smart_cache import CachingStoppedException
            all_cache_samples = train_ds.samples + val_ds.samples
            try:
                smart_cache.prepare(
                    all_cache_samples, build_image_tensor,
                    num_workers=config.cache_workers,
                    stage_label="caching",
                )
            except CachingStoppedException:
                logger.info("Caching interrupted by stop_event")
                cb({"type": "training_cancelled", "stage": "caching",
                    "status_text": "Cancelled during cache build"})
                raise
            # Callbacks are only needed during prepare(). Replace with picklable
            # no-ops so the cache (now attached to datasets) survives pickling
            # when DataLoader workers spawn on Windows — mp.Event and local
            # closures aren't picklable.
            smart_cache._progress_cb = _noop_callback
            smart_cache._stop_check = _never_stop
            train_ds.set_cache(smart_cache)
            val_ds.set_cache(smart_cache)

    # --- Count samples per bucket ---
    bucket_counts: dict[tuple[int, int], int] = {}
    for s in train_ds.samples:
        b = s["bucket"]
        bucket_counts[b] = bucket_counts.get(b, 0) + 1

    # Create model — warm-start from best.pt, rebuild from a resume backup, or
    # start fresh.
    existing_best = checkpoint_dir / config.best_model_name
    use_gradual_unfreeze = num_positives < 50
    cb({
        "type": "training_progress", "stage": "preparing",
        "status_text": "Loading model",
    })
    if resume_state is not None:
        model = _fresh_binary_model(config, dtype=dtype).to(device)
        model.load_state_dict(resume_state["model"])
        use_gradual_unfreeze = bool(resume_state.get("use_gradual_unfreeze", use_gradual_unfreeze))
    elif not config.from_scratch and existing_best.exists():
        try:
            model = load_checkpoint(
                str(existing_best), device=str(device), dtype=dtype,
                model_size=config.model_size,
            ).to(device)
            logger.info("Warm-starting from existing checkpoint: %s", existing_best)
        except Exception:
            logger.warning("Failed to load existing checkpoint, starting from pretrained", exc_info=True)
            model = _fresh_binary_model(config, dtype=dtype).to(device)
    else:
        model = _fresh_binary_model(config, dtype=dtype).to(device)

    # Probe unfrozen = worst-case VRAM, then freeze for epoch 0. Resume reuses
    # the backed-up batch size (skip the probe).
    from bittrainer.autobatch import determine_batch_size

    def _probe_progress(attempt: int, candidate: int, cap: int, status: str) -> None:
        cb({
            "type": "training_progress", "stage": "preparing",
            "status_text": f"Probing batch size (try {attempt}: {candidate}/{cap} — {status})",
        })

    if resume_state is not None:
        eff_bs = int(resume_state["eff_bs"])
        cb({"type": "autobatch", "batch_size": eff_bs, "resumed": True})
    else:
        cb({
            "type": "training_progress", "stage": "preparing",
            "status_text": "Probing optimal batch size",
        })
        auto_result = determine_batch_size(
            model, bucket_counts, device, dtype=dtype,
            use_ema=config.use_ema, progress_callback=_probe_progress,
        )
        eff_bs = auto_result["batch_size"]
        cb({"type": "autobatch", **auto_result})
    freeze_backbone(model)

    # Optimiser: Prodigy_adv with kourkoutas beta and cautious weight decay
    optimizer = _make_optimizer(model, config)

    # Scheduler: cosine annealing (stepped once per epoch)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    # EMA tracks all params from the start; freeze/unfreeze only affects which
    # ones receive gradient updates, but the EMA still mirrors the live tensor
    # values, which is what we want for inference-time smoothing.
    ema = ModelEMA(model, decay=config.ema_decay) if config.use_ema else None

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path = None
    best_metrics: dict = {}
    global_step = 0
    start_epoch = 0
    scheduler_t_max = config.max_epochs

    if resume_state is not None:
        start_epoch = int(resume_state["epoch"])
        global_step = int(resume_state.get("global_step", 0))
        best = resume_state["best"]
        best_val_f1 = best["best_val_f1"]
        best_epoch = best["best_epoch"]
        patience_counter = best["patience_counter"]
        best_checkpoint_path = best["best_checkpoint_path"]
        best_metrics = dict(best.get("best_metrics") or {})
        # Replay the gradual-unfreeze reconstruction so the optimizer param_groups
        # match the epoch we resume INTO, BEFORE loading optimizer/scheduler
        # state. (trainer.py epoch-1 transition: full unfreeze + fresh
        # optimizer/scheduler for the non-gradual path.)
        skip_opt_load = False
        if start_epoch >= 1:
            if use_gradual_unfreeze:
                unfreeze_stage(model, _NUM_STAGES - 1)  # epoch-1 transition
                for e in range(2, start_epoch + 1):
                    si = _NUM_STAGES - e
                    if 0 <= si < _NUM_STAGES:
                        unfreeze_stage(model, si)
            else:
                unfreeze_backbone(model)
                optimizer = _make_optimizer(model, config)
                scheduler_t_max = config.max_epochs - 1
                scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_t_max)
                # Resuming INTO epoch 1: the optimizer/scheduler are freshly
                # created here exactly as the uninterrupted run does — the backup
                # (taken after epoch 0) holds the now-discarded frozen optimizer,
                # so start these fresh rather than loading it.
                skip_opt_load = start_epoch == 1
        if not skip_opt_load:
            optimizer.load_state_dict(resume_state["optimizer"])
            prime_optimizer_after_resume(optimizer)
            restore_optimizer_aux_state(optimizer, resume_state.get("optimizer_aux"), device)
            scheduler.load_state_dict(resume_state["scheduler"])
        if ema is not None and resume_state.get("ema") is not None:
            ema.load_full_state_dict(resume_state["ema"])
        cb({
            "type": "training_resumed", "resumed_from": str(config.resume_from),
            "epoch": start_epoch, "global_step": global_step,
            "best_val_f1": best_val_f1, "best_epoch": best_epoch + 1,
        })

    # Async disk reads + pinned H2D transfers overlap with GPU compute.
    # Workers respawn when train loader rebuilds between epochs (for negative
    # resampling) — accept the ~3s Windows spawn cost for within-epoch async.
    cb({
        "type": "training_progress", "stage": "preparing",
        "status_text": f"Batch size {eff_bs} — spawning data workers",
    })
    _n_workers = max(0, int(config.dataloader_workers))
    _loader_kwargs: dict = {"num_workers": _n_workers, "pin_memory": True}
    if _n_workers > 0:
        _loader_kwargs.update(persistent_workers=True, prefetch_factor=3)

    val_sampler = build_bucket_batch_sampler(val_ds, batch_size=eff_bs)
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler, collate_fn=_collate_bucket_batch, **_loader_kwargs,
    )

    def _rebuild_train_loader() -> DataLoader:
        sampler = build_bucket_batch_sampler(train_ds, batch_size=eff_bs)
        return DataLoader(
            train_ds, batch_sampler=sampler, collate_fn=_collate_bucket_batch, **_loader_kwargs,
        )

    train_loader = _rebuild_train_loader()

    _train_start_mono = time.monotonic()

    cb({
        "type": "training_progress", "stage": "training",
        "status_text": f"Training (epoch {start_epoch}/{config.max_epochs})",
        "epoch": start_epoch, "max_epochs": config.max_epochs,
    })

    def _collect_binary_state(cur_epoch: int) -> dict:
        return {
            "fingerprint": fingerprint,
            "trainer": "binary",
            "epoch": cur_epoch,
            "batch_in_epoch": 0,  # binary resume is epoch-restart
            "global_step": global_step,
            "eff_bs": eff_bs,
            "use_gradual_unfreeze": use_gradual_unfreeze,
            "scheduler_t_max": scheduler_t_max,
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "optimizer_aux": capture_optimizer_aux_state(optimizer),
            "scheduler": scheduler.state_dict(),
            "ema": ema.full_state_dict() if ema is not None else None,
            "best": {
                "best_val_f1": best_val_f1,
                "best_epoch": best_epoch,
                "patience_counter": patience_counter,
                "best_checkpoint_path": best_checkpoint_path,
                "best_metrics": sanitize_for_backup(best_metrics),
            },
        }

    epoch = start_epoch - 1  # so epochs_completed is defined if the loop is empty
    _exc_epoch = start_epoch
    existing_best = checkpoint_dir / config.best_model_name

    with backup_on_exception(
        lambda: _collect_binary_state(_exc_epoch), coordinator.manager, cb=cb,
    ):
        for epoch in range(start_epoch, config.max_epochs):
            _exc_epoch = epoch
            if stop_now_event is not None and stop_now_event.is_set():
                logger.info("Stop-now requested before epoch %d — running final comparison", epoch)
                cb({"type": "stop_now", "epoch": epoch, "max_epochs": config.max_epochs})
                break
            if stop_event is not None and stop_event.is_set():
                logger.info("Graceful stop requested after epoch %d — running final comparison", epoch)
                cb({"type": "graceful_stop", "epoch": epoch, "max_epochs": config.max_epochs})
                break
            if coordinator.paused:
                path = coordinator.save(_collect_binary_state(epoch), reason="pause")
                return _paused_result(epoch, global_step, path)

            # Resample cross-concept negatives so the model sees different
            # negatives each epoch (no-op for legacy per-concept negatives)
            if epoch > 0:
                train_ds.resample_negatives()
                train_loader = _rebuild_train_loader()

            # Unfreezing logic. Skipped for the epoch we resumed INTO — the
            # reconstruction above already put the model/optimizer/scheduler into
            # that epoch's state (re-running it would recreate the optimizer and
            # discard the restored state).
            if not (resume_state is not None and epoch == start_epoch):
                if epoch == 1:
                    if use_gradual_unfreeze:
                        # Unfreeze last stage
                        unfreeze_stage(model, _NUM_STAGES - 1)
                    else:
                        unfreeze_backbone(model)
                        # Re-create optimizer with all params
                        optimizer = _make_optimizer(model, config)
                        scheduler_t_max = config.max_epochs - 1  # 1 epoch already done
                        scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_t_max)
                elif epoch > 1 and use_gradual_unfreeze:
                    stage_idx = _NUM_STAGES - epoch  # 3, 2, 1, 0
                    if 0 <= stage_idx < _NUM_STAGES:
                        unfreeze_stage(model, stage_idx)

            # Train
            epoch_start_mono = time.monotonic()

            def _on_step(step: int, total_steps: int, avg_loss: float) -> None:
                elapsed = time.monotonic() - epoch_start_mono
                throughput = step / elapsed if elapsed > 0 else None
                eta_seconds = (total_steps - step) / throughput if throughput and throughput > 0 else None
                cb({
                    "type": "training_progress",
                    "stage": "training",
                    "status_text": f"Training (epoch {epoch + 1}/{config.max_epochs}, step {step}/{total_steps})",
                    "epoch": epoch + 1,
                    "max_epochs": config.max_epochs,
                    "step": step,
                    "total_steps": total_steps,
                    "eta_seconds": eta_seconds,
                    "throughput": throughput,
                    "throughput_unit": "batch/s",
                    "train_loss": round(avg_loss, 4),
                    "best_val_f1": best_val_f1 if best_val_f1 >= 0 else None,
                    "best_epoch": best_epoch + 1 if best_val_f1 >= 0 else None,
                })

            def _boundary_hook(num_batches: int) -> str | None:
                nonlocal global_step
                global_step += 1
                return coordinator.on_boundary(
                    lambda: _collect_binary_state(epoch), global_step,
                )

            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device, dtype,
                step_callback=_on_step,
                stop_now_event=stop_now_event,
                ema=ema,
                randaugment_n=config.randaugment_n,
                randaugment_m=config.randaugment_m,
                random_erasing_p=config.random_erasing_p,
                boundary_hook=_boundary_hook,
            )
            if coordinator.paused:
                # Pause fired mid-epoch — the boundary hook wrote the backup.
                # Return without the fair-comparison / promotion block.
                return _paused_result(epoch, global_step, coordinator.last_backup_path)
            if stop_now_event is not None and stop_now_event.is_set():
                cb({
                    "type": "stop_now",
                    "epoch": epoch + 1,
                    "max_epochs": config.max_epochs,
                    "status_text": f"Stop-now triggered mid-epoch {epoch + 1} — finishing up",
                })
            scheduler.step()

            # Validate (against EMA weights when enabled — they generalise better)
            eval_model = ema.module if ema is not None else model
            val_result = evaluate(eval_model, val_loader, criterion, device, dtype)
            # Select on F1 at the tuned threshold (what inference ships), not @0.5.
            metrics, _epoch_threshold = _tuned_val_metrics(val_result)
            metrics["val_loss"] = val_result["val_loss"]
            metrics["train_loss"] = train_loss

            # Check improvement
            val_f1 = metrics.get("f1", 0.0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
                best_metrics = metrics.copy()

                ckpt_path = checkpoint_dir / "candidate.pt"
                primary_state = ema.state_dict() if ema is not None else model.state_dict()
                ckpt_meta = {
                    "state_dict": primary_state,
                    "num_classes": 2,
                    "model_size": config.model_size,
                }
                if ema is not None:
                    ckpt_meta["model_state_dict"] = model.state_dict()
                    ckpt_meta["ema_decay"] = config.ema_decay
                torch.save(ckpt_meta, ckpt_path)
                best_checkpoint_path = str(ckpt_path)
            else:
                patience_counter += 1

            # Progress callback
            cb({
                "type": "epoch_complete",
                "stage": "training",
                "status_text": f"Epoch {epoch + 1}/{config.max_epochs} complete (val F1 {val_f1:.3f})",
                "epoch": epoch + 1,
                "max_epochs": config.max_epochs,
                "train_loss": train_loss,
                "val_loss": val_result["val_loss"],
                "val_f1": val_f1,
                "val_precision": metrics.get("precision", 0.0),
                "val_recall": metrics.get("recall", 0.0),
                "val_auprc": metrics.get("auprc", 0.0),
                "best_val_f1": best_val_f1,
                "best_epoch": best_epoch + 1,
            })

            # Epoch-boundary backup: resume at epoch+1 (batch_in_epoch=0). Coherent
            # point — scheduler stepped, best updated.
            if coordinator.enabled:
                coordinator.save(_collect_binary_state(epoch + 1), reason="periodic")

            # Early stopping
            if patience_counter >= config.patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, config.patience)
                break

        # Compare candidate checkpoint against existing best.pt on the CURRENT val
        # set. This ensures a fair apples-to-apples comparison even when the val
        # set has changed between training runs.
        result = _binary_compare_promote(
            config, best_checkpoint_path=best_checkpoint_path, existing_best=existing_best,
            model=model, val_loader=val_loader, criterion=criterion, device=device, dtype=dtype,
            best_val_f1=best_val_f1, best_metrics=best_metrics, best_epoch=best_epoch,
            epochs_completed=epoch + 1, num_positives=num_positives, train_ds=train_ds,
        )

    # Successful completion: the backups are obsolete.
    if coordinator.manager is not None:
        coordinator.manager.delete_all()
    return result


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
                    str(existing_best), map_location=device, weights_only=True,
                )
                old_sd = old_data["state_dict"] if isinstance(old_data, dict) and "state_dict" in old_data else old_data
                old_size = old_data.get("model_size", config.model_size) if isinstance(old_data, dict) else config.model_size
                old_model = create_model(model_size=old_size, pretrained=False, dtype=dtype).to(device)
                old_model.load_state_dict(old_sd)
                old_val_result = evaluate(
                    old_model, val_loader, criterion, device, dtype,
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
                        old_f1, best_val_f1,
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
                        best_val_f1, old_f1,
                    )
                    candidate = Path(best_checkpoint_path)
                    candidate.replace(existing_best)
                    best_checkpoint_path = str(existing_best)
                    model.load_state_dict(_unwrap_state_dict(
                        torch.load(best_checkpoint_path, map_location=device, weights_only=True),
                    ))
                    val_result = evaluate(
                        model, val_loader, criterion, device, dtype,
                    )
                    optimal_threshold = find_optimal_threshold(
                        val_result["labels"], val_result["probs"],
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
                    torch.load(
                        best_checkpoint_path,
                        map_location=device,
                        weights_only=True,
                    ),
                )
                val_result = evaluate(
                    model, val_loader, criterion, device, dtype,
                )
                optimal_threshold = find_optimal_threshold(
                    val_result["labels"], val_result["probs"],
                )
        else:
            # No existing best.pt — promote candidate directly
            candidate = Path(best_checkpoint_path)
            candidate.replace(existing_best)
            best_checkpoint_path = str(existing_best)
            model.load_state_dict(_unwrap_state_dict(
                torch.load(best_checkpoint_path, map_location=device, weights_only=True),
            ))
            val_result = evaluate(model, val_loader, criterion, device, dtype)
            optimal_threshold = find_optimal_threshold(
                val_result["labels"], val_result["probs"],
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
    }
