"""FCMAE self-supervised pretraining for BitTrainer (ConvNeXt V2 paper).

This module adds a fully-convolutional masked-autoencoder (FCMAE) pretraining
path over a large pool of UNLABELED local images. It produces a safetensors
checkpoint holding ONLY the bare timm ConvNeXt V2 trunk (``stem.*``,
``stages.*``, ``norm_pre.*``, ``head.norm.*``) — exactly what
``bittrainer.backbone_init.apply_backbone_init`` consumes — with clean
provenance metadata (``pretrain_method="fcmae"``,
``license_provenance="locally_trained"``, ``external_pretrained_used=false``).
The run rides the shared :class:`~bittrainer.generic.generic_trainer.GenericTrainer`
lifecycle, so pause / backup / resume / graceful-stop all work unchanged.

Why a DENSE-masked approximation
--------------------------------
The official FCMAE uses sparse convolutions so the encoder literally never
touches masked regions. timm ConvNeXt V2 is dense, so we approximate: one random
mask per sample on the final-stage 32-px grid, applied by zeroing the masked
patches of the input AND re-multiplying the feature map by the (upsampled) mask
after the stem and after EVERY stage. All multiplications are out-of-place so
autograd stays intact. Dense 7x7 depthwise convs still smear a little
information across patch boundaries within a stage, but re-masking after each
stage bounds that leak to intra-stage receptive fields and guarantees the
decoder sees ONLY the mask token at masked positions — the ConvNeXt V2 paper's
own note is that FCMAE can be implemented densely by zeroing (the sparse path is
a compute optimisation). The loss is computed on masked patches whose encoder
features were zeroed at every stage boundary, so the task does not collapse.

Why AdamW (not the house Prodigy_adv)
-------------------------------------
Prodigy_adv's D-adaptation is validated in-house only on supervised objectives.
MAE-family reconstruction has a very different curvature (a heavy early transient
while the decoder and mask token learn) that would drive D-adaptation's global
step estimate, and there is no evidence for Prodigy on multi-day MAE pretraining.
A mis-adapted step discovered three days into a run is an expensive failure mode.
AdamW + linear warmup + cosine is the MAE/FCMAE convention with years of
replication, so predictability wins for a long unattended run. See D4 of the
plan. The optimizer identity is folded into the run fingerprint.

Request contract (all training knobs on ``training_config``)
------------------------------------------------------------
``run_id``; optional ``family_name`` / ``architecture`` / ``size_alias`` /
``display_size`` / ``convnextv2_size``; REQUIRED ``candidate_checkpoint_path``;
``images`` (explicit files) and/or ``image_roots`` (walked recursively) — at
least one must yield images; optional ``content_hashes`` (path -> hash) for dedup
+ stable split keys; ``license_provenance`` / ``external_pretrained_used`` /
``release_blocking``. ``training_config`` keys: ``image_size`` (multiple of 32),
``batch_size``, ``accumulation_steps``, ``epochs`` (chunks), ``steps_per_epoch``
(optimizer steps per chunk; 0 = one full pass), ``max_steps`` (optional global
cap), ``mask_ratio``, ``decoder_dim``, ``norm_pix_loss``, ``learning_rate`` (BASE
lr per effective batch 256, scaled internally), ``weight_decay``,
``warmup_epochs``, ``validation_split`` (0 disables the held-out slice),
``val_max_images``, ``device``, ``use_amp`` / ``amp_dtype``,
``dataloader_workers``, ``backup_dir`` / ``backup_every_steps`` / ``resume_from``.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path

import torch
import torch.nn as nn
import xxhash
from PIL import Image
from timm.layers import trunc_normal_
from timm.models.convnext import ConvNeXtBlock
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import bittrainer.backbone_trainer as bb
from bittrainer.training_state import make_fingerprint

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_PATCH_SIZE = 32


class FcmaeTrainingCancelled(RuntimeError):
    """Raised inside the worker thread when the caller cancelled the task.

    Distinct from a graceful stop: it propagates out instead of finalising
    (BackboneTask precedent)."""


# --------------------------------------------------------------------------- #
# Image gathering / deterministic split                                        #
# --------------------------------------------------------------------------- #


def _split_key(path: str, hashes: dict[str, str]) -> str:
    """Stable partition key: the content hash when supplied, else the path.

    Using the content hash keeps the train/val partition (and the dataset
    fingerprint) invariant to files being moved on disk between runs."""
    return str(hashes.get(path) or path)


def _gather_images(request: dict) -> list[str]:
    """Enumerate + dedup + sort the pool of candidate image paths.

    ``images`` are taken as-is; ``image_roots`` are walked recursively for
    known image suffixes. Any filename containing ``-masklabel`` is skipped
    (house dataset hygiene). The union is deduped by ``content_hashes[path]``
    when provided, else by the resolved absolute path, and returned sorted for
    determinism BEFORE any hashing/splitting.
    """
    hashes = dict(request.get("content_hashes") or {})
    candidates: list[str] = [str(p) for p in (request.get("images") or [])]
    for root in request.get("image_roots") or []:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        for entry in root_path.rglob("*"):
            if (
                entry.suffix.lower() in _IMAGE_SUFFIXES
                and "-masklabel" not in entry.name.lower()
            ):
                candidates.append(str(entry))

    seen: dict[str, str] = {}
    for path in candidates:
        if "-masklabel" in Path(path).name.lower():
            continue
        key = hashes.get(path) or str(Path(path).resolve())
        if key in seen:
            continue
        seen[key] = path
    return sorted(seen.values())


def _dataset_fingerprint(paths: list[str], hashes: dict[str, str]) -> str:
    """xxhash64 hex over the sorted partition keys — the dataset-membership id."""
    keys = sorted(_split_key(p, hashes) for p in paths)
    return xxhash.xxh64("\n".join(keys).encode("utf-8")).hexdigest()


def _split_paths(
    paths: list[str], hashes: dict[str, str], val_fraction: float, val_max: int
) -> tuple[list[str], list[str]]:
    """Deterministic held-out slice by hashed partition key (D7).

    ``xxh64(key) % 10_000 / 10_000 < val_fraction`` selects the val slice, so
    re-runs (even with moved files, when ``content_hashes`` is supplied) draw
    the identical partition. NEVER python ``hash()`` (salted per process). The
    val list is truncated to ``val_max`` deterministically by the same digest.
    """
    if val_fraction <= 0:
        return list(paths), []
    train: list[str] = []
    val: list[tuple[int, str]] = []
    for path in paths:
        digest = xxhash.xxh64(_split_key(path, hashes).encode("utf-8")).intdigest()
        if (digest % 10_000) / 10_000 < val_fraction:
            val.append((digest, path))
        else:
            train.append(path)
    val.sort(key=lambda item: (item[0], item[1]))
    val_paths = [path for _digest, path in val[: max(0, int(val_max))]]
    return train, val_paths


# --------------------------------------------------------------------------- #
# Masking / patchify / loss                                                    #
# --------------------------------------------------------------------------- #


def make_random_masks(
    batch: int,
    grid_h: int,
    grid_w: int,
    mask_ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a ``(B, 1, gh, gw)`` float visible-mask (1=visible, 0=masked).

    EXACTLY ``int(round(mask_ratio * gh * gw))`` positions are masked per sample
    (chosen via ``torch.randperm`` so the masked fraction is exact, not
    Bernoulli), honouring ``generator`` when given for deterministic val masks.
    """
    length = grid_h * grid_w
    n_mask = int(round(mask_ratio * length))
    mask = torch.ones(batch, length)
    if n_mask > 0:
        for b in range(batch):
            idx = torch.randperm(length, generator=generator)[:n_mask]
            mask[b, idx] = 0.0
    return mask.reshape(batch, 1, grid_h, grid_w)


def _upsample_mask(grid_mask: torch.Tensor, scale: int) -> torch.Tensor:
    """Nearest-neighbour upsample a grid mask by ``scale`` along H and W."""
    return grid_mask.repeat_interleave(scale, dim=-2).repeat_interleave(scale, dim=-1)


def _patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, 3, H, W) -> (B, L, p*p*3) with L row-major over the patch grid (MAE)."""
    b, c, h, w = images.shape
    p = patch_size
    gh, gw = h // p, w // p
    x = images.reshape(b, c, gh, p, gw, p)
    x = x.permute(0, 2, 4, 3, 5, 1)  # B, gh, gw, p, p, C
    return x.reshape(b, gh * gw, p * p * c)


def fcmae_loss(
    pred: torch.Tensor,
    images: torch.Tensor,
    grid_mask: torch.Tensor,
    *,
    patch_size: int = _PATCH_SIZE,
    norm_pix: bool = True,
) -> torch.Tensor:
    """Normalized-pixel MSE on MASKED patches only (D3).

    ``pred`` is the decoder output ``(B, p*p*3, gh, gw)``; ``images`` are the
    ORIGINAL (un-zeroed, ImageNet-normalized) inputs; ``grid_mask`` is the
    ``(B, 1, gh, gw)`` visible mask (1=visible). Computed in fp32 even under
    autocast. Visible patches contribute exactly zero.
    """
    target = _patchify(images.float(), patch_size)  # (B, L, ppc)
    if norm_pix:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / torch.sqrt(var + 1e-6)
    pred_patches = pred.float().flatten(2).transpose(1, 2)  # (B, L, ppc)
    per_patch = ((pred_patches - target) ** 2).mean(dim=-1)  # (B, L)
    masked = (1.0 - grid_mask).reshape(grid_mask.shape[0], -1)  # 1 on MASKED
    denom = masked.sum().clamp(min=1.0)
    return (per_patch * masked).sum() / denom


# --------------------------------------------------------------------------- #
# Decoder + wrapper module                                                     #
# --------------------------------------------------------------------------- #


class _FcmaeDecoder(nn.Module):
    """Lightweight FCMAE decoder (single ConvNeXt V2 block, per the paper).

    Projects the masked encoder features to ``decoder_dim``, substitutes a
    learned mask token at masked positions, refines with one GRN block and
    predicts per-patch pixels. Output is ``(B, p*p*3, gh, gw)``.
    """

    def __init__(
        self, in_features: int, decoder_dim: int = 512, patch_size: int = _PATCH_SIZE
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_features, decoder_dim, kernel_size=1)
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_dim, 1, 1))
        self.block = ConvNeXtBlock(decoder_dim, use_grn=True)
        self.pred = nn.Conv2d(decoder_dim, patch_size * patch_size * 3, kernel_size=1)
        trunc_normal_(self.mask_token, std=0.02)

    def forward(self, feat: torch.Tensor, grid_mask: torch.Tensor) -> torch.Tensor:
        x = self.proj(feat)
        # Out-of-place: keep visible projections, insert the mask token elsewhere.
        x = x * grid_mask + self.mask_token * (1.0 - grid_mask)
        x = self.block(x)
        return self.pred(x)


class _FcmaeModel(nn.Module):
    """Encoder (timm ConvNeXt V2, num_classes=0) + FCMAE decoder as one module.

    One module so the backup envelope carries a single ``state_dict`` (house
    pattern: ``_BackboneWithHeads``). ``encoder`` holds BARE timm keys under the
    ``encoder.`` namespace; export strips that namespace. The decoder is never
    exported (D6).
    """

    def __init__(
        self, encoder: nn.Module, decoder: _FcmaeDecoder, patch_size: int = _PATCH_SIZE
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.patch_size = patch_size

    def forward_masked(
        self, images: torch.Tensor, grid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Masked forward: zero masked input patches, re-mask features after the
        stem and every stage, then decode. All multiplies are out-of-place."""
        enc = self.encoder
        x = images * _upsample_mask(grid_mask, images.shape[-1] // grid_mask.shape[-1])
        x = enc.stem(x)
        x = x * _upsample_mask(grid_mask, x.shape[-1] // grid_mask.shape[-1])
        for stage in enc.stages:
            x = stage(x)
            x = x * _upsample_mask(grid_mask, x.shape[-1] // grid_mask.shape[-1])
        return self.decoder(x, grid_mask)


# --------------------------------------------------------------------------- #
# Data pipeline                                                                #
# --------------------------------------------------------------------------- #


_WARNED_PATHS: set[str] = set()


class _FcmaeDataset(Dataset):
    """Unlabeled image dataset; unreadable files log once and yield ``None``."""

    def __init__(self, paths: list[str], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        try:
            with Image.open(path) as img:
                return self.transform(img.convert("RGB")), path
        except (OSError, ValueError, Image.DecompressionBombError, SyntaxError):
            if path not in _WARNED_PATHS:
                _WARNED_PATHS.add(path)
                logger.warning(
                    "FCMAE: skipping unreadable image %s", path, exc_info=True
                )
            return None


def _collate_drop_none(batch):
    """Drop failed samples; return ``(images, paths)`` or ``None`` if all failed.

    The path list rides alongside the stacked tensor so validation can seed a
    deterministic per-image mask (D7); ``train_epoch`` ignores it.
    """
    items = [item for item in batch if item is not None]
    if not items:
        return None
    images = torch.stack([item[0] for item in items])
    paths = [item[1] for item in items]
    return images, paths


def _train_transform(image_size: int) -> transforms.Compose:
    """MAE-style, augmentation-light: RandomResizedCrop + hflip (no colour jitter)."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            bb._NORMALIZE,
        ]
    )


def _val_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            bb._NORMALIZE,
        ]
    )


def val_mask_generator(path: str) -> torch.Generator:
    """A CPU generator seeded from the image path so val masks are stable across
    epochs (epoch-to-epoch val losses stay comparable)."""
    generator = torch.Generator()
    generator.manual_seed(xxhash.xxh64(path.encode("utf-8")).intdigest() & 0x7FFFFFFF)
    return generator


# --------------------------------------------------------------------------- #
# Optimizer / schedule / fingerprint helpers                                   #
# --------------------------------------------------------------------------- #


def _param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Two AdamW groups: standard MAE exclusion of norms/biases/mask-token from wd."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith("mask_token"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups: list[dict] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def warmup_cosine_lambda(warmup_epochs: int, epochs: int):
    """Epoch-granular linear-warmup-then-cosine multiplier for ``LambdaLR`` (D4)."""
    warmup = max(0, int(warmup_epochs))
    total = int(epochs)

    def _multiplier(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return _multiplier


def _fcmae_fingerprint(
    model_size: str,
    epochs: int,
    *,
    mask_ratio: float,
    image_size: int,
    dataset_fingerprint: str,
    dataset_count: int,
    steps_per_epoch: int,
) -> dict:
    """Resume-compatibility identity (D8): any change to mask ratio, resolution,
    model size, dataset membership, chunking or epoch budget cleanly mismatches a
    stale backup (``resume_skipped`` -> fresh start)."""
    fingerprint = make_fingerprint(
        class_names=[],
        num_classes=0,
        max_epochs=int(epochs),
        multi_label=False,
        ordinal=False,
        best_model_name="fcmae_candidate",
        model_size=str(model_size),
    )
    fingerprint["trainer"] = "fcmae"
    fingerprint["optimizer"] = "AdamW"
    fingerprint["mask_ratio"] = float(mask_ratio)
    fingerprint["resolution"] = str(image_size)
    fingerprint["dataset"] = f"{dataset_fingerprint}:{dataset_count}"
    fingerprint["steps_per_epoch"] = int(steps_per_epoch)
    return fingerprint


# --------------------------------------------------------------------------- #
# Worker + async entry point                                                   #
# --------------------------------------------------------------------------- #


def _train_fcmae(
    request: dict,
    emit,
    stop: threading.Event,
    pause_event: object | None = None,
    stop_event: object | None = None,
    stop_now_event: object | None = None,
) -> dict:
    """Worker-thread target: drive :class:`GenericTrainer` with an ``FcmaeTask``.

    Same shape as ``bb._train_backbone``. ``stop`` is the cancellation event
    (raised as :class:`FcmaeTrainingCancelled` inside the epoch loop);
    ``pause_event`` rides the generic backup/pause machinery.
    """
    from bittrainer.generic.generic_trainer import GenericTrainer
    from bittrainer.generic.tasks.fcmae_task import FcmaeTask

    task = FcmaeTask(request, cancel_event=stop, stop_event=stop_event)
    return GenericTrainer().run(
        task,
        progress_callback=emit,
        pause_event=pause_event,
        stop_event=task.steps_stop_event,
        stop_now_event=stop_now_event,
    )


async def run_fcmae_pretraining(
    request: dict,
    progress_callback=None,
    *,
    pause_event=None,
    stop_event=None,
    stop_now_event=None,
) -> dict:
    """Pretrain a ConvNeXt V2 trunk with FCMAE; see the module docstring.

    Async by the same worker-thread + queue pattern as ``run_backbone_training``:
    the torch loop runs off the event loop and progress dicts are marshalled back
    to the caller. ``pause_event`` backs up and returns ``{"paused": True, ...}``
    without finalising; ``stop_event`` / ``stop_now_event`` finish early but still
    export the best (or final) trunk.
    """
    return await bb._run_worker_async(
        _train_fcmae,
        request,
        progress_callback,
        pause_event,
        stop_event,
        stop_now_event,
    )
