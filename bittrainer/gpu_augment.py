"""GPU-side image normalisation and augmentation.

Operates on batched uint8 CHW tensors already on GPU.  Replaces the
CPU-side torchvision transforms pipeline used by DataLoader workers.
"""

from __future__ import annotations

import torch

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

_randaugment_cache: dict[tuple[int, int, bool], object] = {}

# RandAugment ops that move pixels rather than recolour them. Spatial groups
# (label = where the subject sits in the frame) must not see these: a translate
# or shear changes the true label while the target stays fixed.
_GEOMETRIC_RA_OPS = ("ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate")


def _make_photometric_randaugment(num_ops: int, magnitude: int):
    from torchvision.transforms import v2

    class _PhotometricRandAugment(v2.RandAugment):
        _AUGMENTATION_SPACE = {
            k: v
            for k, v in v2.RandAugment._AUGMENTATION_SPACE.items()
            if k not in _GEOMETRIC_RA_OPS
        }

    return _PhotometricRandAugment(num_ops=num_ops, magnitude=magnitude)


def _get_randaugment(num_ops: int, magnitude: int, photometric_only: bool = False):
    key = (num_ops, magnitude, photometric_only)
    if key not in _randaugment_cache:
        if photometric_only:
            _randaugment_cache[key] = _make_photometric_randaugment(num_ops, magnitude)
        else:
            from torchvision.transforms import v2
            _randaugment_cache[key] = v2.RandAugment(num_ops=num_ops, magnitude=magnitude)
    return _randaugment_cache[key]


def gpu_randaugment(
    batch: torch.Tensor, num_ops: int, magnitude: int, *, photometric_only: bool = False,
) -> torch.Tensor:
    """Apply RandAugment to a uint8 CHW batch.

    torchvision.v2.RandAugment processes per-sample even when given a batch
    dimension, but kernel ops stay on-device. Returns uint8 batch with the
    same shape and dtype. ``photometric_only`` drops the geometric ops
    (shear/translate/rotate) for label-geometry-sensitive (spatial) groups.
    """
    ra = _get_randaugment(num_ops, magnitude, photometric_only)
    # v2.RandAugment expects [..., C, H, W] uint8; iterate the batch to ensure
    # each sample receives an independent draw of ops + magnitude.
    out = torch.empty_like(batch)
    for i in range(batch.shape[0]):
        out[i] = ra(batch[i])
    return out


def gpu_random_erasing(
    batch: torch.Tensor,
    p: float = 0.25,
    scale: tuple[float, float] = (0.02, 0.20),
    ratio: tuple[float, float] = (0.3, 3.3),
) -> torch.Tensor:
    """Per-image RandomErasing on a normalised float batch.

    Selects a rectangular patch per image (with probability ``p``), and zeroes
    it on the normalised tensor — equivalent to filling with the dataset mean
    after un-normalisation. Operates in-place for memory efficiency.
    """
    B, C, H, W = batch.shape
    device = batch.device
    area = H * W

    keep = torch.rand(B, device=device) >= p
    for i in range(B):
        if keep[i]:
            continue
        for _ in range(10):
            target_area = float(torch.empty(1).uniform_(*scale).item()) * area
            aspect = float(torch.empty(1).uniform_(*ratio).item())
            h = int(round((target_area * aspect) ** 0.5))
            w = int(round((target_area / aspect) ** 0.5))
            if 0 < h < H and 0 < w < W:
                top = int(torch.randint(0, H - h + 1, (1,)).item())
                left = int(torch.randint(0, W - w + 1, (1,)).item())
                batch[i, :, top:top + h, left:left + w] = 0.0
                break
    return batch


def gpu_normalize(
    batch: torch.Tensor,
    mean: torch.Tensor = _IMAGENET_MEAN,
    std: torch.Tensor = _IMAGENET_STD,
) -> torch.Tensor:
    """uint8 [B,3,H,W] → float32 [B,3,H,W], ImageNet-normalised."""
    mean = mean.to(batch.device, dtype=torch.float32).view(1, 3, 1, 1)
    std = std.to(batch.device, dtype=torch.float32).view(1, 3, 1, 1)
    return batch.float().div_(255.0).sub_(mean).div_(std)


def gpu_random_flip(batch: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Per-image random horizontal flip."""
    mask = torch.rand(batch.shape[0], device=batch.device) < p
    if mask.any():
        batch[mask] = batch[mask].flip(-1)
    return batch


def gpu_color_jitter(
    batch: torch.Tensor,
    brightness: float = 0.1,
    contrast: float = 0.1,
    saturation: float = 0.1,
) -> torch.Tensor:
    """Per-image random brightness, contrast, and saturation adjustment."""
    B = batch.shape[0]
    device = batch.device

    if brightness > 0:
        bf = 1.0 + (torch.rand(B, 1, 1, 1, device=device) * 2 * brightness - brightness)
        batch = batch * bf

    if contrast > 0:
        mean = batch.mean(dim=(2, 3), keepdim=True)
        cf = 1.0 + (torch.rand(B, 1, 1, 1, device=device) * 2 * contrast - contrast)
        batch = mean + (batch - mean) * cf

    if saturation > 0:
        gray = 0.2989 * batch[:, 0:1] + 0.5870 * batch[:, 1:2] + 0.1140 * batch[:, 2:3]
        sf = 1.0 + (torch.rand(B, 1, 1, 1, device=device) * 2 * saturation - saturation)
        batch = gray + (batch - gray) * sf

    return batch


def apply_train_augment(
    batch: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    *,
    randaugment_n: int = 0,
    randaugment_m: int = 0,
    random_erasing_p: float = 0.0,
    memory_format: torch.memory_format | None = None,
    hflip: bool = True,
    photometric_only: bool = False,
) -> torch.Tensor:
    """Normalize uint8 batch and apply training augmentation on GPU.

    When ``randaugment_m > 0`` RandAugment runs on uint8 before normalisation.
    When ``random_erasing_p > 0`` RandomErasing runs on the normalised float
    tensor after the existing colour jitter. ``memory_format`` converts the
    final tensor (e.g. channels_last) so the model forward never permutes.

    Spatial groups pass ``hflip=False`` (the trainer flips label-aware via
    ``spatial_hflip_batch`` instead) and ``photometric_only=True`` (geometric
    RandAugment ops would move the subject relative to the frame).
    """
    if randaugment_m > 0 and randaugment_n > 0:
        batch = gpu_randaugment(
            batch, randaugment_n, randaugment_m, photometric_only=photometric_only,
        )
    out = gpu_normalize(batch)
    if hflip:
        out = gpu_random_flip(out)
    out = gpu_color_jitter(out, brightness=0.1, contrast=0.1, saturation=0.1)
    if random_erasing_p > 0:
        out = gpu_random_erasing(out, p=random_erasing_p)
    if dtype != torch.float32:
        out = out.to(dtype=dtype)
    if memory_format is not None:
        out = out.contiguous(memory_format=memory_format)
    return out


def apply_val_transform(
    batch: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    memory_format: torch.memory_format | None = None,
) -> torch.Tensor:
    """Normalize uint8 batch for validation (no augmentation)."""
    out = gpu_normalize(batch)
    if dtype != torch.float32:
        out = out.to(dtype=dtype)
    if memory_format is not None:
        out = out.contiguous(memory_format=memory_format)
    return out
