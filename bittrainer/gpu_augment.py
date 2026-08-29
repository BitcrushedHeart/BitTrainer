"""GPU-side image normalisation and augmentation.

Operates on batched uint8 CHW tensors already on GPU.  Replaces the
CPU-side torchvision transforms pipeline used by DataLoader workers.
"""

from __future__ import annotations

import math

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


def gpu_gaussian_blur(
    batch: torch.Tensor, p: float = 0.0, sigma_max: float = 1.5,
) -> torch.Tensor:
    """Per-sample Gaussian blur on a uint8 CHW batch.

    Forensic augmentation (ISSUE-0847): a realism ranker must not be able to
    read "sharp high-frequency detail = photograph". Sigma is drawn per sample
    from U(0.1, ``sigma_max``); the kernel is the odd size 2*ceil(3*sigma)+1.
    Samples whose Bernoulli draw fails are returned untouched.
    """
    if p <= 0.0:
        return batch
    from torchvision.transforms.v2 import functional as TVF

    mask = torch.rand(batch.shape[0], device=batch.device) < p
    if not bool(mask.any()):
        return batch
    for i in range(batch.shape[0]):
        if not bool(mask[i]):
            continue
        sigma = float(torch.empty(1).uniform_(0.1, max(0.1, sigma_max)).item())
        k = 2 * int(math.ceil(3.0 * sigma)) + 1
        batch[i] = TVF.gaussian_blur(batch[i], kernel_size=[k, k], sigma=[sigma, sigma])
    return batch


def gpu_gaussian_noise(
    batch: torch.Tensor, p: float = 0.0, std: float = 0.03,
) -> torch.Tensor:
    """Per-sample additive Gaussian noise on a uint8 CHW batch.

    ``std`` is a fraction of the 0-1 range and is applied as ``std * 255`` on
    the uint8 scale, then clamped and rounded back to uint8. Sensor noise is
    the other half of the shortcut a realism ranker would otherwise learn
    ("grain = photograph"), so both classes get it.
    """
    if p <= 0.0 or std <= 0.0:
        return batch
    mask = torch.rand(batch.shape[0], device=batch.device) < p
    if not bool(mask.any()):
        return batch
    sel = batch[mask].float()
    noise = torch.randn(sel.shape, device=batch.device, dtype=torch.float32) * (std * 255.0)
    batch[mask] = sel.add_(noise).clamp_(0.0, 255.0).round_().to(torch.uint8)
    return batch


def gpu_jpeg_roundtrip(
    batch: torch.Tensor, p: float = 0.0, quality: tuple[int, int] = (50, 95),
) -> torch.Tensor:
    """Per-sample JPEG encode/decode round-trip on a uint8 CHW batch.

    Quality is drawn per sample from U[quality[0], quality[1]]. Runs after the
    noise so JPEG *quantises* the grain rather than sitting under it — the same
    order a camera produces. Tries the batched on-device torchvision path and
    falls back to per-sample CPU encode/decode when the device rejects it.
    """
    if p <= 0.0:
        return batch
    from torchvision.io import decode_jpeg, encode_jpeg

    mask = torch.rand(batch.shape[0], device=batch.device) < p
    idx = [i for i in range(batch.shape[0]) if bool(mask[i])]
    if not idx:
        return batch
    q_lo, q_hi = int(quality[0]), int(quality[1])
    if q_hi < q_lo:
        q_lo, q_hi = q_hi, q_lo
    qualities = [int(torch.randint(q_lo, q_hi + 1, (1,)).item()) for _ in idx]

    # Encode on CPU: the batched ``decode_jpeg(list, device=cuda)`` path
    # requires CPU-resident encoded bytes (torchvision raises ValueError
    # "Input list must contain tensors on CPU" otherwise — which is exactly
    # what a CUDA-encoded list produced in the first live run, Bitcrush
    # ISSUE-0847). Decoding lands on the batch's device.
    imgs = [batch[i].cpu() for i in idx]
    encoded = [encode_jpeg(img, quality=q) for img, q in zip(imgs, qualities)]
    try:
        decoded = decode_jpeg(encoded, device=batch.device)
    except Exception:  # noqa: BLE001 — any device-path failure falls back to CPU decode
        decoded = [decode_jpeg(e).to(batch.device) for e in encoded]
    for i, out in zip(idx, decoded):
        batch[i] = out
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
    noise_p: float = 0.0,
    noise_std: float = 0.03,
    blur_p: float = 0.0,
    blur_sigma_max: float = 1.5,
    jpeg_p: float = 0.0,
    jpeg_quality: tuple[int, int] = (50, 95),
) -> torch.Tensor:
    """Normalize uint8 batch and apply training augmentation on GPU.

    When ``randaugment_m > 0`` RandAugment runs on uint8 before normalisation.
    When ``random_erasing_p > 0`` RandomErasing runs on the normalised float
    tensor after the existing colour jitter. ``memory_format`` converts the
    final tensor (e.g. channels_last) so the model forward never permutes.

    Spatial groups pass ``hflip=False`` (the trainer flips label-aware via
    ``spatial_hflip_batch`` instead) and ``photometric_only=True`` (geometric
    RandAugment ops would move the subject relative to the frame).

    The forensic knobs (``blur_p``/``noise_p``/``jpeg_p``, all off by default)
    run on the uint8 batch after RandAugment in the order blur -> noise ->
    JPEG: a camera adds sensor noise and *then* compresses, so JPEG must
    quantise the noise rather than sit under it. Each op draws an independent
    Bernoulli per sample. They exist so a realism ranker cannot learn
    "high-frequency noise" or "JPEG artifact" as a stand-in for "photograph" —
    both classes see the same augmentation.
    """
    if randaugment_m > 0 and randaugment_n > 0:
        batch = gpu_randaugment(
            batch, randaugment_n, randaugment_m, photometric_only=photometric_only,
        )
    batch = gpu_gaussian_blur(batch, p=blur_p, sigma_max=blur_sigma_max)
    batch = gpu_gaussian_noise(batch, p=noise_p, std=noise_std)
    batch = gpu_jpeg_roundtrip(batch, p=jpeg_p, quality=jpeg_quality)
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
