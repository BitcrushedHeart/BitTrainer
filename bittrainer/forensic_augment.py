"""CPU-side forensic augmentation, applied per sample in the DataLoader workers.

The forensic chain (blur -> noise -> JPEG round-trip) exists so a realism
ranker cannot learn "sharp high-frequency detail", "sensor grain" or "JPEG
artifact" as a stand-in for "photograph": both classes see the same
augmentation, so the cue carries no signal (Bitcrush ISSUE-0847).

It used to run on the GPU batch inside :func:`bittrainer.gpu_augment.
apply_train_augment`, on the trainer's MAIN thread, between fetching a batch
and launching the forward pass — and the JPEG leg is a CPU codec regardless,
so every selected sample was copied device->host, encoded, decoded and copied
back while the GPU idled. On a live pico@800/batch-31 run that chain was 26.6%
of main-thread wall (JPEG alone 18.9%) and held the GPU at 57% utilisation,
while the six persistent DataLoader workers sat nearly idle (Bitcrush
ISSUE-0861). Doing the same work per-sample in the workers costs nothing on
the critical path.

Ordering note
-------------
Because it now runs in the worker, the forensic chain happens BEFORE the GPU
RandAugment/flip/colour-jitter instead of after RandAugment. That is
acceptable — and arguably more faithful: a real photograph is captured,
noised and compressed first, and any photometric editing happens to the
already-compressed image. Within the chain the original order is preserved:
blur, then noise, then JPEG, so JPEG *quantises* the grain rather than sitting
under it, exactly as a camera produces it.

Randomness
----------
All draws use the global ``torch`` RNG (``torch.rand`` / ``torch.randint``).
PyTorch's DataLoader seeds each worker's ``torch`` RNG from the base seed plus
the worker id and epoch, so per-worker streams diverge without any extra
plumbing. (A ``numpy.random`` global would be shared identically across forked
workers — hence torch, never numpy, here.) With ``num_workers=0`` the
transform simply runs in-process.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ForensicAugment:
    """Per-sample blur/noise/JPEG augmentation for a uint8 CHW tensor.

    Every probability defaults to 0, which makes the transform an exact no-op
    (:attr:`is_active` is ``False`` and ``__call__`` returns the input object
    unchanged), so existing groups train bit-identically.

    Frozen and made only of scalars, so it pickles cleanly into spawned
    Windows DataLoader workers.
    """

    blur_p: float = 0.0
    blur_sigma_max: float = 1.5
    noise_p: float = 0.0
    noise_std: float = 0.03
    jpeg_p: float = 0.0
    jpeg_quality_min: int = 50
    jpeg_quality_max: int = 95

    @property
    def is_active(self) -> bool:
        """True when at least one op can actually fire."""
        return (
            (self.blur_p > 0.0)
            or (self.noise_p > 0.0 and self.noise_std > 0.0)
            or (self.jpeg_p > 0.0)
        )

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply the chain to one uint8 CHW tensor, returning uint8 CHW.

        Disabled ops draw nothing at all (not even their Bernoulli), so turning
        one off never shifts the others' RNG stream.
        """
        if not self.is_active:
            return img
        img = self._blur(img)
        img = self._noise(img)
        return self._jpeg(img)

    # -- ops ------------------------------------------------------------

    def _blur(self, img: torch.Tensor) -> torch.Tensor:
        if self.blur_p <= 0.0 or not _fires(self.blur_p):
            return img
        from torchvision.transforms.v2 import functional as TVF

        sigma_max = max(0.1, float(self.blur_sigma_max))
        sigma = float(torch.empty(1).uniform_(0.1, sigma_max).item())
        k = 2 * int(math.ceil(3.0 * sigma)) + 1
        return TVF.gaussian_blur(img, kernel_size=[k, k], sigma=[sigma, sigma])

    def _noise(self, img: torch.Tensor) -> torch.Tensor:
        if self.noise_p <= 0.0 or self.noise_std <= 0.0 or not _fires(self.noise_p):
            return img
        noise = torch.randn(img.shape, dtype=torch.float32) * (float(self.noise_std) * 255.0)
        return img.float().add_(noise).clamp_(0.0, 255.0).round_().to(torch.uint8)

    def _jpeg(self, img: torch.Tensor) -> torch.Tensor:
        if self.jpeg_p <= 0.0 or not _fires(self.jpeg_p):
            return img
        from torchvision.io import decode_jpeg, encode_jpeg

        q_lo, q_hi = int(self.jpeg_quality_min), int(self.jpeg_quality_max)
        if q_hi < q_lo:
            q_lo, q_hi = q_hi, q_lo
        quality = int(torch.randint(q_lo, q_hi + 1, (1,)).item())
        return decode_jpeg(encode_jpeg(img.contiguous(), quality=quality))


def _fires(p: float) -> bool:
    """One Bernoulli draw off the worker-local torch RNG."""
    return bool(float(torch.rand(1).item()) < p)


def forensic_from_config(config: object) -> ForensicAugment:
    """Build the transform from a train config's ``aug_*`` knobs.

    Accepts anything exposing the :class:`~bittrainer.group_trainer.
    GroupTrainConfig` forensic field names, so trainers stay decoupled from
    this module's field ordering.
    """
    return ForensicAugment(
        blur_p=float(getattr(config, "aug_blur_p", 0.0) or 0.0),
        blur_sigma_max=float(getattr(config, "aug_blur_sigma_max", 1.5) or 1.5),
        noise_p=float(getattr(config, "aug_noise_p", 0.0) or 0.0),
        noise_std=float(getattr(config, "aug_noise_std", 0.03) or 0.0),
        jpeg_p=float(getattr(config, "aug_jpeg_p", 0.0) or 0.0),
        jpeg_quality_min=int(getattr(config, "aug_jpeg_quality_min", 50) or 50),
        jpeg_quality_max=int(getattr(config, "aug_jpeg_quality_max", 95) or 95),
    )
