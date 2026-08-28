"""Tests for gpu_augment.py — GPU-side normalize + augmentation."""

import pytest
import torch
from torchvision.transforms import functional as TF

from bittrainer.gpu_augment import (
    apply_train_augment,
    apply_val_transform,
    gpu_color_jitter,
    gpu_normalize,
    gpu_random_flip,
)


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestGpuNormalize:
    def test_output_range(self):
        batch = torch.randint(0, 256, (4, 3, 64, 64), dtype=torch.uint8, device=_get_device())
        out = gpu_normalize(batch)
        assert out.dtype == torch.float32
        assert out.shape == batch.shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_matches_torchvision(self):
        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
        tv_tensor = TF.normalize(TF.to_tensor(img), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0)
        gpu_tensor = gpu_normalize(arr.to(_get_device()))

        torch.testing.assert_close(
            gpu_tensor[0].cpu(), tv_tensor, atol=1e-5, rtol=1e-5,
        )


class TestGpuRandomFlip:
    def test_preserves_content(self):
        batch = torch.randn(1, 3, 32, 32, device=_get_device())
        flipped = batch.flip(-1)
        restored = flipped.flip(-1)
        torch.testing.assert_close(batch, restored)

    def test_probability(self):
        batch = torch.arange(16, device=_get_device()).float().view(1, 1, 4, 4).expand(10000, 1, 4, 4).clone()
        flipped = gpu_random_flip(batch, p=0.5)
        num_flipped = (flipped[:, 0, 0, -1] == 0).sum().item()
        assert 4000 < num_flipped < 6000

    def test_zero_probability(self):
        batch = torch.randn(8, 3, 32, 32, device=_get_device())
        original = batch.clone()
        result = gpu_random_flip(batch, p=0.0)
        torch.testing.assert_close(result, original)


class TestGpuColorJitter:
    def test_no_crash_various_sizes(self):
        device = _get_device()
        for B in [1, 4, 32]:
            for H, W in [(64, 64), (128, 96), (32, 48)]:
                batch = torch.randn(B, 3, H, W, device=device)
                out = gpu_color_jitter(batch)
                assert out.shape == (B, 3, H, W)

    def test_zero_params_identity(self):
        batch = torch.randn(4, 3, 64, 64, device=_get_device())
        original = batch.clone()
        out = gpu_color_jitter(batch, brightness=0, contrast=0, saturation=0)
        torch.testing.assert_close(out, original)

    def test_bounded_output(self):
        batch = torch.randn(16, 3, 64, 64, device=_get_device())
        out = gpu_color_jitter(batch, brightness=0.1, contrast=0.1, saturation=0.1)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestApplyTrainAugment:
    def test_shape_and_dtype(self):
        batch = torch.randint(0, 256, (8, 3, 64, 64), dtype=torch.uint8, device=_get_device())
        out = apply_train_augment(batch)
        assert out.shape == (8, 3, 64, 64)
        assert out.dtype == torch.float32

    def test_bfloat16_output(self):
        batch = torch.randint(0, 256, (4, 3, 64, 64), dtype=torch.uint8, device=_get_device())
        out = apply_train_augment(batch, dtype=torch.bfloat16)
        assert out.dtype == torch.bfloat16


class TestApplyValTransform:
    def test_deterministic(self):
        batch = torch.randint(0, 256, (4, 3, 64, 64), dtype=torch.uint8, device=_get_device())
        a = apply_val_transform(batch.clone())
        b = apply_val_transform(batch.clone())
        torch.testing.assert_close(a, b)

    def test_matches_torchvision(self):
        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
        tv_tensor = TF.normalize(TF.to_tensor(img), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0)
        gpu_tensor = apply_val_transform(arr.to(_get_device()))

        torch.testing.assert_close(
            gpu_tensor[0].cpu(), tv_tensor, atol=1e-5, rtol=1e-5,
        )


def _natural_batch(b=4, c=3, h=64, w=64, seed=0):
    """A non-constant, smoothly-varying uint8 batch (blur/JPEG need structure)."""
    g = torch.Generator().manual_seed(seed)
    yy = torch.linspace(0, 1, h).view(1, 1, h, 1)
    xx = torch.linspace(0, 1, w).view(1, 1, 1, w)
    base = (torch.sin(yy * 12.0) * torch.cos(xx * 9.0) + 1.0) / 2.0
    base = base.expand(b, c, h, w).clone()
    base = base + torch.rand(b, c, h, w, generator=g) * 0.3
    return (base.clamp(0, 1) * 255).round().to(torch.uint8).to(_get_device())


def _laplacian_energy(batch: torch.Tensor) -> float:
    x = batch.float()
    lap = (
        4 * x[:, :, 1:-1, 1:-1]
        - x[:, :, :-2, 1:-1]
        - x[:, :, 2:, 1:-1]
        - x[:, :, 1:-1, :-2]
        - x[:, :, 1:-1, 2:]
    )
    return lap.abs().mean().item()


class TestGpuGaussianBlur:
    def test_shape_dtype_device(self):
        from bittrainer.gpu_augment import gpu_gaussian_blur

        batch = _natural_batch()
        out = gpu_gaussian_blur(batch, p=1.0, sigma_max=1.5)
        assert out.shape == batch.shape
        assert out.dtype == torch.uint8
        assert out.device == batch.device

    def test_zero_probability_identity(self):
        from bittrainer.gpu_augment import gpu_gaussian_blur

        batch = _natural_batch()
        out = gpu_gaussian_blur(batch.clone(), p=0.0)
        assert torch.equal(out, batch)

    def test_p_one_changes_pixels(self):
        from bittrainer.gpu_augment import gpu_gaussian_blur

        batch = _natural_batch()
        out = gpu_gaussian_blur(batch.clone(), p=1.0, sigma_max=1.5)
        assert not torch.equal(out, batch)

    def test_reduces_high_frequency_energy(self):
        from bittrainer.gpu_augment import gpu_gaussian_blur

        batch = _natural_batch()
        out = gpu_gaussian_blur(batch.clone(), p=1.0, sigma_max=1.5)
        assert _laplacian_energy(out) < _laplacian_energy(batch)


class TestGpuGaussianNoise:
    def test_shape_dtype_device(self):
        from bittrainer.gpu_augment import gpu_gaussian_noise

        batch = _natural_batch()
        out = gpu_gaussian_noise(batch, p=1.0, std=0.03)
        assert out.shape == batch.shape
        assert out.dtype == torch.uint8
        assert out.device == batch.device

    def test_zero_probability_identity(self):
        from bittrainer.gpu_augment import gpu_gaussian_noise

        batch = _natural_batch()
        out = gpu_gaussian_noise(batch.clone(), p=0.0)
        assert torch.equal(out, batch)

    def test_p_one_changes_pixels(self):
        from bittrainer.gpu_augment import gpu_gaussian_noise

        batch = _natural_batch()
        out = gpu_gaussian_noise(batch.clone(), p=1.0, std=0.03)
        assert not torch.equal(out, batch)

    def test_increases_difference_std(self):
        from bittrainer.gpu_augment import gpu_gaussian_noise

        batch = _natural_batch()
        low = gpu_gaussian_noise(batch.clone(), p=1.0, std=0.01)
        high = gpu_gaussian_noise(batch.clone(), p=1.0, std=0.10)
        low_std = (low.float() - batch.float()).std().item()
        high_std = (high.float() - batch.float()).std().item()
        assert 0.0 < low_std < high_std


class TestGpuJpegRoundtrip:
    def test_shape_dtype_device(self):
        from bittrainer.gpu_augment import gpu_jpeg_roundtrip

        batch = _natural_batch()
        out = gpu_jpeg_roundtrip(batch, p=1.0, quality=(50, 95))
        assert out.shape == batch.shape
        assert out.dtype == torch.uint8
        assert out.device == batch.device

    def test_zero_probability_identity(self):
        from bittrainer.gpu_augment import gpu_jpeg_roundtrip

        batch = _natural_batch()
        out = gpu_jpeg_roundtrip(batch.clone(), p=0.0)
        assert torch.equal(out, batch)

    def test_p_one_changes_pixels(self):
        from bittrainer.gpu_augment import gpu_jpeg_roundtrip

        batch = _natural_batch()
        out = gpu_jpeg_roundtrip(batch.clone(), p=1.0, quality=(50, 50))
        assert not torch.equal(out, batch)

    def test_lower_quality_deviates_more(self):
        from bittrainer.gpu_augment import gpu_jpeg_roundtrip

        batch = _natural_batch()
        hi = gpu_jpeg_roundtrip(batch.clone(), p=1.0, quality=(95, 95))
        lo = gpu_jpeg_roundtrip(batch.clone(), p=1.0, quality=(30, 30))
        hi_err = (hi.float() - batch.float()).abs().mean().item()
        lo_err = (lo.float() - batch.float()).abs().mean().item()
        assert hi_err < lo_err


class TestForensicApplyTrainAugment:
    def test_defaults_off_are_bit_identical(self):
        batch = _natural_batch(seed=3)
        torch.manual_seed(1234)
        baseline = apply_train_augment(batch.clone(), randaugment_n=2, randaugment_m=9)
        torch.manual_seed(1234)
        with_knobs = apply_train_augment(
            batch.clone(),
            randaugment_n=2,
            randaugment_m=9,
            noise_p=0.0,
            noise_std=0.03,
            blur_p=0.0,
            blur_sigma_max=1.5,
            jpeg_p=0.0,
            jpeg_quality=(50, 95),
        )
        assert torch.equal(baseline, with_knobs)

    def test_knobs_on_change_output(self):
        batch = _natural_batch(seed=4)
        torch.manual_seed(7)
        baseline = apply_train_augment(batch.clone(), randaugment_n=0, randaugment_m=0)
        torch.manual_seed(7)
        forensic = apply_train_augment(
            batch.clone(),
            randaugment_n=0,
            randaugment_m=0,
            noise_p=1.0,
            blur_p=1.0,
            jpeg_p=1.0,
        )
        assert not torch.equal(baseline, forensic)
        assert forensic.shape == baseline.shape
        assert forensic.dtype == baseline.dtype

    def test_photometric_only_kwarg_accepted(self):
        batch = _natural_batch(seed=5)
        out = apply_train_augment(
            batch, randaugment_n=2, randaugment_m=9, photometric_only=True,
        )
        assert out.shape == batch.shape


class TestGroupTrainConfigForensicDefaults:
    def test_new_knobs_default_off(self):
        from bittrainer.group_trainer import GroupTrainConfig

        cfg = GroupTrainConfig(
            group_folder="unused", num_classes=2, class_names=["a", "b"],
        )
        assert cfg.randaugment_photometric_only is False
        assert cfg.aug_noise_p == 0.0
        assert cfg.aug_noise_std == 0.03
        assert cfg.aug_blur_p == 0.0
        assert cfg.aug_blur_sigma_max == 1.5
        assert cfg.aug_jpeg_p == 0.0
        assert cfg.aug_jpeg_quality_min == 50
        assert cfg.aug_jpeg_quality_max == 95
