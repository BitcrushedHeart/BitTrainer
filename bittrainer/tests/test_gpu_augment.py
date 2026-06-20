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
