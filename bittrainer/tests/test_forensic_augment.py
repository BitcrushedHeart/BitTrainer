"""CPU forensic augmentation runs in the DataLoader workers (Bitcrush ISSUE-0861).

The blur -> noise -> JPEG chain used to run on the trainer's main thread inside
``apply_train_augment`` (26.6% of main-thread wall on a live pico@800 run, JPEG
alone 18.9%), stalling the GPU between batch fetch and forward. It now runs
per-sample in ``GroupDataset.__getitem__`` — i.e. in the (idle) worker
processes.

CPU-only: no CUDA is touched, so this is safe to run alongside a live GPU train.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image

from bittrainer.forensic_augment import ForensicAugment
from bittrainer.group_dataset import GroupDataset

_ALL_ON = ForensicAugment(
    blur_p=1.0, blur_sigma_max=1.5,
    noise_p=1.0, noise_std=0.03,
    jpeg_p=1.0, jpeg_quality_min=40, jpeg_quality_max=60,
)


def _natural_image(h: int = 64, w: int = 64, seed: int = 0) -> torch.Tensor:
    """Non-constant, smoothly varying uint8 CHW tensor (blur/JPEG need structure)."""
    g = torch.Generator().manual_seed(seed)
    yy = torch.linspace(0, 1, h).view(1, h, 1)
    xx = torch.linspace(0, 1, w).view(1, 1, w)
    base = (torch.sin(yy * 12.0) * torch.cos(xx * 9.0) + 1.0) / 2.0
    base = base.expand(3, h, w).clone()
    base = base + torch.rand(3, h, w, generator=g) * 0.3
    return (base.clamp(0, 1) * 255).round().to(torch.uint8)


def _laplacian_energy(img: torch.Tensor) -> float:
    x = img.float().unsqueeze(0)
    lap = (
        4 * x[:, :, 1:-1, 1:-1]
        - x[:, :, :-2, 1:-1]
        - x[:, :, 2:, 1:-1]
        - x[:, :, 1:-1, :-2]
        - x[:, :, 1:-1, 2:]
    )
    return lap.abs().mean().item()


class TestIsActive:
    def test_all_probabilities_zero_is_inactive(self):
        assert ForensicAugment().is_active is False
        assert ForensicAugment(blur_sigma_max=3.0, noise_std=0.5).is_active is False

    def test_any_probability_positive_is_active(self):
        assert ForensicAugment(blur_p=0.1).is_active is True
        assert ForensicAugment(noise_p=0.1).is_active is True
        assert ForensicAugment(jpeg_p=0.1).is_active is True

    def test_inactive_call_is_the_same_object(self):
        img = _natural_image()
        assert ForensicAugment()(img) is img


class TestBlur:
    def test_p_one_changes_pixels_and_keeps_shape(self):
        img = _natural_image(seed=1)
        torch.manual_seed(0)
        out = ForensicAugment(blur_p=1.0)(img)
        assert out.dtype == torch.uint8
        assert out.shape == img.shape
        assert not torch.equal(out, img)

    def test_p_zero_draws_nothing(self):
        """A disabled op must not even draw its Bernoulli — zero overhead, and
        the remaining ops' draws stay in the same RNG positions."""
        img = _natural_image(seed=2)
        torch.manual_seed(21)
        both = ForensicAugment(blur_p=0.0, noise_p=1.0)(img)
        torch.manual_seed(21)
        noise_only = ForensicAugment(noise_p=1.0)(img)
        assert torch.equal(both, noise_only)

    def test_reduces_high_frequency_energy(self):
        img = _natural_image(seed=3)
        torch.manual_seed(0)
        out = ForensicAugment(blur_p=1.0, blur_sigma_max=1.5)(img)
        assert _laplacian_energy(out) < _laplacian_energy(img)


class TestNoise:
    def test_p_one_changes_pixels(self):
        img = _natural_image(seed=4)
        torch.manual_seed(0)
        out = ForensicAugment(noise_p=1.0, noise_std=0.03)(img)
        assert out.dtype == torch.uint8
        assert out.shape == img.shape
        assert not torch.equal(out, img)

    def test_stays_within_uint8_clamp(self):
        img = torch.full((3, 32, 32), 250, dtype=torch.uint8)
        torch.manual_seed(0)
        out = ForensicAugment(noise_p=1.0, noise_std=1.0)(img)
        assert out.dtype == torch.uint8
        assert int(out.min()) >= 0
        assert int(out.max()) <= 255

    def test_larger_std_deviates_more(self):
        img = _natural_image(seed=5)
        torch.manual_seed(0)
        low = ForensicAugment(noise_p=1.0, noise_std=0.01)(img)
        torch.manual_seed(0)
        high = ForensicAugment(noise_p=1.0, noise_std=0.10)(img)
        low_err = (low.float() - img.float()).std().item()
        high_err = (high.float() - img.float()).std().item()
        assert 0.0 < low_err < high_err


class TestJpeg:
    def test_p_one_changes_pixels_and_stays_uint8_chw(self):
        img = _natural_image(seed=6)
        torch.manual_seed(0)
        out = ForensicAugment(jpeg_p=1.0, jpeg_quality_min=30, jpeg_quality_max=30)(img)
        assert out.dtype == torch.uint8
        assert out.shape == img.shape
        assert not torch.equal(out, img)

    def test_lower_quality_deviates_more(self):
        img = _natural_image(seed=7)
        torch.manual_seed(0)
        hi = ForensicAugment(jpeg_p=1.0, jpeg_quality_min=95, jpeg_quality_max=95)(img)
        torch.manual_seed(0)
        lo = ForensicAugment(jpeg_p=1.0, jpeg_quality_min=30, jpeg_quality_max=30)(img)
        hi_err = (hi.float() - img.float()).abs().mean().item()
        lo_err = (lo.float() - img.float()).abs().mean().item()
        assert hi_err < lo_err

    def test_swapped_quality_bounds_are_tolerated(self):
        img = _natural_image(seed=8)
        torch.manual_seed(0)
        out = ForensicAugment(jpeg_p=1.0, jpeg_quality_min=90, jpeg_quality_max=40)(img)
        assert out.shape == img.shape and out.dtype == torch.uint8


class TestFullChain:
    def test_all_three_preserve_shape_dtype(self):
        img = _natural_image(seed=9)
        torch.manual_seed(0)
        out = _ALL_ON(img)
        assert out.dtype == torch.uint8
        assert out.shape == img.shape
        assert not torch.equal(out, img)

    def test_draws_come_from_torch_rng_so_workers_diverge(self):
        """DataLoader seeds each worker's torch RNG; identical seeds must give
        identical draws, different seeds different ones."""
        img = _natural_image(seed=10)
        torch.manual_seed(11)
        a = _ALL_ON(img)
        torch.manual_seed(11)
        b = _ALL_ON(img)
        torch.manual_seed(12)
        c = _ALL_ON(img)
        assert torch.equal(a, b)
        assert not torch.equal(a, c)


# ---------------------------------------------------------------------------
# GroupDataset integration
# ---------------------------------------------------------------------------


def _build_group(root, *, n_train: int = 1, n_val: int = 1, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for split, n in (("train", n_train), ("val", n_val)):
        d = root / "a" / split
        d.mkdir(parents=True, exist_ok=True)
        for j in range(n):
            arr = _natural_image(64, 64, seed=seed * 100 + j).permute(1, 2, 0).numpy()
            arr = np.clip(arr.astype(np.int16) + rng.integers(-4, 5, arr.shape), 0, 255)
            Image.fromarray(arr.astype(np.uint8)).save(d / f"a_{j}.png")


class TestGroupDatasetIntegration:
    def test_train_split_applies_the_transform(self, tmp_path):
        root = tmp_path / "g"
        _build_group(root)
        random.seed(0)
        clean = GroupDataset(root, ["a"], split="train", group_name="g")
        random.seed(0)
        dirty = GroupDataset(
            root, ["a"], split="train", group_name="g", forensic=_ALL_ON,
        )
        torch.manual_seed(0)
        t_clean, _, _ = clean[0]
        torch.manual_seed(0)
        t_dirty, _, _ = dirty[0]
        assert t_dirty.dtype == torch.uint8
        assert t_dirty.shape == t_clean.shape
        assert not torch.equal(t_dirty, t_clean)

    def test_val_split_never_applies_the_transform(self, tmp_path):
        root = tmp_path / "g"
        _build_group(root)
        clean = GroupDataset(root, ["a"], split="val", group_name="g")
        dirty = GroupDataset(root, ["a"], split="val", group_name="g", forensic=_ALL_ON)
        torch.manual_seed(0)
        t_clean, _, _ = clean[0]
        torch.manual_seed(0)
        t_dirty, _, _ = dirty[0]
        assert torch.equal(t_dirty, t_clean)

    def test_inactive_transform_is_a_no_op(self, tmp_path):
        root = tmp_path / "g"
        _build_group(root)
        random.seed(0)
        clean = GroupDataset(root, ["a"], split="train", group_name="g")
        random.seed(0)
        off = GroupDataset(
            root, ["a"], split="train", group_name="g", forensic=ForensicAugment(),
        )
        torch.manual_seed(0)
        t_clean, _, _ = clean[0]
        torch.manual_seed(0)
        t_off, _, _ = off[0]
        assert torch.equal(t_off, t_clean)


class TestSmartCacheStaysClean:
    def test_cache_stores_unaugmented_pixels(self, tmp_path):
        """The SmartCache is populated by ``build_image_tensor`` via
        ``SmartCache.prepare`` — never through ``__getitem__`` — so an active
        forensic transform must not bake augmented pixels into the cache."""
        from bittrainer.cache_builders import build_image_tensor
        from bittrainer.smart_cache import SmartCache

        root = tmp_path / "g"
        _build_group(root)
        random.seed(0)
        ds = GroupDataset(root, ["a"], split="train", group_name="g", forensic=_ALL_ON)
        cache = SmartCache(tmp_path / "cache", tqdm_enabled=False)
        cache.prepare(list(ds.samples), build_image_tensor, num_workers=1)

        sample = ds.samples[0]
        cached = cache.get(sample["path"])
        assert cached is not None
        cached_tensor, _meta = cached
        expected = torch.from_numpy(build_image_tensor(sample))
        assert torch.equal(cached_tensor, expected)

        # And with the cache attached, the dataset still augments on read.
        ds.set_cache(cache)
        torch.manual_seed(0)
        served, _, _ = ds[0]
        assert served.shape == cached_tensor.shape
        assert not torch.equal(served, cached_tensor)


class TestTrainLoopDoesNotRepeatForensicOnGpu:
    def test_train_one_epoch_passes_no_forensic_kwargs(self, monkeypatch):
        """The knobs now reach the workers via GroupDataset; the GPU step loop
        must not run the same chain a second time on the main thread."""
        import torch.nn as nn

        import bittrainer.gpu_augment as ga
        from bittrainer.group_trainer import GroupTrainConfig, _train_one_epoch

        captured: list[dict] = []

        def fake_apply_train_augment(batch, dtype=torch.float32, **kwargs):
            captured.append(kwargs)
            return batch.float().div(255.0)

        monkeypatch.setattr(ga, "apply_train_augment", fake_apply_train_augment)

        config = GroupTrainConfig(
            group_folder="unused",
            num_classes=2,
            class_names=["a", "b"],
            device="cpu",
            channels_last=False,
            record_sample_stats=False,
            aug_blur_p=1.0,
            aug_noise_p=1.0,
            aug_jpeg_p=1.0,
        )
        model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        batches = [
            (
                torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8),
                torch.tensor([0, 1]),
            )
        ]
        _train_one_epoch(
            model,
            batches,
            optimizer,
            config,
            torch.device("cpu"),
            torch.float32,
        )

        assert captured, "apply_train_augment was never called"
        for kwargs in captured:
            for forbidden in (
                "noise_p", "noise_std", "blur_p", "blur_sigma_max",
                "jpeg_p", "jpeg_quality",
            ):
                assert forbidden not in kwargs, (
                    f"_train_one_epoch still passes {forbidden} to apply_train_augment"
                )
