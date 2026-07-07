"""Skin Tone V2 dual-view (Bitcrush Engine ISSUE-0217, spec §8)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from bittrainer.skin_tone_views import (
    SkinToneViewBank,
    ViewParams,
    apply_view,
    load_view_bank,
    maybe_apply_view,
)

IDENTITY = ViewParams(
    wb_gains=(1.0, 1.0, 1.0),
    exposure_gain=1.0,
    saturation_scale=1.0,
    contrast_scale=1.0,
    magnitude=0.0,
    excluded=False,
)

WARM_FIX = ViewParams(
    wb_gains=(0.8, 1.0, 1.35),
    exposure_gain=1.1,
    saturation_scale=1.0,
    contrast_scale=1.0,
    magnitude=0.3,
    excluded=False,
)


def _skin_tensor() -> torch.Tensor:
    t = torch.zeros(3, 32, 32, dtype=torch.uint8)
    t[0], t[1], t[2] = 200, 160, 140
    return t


class TestApplyView:
    def test_identity_params_leave_tensor_unchanged(self):
        t = _skin_tensor()
        out = apply_view(t, IDENTITY)
        assert torch.equal(out, t)

    def test_wb_and_exposure_change_pixels(self):
        t = _skin_tensor()
        out = apply_view(t, WARM_FIX)
        assert out.dtype == torch.uint8
        assert out.shape == t.shape
        assert not torch.equal(out, t)
        # Blue gain > 1 must raise the blue channel; red gain < 1 lowers red.
        assert out[2].float().mean() > t[2].float().mean()
        assert out[0].float().mean() < t[0].float().mean()

    def test_excluded_params_are_never_applied(self):
        t = _skin_tensor()
        excluded = ViewParams(
            wb_gains=(2.0, 1.0, 0.5),
            exposure_gain=2.5,
            saturation_scale=1.6,
            contrast_scale=1.4,
            magnitude=3.0,
            excluded=True,
        )
        assert torch.equal(apply_view(t, excluded), t)

    def test_saturation_contrast_path_runs(self):
        t = _skin_tensor()
        params = ViewParams(
            wb_gains=(1.0, 1.0, 1.0),
            exposure_gain=1.0,
            saturation_scale=1.3,
            contrast_scale=1.2,
            magnitude=0.4,
            excluded=False,
        )
        out = apply_view(t, params)
        assert out.dtype == torch.uint8
        assert not torch.equal(out, t)


class TestBank:
    def test_lookup_by_path_and_basename(self, tmp_path: Path):
        manifest = tmp_path / "skin_tone_normalisation.json"
        manifest.write_text(
            json.dumps(
                {
                    "reference": {},
                    "images": {
                        "F:/somewhere/img_a.png": {
                            "wb_gains": [0.9, 1.0, 1.1],
                            "exposure_gain": 1.05,
                            "saturation_scale": 1.0,
                            "contrast_scale": 1.0,
                            "magnitude": 0.1,
                            "excluded": False,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        bank = load_view_bank(manifest)
        assert bank is not None and len(bank) == 1
        # Exact path (either separator style).
        assert bank.lookup("F:\\somewhere\\img_a.png") is not None
        # Basename fallback for a re-rooted copy of the same file.
        assert bank.lookup("D:/group/Tone 1/train/img_a.png") is not None
        assert bank.lookup("D:/group/other.png") is None

    def test_ambiguous_basenames_disable_fallback(self):
        bank = SkinToneViewBank(
            {
                "a/dup.png": {"wb_gains": [1, 1, 1]},
                "b/dup.png": {"wb_gains": [1, 1, 1]},
            }
        )
        assert bank.lookup("c/dup.png") is None
        assert bank.lookup("a/dup.png") is not None

    def test_missing_or_empty_manifest_disables_feature(self, tmp_path: Path):
        assert load_view_bank(tmp_path / "nope.json") is None
        empty = tmp_path / "empty.json"
        empty.write_text(json.dumps({"images": {}}), encoding="utf-8")
        assert load_view_bank(empty) is None


class TestMaybeApply:
    def test_probability_zero_is_identity(self):
        bank = SkinToneViewBank({"x/img.png": {"wb_gains": [0.8, 1.0, 1.35]}})
        t = _skin_tensor()
        out = maybe_apply_view(t, "x/img.png", bank, probability=0.0, force=False)
        assert torch.equal(out, t)

    def test_force_applies_view(self):
        bank = SkinToneViewBank({"x/img.png": {"wb_gains": [0.8, 1.0, 1.35]}})
        t = _skin_tensor()
        out = maybe_apply_view(t, "x/img.png", bank, probability=0.0, force=True)
        assert not torch.equal(out, t)

    def test_unknown_path_is_identity_even_forced(self):
        bank = SkinToneViewBank({"x/img.png": {"wb_gains": [0.8, 1.0, 1.35]}})
        t = _skin_tensor()
        out = maybe_apply_view(t, "y/unknown.png", bank, probability=1.0, force=True)
        assert torch.equal(out, t)


class TestConfigFields:
    def test_group_train_config_accepts_dual_view_keys(self):
        from bittrainer.group_trainer import GroupTrainConfig

        cfg = GroupTrainConfig(
            group_folder="X:/nowhere",
            num_classes=3,
            class_names=["__none__", "Tone 1", "Tone 2"],
            skin_tone_views_manifest="X:/nowhere/skin_tone_normalisation.json",
            skin_tone_calibration={"version": 1},
            skin_tone_dual_view_prob=0.4,
        )
        assert cfg.skin_tone_views_manifest.endswith(".json")
        assert cfg.skin_tone_calibration == {"version": 1}
        assert cfg.skin_tone_dual_view_prob == 0.4
