"""``run_pairwise_reward_training`` end to end on a toy CPU fixture
(Bitcrush Optimise P8, ISSUE-1133).

Tiny SigLIP image trunk and text tower through the test seams, synthetic images
where the chosen side is brighter than the rejected side, a handful of caption
groups. Pins: the run completes and writes a checkpoint carrying the fields
Optimise publishes (``text_model``, ``reward_head_arch``, ``embed_preproc_sig``,
``val_pref_acc``, ``norm_mapping``); validation is a GROUP-held-out set, never
train agreement; the split is persisted; progress and stop events behave; the
VAE round-trip preparation copies both sides and records what it did; a saved
checkpoint scores new (image, caption) pairs through ``reward_inference``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import bittrainer.reward_inference as ri
from bittrainer.pair_dataset import (
    PairRecord,
    load_pair_manifest,
    vae_roundtrip_prepare,
    write_pair_manifest,
)
from bittrainer.reward_head import apply_norm_mapping
from bittrainer.reward_trainer import RewardTrainConfig, run_pairwise_reward_training
from bittrainer.tests.siglip_fixtures import TINY_DIM, install_tiny_trunk
from bittrainer.tests.text_fixtures import install_tiny_text_model


class _FlagEvent:
    def __init__(self, *, preset: bool = False) -> None:
        self._set = preset

    def set(self) -> None:
        self._set = True

    def is_set(self) -> bool:
        return self._set


def _image(path: Path, seed: int, *, bright: bool) -> Path:
    rng = np.random.default_rng(seed)
    base = 170 if bright else 60
    arr = rng.integers(base, base + 40, (48, 48, 3)).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    return path


def _manifest(root: Path, *, groups: int = 8, per_group: int = 2) -> Path:
    records = []
    n = 0
    for g in range(groups):
        for k in range(per_group):
            chosen = _image(root / "img" / f"g{g}_c{k}.png", n, bright=True)
            rejected = _image(root / "img" / f"g{g}_r{k}.png", 1000 + n, bright=False)
            n += 1
            records.append(
                PairRecord(
                    chosen=str(chosen),
                    rejected=str(rejected),
                    caption=f"a mug beside notebook number {g}",
                    group_id=f"group-{g}",
                    pair_type="normal" if g % 2 else "minor_edits",
                    identities=(f"blake3:c{g}{k}", f"blake3:r{g}{k}"),
                )
            )
    path = root / "pairs.json"
    write_pair_manifest(path, records, extra={"optimise": {"schema": 1}})
    return path


@pytest.fixture
def towers(monkeypatch):
    install_tiny_trunk(monkeypatch, embed_dim=TINY_DIM)
    install_tiny_text_model(monkeypatch, embed_dim=TINY_DIM)


def _cfg(root: Path, manifest: Path, **overrides) -> RewardTrainConfig:
    params = dict(
        manifest_path=str(manifest),
        output_dir=str(root / "out"),
        image_model_size="siglip2_base",
        image_size=32,
        hidden_dim=16,
        max_epochs=12,
        patience=12,
        batch_size=8,
        learning_rate=5e-3,
        val_fraction=0.25,
        seed=1,
        device="cpu",
        dtype="float32",
        embedding_cache_dir=str(root / "embed"),
    )
    params.update(overrides)
    return RewardTrainConfig(**params)


def test_toy_run_completes_and_publishes_the_checkpoint_fields(tmp_path, towers):
    manifest = _manifest(tmp_path)
    frames: list[dict] = []
    result = run_pairwise_reward_training(_cfg(tmp_path, manifest), progress_callback=frames.append)

    assert result["mode"] == "pairwise_reward"
    assert result["epochs_completed"] >= 1
    assert result["pair_count"] == 16
    assert result["type_counts"] == {"minor_edits": 8, "normal": 8}
    assert result["held_out"] is True
    assert result["split"]["val_pairs"] > 0
    assert result["split"]["train_pairs"] + result["split"]["val_pairs"] == 16
    assert 0.0 <= result["val_pref_acc"] <= 1.0
    assert isinstance(result["val_margin"], float)
    # Train agreement is reported under its own name, never as held-out.
    assert "train_pref_acc" in result
    assert result["text_model"] == "google/siglip2-base-patch16-naflex"
    assert result["image_model"] == "siglip2_base"
    assert result["norm_mapping"]["type"] == "sigmoid"
    assert set(result["per_type"]) <= {"minor_edits", "normal"}

    ckpt_path = Path(result["checkpoint_path"])
    assert ckpt_path.is_file() and ckpt_path.suffix == ".pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in (
        "trainer",
        "image_model",
        "text_model",
        "reward_head_arch",
        "embed_preproc_sig",
        "val_pref_acc",
        "val_margin",
        "held_out",
        "norm_mapping",
        "per_type",
        "type_counts",
        "split_path",
        "head_state_dict",
        "vae_prep",
    ):
        assert key in ckpt, key
    assert ckpt["trainer"] == "pairwise_reward"
    assert ckpt["reward_head_arch"]["embed_dim"] == TINY_DIM
    assert ckpt["embed_preproc_sig"].startswith("siglip2_base:")
    assert ckpt["norm_mapping"] == result["norm_mapping"]
    assert ckpt["vae_prep"] is None
    assert ckpt["val_pref_acc"] == result["val_pref_acc"]

    # The split is persisted beside the checkpoint and names whole groups.
    split = json.loads(Path(result["split_path"]).read_text("utf-8"))
    assert not set(split["train_groups"]) & set(split["val_groups"])
    assert len(split["val_groups"]) == 2  # 25% of 8 groups
    # Progress frames carry a stage and the epoch summaries.
    assert any(frame.get("type") == "epoch_complete" for frame in frames)
    assert all("stage" in frame or frame.get("type") == "epoch_complete" for frame in frames)
    assert any(frame.get("stage") == "embedding_build" for frame in frames)


def test_toy_run_learns_the_brightness_preference(tmp_path, towers):
    manifest = _manifest(tmp_path, groups=12)
    result = run_pairwise_reward_training(
        _cfg(tmp_path, manifest, max_epochs=40, patience=40, learning_rate=1e-2)
    )
    # Random SigLIP trunk + bright/dark images: a brightness difference is
    # linearly visible in pooled features, so the held-out set is separable.
    assert result["val_pref_acc"] >= 0.75
    assert result["train_pref_acc"] >= 0.75


def test_single_group_manifest_has_no_held_out_metric(tmp_path, towers):
    records = load_pair_manifest(_manifest(tmp_path, groups=4))
    one = [PairRecord(**{**r.__dict__, "group_id": "only"}) for r in records]
    manifest = tmp_path / "one.json"
    write_pair_manifest(manifest, one)
    result = run_pairwise_reward_training(_cfg(tmp_path, manifest, max_epochs=2))
    assert result["held_out"] is False
    assert result["val_pref_acc"] is None
    assert result["split"]["val_pairs"] == 0
    assert "train_pref_acc" in result
    ckpt = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=False)
    assert ckpt["held_out"] is False and ckpt["val_pref_acc"] is None


def test_existing_split_is_reused_and_stop_event_is_honoured(tmp_path, towers):
    manifest = _manifest(tmp_path)
    first = run_pairwise_reward_training(_cfg(tmp_path, manifest, max_epochs=1))
    split_path = first["split_path"]
    before = Path(split_path).read_text("utf-8")
    stop = _FlagEvent(preset=True)
    second = run_pairwise_reward_training(
        _cfg(tmp_path, manifest, output_dir=str(tmp_path / "out2"), split_path=split_path),
        stop_event=stop,
    )
    assert second["split_path"] == split_path
    assert Path(split_path).read_text("utf-8") == before
    assert second["stopped"] is True
    assert second["epochs_completed"] == 0
    assert second["checkpoint_path"] is None
    # The embedding cache from the first run is reused by the second.
    assert second["embedding_cache_stats"]["images_reused"] == 32
    assert first["embedding_cache_stats"]["images_built"] == 32


def test_vae_roundtrip_prepare_processes_both_sides_and_records_it(tmp_path, towers):
    manifest = _manifest(tmp_path, groups=2)
    records = load_pair_manifest(manifest)
    seen: list[str] = []

    def roundtrip(image: Image.Image) -> Image.Image:
        seen.append(image.size)
        return Image.eval(image, lambda v: 255 - v)

    prepared, meta = vae_roundtrip_prepare(
        records,
        roundtrip,
        tmp_path / "vae",
        vae="sdxl-vae-fp16-fix",
        version="1.0",
        settings={"tile": False},
    )
    assert len(prepared) == len(records)
    assert len(seen) == 2 * len(records)
    for before, after in zip(records, prepared, strict=True):
        assert after.caption == before.caption and after.group_id == before.group_id
        assert after.identities == before.identities
        assert Path(after.chosen).is_file() and Path(after.rejected).is_file()
        assert Path(after.chosen).parent == tmp_path / "vae" / "chosen"
        assert Path(before.chosen).is_file()  # originals untouched
        original = np.asarray(Image.open(before.chosen))
        processed = np.asarray(Image.open(after.chosen))
        assert np.array_equal(processed, 255 - original)
    assert meta == {
        "vae": "sdxl-vae-fp16-fix",
        "version": "1.0",
        "settings": {"tile": False},
        "pairs": len(records),
        "output_dir": str(tmp_path / "vae"),
    }
    prepared_manifest = tmp_path / "vae" / "pairs.json"
    write_pair_manifest(prepared_manifest, prepared, extra={"vae_prep": meta})
    result = run_pairwise_reward_training(
        _cfg(tmp_path, prepared_manifest, max_epochs=1, vae_prep=meta)
    )
    ckpt = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=False)
    assert ckpt["vae_prep"] == meta


def test_saved_checkpoint_scores_new_pairs_under_its_fixed_mapping(tmp_path, towers):
    manifest = _manifest(tmp_path, groups=4)
    result = run_pairwise_reward_training(_cfg(tmp_path, manifest, max_epochs=3))
    model = ri.load_reward_model(result["checkpoint_path"], device="cpu")
    assert model.text_model == result["text_model"]
    assert model.image_model == "siglip2_base"
    assert model.norm_mapping == result["norm_mapping"]
    bright = _image(tmp_path / "new_bright.png", 5, bright=True)
    dark = _image(tmp_path / "new_dark.png", 6, bright=False)
    rewards = model.score([str(bright), str(dark)], ["a mug beside notebook number 1"] * 2)
    assert len(rewards) == 2 and all(isinstance(r, float) for r in rewards)
    scores = [apply_norm_mapping(model.norm_mapping, r) for r in rewards]
    assert all(0 <= s <= 100 for s in scores)
    assert model.embed_preproc_sig == result["embed_preproc_sig"]
