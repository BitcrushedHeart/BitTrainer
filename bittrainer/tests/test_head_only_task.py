"""HeadOnlyTask on the GenericTrainer skeleton (Bitcrush ISSUE-0542 Step 4).

Pins the two behaviours the Engine worker depends on across the migration of
``run_head_only_training`` onto ``GenericTrainer``:

(a) a set ``pause_event`` (and, identically, ``stop_now_event``) makes head-only
    behave like a cancel — there is no backup/resume for head-only, so it returns
    the ``{"cancelled": True, "mode": "head_only"}`` partial result, NOT the
    generic ``{"paused": True, ...}`` envelope; and
(b) a normal run returns the mode / backbone-hash / cache-stats keys the worker
    reads. CPU-only, caching off (single-process).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from bittrainer.group_trainer import GroupTrainConfig
from bittrainer.head_only_trainer import run_head_only_training

_CLASSES = ["a", "b", "c"]


class _FlagEvent:
    """Duck-typed stop/pause event (``.is_set()``)."""

    def __init__(self, *, preset: bool = False) -> None:
        self._set = preset

    def set(self) -> None:
        self._set = True

    def is_set(self) -> bool:
        return self._set


def _build_group(root):
    for ci, c in enumerate(_CLASSES):
        for split, n in (("train", 8), ("val", 4)):
            d = root / c / split
            d.mkdir(parents=True)
            for j in range(n):
                arr = np.random.default_rng(ci * 1000 + j + hash(split) % 50).integers(
                    0, 80, (96, 96, 3)
                ).astype(np.uint8)
                arr[..., ci] = np.clip(arr[..., ci] + 150, 0, 255)  # class-separable
                Image.fromarray(arr).save(d / f"i{j}.png")


def _cfg(root, ckpt_dir, embed_dir):
    return GroupTrainConfig(
        group_folder=str(root), num_classes=3, class_names=_CLASSES,
        device="cpu", dtype="float32", use_cache=False,
        checkpoint_dir=str(ckpt_dir), embedding_cache_dir=str(embed_dir),
        probe_head="linear", head_max_epochs=20, head_patience=6,
    )


def test_pause_behaves_like_stop_now_and_never_paused_envelope(tmp_path):
    """A pre-set pause_event cancels the run (like stop_now) and returns the
    head-only cancelled partial — never a generic ``paused`` envelope."""
    root = tmp_path / "grp"
    _build_group(root)
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()

    paused = run_head_only_training(
        _cfg(root, ckpt_dir, tmp_path / "embed_p"),
        pause_event=_FlagEvent(preset=True),
    )
    stopped = run_head_only_training(
        _cfg(root, ckpt_dir, tmp_path / "embed_s"),
        stop_now_event=_FlagEvent(preset=True),
    )

    assert paused == {"cancelled": True, "mode": "head_only"}
    # pause is observably identical to stop_now for head-only
    assert paused == stopped
    # never the generic pause/resume envelope
    assert "paused" not in paused and "backup_path" not in paused


def test_result_has_mode_and_backbone_hash_keys(tmp_path):
    """A completed run returns the keys the Engine worker reads: the mode marker,
    the backbone-era hash and the embedding-cache stats, plus the deployed
    checkpoint path."""
    root = tmp_path / "grp"
    _build_group(root)
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()

    res = run_head_only_training(_cfg(root, ckpt_dir, tmp_path / "embed"))

    assert res["mode"] == "head_only"
    assert isinstance(res["backbone_hash"], str) and res["backbone_hash"]
    assert isinstance(res["embedding_cache_stats"], dict)
    assert res["probe_head"] == "linear"
    assert "thin_class_warnings" in res
    assert res["checkpoint_path"].endswith("best.pt")
    assert len(res["per_class_f1"]) == 3
