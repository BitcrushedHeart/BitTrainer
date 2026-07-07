"""ΔE-perceptual soft labels (Skin Tone V2, Engine ISSUE-0217 spec §8.1)."""

from __future__ import annotations

import torch

from bittrainer.group_trainer import (
    GroupTrainConfig,
    _build_perceptual_kernel,
    _build_soft_targets,
)

# Four classes + __none__: two perceptually close light tones, one distant
# dark tone. Display order deliberately interleaved to prove softness follows
# ΔE, not index adjacency.
NAMES = ["__none__", "pale skin", "deep brown skin", "fair skin", "tanned skin"]
CENTROIDS = {
    "pale skin": [0.80, 0.03, 0.04],
    "deep brown skin": [0.42, 0.05, 0.05],
    "fair skin": [0.78, 0.032, 0.042],  # ΔE ~0.021 from pale — a near neighbour
    "tanned skin": [0.62, 0.06, 0.07],
}


def _kernel() -> torch.Tensor:
    k = _build_perceptual_kernel(NAMES, CENTROIDS, 0.035, none_index=0)
    assert k is not None
    return k


class TestPerceptualKernel:
    def test_softness_follows_delta_e_not_index(self):
        k = _kernel()
        pale, deep, fair = 1, 2, 3
        # "pale skin" bleeds far more into "fair skin" (ΔE 0.021) than into
        # its INDEX-adjacent "deep brown skin" (ΔE ~0.38).
        assert k[pale, fair] > 10 * k[pale, deep]
        assert k[pale, deep] < 1e-6

    def test_none_stays_hard(self):
        k = _kernel()
        assert k[0, 0] == 1.0
        assert float(k[0, 1:].sum()) == 0.0
        assert float(k[1:, 0].sum()) == 0.0

    def test_rows_normalised(self):
        k = _kernel()
        assert torch.allclose(k.sum(dim=1), torch.ones(len(NAMES)))

    def test_disabled_without_centroids(self):
        assert _build_perceptual_kernel(NAMES, {}, 0.035, none_index=0) is None
        assert (
            _build_perceptual_kernel(NAMES, {"pale skin": [0.8, 0.03, 0.04]}, 0.035, none_index=0)
            is None
        )
        assert _build_perceptual_kernel(NAMES, CENTROIDS, 0.0, none_index=0) is None

    def test_soft_targets_use_kernel_over_label_smoothing(self):
        labels = torch.tensor([1])  # pale skin
        kernel = _kernel()
        soft = _build_soft_targets(
            labels, len(NAMES),
            ordinal=False,
            label_smoothing=0.1,  # would spread uniformly if it applied
            none_index=0,
            perceptual_kernel=kernel,
        )
        # fair skin got real weight; deep brown got essentially none —
        # uniform smoothing would have given them equal shares.
        assert float(soft[0, 3]) > 0.2
        assert float(soft[0, 2]) < 1e-6
        assert torch.allclose(soft.sum(dim=1), torch.ones(1))

    def test_config_accepts_fields(self):
        cfg = GroupTrainConfig(
            group_folder="x",
            num_classes=len(NAMES),
            class_names=NAMES,
            class_similarity_centroids=CENTROIDS,
            perceptual_sigma=0.04,
        )
        assert cfg.perceptual_sigma == 0.04
        assert cfg.class_similarity_centroids["pale skin"][0] == 0.80
