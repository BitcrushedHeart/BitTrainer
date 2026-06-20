"""Regression: the head probe must train on float32 cached features even when the
model runs in a reduced compute dtype (bf16/fp16).

On GPU the model's head weights are bf16/fp16 while the cached embeddings are
float32; feeding float32 into a bf16/fp16 Linear raises
"mat1 and mat2 must have same dtype". The probe trains the tail in float32 and
restores the model dtype afterwards. CPU/float32 tests never exercised this, so
this pins it on bf16 + fp16 models (the dtype-mismatch check fires on CPU too).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bittrainer.group_trainer import GroupTrainConfig
from bittrainer.head_probe import train_head_probe
from bittrainer.model import create_model


class _FakeEmbed:
    def __init__(self, mapping):
        self._m = mapping

    def get_vector(self, path, _smart_cache):
        return self._m.get(path)


def _separable_dataset():
    rng = np.random.default_rng(0)

    def vec(c):
        v = rng.normal(0, 0.1, 640).astype(np.float32)
        v[c * 20:(c + 1) * 20] += 3.0
        return v

    mapping, train, val = {}, [], []
    for c in range(3):
        for i in range(40):
            p = f"tr_{c}_{i}"; mapping[p] = vec(c)
            train.append({"path": p, "label": c, "bucket": (64, 64)})
        for i in range(12):
            p = f"va_{c}_{i}"; mapping[p] = vec(c)
            val.append({"path": p, "label": c, "bucket": (64, 64)})
    return _FakeEmbed(mapping), train, val


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_probe_trains_on_reduced_dtype_model(dtype):
    embed, train, val = _separable_dataset()
    model = create_model(model_size="nano", pretrained=False, num_classes=3).to(dtype)
    assert model.head.fc.weight.dtype == dtype

    cfg = GroupTrainConfig(
        group_folder="/tmp/g", num_classes=3, class_names=["a", "b", "c"],
        head_max_epochs=40, head_patience=8, head_weight_decay=0.0,
    )
    res = train_head_probe(
        model, embed, None, train, val, cfg,
        device=torch.device("cpu"), none_index=-1,
    )

    assert res["macro_f1"] > 0.95  # float32 probe still learns the separable features
    assert model.head.fc.weight.dtype == dtype  # head restored to the model dtype
