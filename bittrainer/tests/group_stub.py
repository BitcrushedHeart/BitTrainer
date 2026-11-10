"""A stand-in trunk and a generated group for the group trainers' tests
(Bitcrush ISSUE-1690).

Even the smallest real ConvNeXt V2 (atto) keeps every core busy for seconds over
a few training steps. What these tests check (call order, the GradScaler, the
GRN layer, the preflight, backups and resume) needs timm's ConvNeXt V2 *class*:
its head, its GRN layers, its stages and their state-dict names. It does not
need its size. :func:`micro_convnext` is that class with four stages of one
block and eight channels, about 4,000 weights; a whole run takes a moment.

Images are drawn here, never read from anywhere: a left half, a right half or
a plain frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from timm.layers.grn import GlobalResponseNorm
from timm.models.convnext import ConvNeXt

import bittrainer.generic.tasks.group_task as group_task
import bittrainer.group_trainer as gt
import bittrainer.model as model_module

CLASSES = ["left", "right", "__none__"]


def micro_convnext(
    *, num_classes: int = len(CLASSES), head_hidden_size: int | None = None, **_unused
) -> ConvNeXt:
    """ConvNeXt V2's block (GRN, no layer scale, 1x1-conv MLP) at toy size.

    Takes ``create_model``'s keywords; the size and the pretrained flag are
    ignored.
    """
    return ConvNeXt(
        num_classes=num_classes,
        depths=(1, 1, 1, 1),
        dims=(8, 8, 8, 8),
        kernel_sizes=3,
        use_grn=True,
        ls_init_value=None,
        conv_mlp=True,
        head_hidden_size=head_hidden_size,
    )


def use_micro_trunk(monkeypatch) -> None:
    """Every model the group trainers build becomes a :func:`micro_convnext`:
    fresh, warm-started, resumed, and a checkpoint loaded back for scoring."""
    for module in (gt, group_task, model_module):
        monkeypatch.setattr(module, "create_model", micro_convnext)


def grn_layers(model: torch.nn.Module) -> list[GlobalResponseNorm]:
    return [m for m in model.modules() if isinstance(m, GlobalResponseNorm)]


def stir_grn(model: torch.nn.Module, seed: int = 0) -> None:
    """GRN starts at zero, where it passes its input straight through; give
    it weights that do something so a changed layer would show."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in grn_layers(model):
            layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator) * 0.5)
            layer.bias.copy_(torch.randn(layer.bias.shape, generator=generator) * 0.5)


def _draw(name: str, size: int, rng: np.random.Generator) -> np.ndarray:
    dark, light = rng.integers(30, 90), rng.integers(150, 220)
    image = np.full((size, size, 3), dark, dtype=np.uint8)
    half = size // 2
    if name == "left":
        image[:, :half] = light
    elif name == "right":
        image[:, half:] = light
    noise = rng.integers(0, 12, (size, size, 3), dtype=np.uint8)
    return image + noise


def make_group(
    root: Path, *, per_class: int = 6, val_per_class: int = 3, size: int = 48, seed: int = 0
) -> Path:
    """``root/<class>/<split>/*.png`` for every class in :data:`CLASSES`."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    for split, count in (("train", per_class), ("val", val_per_class)):
        for name in CLASSES:
            folder = root / name / split
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                Image.fromarray(_draw(name, size, rng)).save(folder / f"{name}_{index}.png")
    return root


def group_config(group_folder: Path, checkpoint_dir: Path, **overrides) -> gt.GroupTrainConfig:
    """A full run on the micro trunk that finishes in seconds on a CPU."""
    base = dict(
        group_folder=str(group_folder),
        num_classes=len(CLASSES),
        class_names=list(CLASSES),
        checkpoint_dir=str(checkpoint_dir),
        max_epochs=1,
        patience=99,
        backbone_variant="atto",
        device="cpu",
        dtype="float32",
        batch_size=4,
        train_resolution=64,
        use_cache=False,
        use_compile=False,
        channels_last=False,
        auto_label_softness=False,
        auto_oversample_none=False,
        use_greedy_soup=False,
        record_sample_stats=False,
        dataloader_workers=0,
        head_max_epochs=2,
        head_patience=1,
        backbone_init={"source": "random_init", "checkpoint_path": None},
    )
    base.update(overrides)
    return gt.GroupTrainConfig(**base)
