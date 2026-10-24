"""Score val passes on cached pooled features (Bitcrush ISSUE-1110).

A frozen-trunk (head-only) model's logits are ``head_tail_logits(model,
pooled_features(model, x))`` — exactly ``model(x)``. The embedding cache already
holds ``pooled_features`` for every val image, computed from the same dataset
tensors through the same ``apply_val_transform`` and autocast dtype, so a val
pass over the cached vectors gives the full pass's logits without running the
trunk (for SigLIP so400m at 384 px that is the whole cost of the pass).

Only the trunk is skipped. Everything else is the full pass's own code path:
the batches come from the val loader's OWN batch sampler (same bucket batches,
same python ``random`` draws, plus the one torch draw ``DataLoader`` makes per
iteration), the head runs under the same autocast, and the loss and metric
code is :mod:`bittrainer.generic.evaluation`'s, shared rather than copied.

When the cached vectors are NOT what the full pass would see — another trunk
(an incumbent trained with a different backbone), a val dataset that
transforms its inputs (a forced Skin Tone view, a torchvision transform), or a
val image without a vector — callers fall back to the full pass.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.nn as nn

from bittrainer.generic.evaluation import _collect_logit_batches, _score_batches
from bittrainer.model import backbone_state_hash, head_tail_logits

logger = logging.getLogger(__name__)

ProgressFn = Callable[[int, int], None]

_HEAD_PREFIX = "head."


def _fit_head_type(head: nn.Module, head_state: dict) -> None:
    """Swap ``head.fc`` for the saved head's type, in place (Bitcrush ISSUE-1496).

    A head search can put a softmax candidate up against a threshold incumbent
    (or the other way round) on the same trunk; the incumbent's head is still
    servable from the cached vectors once ``fc`` has its shape.
    """
    from bittrainer.ordinal_threshold import (
        NONE_MASK_STATE_KEY,
        OrdinalThresholdFC,
        threshold_none_index,
    )

    fc = head.fc
    saved_threshold = "fc.threshold_fc.weight" in head_state
    if saved_threshold == isinstance(fc, OrdinalThresholdFC):
        return
    ref = fc.weight
    if saved_threshold:
        mask = head_state.get("fc.none_mask")
        none_index = (
            threshold_none_index({NONE_MASK_STATE_KEY: mask}) if mask is not None else -1
        )
        rows = int(head_state["fc.threshold_fc.weight"].shape[0])
        new_fc: nn.Module = OrdinalThresholdFC(int(fc.in_features), rows + 1, none_index)
    else:
        new_fc = nn.Linear(
            int(fc.in_features), int(head_state["fc.weight"].shape[0]), bias="fc.bias" in head_state
        )
    head.fc = new_fc.to(device=ref.device, dtype=ref.dtype)


def val_inputs_are_cache_inputs(val_ds: Any) -> bool:
    """True when the val dataset feeds the model the tensors the cache embedded.

    The embedding cache embeds the SmartCache / ``build_image_tensor`` tensor
    of each sample. The val dataset returns that same tensor unless it applies
    a torchvision ``transform``, a forensic chain, or a Skin Tone view (a bank
    at probability 0 without ``force`` is the identity and draws no RNG).
    """
    if getattr(val_ds, "transform", None) is not None:
        return False
    if getattr(val_ds, "_forensic", None) is not None:
        return False
    if getattr(val_ds, "skin_tone_views", None) is not None and (
        bool(getattr(val_ds, "skin_tone_force_view", False))
        or float(getattr(val_ds, "skin_tone_view_prob", 0.0) or 0.0) > 0.0
    ):
        return False
    return True


def _loader_total(loader: Any) -> int | None:
    try:
        return len(loader)
    except TypeError:
        return None


def _cached_batches(
    features: torch.Tensor,
    val_loader: Any,
    device: torch.device,
    *,
    multi_label: bool,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """The val loader's batches, as cached vectors instead of images.

    Mirrors ``DataLoader.__iter__`` with ``num_workers=0``: one torch draw for
    the iterator's base seed (keeps every later torch RNG consumer where the
    full pass would leave it), then the batch sampler's own index lists, with
    labels collated like ``_collate_bucket_batch`` / ``_collate_multilabel_batch``.
    """
    torch.empty((), dtype=torch.int64).random_(generator=val_loader.generator)
    samples = val_loader.dataset.samples
    for indices in val_loader.batch_sampler:
        idx = list(indices)
        x = features[idx].to(device, non_blocking=True)
        raw = [samples[i]["label"] for i in idx]
        if multi_label:
            labels = torch.stack([torch.as_tensor(lbl) for lbl in raw])
        else:
            labels = torch.tensor(raw, dtype=torch.long)
        yield x, labels


def _head_forward(head_model: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    return lambda pooled: head_tail_logits(head_model, pooled)


@torch.no_grad()
def evaluate_cached(
    head_model: nn.Module,
    features: torch.Tensor,
    val_loader: Any,
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    multi_label: bool = False,
    ordinal: bool = False,
    none_index: int = -1,
    thresholds: np.ndarray | None = None,
    progress: ProgressFn | None = None,
) -> dict:
    """:func:`bittrainer.generic.evaluation._evaluate` on cached val vectors.

    ``features[i]`` is the pooled vector of ``val_loader.dataset.samples[i]``.
    """
    head_model.eval()
    return _score_batches(
        _cached_batches(features, val_loader, device, multi_label=multi_label),
        _head_forward(head_model),
        num_classes, device, dtype,
        multi_label=multi_label, ordinal=ordinal, none_index=none_index,
        thresholds=thresholds, progress=progress, total=_loader_total(val_loader),
    )


@torch.no_grad()
def collect_cached_logits(
    head_model: nn.Module,
    features: torch.Tensor,
    val_loader: Any,
    device: torch.device,
    dtype: torch.dtype,
    *,
    progress: ProgressFn | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`bittrainer.generic.evaluation._collect_val_logits` on cached vectors."""
    head_model.eval()
    return _collect_logit_batches(
        _cached_batches(features, val_loader, device, multi_label=False),
        _head_forward(head_model),
        device, dtype, progress=progress, total=_loader_total(val_loader),
    )


class _HeadOnly(nn.Module):
    """Just a ``head`` — what :func:`head_tail_logits` reads."""

    def __init__(self, head: nn.Module) -> None:
        super().__init__()
        self.head = head


class CachedValScorer:
    """Scores checkpoints whose trunk produced the cached val vectors.

    ``model`` is the run's model (its trunk built ``features``); ``features``
    is ``[len(val_ds.samples), D]`` float32 aligned with the val samples.
    """

    def __init__(
        self,
        model: nn.Module,
        features: torch.Tensor,
        backbone_hash: str,
        *,
        multi_label: bool = False,
    ) -> None:
        self.model = model
        self.features = features
        self.backbone_hash = backbone_hash
        self.multi_label = bool(multi_label)

    def model_for_state(self, state: Any) -> nn.Module | None:
        """A head-only module carrying *state*'s head, or ``None`` when the
        checkpoint's trunk is not the one that built the cache (or its head
        does not fit this architecture) — the caller then runs the full pass."""
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            return None
        try:
            if backbone_state_hash(state) != self.backbone_hash:
                return None
            head_state = {
                k[len(_HEAD_PREFIX):]: v for k, v in state.items() if k.startswith(_HEAD_PREFIX)
            }
            head = copy.deepcopy(self.model.head)
            _fit_head_type(head, head_state)
            head.load_state_dict(head_state, strict=True)
        except (RuntimeError, KeyError, TypeError, ValueError, AttributeError) as exc:
            logger.info("CachedValScorer: checkpoint head not servable (%s)", exc)
            return None
        return _HeadOnly(head).eval()

    def model_for_checkpoint(self, path: str) -> nn.Module | None:
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, EOFError, ValueError) as exc:
            logger.info("CachedValScorer: cannot read %s (%s)", path, exc)
            return None
        return self.model_for_state(state)

    def evaluate(
        self,
        head_model: nn.Module,
        val_loader: Any,
        num_classes: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        ordinal: bool = False,
        none_index: int = -1,
        thresholds: np.ndarray | None = None,
        progress: ProgressFn | None = None,
    ) -> dict:
        head_model.to(device)
        return evaluate_cached(
            head_model, self.features, val_loader, num_classes, device, dtype,
            multi_label=self.multi_label, ordinal=ordinal, none_index=none_index,
            thresholds=thresholds, progress=progress,
        )

    def collect_logits(
        self,
        head_model: nn.Module,
        val_loader: Any,
        device: torch.device,
        dtype: torch.dtype,
        *,
        progress: ProgressFn | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_model.to(device)
        return collect_cached_logits(
            head_model, self.features, val_loader, device, dtype, progress=progress,
        )
