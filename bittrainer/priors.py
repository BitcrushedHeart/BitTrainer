"""Inference-time prior correction (Bitcrush ISSUE-0542, origin ISSUE-0490 A).

Extracted verbatim from ``bittrainer.group_trainer``: build the natural /
effective-train log-prior vectors, persist them into the checkpoint, and apply
the ``tau * (log natural - log effective)`` logit delta at finalisation.
``group_trainer`` re-imports every name from here, keeping the objects identical.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from bittrainer.group_dataset import GroupDataset
    from bittrainer.group_trainer import GroupTrainConfig

logger = logging.getLogger(__name__)


def _compute_prior_vectors_from_counts(
    natural_counts: dict[int, int],
    effective_counts: dict[int, int],
    config: GroupTrainConfig,
) -> tuple[dict[str, float], dict[str, float]] | None:
    """Natural + effective-train log-prior vectors for prior correction.

    ``natural`` = raw (un-oversampled) per-class train counts; ``effective`` =
    per-class counts after oversample expansion (the model's real class
    exposure). Both Laplace-smoothed and returned as ``{str(idx): log_prob}``.
    Returns ``None`` for multi-label groups (no single-label prior to correct).
    """
    if config.multi_label:
        return None
    from bittrainer.group_dataset import compute_class_log_priors

    natural = {int(k): int(v or 0) for k, v in natural_counts.items()}
    effective = {int(k): int(v or 0) for k, v in effective_counts.items()}
    log_natural = compute_class_log_priors(natural, config.num_classes)
    log_effective = compute_class_log_priors(effective, config.num_classes)
    return log_natural, log_effective


def _compute_prior_vectors(
    train_ds: GroupDataset,
    config: GroupTrainConfig,
) -> tuple[dict[str, float], dict[str, float]] | None:
    """Prior vectors derived directly from a train dataset (natural counts +
    post-oversample effective counts). Thin wrapper over
    :func:`_compute_prior_vectors_from_counts`."""
    if config.multi_label:
        return None
    return _compute_prior_vectors_from_counts(
        train_ds.get_class_counts(), train_ds.get_effective_class_counts(), config
    )


def _persist_class_priors(
    checkpoint_path: str | None,
    *,
    log_natural: dict[str, float],
    log_effective: dict[str, float],
    tau: float,
) -> None:
    """Write the prior-correction vectors into the checkpoint sidecar meta,
    mirroring :func:`_persist_ordinal_cut_points`. Old checkpoints lack these
    keys, so Engine decode is a byte-identical no-op for them (ISSUE-0490 A)."""
    if not checkpoint_path:
        return
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict):
            ckpt["class_log_prior_natural"] = {str(k): float(v) for k, v in log_natural.items()}
            ckpt["class_log_prior_train_effective"] = {
                str(k): float(v) for k, v in log_effective.items()
            }
            ckpt["prior_tau"] = float(tau)
            torch.save(ckpt, checkpoint_path)
    except Exception:
        logger.warning("Failed to persist class priors to checkpoint", exc_info=True)


def _prior_logit_delta(
    log_natural: dict[str, float],
    log_effective: dict[str, float],
    num_classes: int,
    tau: float,
) -> "np.ndarray":
    """``tau * (log_natural - log_effective)`` as a ``[num_classes]`` vector.

    Added to raw logits BEFORE calibration so the shipped calibration constants
    are fit on the logits inference will actually see (ISSUE-0490 A)."""
    delta = np.zeros(num_classes, dtype=np.float64)
    for i in range(num_classes):
        delta[i] = tau * (
            float(log_natural.get(str(i), 0.0)) - float(log_effective.get(str(i), 0.0))
        )
    return delta


def _apply_and_persist_priors(
    logits: torch.Tensor,
    natural_counts: dict[int, int],
    effective_counts: dict[int, int],
    config: GroupTrainConfig,
    checkpoint_path: str | None,
) -> torch.Tensor:
    """Compute prior-correction vectors, persist them to the checkpoint, and
    return the prior-adjusted val logits.

    Called at finalisation BEFORE calibration so ``calibration_temperature`` and
    ``none_logit_bias`` are fit on the logits inference will actually see
    (decode order: raw -> prior adjustment -> temperature -> none bias). Returns
    the logits unchanged when there is no single-label prior to correct
    (multi-label) so callers can use the result unconditionally (ISSUE-0490 A)."""
    vectors = _compute_prior_vectors_from_counts(natural_counts, effective_counts, config)
    if vectors is None:
        return logits
    log_natural, log_effective = vectors
    _persist_class_priors(
        checkpoint_path,
        log_natural=log_natural,
        log_effective=log_effective,
        tau=config.prior_tau,
    )
    delta = _prior_logit_delta(log_natural, log_effective, config.num_classes, config.prior_tau)
    delta_t = torch.tensor(delta, dtype=torch.float32, device=logits.device)
    return logits.float() + delta_t
