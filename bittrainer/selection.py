"""Checkpoint-selection scoring helpers (Bitcrush ISSUE-0542).

Extracted verbatim from ``bittrainer.group_trainer`` so the epoch-selection
metric zoo (macro-F1 / weighted / balanced, the ``__none__`` guard term, and
the ordinal QWK+macro-F1 composite) lives in one place. ``group_trainer``
re-imports every name from here, so the old import paths keep working and the
objects stay identical.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bittrainer.group_trainer import GroupTrainConfig

logger = logging.getLogger(__name__)

_NONE_CLASS_NAME = "__none__"
_NONE_F1_WEIGHT = 0.10
# weight on macro-F1 in the composite (0 = pure ordinal metric)
_SELECTION_SECONDARY_WEIGHT = 0.40


def _resolve_none_index(class_names: list[str]) -> int:
    """Return the position of the ``__none__`` class, or -1 if absent.

    ``__none__`` is a valid output class the model learns to predict, but it
    must be excluded from any code path that assumes class indices are
    positions on an ordinal scale (Gaussian soft-target smoothing, ordinal
    validation metrics, etc.).
    """
    try:
        return class_names.index(_NONE_CLASS_NAME)
    except ValueError:
        return -1


def _primary_validation_metric(config: GroupTrainConfig) -> str:
    if config.ordinal:
        if config.validation_metric == "guarded_qwk":
            return "guarded_qwk"
        return "qwk" if config.validation_metric == "qwk" else "macro_f1"
    return "macro_f1"


def _has_none_class(config: GroupTrainConfig) -> bool:
    return _resolve_none_index(config.class_names) >= 0


def _guarded_metric_enabled(config: GroupTrainConfig) -> bool:
    return config.none_guard and _has_none_class(config) and not config.multi_label


def _selection_base_f1(metrics: dict, config: GroupTrainConfig) -> float:
    """The non-ordinal selection driver F1 under ``config.selection_metric``.

    ``macro_f1`` (default, balanced-training behaviour), ``weighted_f1``
    (support-weighted, so majority classes count for their real prevalence), or
    ``balanced`` (harmonic mean of the two — punishes collapsing either side,
    consistent with the ordinal composite's harmonic style).
    """
    macro = float(metrics.get("macro_f1") or 0.0)
    metric = getattr(config, "selection_metric", "macro_f1") or "macro_f1"
    if metric == "weighted_f1":
        return float(metrics.get("weighted_f1") or 0.0)
    if metric == "balanced":
        weighted = float(metrics.get("weighted_f1") or 0.0)
        if macro <= 0.0 or weighted <= 0.0:
            return 0.5 * (macro + weighted)
        return 2.0 * macro * weighted / (macro + weighted)
    return macro


def _guarded_score(metrics: dict, config: GroupTrainConfig) -> float:
    none_f1 = float(metrics.get("none_f1") or 0.0)
    if config.ordinal and _primary_validation_metric(config) == "guarded_qwk":
        return float(metrics.get("qwk") or 0.0) + _NONE_F1_WEIGHT * none_f1
    return _selection_base_f1(metrics, config) + _NONE_F1_WEIGHT * none_f1


def _ordinal_primary_score(metrics: dict, config: GroupTrainConfig) -> float:
    """The ordinal driver metric (QWK, plus the __none__ guard term when the
    group uses guarded_qwk) BEFORE it is composited with macro-F1."""
    qwk = float(metrics.get("qwk") or 0.0)
    if _primary_validation_metric(config) == "guarded_qwk" and _guarded_metric_enabled(config):
        return qwk + _NONE_F1_WEIGHT * float(metrics.get("none_f1") or 0.0)
    return qwk


def _composite_selection_score(primary: float, macro_f1: float) -> float:
    """Weighted harmonic mean of the ordinal primary metric and macro-F1.

    The harmonic mean (as in F1 itself) is dominated by whichever component is
    weakest, so a marginal QWK gain cannot buy back a large macro-F1 collapse.
    Degenerate inputs (<= 0, where the harmonic mean is undefined) fall back to
    the weighted arithmetic mean so the ordering stays well-defined.
    """
    p = max(0.0, primary)
    s = max(0.0, macro_f1)
    w_secondary = _SELECTION_SECONDARY_WEIGHT
    w_primary = 1.0 - w_secondary
    if p <= 0.0 or s <= 0.0:
        return w_primary * p + w_secondary * s
    return 1.0 / (w_primary / p + w_secondary / s)


def _metric_score(metrics: dict, config: GroupTrainConfig) -> float:
    # Ordinal groups: select on a composite of the ordinal metric (QWK or
    # guarded QWK) and macro-F1 so a marginal QWK gain cannot override an
    # exact-match (macro-F1) collapse. See the _SELECTION_* constants above.
    if config.ordinal and _primary_validation_metric(config) in ("qwk", "guarded_qwk"):
        primary = _ordinal_primary_score(metrics, config)
        macro_f1 = float(metrics.get("macro_f1") or 0.0)
        return _composite_selection_score(primary, macro_f1)
    # Non-ordinal groups: macro-F1 (with the __none__ guard term when present).
    if _guarded_metric_enabled(config) and not config.ordinal:
        return _guarded_score(metrics, config)
    # Non-ordinal, no __none__ guard: the configured selection metric.
    if not config.ordinal:
        return _selection_base_f1(metrics, config)
    # Ordinal-as-macro_f1 or any other fallthrough: the primary metric as-is.
    metric = _primary_validation_metric(config)
    value = metrics.get("qwk" if metric == "qwk" else "macro_f1")
    return float(value) if value is not None else 0.0


def _score_metric_label(config: GroupTrainConfig) -> str:
    metric = _primary_validation_metric(config)
    if metric == "guarded_qwk":
        return "guarded_qwk"
    if _guarded_metric_enabled(config) and not config.ordinal:
        return "guarded_macro_f1"
    return metric
