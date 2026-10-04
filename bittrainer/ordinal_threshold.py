"""All-threshold ordinal head (Bitcrush ISSUE-1106).

An ordinal group's classes are steps on a scale (plus, usually, ``__none__``).
A softmax head treats them as unrelated labels; the all-threshold head instead
asks one question per boundary — "is the step above k?" — and reads the step as
the number of thresholds passed. In the ISSUE-1063 research it beat both ridge
and softmax heads on held-out people.

:class:`OrdinalThresholdFC` replaces the classifier's ``head.fc`` (the same way
``SpatialCellFC`` does for spatial groups). Its linear layer has ``K - 1``
outputs for ``K`` classes:

* ``R - 1`` threshold logits ``z_0 .. z_{R-2}`` over the ``R`` ordinal classes
  (every class but ``__none__``, in class-index order), then
* one ``__none__`` logit ``u`` — present exactly when the group has a none class
  (so ``R - 1 + 1 = K - 1``; without one ``R = K`` and there are ``K - 1``
  thresholds).

``forward`` emits ``K`` scores that are EXACT log-probabilities (they
log-sum-exp to 0), via the thermometer construction: rank ``j`` is the pattern
"first ``j`` thresholds on, the rest off",

    therm_j = sum_{k<j} log sigma(z_k) + sum_{k>=j} log sigma(-z_k),

normalised over the ``R`` consistent patterns (log-softmax over ``j``), and the
none class is gated by its own logit: ``log P(none) = log sigma(u)`` and
``log P(rank j) = log sigma(-u) + log_softmax(therm)_j``. Since
``therm_{j+1} - therm_j = z_j``, ``log_softmax(therm)`` equals
``log_softmax([0, z_0, z_0 + z_1, ...])`` — the form computed here, exact and
stable at any logit size. Argmax of the scores is the count rule whenever the
thresholds are monotone. Because the scores are ordinary log-probabilities,
softmax, temperature, priors, calibration, the strictness curve and every
Engine consumer keep their one-probability-per-class contract unchanged.

The loss (:func:`ordinal_threshold_loss`) is BCE of each threshold against the
cumulative soft target built from the group's existing soft targets (so the
ordinal sigma sweep still applies), with ``__none__`` rows training only the
none logit.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

THRESHOLD_HEAD = "threshold"
SOFTMAX_HEAD = "softmax"
ORDINAL_HEADS = (SOFTMAX_HEAD, THRESHOLD_HEAD)
THRESHOLD_STATE_KEY = "head.fc.threshold_fc.weight"
NONE_MASK_STATE_KEY = "head.fc.none_mask"


def _has_none(num_classes: int, none_index: int) -> bool:
    return 0 <= int(none_index) < int(num_classes)


def _rank_indices(num_classes: int, none_index: int) -> list[int]:
    return [i for i in range(int(num_classes)) if i != int(none_index)]


class OrdinalThresholdFC(nn.Module):
    """Threshold classifier ``fc``: ``K - 1`` logits in, ``K`` log-probabilities out."""

    def __init__(
        self, in_features: int, num_classes: int, none_index: int = -1
    ) -> None:
        super().__init__()
        num_classes = int(num_classes)
        if num_classes < 2:
            raise ValueError("an ordinal threshold head needs at least two classes")
        has_none = _has_none(num_classes, none_index)
        self.in_features = int(in_features)
        self.num_classes = num_classes
        self.none_index = int(none_index) if has_none else -1
        self.num_ranks = num_classes - int(has_none)
        self.threshold_fc = nn.Linear(self.in_features, num_classes - 1)
        mask = torch.zeros(num_classes)
        if has_none:
            mask[self.none_index] = 1.0
        # Persistent so a checkpoint names its own none position; float so the
        # weight-averaging paths (SWA, soup) carry it through unharmed.
        self.register_buffer("none_mask", mask)
        self.register_buffer(
            "_rank_index",
            torch.tensor(_rank_indices(num_classes, self.none_index), dtype=torch.long),
            persistent=False,
        )

    @property
    def weight(self) -> torch.Tensor:
        """The trainable weight matrix (head probes read ``.weight.dtype``)."""
        return self.threshold_fc.weight

    def threshold_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Raw ``[B, K-1]`` logits: thresholds first, then the none logit."""
        return self.threshold_fc(x)

    def decode(self, raw: torch.Tensor) -> torch.Tensor:
        """``[B, K-1]`` threshold logits -> ``[B, K]`` exact log-probabilities."""
        z = raw.float()
        thresholds = z[:, : self.num_ranks - 1]
        steps = torch.cat(
            [z.new_zeros(z.shape[0], 1), torch.cumsum(thresholds, dim=1)], dim=1
        )
        rank_lp = torch.log_softmax(steps, dim=1)
        if self.none_index < 0:
            return rank_lp
        u = z[:, self.num_ranks - 1]
        rank_lp = rank_lp + F.logsigmoid(-u).unsqueeze(1)
        scores = z.new_empty(z.shape[0], self.num_classes)
        scores = scores.index_copy(1, self._rank_index.to(z.device), rank_lp)
        none_col = torch.tensor([self.none_index], device=z.device)
        return scores.index_copy(1, none_col, F.logsigmoid(u).unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.threshold_fc(x))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # The none position is part of how THIS head was built (from the run's
        # class list); a warm start from a head whose none sat elsewhere may
        # carry the thresholds over, never the position.
        key = prefix + "none_mask"
        incoming = state_dict.get(key)
        if incoming is not None and incoming.shape == self.none_mask.shape:
            state_dict[key] = self.none_mask.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def has_threshold_head(model: nn.Module) -> bool:
    head = getattr(model, "head", None)
    return isinstance(getattr(head, "fc", None), OrdinalThresholdFC)


def install_ordinal_threshold_head(
    model: nn.Module, num_classes: int, none_index: int = -1
) -> OrdinalThresholdFC:
    """Replace ``model.head.fc`` with an :class:`OrdinalThresholdFC`.

    Composes with both head layouts (plain linear and MLP ``pre_logits``) and
    both backbone families: only the final projection is swapped. Device and
    dtype follow the layer being replaced.
    """
    fc = model.head.fc
    head = OrdinalThresholdFC(fc.in_features, num_classes, none_index)
    head = head.to(device=fc.weight.device, dtype=fc.weight.dtype)
    model.head.fc = head
    return head


def threshold_none_index(state_dict: dict, class_names: list[str] | None = None) -> int:
    """The none position of a saved threshold head: its own ``none_mask``
    buffer, else the class list, else none."""
    mask = state_dict.get(NONE_MASK_STATE_KEY)
    if mask is not None:
        mask = mask.float()
        return int(mask.argmax()) if bool((mask > 0.5).any()) else -1
    if class_names:
        from bittrainer.selection import _resolve_none_index

        return _resolve_none_index(list(class_names))
    return -1


def threshold_logits_from_scores(scores: torch.Tensor, none_index: int) -> torch.Tensor:
    """Invert :meth:`OrdinalThresholdFC.decode`: ``[B, K]`` scores -> ``[B, K-1]``.

    Thresholds are the differences of consecutive rank log-probabilities
    (``z_j = l_{j+1} - l_j``; the none gate cancels), and the none logit is
    ``log sigma(u) - log sigma(-u)`` where ``log sigma(-u)`` is the
    log-sum-exp of the rank scores. Lets the loss work from the ``[B, K]``
    scores every call site already passes around.
    """
    s = scores.float()
    num_classes = s.shape[1]
    if not _has_none(num_classes, none_index):
        return s[:, 1:] - s[:, :-1]
    rank_idx = torch.tensor(_rank_indices(num_classes, none_index), device=s.device)
    ranks = s.index_select(1, rank_idx)
    thresholds = ranks[:, 1:] - ranks[:, :-1]
    u = s[:, int(none_index)] - torch.logsumexp(ranks, dim=1)
    return torch.cat([thresholds, u.unsqueeze(1)], dim=1)


def cumulative_targets(
    soft_targets: torch.Tensor, none_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Threshold targets from ``[B, K]`` class soft targets.

    Returns ``(targets, rank_mass)``. ``targets`` is ``[B, K-1]`` in the raw
    logit layout: target ``k`` is ``P(step > k | not none)`` — the tail mass
    above rank ``k`` over the mass on ranks — and, when the group has a none
    class, the last column is ``P(none)``. ``rank_mass`` (``[B]``) is the target
    mass on the ranks, which weighs each row's threshold terms.
    """
    soft = soft_targets.float()
    num_classes = soft.shape[1]
    has_none = _has_none(num_classes, none_index)
    if has_none:
        rank_idx = torch.tensor(
            _rank_indices(num_classes, none_index), device=soft.device
        )
        ranks = soft.index_select(1, rank_idx)
    else:
        ranks = soft
    mass = ranks.sum(dim=1)
    tail = torch.flip(torch.cumsum(torch.flip(ranks, dims=[1]), dim=1), dims=[1])[:, 1:]
    cumulative = tail / mass.clamp(min=1e-8).unsqueeze(1)
    cumulative = cumulative.clamp(0.0, 1.0)
    if has_none:
        cumulative = torch.cat(
            [cumulative, soft[:, int(none_index)].unsqueeze(1)], dim=1
        )
    return cumulative, mass


def ordinal_threshold_loss(
    threshold_logits: torch.Tensor,
    soft_targets: torch.Tensor,
    *,
    none_index: int,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean BCE of the thresholds against their cumulative soft targets.

    Per row: the thresholds' mean BCE weighted by the row's rank mass (a
    ``__none__`` row has none, so it trains only the none logit), plus the none
    logit's BCE against ``P(none)`` when the group has a none class. Rows are
    averaged, or — with ``class_weights`` — weighted by the expected class
    weight under their soft target (the same convention as the soft-CE loss).
    """
    z = threshold_logits.float()
    soft = soft_targets.float()
    num_classes = soft.shape[1]
    has_none = _has_none(num_classes, none_index)
    targets, mass = cumulative_targets(soft, none_index)
    num_thresholds = num_classes - 1 - int(has_none)
    if num_thresholds > 0:
        bce = F.binary_cross_entropy_with_logits(
            z[:, :num_thresholds], targets[:, :num_thresholds], reduction="none"
        )
        row = mass * bce.mean(dim=1)
    else:
        row = z.new_zeros(z.shape[0])
    if has_none:
        row = row + F.binary_cross_entropy_with_logits(
            z[:, num_thresholds], targets[:, num_thresholds], reduction="none"
        )
    if class_weights is not None:
        w = (soft * class_weights.float().unsqueeze(0)).sum(dim=1)
        return (row * w).sum() / w.sum().clamp(min=1e-8)
    return row.mean()


def uses_threshold_head(config) -> bool:
    """True when a group config trains the all-threshold head."""
    return (
        bool(getattr(config, "ordinal", False))
        and getattr(config, "ordinal_head", SOFTMAX_HEAD) == THRESHOLD_HEAD
        and not bool(getattr(config, "multi_label", False))
    )
