"""Dynamic per-class loss-weight controller (ISSUE-0392).

A *soft per-class early-stop* for single-label group training. Each epoch, given
per-class validation F1 (primary signal) and per-class validation loss
(secondary), it detects classes whose smoothed val-F1 has DECLINED from their
per-class peak — the per-class overfitting signature — and shrinks that class's
loss-weight multiplier, throttling the gradient it keeps receiving. Weights are
renormalised to mean 1 so the total loss scale (hence the effective learning
rate) stays constant: this *reallocates* capacity between classes rather than
globally scaling the loss down.

Design guards (see ISSUE-0392 — why the naive "down-weight an overfitting class"
idea needs shaping):
- Triggers on val-F1 DECLINE past a deadband, not on plateau — a class that has
  merely converged is left alone; only one that has started to overfit is
  throttled. (Raw val-loss rises from overconfidence before F1 degrades, so F1
  is the primary trigger; loss only corroborates.)
- A weight FLOOR + COOLDOWN mean a class is throttled, never abandoned — this is
  the guard against trading overfitting for underfitting.
- EMA smoothing + patience stop noisy small-val-set signals from oscillating the
  weights.

The update logic is pure Python (floats/dicts) and unit-testable in isolation;
torch is used only to build the returned weight tensor from the immutable
base-weights vector.
"""

from __future__ import annotations

import torch

_METRICS = ("val_f1", "val_loss", "both")


class DynamicClassWeightController:
    """Between-epoch per-class loss-weight controller.

    ``base_weights`` is the static weight vector to modulate (effective-number
    weights when ``class_balance_mode="reweight"``, otherwise all-ones — both
    have mean 1, so an all-ones base with all-ones multipliers is a numerical
    no-op versus unweighted CE). ``update`` is called once per epoch AFTER
    validation and returns the new ``[num_classes]`` weight tensor to feed the
    next epoch's loss.
    """

    def __init__(
        self,
        num_classes: int,
        base_weights: torch.Tensor,
        *,
        metric: str = "val_f1",
        patience: int = 2,
        ema_decay: float = 0.5,
        decay: float = 0.8,
        floor: float = 0.25,
        ceiling: float = 1.0,
        cooldown: int = 1,
        min_delta: float = 0.005,
    ) -> None:
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}")
        self.num_classes = int(num_classes)
        self._base = base_weights.detach().clone().float()
        self.metric = metric
        self.patience = int(patience)
        self.ema_decay = float(ema_decay)
        self.decay = float(decay)
        self.floor = float(floor)
        self.ceiling = float(ceiling)
        self.cooldown_period = int(cooldown)
        self.min_delta = float(min_delta)

        self.multiplier: list[float] = [1.0] * self.num_classes
        self._f1_ema: list[float | None] = [None] * self.num_classes
        self._loss_ema: list[float | None] = [None] * self.num_classes
        self._peak: list[float | None] = [None] * self.num_classes
        self._loss_at_peak: list[float | None] = [None] * self.num_classes
        self._stale: list[int] = [0] * self.num_classes
        self._cooldown: list[int] = [0] * self.num_classes
        self.adjustments = 0  # cumulative throttles, for observability/tests

    def _ema(self, prev: float | None, x: float) -> float:
        if prev is None:
            return x
        return self.ema_decay * prev + (1.0 - self.ema_decay) * x

    def _primary(self, c: int) -> float | None:
        """Signal where higher == better (F1 directly; negated loss for loss-mode)."""
        if self.metric == "val_loss":
            le = self._loss_ema[c]
            return None if le is None else -le
        return self._f1_ema[c]

    def update(
        self,
        per_class_val_f1: dict[str, float],
        per_class_val_loss: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """Ingest one epoch's per-class metrics; return the next weight tensor.

        Support gate: a class counts as present this epoch iff it has a
        per-class val-loss entry (``compute_multiclass_metrics`` reports F1 as
        0.0 for zero-support classes, which would masquerade as a catastrophic
        decline — so loss-presence, which is omitted for absent classes, is the
        reliable support signal). When no loss dict is supplied we fall back to
        the F1 keys.
        """
        per_class_val_loss = per_class_val_loss or {}
        supported = set(per_class_val_loss) if per_class_val_loss else set(per_class_val_f1)

        for c in range(self.num_classes):
            key = str(c)
            if key not in supported:
                # No val support this epoch — hold all state, but still let any
                # active cooldown tick down so it can't stall indefinitely.
                if self._cooldown[c] > 0:
                    self._cooldown[c] -= 1
                continue

            if key in per_class_val_f1:
                self._f1_ema[c] = self._ema(self._f1_ema[c], float(per_class_val_f1[key]))
            if key in per_class_val_loss:
                self._loss_ema[c] = self._ema(self._loss_ema[c], float(per_class_val_loss[key]))

            improved, declined = self._classify(c)

            if improved:
                self._peak[c] = self._primary(c)
                self._loss_at_peak[c] = self._loss_ema[c]
                self._stale[c] = 0
            elif declined:
                self._stale[c] += 1
            else:  # plateau — neither improving nor declining past the deadband
                self._stale[c] = 0

            if self._cooldown[c] > 0:
                self._cooldown[c] -= 1
            elif self._stale[c] >= self.patience:
                self.multiplier[c] = max(
                    self.floor, min(self.ceiling, self.multiplier[c] * self.decay)
                )
                self._cooldown[c] = self.cooldown_period
                self._stale[c] = 0
                self.adjustments += 1

        return self.current_weights()

    def _classify(self, c: int) -> tuple[bool, bool]:
        """Return (improved, declined) for class ``c`` under the active metric."""
        prim = self._primary(c)
        if prim is None:
            return False, False
        peak = self._peak[c]
        if peak is None:
            return True, False  # first supported observation establishes the peak
        improved = prim > peak + self.min_delta
        declined = prim < peak - self.min_delta
        if self.metric == "both" and declined:
            # F1 declined; require rising val loss to corroborate genuine
            # overfitting (rules out a benign F1 wobble at stable loss).
            loss_now = self._loss_ema[c]
            loss_ref = self._loss_at_peak[c]
            loss_rising = (
                loss_now is not None
                and loss_ref is not None
                and loss_now > loss_ref + self.min_delta
            )
            declined = declined and loss_rising
        return improved, declined

    def current_weights(self) -> torch.Tensor:
        """base_weights * multipliers, renormalised to mean 1 (constant loss scale)."""
        mult = torch.tensor(self.multiplier, dtype=self._base.dtype, device=self._base.device)
        w = self._base * mult
        return w * (self.num_classes / w.sum().clamp(min=1e-8))

    def multipliers(self) -> dict[str, float]:
        """Current per-class multiplier, keyed by ``str(class_index)`` (for epoch_msg)."""
        return {str(c): float(self.multiplier[c]) for c in range(self.num_classes)}

    def to_dict(self) -> dict:
        """Serialise the full mutable controller state (plain floats/ints/lists).

        Everything the between-epoch ``update`` reads or writes — the running
        multiplier, per-class EMAs/peaks, stale/cooldown counters, the
        adjustment tally, and the constructor knobs — so a resumed run's next
        ``update`` behaves as if training had never stopped. The immutable
        ``base_weights`` vector is intentionally NOT serialised; it is rebuilt
        from the config on resume and passed to :meth:`from_dict`.
        """
        return {
            "num_classes": self.num_classes,
            "metric": self.metric,
            "patience": self.patience,
            "ema_decay": self.ema_decay,
            "decay": self.decay,
            "floor": self.floor,
            "ceiling": self.ceiling,
            "cooldown": self.cooldown_period,
            "min_delta": self.min_delta,
            "multiplier": list(self.multiplier),
            "f1_ema": list(self._f1_ema),
            "loss_ema": list(self._loss_ema),
            "peak": list(self._peak),
            "loss_at_peak": list(self._loss_at_peak),
            "stale": list(self._stale),
            "cooldown_state": list(self._cooldown),
            "adjustments": self.adjustments,
        }

    @classmethod
    def from_dict(cls, data: dict, base_weights: torch.Tensor) -> "DynamicClassWeightController":
        """Reconstruct a controller from :meth:`to_dict` + the rebuilt base vector."""
        ctrl = cls(
            int(data["num_classes"]),
            base_weights,
            metric=data.get("metric", "val_f1"),
            patience=int(data.get("patience", 2)),
            ema_decay=float(data.get("ema_decay", 0.5)),
            decay=float(data.get("decay", 0.8)),
            floor=float(data.get("floor", 0.25)),
            ceiling=float(data.get("ceiling", 1.0)),
            cooldown=int(data.get("cooldown", 1)),
            min_delta=float(data.get("min_delta", 0.005)),
        )
        ctrl.multiplier = list(data["multiplier"])
        ctrl._f1_ema = list(data["f1_ema"])
        ctrl._loss_ema = list(data["loss_ema"])
        ctrl._peak = list(data["peak"])
        ctrl._loss_at_peak = list(data["loss_at_peak"])
        ctrl._stale = list(data["stale"])
        ctrl._cooldown = list(data["cooldown_state"])
        ctrl.adjustments = int(data["adjustments"])
        return ctrl
