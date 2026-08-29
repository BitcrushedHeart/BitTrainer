"""Per-sample training statistics export (Bitcrush ISSUE-0850).

``_train_one_epoch`` already computes per-example cross-entropy under
``no_grad`` for the per-class telemetry and then throws the per-image values
away. This module keeps them: one row per image path holding the train-loss
trajectory across epochs (train split) or the per-epoch probability vector and
loss (val split).

The consumer is Bitcrush Engine's Dataset Hygiene, which ranks label issues from
these signals — images that stay high-loss late in training / are "learned
last" are the classic label-noise fingerprint, and val probabilities are
genuinely out-of-sample so they feed a confident-learning estimate directly.

File format (``sample_stats.json``, version 1)::

    {
      "version": 1,
      "class_names": [...],
      "epochs": E,
      "samples": [
        {"path": str, "label": int, "split": "train",
         "train_loss": [float | null] * E},
        {"path": str, "label": int, "split": "val",
         "val_loss": [float | null] * E,
         "val_probs": [[float] * C | null] * E},
      ]
    }

Trajectories are epoch-indexed lists; an epoch a path was absent from (MixUp
batch, dropped sample) holds ``null`` so positions stay aligned.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import torch

_ROUND = 5


def _r(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return round(float(x), _ROUND)


class SampleStatsRecorder:
    """Accumulate per-path, per-epoch losses / probabilities and write JSON."""

    def __init__(self, class_names: Iterable[str]) -> None:
        self.class_names = list(class_names)
        self._rows: dict[str, dict] = {}
        self._max_epoch = -1
        # (path, epoch) → [sum, count] for within-epoch replication averaging.
        self._train_acc: dict[tuple[str, int], list[float]] = {}

    # -- recording ---------------------------------------------------------

    def _row(self, path: str, label: int, split: str) -> dict:
        row = self._rows.get(path)
        if row is None:
            row = {"path": path, "label": int(label), "split": split}
            self._rows[path] = row
        return row

    def record_train(
        self,
        epoch: int,
        paths: list[str],
        labels: list[int],
        losses: Iterable[float],
    ) -> None:
        self._max_epoch = max(self._max_epoch, epoch)
        for path, label, loss in zip(paths, labels, losses, strict=True):
            self._row(path, int(label), "train")
            acc = self._train_acc.setdefault((path, epoch), [0.0, 0])
            acc[0] += float(loss)
            acc[1] += 1

    def record_val(
        self,
        epoch: int,
        paths: list[str],
        labels: list[int],
        probs: torch.Tensor,
    ) -> None:
        """``probs`` is ``[N, C]`` (already softmaxed), row-aligned with ``paths``."""
        self._max_epoch = max(self._max_epoch, epoch)
        p = probs.detach().float().cpu()
        if p.shape[0] != len(paths):
            raise RuntimeError(
                f"sample_stats: {p.shape[0]} prob rows for {len(paths)} val paths"
            )
        for i, (path, label) in enumerate(zip(paths, labels, strict=True)):
            row = self._row(path, int(label), "val")
            per_epoch = row.setdefault("_val", {})
            p_label = float(p[i, int(label)]) if 0 <= int(label) < p.shape[1] else 0.0
            loss = -math.log(max(p_label, 1e-12))
            per_epoch[epoch] = ([_r(v) for v in p[i].tolist()], _r(loss))

    # -- output ------------------------------------------------------------

    @property
    def epochs(self) -> int:
        return self._max_epoch + 1

    def rows(self) -> list[dict]:
        """Materialise epoch-aligned rows (does not mutate internal state)."""
        n = self.epochs
        out: list[dict] = []
        for path, row in self._rows.items():
            item = {"path": path, "label": row["label"], "split": row["split"]}
            if row["split"] == "train":
                traj: list[float | None] = []
                for e in range(n):
                    acc = self._train_acc.get((path, e))
                    traj.append(_r(acc[0] / acc[1]) if acc else None)
                item["train_loss"] = traj
            else:
                per_epoch = row.get("_val", {})
                item["val_probs"] = [per_epoch[e][0] if e in per_epoch else None for e in range(n)]
                item["val_loss"] = [per_epoch[e][1] if e in per_epoch else None for e in range(n)]
            out.append(item)
        return out

    def write(self, path: Path | str) -> Path:
        """Overwrite ``path`` atomically with the current snapshot."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "class_names": self.class_names,
            "epochs": self.epochs,
            "samples": self.rows(),
        }
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp.replace(target)
        return target
