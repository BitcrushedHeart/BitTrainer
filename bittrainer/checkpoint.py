"""Checkpoint comparison and management."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_BEST_NAME = "best.pt"


def compare_checkpoints(
    old_f1: float | None,
    new_f1: float,
) -> str:
    """Compare old and new F1 scores. Returns 'old' or 'new'."""
    if old_f1 is None:
        return "new"
    return "new" if new_f1 >= old_f1 else "old"


def save_if_better(
    *,
    new_checkpoint_path: str,
    new_f1: float,
    concept_folder: str,
    old_f1: float | None,
    best_model_name: str = _BEST_NAME,
) -> dict:
    """Compare new checkpoint against existing best; keep the winner.

    Returns dict with 'kept' ('old'|'new') and 'checkpoint_path'.
    """
    ckpt_dir = Path(concept_folder) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / best_model_name
    new_path = Path(new_checkpoint_path)

    winner = compare_checkpoints(old_f1, new_f1)

    if winner == "new":
        # Move new checkpoint to best.pt (overwrite if exists)
        if new_path != best_path:
            shutil.move(str(new_path), str(best_path))
        logger.info("New checkpoint wins (F1=%.4f vs old=%.4f)", new_f1, old_f1 or 0.0)
        return {"kept": "new", "checkpoint_path": str(best_path)}
    else:
        # Remove candidate, keep old
        if new_path.exists() and new_path != best_path:
            new_path.unlink()
        logger.info("Old checkpoint wins (F1=%.4f vs new=%.4f)", old_f1, new_f1)
        return {"kept": "old", "checkpoint_path": str(best_path)}
