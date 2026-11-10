"""Where a pack run stands, from its ``--out`` folder (Bitcrush ISSUE-1655).

    python -m bittrainer.tools.backbone_status --out DIR

Reads what ``backbone_pack_train`` keeps there: ``status.json`` (running,
paused, finished, failed or refused, with the run's plan and process id) and the
time-stamped frames in ``progress.jsonl``. Prints the epoch and batch, the speed,
the last validation score, skipped optimiser steps, when the run last made
progress and roughly how long is left. Exit code 0 when the run is running,
paused or finished; 1 when it failed, was refused, has been silent too long, or
its process is gone without an ending.

The time left is the training time only (validation and backups add a little),
and the 384px epochs are estimated from the 224px ones by pixel count until one
of them has run.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

_BAD_STATES = frozenset({"failed", "refused", "stopped", "silent"})


def read_run(out: Path) -> tuple[list[dict], dict]:
    """The folder's frames and its status. A line the runner was still writing
    is skipped; a folder with neither file gives ``([], {})``."""
    out = Path(out)
    frames: list[dict] = []
    progress = out / "progress.jsonl"
    if progress.is_file():
        for line in progress.read_text(encoding="utf-8").splitlines():
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    status: dict = {}
    if (out / "status.json").is_file():
        try:
            status = json.loads((out / "status.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status = {}
    return frames, status


def process_alive(status: dict) -> bool | None:
    """Whether the run's process still exists; None when that cannot be told
    from here (another machine, or Windows, where probing a process is unsafe)."""
    pid = status.get("pid")
    if not pid or os.name != "posix" or status.get("host") != socket.gethostname():
        return None
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_tail(epoch: int, plan: dict) -> bool:
    epochs = int(plan.get("epochs") or 0)
    tail = int(plan.get("finetune_epochs") or 0)
    has_tail = int(plan.get("finetune_image_size") or 0) > 0 and 0 < tail < epochs
    return has_tail and epoch > epochs - tail


def _seconds_left(
    plan: dict, epochs_done: list[dict], step: dict | None, epoch: int | None
):
    """Training seconds still ahead, and whether the tail's share is a guess."""
    total = int(plan.get("epochs") or (step or {}).get("epochs") or 0)
    if not total or epoch is None:
        return None, False
    measured: dict[bool, list[float]] = {False: [], True: []}
    for frame in epochs_done:
        if frame.get("train_seconds"):
            measured[_is_tail(int(frame["epoch"]), plan)].append(
                float(frame["train_seconds"])
            )
    per_epoch: dict[bool, float | None] = {
        phase: (sum(values) / len(values) if values else None)
        for phase, values in measured.items()
    }
    if step is not None and step.get("images_per_second"):
        phase = _is_tail(int(step["epoch"]), plan)
        if per_epoch[phase] is None:
            size = step.get("batch_size") or plan.get("batch_size") or 0
            per_epoch[phase] = (
                int(step["batches"]) * int(size) / float(step["images_per_second"])
            )
    estimated = False
    if per_epoch[True] is None and per_epoch[False] is not None:
        main = float(plan.get("image_size") or 0)
        tail = float(plan.get("finetune_image_size") or 0)
        if main and tail:
            per_epoch[True] = per_epoch[False] * (tail / main) ** 2
            estimated = True

    left = 0.0
    first_whole = epoch + 1
    if step is not None and int(step["epoch"]) == epoch:
        share = 1.0 - int(step["batch"]) / max(int(step["batches"]), 1)
        current = per_epoch[_is_tail(epoch, plan)]
        if current is None:
            return None, False
        left += share * current
    for number in range(first_whole, total + 1):
        seconds = per_epoch[_is_tail(number, plan)]
        if seconds is None:
            return None, False
        left += seconds
    uses_tail = any(_is_tail(n, plan) for n in range(first_whole, total + 1))
    return left, estimated and uses_tail


def summarise(
    frames: list[dict],
    status: dict,
    *,
    now: float,
    alive: bool | None = None,
    stale_minutes: float = 20,
) -> dict:
    """One dict for the run: its state, position, numbers and time left."""
    plan = status.get("plan") or {}
    step: dict | None = None
    epochs_done: list[dict] = []
    backup: dict | None = None
    warnings = 0
    last_time: float | None = None
    for frame in frames:
        if frame.get("t") is not None:
            last_time = float(frame["t"])
        stage = frame.get("stage")
        if stage == "training_step":
            step = frame
        elif stage == "training" and "loss" in frame:
            epochs_done.append(frame)
            step = None  # that epoch is over
        elif stage == "warning":
            warnings += 1
        elif frame.get("type") == "backup_complete":
            backup = frame
    last_epoch = epochs_done[-1] if epochs_done else None
    position = step or last_epoch
    epoch = int(position["epoch"]) if position else None
    since = None if last_time is None else max(0.0, now - last_time)

    state = str(status.get("status") or "unknown")
    if state == "running":
        if alive is False:
            state = "stopped"
        elif since is not None and since > stale_minutes * 60:
            state = "silent"

    if state == "finished":
        seconds_left, estimated = 0, False
    else:
        seconds_left, estimated = _seconds_left(plan, epochs_done, step, epoch)
    return {
        "state": state,
        "reason": status.get("reason"),
        "epoch": epoch,
        "epochs": int(plan.get("epochs") or (position or {}).get("epochs") or 0)
        or None,
        "batch": step.get("batch") if step else None,
        "batches": step.get("batches") if step else None,
        "image_size": (position or {}).get("image_size") or plan.get("image_size"),
        "batch_size": (position or {}).get("batch_size") or plan.get("batch_size"),
        "loss": (position or {}).get("loss"),
        "images_per_second": (position or {}).get("images_per_second"),
        "validation_score": (last_epoch or status).get("validation_score"),
        "best_score": (last_epoch or {}).get("best_score"),
        "best_epoch": (last_epoch or status).get("best_epoch"),
        "skipped_steps": (last_epoch or {}).get("skipped_steps_total"),
        "skipped_this_epoch": step.get("skipped_steps") if step else None,
        "warnings": warnings,
        "last_backup_step": backup.get("global_step") if backup else None,
        "seconds_since_progress": since,
        "seconds_left": seconds_left,
        "tail_is_estimated": estimated,
    }


def _duration(seconds: float) -> str:
    minutes = int(round(seconds / 60))
    if minutes < 1:
        return "under a minute"
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours} h {minutes} min"
    days, hours = divmod(hours, 24)
    return f"{days} d {hours} h"


def report_lines(summary: dict) -> list[str]:
    """The summary as a few plain lines."""
    state = summary["state"]
    where = ""
    if summary["epoch"] is not None:
        where = f"epoch {summary['epoch']} of {summary['epochs']}"
        if summary["batch"] is not None:
            where += f", batch {summary['batch']:,} of {summary['batches']:,}"
        if summary["image_size"]:
            where += f" ({summary['image_size']}px, batch {summary['batch_size']})"
    since = summary["seconds_since_progress"]
    # A run stopped before its first report has no position to name.
    at = where or "before its first report"
    head = {
        "running": f"running: {where or 'starting, nothing reported yet'}",
        "silent": (
            f"running, but no progress for {_duration(since or 0)}: {at}. "
            "Look at the machine."
        ),
        "stopped": (
            f"no longer running and it did not say why (its process is gone): {at}. "
            "The same command resumes it from its last backup."
        ),
        "paused": f"paused: {at}. The same command resumes it.",
        "finished": f"finished: {at}",
        "failed": f"failed: {summary['reason'] or 'see error.txt'} ({at})",
        "refused": f"refused before training: {summary['reason'] or 'no reason recorded'}",
    }.get(state, "no status file here: not a run folder, or a run from an older runner")
    lines = [head]
    if summary["images_per_second"]:
        line = f"speed {summary['images_per_second']:.1f} images/s"
        if summary["loss"] is not None:
            line += f"; loss {summary['loss']:.4f}"
        lines.append(line)
    if summary["validation_score"] is not None:
        line = f"last validation {summary['validation_score']:.4f}"
        if summary["best_score"] is not None and summary["best_epoch"]:
            line += (
                f" (best {summary['best_score']:.4f} at epoch {summary['best_epoch']})"
            )
        lines.append(line)
    if summary["skipped_steps"] is not None or summary["skipped_this_epoch"]:
        line = f"skipped optimiser steps: {summary['skipped_steps'] or 0} in finished epochs"
        if summary["skipped_this_epoch"] is not None:
            line += f", {summary['skipped_this_epoch']} in this one"
        lines.append(line)
    if summary["warnings"]:
        lines.append(
            f"warnings from the run: {summary['warnings']} (see progress.jsonl)"
        )
    tail = []
    if summary["last_backup_step"] is not None:
        tail.append(f"last backup at step {summary['last_backup_step']:,}")
    if since is not None and state in ("running", "paused", "stopped"):
        tail.append(f"last progress {_duration(since)} ago")
    if tail:
        lines.append("; ".join(tail))
    left = summary["seconds_left"]
    if left and state != "finished":
        line = f"training time left: about {_duration(left)}"
        if summary["tail_is_estimated"]:
            line += " (the larger-image epochs are an estimate until one has run)"
        lines.append(line)
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bittrainer.tools.backbone_status",
        description="Where a pack run stands, from its --out folder.",
    )
    parser.add_argument("--out", required=True, help="the run's --out folder")
    parser.add_argument("--stale-minutes", type=float, default=20)
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)

    frames, status = read_run(Path(args.out))
    summary = summarise(
        frames,
        status,
        now=time.time(),
        alive=process_alive(status),
        stale_minutes=args.stale_minutes,
    )
    print(
        json.dumps(summary, indent=1) if args.json else "\n".join(report_lines(summary))
    )
    return 1 if summary["state"] in _BAD_STATES else 0


if __name__ == "__main__":
    raise SystemExit(main())
