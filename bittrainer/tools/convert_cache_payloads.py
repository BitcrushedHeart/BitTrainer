"""Convert SmartCache ``.pt`` payloads from pickled ndarrays to torch tensors.

Bitcrush ISSUE-0847 cache audit: ``torch.save({"tensor": <np.ndarray>})`` goes
through pickle protocol 2, which has no BINBYTES opcode, so the uint8 buffer
is stored as a latin-1 ``str`` re-encoded as UTF-8 — every byte >= 0x80 costs
two bytes (~1.44x inflation) and ``torch.load`` is ~5x slower. The writer now
stores a ``torch.Tensor``; this tool upgrades existing files in place.

Usage::

    python -m bittrainer.tools.convert_cache_payloads <cache_dir> [--workers N] [--dry-run]

Each file is loaded, re-saved to a temp file in the same directory and
atomically swapped in with ``os.replace`` — a concurrent reader sees either
the old or the new complete file, never a partial one. Files whose payload is
already a ``torch.Tensor`` are skipped, so the run is idempotent. Metadata keys
are carried over untouched.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import pickle
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

_LOAD_ERRORS = (
    OSError,
    RuntimeError,
    EOFError,
    ValueError,
    KeyError,
    TypeError,
    pickle.UnpicklingError,
    zipfile.BadZipFile,
)


@dataclass
class ConvertReport:
    """Counts and byte totals for one converter run."""

    dry_run: bool = False
    converted: int = 0
    skipped: int = 0
    failed: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    converted_bytes_before: int = 0
    converted_bytes_after: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.converted + self.skipped + self.failed

    def summary(self) -> str:
        mode = "dry-run" if self.dry_run else "converted"
        lines = [
            f"{mode}: {self.converted} converted, {self.skipped} skipped "
            f"(already tensor), {self.failed} failed, {self.total} total",
            f"bytes before: {self.bytes_before:,}  after: {self.bytes_after:,}  "
            f"saved: {self.bytes_before - self.bytes_after:,}",
        ]
        if self.converted_bytes_before:
            ratio = self.converted_bytes_after / self.converted_bytes_before
            lines.append(
                f"converted files: {self.converted_bytes_before:,} -> "
                f"{self.converted_bytes_after:,} bytes (ratio {ratio:.3f})"
            )
        if self.dry_run and self.converted:
            lines.append("(dry-run: 'after' equals 'before'; no files were touched)")
        for path, err in self.failures[:20]:
            lines.append(f"  FAILED {path}: {err}")
        if len(self.failures) > 20:
            lines.append(f"  ... {len(self.failures) - 20} more failures")
        return "\n".join(lines)


def _convert_one(pt_path: Path, *, dry_run: bool) -> tuple[str, int, int, str | None]:
    """Return ``(status, bytes_before, bytes_after, error)`` for one file.

    ``status`` is ``"converted"``, ``"skipped"`` or ``"failed"``.
    """
    try:
        size_before = pt_path.stat().st_size
    except OSError as exc:
        return "failed", 0, 0, str(exc)
    try:
        payload = torch.load(pt_path, weights_only=False, map_location="cpu")
        if not isinstance(payload, dict) or "tensor" not in payload:
            return "failed", size_before, size_before, "payload has no 'tensor' key"
        tensor = payload["tensor"]
        if isinstance(tensor, torch.Tensor):
            return "skipped", size_before, size_before, None
        if not isinstance(tensor, np.ndarray):
            return (
                "failed",
                size_before,
                size_before,
                f"unexpected tensor payload type {type(tensor).__name__}",
            )
        if dry_run:
            return "converted", size_before, size_before, None

        new_payload = dict(payload)
        new_payload["tensor"] = torch.from_numpy(np.ascontiguousarray(tensor))
        tmp_path = pt_path.with_name(
            f"{pt_path.name}.{os.getpid()}.{threading.get_ident()}.convert.tmp"
        )
        try:
            torch.save(new_payload, tmp_path)
            os.replace(tmp_path, pt_path)
        except _LOAD_ERRORS as exc:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return "failed", size_before, size_before, f"write failed: {exc}"
        return "converted", size_before, pt_path.stat().st_size, None
    except _LOAD_ERRORS as exc:
        return "failed", size_before, size_before, f"{type(exc).__name__}: {exc}"


def convert_cache_dir(
    cache_dir: Path | str,
    *,
    workers: int = 4,
    dry_run: bool = False,
    progress_every: int = 0,
    out=None,
) -> ConvertReport:
    """Convert every legacy ``.pt`` under *cache_dir* (non-recursive).

    Safe to run while a trainer reads the cache: each replacement is atomic.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"cache dir not found: {cache_dir}")
    files = sorted(p for p in cache_dir.iterdir() if p.suffix == ".pt" and p.is_file())
    report = ConvertReport(dry_run=dry_run)
    workers = max(1, int(workers))
    start = time.monotonic()
    done = 0

    def _record(pt_path: Path, result: tuple[str, int, int, str | None]) -> None:
        status, before, after, err = result
        report.bytes_before += before
        report.bytes_after += after
        if status == "converted":
            report.converted += 1
            report.converted_bytes_before += before
            report.converted_bytes_after += after
        elif status == "skipped":
            report.skipped += 1
        else:
            report.failed += 1
            report.failures.append((str(pt_path), err or "unknown error"))

    if workers == 1:
        for pt_path in files:
            _record(pt_path, _convert_one(pt_path, dry_run=dry_run))
            done += 1
            if out is not None and progress_every and done % progress_every == 0:
                print(f"  {done}/{len(files)} ({time.monotonic() - start:.0f}s)", file=out)
        return report

    # Bounded in-flight window: submitting every file up front keeps 160k+
    # futures (and their pending results) alive for the whole run. Only
    # ``2 * workers`` files are ever queued or loaded at once, and the payloads
    # are dropped as soon as each is recorded.
    import gc

    window = max(2, 2 * workers)
    pending: dict[concurrent.futures.Future, Path] = {}
    queue = iter(files)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or True:
            while len(pending) < window:
                nxt = next(queue, None)
                if nxt is None:
                    break
                pending[pool.submit(_convert_one, nxt, dry_run=dry_run)] = nxt
            if not pending:
                break
            finished, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for fut in finished:
                _record(pending.pop(fut), fut.result())
                done += 1
                if out is not None and progress_every and done % progress_every == 0:
                    gc.collect()
                    print(
                        f"  {done}/{len(files)} ({time.monotonic() - start:.0f}s, "
                        f"rss {_rss_mb():.0f} MB)",
                        file=out,
                    )
    return report


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:  # noqa: BLE001
        return float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bittrainer.tools.convert_cache_payloads",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("cache_dir", help="SmartCache directory containing the .pt files")
    parser.add_argument("--workers", type=int, default=4, help="thread-pool size (default 4)")
    parser.add_argument(
        "--dry-run", action="store_true", help="count legacy files; write nothing"
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="print a progress line every N files (0 disables)",
    )
    args = parser.parse_args(argv)
    try:
        report = convert_cache_dir(
            args.cache_dir,
            workers=args.workers,
            dry_run=args.dry_run,
            progress_every=args.progress_every,
            out=sys.stdout,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(report.summary())
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
