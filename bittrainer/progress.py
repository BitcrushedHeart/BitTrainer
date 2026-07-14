"""Structured training progress: stage model + stamped, throttled emitter.

Every phase of a run emits through one ProgressEmitter so the UI gets a single
monotonic ``seq`` stream with a ``stage`` on every frame — nothing looks hung.
Messages stay plain dicts over the multiprocessing queue (Windows spawn-safe).
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Callable


class Stage(str, Enum):
    preparing = "preparing"
    scanning = "scanning"
    downloading_model = "downloading_model"
    loading_model = "loading_model"
    face_detection = "face_detection"
    caching = "caching"
    embedding_build = "embedding_build"
    autobatch = "autobatch"
    compiling = "compiling"
    training = "training"
    validating = "validating"
    comparing = "comparing"
    saving = "saving"
    promoting = "promoting"
    backing_up = "backing_up"
    resuming = "resuming"


class ProgressEmitter:
    """Stamps every frame with ``stage``/``seq``/``elapsed_s`` and throttles hot loops.

    ``stage()`` emits immediately on transitions; ``step()`` is time-throttled
    (default ~4 Hz) except for the final step of a loop; ``raw()`` passes
    legacy dict messages through with the same stamping, so existing call
    sites join the stream by swapping ``cb`` for ``emitter.raw``.
    """

    def __init__(self, cb: Callable[[dict], None], *, min_interval: float = 0.25) -> None:
        self._cb = cb
        self._min_interval = min_interval
        self._last_emit = 0.0
        self._seq = 0
        self._start = time.monotonic()
        self._stage: str | None = None

    def _send(self, msg: dict) -> None:
        self._seq += 1
        msg.setdefault("type", "training_progress")
        if self._stage is not None:
            msg.setdefault("stage", self._stage)
        msg["seq"] = self._seq
        msg["elapsed_s"] = round(time.monotonic() - self._start, 2)
        self._cb(msg)

    def stage(self, stage: Stage | str, status_text: str | None = None, **fields: object) -> None:
        self._stage = str(getattr(stage, "value", stage))
        self._last_emit = 0.0
        msg: dict = {"stage": self._stage, **fields}
        if status_text is not None:
            msg["status_text"] = status_text
        self._send(msg)

    def step(
        self,
        step: int,
        total: int,
        status_text: str | None = None,
        *,
        force: bool = False,
        **fields: object,
    ) -> None:
        now = time.monotonic()
        if not force and step < total and (now - self._last_emit) < self._min_interval:
            return
        self._last_emit = now
        msg: dict = {"step": step, "total_steps": total, **fields}
        if status_text is not None:
            msg["status_text"] = status_text
        self._send(msg)

    def raw(self, msg: dict) -> None:
        incoming = dict(msg)
        stage = incoming.get("stage")
        if stage is not None:
            self._stage = str(stage)
        self._send(incoming)
