"""Tests for progress.py — stage transitions, throttling, seq stamping."""

import bittrainer.progress as progress_mod
from bittrainer.progress import ProgressEmitter, Stage


class TestStage:
    def test_stage_emits_immediately(self):
        frames = []
        em = ProgressEmitter(frames.append)
        em.stage(Stage.scanning, "Scanning dataset")
        em.stage(Stage.caching, "Caching")
        assert [f["stage"] for f in frames] == ["scanning", "caching"]
        assert frames[0]["status_text"] == "Scanning dataset"
        assert frames[0]["type"] == "training_progress"

    def test_seq_monotonic_across_methods(self):
        frames = []
        em = ProgressEmitter(frames.append)
        em.stage(Stage.training)
        em.step(1, 2, force=True)
        em.raw({"type": "epoch_complete", "epoch": 1})
        assert [f["seq"] for f in frames] == [1, 2, 3]
        assert all("elapsed_s" in f for f in frames)

    def test_raw_inherits_current_stage(self):
        frames = []
        em = ProgressEmitter(frames.append)
        em.stage(Stage.training)
        em.raw({"type": "autobatch", "batch_size": 28})
        assert frames[-1]["stage"] == "training"

    def test_raw_stage_updates_current(self):
        frames = []
        em = ProgressEmitter(frames.append)
        em.raw({"stage": "compiling", "status_text": "Compiling model (shape 1/9)"})
        em.raw({"type": "autobatch"})
        assert frames[-1]["stage"] == "compiling"


class TestStepThrottle:
    def test_steps_within_interval_are_dropped(self, monkeypatch):
        clock = {"t": 100.0}
        monkeypatch.setattr(progress_mod.time, "monotonic", lambda: clock["t"])
        frames = []
        em = ProgressEmitter(frames.append, min_interval=0.25)
        em.step(1, 100)
        em.step(2, 100)  # same instant — dropped
        clock["t"] += 0.3
        em.step(3, 100)  # past interval — emitted
        steps = [f["step"] for f in frames]
        assert steps == [1, 3]

    def test_final_step_always_emits(self, monkeypatch):
        clock = {"t": 100.0}
        monkeypatch.setattr(progress_mod.time, "monotonic", lambda: clock["t"])
        frames = []
        em = ProgressEmitter(frames.append, min_interval=0.25)
        em.step(99, 100)
        em.step(100, 100)  # final — bypasses throttle
        assert [f["step"] for f in frames] == [99, 100]

    def test_stage_transition_resets_throttle(self, monkeypatch):
        clock = {"t": 100.0}
        monkeypatch.setattr(progress_mod.time, "monotonic", lambda: clock["t"])
        frames = []
        em = ProgressEmitter(frames.append, min_interval=0.25)
        em.step(1, 100)
        em.stage(Stage.validating)
        em.step(1, 50)  # new stage — first step emits despite the clock not moving
        assert frames[-1] == {
            "step": 1, "total_steps": 50, "type": "training_progress",
            "stage": "validating", "seq": 3, "elapsed_s": 0.0,
        }
