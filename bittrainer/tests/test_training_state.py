"""Backup/pause/resume primitives (Bitcrush ISSUE-0405).

Unit-level coverage for ``bittrainer.training_state`` plus the additive
serialisation hooks the backup envelope leans on (EMA ``full_state_dict``,
``_SWA.load_state_dict``, ``DynamicClassWeightController.to_dict/from_dict``).
CPU-only, no training loop, tiny tensors.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from adv_optm import Prodigy_adv

from bittrainer.training_state import (
    BACKUP_FORMAT_VERSION,
    TrainingStateManager,
    _FixedBatchSampler,
    backup_on_exception,
    capture_rng_states,
    capture_optimizer_aux_state,
    fingerprint_matches,
    make_fingerprint,
    prime_optimizer_after_resume,
    restore_optimizer_aux_state,
    restore_rng_states,
    sanitize_for_backup,
)

_DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# TrainingStateManager: save/load, rotation, corruption, tmp files
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_preserves_tensors_counters_and_rng(tmp_path):
    """Regression guard for the backup envelope: tensors, counters and RNG
    survive a save→load round-trip byte-for-byte, and version/reason are stamped."""
    mgr = TrainingStateManager(tmp_path / "backups")
    rng = capture_rng_states(_DEVICE)
    state = {
        "global_step": 42,
        "epoch": 3,
        "model": {"w": torch.arange(6, dtype=torch.float32).reshape(2, 3)},
        "rng_now": rng,
    }
    path = mgr.save(state, reason="periodic")
    assert path.name == "backup_00000042_0000.pt"  # first save = seq 0

    loaded = mgr.load_latest()
    assert loaded is not None
    assert loaded["version"] == BACKUP_FORMAT_VERSION
    assert loaded["reason"] == "periodic"
    assert loaded["global_step"] == 42
    assert torch.equal(loaded["model"]["w"], state["model"]["w"])
    # RNG round-trips: restoring it reproduces the same draw.
    restore_rng_states(loaded["rng_now"], _DEVICE)
    a = torch.rand(3)
    restore_rng_states(rng, _DEVICE)
    b = torch.rand(3)
    assert torch.equal(a, b)


def test_save_does_not_mutate_callers_dict(tmp_path):
    mgr = TrainingStateManager(tmp_path / "b")
    state = {"global_step": 1}
    mgr.save(state, reason="pause")
    assert "version" not in state and "saved_at" not in state


def test_rotation_keeps_exactly_last_two(tmp_path):
    """Regression guard: keep=2 rotation deletes the oldest on the 3rd save and
    leaves both survivors loadable."""
    mgr = TrainingStateManager(tmp_path / "b", keep=2)
    for step in (10, 20, 30):
        mgr.save({"global_step": step, "val": step}, reason="periodic")
    backups = mgr.list_backups()
    assert len(backups) == 2
    steps = sorted(TrainingStateManager._sort_key(p)[0] for p in backups)
    assert steps == [20, 30]
    # newest loads and is the step-30 save
    assert mgr.load_latest()["global_step"] == 30
    # both survivors are individually loadable
    for p in backups:
        assert torch.load(p, map_location="cpu", weights_only=False)["global_step"] in (20, 30)


def test_same_step_saves_disambiguated_by_seq(tmp_path):
    mgr = TrainingStateManager(tmp_path / "b", keep=5)
    mgr.save({"global_step": 7, "tag": "a"}, reason="periodic")
    mgr.save({"global_step": 7, "tag": "b"}, reason="periodic")
    assert len(mgr.list_backups()) == 2
    assert mgr.load_latest()["tag"] == "b"  # later save wins the (step, seq) sort


def test_corrupt_newest_falls_back_to_older(tmp_path):
    """Regression guard: a truncated newest backup is skipped; load_latest
    returns the next-older valid one."""
    mgr = TrainingStateManager(tmp_path / "b", keep=3)
    mgr.save({"global_step": 100, "tag": "old"}, reason="periodic")
    newest = mgr.save({"global_step": 200, "tag": "new"}, reason="periodic")
    # Truncate the newest file to unpickleable garbage.
    with open(newest, "r+b") as f:
        f.truncate(3)
    loaded = mgr.load_latest()
    assert loaded is not None
    assert loaded["tag"] == "old"


def test_stray_tmp_files_ignored(tmp_path):
    """Regression guard: leftover .tmp files never masquerade as backups."""
    mgr = TrainingStateManager(tmp_path / "b", keep=5)
    real = mgr.save({"global_step": 5, "tag": "real"}, reason="periodic")
    (mgr.backup_dir / "backup_00000009_0009.pt.tmp").write_bytes(b"garbage")
    backups = mgr.list_backups()
    assert backups == [real]
    assert mgr.load_latest()["tag"] == "real"


def test_delete_all_removes_files_and_dir(tmp_path):
    mgr = TrainingStateManager(tmp_path / "b")
    mgr.save({"global_step": 1}, reason="periodic")
    mgr.delete_all()
    assert not (tmp_path / "b").exists()


def test_load_latest_none_when_empty(tmp_path):
    mgr = TrainingStateManager(tmp_path / "b")
    assert mgr.load_latest() is None
    assert mgr.list_backups() == []


# ---------------------------------------------------------------------------
# RNG capture/restore: python random continuity
# ---------------------------------------------------------------------------


def test_python_random_stream_continues_identically_after_restore():
    """Regression guard: capture_rng_states snapshots python ``random`` (the
    generator the samplers shuffle with) so the stream resumes identically."""
    random.seed(1234)
    _ = [random.random() for _ in range(5)]
    snap = capture_rng_states(_DEVICE)
    expected = [random.random() for _ in range(5)]

    # Perturb the global python RNG, then restore and re-draw.
    random.seed(999)
    _ = [random.random() for _ in range(3)]
    restore_rng_states(snap, _DEVICE)
    got = [random.random() for _ in range(5)]
    assert got == expected


def test_restore_survives_torch_save_roundtrip(tmp_path):
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    snap = capture_rng_states(_DEVICE)
    torch.save(snap, tmp_path / "rng.pt")
    reloaded = torch.load(tmp_path / "rng.pt", map_location="cpu", weights_only=False)
    expected = (torch.rand(2), np.random.rand(2), random.random())
    restore_rng_states(reloaded, _DEVICE)
    got = (torch.rand(2), np.random.rand(2), random.random())
    assert torch.equal(got[0], expected[0])
    assert np.allclose(got[1], expected[1])
    assert got[2] == expected[2]


# ---------------------------------------------------------------------------
# EMA full_state_dict roundtrip (n_updates preserved => same effective decay)
# ---------------------------------------------------------------------------


def test_ema_full_state_roundtrip_preserves_effective_decay():
    """Regression guard: restoring EMA via full_state_dict preserves n_updates,
    so the next effective decay matches an uninterrupted EMA exactly."""
    from bittrainer.ema import ModelEMA

    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    ema = ModelEMA(model, decay=0.999, warmup=10)
    for _ in range(5):
        with torch.no_grad():
            model.weight.add_(0.01)
        ema.update(model)
    assert ema.n_updates == 5
    payload = ema.full_state_dict()

    fresh = ModelEMA(nn.Linear(4, 3), decay=0.5, warmup=99)
    fresh.load_full_state_dict(payload)
    assert fresh.n_updates == 5
    assert fresh.decay == 0.999 and fresh.warmup == 10
    # next effective decay identical
    assert fresh._effective_decay() == ema._effective_decay()
    for a, b in zip(fresh.module.parameters(), ema.module.parameters()):
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# _SWA roundtrip
# ---------------------------------------------------------------------------


def test_swa_roundtrip_continues_average():
    """Regression guard: _SWA.load_state_dict restores (avg, n) so a resumed
    SWA keeps compositing the uniform average correctly."""
    from bittrainer.group_trainer import _SWA

    torch.manual_seed(0)
    models = [nn.Linear(3, 2) for _ in range(4)]

    control = _SWA()
    for m in models:
        control.update(m)

    partial = _SWA()
    partial.update(models[0])
    partial.update(models[1])
    # Serialise partial state, restore into a fresh instance, finish.
    avg, n = partial.state_dict(), partial.n
    resumed = _SWA()
    resumed.load_state_dict({k: v.clone() for k, v in avg.items()}, n)
    resumed.update(models[2])
    resumed.update(models[3])

    assert resumed.n == control.n == 4
    for k in control.state_dict():
        assert torch.allclose(resumed.state_dict()[k], control.state_dict()[k], atol=1e-6)


# ---------------------------------------------------------------------------
# DCW to_dict/from_dict roundtrip
# ---------------------------------------------------------------------------


def test_dcw_to_from_dict_roundtrip_matches_update_behavior():
    """Regression guard: DCW serialises its full mutable state, so a restored
    controller's subsequent update() produces identical multipliers."""
    from bittrainer.dynamic_class_weights import DynamicClassWeightController

    base = torch.ones(3)
    ctrl = DynamicClassWeightController(3, base, patience=1, cooldown=0, decay=0.5, min_delta=0.001)
    # Drive some history: class 1 peaks then declines (should throttle).
    ctrl.update({"0": 0.5, "1": 0.8, "2": 0.5}, {"0": 1.0, "1": 1.0, "2": 1.0})
    ctrl.update({"0": 0.5, "1": 0.6, "2": 0.5}, {"0": 1.0, "1": 1.2, "2": 1.0})

    payload = ctrl.to_dict()
    restored = DynamicClassWeightController.from_dict(payload, torch.ones(3))
    assert restored.multiplier == ctrl.multiplier
    assert restored.adjustments == ctrl.adjustments

    # Both continue identically on the same next epoch.
    nxt_f1 = {"0": 0.5, "1": 0.55, "2": 0.5}
    nxt_loss = {"0": 1.0, "1": 1.4, "2": 1.0}
    w_a = ctrl.update(nxt_f1, nxt_loss)
    w_b = restored.update(nxt_f1, nxt_loss)
    assert torch.allclose(w_a, w_b)
    assert ctrl.multiplier == restored.multiplier


# ---------------------------------------------------------------------------
# Prodigy_adv optimizer roundtrip (KEY RISK)
# ---------------------------------------------------------------------------


def _tiny_train_step(model, opt, x, y):
    opt.zero_grad()
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    return loss


def test_prodigy_optimizer_state_roundtrip_bit_exact():
    """KEY RISK: a Prodigy_adv state_dict → fresh optimizer → load → step
    produces bit-identical params vs an uninterrupted optimizer, and the
    d/k/d_numerator group scalars survive the round-trip."""
    torch.manual_seed(0)
    x = torch.randn(16, 5)
    y = torch.randint(0, 3, (16,))

    def _fresh_model():
        torch.manual_seed(123)
        return nn.Linear(5, 3)

    def _fresh_opt(m):
        return Prodigy_adv(
            m.parameters(), lr=1.0, d_coef=0.9, weight_decay=0.01,
            betas=(0.9, 0.999), kourkoutas_beta=True, k_warmup_steps=5, cautious_wd=True,
        )

    # Control: k+1 uninterrupted steps.
    ctrl_m = _fresh_model()
    ctrl_o = _fresh_opt(ctrl_m)
    k = 4
    for _ in range(k + 1):
        _tiny_train_step(ctrl_m, ctrl_o, x, y)

    # Interrupted: k steps, snapshot (state_dict + hidden aux), fresh optimizer,
    # load, restore aux, one more step.
    int_m = _fresh_model()
    int_o = _fresh_opt(int_m)
    for _ in range(k):
        _tiny_train_step(int_m, int_o, x, y)
    opt_state = int_o.state_dict()
    aux_state = capture_optimizer_aux_state(int_o)

    resumed_o = _fresh_opt(int_m)
    resumed_o.load_state_dict(opt_state)
    prime_optimizer_after_resume(resumed_o)  # Prodigy d_numerator beta3 re-seed
    restore_optimizer_aux_state(resumed_o, aux_state, _DEVICE)  # Kourkoutas accumulators
    # group scalars survived
    assert resumed_o.param_groups[0]["k"] == int_o.param_groups[0]["k"]
    assert resumed_o.param_groups[0]["d"] == int_o.param_groups[0]["d"]
    assert resumed_o.param_groups[0]["d_numerator"] == int_o.param_groups[0]["d_numerator"]

    _tiny_train_step(int_m, resumed_o, x, y)

    # Bit-exact: interrupted+resumed final params equal the uninterrupted control.
    for a, b in zip(ctrl_m.parameters(), int_m.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Fingerprint + fixed sampler + sanitise + exception backup
# ---------------------------------------------------------------------------


def test_fingerprint_mismatch_detected():
    a = make_fingerprint(
        class_names=["x", "y"], num_classes=2, max_epochs=4,
        multi_label=False, ordinal=False, best_model_name="best.pt", model_size="atto",
    )
    same = dict(a)
    changed = make_fingerprint(
        class_names=["x", "z"], num_classes=2, max_epochs=4,
        multi_label=False, ordinal=False, best_model_name="best.pt", model_size="atto",
    )
    assert fingerprint_matches(a, same)
    assert not fingerprint_matches(a, changed)
    assert not fingerprint_matches(None, a)


def test_fixed_batch_sampler_replays_exact_order():
    sched = [[0, 1], [2, 3], [4]]
    fbs = _FixedBatchSampler(sched)
    assert len(fbs) == 3
    assert list(fbs) == sched
    # slicing for resume yields the tail
    tail = _FixedBatchSampler(sched[1:])
    assert list(tail) == [[2, 3], [4]]


def test_sanitize_for_backup_converts_numpy():
    out = sanitize_for_backup({"f": np.float64(1.5), "a": np.array([1, 2]), "n": [np.int64(3)]})
    assert out == {"f": 1.5, "a": [1, 2], "n": [3]}
    assert isinstance(out["f"], float)


def test_backup_on_exception_saves_and_reraises(tmp_path):
    """Regression guard: an exception inside the wrapped block writes a
    reason='exception' backup and the ORIGINAL exception still propagates."""
    mgr = TrainingStateManager(tmp_path / "b")

    class Boom(RuntimeError):
        pass

    def _collect():
        return {"global_step": 11, "epoch": 2, "note": "snapshot"}

    events = []
    try:
        with backup_on_exception(_collect, mgr, cb=events.append):
            raise Boom("kaboom")
    except Boom:
        pass
    else:  # pragma: no cover
        raise AssertionError("original exception was swallowed")

    loaded = mgr.load_latest()
    assert loaded["reason"] == "exception"
    assert loaded["global_step"] == 11
    assert events and events[0]["type"] == "backup_complete"
    assert events[0]["reason"] == "exception"


def test_backup_failure_does_not_mask_original_exception(tmp_path):
    """A failing backup must never replace the real error."""
    class BadManager:
        def save(self, *_a, **_k):
            raise OSError("disk full")

    def _collect():
        return {"global_step": 1}

    with_pytest_raises = False
    try:
        with backup_on_exception(_collect, BadManager()):
            raise ValueError("real error")
    except ValueError as e:
        with_pytest_raises = str(e) == "real error"
    assert with_pytest_raises
