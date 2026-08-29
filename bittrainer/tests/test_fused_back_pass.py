"""Fused back pass for the group full fine-tune (Bitcrush ISSUE-0862).

``Prodigy_adv.supports_fused_back_pass`` is True: ``step_parameter(p, group, i)``
may run the moment a parameter's gradient has finished accumulating, i.e. from a
``register_post_accumulate_grad_hook`` fired inside ``backward()``. That hides
the per-parameter Python/kernel-launch cost of the optimizer behind the rest of
the backward instead of paying it as a separate serial phase (29.6 % of the
trainer main thread on a live pico fine-tune profile).

These tests pin the contract of ``group_trainer._train_one_epoch``'s fused mode:

* it produces the same parameters as ``backward(); step(); zero_grad()`` when
  clipping never engages,
* it clips PER TENSOR (the only thing a per-parameter hook can see),
* gradients are freed by the hooks (never by ``optimizer.zero_grad()``),
* it falls back to the standard path under gradient accumulation or when
  ``fused_back_pass=False``,
* and it leaves no hook handles behind, so a second epoch steps exactly once
  per batch.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn

import bittrainer.group_trainer as gt
from bittrainer.generic.optimizer import make_optimizer

_DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


class _TinyConv(nn.Module):
    """Smallest model that still has 2-D and 1-D params (so ``wd_exclusions``
    builds two real param groups and the ``{id(p): (group, i)}`` map has to span
    them) and a real conv backward."""

    def __init__(self, num_classes: int, scale: float = 1.0) -> None:
        super().__init__()
        torch.manual_seed(1)
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.head = nn.Linear(4, num_classes)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x).mean(dim=(2, 3))) * self.scale


def _cfg(**kw) -> gt.GroupTrainConfig:
    base = dict(
        group_folder=".",
        num_classes=3,
        class_names=["a", "b", "c"],
        device="cpu",
        dtype="float32",
        channels_last=False,
        use_compile=False,
        randaugment_n=0,
        randaugment_m=0,
        random_erasing_p=0.0,
        aug_noise_p=0.0,
        aug_blur_p=0.0,
        aug_jpeg_p=0.0,
        use_mixup=False,
    )
    base.update(kw)
    return gt.GroupTrainConfig(**base)


def _batches(n: int = 3, batch: int = 2, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randint(0, 255, (batch, 3, 16, 16), dtype=torch.uint8, generator=g),
            torch.randint(0, 3, (batch,), generator=g),
        )
        for _ in range(n)
    ]


def _fresh_batches(batches):
    """The loop normalises in place-ish; hand every run its own copy."""
    return [(im.clone(), lb.clone()) for im, lb in batches]


def _optimizer(model: nn.Module):
    return make_optimizer(model, llrd=False, wd_exclusions=True)


def _run_epoch(model, optimizer, config, batches, *, seed: int = 7, **kw):
    # Seed immediately before the loop: the loop's own hflip augmentation draws
    # from the global RNG, so both arms of an equivalence test must start from
    # the identical stream.
    torch.manual_seed(seed)
    return gt._train_one_epoch(
        model,
        _fresh_batches(batches),
        optimizer,
        config,
        _DEVICE,
        torch.float32,
        **kw,
    )


def _params(model: nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


# --------------------------------------------------------------------------- #
# Prodigy preconditions                                                        #
# --------------------------------------------------------------------------- #


def test_prodigy_advertises_fused_back_pass_and_inits_before_first_backward():
    """``step_parameter`` reads ``self.beta1`` / ``self.beta3``, which only
    ``init_step()`` sets. Prodigy's constructor calls it, so the very first
    hook-driven ``step_parameter`` of a run is already primed — the fused loop
    only has to call ``calculate_d(); init_step()`` AFTER each backward."""
    model = _TinyConv(3)
    opt = _optimizer(model)
    assert opt.supports_fused_back_pass is True
    assert hasattr(opt, "step_parameter")
    # init_step() ran in __init__ (no step has been taken yet).
    assert opt.beta1 == pytest.approx(0.9)
    assert opt.beta3 == pytest.approx(math.sqrt(0.999))


def test_prodigy_init_step_runs_again_after_load_state_dict():
    model = _TinyConv(3)
    opt = _optimizer(model)
    other = _optimizer(_TinyConv(3))
    del other.beta1  # prove load_state_dict re-primes rather than relying on __init__
    other.load_state_dict(opt.state_dict())
    assert other.beta1 == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# (a) Numeric equivalence when clipping never engages                          #
# --------------------------------------------------------------------------- #


def _equivalence(clip: float, scale: float = 1.0):
    batches = _batches(3)
    base = _TinyConv(3, scale=scale)
    start = _params(base)

    std_model = copy.deepcopy(base)
    std_opt = _optimizer(std_model)
    _run_epoch(std_model, std_opt, _cfg(clip_grad_norm=clip, fused_back_pass=False), batches)

    fused_model = copy.deepcopy(base)
    fused_opt = _optimizer(fused_model)
    _run_epoch(fused_model, fused_opt, _cfg(clip_grad_norm=clip, fused_back_pass=True), batches)

    return start, std_model, fused_model, std_opt, fused_opt


def test_fused_matches_standard_when_clipping_is_off():
    start, std_model, fused_model, std_opt, fused_opt = _equivalence(clip=0.0)
    for a, b in zip(std_model.parameters(), fused_model.parameters()):
        assert torch.allclose(a, b, rtol=1e-6, atol=1e-9)
    # Non-vacuous: the run actually moved the weights and drove Prodigy's
    # d-adaptation accumulators.
    assert any(not torch.equal(s, p) for s, p in zip(start, std_model.parameters()))
    assert float(std_opt.d_numerator) != 0.0
    assert float(fused_opt.d_numerator) == pytest.approx(float(std_opt.d_numerator), rel=1e-5)


def test_fused_matches_standard_when_global_norm_stays_under_max(monkeypatch):
    """With the shipped default (``clip_grad_norm=1.0``) and gradients whose
    global norm never reaches 1.0, per-tensor and global clipping are the same
    no-op, so the two paths must agree exactly."""
    seen: list[float] = []
    real_clip = gt.clip_gradients

    def _spy(module_or_params, max_norm):
        params = [p for p in module_or_params.parameters() if p.grad is not None]
        seen.append(float(torch.norm(torch.stack([p.grad.norm() for p in params]))))
        return real_clip(module_or_params, max_norm)

    monkeypatch.setattr(gt, "clip_gradients", _spy)
    # scale=0.05 keeps the logit magnitudes (and so the CE gradients) small.
    _start, std_model, fused_model, _so, _fo = _equivalence(clip=1.0, scale=0.05)
    assert seen, "the standard path never clipped"
    assert max(seen) < 1.0, f"global norm reached {max(seen)}; test premise broken"
    for a, b in zip(std_model.parameters(), fused_model.parameters()):
        assert torch.allclose(a, b, rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------- #
# (b) Per-tensor clipping                                                      #
# --------------------------------------------------------------------------- #


def test_fused_clips_each_tensor_to_max_norm():
    """The hook only ever sees one tensor, so fused mode clips per tensor. Each
    tensor's norm is <= the global norm, so this NEVER clips more than global
    clipping does; it clips LESS whenever the global norm exceeds max_norm."""
    max_norm = 1e-3
    model = _TinyConv(3, scale=20.0)
    opt = _optimizer(model)
    norms: list[float] = []
    real = opt.step_parameter

    def _spy(p, group, i=None):
        if p.grad is not None:
            norms.append(float(p.grad.norm()))
        return real(p, group, i)

    opt.step_parameter = _spy
    _run_epoch(model, opt, _cfg(clip_grad_norm=max_norm, fused_back_pass=True), _batches(3))

    assert norms, "step_parameter never ran — fused mode did not engage"
    assert max(norms) <= max_norm * 1.001


def test_standard_path_still_clips_the_global_norm():
    max_norm = 1e-3
    model = _TinyConv(3, scale=20.0)
    opt = _optimizer(model)
    totals: list[float] = []
    real = opt.step

    def _spy(*a, **k):
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        totals.append(float(torch.norm(torch.stack([g.norm() for g in grads]))))
        return real(*a, **k)

    opt.step = _spy
    _run_epoch(model, opt, _cfg(clip_grad_norm=max_norm, fused_back_pass=False), _batches(3))

    assert totals
    assert max(totals) <= max_norm * 1.001


# --------------------------------------------------------------------------- #
# (c) The hooks free the gradients                                             #
# --------------------------------------------------------------------------- #


def test_fused_leaves_no_gradients_after_backward():
    model = _TinyConv(3)
    opt = _optimizer(model)
    snapshots: list[list[bool]] = []
    real = opt.calculate_d

    def _spy():
        # calculate_d runs after backward() has returned, i.e. after every hook
        # has fired; in fused mode every grad must already be released.
        snapshots.append([p.grad is None for p in model.parameters()])
        return real()

    opt.calculate_d = _spy
    _run_epoch(model, opt, _cfg(fused_back_pass=True), _batches(3))

    assert len(snapshots) == 3
    assert all(all(s) for s in snapshots)
    assert all(p.grad is None for p in model.parameters())


def test_fused_never_calls_optimizer_step_or_zero_grad():
    model = _TinyConv(3)
    opt = _optimizer(model)
    calls: list[str] = []
    real_step, real_zero = opt.step, opt.zero_grad

    opt.step = lambda *a, **k: (calls.append("step"), real_step(*a, **k))[1]

    def _zero(*a, **k):
        calls.append("zero_grad")
        return real_zero(*a, **k)

    opt.zero_grad = _zero
    _run_epoch(model, opt, _cfg(fused_back_pass=True), _batches(3))

    assert "step" not in calls
    # Exactly one top-of-epoch zero_grad (clearing anything left by a previous
    # phase); never one per boundary.
    assert calls == ["zero_grad"]


# --------------------------------------------------------------------------- #
# (d)/(e) Fallbacks                                                            #
# --------------------------------------------------------------------------- #


def _step_spy(opt) -> list[int]:
    calls: list[int] = []
    real = opt.step

    def _step(*a, **k):
        calls.append(1)
        return real(*a, **k)

    opt.step = _step
    return calls


def test_grad_accumulation_falls_back_to_the_standard_path():
    model = _TinyConv(3)
    opt = _optimizer(model)
    calls = _step_spy(opt)
    _run_epoch(
        model,
        opt,
        _cfg(fused_back_pass=True, grad_accum_steps=2),
        _batches(4),
    )
    assert len(calls) == 2  # 4 batches / accum 2
    assert not any(getattr(p, "_post_accumulate_grad_hooks", None) for p in model.parameters())


def test_fused_back_pass_false_uses_the_standard_path():
    model = _TinyConv(3)
    opt = _optimizer(model)
    calls = _step_spy(opt)
    _run_epoch(model, opt, _cfg(fused_back_pass=False), _batches(3))
    assert len(calls) == 3
    assert not any(getattr(p, "_post_accumulate_grad_hooks", None) for p in model.parameters())


def test_optimizer_without_fused_support_falls_back():
    model = _TinyConv(3)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    calls = _step_spy(opt)
    _run_epoch(model, opt, _cfg(fused_back_pass=True), _batches(3))
    assert len(calls) == 3


def test_config_default_enables_fused_back_pass():
    assert _cfg().fused_back_pass is True


# --------------------------------------------------------------------------- #
# (f) Hook lifetime                                                            #
# --------------------------------------------------------------------------- #


def test_hooks_are_removed_and_a_second_epoch_steps_once_per_batch():
    model = _TinyConv(3)
    opt = _optimizer(model)
    calls: list[int] = []
    real = opt.step_parameter

    def _spy(p, group, i=None):
        calls.append(1)
        return real(p, group, i)

    opt.step_parameter = _spy
    cfg = _cfg(fused_back_pass=True)
    batches = _batches(3)

    _run_epoch(model, opt, cfg, batches)
    assert not any(getattr(p, "_post_accumulate_grad_hooks", None) for p in model.parameters())
    n_params = sum(1 for _ in model.parameters())
    assert len(calls) == 3 * n_params

    _run_epoch(model, opt, cfg, batches)
    assert not any(getattr(p, "_post_accumulate_grad_hooks", None) for p in model.parameters())
    # Exactly one more pass — a leaked handle from epoch 1 would double this.
    assert len(calls) == 6 * n_params


def test_hooks_are_removed_when_the_epoch_raises():
    model = _TinyConv(3)
    opt = _optimizer(model)

    def _boom(*_a, **_k):
        raise RuntimeError("boom")

    opt.calculate_d = _boom
    with pytest.raises(RuntimeError, match="boom"):
        _run_epoch(model, opt, _cfg(fused_back_pass=True), _batches(3))
    assert not any(getattr(p, "_post_accumulate_grad_hooks", None) for p in model.parameters())


# --------------------------------------------------------------------------- #
# Surrounding machinery keeps working                                          #
# --------------------------------------------------------------------------- #


def test_boundary_hook_and_ema_still_fire_once_per_step_in_fused_mode():
    from bittrainer.ema import ModelEMA

    model = _TinyConv(3)
    opt = _optimizer(model)
    ema = ModelEMA(model, decay=0.9)
    updates: list[int] = []
    real_update = ema.update

    def _update(m):
        updates.append(1)
        return real_update(m)

    ema.update = _update
    seen: list[int] = []
    loss, per_class = _run_epoch(
        model,
        opt,
        _cfg(fused_back_pass=True),
        _batches(3),
        ema=ema,
        boundary_hook=lambda n: seen.append(n),
        sample_loss_sink=lambda i, losses: seen.append(-1),
    )
    assert seen == [1, -1, 2, -1, 3, -1]
    assert len(updates) == 3
    assert loss > 0
    assert per_class


def test_boundary_hook_sees_a_coherent_optimizer_state():
    """Backups fire from ``boundary_hook``. In fused mode the step is finished
    (``calculate_d`` + ``init_step`` have run) and no gradient is in flight when
    it fires, so a restored state resumes exactly like the standard path's."""
    model = _TinyConv(3)
    opt = _optimizer(model)
    seen: list[tuple[bool, float, int]] = []

    def _hook(n: int):
        seen.append(
            (
                all(p.grad is None for p in model.parameters()),
                float(opt.d_denom),  # init_step() reset the accumulator
                opt.param_groups[0]["k"],  # calculate_d() advanced the counter
            )
        )

    _run_epoch(model, opt, _cfg(fused_back_pass=True), _batches(3), boundary_hook=_hook)

    assert [s[0] for s in seen] == [True, True, True]
    assert [s[1] for s in seen] == [0.0, 0.0, 0.0]
    assert [s[2] for s in seen] == [1, 2, 3]


def test_optimizer_state_round_trips_after_a_fused_epoch():
    """Resume path: a fused epoch must leave a state_dict another optimizer can
    load (Kourkoutas registers its per-layer EMA into ``optimizer.state``)."""
    model = _TinyConv(3)
    opt = _optimizer(model)
    _run_epoch(model, opt, _cfg(fused_back_pass=True), _batches(3))

    clone = _TinyConv(3)
    clone.load_state_dict(model.state_dict())
    reloaded = _optimizer(clone)
    reloaded.load_state_dict(opt.state_dict())
    assert reloaded.param_groups[0]["k"] == 3
    assert reloaded.param_groups[0]["d"] == pytest.approx(opt.param_groups[0]["d"])
    # A further fused epoch on the restored optimizer still runs.
    _run_epoch(clone, reloaded, _cfg(fused_back_pass=True), _batches(3))
    assert reloaded.param_groups[0]["k"] == 6


def test_fused_honours_start_batch_and_mixup_and_early_outs():
    class _Flag:
        def __init__(self, value: bool) -> None:
            self._value = value

        def is_set(self) -> bool:
            return self._value

    # Mid-epoch resume: absolute batch positions still drive the boundary hook.
    model = _TinyConv(3)
    opt = _optimizer(model)
    seen: list[int] = []
    _run_epoch(
        model,
        opt,
        _cfg(fused_back_pass=True),
        _batches(2),
        start_batch=5,
        boundary_hook=seen.append,
    )
    assert seen == [6, 7]

    # MixUp batches still step (soft targets, no per-class telemetry).
    model = _TinyConv(3)
    opt = _optimizer(model)
    loss, per_class = _run_epoch(
        model,
        opt,
        _cfg(fused_back_pass=True, use_mixup=True, mixup_prob=1.0),
        _batches(3),
        mixup_enabled=True,
    )
    assert loss > 0
    assert per_class == {}
    assert opt.param_groups[0]["k"] == 3

    # stop_now / pause break BEFORE a forward, so no half-stepped parameters.
    model = _TinyConv(3)
    opt = _optimizer(model)
    _run_epoch(
        model,
        opt,
        _cfg(fused_back_pass=True),
        _batches(3),
        stop_now_event=_Flag(True),
    )
    assert opt.param_groups[0]["k"] == 0
    assert not any(getattr(p, "_post_accumulate_grad_hooks", None) for p in model.parameters())

    model = _TinyConv(3)
    opt = _optimizer(model)
    _run_epoch(model, opt, _cfg(fused_back_pass=True), _batches(3), pause_event=_Flag(True))
    assert opt.param_groups[0]["k"] == 0

    # boundary_hook returning "stop" halts after a completed step.
    model = _TinyConv(3)
    opt = _optimizer(model)
    _run_epoch(
        model,
        opt,
        _cfg(fused_back_pass=True),
        _batches(3),
        boundary_hook=lambda _n: "stop",
    )
    assert opt.param_groups[0]["k"] == 1
    assert all(p.grad is None for p in model.parameters())


def test_fused_mode_announces_itself(monkeypatch):
    messages: list[dict] = []
    model = _TinyConv(3)
    opt = _optimizer(model)
    cfg = _cfg(fused_back_pass=True, progress_callback=messages.append)
    _run_epoch(model, opt, cfg, _batches(2))
    _run_epoch(model, opt, cfg, _batches(2))
    # Once per RUN, not once per epoch.
    assert len(messages) == 1
    assert "fused" in messages[0]["status_text"].lower()


def test_fallback_announces_its_reason():
    messages: list[dict] = []
    model = _TinyConv(3)
    opt = _optimizer(model)
    cfg = _cfg(fused_back_pass=True, grad_accum_steps=2, progress_callback=messages.append)
    _run_epoch(model, opt, cfg, _batches(2))
    assert len(messages) == 1
    text = messages[0]["status_text"].lower()
    assert "accum" in text
