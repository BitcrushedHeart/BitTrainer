"""Sharpness-Aware Minimisation wrapper for any optimizer."""

from __future__ import annotations

import torch


class SAM:
    """Wraps an existing optimizer to seek flatter loss minima.

    SAM perturbs weights toward the steepest ascent direction, then
    computes the update gradient at that perturbed point.  The effect
    is that the optimiser converges to flatter regions of the loss
    landscape, which generalise better — particularly on small datasets.

    Usage per training step::

        sam.zero_grad()
        loss.backward()
        sam.first_step()          # perturb weights toward gradient

        sam.zero_grad()
        loss2.backward()          # recompute gradient at perturbed point
        sam.second_step()         # restore weights and apply update
    """

    def __init__(self, optimizer: torch.optim.Optimizer, rho: float = 0.05):
        self.optimizer = optimizer
        self.rho = rho
        self._e_w: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def first_step(self) -> None:
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self._e_w[id(p)] = e_w

    @torch.no_grad()
    def second_step(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                key = id(p)
                if key in self._e_w:
                    p.sub_(self._e_w[key])
        self.optimizer.step()
        self._e_w.clear()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    @property
    def param_groups(self) -> list:
        return self.optimizer.param_groups

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.optimizer.param_groups[0]["params"][0].device
        norms = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norms.append(p.grad.norm(p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)
