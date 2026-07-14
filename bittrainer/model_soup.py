"""Greedy model soup (Wortsman et al., 2022) for group training.

Averages the WEIGHTS of the strongest epochs into a single deployable model,
greedily accepting an epoch into the soup only when it does not lower the
validation selection score. By construction the soup's val score is >= the best
single epoch, so adopting it (only when it strictly wins) can only help on val —
a zero-inference-cost alternative to logit ensembling (one model, one forward
pass, one checkpoint). ConvNeXt V2 is LayerNorm-only, so there are no BatchNorm
running-stat pitfalls; integer buffers are carried through unaveraged.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def average_state_dicts(states: list[dict]) -> dict:
    """Element-wise mean of compatible state_dicts.

    Floating tensors are averaged in float32 and cast back to their original
    dtype; non-float tensors (e.g. counters) are taken from the first dict.
    """
    if not states:
        raise ValueError("average_state_dicts requires at least one state dict")
    if len(states) == 1:
        return {k: v.clone() for k, v in states[0].items()}
    out: dict = {}
    n = len(states)
    for k, ref in states[0].items():
        if torch.is_floating_point(ref):
            acc = ref.detach().float().clone()
            for s in states[1:]:
                acc += s[k].detach().float()
            out[k] = (acc / n).to(ref.dtype)
        else:
            out[k] = ref.clone()
    return out


def greedy_soup(
    candidates: list[tuple[float, dict]],
    eval_fn: Callable[[dict], float],
) -> tuple[dict, float, list[int]]:
    """Build a greedy weight soup from ``(val_score, state_dict)`` candidates.

    Sorts candidates by their recorded val score (desc), starts the soup from the
    best one, and adds each remaining candidate only if the freshly-averaged soup
    scores no worse than the current soup under ``eval_fn`` (higher = better).
    Returns ``(soup_state, soup_score, accepted_original_indices)``. ``soup_score``
    is ``eval_fn`` of the returned soup and is >= ``eval_fn`` of the single best
    candidate by construction.
    """
    if not candidates:
        raise ValueError("greedy_soup requires at least one candidate")
    order = sorted(range(len(candidates)), key=lambda i: candidates[i][0], reverse=True)
    soup_members = [candidates[order[0]][1]]
    soup_score = eval_fn(average_state_dicts(soup_members))
    accepted = [order[0]]
    for i in order[1:]:
        trial = average_state_dicts(soup_members + [candidates[i][1]])
        trial_score = eval_fn(trial)
        if trial_score >= soup_score:
            soup_members.append(candidates[i][1])
            soup_score = trial_score
            accepted.append(i)
    return average_state_dicts(soup_members), soup_score, accepted
