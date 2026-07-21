"""head-only training forwards effective class counts to finalisation.

Without effective counts, _compare_promote_finalize computes a zero-delta prior
(natural == effective) and silently ships no correction. The head-only path must
forward train_ds.get_effective_class_counts() the same way run_group_training
does (ISSUE-0490 A, audit finding 3). Unit-level via source contract — no
training run, no GPU.
"""

from __future__ import annotations

import inspect

import bittrainer.group_trainer as gt
from bittrainer.generic.tasks.head_only_task import HeadOnlyTask


def test_head_only_forwards_effective_class_counts():
    # ISSUE-0542: the head-only body moved onto HeadOnlyTask (run_head_only_training
    # is now a thin wrapper). The effective-count forwarding lives in finalize().
    src = inspect.getsource(HeadOnlyTask.finalize)
    assert "effective_class_counts=self.train_ds.get_effective_class_counts()" in src


def test_finalize_accepts_effective_class_counts_kwarg():
    sig = inspect.signature(gt._compare_promote_finalize)
    assert "effective_class_counts" in sig.parameters
