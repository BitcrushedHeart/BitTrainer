"""Trainer-agnostic building blocks (Bitcrush ISSUE-0542).

Shared factories and evaluation helpers extracted from the per-trainer modules
so the group / binary / multihead / dual-branch trainers build their optimizer
and score their validation logits through one implementation each.
"""
