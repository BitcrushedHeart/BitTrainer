"""Tiny SigLIP text encoders for tests (Bitcrush Optimise P8).

``install_tiny_text_model`` monkeypatches the ``bittrainer.siglip_text
._build_text_model`` seam so every text tower is a few-kilobyte hashed
bag-of-words encoder with the Hugging Face surface the tower relies on: a
tokenizer callable returning ``input_ids`` / ``attention_mask`` and an encoder
whose output carries ``pooler_output``. No test downloads SigLIP2 weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

_VOCAB = 1024


class TinyTokenizer:
    """Whitespace tokens hashed into a fixed vocabulary; pads to ``max_length``."""

    def __call__(
        self,
        captions: list[str],
        *,
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int = 64,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        ids = torch.zeros(len(captions), max_length, dtype=torch.long)
        mask = torch.zeros(len(captions), max_length, dtype=torch.long)
        for row, caption in enumerate(captions):
            words = caption.lower().split()[:max_length]
            for col, word in enumerate(words):
                ids[row, col] = 1 + (sum(ord(c) * (31**k) for k, c in enumerate(word)) % (_VOCAB - 1))
                mask[row, col] = 1
        return {"input_ids": ids, "attention_mask": mask}


class TinyTextEncoder(nn.Module):
    """Mean of token embeddings -> linear head, exposed as ``pooler_output``."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        torch.manual_seed(4321)
        self.embeddings = nn.Embedding(_VOCAB, embed_dim, padding_idx=0)
        self.head = nn.Linear(embed_dim, embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        emb = self.embeddings(input_ids)
        mask = attention_mask.unsqueeze(-1).to(emb.dtype)
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return SimpleNamespace(pooler_output=self.head(pooled), last_hidden_state=emb)


def install_tiny_text_model(monkeypatch, *, embed_dim: int, unavailable=()):
    """Route every text-model build to a tiny encoder; record the ids asked for.

    ``unavailable`` names model ids whose build raises
    :class:`bittrainer.siglip_text.TextTowerUnavailable`, which is how the
    fallback path is exercised without a network.
    """
    import bittrainer.siglip_text as st

    calls: list[str] = []
    blocked = set(unavailable)

    def _build(model_id: str):
        calls.append(model_id)
        if model_id in blocked:
            raise st.TextTowerUnavailable(f"{model_id} is not available in this test")
        return TinyTextEncoder(embed_dim), TinyTokenizer()

    monkeypatch.setattr(st, "_build_text_model", _build)
    return calls
