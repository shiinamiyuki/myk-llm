"""Input embeddings: the token table plus optional hashed n-gram tables."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig, NgramEmbeddingConfig

# Odd 61-bit multipliers for the rolling n-gram hash (int64 arithmetic wraps on overflow,
# which is fine for hashing).
_HASH_MULTIPLIERS = (0x1F3D5B79A2C4E681, 0x0DDC0FFEEBADF00D, 0x1B873593CC9E2D51, 0x0000000100000001B3)


def ngram_bucket_ids(tokens: torch.Tensor, order: int, context: int, num_buckets: int) -> torch.Tensor:
    """Bucket id of the `order` tokens ending at each position.

    `tokens` is (batch, context + seq_len): `context` tokens of left context (real
    preceding tokens, or a sentinel id where there are none) followed by the sequence.
    Returns (batch, seq_len) int64 ids in [0, num_buckets).
    """
    seq_len = tokens.size(1) - context
    hashed = torch.full_like(tokens[:, context:], order)
    for offset in range(order):
        window = tokens[:, context - offset : context - offset + seq_len]
        hashed = hashed * _HASH_MULTIPLIERS[offset % len(_HASH_MULTIPLIERS)] + window + 1
        hashed = hashed ^ (hashed >> 29)
    return torch.remainder(hashed, num_buckets)


class NgramEmbedding(nn.Module):
    def __init__(self, config: NgramEmbeddingConfig, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.orders = list(config.orders)
        self.num_buckets = config.num_buckets
        self.sentinel = vocab_size  # stands in for "no token" left of the sequence start
        self.tables = nn.ModuleList(nn.Embedding(config.num_buckets, config.dim) for _ in self.orders)
        self.layers = list(config.layers)
        needs_proj = config.dim != hidden_size or self.layers != [0]
        # projs[k] feeds block layers[k]; keyed by position so parameter names are stable.
        self.projs = nn.ModuleList(
            nn.Linear(config.dim, hidden_size, bias=False) if needs_proj else nn.Identity() for _ in self.layers
        )

    @property
    def context(self) -> int:
        return max(self.orders) - 1

    def forward(self, input_ids: torch.Tensor, prefix_ids: torch.Tensor | None) -> dict[int, torch.Tensor]:
        """{block index: (batch, seq_len, hidden) update to add to that block's input}."""
        features = self.lookup(input_ids, prefix_ids)
        return {layer: proj(features) for layer, proj in zip(self.layers, self.projs)}

    def lookup(self, input_ids: torch.Tensor, prefix_ids: torch.Tensor | None) -> torch.Tensor:
        """Summed table rows, (batch, seq_len, dim)."""
        batch, context = input_ids.size(0), self.context
        left = torch.full((batch, context), self.sentinel, dtype=input_ids.dtype, device=input_ids.device)
        if prefix_ids is not None and prefix_ids.size(1) > 0:
            tail = prefix_ids[:, -context:]
            left[:, context - tail.size(1) :] = tail
        tokens = torch.cat([left, input_ids], dim=1)
        return sum(
            table(ngram_bucket_ids(tokens, order, context, self.num_buckets))
            for order, table in zip(self.orders, self.tables)
        )


class Embeddings(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.token = nn.Embedding(config.vocab_size, config.hidden_size)
        self.ngram = (
            NgramEmbedding(config.ngram, config.vocab_size, config.hidden_size) if config.ngram.orders else None
        )

    @property
    def context(self) -> int:
        """How many preceding tokens the embedding looks at (needed during decoding)."""
        return self.ngram.context if self.ngram is not None else 0

    def forward(
        self, input_ids: torch.Tensor, prefix_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Token embeddings, plus the n-gram updates keyed by the block they feed."""
        updates = self.ngram(input_ids, prefix_ids) if self.ngram is not None else {}
        return self.token(input_ids), updates
