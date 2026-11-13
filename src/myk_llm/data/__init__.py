"""Data loaders. Every loader yields `Batch`es for this rank and supports
`state_dict()` / `load_state_dict()` for exact resume."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import IGNORE_INDEX, Batch, DataConfig, SourceConfig, collate
from .filters import DocFilter


def resolve_eos_id(config: DataConfig) -> int:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(config.tokenizer) / "tokenizer.json"))
    eos_id = tokenizer.token_to_id(config.eos_token)
    if eos_id is None:
        raise ValueError(f"eos_token {config.eos_token!r} is not in the tokenizer vocab")
    return int(eos_id)


def build_loader(config: DataConfig, rank: int = 0, world_size: int = 1) -> Any:
    if config.kind == "shards":
        from .shards import ShardLoader

        return ShardLoader(config, rank, world_size)
    if config.kind == "stream":
        from .streaming import StreamLoader

        return StreamLoader(config, resolve_eos_id(config), rank, world_size)
    if config.kind == "sft":
        from .sft import ChatLoader

        return ChatLoader(config, rank, world_size)
    raise ValueError(f"unknown data kind {config.kind!r}")


def build_fixed_batches(config: DataConfig, num_batches: int, seed: int) -> list[Batch]:
    """A fixed validation set: `num_batches` global batches of `micro_batch_size` rows,
    identical on every rank (the evaluator splits the batches across ranks)."""
    if config.kind == "stream":
        from .streaming import take_windows

        rows = take_windows(config, resolve_eos_id(config), num_batches * config.micro_batch_size, seed)
        size = config.micro_batch_size
        return [collate(rows[i * size : (i + 1) * size]) for i in range(num_batches)]
    from dataclasses import replace

    loader = build_loader(replace(config, seed=seed), rank=0, world_size=1)
    if config.kind == "shards":
        return [loader.next_batch() for _ in range(num_batches)]
    return [loader.batch_at(i) for i in range(num_batches)]


__all__ = ["IGNORE_INDEX", "Batch", "DataConfig", "DocFilter", "SourceConfig", "build_fixed_batches", "build_loader"]
