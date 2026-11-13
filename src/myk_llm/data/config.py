"""Data configuration and the common batch type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from .filters import DocFilter

IGNORE_INDEX = -100


@dataclass
class SourceConfig:
    """One corpus. `path` is a pre-tokenized directory (`kind: shards`) or a tokenized
    chat prefix (`kind: sft`); the `hf_*` fields describe a HuggingFace dataset, used by
    `kind: stream` and by the pre-tokenization script."""

    name: str
    weight: float = 1.0
    path: str = ""
    hf_path: str = ""
    hf_config: str | None = None
    hf_data_dir: str | None = None
    hf_split: str = "train"
    text_key: str = "text"
    script: Literal["latin", "cjk"] | None = None  # expected writing system, for DocFilter
    curated: bool = False  # already quality-filtered upstream: skip DocFilter heuristics


@dataclass
class DataConfig:
    kind: Literal["shards", "stream", "sft"] = "shards"
    tokenizer: str = "_download_models/Qwen3.5-0.8B-pruned64k"
    seq_len: int = 2048
    micro_batch_size: int = 16  # sequences per GPU per micro-step
    sources: list[SourceConfig] = field(default_factory=list)
    # Block-diagonal attention over packed documents; also masks the targets that
    # cross a document seam.
    document_masking: bool = True
    seed: int = 0
    eos_token: str = "<|endoftext|>"  # document separator (pretraining)
    # Streaming only:
    shuffle_buffer: int = 10_000
    num_workers: int = 4
    doc_filter: DocFilter = field(default_factory=DocFilter)


@dataclass
class Batch:
    input_ids: torch.Tensor  # (batch, seq_len) int64
    targets: torch.Tensor  # (batch, seq_len) int64, IGNORE_INDEX where unsupervised
    seqlens: list[torch.Tensor] | None = None  # per-row document lengths (document masking)

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            self.input_ids.to(device, non_blocking=True),
            self.targets.to(device, non_blocking=True),
            self.seqlens,
        )

    def pin_memory(self) -> "Batch":
        return Batch(self.input_ids.pin_memory(), self.targets.pin_memory(), self.seqlens)

    @property
    def num_targets(self) -> int:
        return int((self.targets != IGNORE_INDEX).sum())


def collate(rows: list[dict]) -> Batch:
    import numpy as np

    input_ids = torch.as_tensor(np.stack([r["input_ids"] for r in rows]), dtype=torch.long)
    targets = torch.as_tensor(np.stack([r["targets"] for r in rows]), dtype=torch.long)
    seqlens = [torch.as_tensor(r["seqlens"], dtype=torch.int32) for r in rows] if "seqlens" in rows[0] else None
    return Batch(input_ids, targets, seqlens)
