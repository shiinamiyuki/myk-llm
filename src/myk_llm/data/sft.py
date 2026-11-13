"""Supervised fine-tuning (SFT) data: chat tokenization and packed, loss-masked windows.

On-disk format of a tokenized chat set (prefix `P`, written by `write_chat_dataset`):

    P.tokens.bin     uint32 token ids, conversations concatenated
    P.loss_mask.bin  uint8, 1 on assistant tokens (the only supervised positions)
    P.spans.npy      int64 (num_conversations, 2) [start, end) offsets
    P.meta.json

Windows are drawn uniformly over token positions and span conversation boundaries
(short conversations are packed rather than dropped); document masking keeps packed
conversations from attending to each other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, cast

import numpy as np
from transformers import PreTrainedTokenizerBase

from .config import IGNORE_INDEX, Batch, DataConfig, collate

_ROLE_ALIASES = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "bot": "assistant", "system": "system"}


def normalize_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    """Accept {'messages': [{'role','content'}]} and ShareGPT {'conversations': [{'from','value'}]}."""
    messages = example.get("messages", example.get("conversations"))
    if not isinstance(messages, list) or not messages:
        raise ValueError("chat examples need a non-empty 'messages' or 'conversations' list")
    normalized = []
    for message in messages:
        role, content = message.get("role", message.get("from")), message.get("content", message.get("value"))
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("chat messages need string role/content (or from/value)")
        normalized.append({"role": _ROLE_ALIASES.get(role.lower(), role), "content": content})
    return normalized


def encode_chat(tokenizer: PreTrainedTokenizerBase, messages: list[dict[str, str]]) -> tuple[list[int], list[int]]:
    """Token ids + assistant loss mask. The chat template must wrap assistant content in
    `{% generation %}` (see the tokenizer's chat_template.jinja), which makes the mask
    exact: assistant content and its end-of-turn token, nothing else."""
    encoding = cast(
        dict[str, list],
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_assistant_tokens_mask=True
        ),
    )
    ids = [int(t) for t in encoding["input_ids"]]
    mask = encoding.get("assistant_masks")
    if not mask or len(mask) != len(ids):
        raise ValueError("the chat template did not produce an assistant mask ({% generation %} tags missing?)")
    return ids, [int(m) for m in mask]


def write_chat_dataset(tokenizer: PreTrainedTokenizerBase, examples: Iterable[dict[str, Any]], prefix: str | Path, tokenizer_path: str) -> dict[str, Any]:
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    tokens: list[int] = []
    loss_mask: list[int] = []
    spans: list[tuple[int, int]] = []
    skipped = 0
    for example in examples:
        ids, mask = encode_chat(tokenizer, normalize_messages(example))
        if len(ids) < 2 or not any(mask):
            skipped += 1
            continue
        spans.append((len(tokens), len(tokens) + len(ids)))
        tokens.extend(ids)
        loss_mask.extend(mask)
    np.asarray(tokens, dtype=np.uint32).tofile(f"{prefix}.tokens.bin")
    np.asarray(loss_mask, dtype=np.uint8).tofile(f"{prefix}.loss_mask.bin")
    np.save(f"{prefix}.spans.npy", np.asarray(spans, dtype=np.int64))
    meta = {
        "mode": "chat", "tokenizer": tokenizer_path, "num_tokens": len(tokens), "num_spans": len(spans),
        "token_dtype": "uint32", "has_loss_mask": True, "skipped": skipped,
    }
    Path(f"{prefix}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class ChatDataset:
    def __init__(self, prefix: str | Path) -> None:
        meta = json.loads(Path(f"{prefix}.meta.json").read_text(encoding="utf-8"))
        if not meta.get("has_loss_mask"):
            raise ValueError(f"{prefix} has no loss mask; tokenize it as chat data")
        self.num_tokens = int(meta["num_tokens"])
        self.tokens = np.memmap(f"{prefix}.tokens.bin", dtype=np.uint32, mode="r", shape=(self.num_tokens,))
        self.loss_mask = np.memmap(f"{prefix}.loss_mask.bin", dtype=np.uint8, mode="r", shape=(self.num_tokens,))
        self.doc_ends = np.load(f"{prefix}.spans.npy")[:, 1]

    def window(self, start: int, seq_len: int, document_masking: bool) -> dict[str, np.ndarray]:
        chunk = np.asarray(self.tokens[start : start + seq_len + 1], dtype=np.int64)
        mask = np.asarray(self.loss_mask[start + 1 : start + seq_len + 1], dtype=bool)
        row = {"input_ids": chunk[:-1], "targets": np.where(mask, chunk[1:], IGNORE_INDEX)}
        if document_masking:
            lo, hi = start, start + seq_len
            cuts = self.doc_ends[np.searchsorted(self.doc_ends, lo, side="right") : np.searchsorted(self.doc_ends, hi, side="left")]
            row["seqlens"] = np.diff(np.concatenate(([lo], cuts, [hi]))).astype(np.int32)
        return row


class ChatLoader:
    """Deterministic, resumable, rank-sliced SFT batches. Micro-step s draws its window
    starts from a generator seeded with (seed, s), so the state is just `s`."""

    def __init__(self, config: DataConfig, rank: int = 0, world_size: int = 1) -> None:
        if len(config.sources) != 1 or not config.sources[0].path:
            raise ValueError("sft data takes exactly one source with a tokenized `path` prefix")
        self.config, self.rank, self.world_size = config, rank, world_size
        self.dataset = ChatDataset(config.sources[0].path)
        if self.dataset.num_tokens < config.seq_len + 1:
            raise ValueError("chat dataset is shorter than one window")
        self.step = 0

    def batch_at(self, step: int) -> Batch:
        per_rank = self.config.micro_batch_size
        rng = np.random.default_rng([self.config.seed, step])
        starts = rng.integers(0, self.dataset.num_tokens - self.config.seq_len, size=per_rank * self.world_size)
        mine = starts[self.rank * per_rank : (self.rank + 1) * per_rank]
        return collate([self.dataset.window(int(s), self.config.seq_len, self.config.document_masking) for s in mine])

    def __iter__(self) -> Iterator[Batch]:
        while True:
            batch = self.batch_at(self.step)
            self.step += 1
            yield batch

    def state_dict(self) -> dict[str, Any]:
        return {"kind": "sft", "step": self.step}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step = int(state["step"])

    def close(self) -> None:
        pass
