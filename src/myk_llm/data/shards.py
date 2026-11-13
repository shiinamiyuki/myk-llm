"""Pre-tokenized token shards: the default, reproducible data path for real runs.

On-disk layout of one source directory (written by `ShardWriter`):

    meta.json              {"tokenizer", "eos_id", "num_tokens", "shards": [{"file", "num_tokens", "num_docs"}]}
    shard_00000.bin        uint32 token ids; every document is followed by the EOS id
    shard_00000.docs.npy   int64 exclusive end offset of each document within the shard

Reading (`ShardLoader`) is fully deterministic: each source's fixed-length windows are
permuted by (seed, epoch) and the sources are interleaved by a token-balanced rule. All
ranks walk the same global window sequence and each takes its own slice, so the only
state to checkpoint is a few counters, and resuming on a different number of GPUs keeps
the same global order.
"""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .config import IGNORE_INDEX, Batch, DataConfig, SourceConfig, collate

TOKEN_DTYPE = np.uint32


class ShardWriter:
    """Appends documents to fixed-size token shards."""

    def __init__(
        self, directory: str | Path, eos_id: int, tokenizer: str, shard_tokens: int = 1 << 28, prefix: str = "shard"
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / "meta.json").exists():
            raise FileExistsError(f"{self.directory} already holds a shard set")
        self.eos_id, self.tokenizer, self.shard_tokens, self.prefix = eos_id, tokenizer, shard_tokens, prefix
        self.shards: list[dict[str, Any]] = []
        self._tokens: list[np.ndarray] = []
        self._doc_ends: list[int] = []
        self._size = 0

    @property
    def num_tokens(self) -> int:
        return sum(s["num_tokens"] for s in self.shards) + self._size

    def add(self, ids: list[int]) -> None:
        doc = np.asarray([*ids, self.eos_id], dtype=TOKEN_DTYPE)
        self._tokens.append(doc)
        self._size += len(doc)
        self._doc_ends.append(self._size)
        if self._size >= self.shard_tokens:
            self._flush()

    def _flush(self) -> None:
        if not self._size:
            return
        name = f"{self.prefix}_{len(self.shards):05d}"
        np.concatenate(self._tokens).tofile(self.directory / f"{name}.bin")
        np.save(self.directory / f"{name}.docs.npy", np.asarray(self._doc_ends, dtype=np.int64))
        self.shards.append({"file": f"{name}.bin", "num_tokens": self._size, "num_docs": len(self._doc_ends)})
        self._tokens, self._doc_ends, self._size = [], [], 0

    def close(self, extra_meta: dict[str, Any] | None = None, write_meta: bool = True) -> list[dict[str, Any]]:
        """Flush the last shard; write meta.json (or return the shard list so several
        writers sharing a directory can be merged with `write_meta`)."""
        self._flush()
        if write_meta:
            write_meta_file(self.directory, self.shards, self.eos_id, self.tokenizer, extra_meta)
        return self.shards


def write_meta_file(
    directory: Path, shards: list[dict[str, Any]], eos_id: int, tokenizer: str, extra_meta: dict[str, Any] | None = None
) -> None:
    meta = {
        "tokenizer": tokenizer,
        "eos_id": eos_id,
        "num_tokens": sum(s["num_tokens"] for s in shards),
        "token_dtype": "uint32",
        "shards": shards,
        **(extra_meta or {}),
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


class ShardSource:
    """Random access to one shard set, as fixed windows of seq_len + 1 tokens."""

    def __init__(self, config: SourceConfig, seq_len: int) -> None:
        self.config = config
        self.directory = Path(config.path)
        self.meta = json.loads((self.directory / "meta.json").read_text(encoding="utf-8"))
        self.seq_len = seq_len
        self._tokens: dict[int, np.ndarray] = {}
        self._doc_ends: dict[int, np.ndarray] = {}
        shard_ids, starts = [], []
        for index, shard in enumerate(self.meta["shards"]):
            count = max(0, (shard["num_tokens"] - 1) // seq_len)  # windows need seq_len + 1 tokens
            shard_ids.append(np.full(count, index, dtype=np.int32))
            starts.append(np.arange(count, dtype=np.int64) * seq_len)
        self.window_shard = np.concatenate(shard_ids) if shard_ids else np.zeros(0, np.int32)
        self.window_start = np.concatenate(starts) if starts else np.zeros(0, np.int64)
        if len(self.window_shard) == 0:
            raise ValueError(f"{self.directory}: no complete window of {seq_len + 1} tokens")

    @property
    def eos_id(self) -> int:
        return int(self.meta["eos_id"])

    def __len__(self) -> int:
        return len(self.window_shard)

    def _shard(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if index not in self._tokens:
            shard = self.meta["shards"][index]
            self._tokens[index] = np.memmap(
                self.directory / shard["file"], dtype=TOKEN_DTYPE, mode="r", shape=(shard["num_tokens"],)
            )
            self._doc_ends[index] = np.load(self.directory / shard["file"].replace(".bin", ".docs.npy"))
        return self._tokens[index], self._doc_ends[index]

    def window(self, window_index: int, document_masking: bool) -> dict[str, np.ndarray]:
        tokens, doc_ends = self._shard(int(self.window_shard[window_index]))
        start = int(self.window_start[window_index])
        chunk = np.asarray(tokens[start : start + self.seq_len + 1], dtype=np.int64)
        row = {"input_ids": chunk[:-1], "targets": chunk[1:].copy()}
        if document_masking:
            lo, hi = start, start + self.seq_len
            # Document ends strictly inside the window are cut points; the EOS belongs to
            # the document it ends, and its target (the next document's first token) is
            # masked because it crosses the seam.
            cuts = doc_ends[np.searchsorted(doc_ends, lo, side="right") : np.searchsorted(doc_ends, hi, side="left")]
            row["targets"][cuts - lo - 1] = IGNORE_INDEX
            row["seqlens"] = np.diff(np.concatenate(([lo], cuts, [hi]))).astype(np.int32)
        return row


@dataclass
class _SourceCursor:
    epoch: int = 0
    position: int = 0
    consumed: int = 0  # windows taken, for the token-balanced interleave


class ShardLoader:
    """Deterministic, resumable, rank-sliced batches over one or more shard sets."""

    def __init__(self, config: DataConfig, rank: int = 0, world_size: int = 1, prefetch: int = 4) -> None:
        if not config.sources:
            raise ValueError("data.sources is empty")
        self.config, self.rank, self.world_size = config, rank, world_size
        self.sources = [ShardSource(source, config.seq_len) for source in config.sources]
        self.weights = np.asarray([s.weight for s in config.sources], dtype=np.float64)
        if np.any(self.weights <= 0):
            raise ValueError("source weights must be positive")
        self.cursors = [_SourceCursor() for _ in self.sources]
        self._orders: dict[tuple[int, int], np.ndarray] = {}
        self.prefetch = prefetch
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- deterministic window sequence -------------------------------------------------
    def _order(self, source_index: int, epoch: int) -> np.ndarray:
        key = (source_index, epoch)
        if key not in self._orders:
            if len(self._orders) > 2 * len(self.sources):
                self._orders.clear()
            rng = np.random.default_rng([self.config.seed, source_index, epoch])
            self._orders[key] = rng.permutation(len(self.sources[source_index]))
        return self._orders[key]

    def _next_window(self) -> tuple[int, int]:
        pick = int(np.argmin([c.consumed for c in self.cursors] / self.weights))
        cursor = self.cursors[pick]
        order = self._order(pick, cursor.epoch)
        window = int(order[cursor.position])
        cursor.position += 1
        cursor.consumed += 1
        if cursor.position == len(order):
            cursor.epoch, cursor.position = cursor.epoch + 1, 0
        return pick, window

    def next_batch(self) -> Batch:
        """The next global batch's slice for this rank (advances the shared state)."""
        per_rank = self.config.micro_batch_size
        picks = [self._next_window() for _ in range(per_rank * self.world_size)]
        mine = picks[self.rank * per_rank : (self.rank + 1) * per_rank]
        return collate([self.sources[s].window(w, self.config.document_masking) for s, w in mine])

    # -- iteration with a background prefetch thread ------------------------------------
    def __iter__(self) -> Iterator[Batch]:
        self._start()
        assert self._queue is not None
        while True:
            item = self._queue.get()
            if isinstance(item, BaseException):
                raise item
            batch, state = item
            self._consumed_state = state
            yield batch

    def _start(self) -> None:
        if self._thread is not None:
            return
        self._queue = queue.Queue(maxsize=self.prefetch)
        self._consumed_state = self._snapshot()

        def worker() -> None:
            try:
                while not self._stop.is_set():
                    batch = self.next_batch()
                    batch.input_ids, batch.targets = batch.input_ids.pin_memory(), batch.targets.pin_memory()
                    self._queue.put((batch, self._snapshot()))  # type: ignore[union-attr]
            except BaseException as error:  # surface loader errors in the training loop
                self._queue.put(error)  # type: ignore[union-attr]

        self._thread = threading.Thread(target=worker, daemon=True, name="shard-loader")
        self._thread.start()

    def _snapshot(self) -> list[dict[str, int]]:
        return [vars(c).copy() for c in self.cursors]

    def state_dict(self) -> dict[str, Any]:
        """State after the last batch handed to the training loop (not the prefetched ones)."""
        cursors = self._consumed_state if self._thread is not None else self._snapshot()
        return {"kind": "shards", "cursors": cursors}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._thread is not None:
            raise RuntimeError("load_state_dict must be called before iterating")
        self.cursors = [_SourceCursor(**c) for c in state["cursors"]]

    def close(self) -> None:
        self._stop.set()
