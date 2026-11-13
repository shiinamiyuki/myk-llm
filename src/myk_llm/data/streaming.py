"""On-the-fly streaming corpus: HuggingFace streaming datasets, tokenized and packed live.

Handy for quick local experiments (no pre-tokenization step); for long multi-GPU runs
prefer pre-tokenized shards (`myk_llm.data.shards`), which do not depend on network
access and resume trivially.

Sources are interleaved by a token budget, tokenized, and packed into fixed windows;
documents are separated by an end-of-sequence (EOS) token and every window carries the
per-document lengths for document-masked attention.

Design notes:
  - Each rank reads a disjoint part of every source (`split_dataset_by_node`), and each
    DataLoader worker a disjoint part of that (HuggingFace auto-shards over workers), so
    keep num_workers * world_size <= the smallest source's shard count.
  - All mutable iteration state lives on object attributes, so `state_dict()` gives an
    exact snapshot that torchdata's StatefulDataLoader checkpoints per worker; the
    trainer saves one loader state per rank.
  - Each source is infinite: when exhausted it reshuffles with a new seed and loops.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, cast

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from datasets import load_dataset
from tokenizers import Tokenizer

from .config import IGNORE_INDEX, Batch, DataConfig, SourceConfig, collate
from .filters import DocFilter


def open_hf_stream(source: SourceConfig, seed: int, shuffle_buffer: int, rank: int = 0, world_size: int = 1) -> Any:
    ds = load_dataset(
        source.hf_path, name=source.hf_config, data_dir=source.hf_data_dir, split=source.hf_split, streaming=True
    )
    if world_size > 1:
        from datasets.distributed import split_dataset_by_node

        ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return ds


class _TokenizedSource:
    """One streaming source, tokenized in batches; infinite via reshuffle + loop.

    Holds a small queue of tokenized documents so `state_dict` can restore exactly:
    the HuggingFace stream position plus the queued-but-unconsumed documents.
    """

    def __init__(
        self,
        source: SourceConfig,
        tokenizer: Tokenizer,
        seed: int,
        shuffle_buffer: int,
        eos_id: int,
        doc_filter: DocFilter,
        rank: int,
        world_size: int,
        encode_batch_size: int = 64,
    ) -> None:
        self.source, self.tokenizer, self.seed = source, tokenizer, seed
        self.shuffle_buffer, self.eos_id, self.doc_filter = shuffle_buffer, eos_id, doc_filter
        self.rank, self.world_size, self.encode_batch_size = rank, world_size, encode_batch_size
        self.n_kept = self.n_dropped = self.n_dropped_lang = 0
        self.epoch = 0
        self._queue: list[list[int]] = []
        self._ds: Any = None
        self._raw_iter: Iterator[dict[str, Any]] | None = None
        self._pending_hf_state: dict[str, Any] | None = None

    def _new_dataset(self) -> Any:
        return open_hf_stream(self.source, self.seed + self.epoch, self.shuffle_buffer, self.rank, self.world_size)

    def _ensure_iter(self) -> None:
        if self._raw_iter is not None:
            return
        self._ds = self._new_dataset()
        if self._pending_hf_state is not None:
            self._ds.load_state_dict(self._pending_hf_state)
            self._pending_hf_state = None
        self._raw_iter = iter(self._ds)

    def _refill(self) -> None:
        self._ensure_iter()
        texts: list[str] = []
        while len(texts) < self.encode_batch_size:
            try:
                example = next(cast(Iterator[dict[str, Any]], self._raw_iter))
            except StopIteration:
                self.epoch += 1
                self._ds = self._new_dataset()
                self._raw_iter = iter(self._ds)
                continue
            text = example.get(self.source.text_key)
            if not (isinstance(text, str) and text.strip()):
                continue
            if not self.source.curated and self.doc_filter.reject_text(text, self.source.script):
                self.n_dropped_lang += 1
                continue
            texts.append(text)
        for encoding in self.tokenizer.encode_batch(texts, add_special_tokens=False):
            ids = encoding.ids
            if not ids:
                continue
            if self.doc_filter.max_doc_tokens and len(ids) > self.doc_filter.max_doc_tokens:
                ids = ids[: self.doc_filter.max_doc_tokens]
            if not self.source.curated and self.doc_filter.reject(ids):
                self.n_dropped += 1
                continue
            self.n_kept += 1
            self._queue.append([*ids, self.eos_id])

    def next_doc(self) -> list[int]:
        if not self._queue:
            self._refill()
        return self._queue.pop(0)

    def state_dict(self) -> dict[str, Any]:
        self._ensure_iter()
        return {"epoch": self.epoch, "queue": [list(d) for d in self._queue], "hf": self._ds.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch = int(state.get("epoch", 0))
        self._queue = [list(d) for d in state.get("queue", [])]
        self._pending_hf_state = state.get("hf")
        self._raw_iter = None  # rebuilt lazily, applying the pending HF state


class PackedStreamDataset(IterableDataset):
    """Token-balanced interleave of several streaming sources, packed into windows."""

    def __init__(self, config: DataConfig, eos_id: int, rank: int = 0, world_size: int = 1, seed: int | None = None) -> None:
        super().__init__()
        if not config.sources:
            raise ValueError("data.sources is empty")
        self.config, self.eos_id, self.rank, self.world_size = config, eos_id, rank, world_size
        self.seed = config.seed if seed is None else seed
        self.seq_len = config.seq_len
        self._weights = np.asarray([s.weight for s in config.sources], dtype=np.float64)
        self._sources: list[_TokenizedSource] | None = None
        self._cum_tokens = np.zeros(len(config.sources), dtype=np.float64)
        self._residual: list[int] = []
        self._residual_segs: list[int] = []
        self._seg_counter = 0
        self._pending_state: dict[str, Any] | None = None

    def _ensure_setup(self) -> None:
        if self._sources is not None:
            return
        worker = get_worker_info()
        # Decorrelate shuffle buffers per worker; stable across resume (fixed worker count).
        seed = self.seed + 1009 * (worker.id if worker is not None else 0)
        tokenizer = Tokenizer.from_file(str(Path(self.config.tokenizer) / "tokenizer.json"))
        self._sources = [
            _TokenizedSource(
                source, tokenizer, seed, self.config.shuffle_buffer, self.eos_id, self.config.doc_filter,
                self.rank, self.world_size,
            )
            for source in self.config.sources
        ]
        if self._pending_state is not None:
            self._apply_state(self._pending_state)
            self._pending_state = None

    def _produce_window(self) -> dict[str, np.ndarray]:
        sources = cast(list[_TokenizedSource], self._sources)
        while len(self._residual) < self.seq_len + 1:
            pick = int(np.argmin(self._cum_tokens / self._weights))
            doc = sources[pick].next_doc()
            self._cum_tokens[pick] += len(doc)
            self._residual.extend(doc)
            self._residual_segs.extend([self._seg_counter] * len(doc))
            self._seg_counter += 1

        window = self._residual[: self.seq_len + 1]
        segs = np.asarray(self._residual_segs[: self.seq_len + 1], dtype=np.int64)
        self._residual = self._residual[self.seq_len :]
        self._residual_segs = self._residual_segs[self.seq_len :]

        row: dict[str, np.ndarray] = {
            "input_ids": np.asarray(window[:-1], dtype=np.int64),
            "targets": np.asarray(window[1:], dtype=np.int64),
        }
        if self.config.document_masking:
            boundaries = np.flatnonzero(np.diff(segs[:-1])) + 1
            row["seqlens"] = np.diff(np.concatenate(([0], boundaries, [self.seq_len]))).astype(np.int32)
            row["targets"][segs[:-1] != segs[1:]] = IGNORE_INDEX  # targets crossing a seam
        return row

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        self._ensure_setup()
        while True:
            yield self._produce_window()

    def state_dict(self) -> dict[str, Any]:
        if self._sources is None:
            return self._pending_state or {}
        return {
            "cum_tokens": self._cum_tokens.tolist(),
            "residual": list(self._residual),
            "residual_segs": list(self._residual_segs),
            "seg_counter": self._seg_counter,
            "sources": [source.state_dict() for source in self._sources],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._sources is None:
            self._pending_state = state  # applied when the worker builds its sources
        else:
            self._apply_state(state)

    def _apply_state(self, state: dict[str, Any]) -> None:
        if not state:
            return
        self._cum_tokens = np.asarray(state["cum_tokens"], dtype=np.float64)
        self._residual = list(state["residual"])
        self._residual_segs = list(state["residual_segs"])
        self._seg_counter = int(state["seg_counter"])
        for source, source_state in zip(cast(list[_TokenizedSource], self._sources), state["sources"]):
            source.load_state_dict(source_state)


class StreamLoader:
    """Per-rank StatefulDataLoader over `PackedStreamDataset`, yielding `Batch`es."""

    def __init__(self, config: DataConfig, eos_id: int, rank: int = 0, world_size: int = 1) -> None:
        from torchdata.stateful_dataloader import StatefulDataLoader

        self.loader = StatefulDataLoader(
            PackedStreamDataset(config, eos_id, rank, world_size),
            batch_size=config.micro_batch_size,
            num_workers=config.num_workers,
            collate_fn=collate,
            pin_memory=True,
            drop_last=True,
            persistent_workers=config.num_workers > 0,
            prefetch_factor=4 if config.num_workers > 0 else None,
        )

    def __iter__(self) -> Iterator[Batch]:
        return iter(self.loader)

    def state_dict(self) -> dict[str, Any]:
        return {"kind": "stream", "loader": self.loader.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.loader.load_state_dict(state["loader"])

    def close(self) -> None:
        pass


def take_windows(config: DataConfig, eos_id: int, num_rows: int, seed: int) -> list[dict[str, np.ndarray]]:
    """A fixed list of windows (for a monitoring validation set) from a dedicated seed."""
    iterator = iter(PackedStreamDataset(config, eos_id, seed=seed))
    return [next(iterator) for _ in range(num_rows)]
