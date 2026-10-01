"""Infinite stream of packed training batches of freshly generated problems.

Problems are packed into rows of `row_len` tokens; each problem is its own document
(document-masked attention, so problems never see each other) and the loss covers
only the completion (CoT + answer + EOS), never the prompt.
"""

from __future__ import annotations

import random
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..data.config import IGNORE_INDEX, Batch
from .tasks import CharTokenizer, TaskConfig, add_fillers, render, sample_problem


def encode_example(tokenizer: CharTokenizer, prompt: str, completion: str) -> tuple[list[int], list[int]]:
    """Token ids and next-token targets (IGNORE_INDEX except where the next token is
    part of the completion)."""
    ids = tokenizer.encode(prompt + completion)
    targets = [IGNORE_INDEX] * len(ids)
    for i in range(len(prompt) - 1, len(ids) - 1):
        targets[i] = ids[i + 1]
    return ids, targets


class PackedArithmetic(IterableDataset):
    def __init__(self, task: TaskConfig, batch_rows: int, row_len: int, seed: int) -> None:
        super().__init__()
        self.task, self.batch_rows, self.row_len, self.seed = task, batch_rows, row_len, seed
        self.tokenizer = CharTokenizer()

    def _rows(self, rng: random.Random) -> Iterator[tuple[list[int], list[int], list[int]]]:
        pending: tuple[list[int], list[int]] | None = None
        while True:
            ids: list[int] = []
            targets: list[int] = []
            lengths: list[int] = []
            while True:
                if pending is None:
                    prompt, completion = render(*sample_problem(rng, self.task), self.task)
                    prompt = add_fillers(prompt, rng, self.task.filler_prob, self.task.max_fillers)
                    pending = encode_example(self.tokenizer, prompt, completion)
                if len(pending[0]) > self.row_len:
                    raise ValueError(f"an example ({len(pending[0])} tokens) exceeds row_len {self.row_len}")
                if len(ids) + len(pending[0]) > self.row_len:
                    break
                ids += pending[0]
                targets += pending[1]
                lengths.append(len(pending[0]))
                pending = None
            fill = self.row_len - len(ids)
            if fill:  # padding forms its own (unsupervised) document
                ids += [self.tokenizer.pad_id] * fill
                targets += [IGNORE_INDEX] * fill
                lengths.append(fill)
            yield ids, targets, lengths

    def __iter__(self) -> Iterator[Batch]:
        worker = get_worker_info()
        rng = random.Random(self.seed * 1000 + (worker.id if worker else 0))
        rows = self._rows(rng)
        while True:
            chunk = [next(rows) for _ in range(self.batch_rows)]
            yield Batch(
                torch.tensor(np.array([r[0] for r in chunk]), dtype=torch.long),
                torch.tensor(np.array([r[1] for r in chunk]), dtype=torch.long),
                [torch.tensor(r[2], dtype=torch.int32) for r in chunk],
            )


def build_loader(task: TaskConfig, batch_rows: int, row_len: int, seed: int, num_workers: int):
    from torch.utils.data import DataLoader

    return DataLoader(
        PackedArithmetic(task, batch_rows, row_len, seed),
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=8 if num_workers > 0 else None,
    )
