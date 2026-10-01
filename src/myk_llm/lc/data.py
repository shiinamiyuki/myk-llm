"""Infinite stream of packed training batches of freshly generated retrieval examples.

Examples (lengths uniform in [min_len, max_len]) are packed into rows of max_len
tokens; each example is its own document (document-masked attention) and the loss
covers only the answers.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..data.config import IGNORE_INDEX, Batch
from .tasks import PAD, Example, TaskConfig, make_copy_example, make_example


def answer_targets(example: Example) -> np.ndarray:
    """Next-token targets: IGNORE_INDEX except where the next token is an answer token."""
    targets = np.full(len(example.ids), IGNORE_INDEX, dtype=np.int64)
    for start, end in example.answers:
        targets[start - 1 : end - 1] = example.ids[start:end]
    return targets


class PackedRetrieval(IterableDataset):
    def __init__(self, task: TaskConfig, batch_rows: int, seed: int) -> None:
        super().__init__()
        self.task, self.batch_rows, self.seed = task, batch_rows, seed

    def _rows(self, rng: np.random.Generator) -> Iterator[tuple[np.ndarray, np.ndarray, list[int]]]:
        row_len, pending = self.task.max_len, None
        while True:
            ids, targets, lengths, used = [], [], [], 0
            while True:
                if pending is None:
                    if self.task.copy_fraction and rng.random() < self.task.copy_fraction:
                        example = make_copy_example(rng, self.task)
                    else:
                        example = make_example(rng, self.task, self.task.sample_length(rng), self.task.sample_records(rng))
                    pending = (example.ids, answer_targets(example))
                if used + len(pending[0]) > row_len:
                    break
                ids.append(pending[0])
                targets.append(pending[1])
                lengths.append(len(pending[0]))
                used += len(pending[0])
                pending = None
            if used < row_len:  # padding forms its own (unsupervised) document
                ids.append(np.full(row_len - used, PAD, dtype=np.int64))
                targets.append(np.full(row_len - used, IGNORE_INDEX, dtype=np.int64))
                lengths.append(row_len - used)
            yield np.concatenate(ids), np.concatenate(targets), lengths

    def __iter__(self) -> Iterator[Batch]:
        worker = get_worker_info()
        rng = np.random.default_rng([self.seed, worker.id if worker else 0])
        rows = self._rows(rng)
        while True:
            chunk = [next(rows) for _ in range(self.batch_rows)]
            yield Batch(
                torch.from_numpy(np.stack([r[0] for r in chunk])),
                torch.from_numpy(np.stack([r[1] for r in chunk])),
                [torch.tensor(r[2], dtype=torch.int32) for r in chunk],
            )


def build_loader(task: TaskConfig, batch_rows: int, seed: int, num_workers: int):
    from torch.utils.data import DataLoader

    return DataLoader(
        PackedRetrieval(task, batch_rows, seed),
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=8 if num_workers > 0 else None,
    )
