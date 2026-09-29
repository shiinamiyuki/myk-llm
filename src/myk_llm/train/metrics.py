"""Lightweight training-metrics logging: durable JSONL plus optional TensorBoard.

The JSONL file is the always-on backbone (one record per logged step, appended so
it survives crashes and resumes). TensorBoard is an optional live-view layer over
the same numbers.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class MetricsLogger:
    def __init__(self, jsonl_path: str | Path, tensorboard_dir: str | Path | None = None) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.jsonl_path.open("a", encoding="utf-8")
        self._tb: Any = None
        if tensorboard_dir is not None:
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(log_dir=str(tensorboard_dir))

    def log(self, step: int, metrics: dict[str, float], wall_time: float | None = None) -> None:
        record: dict[str, Any] = {"step": step, "time": wall_time if wall_time is not None else time.time(), **metrics}
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()
        if self._tb is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._tb.add_scalar(key, value, step)

    def close(self) -> None:
        self._handle.close()
        if self._tb is not None:
            self._tb.close()
