"""Hashed n-gram tables trained with sparse (lazy) Adam, kept on the GPU or in host memory.

A step looks up only a small part of the tables: 131K tokens x 2 orders touch ~200K
distinct rows whatever the table size. Dense AdamW (`optimizer.ngram.update: dense`)
still materializes a full-size gradient and updates every row every step. `lazy`
instead, per optimizer step:

1. takes the step's distinct row ids (all its micro-batches) and gathers those rows,
   with their Adam moments and the step each was last updated at, into compact
   buffers on the GPU;
2. runs forward/backward against the compact rows (a leaf tensor: its gradient is
   compact too, and enters the global gradient-norm clip like any parameter);
3. applies Adam to the compact rows on the GPU and writes them back.

Rows the step does not touch keep their weights and moments unchanged (LazyAdam, as
torch.optim.SparseAdam; bias correction by the global step count). Dense Adam keeps
moving such a row along its decaying momentum, and weight decay keeps shrinking it.
`catch_up` restores that: when a row last updated at step s is gathered again at step
t, the t - 1 - s updates dense Adam would have made with zero gradients are applied
first, in closed form (exact but for eps, bias correction and a learning rate that
changed within the gap; see `_catch_up`). Lookups outside training (evaluation, export)
see the caught-up rows too.

`placement: host` keeps the weights, moments and last-update steps in host memory
(12 bytes per parameter in fp32, 8 with the rowwise second moment, plus 4 per row), so
the tables are bounded by host RAM rather than GPU memory. The row ids depend only on
the input tokens, so a CPU worker thread gathers the next step's rows and copies them to
the GPU while the current step computes (`prefetch`). That prefetch runs before the
current step's write-back, so the rows both steps touch are stale in it; they are
patched on the GPU from the current step's updated rows. The result is the same as
`placement: gpu`, bit for bit.

Worker queue (one thread, FIFO), step t uses buffer slot t % 2:

    ... gather(t+1) | scatter(t) | gather(t+2) | scatter(t+1) ...

gather(t+1) is enqueued at the start of step t, after the patch of step t (which reads
slot (t-1) % 2, the slot gather(t+1) overwrites); scatter(t) after step t's Adam update.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.embeddings import NgramEmbedding
from .config import NgramOptimConfig

STATE_NAMES = ("weight", "exp_avg", "exp_avg_sq", "last_step")


@dataclass
class _Slot:
    """Compact buffers of one step: sorted row ids and the rows' state (STATE_NAMES);
    the first `n` rows are valid."""

    ids: torch.Tensor
    state: list[torch.Tensor]
    n: int = 0

    def view(self) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return self.ids[: self.n], [s[: self.n] for s in self.state]


def _state_like(rows: int, dim: int, rowwise: bool, device: torch.device, pin: bool = False) -> list[torch.Tensor]:
    """Empty weight, exp_avg, exp_avg_sq, last_step for `rows` rows."""
    def make(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.empty(*shape, dtype=dtype, device=device, pin_memory=pin)

    return [make(rows, dim), make(rows, dim), make(rows) if rowwise else make(rows, dim), make(rows, dtype=torch.int32)]


def _empty_slot(capacity: int, dim: int, rowwise: bool, device: torch.device, pin: bool) -> _Slot:
    ids = torch.empty(capacity, dtype=torch.int64, device=device, pin_memory=pin)
    return _Slot(ids, _state_like(capacity, dim, rowwise, device, pin))


class SparseNgramTables:
    """Owns the n-gram tables of `module` (stacked into one (orders x buckets, dim)
    table), their Adam state, and the per-step compact rows.

    The module's own table parameters are replaced by empty placeholders; the module
    reads rows through `row_source`. Construct this while the model is still on the
    CPU, before moving it to the GPU, so that the full tables never reach the GPU with
    `placement: host`.
    """

    def __init__(self, module: NgramEmbedding, config: NgramOptimConfig, lr: float, betas: tuple[float, float],
                 eps: float, device: torch.device, max_rows_per_step: int) -> None:
        if config.update != "lazy":
            raise ValueError("SparseNgramTables implements optimizer.ngram.update=lazy")
        self.module, self.config, self.device = module, config, device
        self.host = config.placement == "host"
        self.prefetch = self.host and config.prefetch
        self.rowwise = config.second_moment == "rowwise"
        self.lr, self.betas, self.eps = lr * config.lr_scale, betas, eps
        self.lr_multiplier = 1.0
        self.num_buckets, self.num_orders = module.num_buckets, len(module.tables)
        self.dim = module.tables[0].embedding_dim
        num_rows = self.num_orders * self.num_buckets
        store = torch.device("cpu") if self.host else device

        # Weights first, freeing each module table once copied, then the moments: the
        # peak is one table copy above the final state, not two.
        weight = torch.empty(num_rows, self.dim, device=store)
        for k, table in enumerate(module.tables):
            weight[k * self.num_buckets : (k + 1) * self.num_buckets].copy_(table.weight.detach())
            table.weight = nn.Parameter(torch.empty(0, self.dim), requires_grad=False)
        self.state = [weight, torch.zeros_like(weight),
                      torch.zeros(num_rows, device=store) if self.rowwise else torch.zeros_like(weight),
                      torch.zeros(num_rows, dtype=torch.int32, device=store)]  # STATE_NAMES
        self.step_count = 0
        module.row_source = self.rows_for

        capacity = min(max_rows_per_step, num_rows)
        self.gpu_slots = [_empty_slot(capacity, self.dim, self.rowwise, device, pin=False) for _ in range(2)]
        if self.host:
            self.pinned_slots = [_empty_slot(capacity, self.dim, self.rowwise, torch.device("cpu"), pin=True)
                                 for _ in range(2)]
            self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ngram-tables")
            self.h2d_stream = torch.cuda.Stream(device)
            self.d2h_stream = torch.cuda.Stream(device)
            # The GPU slots come from the caching allocator on the current stream, which
            # may hand out blocks that kernels queued there still use. The copy streams
            # must wait for those before they first write the slots, and freeing the
            # slots must wait for the copy streams (record_stream).
            current = torch.cuda.current_stream(device)
            for side in (self.h2d_stream, self.d2h_stream):
                side.wait_stream(current)
                for slot in self.gpu_slots:
                    for tensor in (slot.ids, *slot.state):
                        tensor.record_stream(side)
        self._slot = 1  # slot of the current / last step
        self._active: _Slot | None = None  # the current step's rows (between begin_step and step)
        self._leaf: torch.Tensor | None = None
        self._last: _Slot | None = None  # the previous step's updated rows (host prefetch patch)
        self._pending: Future | None = None  # gather of the next step
        self._stats = self._zero_stats()

    @property
    def weight(self) -> torch.Tensor:
        return self.state[0]

    @property
    def exp_avg(self) -> torch.Tensor:
        return self.state[1]

    @property
    def exp_avg_sq(self) -> torch.Tensor:
        return self.state[2]

    # -- sizes -------------------------------------------------------------------------
    def num_parameters(self) -> int:
        return self.weight.numel()

    def state_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.state)

    # -- the training step -------------------------------------------------------------
    def _unique_ids(self, input_ids: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        ids = torch.cat([self.module.row_ids(x.to(device, non_blocking=True)).flatten() for x in input_ids])
        return torch.unique(ids)  # sorted

    def begin_step(self, input_ids: list[torch.Tensor], next_input_ids: list[torch.Tensor] | None = None) -> None:
        """Stage the rows of this optimizer step's micro-batches (`input_ids`: their
        token ids, on any device), and with host prefetch start gathering the next
        step's rows."""
        if self._active is not None:
            raise RuntimeError("begin_step called twice without step()")
        self._slot = slot = 1 - self._slot
        buf = self.gpu_slots[slot]
        if not self.host:
            ids = self._unique_ids(input_ids, self.device)
            buf.n = n = ids.numel()
            buf.ids[:n].copy_(ids)
            for src, dst in zip(self.state, buf.state):
                torch.index_select(src, 0, ids, out=dst[:n])
        else:
            start = time.perf_counter()
            if self._pending is None:  # no prefetch: gather now, after the last write-back
                self._pending = self.worker.submit(self._gather, input_ids, slot, None)
            got_slot, n, ready = self._pending.result()
            assert got_slot == slot
            self._pending = None
            torch.cuda.current_stream(self.device).wait_event(ready)
            self._stats["wait_s"] += time.perf_counter() - start
            buf.n = n
            if self._last is not None and self.prefetch:
                self._patch(buf, self._last)
            if self.prefetch and next_input_ids is not None:
                patched = torch.cuda.Event()
                patched.record(torch.cuda.current_stream(self.device))
                self._pending = self.worker.submit(self._gather, next_input_ids, 1 - slot, patched)
        if self.config.catch_up:
            with torch.no_grad():
                self._catch_up(*buf.view()[1], step=self.step_count + 1)
        self._active = buf
        self._leaf = buf.state[0][: buf.n].detach().requires_grad_()
        self._stats["rows"] += buf.n

    def parameters(self) -> list[torch.Tensor]:
        """The current step's compact rows (for gradient clipping)."""
        return [self._leaf] if self._leaf is not None else []

    def rows_for(self, ids: torch.Tensor) -> torch.Tensor:
        """Summed rows (batch, seq_len, dim) of row ids (orders, batch, seq_len)."""
        if self._active is not None and torch.is_grad_enabled():
            staged = self._active.ids[: self._active.n]
            return F.embedding(torch.searchsorted(staged, ids), self._leaf).sum(0)
        if self._active is not None:
            raise RuntimeError("no-grad forward between begin_step and step")
        if not self.host and not self.config.catch_up:
            return F.embedding(ids, self.weight).sum(0)
        # Evaluation / generation: read this batch's rows now.
        unique, inverse = torch.unique(ids, return_inverse=True)
        return F.embedding(inverse, self._read(unique)).sum(0)

    @torch.no_grad()
    def _read(self, ids: torch.Tensor) -> torch.Tensor:
        """Current weights of rows `ids` (on the GPU), caught up with `catch_up`."""
        names = STATE_NAMES if self.config.catch_up else STATE_NAMES[:1]
        if self.host:
            self.flush()
            ids = ids.cpu()
        rows = [s.index_select(0, ids).to(self.device) for s, _ in zip(self.state, names)]
        if self.config.catch_up:
            self._catch_up(*rows, step=self.step_count + 1)
        return rows[0]

    def set_lr_multiplier(self, multiplier: float) -> None:
        self.lr_multiplier = multiplier

    @torch.no_grad()
    def step(self) -> None:
        """Adam on the staged rows, then write them back."""
        buf, leaf = self._active, self._leaf
        if buf is None or leaf is None:
            raise RuntimeError("step() without begin_step()")
        self._active = self._leaf = None
        self.step_count += 1
        self._stats["steps"] += 1
        ids, (weight, exp_avg, exp_avg_sq, last_step) = buf.view()
        if leaf.grad is not None:
            self._adam(weight, leaf.grad, exp_avg, exp_avg_sq)
        last_step.fill_(self.step_count)
        if not self.host:
            for dst, src in zip(self.state, buf.view()[1]):
                dst.index_copy_(0, ids, src)
            return
        slot, pinned = self._slot, self.pinned_slots[self._slot]
        pinned.n = buf.n
        self.d2h_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.d2h_stream):
            for dst, src in zip(pinned.view()[1], buf.view()[1]):
                dst.copy_(src, non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.d2h_stream)
        self.worker.submit(self._scatter, slot, done)
        self._last = buf

    def _adam(self, weight: torch.Tensor, grad: torch.Tensor, exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor) -> None:
        """torch.optim.AdamW's update rule on the compact rows."""
        beta1, beta2 = self.betas
        lr = self.lr * self.lr_multiplier
        if self.config.weight_decay:
            weight.mul_(1 - lr * self.config.weight_decay)
        exp_avg.lerp_(grad, 1 - beta1)
        if self.rowwise:
            exp_avg_sq.mul_(beta2).add_(grad.square().mean(1), alpha=1 - beta2)
            second = exp_avg_sq[:, None]
        else:
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            second = exp_avg_sq
        bias1 = 1 - beta1**self.step_count
        bias2 = 1 - beta2**self.step_count
        denom = (second.sqrt() / math.sqrt(bias2)).add_(self.eps)
        weight.addcdiv_(exp_avg, denom, value=-lr / bias1)

    def _catch_up(self, weight: torch.Tensor, exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor, last_step: torch.Tensor,
                  step: int) -> None:
        """Bring rows last updated at `last_step` to the start of `step`, as dense AdamW
        would have with zero gradients in between. Skipped step j (1..k) decays the row
        by d = 1 - lr wd, then moves it by lr m_j / sqrt(v_j) = lr r^j m / sqrt(v) with
        m_j = beta1^j m, v_j = beta2^j v, r = beta1 / sqrt(beta2). Over the gap:
            w_k = d^k (w - lr m / sqrt(v) sum_{j=1..k} q^j),  q = r / d,
        and the moments end at beta1^k m and beta2^k v. Approximations: the current lr and
        bias correction (step - 1) for the whole gap, and eps not scaled with sqrt(v_j);
        r^j halves every ~9 steps, so only the last few skipped steps matter."""
        if step <= 1:
            return
        beta1, beta2 = self.betas
        lr = self.lr * self.lr_multiplier
        decay = 1 - lr * self.config.weight_decay
        skipped = (step - 1 - last_step).clamp_(min=0).float()
        q = beta1 / math.sqrt(beta2) / decay
        drift = (1 - torch.pow(q, skipped)).mul_(q / (1 - q))
        bias = math.sqrt(1 - beta2 ** (step - 1)) / (1 - beta1 ** (step - 1))
        second = exp_avg_sq[:, None] if self.rowwise else exp_avg_sq
        weight.addcdiv_(exp_avg * drift.mul_(lr * bias)[:, None], second.sqrt().add_(self.eps), value=-1)
        if self.config.weight_decay:
            weight.mul_(torch.pow(decay, skipped)[:, None])
        exp_avg.mul_(torch.pow(beta1, skipped)[:, None])
        decay2 = torch.pow(beta2, skipped)
        exp_avg_sq.mul_(decay2 if self.rowwise else decay2[:, None])

    def _patch(self, buf: _Slot, last: _Slot) -> None:
        """Overwrite the rows of `buf` that the previous step also touched (stale in the
        prefetch) with that step's updated rows."""
        ids, last_ids = buf.ids[: buf.n], last.ids[: last.n]
        where = torch.searchsorted(last_ids, ids).clamp_(max=last.n - 1)
        hit = last_ids[where] == ids
        # torch.where over all rows rather than hit.nonzero(): no GPU -> CPU sync, so
        # the CPU keeps running ahead of the GPU.
        for to, frm in zip(buf.view()[1], last.view()[1]):
            mask = hit if to.dim() == 1 else hit[:, None]
            torch.where(mask, frm.index_select(0, where), to, out=to)

    # -- worker-thread tasks (host placement) ------------------------------------------
    def _gather(self, input_ids: list[torch.Tensor], slot: int, after: torch.cuda.Event | None):
        start = time.perf_counter()
        ids = self._unique_ids([x.cpu() for x in input_ids], torch.device("cpu"))
        pinned, gpu = self.pinned_slots[slot], self.gpu_slots[slot]
        n = pinned.n = ids.numel()
        if n > pinned.ids.numel():
            raise RuntimeError(f"{n} rows in one step exceed the staging capacity {pinned.ids.numel()}")
        pinned.ids[:n].copy_(ids)
        for src, dst in zip(self.state, pinned.state):
            torch.index_select(src, 0, ids, out=dst[:n])
        with torch.cuda.device(self.device), torch.cuda.stream(self.h2d_stream):
            if after is not None:
                self.h2d_stream.wait_event(after)
            for src, dst in zip([pinned.ids, *pinned.state], [gpu.ids, *gpu.state]):
                dst[:n].copy_(src[:n], non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self.h2d_stream)
        self._stats["gather_s"] += time.perf_counter() - start
        return slot, n, ready

    def _scatter(self, slot: int, done: torch.cuda.Event) -> None:
        done.synchronize()
        start = time.perf_counter()
        ids, rows = self.pinned_slots[slot].view()
        for dst, src in zip(self.state, rows):
            dst.index_copy_(0, ids, src)
        self._stats["scatter_s"] += time.perf_counter() - start

    def flush(self) -> None:
        """Wait for all queued write-backs (and a pending prefetch)."""
        if self.host:
            self.worker.submit(lambda: None).result()

    # -- logging, checkpoints, export --------------------------------------------------
    @staticmethod
    def _zero_stats() -> dict[str, float]:
        return {"steps": 0, "rows": 0, "wait_s": 0.0, "gather_s": 0.0, "scatter_s": 0.0}

    def pop_stats(self) -> dict[str, float]:
        """Per-step averages since the last call: rows touched, main-thread wait for the
        rows (ms), worker gather / scatter time (ms)."""
        stats, steps = self._stats, max(1, self._stats["steps"])
        self._stats = self._zero_stats()
        out = {"ngram/rows_per_step": stats["rows"] / steps}
        if self.host:
            out.update({"ngram/wait_ms": 1e3 * stats["wait_s"] / steps, "ngram/gather_ms": 1e3 * stats["gather_s"] / steps,
                        "ngram/scatter_ms": 1e3 * stats["scatter_s"] / steps})
        return out

    def state_dict(self) -> dict[str, Any]:
        self.flush()
        return {**dict(zip(STATE_NAMES, self.state)), "step": self.step_count}

    def load_state_dict(self, state: dict[str, Any], load_optimizer: bool = True) -> None:
        self.flush()
        self._pending = self._last = None  # staged rows predate the loaded tables
        self.weight.copy_(state["weight"])
        if load_optimizer:
            for name, tensor in zip(STATE_NAMES[1:], self.state[1:]):
                tensor.copy_(state[name])
            self.step_count = int(state["step"])

    def save(self, directory: Path) -> None:
        torch.save(self.state_dict(), directory / "ngram_tables.pt")

    def load(self, directory: Path, load_optimizer: bool = True) -> None:
        path = directory / "ngram_tables.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: checkpoint was not trained with optimizer.ngram.update=lazy")
        self.load_state_dict(torch.load(path, map_location="cpu", mmap=True, weights_only=True), load_optimizer)

    def export_weights(self, chunk_rows: int = 1 << 20) -> dict[str, torch.Tensor]:
        """The tables (caught up with `catch_up`) under the model's own parameter names,
        on the CPU (for model.safetensors)."""
        self.flush()
        num_rows = self.weight.size(0)
        if self.config.catch_up:
            weight = torch.cat([
                self._read(torch.arange(start, min(start + chunk_rows, num_rows), device=self.device)).cpu()
                for start in range(0, num_rows, chunk_rows)
            ])
        else:
            weight = self.weight.cpu()
        nb = self.num_buckets
        return {f"embed.ngram.tables.{k}.weight": weight[k * nb : (k + 1) * nb] for k in range(self.num_orders)}

    def close(self) -> None:
        if self.host:
            self.flush()
            self.worker.shutdown()
