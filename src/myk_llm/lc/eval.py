"""Exact-match retrieval accuracy on a grid of context lengths x needle depths.

Scoring is teacher-forced: a query counts as correct when the argmax at every answer
position equals the target. That is exactly greedy-decoding exact match (if each step's
argmax is the gold token given the gold prefix, greedy decoding reproduces the gold
answer), but needs one forward pass per batch instead of one per generated token.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from ..model import MykLLM
from .tasks import Example, TaskConfig, make_example


def score_batch(model: MykLLM, examples: list[Example], iterations: int | None = None) -> list[bool]:
    """Per-query exact match for equal-length examples, one document per row
    (`iterations`: core iterations of a looped model, default its own)."""
    device = model.device
    ids = torch.from_numpy(np.stack([e.ids for e in examples])).to(device)
    rows, positions, spans = [], [], []
    for row, example in enumerate(examples):
        for start, end in example.answers:
            spans.append((len(positions), len(positions) + end - start))
            rows += [row] * (end - start)
            positions += range(start - 1, end - 1)
    rows_t = torch.tensor(rows, device=device)
    positions_t = torch.tensor(positions, device=device)
    loop_args = {} if iterations is None else {"iterations": iterations}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model(ids, seqlens=[torch.tensor([ids.size(1)])] * ids.size(0), **loop_args).hidden
        predicted = model.logits(hidden[rows_t, positions_t]).argmax(-1)
    hits = (predicted == ids[rows_t, positions_t + 1]).cpu().tolist()
    return [all(hits[a:b]) for a, b in spans]


@torch.no_grad()
def evaluate(
    model: MykLLM, task: TaskConfig, lengths: list[int], queries: int, seed: int, tokens_per_batch: int = 65536,
    iterations: int | None = None, hops: list[int] | None = None,
) -> dict:
    """{"accuracy": {L: acc}, "depth": {L: [acc per depth bucket]}, "hops": {L: {h: acc}}},
    with about `queries` queries per length; hop counts are drawn from `hops` (default
    task.hops)."""
    was_training = model.training
    model.eval()
    if hops is not None:
        task = replace(task, hops=hops)
    results: dict[str, dict] = {"accuracy": {}, "depth": {}, "hops": {}}
    buckets = task.depth_buckets
    for length in lengths:
        rng = np.random.default_rng([seed, length])
        examples = [make_example(rng, task, length) for _ in range(-(-queries // task.num_queries))]
        per_batch = max(1, tokens_per_batch // length)
        correct: list[bool] = []
        for start in range(0, len(examples), per_batch):
            correct += score_batch(model, examples[start : start + per_batch], iterations)
        depths = [d for e in examples for d in e.depths]
        hops = [h for e in examples for h in e.hops]
        depth_hits = np.zeros(buckets)
        depth_counts = np.zeros(buckets)
        for ok, depth in zip(correct, depths):
            bucket = min(int(depth * buckets), buckets - 1)
            depth_hits[bucket] += ok
            depth_counts[bucket] += 1
        results["accuracy"][length] = float(np.mean(correct))
        results["depth"][length] = [float(h / c) if c else float("nan") for h, c in zip(depth_hits, depth_counts)]
        results["hops"][length] = {
            h: float(np.mean([ok for ok, k in zip(correct, hops) if k == h])) for h in sorted(set(hops))
        }
    model.train(was_training)
    return results
