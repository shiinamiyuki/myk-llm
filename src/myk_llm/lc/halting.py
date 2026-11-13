"""Early-exit analysis of looped lc models (notes/loop.md): does the loop settle once a
query's chain is resolved, so that a convergence signal could stop it?

    python -m myk_llm.lc.halting runs/lc/loop-curriculum/chain-h6-s{0,1,2,3} [--hops 1 2 ...]

Each checkpoint runs once at --iterations core iterations and is read out after every
iteration (the core must be the model's last blocks, so the readout is the final norm
and the output head on the loop state). Per query, over the positions that predict its
answer:

  delta_t    ||h_t - h_{t-1}|| / ||h_t||, h = the final-normed state (the add-injected
             state itself grows by the input every iteration)
  kl_t       KL(p_t || p_{t-1}) of the next-token distributions
  conf_t     the probability of the argmax token
  correct_t  exact match of the answer read at iteration t

(max over the answer positions for delta and kl, min for conf). An exit rule stops a
query at the first iteration t >= 2 whose signal passes a threshold, else at the budget
(default: the top of loop.train_iterations), and is scored by accuracy and mean
iterations. Context tokens run every iteration either way, so this measures how many
iterations each query needs, not a full per-token exit (which would also drop the
exited tokens' keys / values from later iterations).

Per-query arrays go to <run>/halting.npz (reused unless --force); the tables pool the
runs of one configuration over seeds (trailing -s<k>).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ..arith.sweep import SEED_SUFFIX
from ..configuration import load_config
from ..model import MykLLM
from .tasks import Example, make_example
from .train import LcConfig

SIGNALS = ("delta", "kl", "conf", "correct")
RULES = [("delta", "<", t) for t in (0.2, 0.1, 0.05, 0.02, 0.01)] + [("kl", "<", t) for t in (0.1, 0.01, 1e-3, 1e-4)] \
    + [("conf", ">", t) for t in (0.9, 0.99, 0.999)]
SHOWN = (2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24)


@torch.no_grad()
def trace(model: MykLLM, examples: list[Example], iterations: int) -> dict[str, np.ndarray]:
    """{signal: [queries, iterations]} for equal-length examples (delta / kl are inf at
    the first iteration)."""
    loop = model.config.loop
    if not loop.core or loop.prelude + loop.core != len(model.layers):
        raise ValueError("needs a looped model whose core is its last blocks")
    device = model.device
    ids = torch.from_numpy(np.stack([e.ids for e in examples])).to(device)
    rows, positions, spans = [], [], []  # spans: each query's slice of the answer positions
    for row, example in enumerate(examples):
        for start, end in example.answers:
            spans.append((len(positions), len(positions) + end - start))
            rows += [row] * (end - start)
            positions += range(start - 1, end - 1)
    rows_t, positions_t = torch.tensor(rows, device=device), torch.tensor(positions, device=device)
    readouts: list[torch.Tensor] = []
    handle = model.layers[-1].register_forward_hook(
        lambda module, args, output: readouts.append(model.final_norm(output[0][rows_t, positions_t]).float()))
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(ids, seqlens=[torch.tensor([ids.size(1)])] * ids.size(0), iterations=iterations)
    finally:
        handle.remove()
    hidden = torch.stack(readouts)  # (iterations, positions, hidden)
    logp = torch.log_softmax(model.logits(hidden).float(), dim=-1)
    inf = torch.full_like(hidden[:1, :, 0], float("inf"))
    per_position = {
        "delta": torch.cat([inf, (hidden[1:] - hidden[:-1]).norm(dim=-1) / hidden[1:].norm(dim=-1)]),
        "kl": torch.cat([inf, (logp[1:].exp() * (logp[1:] - logp[:-1])).sum(-1)]),
        "conf": logp.max(-1).values.exp(),
        "correct": (logp.argmax(-1) == ids[rows_t, positions_t + 1]).float(),
    }
    out = {}
    for name, values in per_position.items():
        reduce = torch.amin if name in ("conf", "correct") else torch.amax
        out[name] = torch.stack([reduce(values[:, a:b], dim=1) for a, b in spans]).cpu().numpy()
    return out


def exit_iterations(signal: np.ndarray, op: str, threshold: float, budget: int) -> np.ndarray:
    """Per query, the first iteration (1-based, >= 2) whose signal is below (op "<") or
    above (">") the threshold, else `budget`."""
    window = signal[:, 1:budget]
    passed = window < threshold if op == "<" else window > threshold
    return np.where(passed.any(1), passed.argmax(1) + 2, budget)


def analyze(run: Path, length: int, examples: int, iterations: int, hops: list[int] | None, force: bool) -> dict:
    cache = run / "halting.npz"
    settings = np.array([length, examples, iterations, *(hops or [])])
    if cache.exists() and not force:
        stored = np.load(cache)
        if np.array_equal(stored["settings"], settings):
            return dict(stored)
    config = load_config(LcConfig, run / "config.yaml")
    model = MykLLM(config.model).cuda()
    model.load_state_dict(torch.load(run / "model.pt", map_location="cuda")["state_dict"])
    model.eval()
    task = replace(config.task, hops=hops or config.task.eval_hops or config.task.hops)
    rng = np.random.default_rng([12345, length])
    pool = [make_example(rng, task, length) for _ in range(examples)]
    per_batch = max(1, 65536 // length)
    parts = [trace(model, pool[i : i + per_batch], iterations) for i in range(0, len(pool), per_batch)]
    loop = config.model.loop
    result = {name: np.concatenate([p[name] for p in parts]) for name in SIGNALS}
    result |= {"hops": np.array([h for e in pool for h in e.hops]), "settings": settings,
               "budget": np.array(loop.train_iterations[1] if loop.train_iterations else loop.iterations)}
    np.savez(cache, **result)
    return result


def tables(name: str, data: dict, budget: int) -> list[str]:
    hops = sorted(set(data["hops"].tolist()))
    shown = [t for t in SHOWN if t <= data["correct"].shape[1]]
    lines = [f"\n### {name} ({len(data['hops'])} queries, exit budget {budget})\n"]
    for signal, title, fmt in (("correct", "accuracy (%)", lambda v: f"{100 * v:.0f}"),
                               ("delta", "median state change delta", lambda v: f"{v:.3f}"),
                               ("kl", "median KL to the previous iteration", lambda v: f"{v:.0e}")):
        lines += [f"\n{title} by hop count (rows) and iteration (columns)\n",
                  "| hops | " + " | ".join(str(t) for t in shown) + " |", "|---|" + "---|" * len(shown)]
        for h in hops:
            values = data[signal][data["hops"] == h]
            stat = values.mean(0) if signal == "correct" else np.median(values, 0)
            lines.append(f"| {h} | " + " | ".join(fmt(stat[t - 1]) for t in shown) + " |")
    lines += ["\nExit rules: accuracy % (mean iterations) by hop count\n",
              "| rule | all | " + " | ".join(str(h) for h in hops) + " |", "|---|---|" + "---|" * len(hops)]
    correct = data["correct"]
    ever = correct[:, :budget].max(1) > 0
    rules = [(f"fixed r = {budget}", np.full(len(correct), budget)),
             ("oracle (first correct iteration)", np.where(ever, correct[:, :budget].argmax(1) + 1, budget))]
    rules += [(f"{signal} {op} {threshold:g}", exit_iterations(data[signal], op, threshold, budget))
              for signal, op, threshold in RULES]
    for label, exits in rules:
        hit = correct[np.arange(len(exits)), exits - 1]
        cells = [f"{100 * hit[m].mean():.0f} ({exits[m].mean():.1f})" for m in [slice(None)] + [data["hops"] == h for h in hops]]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="lc run directories (config.yaml + model.pt)")
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--examples", type=int, default=500, help="examples per run (task.num_queries each)")
    parser.add_argument("--iterations", type=int, default=24, help="core iterations to trace")
    parser.add_argument("--budget", type=int, default=None, help="exit budget (default: top of loop.train_iterations)")
    parser.add_argument("--hops", type=int, nargs="*", default=None, help="default: task.eval_hops, else task.hops")
    parser.add_argument("--force", action="store_true", help="recompute cached traces")
    args = parser.parse_args()
    groups: dict[str, list[dict]] = {}
    for run in map(Path, args.runs):
        data = analyze(run, args.length, args.examples, args.iterations, args.hops, args.force)
        groups.setdefault(f"{run.parent.name}/{SEED_SUFFIX.sub('', run.name)}", []).append(data)
    for name, runs in groups.items():
        pooled = {key: np.concatenate([r[key] for r in runs]) for key in (*SIGNALS, "hops")}
        print("\n".join(tables(f"{name}, {len(runs)} runs", pooled, args.budget or int(runs[0]["budget"]))))


if __name__ == "__main__":
    main()
