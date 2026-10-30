"""Run a grid of retrieval experiments and summarize them.

Same sweep files and resume behavior as `myk_llm.arith.sweep` (see there):

    python -m myk_llm.lc.sweep configs/lc/sweeps/e2-length.yaml
    python -m myk_llm.lc.sweep configs/lc/sweeps/e2-length.yaml --report --at 8192

The report averages runs that differ only in their trailing `-s<k>` seed label.
`eff@90` / `eff@50` is the longest context length up to which every evaluated length
stays at >= 90% / >= 50% accuracy (0 if the shortest already fails). `--at L` adds
accuracy per needle-depth bucket and per hop count at length L, and the final scans
(hop extrapolation, core iteration counts of looped models) at L.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ..arith.sweep import SEED_SUFFIX, plan, run_all

METRICS = ("accuracy", "accuracy_late")


def effective_length(accuracy: dict[str, float], threshold: float) -> int:
    longest = 0
    for length in sorted(accuracy, key=int):
        if accuracy[length] < threshold:
            break
        longest = int(length)
    return longest


def _k(tokens: float) -> str:
    return f"{tokens / 1024:.3g}K" if tokens >= 1024 else f"{tokens:.0f}"


def _group(results: list[dict]) -> dict[str, dict[str, list[dict]]]:
    """{task: {config name without seed: [runs]}}"""
    groups: dict[str, dict[str, list[dict]]] = {}
    for r in results:
        task = r["run_name"].split("-")[0]
        groups.setdefault(task, {}).setdefault(SEED_SUFFIX.sub("", r["run_name"]), []).append(r)
    return groups


def report(sweep_dir: Path, metric: str = "accuracy", at: int | None = None) -> str:
    results = [json.loads(p.read_text()) for p in sorted(sweep_dir.glob("*/results.json"))]
    results = [r for r in results if r.get(metric)]
    if not results:
        return f"(no finished runs with {metric})"
    lines = []
    for task, configs in _group(results).items():
        lengths = sorted({int(k) for runs in configs.values() for r in runs for k in r[metric]})
        train_len = next(iter(configs.values()))[0]["train_len"]
        lines += [
            f"\n### {task}, {metric} (trained on <= {train_len} tokens; columns are context lengths)\n",
            "| run | seeds | eff@90 | eff@50 | " + " | ".join(_k(n) for n in lengths) + " | min |",
            "|---|---|---|---|" + "---|" * len(lengths) + "---|",
        ]
        rows = []
        for name, runs in configs.items():
            eff90 = np.mean([effective_length(r[metric], 0.9) for r in runs])
            eff50 = np.mean([effective_length(r[metric], 0.5) for r in runs])
            cells = [f"{100 * np.mean([r[metric].get(str(n), np.nan) for r in runs]):.0f}" for n in lengths]
            minutes = np.mean([r["train_minutes"] for r in runs])
            rows.append(((-eff50, -eff90, name), f"| {name} | {len(runs)} | {_k(eff90)} | {_k(eff50)} | "
                         + " | ".join(cells) + f" | {minutes:.1f} |"))
        lines += [row for _, row in sorted(rows)]
        if at is not None:
            lines += _detail(configs, at)
            lines += _scan(configs, at)
    return "\n".join(lines)


def _scan(configs: dict[str, list[dict]], length: int) -> list[str]:
    """Accuracy per hop count in the final scans (lc.train `eval_iterations` /
    `task.eval_hops`) at `length`: one row per run and core iteration count."""
    key = str(length)
    cells: dict[tuple[str, int | None], list[dict[str, float]]] = {}
    for name, runs in configs.items():
        for r in runs:
            for scan in r.get("scan", []):
                if key in scan["hops"]:
                    cells.setdefault((name, scan["iterations"]), []).append(scan["hops"][key])
    if not cells:
        return []
    hops = sorted({h for runs in cells.values() for r in runs for h in r}, key=int)
    lines = [
        f"\nFinal scan at {_k(length)}: accuracy by hop count per core iteration count (- = the model's default)\n",
        "| run | iterations | seeds | " + " | ".join(f"{h} hops" for h in hops) + " |",
        "|---|---|---|" + "---|" * len(hops),
    ]
    for (name, iterations), runs in sorted(cells.items(), key=lambda item: (item[0][0], item[0][1] or 0)):
        row = [f"{100 * np.mean([r.get(h, np.nan) for r in runs]):.0f}" for h in hops]
        lines.append(f"| {name} | {iterations or '-'} | {len(runs)} | " + " | ".join(row) + " |")
    return lines


def _detail(configs: dict[str, list[dict]], length: int) -> list[str]:
    key = str(length)
    runs_at = {name: [r for r in runs if key in r["depth"]] for name, runs in configs.items()}
    runs_at = {name: runs for name, runs in runs_at.items() if runs}
    if not runs_at:
        return [f"\n(no evaluation at length {length})"]
    buckets = len(next(iter(runs_at.values()))[0]["depth"][key])
    lines = [
        f"\nAccuracy by needle depth at {_k(length)} (final checkpoint; 0 = start of the context)\n",
        "| run | " + " | ".join(f"{100 * i // buckets}%" for i in range(buckets)) + " |",
        "|---|" + "---|" * buckets,
    ]
    for name, runs in sorted(runs_at.items()):
        depth = np.nanmean([r["depth"][key] for r in runs], axis=0)
        lines.append(f"| {name} | " + " | ".join(f"{100 * d:.0f}" for d in depth) + " |")
    hops = sorted({h for runs in runs_at.values() for r in runs for h in r["hops"][key]}, key=int)
    if len(hops) > 1:
        lines += [
            f"\nAccuracy by hop count at {_k(length)}\n",
            "| run | " + " | ".join(f"{h} hops" for h in hops) + " |",
            "|---|" + "---|" * len(hops),
        ]
        for name, runs in sorted(runs_at.items()):
            cells = [f"{100 * np.mean([r['hops'][key].get(h, np.nan) for r in runs]):.0f}" for h in hops]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep")
    parser.add_argument("--report", action="store_true", help="only print the summary table")
    parser.add_argument("--metric", choices=METRICS, default=None, help="accuracy to tabulate (default: all available)")
    parser.add_argument("--at", type=int, default=None, help="add depth / hop tables at this context length")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="runs/lc")
    args = parser.parse_args()
    sweep = yaml.safe_load(Path(args.sweep).read_text(encoding="utf-8"))
    sweep_dir = Path(args.output_dir) / sweep["name"]
    if not args.report:
        run_all(plan(sweep), sweep_dir, "myk_llm.lc.train", args.dry_run)
    metrics = [args.metric] if args.metric else list(METRICS)
    tables = [report(sweep_dir, m, args.at if m == metrics[0] else None) for m in metrics]
    table = "\n".join(t for t in tables if not t.startswith("(no finished")) or "(no finished runs)"
    print(table)
    if sweep_dir.exists():
        (sweep_dir / "summary.md").write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
