"""Run a grid of arithmetic experiments and summarize them.

A sweep file names a base config per task and axes of named override lists; every
combination is one run (a fresh process), and finished runs are skipped, so an
interrupted sweep resumes by re-running the command:

    python -m myk_llm.arith.sweep configs/arith/sweeps/format-position.yaml
    python -m myk_llm.arith.sweep configs/arith/sweeps/format-position.yaml --report   # table only

Sweep file:
    name: format-position
    configs: {add: configs/arith/add.yaml, mul: configs/arith/mul.yaml}
    axes:
      format: {direct: [task.format=direct], cot: [task.format=cot]}
    overrides: [steps=3000]          # applied to every run
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


def plan(sweep: dict) -> list[tuple[str, str, list[str]]]:
    """[(run name, config path, overrides)] for every combination."""
    axes = sweep.get("axes", {})
    runs = []
    for task_name, config_path in sweep["configs"].items():
        for combo in itertools.product(*[list(values.items()) for values in axes.values()]):
            name = "-".join([task_name, *[label for label, _ in combo]])
            overrides = [*sweep.get("overrides", []), *[o for _, group in combo for o in group]]
            runs.append((name, config_path, overrides))
    return runs


SEED_SUFFIX = re.compile(r"-s\d+$")


METRICS = ("accuracy", "accuracy_late", "fillers_accuracy", "fillers_accuracy_late")


def extra_digits(result: dict, metric: str = "accuracy", threshold: float = 0.5) -> int:
    """How many lengths beyond training stay at or above `threshold` accuracy, counting
    up from train_digits + 1 until the first failure."""
    count, length = 0, result["train_digits"] + 1
    while result[metric].get(str(length), 0.0) >= threshold:
        count, length = count + 1, length + 1
    return count


def report(sweep_dir: Path, metric: str = "accuracy") -> str:
    """Markdown tables. Runs that differ only in a trailing `-s<k>` (seed) label are
    averaged; `spread` is the largest max-min gap across seeds at any length, and
    `+digits` the mean number of lengths past training at >= 50% accuracy. Only
    lengths past training are shown, up to the last one any run gets right.
    `metric` picks the accuracy: final checkpoint ("accuracy"), mean of the
    second-half evaluations ("accuracy_late"), or the same on prompts with fillers."""
    results = [json.loads(p.read_text()) for p in sorted(sweep_dir.glob("*/results.json"))]
    results = [{**r, "accuracy": r[metric]} for r in results if metric in r]
    if not results:
        return f"(no finished runs with {metric})"
    lines = []
    for task in sorted({r["run_name"].split("-")[0] for r in results}):
        group = [r for r in results if r["run_name"].split("-")[0] == task]
        train_digits = group[0]["train_digits"]
        lengths = sorted({int(d) for r in group for d in r["accuracy"]} - set(range(train_digits + 1)))
        while lengths and all(r["accuracy"].get(str(lengths[-1]), 0.0) == 0.0 for r in group):
            lengths.pop()
        configs: dict[str, list[dict]] = {}
        for r in group:
            configs.setdefault(SEED_SUFFIX.sub("", r["run_name"]), []).append(r)
        lines += [
            f"\n### {task}, {metric} (trained on 1-{train_digits} digits; columns are longer lengths)\n",
            "| run | seeds | in-dist | +digits | spread | " + " | ".join(map(str, lengths)) + " | min |",
            "|---|---|---|---|---|" + "---|" * len(lengths) + "---|",
        ]
        rows = []
        for name, runs in configs.items():
            acc = {d: [r["accuracy"].get(str(d), 0.0) for r in runs] for d in lengths}
            spread = max((max(v) - min(v) for v in acc.values()), default=0.0)
            plus = sum(extra_digits(r) for r in runs) / len(runs)
            train_lengths = [str(d) for d in range(1, train_digits + 1)]
            cells = [f"{100 * sum(v) / len(v):.0f}" for v in acc.values()]
            in_dist = 100 * sum(
                sum(r["accuracy"].get(d, 0.0) for d in train_lengths if d in r["accuracy"])
                / max(1, sum(d in r["accuracy"] for d in train_lengths))
                for r in runs
            ) / len(runs)
            minutes = sum(r["train_minutes"] for r in runs) / len(runs)
            rows.append((-plus, f"| {name} | {len(runs)} | {in_dist:.1f} | {plus:.1f} | {100 * spread:.0f} | "
                         + " | ".join(cells) + f" | {minutes:.1f} |"))
        lines += [row for _, row in sorted(rows)]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep")
    parser.add_argument("--report", action="store_true", help="only print the summary table")
    parser.add_argument("--metric", choices=METRICS, default=None, help="accuracy to tabulate (default: all available)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="runs/arith")
    args = parser.parse_args()
    sweep = yaml.safe_load(Path(args.sweep).read_text(encoding="utf-8"))
    sweep_dir = Path(args.output_dir) / sweep["name"]

    if not args.report:
        runs = plan(sweep)
        for index, (name, config_path, overrides) in enumerate(runs, start=1):
            if (sweep_dir / name / "results.json").exists():
                print(f"[{index}/{len(runs)}] {name}: done, skipping", flush=True)
                continue
            command = [sys.executable, "-m", "myk_llm.arith.train", config_path, *overrides,
                       f"run_name={name}", f"output_dir={sweep_dir}"]
            print(f"[{index}/{len(runs)}] {name}: {' '.join(overrides)}", flush=True)
            if not args.dry_run:
                log = sweep_dir / name / "train.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("w") as handle:
                    code = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT)
                if code:
                    print(f"  failed (exit {code}); see {log}", flush=True)
    metrics = [args.metric] if args.metric else list(METRICS)
    table = "\n".join(t for t in (report(sweep_dir, m) for m in metrics) if not t.startswith("(no finished"))
    table = table or "(no finished runs)"
    print(table)
    if sweep_dir.exists():
        (sweep_dir / "summary.md").write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
