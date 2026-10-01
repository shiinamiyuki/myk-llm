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


def report(sweep_dir: Path) -> str:
    """Markdown table: in-distribution mean accuracy, then accuracy per length."""
    results = [json.loads(p.read_text()) for p in sorted(sweep_dir.glob("*/results.json"))]
    if not results:
        return "(no finished runs)"
    lines = []
    for task in sorted({r["run_name"].split("-")[0] for r in results}):
        group = [r for r in results if r["run_name"].split("-")[0] == task]
        lengths = sorted({int(d) for r in group for d in r["accuracy"]})
        train_digits = group[0]["train_digits"]
        header = [f"{d}{'*' if d > train_digits else ''}" for d in lengths]
        lines += [
            f"\n### {task} (trained on 1-{train_digits} digits; * = longer than training)\n",
            "| run | in-dist | " + " | ".join(header) + " | min |",
            "|---|---|" + "---|" * len(lengths) + "---|",
        ]
        for r in sorted(group, key=lambda r: r["run_name"]):
            cells = [f"{100 * r['accuracy'].get(str(d), float('nan')):.0f}" for d in lengths]
            lines.append(f"| {r['run_name']} | {100 * r['in_distribution']:.1f} | " + " | ".join(cells) + f" | {r['train_minutes']:.1f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep")
    parser.add_argument("--report", action="store_true", help="only print the summary table")
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
    table = report(sweep_dir)
    print(table)
    if sweep_dir.exists():
        (sweep_dir / "summary.md").write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
