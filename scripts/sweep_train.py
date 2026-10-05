"""Run a grid of pretraining runs (`python -m myk_llm.train`) and summarize them.

Same sweep files and resume behavior as `myk_llm.arith.sweep`: every combination of the
axes is one run, a finished run (one with a results.json) is skipped. After each run its
final validation metrics are written to <sweep_dir>/<run>/results.json:

    python scripts/sweep_train.py configs/ngram-small/sweeps/main.yaml
    python scripts/sweep_train.py configs/ngram-small/sweeps/main.yaml --report
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml

from myk_llm.arith.sweep import plan, run_all

SEED_SUFFIX = re.compile(r"-s\d+$")


def summarize(run_dir: Path) -> dict | None:
    """Final validation metrics, throughput and parameter counts of a finished run."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    evals = [r for r in rows if any(k.startswith("val/") for k in r)]
    if not evals:
        return None
    final = evals[-1]
    speeds = [r["perf/tokens_per_s"] for r in rows if "perf/tokens_per_s" in r][len(rows) // 10 :]
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    return {
        "run_name": run_dir.name,
        "step": final["step"],
        "steps": config["steps"],
        "val": {k.removeprefix("val/"): v for k, v in final.items() if k.startswith("val/")},
        "tokens_per_s": float(np.median(speeds)) if speeds else None,
        "params": {k.removeprefix("params/"): v for r in rows for k, v in r.items() if k.startswith("params/")},
    }


def report(sweep_dir: Path) -> str:
    results = [json.loads(p.read_text()) for p in sorted(sweep_dir.glob("*/results.json"))]
    if not results:
        return "(no finished runs)"
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(SEED_SUFFIX.sub("", r["run_name"]), []).append(r)
    sets = sorted({k.split("/")[0] for r in results for k in r["val"] if k.endswith("/bpb")})
    lines = [
        "| run | seeds | " + " | ".join(f"{s} bpb" for s in sets) + " | mean bpb | steps | tok/s |",
        "|---|---|" + "---|" * len(sets) + "---|---|---|",
    ]
    rows = []
    for name, runs in groups.items():
        bpb = {s: np.mean([r["val"][f"{s}/bpb"] for r in runs]) for s in sets}
        mean = float(np.mean(list(bpb.values())))
        speed = np.mean([r["tokens_per_s"] or np.nan for r in runs])
        rows.append((mean, f"| {name} | {len(runs)} | " + " | ".join(f"{bpb[s]:.4f}" for s in sets)
                     + f" | {mean:.4f} | {runs[0]['steps']} | {speed / 1e3:.0f}K |"))
    return "\n".join(lines + [row for _, row in sorted(rows)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep")
    parser.add_argument("--report", action="store_true", help="only print the summary table")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="runs/sweeps")
    args = parser.parse_args()
    sweep = yaml.safe_load(Path(args.sweep).read_text())
    sweep_dir = Path(args.output_dir) / sweep["name"]
    if not args.report:
        for run in plan(sweep):
            run_all([run], sweep_dir, "myk_llm.train", dry_run=args.dry_run)
            summary = None if args.dry_run else summarize(sweep_dir / run[0])
            if summary is not None and summary["step"] >= summary["steps"]:
                (sweep_dir / run[0] / "results.json").write_text(json.dumps(summary, indent=1))
    print(report(sweep_dir))


if __name__ == "__main__":
    main()
