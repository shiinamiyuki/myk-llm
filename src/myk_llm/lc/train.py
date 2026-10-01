"""Train a small model on synthetic long-context retrieval (single GPU).

    python -m myk_llm.lc.train configs/lc/needle.yaml [key.path=value ...]

Writes <output_dir>/<run_name>/: config.yaml, metrics.jsonl, model.pt and results.json
(exact-match accuracy per context length, per needle depth and per hop count).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..configuration import dump_yaml, load_config
from ..model import MykLLM, ModelConfig
from ..train.config import ObjectiveConfig, OptimizerConfig, ScheduleConfig
from ..train.metrics import MetricsLogger
from ..train.objectives import Objective
from ..train.optim import build_optimizers, lr_multiplier, set_lr
from .data import build_loader
from .eval import evaluate
from .tasks import VOCAB_SIZE, TaskConfig


@dataclass
class LcConfig:
    run_name: str = "lc"
    output_dir: str = "runs/lc"
    seed: int = 0
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)  # see configs/lc/base.yaml
    steps: int = 5000
    batch_rows: int = 32  # rows of task.max_len tokens per step
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    compile: bool = True
    num_workers: int = 4
    log_interval: int = 100
    eval_interval: int = 1000
    eval_queries: int = 200  # per length, during training
    final_eval_queries: int = 1000  # per length, at the end


def late_mean(late: list[tuple[int, dict[int, float]]], learned_step: int | None, lengths: list[int]) -> dict:
    """Mean accuracy per length over the second-half evaluations at or after the
    transition (all of them when retrieval was never learned)."""
    kept = [a for step, a in late if learned_step is None or step >= learned_step]
    return {length: float(np.mean([a[length] for a in kept])) for length in lengths} if kept else {}


def train(config: LcConfig) -> dict:
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    task = config.task
    if config.model.vocab_size < VOCAB_SIZE:
        raise ValueError(f"model.vocab_size must be >= {VOCAB_SIZE}")
    if max(task.eval_lengths + [task.max_len]) > config.model.max_seq_len:
        raise ValueError("model.max_seq_len must cover the longest evaluation length (RoPE tables)")
    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(dump_yaml(config), encoding="utf-8")

    model = MykLLM(config.model).to(device)
    optimizers = build_optimizers(model, config.optimizer)
    base_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
    if config.compile:
        model.compile_blocks()
    objective = Objective(config.objective, device)
    loader = iter(build_loader(task, config.batch_rows, config.seed, config.num_workers))
    metrics = MetricsLogger(run_dir / "metrics.jsonl")
    lengths = task.eval_lengths
    late: list[tuple[int, dict[int, float]]] = []  # (step, accuracy) of second-half evaluations
    # Retrieval appears abruptly after a plateau whose length varies with seed and
    # architecture; record when it happened so a late transition is not read as inability.
    learned_step = None

    def in_distribution(accuracy: dict[int, float]) -> float:
        values = [a for length, a in accuracy.items() if length <= task.max_len]
        return float(np.mean(values)) if values else float("nan")
    params = model.num_parameters(exclude_embeddings=True)
    print(f"[{config.run_name}] {params / 1e6:.2f}M non-embedding params, layers {''.join(config.model.layer_types)}, "
          f"{config.batch_rows}x{task.max_len} tokens/step, {config.steps} steps", flush=True)

    def log_eval(step: int, result: dict, prefix: str) -> None:
        metrics.log(step, {f"{prefix}acc/{length}": a for length, a in result["accuracy"].items()})
        print(f"[{config.run_name}] step {step} {prefix}acc "
              + " ".join(f"{length}:{a:.2f}" for length, a in result["accuracy"].items()), flush=True)

    started = window_start = time.perf_counter()
    loss_sum = token_sum = 0.0
    for step in range(1, config.steps + 1):
        set_lr(optimizers, base_lrs, lr_multiplier(step, config.steps, config.schedule))
        batch = next(loader).to(device)
        num_targets = batch.num_targets
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch.input_ids, seqlens=batch.seqlens, head=objective.head(batch, float(num_targets)))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_clip or float("inf"))
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        loss_sum += float(out.sums["ce"])
        token_sum += num_targets

        if step % config.log_interval == 0:
            elapsed = time.perf_counter() - window_start
            record = {"train/loss": loss_sum / token_sum, "perf/tokens_per_s": config.log_interval * batch.input_ids.numel() / elapsed}
            metrics.log(step, record)
            print(f"[{config.run_name}] step {step} loss {record['train/loss']:.4f} {record['perf/tokens_per_s']:,.0f} tok/s", flush=True)
            loss_sum = token_sum = 0.0
            window_start = time.perf_counter()
        if config.eval_interval and step % config.eval_interval == 0 and step != config.steps:
            result = evaluate(model, task, lengths, config.eval_queries, seed=step)
            log_eval(step, result, "")
            if learned_step is None and in_distribution(result["accuracy"]) >= 0.9:
                learned_step = step
            if step >= config.steps // 2:
                late.append((step, result["accuracy"]))
            window_start = time.perf_counter()

    # Save before the (slower) final evaluation so an interrupted eval can be redone.
    torch.save({"config": dump_yaml(config), "state_dict": model.state_dict()}, run_dir / "model.pt")
    final = evaluate(model, task, lengths, config.final_eval_queries, seed=12345)
    log_eval(config.steps, final, "final_")
    metrics.close()
    if learned_step is None and in_distribution(final["accuracy"]) >= 0.9:
        learned_step = config.steps
    results = {
        "run_name": config.run_name,
        "train_len": task.max_len,
        **final,
        # Single checkpoints are noisy past the training length; the mean over the
        # second-half evaluations from the transition on (learned_step) is steadier.
        "accuracy_late": late_mean(late, learned_step, lengths),
        "in_distribution": in_distribution(final["accuracy"]),
        "learned_step": learned_step,  # first evaluation at >= 90% in distribution (None: never)
        "non_embedding_params": params,
        "train_minutes": (time.perf_counter() - started) / 60,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    train(load_config(LcConfig, args.config, args.overrides))


if __name__ == "__main__":
    main()
