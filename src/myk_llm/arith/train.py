"""Train a small character-level model on arithmetic (single GPU).

    python -m myk_llm.arith.train configs/arith/add.yaml [key.path=value ...]

Writes <output_dir>/<run_name>/: config.yaml, metrics.jsonl, model.pt and
results.json (final exact-match accuracy per operand length).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..configuration import dump_yaml, load_config
from ..model import ModelConfig
from ..train.config import ObjectiveConfig, OptimizerConfig, ScheduleConfig
from ..train.metrics import MetricsLogger
from ..train.objectives import Objective
from ..train.optim import build_optimizers, lr_multiplier, set_lr
from .data import build_loader
from .model import ArithModel, DigitEmbeddingConfig, evaluate_lengths
from .tasks import TaskConfig


@dataclass
class ArithConfig:
    run_name: str = "arith"
    output_dir: str = "runs/arith"
    seed: int = 0
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)  # see configs/arith/base.yaml
    digit_embedding: DigitEmbeddingConfig = field(default_factory=DigitEmbeddingConfig)
    steps: int = 5000
    batch_rows: int = 64
    row_len: int = 1024
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    compile: bool = True
    num_workers: int = 4
    log_interval: int = 100
    eval_interval: int = 1000
    eval_examples: int = 200  # per length, during training
    final_eval_examples: int = 1000  # per length, at the end


def train(config: ArithConfig) -> dict:
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(dump_yaml(config), encoding="utf-8")

    model = ArithModel(config.model, config.digit_embedding).to(device)
    optimizers = build_optimizers(model.lm, config.optimizer)
    if model.digit_embed is not None:
        optimizers[-1].add_param_group({"params": [model.digit_embed.weight], "weight_decay": 0.0, "lr": config.optimizer.lr})
    base_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
    if config.compile:
        model.lm.compile_blocks()
    objective = Objective(config.objective, device)
    loader = iter(build_loader(config.task, config.batch_rows, config.row_len, config.seed, config.num_workers))
    metrics = MetricsLogger(run_dir / "metrics.jsonl")
    lengths = config.task.eval_lengths()
    params = model.lm.num_parameters(exclude_embeddings=True)
    print(f"[{config.run_name}] {params / 1e6:.2f}M non-embedding params, "
          f"{config.batch_rows}x{config.row_len} tokens/step, {config.steps} steps", flush=True)

    started = window_start = time.perf_counter()
    loss_sum = token_sum = 0.0
    for step in range(1, config.steps + 1):
        set_lr(optimizers, base_lrs, lr_multiplier(step, config.steps, config.schedule))
        batch = next(loader).to(device)
        num_targets = batch.num_targets
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch, objective.head(batch, float(num_targets)))
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
            accuracy = evaluate_lengths(model, config.task, lengths, config.eval_examples, seed=step)
            metrics.log(step, {f"acc/{d}": a for d, a in accuracy.items()})
            print(f"[{config.run_name}] step {step} acc " + " ".join(f"{d}:{a:.2f}" for d, a in accuracy.items()), flush=True)
            window_start = time.perf_counter()

    accuracy = evaluate_lengths(model, config.task, lengths, config.final_eval_examples, seed=12345)
    results = {
        "run_name": config.run_name,
        "accuracy": accuracy,
        "train_digits": config.task.train_digits,
        "in_distribution": sum(a for d, a in accuracy.items() if d <= config.task.train_digits)
        / max(1, sum(d <= config.task.train_digits for d in accuracy)),
        "non_embedding_params": params,
        "train_minutes": (time.perf_counter() - started) / 60,
    }
    metrics.log(config.steps, {f"final_acc/{d}": a for d, a in accuracy.items()})
    metrics.close()
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    torch.save({"config": dump_yaml(config), "state_dict": model.state_dict()}, run_dir / "model.pt")
    print(f"[{config.run_name}] final acc " + " ".join(f"{d}:{a:.3f}" for d, a in accuracy.items()), flush=True)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    train(load_config(ArithConfig, args.config, args.overrides))


if __name__ == "__main__":
    main()
