"""Training throughput and memory of a config, on random tokens (single GPU).

Runs the real model, objective and optimizers for a few steps, so architecture
variants can be compared on speed before spending a full run:

    python scripts/benchmark_throughput.py configs/experiments/ablation.yaml
    python scripts/benchmark_throughput.py configs/experiments/hybrid-deltanet.yaml data.micro_batch_size=8
"""

from __future__ import annotations

import argparse
import time

import torch

from myk_llm.configuration import load_config
from myk_llm.data import Batch
from myk_llm.model import MykLLM
from myk_llm.train.config import TrainConfig
from myk_llm.train.distributed import DistInfo, parallelize
from myk_llm.train.objectives import Objective
from myk_llm.train.optim import build_optimizers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    config = load_config(TrainConfig, args.config, args.overrides)

    torch.set_float32_matmul_precision("high")
    info = DistInfo(0, 1, 0, torch.device("cuda"))
    model = parallelize(MykLLM(config.model).cuda(), config.parallel, info)
    optimizers = build_optimizers(model, config.optimizer)
    objective = Objective(config.objective, info.device)
    batch_size, seq_len = config.data.micro_batch_size, config.data.seq_len
    ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len + 1), device="cuda")
    # Documents of ~seq_len/4 tokens, to exercise document masking realistically.
    seqlens = [torch.tensor([seq_len // 4] * 3 + [seq_len - 3 * (seq_len // 4)]) for _ in range(batch_size)]
    batch = Batch(ids[:, :-1], ids[:, 1:], seqlens if config.data.document_masking else None)

    def step() -> None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch.input_ids, seqlens=batch.seqlens, head=objective.head(batch, float(batch.num_targets)))
        out.loss.backward()
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    params = model.num_parameters()
    print(
        f"params {params / 1e6:.1f}M (non-embedding {model.num_parameters(exclude_embeddings=True) / 1e6:.1f}M) | "
        f"layers {''.join(config.model.layer_types)} | batch {batch_size}x{seq_len} | "
        f"{batch_size * seq_len * args.steps / elapsed:,.0f} tok/s | step {1000 * elapsed / args.steps:.0f} ms | "
        f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
    )


if __name__ == "__main__":
    main()
