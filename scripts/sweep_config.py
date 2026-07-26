"""Benchmark one model-size configuration: params and training throughput.

Holds hidden_size fixed at 1024 (8 heads x 128 head_dim) so the Qwen embedding
can be reused, and sweeps the free knobs: layer count, ffn multiplier, attention
pattern, plus the run-time levers (batch size, sequence length, gradient
checkpointing). Uses random tokens and the real Muon optimizer so the measured
throughput matches the training loop. Run it repeatedly to sweep a grid, e.g.:

    for L in 25 29 33; do
        python scripts/sweep_config.py --num-layers $L --batch-size 8 --seq-len 2048
    done

Outputs: total params, params excluding the (tied) token embedding, and training
throughput in tokens/sec.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "myk-llm"))

import training as T  # noqa: E402
from model import MykLLM, MykLLMConfig  # noqa: E402

VOCAB_SIZE = 65536
HEAD_DIM = 128
NUM_ATTENTION_HEADS = 8  # hidden = 8 * 128 = 1024, fixed to match Qwen embeddings
NUM_KV_HEADS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-layers", type=int, default=33)
    parser.add_argument("--ffn-multiplier", type=float, default=4.0)
    parser.add_argument(
        "--attn-pattern",
        type=int,
        default=4,
        help="full-attention period P: layer i is full attention when i %% P == 0, else SWA (P=4 is the default schedule)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--sliding-window-size", type=int, default=512)
    parser.add_argument("--ffn-multiple-of", type=int, default=256)
    parser.add_argument("--loss-chunk-size", type=int, default=2048)
    parser.add_argument("--warmup-iters", type=int, default=6)
    parser.add_argument("--timed-iters", type=int, default=15)
    return parser.parse_args()


def build_muon_args() -> argparse.Namespace:
    return argparse.Namespace(
        optimizer="muon",
        lr=3e-4,
        weight_decay=0.1,
        muon_lr=None,
        muon_weight_decay=None,
        muon_momentum=0.95,
        muon_ns_steps=5,
        muon_nesterov=True,
        muon_adjust_lr_fn="match_rms_adamw",
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("benchmark is CUDA-only; no GPU is available")

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats()

    config = MykLLMConfig(
        vocab_size=VOCAB_SIZE,
        max_seq_len=args.seq_len,
        head_dim=HEAD_DIM,
        num_kv_heads=NUM_KV_HEADS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_layers=args.num_layers,
        sliding_window_size=args.sliding_window_size,
        ffn_multiplier=args.ffn_multiplier,
        ffn_multiple_of=args.ffn_multiple_of,
        full_attn_period=args.attn_pattern,
        tie_embeddings=True,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    model = MykLLM(config).to(device=device)
    model.train()
    optimizer = T.build_optimizer(model, build_muon_args())

    total_params = model.num_parameters()
    embedding_params = config.vocab_size * config.hidden_size
    params_less_embedding = total_params - embedding_params

    input_ids = torch.randint(0, VOCAB_SIZE, (args.batch_size, args.seq_len), device=device)
    targets = torch.randint(0, VOCAB_SIZE, (args.batch_size, args.seq_len), device=device)

    def run_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss, _ = model(input_ids, targets=targets, loss_chunk_size=args.loss_chunk_size)
        loss.backward()
        optimizer.step()

    pattern = config.attention_pattern
    num_full = sum(1 for layer in pattern if layer == "full_attention")
    header = (
        f"layers={args.num_layers} ffn_mult={args.ffn_multiplier} attn_period={args.attn_pattern} "
        f"(full={num_full}/{args.num_layers}) bs={args.batch_size} seq_len={args.seq_len} "
        f"grad_checkpointing={args.gradient_checkpointing}"
    )

    try:
        for _ in range(args.warmup_iters):
            run_step()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.timed_iters):
            run_step()
        torch.cuda.synchronize()
        seconds_per_step = (time.perf_counter() - start) / args.timed_iters
        tokens_per_sec = args.batch_size * args.seq_len / seconds_per_step
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(header)
        print(
            f"  total_params={total_params / 1e6:.2f}M  params_less_embedding={params_less_embedding / 1e6:.2f}M  "
            f"throughput={tokens_per_sec:,.0f} tok/s  (step={seconds_per_step * 1e3:.0f}ms peak_mem={peak_gb:.2f}GB)"
        )
    except torch.OutOfMemoryError:
        print(header)
        print(
            f"  total_params={total_params / 1e6:.2f}M  params_less_embedding={params_less_embedding / 1e6:.2f}M  "
            f"throughput=OOM"
        )


if __name__ == "__main__":
    main()
