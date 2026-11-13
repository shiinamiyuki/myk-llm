"""Prefill time and inference-cache size of model variants (random weights, one GPU).

The prefill is generate()'s: the whole prompt in one pass with `use_cache=True,
keep_last=1`, so a decoder-decoder (model.cross, "X" blocks) runs its cross-decoder on
the last position only. "full" is the same pass over every position (a training-style
forward, no cache), for comparison. Blocks run eagerly, as in generate(). The cache size
counts every tensor the cache holds (attention keys / values, DeltaNet recurrent and
convolution states) after the prefill.

Any config with a `model:` section works (pretraining or lc); each `--variants` entry is
a comma-separated list of extra overrides, one table row per variant and length:

    python scripts/benchmark_prefill.py configs/lc/base.yaml model.attention.global_rope=none \
        --variants model.layer_pattern=LLLLXXXX model.layer_pattern=LLLG --lengths 8192 65536
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import torch

from myk_llm.configuration import apply_overrides, from_dict, load_yaml_tree
from myk_llm.model import MykLLM, ModelConfig


def tensor_bytes(entry: Any) -> int:
    if isinstance(entry, torch.Tensor):
        return entry.numel() * entry.element_size()
    if isinstance(entry, (list, tuple)):
        return sum(tensor_bytes(e) for e in entry)
    if isinstance(entry, dict):
        return sum(tensor_bytes(e) for e in entry.values())
    if entry is None:
        return 0
    return sum(tensor_bytes(v) for v in vars(entry).values())


def timed(fn, repeats: int) -> float:
    """Median milliseconds of `fn()` over `repeats` calls, after one warm-up call."""
    fn()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - start))
    return sorted(times)[len(times) // 2]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*", help="key.path=value, applied to every variant")
    parser.add_argument("--variants", nargs="+", default=[""], help="comma-separated overrides per variant")
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384, 65536])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    tree = apply_overrides(load_yaml_tree(args.config), args.overrides)
    torch.set_float32_matmul_precision("high")

    print("| variant | layers | length | prefill ms | full ms | cache MB |")
    print("|---|---|---|---|---|---|")
    for variant in args.variants:
        extra = [o for o in variant.split(",") if o]
        config = from_dict(ModelConfig, apply_overrides(tree, extra)["model"], "model")
        config.max_seq_len = max(config.max_seq_len, *args.lengths)
        model = MykLLM(config).cuda().to(torch.bfloat16).eval()
        for length in args.lengths:
            ids = torch.randint(0, config.vocab_size, (args.batch, length), device="cuda")
            prefill = timed(lambda: model(ids, use_cache=True, keep_last=1), args.repeats)
            full = timed(lambda: model(ids), args.repeats)
            cache = model(ids, use_cache=True, keep_last=1).cache
            megabytes = sum(tensor_bytes(entry) for entry in cache) / 2**20
            label = ",".join(o.removeprefix("model.") for o in extra) or "-"
            print(f"| {label} | {config.layout} | {length} | {prefill:.1f} | {full:.1f} | {megabytes:.1f} |", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
