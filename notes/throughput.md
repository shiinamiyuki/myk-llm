# Training throughput — profiling notes

> **Note (2026-09):** measured with the old `myk-llm/training.py`; the refactored trainer matches its throughput (200M, 32x2048: ~128K tok/s). Re-measure with `scripts/benchmark_throughput.py`.

Session goal: optimize `myk-llm/training.py` throughput for the planned base
run — **pruned 64K tokenizer** (`_download_models/Qwen3.5-0.8B-pruned64k`,
`vocab=65536`), **seq_len=2048**. Profiled on the target hardware below.

## Setup
- **GPU:** RTX 5090, 32 GB, compute 12.0 (Blackwell). torch 2.11+cu130, bf16.
- **Model:** default config — 25 layers, 8 heads × 128 head_dim (hidden 1024),
  2 KV heads, SWA window 512, tied embeddings → **447M params**.
- **Bench config:** seq_len 2048, batch_size 8 (→ 16384 tokens/step),
  grad_accum 1, `torch.autocast(bf16)`, fp32 master weights.
- Data: `data/input.txt` tokenized with the pruned 64K tokenizer
  (`.cache/tokenized/input64k`, 351K tokens) — enough to exercise the pipeline.

## Baseline
End-to-end (real loop, Muon, real data): **2.70 it/s ≈ 44.2K tokens/sec**.

Per-step breakdown (~370ms), measured by syncing between phases:

| phase | time | note |
|---|---|---|
| forward | ~139ms | flash-attn + fused linear-CE |
| backward | ~158ms | |
| **Muon `opt.step`** | **~69ms** | ~19% of the step |
| `get_batch` | 0.1ms | memmap, cached — **not** a bottleneck |

Compute is already healthy: fwd+bwd ≈ 6·P·T / time ≈ **~70% of the 5090's bf16
peak** (MFU). This is not a "wildly underutilized GPU" situation.

## Change applied: `--loss-chunk-size` default 256 → 2048
The fused linear-CE (`fused_linear_cross_entropy`) chunks tokens and runs one
GEMM per chunk. At chunk 256 that's 64 tiny GEMMs/step; bigger chunks = bigger
GEMMs = better tensor-core utilization, **numerically identical** loss (chunking
is along the token/row axis, not vocab — each row's full softmax over V is
unchanged; only the fp32 `grad_weight` accumulation order differs).

Isolated sweep (bs=8, seq 2048, AdamW to isolate model from optimizer):

| chunk | tok/s | peak mem |
|---|---|---|
| 256 | 51.0K | 24.10 GB |
| 512 | 53.5K | 24.13 GB |
| 1024 | 54.8K | 24.20 GB |
| **2048** | **55.5K** | **24.47 GB** |
| 4096 | 55.6K | 25.01 GB |
| 8192 | 56.2K | 26.09 GB |
| 16384 | 56.2K | 26.25 GB |

Knee at **2048**: +8.6% over 256 for +0.4 GB; ~flat after. Chosen as the new
default (`training.py`, `--loss-chunk-size`).

**End-to-end validation** (real loop, Muon, real data): **2.70 → 2.91 it/s
(+7.8%)**, ≈ 44.2K → 47.7K tokens/sec. No change to training dynamics.

## torch.compile state — leave as is
Per-block `@torch.compile` (on `RMSNorm`, `SwiGLU`, `TransformerBlock`) is
already a big win and should stay:

| config | tok/s | peak mem |
|---|---|---|
| eager (compile disabled) | 38.5K | 32.04 GB |
| **per-block compile (current)** | **50.9K** | **24.10 GB** |

i.e. +32% tok/s and −8 GB vs eager. The per-block compile also **beats** a
single full-model `torch.compile(model)` (24.5 GB vs 26.0 GB, same speed), and
stacking an *outer* compile on top of the baked-in decorators just OOMs
(double-compile). No restructuring worth doing.

## CUDA graphs / `torch.compile(mode="reduce-overhead")` — does NOT help
`reduce-overhead` *is* CUDA graphs under the hood. Tested on both paths:

| path | base | reduce-overhead | verdict |
|---|---|---|---|
| fwd/bwd | 294.8ms / 55.6K | 294.0ms / 55.7K, **+1.5 GB** | wash, costs memory |
| Muon step | 68.9ms | 67.5ms (~2%) | wash |

**Why it doesn't apply.** CUDA graphs only win when CPU *launch overhead*
dominates — many tiny kernels finishing faster than the host can queue them
(small batch/seq, or autoregressive decode). At seq 2048 × bs 8 we're not there:

- **fwd/bwd** is compute-bound dense GEMMs — no idle launch gap to collapse.
  Also `flash_attn_func` forces graph breaks and the fused-CE's
  `n_valid = (targets != ignore_index).sum().item()` is a hard host sync, so a
  single clean capture isn't even possible; reduce-overhead can only wrap
  fragments and adds ~1.5 GB of static graph pools — actively bad since we're
  **memory-bound** (bs=16 already OOMs).
- **Muon** is *not* launch-bound (my initial guess was wrong — the reduce-overhead
  test refutes it). Its 69ms is real FLOPs: Newton-Schulz orthogonalization is
  ~10 TFLOP of `G@Gᵀ` + two `addmm`s × 5 steps across ~100 weight matrices
  (`torch.optim._muon`, already runs in bf16). Compute, not launches.

Where graphs *would* pay: the `generate()` decode loop (single-token steps, tiny
launch-bound kernels) — that's inference latency, not training throughput.

## Remaining levers (not pulled this session)
- **Muon ~19% of the step is genuine compute.** Only real knobs:
  - fewer `--muon-ns-steps` (5 → 3): algorithmic, changes update quality.
  - `--grad-accum-steps`: Muon runs once per *optimizer* step, so accum=2 nearly
    halves its per-token cost — but it changes effective batch size, so it's a
    training-config decision, not a free throughput win.
- **Batch size is memory-bound**, not compute-bound: bs=8 peaks ~24 GB, bs=16
  OOMs. And since we're already ~70% MFU, a bigger batch wouldn't raise tok/s
  much anyway — it would only amortize the fixed Muon cost (same as grad-accum).
- torch's `Muon` has no `foreach`/fused path; a custom batched Muon could help
  but is out of scope (rewriting the optimizer, touches dynamics).

## Model-size exploration (hidden stays 1024 for embedding reuse)
Constraint: `hidden_size = num_attention_heads · head_dim = 8 · 128 = 1024` must
match Qwen for embedding reuse, so we only scale **layers** or **ffn_multiplier**.
`num_layers` must follow `1 + 4N` for the default attention schedule
(25, 29, 33, 37, …). Same bench: seq 2048, Muon, chunk 2048.

**Without gradient checkpointing** (bs=8 is the memory limit):

| config | params | tok/s | peak | note |
|---|---|---|---|---|
| L25 ffn4 (old default) | 447M | 48.2K | 23.0 GB | baseline |
| L29 ffn4 | 508M | 42.2K | 26.3 GB | comfortable headroom |
| **L33 ffn4 (new default)** | **569M** | **37.5K** | **29.7 GB** | at the ceiling |
| L25 ffn6 | 605M | 37.3K | 29.6 GB | at the ceiling |
| L37 ffn4 (630M) / L25 ffn8 (762M) | — | — | **OOM** | |

**With `--gradient-checkpointing`** (unlocks bigger models + large batch):

| config | params | bs | tok/s | peak |
|---|---|---|---|---|
| L33 ffn4 | 569M | 8 | 30.5K | 8.4 GB |
| L33 ffn4 | 569M | 24 | 34.4K | 14.6 GB |
| L33 ffn4 | 569M | 40 | 35.5K | 20.9 GB |
| L37 ffn4 | 630M | 24 | 30.9K | 15.9 GB |
| L25 ffn8 | 762M | 16 | 26.4K | 13.4 GB |

Findings:
- **Depth ≈ width at equal budget.** L33-ffn4 (569M) and L25-ffn6 (605M) tie on
  both throughput (37.5K vs 37.3K) and memory (~29.6 GB). The choice is on quality
  grounds — for sub-1B LLMs, **adding layers beats widening ffn** (keeps
  `ff_hidden` modest), so we scale depth.
- **No-gc ceiling is ~570–600M at bs=8**, but those sit right at 30 GB — too tight
  for a real run (one fragmentation spike OOMs).
- **gradient_checkpointing is the enabler past ~0.6B.** ~15–18% throughput tax
  (recompute), 3–4× memory cut (L33: 29.7→8.4 GB). It never beats the same-size
  no-gc config on raw tok/s; its value is headroom and reach.

**Decision:** new default `--num-layers 33` (ffn stays 4.0) → **569M**, the
~0.6B target. `num_layers = 33 = 1 + 4·8`, embedding reuse unaffected.
Recommended real-run config (safe memory + larger effective batch):
`--gradient-checkpointing --batch-size 32` (or 40) → ~35K tok/s at ~21 GB.
For full-speed / no gc, bs=8 works but runs at the 30 GB edge.
A conservative alternative is L29 (508M): 42.2K tok/s, 26.3 GB, no gc.
