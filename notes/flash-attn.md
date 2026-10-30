# FlashAttention in Triton

`src/myk_llm/ops/flash_attn.py` replaces the flash-attn package, which has no build for
sm_120 (RTX 5090) with Python 3.14 and CUDA 13. It keeps flash-attn's function names and
arguments, so the call sites only changed their import:

- `flash_attn_func`: dense (batch, seq, heads, head_dim). Used for evaluation, and for decoding with a growing cache.
- `flash_attn_varlen_func`: packed documents with `cu_seqlens`. Used for document-masked training.
- `flash_attn_with_kvcache`: static key/value (KV) cache, forward only, CUDA-graph capturable. Used by `scripts/benchmark_decode.py`.

It supports causal or bidirectional attention, a sliding window, grouped-query attention
(GQA), bf16/fp16 inputs, and head_dim in {16, 32, 64, 128, 256}. The masking matches
flash-attn: queries are aligned to the end of the keys. Measured 2026-10-30 on one RTX
5090 with torch 2.15 + CUDA 13.2 and Triton 3.9.

## Design

- **Forward.** One program handles one query block of one (sequence, head). It runs an
  online softmax in fp32 with `exp2`, and saves a base-2 log-sum-exp for the backward pass.
  A query block's key range is split three ways: masked edge blocks on the left (where the
  window starts), unmasked full blocks in the middle, and masked edge blocks on the right
  (the causal diagonal and the end of the sequence). Blocks outside a sliding window are
  never visited.
- **Backward.** Two kernels, with no atomics, so the result is deterministic:
  - A dQ kernel, one program per query block. It also computes delta = rowsum(dO · O),
    which saves a separate preprocessing launch.
  - A dK/dV kernel, one program per key block. It loops over every query head of its GQA
    group, so dK/dV are written once in the (batch, seq, kv_heads, head_dim) layout.

  This costs 7 tile matmuls per (query, key) block pair; flash-attn's fp32 atomic-add dQ
  costs 5. A fused atomic-dQ variant was only 5–10% faster on causal 2K and was slower with
  a window, so it was not kept.
- **Tiles**, swept on the 5090:

  | kernel | head_dim | tile | warps | stages |
  |---|---|---|---|---|
  | forward | 128 | 128×32 | 8 | 3 |
  | forward | ≤ 64 | 64×32 | 4 | 3 |
  | backward | all | 32×32 | 4 | 2 |
  | decode (≤ 16 queries) | all | 16×64 | | |

  In the backward pass, larger tiles were slower: sm_120 has 99 KB of shared memory per
  block and runs `mma.sync`.
- **Wrapper: `torch.autograd.Function`, not `torch.library.custom_op`.** Dynamo traces
  through the Function and launches the Triton kernels from the compiled graph, with
  `fullgraph=True` and no graph break. The custom-op dispatch costs more than the kernels
  at small sizes. Forward + backward of a tiny attention block:

  | wrapper | eager | compiled |
  |---|---|---|
  | custom_op | 244 µs | 301 µs |
  | autograd.Function | 179 µs | 157 µs |
  | SDPA | 107 µs | 207 µs |

## Speed

From `python scripts/benchmark_attention.py`. TFLOPS count visible (query, key) pairs only,
and the backward pass is counted as 2.5× the forward. Times are fwd / fwd+bwd in ms.

| shape (bf16) | ours | SDPA flash | SDPA cuDNN | FlexAttention |
|---|---|---|---|---|
| 8×2048, 8/2 heads ×128, causal | **0.385** / 1.706 | 0.404 / **1.431** | 0.367 / 1.601 | 0.412 / 2.448 |
| 8×2048, window 512 | **0.198 / 0.783** | — | — | 0.212 / 1.268 |
| 8×2048, 4 packed documents | **0.199 / 0.804** | — | — | 0.211 / 1.324 |
| 2×8192, causal | 1.387 / 6.035 | 1.524 / **4.848** | **1.370** / 5.184 | 1.528 / 8.097 |
| 16×1024, 4/4 heads ×64, causal | **0.058** / 0.236 | 0.066 / **0.226** | 0.080 / 0.281 | 0.070 / 0.264 |
| 16×1024, window 256 | **0.029 / 0.113** | — | — | 0.045 / 0.170 |

- The forward pass is level with the best SDPA backend: 179–198 TFLOPS at head_dim 128.
- Forward + backward reaches 84% of SDPA's flash backend at 2K causal and 80% at 8K. The
  extra dQ recomputation is the gap; see the backward-pass design above.
- With sliding windows and document masks, which SDPA cannot express without a dense mask,
  ours is 1.5–1.7× faster than FlexAttention for forward + backward.

## End to end

- **Training.** `scripts/benchmark_throughput.py configs/experiments/ablation.yaml
  data.micro_batch_size=16` (200M, GLGLGLGLG): **123.7K tok/s, 14.8 GB peak**. flash-attn
  measured 121.7K tok/s and 14.8 GB (`fp8.md`), on the older torch 2.11.
- **Decoding.** `scripts/benchmark_decode.py configs/ngram-small/v64k.yaml`, eager, ms per
  iteration (old flash-attn numbers from `runs/ngram-small-decode.txt`):

  | batch, drafts K | ours | flash-attn |
  |---|---|---|
  | 1, 0 | 0.455 | 0.470 |
  | 1, 2 | 0.816 | 0.797 |
  | 32, 0 | 0.638 | 0.711 |
  | 32, 2 | 0.990 | 1.179 |

  With `--compile`: 0.249 / 0.471 / 0.346 / 0.588. The static-cache check, max |Δlogit|
  between a full forward and prefill + one cached step, is 0.016–0.021, the same as before.

## Not done

- A faster backward, i.e. closing the 15–20% gap to the CUDA kernel. Options: atomic dQ
  with wider key tiles, or a TMA (Tensor Memory Accelerator) / warp-specialized variant.
- Split-KV decoding. A single long sequence decodes with only `heads` programs.
- head_dim must be a power of two.
