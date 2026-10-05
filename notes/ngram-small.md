# Small vocabulary + hashed n-gram embeddings — small-scale study

Status (2026-10-05): main grid, second seeds, table-size scan and large tables on every
vocabulary done (below). MTP acceptance per vocabulary not measured yet.

Question: can a smaller vocabulary plus hashed n-gram input embeddings (`model.ngram`)
match a large vocabulary? Motivation: a smaller output head decodes faster, and the n-gram
lookup is a sparse gather whose cost does not grow with the table, so the table can be
made large for free at inference. Second question: inject the n-grams at block 0 (added
to the token embedding) or block 1 (after one attention layer)?

## Setup

- Tokenizers: the Qwen3.5 BPE pruned on `data/corpus_{en,zh}.txt` to 64K (the existing
  `Qwen3.5-0.8B-pruned64k`), 32K and 16K (`scripts/prune_tokenizer.py --target-vocab`).
- Data: FineWeb-Edu en + zh streamed by `scripts/pretokenize.py` into
  `data/shards-small/<vocab>/fineweb-edu-{en,zh}` (64K: 191M en + 286M zh tokens; the
  smaller vocabularies proportionally more), 2000 val documents each; mix 0.4 en / 0.6 zh.
- Model (`configs/ngram-small/base.yaml`): hidden 512, 8 layers GL (8 heads x 64, 2 KV
  heads, window 512, RoPE on the first global layer), 30.4M non-embedding parameters,
  tied embeddings, random init (no Qwen warm start: hidden 512). Muon 5e-4, WSD.
- Same text for every vocabulary: 64K trains 3000 steps x 131K tokens (393M tokens),
  32K 3210 steps, 16K 3530 steps (x 1.07 / 1.177 tokens per byte, measured below).
- n-gram: 2- and 3-grams, `num_buckets` rows per order, dim 128, projection to 512.
- Metric: validation bits per byte (bpb) on en and zh, 8 batches each, final step.
- Runner: `scripts/sweep_train.py` (grid over `python -m myk_llm.train`, resumable,
  report). Sweeps in `configs/ngram-small/sweeps/`. ~15-20 min per run on one RTX 5090.

Compression (bytes per token on a corpus sample):

| vocab | en | zh | tokens for the same text |
|---|---|---|---|
| 64K | 4.44 | 4.67 | 1.00 |
| 32K | 4.16 | 4.34 | 1.07 |
| 16K | 3.78 | 3.96 | 1.18 |

## Results: vocabulary x n-gram placement

Mean of en / zh bpb (lower is better); 2 seeds where marked, else 1. Training
throughput in brackets.

| | no n-gram | block 0 | block 1 | blocks 0+1 |
|---|---|---|---|---|
| 64K (342K tok/s) | 1.2088 (2: 1.2091 / 1.2086) | **1.2046** | 1.2053 (2: 1.2049 / 1.2057) | 1.2056 |
| 32K (442K tok/s) | 1.2215 (2: 1.2228 / 1.2203) | 1.2137 | **1.2128** (2: 1.2124 / 1.2131) | 1.2151 |
| 16K (518K tok/s) | 1.2404 (2: 1.2378 / 1.2431) | 1.2315 | **1.2293** (2: 1.2288 / 1.2297) | 1.2332 |

Seed spread: 0.0005-0.005 without n-grams, 0.0007-0.0009 with them.

- **Same text, smaller vocabulary is worse**: +0.013 bpb at 32K, +0.032 at 16K.
- **The n-gram gain grows as the vocabulary shrinks**: 0.0035 at 64K, 0.0087 at 32K,
  0.011 at 16K (block 1, 2 seeds). With the 64K-row table, n-grams recover ~2/3 of the
  32K gap and ~1/3 of the 16K gap.
- **Block 0 vs block 1: within noise.** Block 1 is ahead by 0.001 at 32K and 0.002 at 16K,
  block 0 by 0.0007 at 64K. Both blocks at once is worse than either alone (one seed).
- Smaller vocabularies train faster per token (the fused output head dominates a model
  this small): 16K takes ~22% less time than 64K for the same text.

## Results: n-gram table size (block 1)

Mean bpb; rows per order (2 orders, dim 128); training throughput in brackets. One seed
except the 64K-row column (2 seeds).

| vocab | none | 64K rows (16.8M) | 256K (67M) | 1M (268M) | 4M (1.07B) |
|---|---|---|---|---|---|
| 64K | 1.2088 | 1.2053 | | **1.1919** (335K) | |
| 32K | 1.2215 | 1.2128 | 1.2050 (441K) | 1.1960 (431K) | **1.1881** (392K) |
| 16K | 1.2404 | 1.2293 | | 1.2050 (502K) | |

- **The table keeps paying off**: at 32K every 4x more rows gives -0.008 to -0.009 bpb,
  with no sign of saturation at 4M rows (on 400M training tokens).
- **It narrows the vocabulary gap but does not close it at equal table size**: with 1M
  rows, 32K is 0.004 behind 64K (0.013 without n-grams) and 16K 0.013 behind (0.032).
  The gain from the 1M table grows as the vocabulary shrinks: -0.017 at 64K, -0.026 at
  32K, -0.035 at 16K.
- **A bigger table beats a bigger vocabulary**: 32K + 4M rows (1.1881) is ahead of 64K +
  1M rows (1.1919) and far ahead of 64K without n-grams (1.2088). 64K + 4M rows was not
  run.
- Training cost: AdamW updates the whole table every step (dense gradients and optimizer
  state): -2.5% throughput at 1M rows, -11% at 4M; peak memory 27.8GB at 4M rows (1.07B
  table parameters plus fp32 Adam state) on the 32GB card. Inference cost does not depend
  on the table size (two row gathers per token).

## Decode speed

`scripts/benchmark_decode.py`: random weights, one decode iteration captured as a CUDA
graph over static KV caches (attention switched to `flash_attn_with_kvcache` for the
benchmark; the model's own cache grows with torch.cat and cannot be captured), blocks
torch.compiled, 1024 tokens in the cache. With multi-token prediction (MTP)
speculative decoding of K drafts: verify K + 1 tokens with the main model, then K MTP
modules each run one block + the output head; with per-draft acceptance a an iteration
yields 1 + a + ... + a^K tokens. KB/s = tokens/s x bytes per token (en/zh mix: 4.58 /
4.27 / 3.89).

200M (9 layers GL, hidden 1024), KB/s (higher is better):

| | 64K | 32K | 16K |
|---|---|---|---|
| batch 1, no MTP | 9.4 | 9.6 | 9.2 |
| batch 1, K = 3, a = 0.7 | 11.5 | 14.5 | 14.2 |
| batch 1, K = 3, a = 0.8 | 13.4 | **16.9** | 16.5 |
| batch 1, K = 3, a = 0.9 | 15.6 | **19.7** | 19.2 |
| batch 32, no MTP | 175 | 179 | 169 |
| batch 32, K = 3, a = 0.8 | 233 | 256 | 251 |

ms per iteration at batch 1: 64K 0.489 (K=0) / 1.008 (K=3); 32K 0.444 / 0.747; 16K
0.424 / 0.695. Full tables: `runs/200m-decode-compiled.txt` (also uncompiled:
`runs/200m-decode.txt`, small model: `runs/ngram-small-decode.txt`).

- **Without MTP the vocabulary barely matters per byte (within 4%)**: the head saved per
  token is paid back in more tokens per byte, and the body dominates.
- **With MTP the head runs 2K + 1 times per iteration** (K + 1 verified positions, K
  drafts), so a smaller vocabulary pays off: 32K is 20-26% faster than 64K at K = 3.
  64K at a = 0.9 is still slower than 32K at a = 0.8.
- **16K never beats 32K at equal acceptance** (2-4% slower); it would need a slightly
  higher acceptance rate, which shorter tokens may well give. Acceptance per vocabulary
  is not measured yet (would need MTP-trained models).
- **The n-gram lookup costs 8% latency at batch 1, 4% at batch 32** in the current
  unfused form (hashing, two gathers, projection as separate small kernels); the table
  size itself does not matter. A fused kernel should make it nearly free.
- The 30M small model is launch-bound (0.4-0.5 ms per step whatever the vocabulary), so
  its decode numbers say nothing about vocabulary.

## Reading for the real model

At equal table size 64K is slightly better per byte (0.004 bpb at 1M rows, ~4x the seed
spread), while 32K trains ~20% faster per byte here and, with MTP speculative decoding,
decodes 20-26% faster. Since the table is free at inference and costs only optimizer
memory and time in training, the practical trade is "32K + a larger table" vs "64K":
here 32K + 4M rows beats 64K + 1M rows. 16K is dominated by 32K: worse quality and no
speed gain at equal acceptance. To decide for 200M / 600M: measure MTP acceptance per
vocabulary, and check the quality gap at 200M (the vocabulary is a smaller share there).

## Caveats

- Small model (30M body); the vocabulary's share of parameters and compute is larger
  here than at 200M, so the quality gap between vocabularies may be smaller at 200M.
- Tokenizers were pruned on the same corpus family that the compression was measured on.
- 400M training tokens; large tables may need more data to pay off (or pay off more).
