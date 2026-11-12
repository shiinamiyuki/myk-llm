# Parallel blocks

Status (2026-11-12): implemented (`model.parallel_block`); small-LM sweep done
(`configs/ngram-small/sweeps/parallel.yaml`, `parallel-seeds.yaml`).

Question: (1) how much language-modeling quality does a parallel block lose against the
sequential one, for a sliding-window hybrid (LLLG) and a Gated DeltaNet hybrid (DDDG)?
(2) Does looping the stack 2 or 3 times (notes/loop.md) win it back?

Summary:
- **The parallel block costs bits per byte (bpb)**: +0.011 for LLLG, +0.005 for DDDG
  (2 seeds each; per-arm seed spread <= 0.0016). The LLLG cost is about what halving
  the vocabulary costs on this testbed (+0.013, notes/ngram-small.md). It is the same
  from step 1500 to the end, not a slow start.
- **One extra loop iteration wins it back**: looping the whole 8-block stack twice cuts
  the parallel/sequential gap to +0.0017 (LLLG) and +0.0002 (DDDG), and 3 times to
  +0.0021 / -0.0009 (1 seed each), within seed noise. The parallel block gains 3-5x more
  from the first extra iteration than the sequential one does (LLLG -0.0125 vs -0.0029,
  DDDG -0.0057 vs -0.0011); after it, both gain about the same per iteration. Parallel
  r = 2 also beats sequential r = 1 (by 0.001), parallel r = 3 by 0.004 / 0.0065.
- **It is not a good trade for speed**: unfused, the parallel block runs at the same
  speed as the sequential one here, and r = 2 costs 1.5x the training time (r = 3: 2x),
  so at equal cost the sequential block is never worse. The reading is the other way
  round: in a looped model the parallel block is about free, so a looped model can take
  its speedups (fused input projections, attention and FFN at the same time, less
  communication under tensor parallelism) at no visible quality cost.

## Model (`ModelConfig.parallel_block`)

```
sequential (default):  x = x + mixer(norm1(x));  x = x + ffn(norm2(x))
parallel:              h = norm(x);  x = x + mixer(h) + ffn(h)
```

GPT-J / PaLM style: one RMSNorm per block (no FFN norm); residual gates (`model.residual`)
apply to each update as before. Applies to every block (mixer type, looped core,
cross-decoder, MTP module). The input projections of mixer and FFN are not fused, so
there is no speedup yet: identical tokens/s at this size, 0.5-2 GB less activation
memory.

## Setup

- Testbed: the small LM of notes/ngram-small.md (hidden 512, 8 blocks, 8 heads x 64 with
  2 KV heads, window 512, RoPE on the first global layer, 64K vocabulary, ~393M
  FineWeb-Edu en + zh tokens, 3000 steps, Muon, WSD). Metric: mean of en / zh
  validation bpb at the end of training.
- Patterns tile over the 8 blocks: LLLGLLLG (30.4M non-embedding parameters) and
  DDDGDDDG (34.4M; Gated DeltaNet 4 heads x 128, short conv, output gate).
- Loop: `loop.core=8`, the whole stack is the core (no prelude / coda), add injection
  of the embedding every iteration, fixed r in training and evaluation, residual init
  scaled for the effective depth (16 / 24 blocks). Same parameters, r x the block compute.
- Looped arms train with micro-batches of 16 (memory); the global batch and the data
  stream are unchanged. They evaluate with `eval.batches=16` (see the pitfall below).
- Seeds: r = 1 arms seeds 0 and 1, looped arms seed 0.

## Results

Mean bpb (en / zh in the last columns); r = 1 is the mean of 2 seeds (per seed in
brackets). Throughput is training tokens/s on one RTX 5090.

| pattern | block | r = 1 | r = 2 | r = 3 | tok/s (r = 1 / 2 / 3) |
|---|---|---|---|---|---|
| LLLG | seq | 1.2084 (1.2091 / 1.2076) | 1.2055 | 1.2023 | 346K / 231K / 172K |
| LLLG | par | 1.2197 (1.2189 / 1.2205) | 1.2072 | 1.2044 | 346K / 231K / 173K |
| | par - seq | **+0.0113** | +0.0017 | +0.0021 | |
| DDDG | seq | 1.2012 (1.2014 / 1.2011) | 1.2001 | 1.1956 | 280K / 181K / 132K |
| DDDG | par | 1.2060 (1.2063 / 1.2056) | 1.2003 | 1.1947 | 280K / 181K / 133K |
| | par - seq | **+0.0047** | +0.0002 | -0.0009 | |

en / zh bpb:

| arm | r = 1 | r = 2 | r = 3 |
|---|---|---|---|
| LLLG seq | 1.2025 / 1.2142 | 1.2002 / 1.2108 | 1.1989 / 1.2057 |
| LLLG par | 1.2125 / 1.2270 | 1.2016 / 1.2128 | 1.1991 / 1.2097 |
| DDDG seq | 1.1953 / 1.2071 | 1.1951 / 1.2051 | 1.1914 / 1.1998 |
| DDDG par | 1.2001 / 1.2118 | 1.1954 / 1.2052 | 1.1904 / 1.1991 |

par - seq during training (seed 0, mean bpb; before step 3000 the looped arms are on
the 128-window set, the same for both blocks):

| arm | 500 | 1000 | 1500 | 2000 | 2750 | 3000 |
|---|---|---|---|---|---|---|
| LLLG r = 1 | +0.0137 | +0.0166 | +0.0077 | +0.0081 | +0.0093 | +0.0099 |
| LLLG r = 2 | +0.0022 | +0.0099 | -0.0011 | +0.0003 | +0.0016 | +0.0017 |
| DDDG r = 1 | +0.0013 | +0.0077 | +0.0040 | +0.0029 | +0.0047 | +0.0049 |
| DDDG r = 2 | -0.0036 | +0.0019 | -0.0005 | -0.0014 | -0.0005 | +0.0002 |

Readings:
- **The loss is real and pattern-dependent**: both seeds agree (LLLG +0.0098 / +0.0129,
  DDDG +0.0049 / +0.0045), on en and zh, and the gap is flat from step 1500. DDDG loses
  half as much as LLLG; why (the DeltaNet mixer's own output gate and short conv, its
  2x mixer parameters, or a stronger mixer in general) is untested.
- **It is a depth effect**: a parallel block's FFN cannot read its own block's mixer
  output, so the stack has 8 serial steps instead of 16 sublayers. One more iteration
  (16 serial steps, like the sequential r = 1) removes almost the whole loss, and from
  then on the two blocks improve at the same rate (r = 2 -> 3: LLLG -0.0028 vs -0.0032,
  DDDG -0.0056 vs -0.0045). The sequential stack itself gains little from its first
  extra iteration on DDDG (-0.0011) and more from the second (-0.0045).
- **Caveats**: r = 2 / 3 are one seed; their gaps (<= 0.0021) are within seed noise
  (<= 0.0016 per arm here, up to 0.005 on this testbed before). Only one scale (30M
  non-embedding, 393M tokens), only whole-stack loops and fixed r.

## Pitfall: the validation set depends on the micro-batch size

`eval.batches` counts batches of `data.micro_batch_size` rows, so the looped arms
(micro-batch 16) were first scored on 128 windows per language instead of 256. On that
set the sequential r = 2 arms looked worse than r = 1 on en (+0.002 / +0.005) and much
better on zh (-0.010 / -0.008), a split that disappears on the full set. The final checkpoints were rescored on the 256-window set (rerunning a
finished run validates its final checkpoint); 16 x 16 rows reproduce 8 x 32 exactly
(checked on an r = 1 checkpoint). The original r = 2 scores are kept as
`results-eval128.json` / `train-original.log` in `runs/sweeps/parallel-lm/`. Keep
`eval.batches x micro_batch_size` fixed across a sweep.

Next: fuse the mixer and FFN input projections to measure the parallel block's actual
speedup; whether a partial loop (e.g. a middle core with prelude / coda) also closes the
gap; the lc chain task (notes/loop.md), where serial depth decides the hop reach.
