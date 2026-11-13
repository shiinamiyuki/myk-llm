# Parallel blocks

Status (2026-11-13): implemented (`model.parallel_block`, `model.parallel_streams`);
small-LM sweep done (`configs/ngram-small/sweeps/parallel.yaml`, `parallel-seeds.yaml`);
multi-hop chain sweeps done (`configs/lc/sweeps/parallel-chain.yaml`,
`parallel-chain-seeds.yaml`, `parallel-chain-cot.yaml`); speed measured (training and
inference, one RTX 5090).

Question: (1) how much language-modeling quality does a parallel block lose against the
sequential one, for a sliding-window hybrid (LLLG) and a Gated DeltaNet hybrid (DDDG)?
(2) Does looping the stack 2 or 3 times (notes/loop.md) win it back? Then both
questions again on direct multi-hop chains, which need serial depth.

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
- **It is not a good trade for training speed**: the parallel block trains at the
  sequential block's speed (fusing its input matmuls does not help on one GPU, see
  Speed), and r = 2 costs 1.5x the training time (r = 3: 2x), so at equal cost the
  sequential block is never worse. The reading is the other way round: in a looped
  model the parallel block is about free, so a looped model can take its inference
  speedup (decoding 8-16% faster with the FFN on a second CUDA stream) and, under tensor
  parallelism, its saved communication at no visible quality cost.
- **Speed**: fusing the mixer's and the FFN's input matmuls (general matrix multiply
  (GEMM) fusion) cannot speed up training on one GPU (the matmuls are already at peak:
  < 0.5% to gain) and made it 5-7.5% slower (the backward concatenates the gradients),
  so it was dropped. Running the FFN on a second CUDA stream (`model.parallel_streams`)
  makes CUDA-graph decoding 8-16% faster than one stream from 30M to 1.5B non-embedding
  parameters (1% for DDDG at batch 32, hidden 2048), but eager, launch-bound decoding
  10-20% slower, so it is off by default.
- **Multi-hop chains tell the same story through seed noise**: unlooped, the parallel
  block learned direct 1-4-hop answers within 24000 steps in 1 of 8 runs against 3 of 8
  sequential runs (not significant; mean 2-4-hop accuracy about equal). Looped, every
  run of both blocks learns: r = 2 at 7500-18000 steps (4 hops 78-99%), r = 3 at
  3500-9000 (95-100%), and the two blocks cannot be told apart. A fixed loop does not
  extrapolate hops (5 hops 21-75%, 6 hops mostly at chance).
- **With chain-of-thought (CoT) answers the parallel block costs nothing, unlooped**:
  writing each intermediate key turns a k-hop chain into k one-hop lookups, so serial
  depth no longer matters. Parallel and sequential learn at the same step (LLLG 1500,
  DDDG 1000, both seeds), score 100% on 1-4 hops to 16K, and both extrapolate to 8 hops
  (97-100% at 2K and 8K). Unlooped CoT beats every looped direct-answer run, in
  1000-1500 steps against 3500-18000.

## Model (`ModelConfig.parallel_block`)

```
sequential (default):  x = x + mixer(norm1(x));  x = x + ffn(norm2(x))
parallel:              h = norm(x);  x = x + mixer(h) + ffn(h)
```

GPT-J / PaLM style: one RMSNorm per block (no FFN norm); residual gates (`model.residual`)
apply to each update as before. Applies to every block (mixer type, looped core,
cross-decoder, MTP module). Same training tokens/s as the sequential block, 0.5-2 GB
less activation memory; at inference `model.parallel_streams` runs the FFN on a second
CUDA stream (see Speed).

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

## Multi-hop chains (2026-11-13)

`configs/lc/sweeps/parallel-chain.yaml` (r = 1 seeds 2-3: `parallel-chain-seeds.yaml`):
does the parallel block cost direct multi-hop answers, which depend on serial depth
(notes/lc-retrieval.md E3, notes/loop.md), and does the loop repair that too? E3b's
direct chain setting: 8 chains of 1-4 hops in noise, 4 queried, answers without
intermediate keys, trained on 256-2048 tokens, NoPE global layers, 24000 steps, on the
lc testbed (hidden 256, 8 blocks; 8.4M / 8.8M non-embedding parameters for LLLG / DDDG).
The loop is the one above (whole stack, add injection, fixed r). The final scan
evaluates 1-6 hops at 2K and 8K; chance is ~15-20%. Minutes per run (parallel =
sequential): LLLG 16 / 31 / 45, DDDG 24 / 44 / 65 for r = 1 / 2 / 3. Full tables:
`runs/lc/parallel-chain/summary.md`.

Final accuracy at 2K per hop count (1 / 2 / 3 / 4) per seed; "learned" = first
evaluation at >= 90% overall:

| arm | r | seed 0 | seed 1 | learned |
|---|---|---|---|---|
| LLLG seq | 2 | 100 / 100 / 98 / 88 | 100 / 100 / 100 / 99 | 14000, 8000 |
| LLLG par | 2 | 100 / 100 / 99 / 97 | 100 / 100 / 100 / 97 | 12000, 7500 |
| DDDG seq | 2 | 100 / 98 / 96 / 78 | 100 / 99 / 96 / 90 | 18000, 11500 |
| DDDG par | 2 | 100 / 99 / 94 / 89 | 100 / 99 / 97 / 93 | 13000, 13000 |
| LLLG seq | 3 | 100 / 100 / 99 / 96 | 100 / 100 / 100 / 95 | 3500, 3500 |
| LLLG par | 3 | 100 / 100 / 100 / 99 | 100 / 100 / 100 / 100 | 3500, 8500 |
| DDDG seq | 3 | 100 / 99 / 98 / 99 | 100 / 100 / 99 / 97 | 5500, 7000 |
| DDDG par | 3 | 100 / 99 / 99 / 96 | 100 / 100 / 98 / 98 | 8000, 9000 |

r = 1, 4 seeds:

| arm | seed 0 | seed 1 | seed 2 | seed 3 | learned |
|---|---|---|---|---|---|
| LLLG seq | 100 / 17 / 18 / 23 | 100 / 97 / 92 / 70 | 100 / 55 / 20 / 19 | 100 / 97 / 95 / 73 | -, 17500, -, 12500 |
| LLLG par | 100 / 95 / 94 / 66 | 100 / 92 / 71 / 32 | 100 / 88 / 82 / 48 | 100 / 18 / 20 / 19 | 14500, -, -, - |
| DDDG seq | 99 / 15 / 20 / 17 | 100 / 98 / 93 / 71 | 99 / 89 / 61 / 35 | 100 / 13 / 17 / 18 | -, 22000, -, - |
| DDDG par | 100 / 60 / 23 / 20 | 100 / 53 / 16 / 16 | 99 / 83 / 66 / 30 | 100 / 59 / 22 / 23 | never |

Final scan at 2K, mean of the seeds (8K is at most 10 points lower). The
scan contexts also hold 5-6-hop chains, so 1-4 hops score a few points below the
tables above:

| arm | r | 1 | 2 | 3 | 4 | 5 | 6 | accuracy at 16K |
|---|---|---|---|---|---|---|---|---|
| LLLG seq | 1 | 100 | 64 | 56 | 44 | 26 | 19 | 59 |
| LLLG par | 1 | 100 | 70 | 65 | 37 | 26 | 18 | 59 |
| DDDG seq | 1 | 99 | 53 | 44 | 28 | 20 | 16 | 53 |
| DDDG par | 1 | 100 | 59 | 27 | 20 | 19 | 17 | 49 |
| LLLG seq | 2 | 100 | 98 | 99 | 94 | 52 | 21 | 94 |
| LLLG par | 2 | 100 | 100 | 100 | 98 | 63 | 25 | 95 |
| DDDG seq | 2 | 100 | 99 | 93 | 82 | 30 | 18 | 92 |
| DDDG par | 2 | 100 | 100 | 93 | 89 | 38 | 22 | 94 |
| LLLG seq | 3 | 100 | 100 | 100 | 96 | 56 | 29 | 96 |
| LLLG par | 3 | 100 | 100 | 100 | 98 | 58 | 22 | 97 |
| DDDG seq | 3 | 100 | 99 | 97 | 96 | 43 | 18 | 97 |
| DDDG par | 3 | 100 | 98 | 97 | 93 | 43 | 36 | 97 |

Readings:
- **Unlooped, the parallel block may learn more slowly; the seeds decide**: within
  24000 steps 3 of 8 sequential runs pass 90% (LLLG 2 of 4, DDDG 1 of 4) against 1 of 8
  parallel runs (LLLG 1, DDDG 0); 3 vs 1 of 8 is not significant (Fisher p = 0.57). Mean
  accuracy on 2-4 hops (tables above) is about equal: LLLG 60 parallel vs 56
  sequential, DDDG 39 vs 46. Most runs are mid-transition at 24000 steps (e.g. LLLG par
  seeds 1-2: 88-92% on 2 hops, 32-48% on 4), so r = 1 measures learning speed at the
  edge of the budget. DDDG par is the only arm with no failed seed (2 hops 53-83% in
  all 4) and no learned one.
- **The loop makes the task learnable and then fast, for both blocks**: at r = 2 all 8
  runs learn (7500-18000 steps) and reach 78-99% on 4 hops; at r = 3 they learn at
  3500-9000 steps and reach 95-100%. The learned step falls with r for every arm.
- **After the loop the parallel block costs nothing**: final accuracy is the same
  (4 hops at r = 2: parallel 89-97, sequential 78-99; at r = 3: 96-100 vs 95-99), and
  learning speed goes either way with 2 seeds: at r = 2 the parallel block learns first
  in 3 of 4 seed pairs, at r = 3 the sequential block learns first or ties in all 4
  (DDDG by 2000-2500 steps). Parallel r = 2 (16 serial sublayers, like sequential r = 1,
  at 2x its compute) learns in 4 of 4 runs, sequential r = 1 in 3 of 8.
- **Little hop extrapolation from a fixed r**: 5 hops reach 21-75% per run with a large
  seed spread (DDDG par r = 3: 21 / 65) and no ordering between the blocks; r = 3 adds
  5-13 points over r = 2 on DDDG and -5 to +4 on LLLG. 6 hops stay at <= 27% except
  DDDG par r = 3 seed 1 (45 / 49% at 2K / 8K) and LLLG seq r = 3 seed 0 (36 / 21%).
  Same as notes/loop.md: with a fixed r the loop is a weight-tied deep stack;
  extrapolating hops took random r in training.
- **Length**: from r = 2 every arm keeps 92-97% to 16K (late accuracy); at r = 1 the
  learned runs lose up to 20 points on 4 hops from 2K to 8K (LLLG par seed 0: 67 -> 47).
- Caveats: 2 seeds for r = 2 / 3 (4 for r = 1), learning at r = 1 is seed-dependent
  and unfinished at 24000 steps, one model size, 1-4 training hops only.

### With chain-of-thought answers

`configs/lc/sweeps/parallel-chain-cot.yaml`: the same chains, but the answer writes every
intermediate key before the value (E3's CoT setting: `task.cot`, NoPE global layers, 8000
steps), unlooped, 2 seeds. Each written key is one single-hop lookup, so the question is
only whether the parallel block learns the lookup-and-copy as fast. The final scan
evaluates 1-8 hops (training has 1-4); evaluation is teacher-forced exact match on the
whole answer, which equals greedy decoding. Minutes per run (parallel = sequential):
LLLG 5.6, DDDG 8.0. Full tables: `runs/lc/parallel-chain-cot/summary.md`.

Final scan per seed, accuracy for 1 / 2 / ... / 8 hops; "learned" = first evaluation at
>= 90% overall:

| arm | seed | 2K | 8K | learned |
|---|---|---|---|---|
| LLLG seq | 0 | 100 / 100 / 100 / 99 / 100 / 100 / 99 / 98 | 100 / 100 / 100 / 100 / 98 / 100 / 100 / 99 | 1500 |
| LLLG seq | 1 | 100 / 100 / 100 / 99 / 100 / 100 / 100 / 98 | 100 / 100 / 100 / 99 / 99 / 100 / 98 / 100 | 1500 |
| LLLG par | 0 | 100 / 100 / 100 / 100 / 100 / 100 / 100 / 97 | 100 / 100 / 100 / 98 / 100 / 100 / 99 / 100 | 1500 |
| LLLG par | 1 | 100 / 99 / 100 / 100 / 100 / 100 / 99 / 97 | 100 / 100 / 100 / 100 / 100 / 100 / 99 / 99 | 1500 |
| DDDG seq | 0 | 100 / 100 / 100 / 100 / 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 / 100 / 99 / 100 / 100 | 1000 |
| DDDG seq | 1 | 100 / 100 / 100 / 100 / 100 / 100 / 100 / 97 | 100 / 100 / 100 / 100 / 100 / 99 / 98 / 100 | 1000 |
| DDDG par | 0 | 100 / 100 / 100 / 100 / 100 / 99 / 100 / 98 | 100 / 100 / 100 / 100 / 100 / 100 / 99 / 100 | 1000 |
| DDDG par | 1 | 100 / 100 / 100 / 100 / 100 / 100 / 100 / 97 | 100 / 100 / 100 / 100 / 100 / 100 / 100 / 98 | 1000 |

Readings:
- **Parallel = sequential**: the same learned step in every seed pair (LLLG 1500, DDDG
  1000, as E3's sequential runs), 100% on 1-4 hops from 512 to 16K tokens (final and
  late accuracy), 97-100% on every hop count in the scan. Nothing separates the blocks.
- **CoT extrapolates hops, the loop does not**: 5-8 hops, never trained, stay at
  97-100% at 2K and 8K, against 21-75% (5 hops) and mostly chance (6 hops) for the
  direct answers at r = 2 / 3. Each step of the written chain is the same one-hop
  lookup, so the number of hops is the number of decoding steps, not a depth the
  network has to have.
- **CoT is the cheaper repair**: unlooped CoT learns in 1000-1500 steps and 6-8 minutes,
  direct answers at r = 3 in 3500-9000 steps and 45-65 minutes (24000-step runs), and
  still reach only 93-98% on 4 hops in the scan. The price is answer length: a k-hop
  CoT answer decodes k - 1 intermediate keys before the value (the loop's price is r
  forward passes per token instead).
- Caveats: 2 seeds, one model size; the task is easy for CoT (every step is a copy of
  the record that follows the last written key).

## Speed (2026-11-13)

How much faster can a parallel block train and decode on one RTX 5090? Training: one
matmul for the mixer's and the FFN's input projections (GEMM fusion, PaLM's speed
argument besides tensor parallelism); inference: the mixer and the FFN on two CUDA
streams. Without either, the parallel block trains at the sequential block's speed and
decodes 1-3% faster in a CUDA graph (one RMSNorm less per block).

### Training: GEMM fusion (tested, dropped)

The tested version (not kept) concatenated the weights of the mixer's input projections
(attention q / k / v and output gate; Gated DeltaNet q / k / v / a / b / g) and the FFN's
up / gate projection on the fly, ran one matmul and split the result; the parameters,
checkpoints and Muon's per-matrix updates were unchanged, and Gated DeltaNet ran fla's
kernels from the projected columns. Outputs were bit-identical to separate matmuls,
gradients within 0.7% (bf16 rounding).

Small-LM testbed (32 x 2048 tokens per micro-batch, compiled blocks), training tokens/s:

| pattern | sequential | parallel | parallel, fused |
|---|---|---|---|
| LLLG | 355K | 354K | 328K (-7.5%) |
| DDDG | 280K | 280K | 266K (-5%) |

Separate vs fused input matmuls (q / k / v + SwiGLU up / gate at the FFN width 4x hidden;
bf16), time saved by fusing, per token count M:

| hidden | M = 1 | 8 | 64 | 512 | 4096 | 65536 |
|---|---|---|---|---|---|---|
| 512 | 43% | 46% | 37% | 25% | 13% | 0.4% |
| 1024 | 16% | 39% | 31% | 20% | 7% | 0.5% |
| 2048 | 54% | 23% | 32% | 12% | 3.5% | 0.0% |

- At training token counts (65K per micro-batch) the matmuls run near the GPU's peak
  whether fused or not; profiling one LLLG block (forward + backward), the matmuls take
  6.8 ms per step either way.
- The fused backward needs the gradients of all parts as one contiguous matrix; autograd
  assembles it with one extra copy kernel per block (0.83 ms, against 0.12 ms for the
  attention's own q / k / v concatenation), which is the whole slowdown. A custom
  backward that keeps the parts separate would only remove the loss: the ceiling is the
  < 0.5% of the forward matmul.
- Fusion pays for few tokens (decoding: 15-55%), but only with weights stored merged:
  concatenating them on every call cost more than it saved from hidden 1024 (hidden 2048
  LLLG decode 2.60 -> 4.08 ms), since decoding is bound by reading the weights. Merged
  inference weights (as inference engines load them) are not implemented. With two
  streams, fusing across mixer and FFN also removes their overlap (the fused matmul has
  to finish before either branch starts).

### Inference: two CUDA streams (`model.parallel_streams`)

At inference (CUDA, autograd off) a parallel block launches the FFN on a side stream that
first waits for the current one, runs the mixer on the current stream, then joins the
two. torch.compile traces the fork / join into one graph with both streams, and a CUDA
graph captures them as two branches. Outputs are bit-identical to one stream (eager,
compiled and in a CUDA graph). Off by default.

Decode (`scripts/benchmark_decode.py --compile`: compiled blocks, one step captured in a
CUDA graph, 1024 cached tokens, 64K vocabulary, no MTP), ms per step at batch 1 / 32:

| size | pattern | sequential | parallel | parallel + streams | streams vs one |
|---|---|---|---|---|---|
| hidden 512, 8 blocks (30-34M) | LLLG | 0.243 / 0.337 | 0.236 / 0.329 | 0.204 / 0.285 | -14 / -13% |
| | DDDG | 0.266 / 0.446 | 0.261 / 0.436 | 0.220 / 0.403 | -16 / -8% |
| hidden 1024, 16 blocks (243-275M) | LLLG | 0.786 / 1.164 | 0.776 / 1.151 | 0.685 / 1.042 | -12 / -9% |
| | DDDG | 0.837 / 1.355 | 0.827 / 1.343 | 0.726 / 1.230 | -12 / -8% |
| hidden 2048, 24 blocks (1.46-1.65B) | LLLG | 2.606 / 3.555 | 2.582 / 3.521 | 2.215 / 3.021 | -14 / -14% |
| | DDDG | 2.903 / 3.617 | 2.883 / 3.579 | 2.564 / 3.528 | -11 / -1% |

Non-embedding parameters in brackets. Heads: hidden 512 as the testbed (DeltaNet 4 x
128); hidden 1024 8 x 128 (2 KV heads), DeltaNet 8 x 128; hidden 2048 16 x 128 (4 KV
heads), DeltaNet 16 x 128; FFN 4x hidden.

Eager `generate()` (no CUDA graph; prompt 512, 128 greedy tokens), ms per step at batch
1 / 32:

| size | pattern | sequential | parallel | parallel + streams |
|---|---|---|---|---|
| hidden 512 | LLLG | 2.28 / 2.30 | 2.08 / 2.13 | 2.41 / 2.50 |
| | DDDG | 3.68 / 3.80 | 3.32 / 3.43 | 3.79 / 3.86 |
| hidden 2048 | LLLG | 6.43 / 8.71 | 5.74 / 7.97 | 6.90 / 9.05 |
| | DDDG | 10.58 / 13.10 | 9.89 / 12.11 | 11.03 / 13.32 |

Prefill (`scripts/benchmark_prefill.py`, eager as in `generate()`, batch 1), ms at 2K /
8K / 32K tokens:

| size | pattern | sequential | parallel | parallel + streams |
|---|---|---|---|---|
| hidden 512 | LLLG | 2.4 / 5.7 / 31.0 | 2.2 / 5.5 / 30.6 | 2.4 / 5.1 / 29.7 |
| | DDDG | 6.6 / 8.0 / 35.3 | 6.4 / 7.9 / 35.0 | 6.7 / 7.3 / 33.8 |
| hidden 2048 | LLLG | 35.2 / 146 / 689 | 35.0 / 145 / 678 | 33.0 / 141 / 675 |
| | DDDG | 40.7 / 159 / 737 | 40.4 / 158 / 724 | 37.8 / 153 / 718 |

Readings:
- **Decoding in a CUDA graph is 8-16% faster with two streams** (one exception: DDDG at
  batch 32, hidden 2048, -1%). A decode step is a chain of small kernels that each fill
  part of the GPU; the FFN beside the mixer shortens the chain. Against the sequential
  block that is 10-17% (DDDG batch 32: 2-10%).
- **Eager decoding is 10-20% slower with two streams**: without a CUDA graph the CPU is
  the bottleneck (2-13 ms per step against 0.2-3.6 ms in a graph), and each block adds
  two cross-stream waits to it. Hence off by default; turn it on with CUDA graphs. The
  parallel block alone decodes 7-11% faster than the sequential one eagerly (the eager
  RMSNorm it drops is ~6 kernels).
- **Prefill gains 0.5-8% from 8K tokens** (hidden 512: -7 / -8% at 8K, -3% at 32K;
  hidden 2048: -3% at 8K, < 1% at 32K, where attention dominates); at 2K tokens hidden
  512 is launch-bound and streams cost 5-9%, hidden 2048 gains 6%.

Next: merged inference weights (one matmul for the mixer's input projections, also for
sequential blocks: fusing Gated DeltaNet's six projections alone made hidden-512 DDDG
decoding 16% faster in a test, with a per-call concatenation that does not scale);
whether a partial loop (e.g. a middle core with prelude / coda) also closes the gap.
