# Decoder-decoder (YOCO)

Status (2026-11-09): implemented (`model.cross`, "X" blocks, looped cross-decoder);
E1-E6 done.

Summary:
- Quality: free at this scale. Single-hop retrieval is perfect to 64K (E1); multi-hop
  chains match the unshared stacks with one global KV instead of four, where one global
  layer alone is at chance from 2 hops (E2); small-LM bpb is 0.0065 better than the GL
  baseline for LLLLXXXX (2 seeds) and tied for the literal GLGLXXXX layer reuse (E3).
- A cross-decoder chases pointers one hop per block: hop reach <= number of "X" blocks,
  in every run (E4); an unshared stack trained on 1-6 hops learned nothing past 1 hop.
- Inference: a quarter of the cache and a 4.6x faster 64K prefill for LLLLXXXX; decode
  per token is unchanged (each cross block still reads the full shared KV), and slower
  for GLGLXXXX, which has six full-context reads per token (E5).
- A looped cross-decoder (one weight-shared "X" block iterated r times, `LLLL(X)xr`)
  makes the hop reach a test-time knob: r iterations answer exactly r hops. It is the
  best multi-hop model so far: 100% on 1-4 hops in all 4 runs, against 85% for LLLLXXXX
  and 77% for the looped DG core of notes/loop.md at 4 hops. Past the 1-4 hop training
  it answers 5 hops at 90-98% and 6 hops at 58-68%. Unlike the DG loop, a fixed r = 4
  extrapolates as well as random r. 1-6 hops from scratch did not learn, with random or
  fixed r. Continuing the 1-4 hop models on 1-6 hops did: 96-100% on 1-6 hops, 94% on 7
  and 84% on 8 (the looped DG core after the same curriculum: 62 / 44%). r costs decode
  time (r full-context reads per token, like r "X" blocks), not cache or prefill (E6).

Question: split the stack in two halves, where the second half has no keys / values of
its own and attends to one shared KV (key / value) cache computed once by the first half
(YOCO, "You Only Cache Once", Sun et al., 2024). What does that cost in retrieval,
multi-hop reasoning and language-modeling quality, and what does it buy at inference
(one global KV cache; a prefill that skips the second half for all but the last
position)?

## Model (`ModelConfig.cross`, layer code "X")

```
x = embed(tokens)
x = self_decoder(x)                  # the blocks before the first "X": L / D / G layers
K, V = shared_kv(x)                  # cross.kv=project: Linear(norm(x)), once per position
                                     # cross.kv=layer:   the K, V of the self-decoder's last G layer
y = x                                # (kept positions only, see keep_last)
for each "X" block:                  # cross-decoder
    y = y + CrossAttn(norm(y), K, V) # own queries; causal over the shared K, V
    y = y + FFN(norm(y))
```

- Written as a layer pattern with the "X" blocks last, in full: `LLLLXXXX`,
  `DDDDXXXX`, `GLGLXXXX` (+ `cross.kv=layer`). Validation rejects "X" anywhere but a
  suffix, a cross-decoder without a self-decoder, a loop core across the seam between
  the two halves (E6) and MTP "X" blocks.
- The cross-decoder never feeds the shared KV: a token's keys / values depend only on the
  self-decoder, so earlier positions reach later ones only through the self-decoder's
  states and the one shared KV.
- Cache: one entry per block call, as before; the cross-decoder blocks' entries are
  None, and `project` adds one entry (the shared KV) between the two halves. So the cache
  is the self-decoder's states (sliding windows, DeltaNet states) plus one global KV.
- `forward(keep_last=n)`: only the last n positions' outputs are needed (`generate()`'s
  prefill uses n = 1); the cross-decoder and the final norm then run on those positions
  only (YOCO's prefill saving: the second half costs one token, not the prompt). The cache
  is the same as after a full pass (tested).
- Position: the cross-decoder queries get RoPE exactly when the shared keys do, i.e. under
  `global_rope: all`, or `first` when the shared KV is the first global attention of the
  stack (`project` with no "G" in the self-decoder, or `layer` reusing the first "G").
- `cross.kv=layer` is the literal "the second half reuses the first half's full
  attention". If the reused G layer is the last self-decoder block, it is itself just a
  cross-attention block whose K / V projection the others share, so `LLLGXXXX` (layer)
  is `LLL` + five cross blocks; it differs from `project` when blocks follow the reused
  layer in the self-decoder (`GLGLXXXX`: the KV is computed at block 2, block 3 still runs).
- FSDP2 (fully sharded data parallel) with activation checkpointing gives the unsharded
  gradients for both KV sources (tested on one GPU, world size 1; not yet on several).
- Benchmarks: `scripts/benchmark_prefill.py` (prefill time, full-pass time, cache size
  per variant and length) and `scripts/benchmark_decode.py` (static-cache CUDA-graph
  decode; the shared KV gets one static buffer that every cross block reads; attention
  layers only, DeltaNet states are not supported there).

## E1: needle retrieval (`configs/lc/sweeps/yoco-needle.yaml`)

lc needle task (8 records, trained on <= 2K tokens, 8000 steps, `global_rope: none`), two
seeds each; evaluated out to 64K (32x the training length).

| layers | KV cache | 2K | 16K | 64K | learned (steps, s0 / s1) |
|---|---|---|---|---|---|
| LLLLXXXX (project) | 4 windows + 1 global | 100 | 100 | 100 | 1000 / 1500 |
| LLLGXXXX (layer) | 3 windows + 1 global | 100 | 100 | 100 | 1000 / 1000 |
| DDDDXXXX (project) | 4 DeltaNet states + 1 global | 100 | 100 | 100 | 1000 / 1000 |
| LLLLGGGG | 4 windows + 4 global | 100 | 100 | 100 (99 late) | 1000 / 1000 |
| DDDDGGGG | 4 DeltaNet states + 4 global | 100 | 100 | 100 (99 late) | 1000 / 1000 |

Saturated, as E4 of `notes/lc-retrieval.md` predicted (one NoPE global layer last is
already perfect on needle): sharing one KV across the second half costs nothing on
single-hop retrieval, at any depth and length. Not discriminating; the multi-hop chain
sweep is the real test.

## E2: multi-hop chains, direct answers (`configs/lc/sweeps/yoco-chain.yaml`)

lc chain task (8 chains of 1-4 hops in noise, 4 queried, the answer is the end of the
chain with no intermediate keys written), 24000 steps, `global_rope: none`, the settings
of loop-confirm. Accuracy at 2K by hop count (final checkpoint): the 4-hop column per
seed (s0 / s1 / s2 / s3), the others as means; "learned" is the first eval step at >= 90%
in distribution (- = never). The 1-global-layer arms have two seeds.

| layers | global KVs | 1 hop | 2 hops | 3 hops | 4 hops | 4 hops per seed | 5 hops (untrained) | 16K (all hops) | learned |
|---|---|---|---|---|---|---|---|---|---|
| LLLLXXXX | 1 | 100 | 92 | 92 | 85 | 83 / 83 / 89 / 87 | 16 | 92 | 7K / 10.5K / 6K / 8K |
| DDDDXXXX | 1 | 100 | 94 | 90 | 76 | 87 / 86 / 80 / 51 | 14 | 90 | 12.5K / 9K / 8.5K / - |
| LLLLGGGG | 4 | 100 | 94 | 91 | 73 | 36 / 83 / 97 / 78 | 26 | 84 | - / 13.5K / 8K / 13.5K |
| DDDDGGGG | 4 | 99 | 93 | 87 | 69 | 83 / 28 / 78 / 89 | 24 | 86 | 18.5K / - / 11K / 17K |
| DDDDDDDG | 1 | 100 | 16 | 17 | 20 | 19 / 21 | 14 | 37 | - |
| LLLLLLLG | 1 | 100 | 16 | 15 | 16 | 16 / 16 | 21 | 34 | - |
| DGDGDGDG (loop-confirm) | 4 | 100 | 91 | 89 | 85 | | 31 | 89 | |

- At the same cache (one global KV), the cross-decoder turns a model that cannot do two
  hops (one global layer, at chance from 2 hops; E3b's DDDDDDDG was 57 / 17 / 14) into one
  that does four: 76-85% at 4 hops, flat out to 16K (8x the training length).
- Against the unshared controls (same layer types, four global KV caches), the
  cross-decoder is at least as good, not clearly better: means 85 / 76 vs 73 / 69 at 4
  hops, with overlapping spreads (the best single run is an unshared one, LLLLGGGG s2 at
  97). Every arm except LLLLXXXX has one of four seeds that did not learn 4 hops (28-51%);
  LLLLXXXX learned on all four, and soonest (6K-10.5K steps vs 8K-18.5K). With s0 / s1
  only, this looked like a reliability win for both cross-decoders; s2 / s3 put DDDDXXXX
  with the controls.
- Reading: with fixed keys / values, each cross block can only re-query the same memory,
  which is exactly the pointer chase (one hop per block: query -> record key -> its value
  -> the next query), and all blocks train the one shared KV. E4 tests that reading.
- Hop reach stops at the trained 4 hops (5 hops at chance), which E2 alone cannot tell
  from "one hop per cross block"; E4 can.

## E3: small LM (`configs/ngram-small/sweeps/yoco.yaml`)

The small-scale LM testbed (hidden 512, 8 layers, 2048-token context, 512-token windows,
~400M FineWeb-Edu en + zh tokens, 64K vocabulary, `global_rope: first`); validation
bits per byte (bpb), means over seeds 0 and 1 (`yoco-seeds.yaml`; LLLLLLLG seed 0 only),
with the per-seed mean bpb.

| layers | global KV caches | non-embedding params | en bpb | zh bpb | mean bpb | per seed |
|---|---|---|---|---|---|---|
| LLLLXXXX (project) | 1 | 30.02M | 1.1970 | 1.2078 | 1.2024 | 1.2040 / 1.2008 |
| GLGLGLGL (GL, baseline) | 4 | 30.42M | 1.2033 | 1.2144 | 1.2089 | 1.2089 / 1.2088 |
| GLGLXXXX (layer) | 2 | 29.89M | 1.2047 | 1.2134 | 1.2091 | 1.2082 / 1.2099 |
| LLLLGGGG | 4 | 30.42M | 1.2064 | 1.2202 | 1.2133 | 1.2130 / 1.2136 |
| LLLLLLLG | 1 | 30.42M | 1.2074 | 1.2245 | 1.2160 | 1.2160 |

- No LM cost at this scale, with 1.3% fewer parameters: the literal design (GLGLXXXX
  reusing block 2's KV) ties the baseline at half its global KV cache, and the YOCO
  layout LLLLXXXX is 0.0065 bpb better at a quarter of it, on both seeds (by 0.005 and
  0.008, against a 0.0001 seed spread of the baseline), on en and zh.
- Same direction as the chains: sharing one KV across the second half (LLLLXXXX) beats
  giving each of those layers its own (LLLLGGGG) by 0.011 bpb; one global layer and no
  cross-decoder (LLLLLLLG) is the worst, mostly on zh.
- At 2048 tokens with 512-token windows this says little about long-context LM quality;
  the lc sweeps are the long-context side.

## E4: split ratio, trained on 1-6 hops (`configs/lc/sweeps/yoco-split.yaml`)

E2's chains with 1-6 hops in training (min_len 320), 24000 steps; the number of cross
blocks out of 8 varies. Accuracy at 2K by hop count, seeds s0 / s1 (chance ~15).

| layers | cross blocks | 1 hop | 2 hops | 3 hops | 4 hops | 5 hops | 6 hops |
|---|---|---|---|---|---|---|---|
| LLLLLLXX | 2 | 99 / 99 | 90 / 15 | 16 / 14 | 12 / 10 | 14 / 16 | 11 / 15 |
| LLLLXXXX | 4 | 98 / 98 | 85 / 98 | 84 / 96 | 83 / 96 | 16 / 16 | 17 / 18 |
| LLXXXXXX | 6 | 99 / 99 | 96 / 96 | 97 / 96 | 98 / 96 | 97 / 97 | 94 / 16 |
| LLGGGGGG (unshared) | - | 99 / 14 | 13 / 14 | 16 / 16 | 15 / 16 | 14 / 15 | 17 / 14 |

- Hop reach <= number of cross blocks, in every run, and either equal to it or one
  short: one hop per cross block. LLLLXXXX is at chance on 5 and 6 hops although both
  are in its training data. This is close to structural: the shared KV comes from a
  sliding-window self-decoder (window 256), so records farther apart than the windows
  cannot be linked inside it, and each hop must be one lookup from the query position,
  one attention per cross block.
- The unshared six-global-layer control learned nothing past 1 hop in 24000 steps (one
  seed not even 1 hop): with 5-6 hop chains in the mix, the free-form stack does not find
  the pointer chase, while the cross-decoder's fixed memory + re-query structure does.
- So a cross-decoder is a pointer-chasing machine with depth = its block count. Its
  natural extension is a looped cross-decoder: one (or a few) weight-shared "X" blocks
  iterated r times over the same shared KV, which would set the hop reach at test time
  at no cache cost (validation currently rejects "X" with `loop.core`; see notes/loop.md
  for the injected loops that already extend hops).

## E5: inference cost (`scripts/benchmark_prefill.py`, `scripts/benchmark_decode.py`)

ngram-small model shape (hidden 512, 8 layers, 2 KV heads x 64, 512-token windows, 64K
vocabulary), random weights, bf16, idle GPU. Raw output: `runs/yoco-bench.md`.

Prefill (batch 1, eager; "prefill" = generate()'s pass with the cache and `keep_last=1`,
"full" = every position, no cache, i.e. the training forward):

| layers | prefill ms 16K | prefill ms 64K | full ms 64K | cache MB 64K |
|---|---|---|---|---|
| GLGLGLGL | 14.5 | 122.1 | 122.6 | 129.0 |
| GLGLXXXX (layer) | 8.0 | 63.6 | 159.0 | 64.5 |
| LLLLXXXX | 5.8 | 26.7 | 126.4 | 33.0 |
| LLLLGGGG | 14.6 | 122.2 | 122.6 | 129.0 |
| LLLLLLLG | 11.3 | 67.7 | 68.0 | 33.8 |
| DDDDXXXX | 9.9 | 40.8 | 144.5 | 34.1 |
| DDDDGGGG | 18.7 | 135.6 | 140.3 | 130.1 |

Decode (static caches, one CUDA graph per token, no MTP), ms per token:

| layers | 8K, batch 1 | 8K, batch 32 | 32K, batch 1 | 32K, batch 32 |
|---|---|---|---|---|
| GLGLGLGL | 0.665 | 0.973 | 1.383 | 1.973 |
| GLGLXXXX (layer) | 0.735 | 1.091 | 1.771 | 2.589 |
| LLLLXXXX | 0.703 | 1.028 | 1.383 | 2.024 |
| LLLLGGGG | 0.664 | 0.966 | 1.381 | 1.961 |
| LLLLLLLG | 0.559 | 0.793 | 0.730 | 1.042 |

- Cache: LLLLXXXX keeps a quarter of the baseline's (33 vs 129 MB at 64K: one global KV
  plus four 512-token windows), the same as one global layer; GLGLXXXX (layer) half.
- Prefill: 4.6x faster than the baseline at 64K for LLLLXXXX (the cross-decoder runs on
  one position, so the prompt pays for four sliding-window layers and one KV
  projection), 1.9x for GLGLXXXX, 3.3x for DDDDXXXX vs DDDDGGGG. Faster even than
  LLLLLLLG, whose one global layer still attends over the whole prompt.
- Decode latency is not reduced: every cross block still reads the whole shared KV per
  token, so LLLLXXXX costs what four global layers cost (2.02 vs 1.97 ms at 32K, batch
  32), and GLGLXXXX, with six full-context reads (two G + four X), is 30% slower than
  the baseline. Decode reads scale with the number of full-context attentions, not with
  the number of caches; only LLLLLLLG (one read) is faster.
- Capacity does not turn into throughput here. At 32K, batch 32 / 64 / 128: GLGLGLGL
  16.2 / 13.6 / 7.8 Ktok/s, LLLLXXXX 15.8 / 13.4 / 7.8, LLLLLLLG 30.8 / 35.7 / 26.7
  (`runs/yoco-bench-batch.md`). The baseline's batch-128 cache still fits on this 32 GB GPU,
  and past batch 32 every variant loses throughput (the KV reads run at 1.1 TB/s at batch 32,
  0.5 TB/s at batch 128): a static-cache attention kernel issue at multi-GB caches, not
  specific to YOCO, and not investigated. YOCO's 4x smaller cache matters where memory
  binds (large models, long contexts), as the batch / context it allows, not as a
  per-token speedup.
- Training cost: the full pass is the baseline's for LLLLXXXX (126 vs 123 ms at 64K;
  338K vs 343K tok/s in E3). GLGLXXXX's full pass has six full-context attentions and is
  30% slower at 64K (no difference at the 2K training length).

## E6: looped cross-decoder (`configs/lc/sweeps/yoco-loop.yaml`)

E4's one hop per cross block suggests sharing the cross block's weights and iterating it:
the loop core of `model.loop` may now lie inside the cross-decoder (`prelude` >= the
first "X" block), e.g. `LLLL(X)xr` = `layer_pattern: LLLLX, num_layers: 5,
loop.prelude: 4, loop.core: 1`:

```
x = self_decoder(x); K, V = shared_kv(x)
s = 0
repeat r times:                    # one weight-shared X block (or several)
    s = inject(s, x)               # add: s + x (r = 1 is the unlooped model)
    s = X(s, K, V)                 # re-query the same shared KV
y = coda(s)                        # X blocks after the core, if any
```

- Only the query state iterates; the memory (shared KV) is fixed, so every iteration
  can apply the same lookup. The self-decoder loop of notes/loop.md instead re-runs its
  core over the whole context each iteration.
- Cross-decoder blocks keep no state, so the cache does not depend on r: a prompt cached
  at one iteration count can be decoded at any other (tested), and r costs compute on
  the query tokens only (with `keep_last`, a prefill pays r X-block calls for one token).
- The loop core may also lie in the self-decoder, before the "X" blocks (e.g.
  `(LL)x2XX`); a core across the seam is rejected.
- FSDP2 with a block called several times per forward is still untested (as for the
  self-decoder loop).

Arms (24000 steps, `global_rope: none`, chain settings of yoco-chain / loop-hops, 2
seeds): `r2to10` (r drawn from 2-10 per step, evaluated at r = 10; the reference is
loop-hops' r2to10 with a DG core: 98 / 90 / 86 / 77 / 49 / 30 / 14 % at 1-6, 8 hops and
r = 10), `r4` (fixed r = 4: LLLLXXXX's compute with a quarter of its cross-decoder
parameters), `h6r2to10` (1-6 hops from scratch, which the DG loop did not learn: 57 / 44
/ 36 / 28 / 24 % at 2-6 hops). ~21 min per run (r4: 17), 5.25M non-embedding
parameters (five blocks).

Final accuracy at 2K on the trained hops (default r): 100 / 100 / 100 / 100 for r2to10
and r4 (99.6-99.9% overall, both seeds of both arms; learned, i.e. crossed 90%, at
13000 / 15000 and 7500 / 12000 steps). Over context lengths 512-16K: r2to10 100 / 100 / 100 / 100
/ 99 / 98, r4 100 everywhere. h6r2to10 is at chance on every hop count, 1 hop included,
in both seeds (training loss 0.096 / 0.098 at step 24000, still creeping down; r2to10:
0.010 / 0.007).

Final scan at 2K, mean of 2 seeds; 5-8 hops are past the training (10 and 12 hops are
at chance, <= 21%, for every model and r):

| arm | r | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|---|
| r2to10 | 1 | 100 | 30 | 16 | 20 | 14 | 17 | 16 |
| r2to10 | 2 | 100 | 100 | 29 | 15 | 14 | 15 | 20 |
| r2to10 | 3 | 100 | 100 | 100 | 22 | 17 | 17 | 21 |
| r2to10 | 4 | 100 | 100 | 100 | 98 | 23 | 18 | 16 |
| r2to10 | 5 | 100 | 100 | 100 | 99 | 91 | 25 | 19 |
| r2to10 | 6 | 100 | 100 | 100 | 99 | 93 | 58 | 22 |
| r2to10 | 10 | 100 | 100 | 100 | 98 | 92 | 58 | 22 |
| r2to10 | 24 | 99 | 99 | 100 | 96 | 87 | 53 | 21 |
| r4 | 1 | 99 | 3 | 2 | 1 | 1 | 2 | 3 |
| r4 | 2 | 100 | 100 | 1 | 2 | 3 | 0 | 4 |
| r4 | 3 | 100 | 100 | 100 | 1 | 1 | 2 | 1 |
| r4 | 4 | 100 | 100 | 100 | 100 | 4 | 3 | 3 |
| r4 | 5 | 100 | 100 | 100 | 100 | 98 | 10 | 5 |
| r4 | 6 | 100 | 100 | 100 | 100 | 96 | 68 | 7 |
| r4 | 10 | 100 | 100 | 99 | 97 | 90 | 61 | 12 |
| r4 | 24 | 100 | 93 | 85 | 86 | 73 | 42 | 8 |

Per seed: 5 hops at r = 5, r2to10 94 / 89, r4 97 / 98; 6 hops at r = 6, r2to10 74 / 42,
r4 57 / 79. r4 at r = 24 on 2-4 hops: s0 100 / 100 / 98, s1 87 / 71 / 73. At 8K the scan
is within 5 points of 2K on 1-5 hops, 6 hops lose up to 10 (r2to10, r = 10: 99 / 100 /
100 / 100 / 87 / 48; r4, r = 6: 100 / 100 / 100 / 98 / 93 / 67).

- **One iteration per hop, exactly.** In both arms r iterations answer every chain of
  up to r hops and fail at r + 1 (4 hops: 22% at r = 3, 98% at r = 4). E4's one hop per
  cross block carries over to iterations of one block, and more sharply than in the DG
  loop, which needed about r = 6 for 4 hops.
- **The best multi-hop model of the lc experiments so far.** 100% on 1-4 hops in all four
  runs. LLLLXXXX (yoco-chain, four distinct cross blocks, 8 blocks of parameters) gets
  85% at 4 hops (4-seed mean), the looped DG core 77% (loop-hops r2to10, r = 10).
- **One hop past the training, partly two.** 5 hops at 90-98% and 6 hops at 58-68% at
  r = hops-10, against 49 / 30% for the DG loop. 8 hops stay at chance at every r, so what
  stops longer chains is not the iteration count.
- **A fixed r works, unlike in the DG loop.** There, a fixed r = 4 (add4) failed
  completely at any other r ("only a weight-tied deep stack", notes/loop.md). Here r4
  answers k hops at r = k for every k <= 4 and 5 hops at r = 5 (98%). The step it
  learned holds still once the answer is reached: 1 hop at r = 4 means three idle
  iterations. With too few iterations it is below chance (0-4%) where r2to10 is near it
  (14-30%); it seems to output a definite wrong token, probably the node reached after
  r hops (not checked). Random r is not needed for the knob here. What it buys is
  stability at large r: at r = 24, r2to10 loses <= 5 points on 1-5 hops, while r4
  loses up to 29 points on 2-4 hops in s1. r4 also learned sooner (7500 / 12000 steps
  against 13000 / 15000).
- Why the cross loop beats the DG loop is not established. A likely reason is that
  only the query state iterates, over a fixed memory, so each iteration can apply the
  same lookup; the DG core instead re-encodes the whole context every iteration.
- **1-6 hops from scratch did not learn at all**, not even the 1-hop chains. The looped
  DG core learned 1 hop in that setting, and yoco-split's LLXXXXXX (six distinct cross
  blocks) learned 1-6 hops. A fixed r = 6 does not help either (`h6r6`, below).

Follow-up on 1-6 hops (`yoco-loop-curriculum.yaml`, and the `h6r6` arm of
`yoco-loop.yaml`):
- `h6r6`: fixed r = 6 on 1-6 hops from scratch, 24000 steps. At chance on every hop count
  in both seeds (loss 0.096 / 0.098, as h6r2to10). Fixed r learned 1-4 hops sooner than
  random r, but it does not make 1-6 hops learnable from scratch.
- Curriculum, as loop-curriculum.yaml did for the DG core: continue the final r2to10
  model of each seed for 16000 steps with a fresh optimizer and schedule, on 1-6 hops
  (h6, min_len 320) or on 1-4 hops again (h4, the control for the extra training). The
  scan adds 7 hops and r = 7. 14.5 min per run.

Both h6 seeds crossed 90% within 1000 steps (the DG core: 3000-12500). Final accuracy at
2K on 1-6 hops: s0 99 / 99 / 98 / 98 / 97 / 95, s1 100 / 100 / 100 / 99 / 100 / 100;
512-16K all 97-100%. Final scan at 2K, mean of 2 seeds (12 hops at chance everywhere):

| arm | r | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| start (r2to10 above) | 10 | 100 | 100 | 100 | 98 | 92 | 58 | | 22 | 17 |
| h4 | 10 | 100 | 100 | 99 | 99 | 94 | 58 | 29 | 21 | 18 |
| h6 | 4 | 100 | 98 | 97 | 97 | 25 | 17 | 15 | 16 | 18 |
| h6 | 6 | 100 | 98 | 97 | 97 | 98 | 96 | 23 | 20 | 16 |
| h6 | 7 | 100 | 98 | 97 | 98 | 98 | 96 | 94 | 25 | 17 |
| h6 | 8 | 100 | 98 | 97 | 98 | 98 | 96 | 94 | 84 | 15 |
| h6 | 10 | 100 | 98 | 97 | 98 | 99 | 96 | 93 | 83 | 24 |
| h6 | 24 | 99 | 98 | 97 | 98 | 99 | 96 | 94 | 76 | 28 |
| DG core h6 (notes/loop.md, 4 seeds) | 8 | 100 | 95 | 93 | 89 | 89 | 81 | 62 | 44 | |

Per seed, h6: 7 hops at r = 7: 90 / 97; 8 hops at r = 8: 78 / 90. At 8K, h6 at r = 8 is
99 / 99 / 97 / 99 / 98 / 97 / 93 / 79.

- **The curriculum extends the reach by the two hops it adds.** Trained on 1-4 hops, the
  model answers 5 hops at about 92% and 6 at about 58%. Trained on 1-6, it answers 7 at
  94% and 8 at 84%. 10 hops stay near chance (15-32%). h4, the same extra training on
  1-4 hops, leaves 5-8 hops where they were, so the gain comes from the longer chains.
- **Still one iteration per hop**, now up to 8 (r = 7: 94% on 7 hops, 25% on 8), and
  stable to r = 24 except on 8 hops (84 -> 76).
- **Ahead of the DG core after the same curriculum** on 2-8 hops (r = 8:
  94 / 84 vs 62 / 44% on 7 / 8 hops), with 2 seeds against 4.
- From scratch, long chains are not learnable at this budget with random or fixed r.
  The cross loop needs the curriculum; six distinct cross blocks did not (E4).
- Next, if this line continues: whether the reach keeps growing by two hops per stage
  (1-8 -> 9 and 10 hops), and whether a 1-2 -> 1-4 -> 1-6 schedule from scratch replaces
  the separate runs.

Cost (`runs/yoco-loop-bench.md`, E5 setup: ngram-small shape, random weights, static-cache
CUDA-graph decode). `LLLL(X)xr` has 19.0M non-embedding parameters, against 30.0M for
LLLLXXXX:

| layers | decode ms 32K, batch 1 | decode ms 32K, batch 32 | prefill ms 64K | full ms 64K | cache MB 64K |
|---|---|---|---|---|---|
| GLGLGLGL (E5) | 1.383 | 1.973 | 122.1 | 122.6 | 129.0 |
| LLLLXXXX (E5) | 1.383 | 2.024 | 26.7 | 126.4 | 33.0 |
| LLLL(X)x1 | 0.576 | 0.815 | | | |
| LLLL(X)x4 | 1.389 | 2.027 | 26.7 | 125.9 | 33.0 |
| LLLL(X)x10 | 3.038 | 4.452 | 29.7 | 279.5 | 33.0 |

- Each iteration reads the whole shared KV per token, like an "X" block. So r = 4 costs
  exactly what LLLLXXXX costs at decode, and r = 10 costs 2.2x the GL baseline per
  token. r = 10 is only needed for chains past 4 hops.
- The prefill (the iterations run on the last position only) and the cache do not
  depend on r. Training does: the full pass is 2.2x longer at r = 10 than at r = 4, and
  r2to10 (mean r = 6) trained in 21.5 min against 17.3 for r4 at the lc scale.
