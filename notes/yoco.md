# Decoder-decoder (YOCO)

Status (2026-11-08): implemented (`model.cross`, "X" blocks); E1-E5 done.

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
- Next: a looped cross-decoder (weight-shared "X" blocks iterated r times over the
  shared KV) to set the hop reach at test time (E4).

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
  suffix, a cross-decoder without a self-decoder, looped cores and MTP "X" blocks.
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
