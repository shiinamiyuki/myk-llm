# Looped transformer

Status (2026-10-30): implemented (`model.loop`); first sweep (lc chains, 2 seeds) done.

Question: can a few blocks with shared weights, run several times (recurrent depth),
replace a deeper stack on tasks that need serial depth, and can they then trade compute
for accuracy at test time by running more iterations? The lc chain task is the probe:
E3 found that direct (no chain-of-thought (CoT)) multi-hop answers depend on the number
of global layers (1:7 hybrids fail at 3-4 hops) and are learned late, i.e. hops cost
depth.

## Model (`ModelConfig.loop`)

```
x = embed(tokens)
x = prelude(x)                       # loop.prelude blocks, run once
s = 0
repeat r times:                      # loop.core blocks, shared weights
    s = inject(s, x)                 # add: s + x;  concat: Linear([norm(s), norm(x)])
    s = core(s)
y = coda(s)                          # the remaining blocks, run once
```

- Layer types still come from `layer_pattern` over the `num_layers` distinct blocks; the
  core is blocks `[prelude, prelude + core)`. `ModelConfig.layout` prints e.g. `(DG)x4`.
- Input injection every iteration (Yang et al., 2024; Fan et al., 2024 for length
  generalization; Geiping et al., 2025 use the concat adapter). With `add` and r = 1 the
  model is exactly the unlooped one.
- r per pass: `forward(iterations=...)` if given (test-time depth), else drawn uniformly
  from `loop.train_iterations` in training mode, else `loop.iterations`. The draw uses a
  generator seeded like the model init, so every FSDP rank runs the same blocks.
- `loop.backprop_iterations = k`: truncated backpropagation through time (BPTT), the
  first r - k iterations run without autograd; the prelude still gets gradients through
  the injected input of the tracked ones.
- Residual projections are initialized for the effective depth
  (`num_layers + core * (iterations - 1)`).
- Caches: one entry per block call, so a cache only fits the iteration count it was
  built with (checked).
- Not tested yet: FSDP2 with a block called several times per forward (the lc testbed
  is single-GPU).

## Testbed additions (lc)

- `task.eval_hops`: hop counts of a final scan (e.g. 1-8 after training on 1-4).
- `eval_iterations`, `scan_lengths` (LcConfig): the final checkpoint is evaluated once
  per test-time iteration count; `results.json["scan"]`, tabulated by
  `python -m myk_llm.lc.sweep ... --report --at L`.

## First sweep: multi-hop chains (2026-10-30)

`configs/lc/sweeps/loop.yaml`: lc chain task (E3) with direct answers, trained on 1-4
hops at 256-2048 tokens, NoPE global layers, 16000 steps, 2 seeds; ~14-16 min per run
for every architecture (looped: 1.35M tokens/s, DG8 1.24M). The final scan evaluates
1-8 hops at 2K and 8K for test-time iteration counts r = 1-16. Chance is ~15-20% (one
of the 8 items' values). Full tables: `runs/lc/loop/summary.md`.

| arch | distinct blocks | non-embedding params | depth |
|---|---|---|---|
| DG8 | 8 | 8.7M | 8 (compute-matched baseline, E3's DG) |
| DG2 | 2 | 2.2M | 2 (parameter-matched baseline) |
| add4 / concat4 | 2 | 2.2M | (DG)x4, r fixed |
| add2to6 / concat2to6 | 2 | 2.2M | (DG)x2-6, r drawn per step; evaluated at 4 by default |

Final accuracy at 2K per hop count (1 / 2 / 3 / 4) at the default depth, per seed;
"learned" = first evaluation at >= 90% overall:

| arch | seed 0 | seed 1 | learned |
|---|---|---|---|
| DG8 | 100 / 97 / 65 / 30 | 100 / 96 / 93 / 88 | -, 11000 |
| DG2 | 100 / 16 / 15 / 15 | 100 / 16 / 15 / 12 | never |
| add4 | 99 / 94 / 59 / 17 | 100 / 93 / 95 / 73 | never |
| add2to6 | 95 / 71 / 65 / 33 | 99 / 90 / 88 / 20 | never |
| concat4 | 100 / 94 / 18 / 19 | 100 / 94 / 81 / 39 | never |
| concat2to6 | 100 / 94 / 90 / 87 | 99 / 70 / 46 / 36 | 10500, - |

Test-time depth, accuracy at 2K per hop count (mean of 2 seeds; 8K is within a few
points):

| arch | r | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|---|
| DG8 | - | 99 | 91 | 76 | 51 | 16 | 21 | 21 |
| add4 | 2 / 4 / 6 / 8 / 12 | 2 / 100 / 100 / 97 / 43 | 0 / 94 / 92 / 76 / 24 | 0 / 78 / 78 / 59 / 16 | 0 / 37 / 39 / 28 / 5 | <= 22 | <= 19 | <= 21 |
| concat4 | 2 / 4 / 6 / 8 | 0 / 100 / 16 / 12 | 0 / 92 / 13 / 5 | 0 / 40 / 3 / 3 | 0 / 20 / 3 / 2 | <= 18 | <= 16 | <= 19 |
| add2to6 | 2 / 4 / 6 / 8 / 16 | 94 / 93 / 93 / 91 / 66 | 32 / 72 / 69 / 68 / 55 | 23 / 61 / 60 / 60 / 47 | 19 / 20 / 46 / 44 / 38 | 17 / 15 / 25 / 28 / 20 | <= 18 | <= 18 |
| concat2to6 | 1 / 2 / 4 / 6 / 16 | 66 / 99 / 100 / 100 / 97 | 0 / 77 / 74 / 70 / 47 | 1 / 28 / 59 / 60 / 36 | 0 / 19 / 50 / 47 / 28 | <= 19 | <= 19 | <= 22 |

In-training accuracy at 2K was still rising at 16000 steps for add2to6 s0, concat2to6
s1 and concat4 s1 (e.g. add2to6 s0: 41% at step 8000, 66% at the last evaluation).

Readings (2 seeds; seed spread is as large as most differences):
- **The loop buys depth, not capacity**: the 2-block core run 4 times answers 2-4 hops
  in the range of DG8 at a quarter of the parameters and the same compute; the
  unlooped 2-block model only ever answers 1 hop. The best looped seed (concat2to6 s0:
  100 / 94 / 90 / 87, learned at 10500) matches the best DG8 seed (s1, learned at 11000);
  DG8 s0 (97 / 65 / 30 on 2-4 hops) is no better than the looped runs. Two seeds cannot
  rank DG8 against the loops.
- **A fixed r is only a weight-tied deep stack**: add4 / concat4 fail completely (0%,
  even on 1 hop) with fewer iterations than trained; with more, add degrades slowly (r =
  6 is fine, r = 8 loses ~18 points on 2 hops, r >= 12 collapses) and concat collapses at
  once (r = 6: 16% on 1 hop).
- **Training with random r makes r a real knob, and hops cost iterations**: add2to6 and
  concat2to6 degrade gracefully up to r = 16, and accuracy on h hops appears once r is
  large enough: concat2to6 s0 answers 2 hops at r = 2 (94%) but 3-4 hops only from
  r = 4 (79 / 81%); add2to6 s1 is at 16% on 4 hops at r = 4 and 69% at r = 6. Roughly one
  more iteration (one more global-layer pass) per extra hop, as E3's depth reading
  predicts.
- **Almost no hop extrapolation**: 6 and 8 hops stay at chance for every model and r;
  the only signal past the training hops is add2to6 s1 on 5 hops (34-38% at r = 6-8).
  Iterations beyond the trained range never help; accuracy declines slowly from r = 8.
- **The default depth underestimates random-r models**: evaluation (in training and
  the main table) runs at loop.iterations = 4, the middle of the range, while add2to6
  s1 needs r = 6 for 4 hops. Evaluate them at the top of the training range.
- Length: looped models keep their accuracy to 8K like DG8 (NoPE global layer last).
- add vs concat: no clear winner. concat with random r is the only variant that does
  anything at r = 1 (58-73% on 1 hop) and keeps 1 hop at ~100% for every r >= 2.

Next: more seeds and steps for DG8 vs add2to6 / concat2to6, evaluated at r = 6; a wider
r range (2-10) with truncated backpropagation to see whether hops keep following
iterations; training on 1-6 hops to test whether extrapolation needs harder training
chains or an adaptive halting signal.
