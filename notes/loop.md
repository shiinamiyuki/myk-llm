# Looped transformer

Status (2026-11-08): implemented (`model.loop`); four lc chain sweeps done (first
sweep, 4-seed follow-up, hop extrapolation, curriculum) and a zero-training early-exit
analysis (`myk_llm.lc.halting`).

Question: can a few blocks with shared weights, run several times (recurrent depth),
replace a deeper stack on tasks that need serial depth, and can they then trade compute
for accuracy at test time by running more iterations? The lc chain task is the probe:
E3 found that direct (no chain-of-thought (CoT)) multi-hop answers depend on the number
of global layers (1:7 hybrids fail at 3-4 hops) and are learned late, i.e. hops cost
depth.

Summary so far (2-block D + NoPE G core, add injection unless noted):
- The looped core matches the 8-layer stack (DG8) on 1-4 hops at a quarter of the
  parameters and the same compute (loop-confirm).
- Train with a random iteration count r per step, else r is not a test-time knob; a
  wider range (2-10 vs 2-6) learns more reliably and stays usable to r = 16 (loop-hops).
  Keep full backpropagation; truncating to the last 4 iterations hurts.
- About one iteration per hop: h hops need r >= h (sometimes h - 1).
- Extrapolation reaches 1-2 hops past the trained ones, relative to them: trained on
  1-4 hops, 5 hops ~50% and 6 ~30%; continued on 1-6 hops, 7 hops ~60% and 8 ~43%;
  further hops are at chance for every r. More training on the same hops does not
  extend it (loop-curriculum). add extrapolates, concat injection does not.
- Long chains need a curriculum: 1-6 hops from scratch did not learn in 24000 steps,
  continuing a 1-4 hop model learned them in 3000-12500.
- Same-seed reruns diverge (nondeterministic kernels): compare rates over seeds.
- Early exit works without training on output signals, not on state convergence: the
  loop state never reaches a fixed point, but the prediction settles one iteration
  after the chain is resolved; exiting at confidence > 0.9 keeps the accuracy at about
  half the iterations on the trained hops (per-query exits, context at full depth).

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
- `scan_iterations`, `scan_lengths` (LcConfig): the final checkpoint is evaluated once
  per test-time iteration count; `results.json["scan"]`, tabulated by
  `python -m myk_llm.lc.sweep ... --report --at L`.
- `eval_iterations` (LcConfig): iteration count of the in-training and final
  evaluations (hence of "learned" and the main table) without changing the model.

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
  s1 needs r = 6 for 4 hops. Evaluate them at the top of the training range. (Only at
  16000 steps: after 24000, r = 4 is enough for 1-4 hops; see the follow-up.)
- Length: looped models keep their accuracy to 8K like DG8 (NoPE global layer last).
- add vs concat: no clear winner. concat with random r is the only variant that does
  anything at r = 1 (58-73% on 1 hop) and keeps 1 hop at ~100% for every r >= 2.

## Follow-up: 4 seeds, 24000 steps (2026-10-31)

`configs/lc/sweeps/loop-confirm.yaml`: DG8 vs add2to6 / concat2to6 as above, 4 seeds,
24000 steps; the looped models (unchanged) are evaluated at r = 6 during training and in
the main table (`eval_iterations`), and the scan adds r = 3 and 5. ~21-25 min per run;
training throughput DG8 1.24M tokens/s, add 1.29M, concat 1.11M (the concat adapter
costs ~15% at this size). Full tables: `runs/lc/loop-confirm/summary.md`.

Final accuracy at 2K per hop count (1 / 2 / 3 / 4), looped models at r = 6:

| arch | seed 0 | seed 1 | seed 2 | seed 3 | learned |
|---|---|---|---|---|---|
| DG8 | 100 / 94 / 95 / 87 | 100 / 97 / 97 / 90 | 100 / 96 / 94 / 88 | 100 / 77 / 68 / 73 | 11500, 10500, 9000, - |
| add2to6 | 100 / 97 / 96 / 90 | 100 / 98 / 97 / 94 | 99 / 88 / 44 / 15 | 100 / 95 / 94 / 86 | 10500, 9000, -, 17500 |
| concat2to6 | 100 / 94 / 91 / 88 | 100 / 99 / 97 / 94 | 100 / 79 / 68 / 64 | 100 / 92 / 92 / 88 | 11000, 14000, -, 13500 |

Mean over all 4 seeds, all queries at 2K: DG8 91, add2to6 87, concat2to6 90 (8K: 90 /
87 / 90; accuracy_late 88 / 83 / 87).

Final scan at 2K, mean of 4 seeds (r = 5 and 8 are within 2 points of r = 6). The scan
contexts also hold 5-8-hop chains, so 1-4 hops score a few points below the table above:

| arch | r | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|---|
| DG8 | - | 100 | 87 | 88 | 81 | 31 | 21 | 20 |
| add2to6 | 1 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| add2to6 | 2 | 99 | 63 | 34 | 16 | 18 | 20 | 20 |
| add2to6 | 3 | 98 | 90 | 74 | 41 | 20 | 19 | 20 |
| add2to6 | 4 | 98 | 89 | 77 | 67 | 38 | 26 | 23 |
| add2to6 | 6 | 99 | 88 | 77 | 69 | 48 | 29 | 23 |
| add2to6 | 12 | 91 | 83 | 74 | 65 | 47 | 28 | 21 |
| add2to6 | 16 | 69 | 66 | 63 | 53 | 38 | 23 | 17 |
| concat2to6 | 1 | 27 | 1 | 1 | 1 | 1 | 0 | 1 |
| concat2to6 | 2 | 98 | 83 | 41 | 21 | 20 | 15 | 19 |
| concat2to6 | 3 | 98 | 86 | 76 | 35 | 23 | 21 | 20 |
| concat2to6 | 4 | 98 | 85 | 75 | 67 | 24 | 21 | 18 |
| concat2to6 | 6 | 98 | 84 | 74 | 67 | 28 | 21 | 19 |
| concat2to6 | 12 | 96 | 76 | 67 | 60 | 25 | 22 | 19 |
| concat2to6 | 16 | 94 | 69 | 59 | 52 | 26 | 22 | 20 |

Readings:
- **Confirmed: the 2-block looped core matches DG8 at a quarter of the parameters.**
  Every architecture learns in 3 of 4 seeds, at similar steps (DG8 9000-11500, add
  9000-17500, concat 11000-14000), and the learned seeds reach the same accuracy (4
  hops: DG8 87-90, add 86-94, concat 88-94). The one unlearned seed per architecture
  was still at 57-78% and mostly rising at 24000 steps; seeds differ more than
  architectures do. The match also holds at DG8's compute: the learned looped seeds at
  r = 4 are within 3 points of r = 6 on 1-4 hops.
- **One iteration per hop**: h hops need r >= h, and more iterations add nothing on
  the trained hops. Per seed the steps are sharp, e.g. concat s0 / s3 on 3 hops 25-29%
  at r = 2 -> 88% at r = 3, on 4 hops 19-20% at r = 3 -> 79-84% at r = 4; add s0 on 3
  hops 57 -> 91 (r = 2 -> 3), on 4 hops 69 -> 86 (r = 3 -> 4). Each iteration is one
  D + one G layer, and DG8 (4 G layers) handles 4 hops: one global-attention pass per
  hop, as E3 suggested. (concat s1 is the exception: 3 hops at 77% already at r = 2.)
  The first sweep's r = 4 deficit (add2to6 s1: 4 hops 16% at r = 4) was undertraining;
  the same seed is at 89% after 24000 steps.
- **add extrapolates past the trained hops with extra iterations; concat does not.**
  add s1 / s3 answer 5 hops at 72-76% from r = 5 (s1: 42% at r = 4 -> 71 at r = 5 ->
  76 at r = 6), s3 also 6 hops at ~50% and 8 hops at 30-33% (chance ~20%), stable up
  to r = 12; DG8's best seed reaches 43% on 5 hops. Only 2 of the 3 learned add seeds
  do it (s0 stays at chance). concat s0 / s3 are at chance on 5 hops for every r, and
  concat s1's ~50% does not grow with r. The add loop seems to learn a reusable hop
  step; concat something tied to the trained depth.
- **Too many iterations**: r <= 8 is safe; add loses 1 hop from r = 12 (91%, 69% at
  r = 16); concat keeps 1 hop (94% at r = 16) but loses ~15 points on 2-4 hops.
- 8K stays within ~5 points of 2K for every architecture.

## Hop extrapolation: wider r range, truncated BPTT, longer chains (2026-10-31)

`configs/lc/sweeps/loop-hops.yaml`: add injection, 2-block DG core, 24000 steps, 4
seeds, evaluated at the top of each arm's r range. The final scan covers 1-12 hops
(1-6, 8, 10, 12) at r = 1-24; its contexts hold more long chains than loop-confirm's,
so r2to6 (loop-confirm's add2to6, same seeds) runs again to be scanned the same way.

| arm | trained hops | train r | backprop | min / run | mean 2K | 1 / 2 / 3 / 4 hops | learned (s0-s3) |
|---|---|---|---|---|---|---|---|
| r2to6 | 1-4 | 2-6 | all | 21 | 80 | 99 / 77 / 74 / 70 | -, 11000, 18500, 19500 |
| r2to10 | 1-4 | 2-10 | all | 31 | 96 | 99 / 96 / 97 / 94 | 13000, 18500, 11000, 7500 |
| r2to10bp4 | 1-4 | 2-10 | last 4 | 24 | 77 | 97 / 81 / 70 / 62 | 20000, -, -, 19000 |
| h6r2to10 | 1-6 (min_len 320) | 2-10 | all | 31 | 49 | 98 / 61 / 45 / 34 (5: 33, 6: 26) | -, -, -, - |

Final scan at 2K, mean of 4 seeds; 10 and 12 hops are at chance (<= 22%) for every arm
and r:

| arm | r | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|---|
| r2to6 | 3 | 99 | 73 | 44 | 17 | 18 | 17 | 14 |
| r2to6 | 4 | 99 | 73 | 63 | 45 | 23 | 17 | 12 |
| r2to6 | 6 | 98 | 70 | 62 | 56 | 32 | 23 | 12 |
| r2to6 | 10 | 93 | 66 | 62 | 55 | 31 | 23 | 14 |
| r2to6 | 16 | 53 | 41 | 40 | 35 | 23 | 15 | 9 |
| r2to6 | 24 | 20 | 24 | 27 | 24 | 12 | 8 | 5 |
| r2to10 | 3 | 99 | 93 | 45 | 19 | 14 | 15 | 17 |
| r2to10 | 4 | 99 | 93 | 85 | 52 | 26 | 21 | 18 |
| r2to10 | 6 | 99 | 91 | 86 | 76 | 46 | 30 | 15 |
| r2to10 | 10 | 98 | 90 | 86 | 77 | 49 | 30 | 14 |
| r2to10 | 16 | 98 | 90 | 84 | 77 | 46 | 30 | 15 |
| r2to10 | 24 | 90 | 74 | 60 | 53 | 26 | 15 | 10 |
| r2to10bp4 | 1 | 66 | 2 | 4 | 3 | 3 | 4 | 4 |
| r2to10bp4 | 4 | 96 | 73 | 60 | 58 | 21 | 17 | 15 |
| r2to10bp4 | 10 | 95 | 74 | 60 | 57 | 25 | 17 | 16 |
| r2to10bp4 | 24 | 92 | 70 | 56 | 55 | 22 | 18 | 16 |
| h6r2to10 | 4 | 99 | 60 | 42 | 22 | 15 | 17 | 16 |
| h6r2to10 | 6 | 98 | 59 | 43 | 35 | 28 | 19 | 16 |
| h6r2to10 | 10 | 98 | 57 | 44 | 36 | 28 | 24 | 16 |

Readings:
- **Seeds do not reproduce**: r2to6 trains exactly like loop-confirm's add2to6 with the
  same seeds, yet s0 learned at 10500 there and never here, s2 the reverse. The losses
  agree to 5 digits at step 200 and differ from step 500: nondeterministic GPU kernels,
  amplified by the plateau before the transition. A seed label means nothing past a
  few hundred steps; only rates and means over seeds count. Pooled, add2to6 / r2to6
  learned in 5 of 8 runs.
- **Training on r = 2-10 learns more reliably and better**: 4 of 4 seeds learned
  (7500-18500), 90-98% on 4 hops, and the scan is ahead of r2to6 on every trained hop
  count (4 hops at r = 6: 76 vs 56). Against r2to6's pooled 5 of 8 this is suggestive,
  not significant, and r2to10 also spends 1.5x the compute per step (mean r 6 vs 4).
  r2to10bp4 has r2to10's forward passes but is worse than r2to6, so the gain needs
  gradients through the deeper iterations, not just the deeper forward passes.
- **The usable test-time r grows with the trained range**: r2to10 holds every hop
  count up to r = 16 and loses ~20-30 points at r = 24; r2to6 drops from r = 10-12 and
  collapses at 16-24 (s3 at 0% from r = 16).
- **Extrapolation still stops 1-2 hops past training**: r2to10 answers 5 hops at
  46-49% and 6 hops at 30% from r = 6 (best seed s3: 73% and 42-48%, up from 36% and 23%
  at r = 4), similar to r2to6 (s1: 61% / 41-44%). 8, 10 and 12 hops stay at chance for
  every arm and every r up to 24. A wider iteration range does not extend the reach;
  past r = 6-8 extra iterations add nothing.
- **Truncated BPTT (last 4 of 2-10 iterations) hurts**: 2 of 4 seeds learned, late
  (19000-20000); no 5-hop signal (~25%); it saves only a quarter of the time here (24
  vs 31 min). Its r curve is the flattest (1 hop: 66% at r = 1, 92% at r = 24). Use full
  backpropagation at this scale.
- **Training on 1-6 hops did not learn in 24000 steps** (0 of 4 at 90%): the best seed
  (s1) ends at 85 / 82 / 73 / 73 / 56 on 2-6 hops and was still rising (36% at step
  15000, 78% at 21000); the others are at 30-52%, multi-hop partly or not learned. The
  longer chains delay the multi-hop transition past the budget, so the arm cannot yet
  say whether training on more hops extends the reach. (s1: 5 hops from r = 6, 6 hops
  only at r >= 10 (20% at r = 6, 38% at 10), 8 hops at chance.)

## Curriculum: continuing on 1-6 hops (2026-11-01)

`configs/lc/sweeps/loop-curriculum.yaml`: each run continues the final loop-hops r2to10
model of its seed (1-4 hops, r = 2-10; all 4 learned) for 16000 steps with a fresh
optimizer and cosine schedule (`init_from`), on 1-6 hops (h6, min_len 320) or on 1-4
hops again (h4, the control for the extra training); "start" rescans the starting
checkpoints (steps=0). The scan adds 7 hops and r = 7. ~21 min per continued run.

Final accuracy at 2K on the trained hops (r = 10): start 99 / 96 / 97 / 94; h4 100 / 98
/ 98 / 97; h6 99 / 96 / 96 / 92 / 93 / 90 (1-6 hops). h6 crossed 90% in every seed, after
3000-12500 steps (from scratch, 0 of 4 in 24000 steps); h4 within 500-2500.

Final scan at 2K, mean of 4 seeds (r = 10 and 12 are within 2 points of r = 8; 10 and
12 hops are at chance, <= 27%, everywhere):

| arm | r | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| start | 4 | 98 | 94 | 88 | 54 | 31 | 18 | 18 | 15 |
| start | 6 | 97 | 94 | 90 | 81 | 52 | 26 | 18 | 17 |
| start | 8 | 97 | 93 | 89 | 80 | 53 | 27 | 17 | 18 |
| start | 24 | 90 | 78 | 63 | 50 | 26 | 16 | 7 | 9 |
| h4 | 4 | 100 | 97 | 90 | 71 | 36 | 19 | 17 | 21 |
| h4 | 6 | 100 | 96 | 91 | 87 | 53 | 28 | 16 | 20 |
| h4 | 8 | 100 | 96 | 91 | 88 | 54 | 29 | 15 | 18 |
| h4 | 24 | 71 | 39 | 31 | 26 | 8 | 6 | 4 | 5 |
| h6 | 4 | 100 | 98 | 93 | 67 | 48 | 32 | 18 | 22 |
| h6 | 5 | 99 | 97 | 93 | 90 | 76 | 54 | 33 | 26 |
| h6 | 6 | 99 | 96 | 94 | 89 | 88 | 74 | 50 | 34 |
| h6 | 7 | 100 | 95 | 93 | 90 | 89 | 82 | 58 | 42 |
| h6 | 8 | 100 | 95 | 93 | 89 | 89 | 81 | 62 | 44 |
| h6 | 16 | 95 | 91 | 90 | 85 | 86 | 76 | 55 | 38 |
| h6 | 24 | 54 | 47 | 45 | 41 | 44 | 35 | 24 | 19 |

Per seed (h6, r = 8; 7 / 8 hops): s0 69 / 49, s1 67 / 35, s2 29 / 23, s3 84 / 68. h4 is
at chance on 7 hops in every seed.

Readings:
- **The reach is relative to the trained hops**: after 1-6 hops the model answers 7
  hops at 62% and 8 at 44% (best seed 84 / 68), where the same model continued on 1-4
  hops stays at chance from 7 hops and at the start's 53% / 28% on 5 / 6 hops. Training
  longer chains moves the limit by the 2 hops added; extra training on the old ones
  does not. 10 and 12 hops stay at chance, so the reach is still +1-2 hops, not a
  general loop. The seed that extrapolated least before (s2: 5 hops 30%) also does
  least after (7 hops 29%).
- **Iterations still track hops**: in h6, 5 hops go 48 -> 76 -> 88 for r = 4 -> 5 -> 6,
  6 hops 54 -> 74 -> 82 for r = 5 -> 6 -> 7, 7 hops 50 -> 58 -> 62 for r = 6 -> 7 -> 8; the
  best seed (s3) does 7 hops at 75% already at r = 6 (one hop fewer than iterations).
- **A curriculum is what makes 1-6 hops learnable here**: from the 1-4 hop model they
  take 3000-12500 steps; from scratch 24000 steps were not enough (loop-hops).
- **Continued training narrows the safe r range**: at r = 24, start keeps 90% on 1
  hop, h4 71%, h6 54% (r <= 16 is fine for all). The fresh schedule's second warmup,
  or simply more training at r <= 10, makes far-out iterations drift more.

Next: whether the reach keeps growing with the curriculum (1-6 -> 1-8 -> ...) or
saturates; an iterated-step objective (supervise the state after each iteration, e.g.
the key reached so far) to turn the +1-2 hop reach into a general loop; ~8 seeds for
headline claims.

## Early exit without training (2026-11-08)

The model always runs its full iteration budget (no halting). Does a convergence signal
tell when a query is done? `python -m myk_llm.lc.halting <runs> --hops 1 2 3 4 5 6 7 8
10` (`src/myk_llm/lc/halting.py`) runs each checkpoint once at 24 iterations and reads
the prediction out after every iteration (final norm + head on the loop state; these
models have no coda). Per query, over its answer positions: delta = relative change of
the final-normed state (the add-injected state itself grows by the input every
iteration), KL to the previous iteration's next-token distribution, confidence (the
argmax probability), exact match. An exit rule stops a query at the first iteration
>= 2 that passes its threshold, else at the budget (the top of the training range).
Models: h6 (loop-curriculum, budget 10), start (= loop-hops r2to10, budget 10), r2to6
(loop-hops, budget 6); 4 seeds x 2000 queries at 2K, pooled.

h6, median per hop count (rows) after iteration t (columns), KL / delta:

| hops | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 7 / 1.20 | 5e-4 / 0.45 | 2e-5 / 0.29 | 8e-6 / 0.21 | 4e-6 / 0.16 | 2e-6 / 0.11 | 2e-6 / 0.09 | 9e-7 / 0.08 | 2e-6 / 0.07 |
| 2 | 7 / 1.21 | 2 / 0.58 | 6e-4 / 0.27 | 9e-5 / 0.21 | 3e-5 / 0.19 | 3e-5 / 0.12 | 3e-5 / 0.09 | 2e-5 / 0.08 | 4e-5 / 0.08 |
| 3 | 7 / 1.21 | 0.3 / 0.56 | 0.4 / 0.42 | 6e-4 / 0.22 | 9e-5 / 0.20 | 5e-5 / 0.12 | 4e-5 / 0.09 | 4e-5 / 0.08 | 1e-4 / 0.08 |
| 4 | 7 / 1.20 | 0.04 / 0.53 | 0.8 / 0.48 | 0.05 / 0.33 | 4e-4 / 0.21 | 1e-4 / 0.13 | 8e-5 / 0.11 | 8e-5 / 0.08 | 3e-4 / 0.08 |
| 5 | 7 / 1.20 | 0.04 / 0.55 | 0.2 / 0.48 | 0.2 / 0.39 | 4e-3 / 0.26 | 2e-4 / 0.15 | 1e-4 / 0.11 | 7e-5 / 0.08 | 4e-4 / 0.08 |
| 6 | 7 / 1.20 | 0.04 / 0.54 | 0.06 / 0.48 | 0.2 / 0.43 | 0.03 / 0.32 | 1e-3 / 0.20 | 3e-4 / 0.12 | 2e-4 / 0.08 | 7e-4 / 0.07 |
| 8 | 7 / 1.20 | 0.04 / 0.54 | 0.05 / 0.48 | 0.09 / 0.43 | 0.07 / 0.34 | 0.02 / 0.25 | 3e-3 / 0.15 | 1e-3 / 0.09 | 4e-3 / 0.07 |

Exit rules, accuracy % (mean iterations) on the trained hops and past them:

| rule | h6: 1-6 hops | h6: 7-10 | start: 1-4 | start: 5-10 | r2to6: 1-4 | r2to6: 5-10 |
|---|---|---|---|---|---|---|
| fixed (budget) | 91.5 (10) | 46.7 (10) | 90.0 (10) | 27.3 (10) | 73.5 (6) | 22.4 (6) |
| delta < 0.2 | 91.6 (6.4) | 47.3 (7.8) | 90.1 (5.6) | 26.6 (6.1) | 73.9 (4.7) | 22.2 (5.0) |
| KL < 1e-2 | 89.0 (5.1) | 45.5 (7.2) | 89.5 (4.7) | 26.6 (6.6) | 73.5 (4.2) | 22.6 (5.2) |
| KL < 1e-3 | 91.6 (6.0) | 46.9 (9.1) | 90.2 (5.6) | 27.3 (9.0) | 73.5 (4.8) | 22.6 (5.8) |
| confidence > 0.9 | 91.4 (4.8) | 46.1 (8.5) | 90.1 (4.5) | 27.2 (8.7) | 73.6 (4.2) | 22.3 (5.8) |
| confidence > 0.99 | 91.5 (5.8) | 46.7 (9.3) | 90.0 (5.5) | 27.3 (9.6) | 73.5 (4.5) | 22.4 (6.0) |

h6 at confidence > 0.9, mean exit iteration per hop count 1-8: 2.2, 3.9, 4.7, 5.6, 6.0,
6.6, 7.7, 8.6 (first correct iteration, counting lucky guesses: 2.0, 2.9, 3.5, 4.2,
4.6, 5.2, 5.8, 6.7).

Readings:
- **The loop state does not converge**: delta decays like ~1/t for every query, solved
  or not, to ~0.07 (r2to6: 0.10-0.15), and past t = 8 it is the same for 1 and 10 hops.
  Thresholds below 0.1 never fire within the budget; delta < 0.2 saves iterations by
  exiting everyone at about the same time (h6: 5.7 iterations for 1 hop, 7.9 for 10).
- **The prediction converges one iteration after the chain is resolved**: KL to the
  previous iteration drops by 2-3 orders of magnitude at t = h + 1 (1 hop at 3, 2 at 4,
  ..., 6 at 8), the one-iteration-per-hop pattern seen inside a single pass. Later
  iterations keep moving the state along directions the readout ignores.
- **Output-signal exits keep the accuracy at about half the iterations**: on the
  trained hops confidence > 0.9 matches the fixed budget (h6 91.4 vs 91.5%, start 90.1
  vs 90.0%) at 4.8 / 4.5 of 10 iterations (52-55% saved; r2to6 31% of 6), and exits
  scale with hop count, a median 0 (mean 1) iterations after the first correct one.
  KL < 1e-3 is as accurate but waits ~1 iteration more; KL < 1e-2 starts losing (-2.5
  points on h6).
- **Unsolvable chains spend the budget**: past the reach the model is not confident
  and runs to 8.5-9.6 of 10 iterations, which is where the remaining compute goes
  (a rule cannot tell "not yet" from "never"). Since the output drifts again from r ~16
  (r2to6: ~10), the budget should stay at the top of the training range.
- Caveat: these are per-query exits with every context token at full depth. Prefill
  compute is dominated by the context, so the saving applies to decoding, where an
  exited token's keys / values for later iterations would have to be filled (e.g. with
  its last iteration's, as Huginn does). Untested so far, as is exiting context tokens.

Next for halting: per-token exit in decoding with that key / value fill, to check the
accuracy holds; a learned halting head looks unnecessary on this task, since
confidence already tracks the hop count.
