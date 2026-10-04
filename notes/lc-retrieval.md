# Long-context retrieval testbed — experiment design

Status (2026-10-03): implemented (`src/myk_llm/lc/`, `configs/lc/`), calibrated (below);
sweeps E1-E4 written; E2, E3 and E4 done (results below). E5 is to be written from E2/E4's results; E6 needs
the server checkpoints.

Question: how do the architectures this repo can build (global / sliding-window
attention, Gated DeltaNet, their hybrids, RoPE placement) differ in their ability to
**retrieve** information from long contexts, and how far does that ability extend past
the training length? The answers should feed the long-context stage of the real model
(training.md §4: which layer pattern, which RoPE setup, how long to train).

Approach: the same recipe as the arithmetic testbed (arith.md): tiny models trained from
scratch on generated tasks with ground truth, many seeds, minutes per run. Then one
transfer check on the real 200M ablation models (E6). No model code changes are needed:
every architecture axis below is already a `ModelConfig` field.

## Cost (measured)

8 layers, hidden 256, 4 heads x 64, window 256, RTX 5090, bf16, compiled blocks:

| pattern | train 2K | train 8K | eval forward 32K | eval forward 64K |
|---|---|---|---|---|
| G | 1.39M tok/s | 0.92M | 33 ms, 0.5 GB | 99 ms, 0.6 GB |
| GLLL | 1.52M | 1.35M | 13 ms | 36 ms |
| GD | 1.17M | 0.95M | 24 ms | 67 ms |
| DDDG | 1.07M | 0.95M | 17 ms | 43 ms |
| D | 0.99M | 0.95M | 13 ms | 26 ms, 1.0 GB |

A 5000-step run at 64K tokens/step (330M tokens) is ~4-6 min; evaluating up to 64K
context is nearly free, so every run can be evaluated far past its training length.

## Tasks

Token-level synthetic vocabulary (464 ids): specials (`PAD`, `EOS`, `?` query, `;`
separator), a key alphabet (128 symbols; a key = 2 tokens, 16K keys), a value alphabet
(64 symbols; a value = 4 tokens, chance accuracy ~1e-7) and a noise alphabet (256
symbols). Like arith, examples are packed into rows with document masking, and the loss
covers only the answers. One generator covers both tasks below: an *item* is a chain of
h hops, and h = 1 is a plain record.

**A. Associative recall** (`hops: [1]`). A context of `N` records `k k v v v v ;`
at random positions among noise tokens, padded to length `L`; then `Q` queries
`? k k` -> `v v v v EOS`, each about a record anywhere in the context. Two regimes
(`configs/lc/needle.yaml`, `configs/lc/capacity.yaml`):

| regime | N | noise | measures |
|---|---|---|---|
| capacity (MQAR, Zoology) | large (16-512), grows with L | none | how many associations the model can hold (state size) |
| needle (multi-key NIAH, RULER) | small (Q-8) | fills the rest | finding a few needles in a long, mostly irrelevant context |

Every query records its needle's depth (position / L), so accuracy can be reported per
length x depth cell. Q queries per example (instead of one) make the training signal
less sparse. Optional: queries for absent keys with answer `NONE` (false recall).

**B. Multi-hop chains** (`hops: [1, 2, 3, 4]`, `configs/lc/chain.yaml`; RULER's
variable tracking). Records are pointers `k_a k_b ;` and terminals `k_z v v v v ;`,
scattered in random order among noise; the query `? k_a` asks for the value reached after `h` hops (h = 1-4). Two answer
formats: direct (`v v v v EOS`) or chain-of-thought (CoT; every intermediate key, then
the value), which turns one h-hop lookup into h single-hop lookups — the same format
question as in arith, now for retrieval.

**C. Copy** (not implemented as an evaluation task). Copy examples are used in the
training mix instead (see calibration).

## Evaluation

- **Teacher-forced exact match:** one forward pass; a query is correct when the argmax
  at every answer position equals the target. This is exactly greedy-decoding exact
  match (if every step's argmax equals the gold token, greedy reproduces the gold
  answer), at the cost of a single forward even at 64K.
- **Grid:** length L in {512, 1K, 2K, 4K, 8K, 16K, 32K, 64K} x 10 depth buckets, 500+
  queries per cell. Training uses L uniform in [L_min, L_train], L_train = 2K by default.
- **Length is decoupled from load:** at evaluation the needle regime keeps N fixed while
  L grows (pure distance / dilution), the capacity regime keeps the record density fixed
  (N grows with L). Otherwise "fails at 16K" mixes up "too far" and "too many".
- **Summaries:** effective length (largest L at >= 90% / >= 50%), length x depth
  heatmaps, accuracy vs N curves; mean of the evaluations over the second half of
  training (`accuracy_late`), since arith showed that single checkpoints are noisy
  out of distribution. >= 2 seeds per cell, 3 where cells disagree.

## Architectures

8 layers, hidden 256 (~8M non-embedding parameters), 4 heads x 64, sliding window
256 (< L_train, so local layers cannot see across the context); DeltaNet 4 heads x 64.
Patterns tile, so GLLL / DDDG appear twice.

| id | pattern | position | why |
|---|---|---|---|
| G-rope | G | RoPE all | standard transformer |
| G-nope | G | none | arith: only position-free attention extrapolated |
| GL | GL | RoPE on first G + local layers | the 200M recipe |
| GLLL | GLLL | same | the 600M recipe (1 global : 3 local) |
| GLLL-nope | GLLL | global layers NoPE | locals keep RoPE |
| L | L | RoPE | negative control: cannot retrieve past the window |
| D / D-noconv | D | — | pure Gated DeltaNet, with / without short conv |
| GD | GD | none | 1 : 1 hybrid (hybrid-deltanet experiment) |
| DDDG | DDDG | none | 3 : 1 hybrid (Qwen3-Next-style ratio) |

Plus attention variants on GLLL: QK-norm + output gate (`attn-qknorm-gate`), window
64 / 256 / 1024, global RoPE theta 1e4 vs 1e6.

## Experiments

**E0 — calibration (done; see below).**

**E1 — capacity.** `assoc` capacity regime, L = 2K, N in {32, 128, 256} (7 tokens per
record, so 256 records fill most of the 2K); patterns G-rope, L, D, D-noconv, GD, DDDG,
GLLL; plus D with head_dim 128 (4x the state).
Hypotheses: attention is flat in N; D degrades once N outgrows its state, and the
degradation tracks state size; one global layer (GD, DDDG) restores attention-level
recall. 48 runs.

**E2 — retrieval at length and extrapolation (main experiment).** `assoc` needle regime,
train L <= 2K, evaluate to 64K, every architecture above. Questions: which positional
setups keep retrieving past 2K (arith: RoPE never extrapolated; does retrieval behave
the same way?); does NoPE fail gradually through softmax dilution as L grows; does D
show a recency bias (early needles lost); do window layers with RoPE (GL, GLLL) inherit
RoPE's failure even when the global layers are NoPE? 20 runs.

**E3 — multi-hop.** `chain`, h = 1-4 mixed in training, evaluate per h up to 16K,
direct vs CoT; G-rope, G-nope, GLLL, DDDG, GD, D. Hypotheses: direct h-hop answers need
several global layers (DDDG with 2 global layers fails at h >= 3?); CoT removes that
limit and the pure-D gap shrinks. 24 runs as written; run with 4 more architectures (40)
plus 6 longer runs (results below).

**E4 — hybrid design for the real model.** On E2's task: ratio of global layers
(1:1, 1:3, 1:7), placement of the global layer in the repeating group (first / last),
mixer of the non-global layers (L vs D), window size, global RoPE first / all / none,
theta. Output: a concrete recommendation for the long-context stage. 20 runs, plus a
14-run follow-up (results below).

**E5 — train long vs extrapolate.** The best 3 configurations from E2/E4: trained at
2K only; at 2K then 10% of steps at 8K (the staged recipe of training.md §4c, in
miniature); at 8K throughout. Does a short long-context phase recover what
extrapolation misses?

**E6 — transfer check on the real 200M models.** Needle in natural text: a sentence
like "The access code for Room 42 is 8 3 1 0 6." inserted at depth d into FineWeb
validation text, a question at the end, scored by teacher-forced exact match. Run on
the ablation checkpoints (baseline GL, `hybrid-deltanet`, `attn-rope-all`,
`attn-qknorm-gate`) at 1K-8K, before and after an 8K continuation
(`configs/pretrain/200m-long.yaml`). Checks whether the synthetic ranking predicts the
real one. Needs the trained checkpoints (server).

Run counts as written: E1 48, E2 20, E3 24, E4 20 runs. With 8000 steps a run takes
~8-12 min (attention faster than DeltaNet, plus evaluations up to 64K), so E1-E4 are
~18 GPU-hours on one 5090; E2 alone (the main result) is ~3.5 h.

## Calibration (E0, 2026-10-01)

Retrieval is learned through an abrupt transition after a plateau at chance, and with
the first version of the training data the plateau outlasted every run:

1. **Answer-only loss alone never formed the circuit.** G-rope on the needle task
   stayed at ~15% exact match for 5000 steps. Diagnosis: the first value token was
   right 19% of the time (chance among 8 records: 12.5%); given it, the rest was copied
   at 97%. So the model copied *a* value but never matched the key. Same on dense
   multi-query associative recall (MQAR): 8000 steps at chance, with Muon or AdamW, 2 or
   8 layers. The data was checked: a one-step induction oracle (find the earlier
   occurrence of the query key, read the next token) scores 100%.
2. **A plain PyTorch transformer gets out of the plateau, MykLLM does not.** Same
   data, 16 records: the reference model jumps from chance to 100% between 6K and 12K
   steps (4 records: before 6K); MykLLM stays at chance for 12K steps (RoPE base 1e6 vs
   1e4, tied vs untied embeddings, init std 0.02 vs 0.1: no difference). At 4 records it
   stalls at ~50%, which is what answering "some record not asked yet" scores
   ((1/4 + 1/3 + 1/2 + 1) / 4): elimination, no key matching.
3. **MykLLM's induction machinery is fine:** on the classic repeated-random-sequence
   task (dense loss) it reaches 100% within 500 steps.
4. **Fix: copy examples in the training mix** (`task.copy_fraction: 0.25`: a random
   span, then the same span, loss on the repeat). Their dense loss forms the induction
   circuit, which retrieval then uses: MQAR with 16 records reached ~100% by 4K steps.
   This mirrors real pretraining, where induction heads come from repeated text. Also in
   the mix: 1..N records per example (`min_records: 1`) and log-uniform lengths
   (128-2048), so short dense examples are common. Separately, the query format lost its
   `>` separator (`? k k` instead of `? k k >`): the query then ends with the key, so
   a plain induction head (match the current token) answers it.

With that mix (needle task, 5000 steps; `learned_step` = first evaluation at >= 90% in
distribution):

| arch | learned at | 512 | 2K | 4K | 8K | 16K | 32K | 64K |
|---|---|---|---|---|---|---|---|---|
| G-rope | 1500 | 100 | 100 | 96 | 62 | 4 | 0 | 0 |
| D | 4500 | 92 | 92 | 93 | 91 | 88 | 74 | 35 |

G-rope at 8K, by needle depth (0% = start of the context): 29, 25, 22, 55, 58, 71, 78,
90, 95, 95 — it finds needles within the training distance and loses the far ones. D at
16K is flat across depths (77-99). The sweeps use 8000 steps so the slowest learners
(DeltaNet) are past the transition with margin; `learned_step` is reported per run so
a late transition is not mistaken for inability.

## E2 results: retrieval at length (2026-10-02)

`configs/lc/sweeps/e2-length.yaml`, 2 seeds, 8000 steps, trained on 128-2048 tokens;
8 records in noise, 4 queried; final checkpoint, 1000 queries per length. eff@90 /
eff@50: longest length up to which accuracy stays >= 90% / 50% (mean of seeds).
Full table: `runs/lc/e2-length/summary.md`.

| arch | learned at | eff@90 | 2K | 4K | 8K | 16K | 32K | 64K |
|---|---|---|---|---|---|---|---|---|
| DDDG (global NoPE) | 500-1000 | 64K | 100 | 100 | 100 | 100 | 100 | 100 |
| GLLL, global NoPE | 1500 | 48K | 100 | 100 | 100 | 100 | 98 | 84 |
| G NoPE | 1500 | 32K | 100 | 100 | 100 | 100 | 97 | 74 |
| GLLL, RoPE on first G (600M recipe) | 1500 | 24K | 100 | 100 | 100 | 99 | 90 | 61 |
| GL, RoPE on first G (200M recipe) | 1500 | 24K | 100 | 100 | 100 | 99 | 87 | 42 |
| D | 3000-3500 | 24K | 100 | 99 | 99 | 98 | 90 | 32 |
| D, no short conv | 3500-4000 | 18K | 99 | 98 | 90 | 53 | 47 | 30 |
| GD (global NoPE) | 1000-1500 | 16K | 100 | 100 | 99 | 97 | 68 | 22 |
| G RoPE | 1000 | 4K | 100 | 95 | 59 | 18 | 3 | 0 |
| L (window 256) | 6000-6500 | 1K | 76 | 39 | 19 | 10 | 6 | 2 |

Seed spread is large past 16K for some rows (64K: GL 76 / 8, GLLL 89 / 32, D 22 / 42,
D no-conv 0 / 59 — its seed 0 already fails at 16K, 8%); the top three rows are
consistent across seeds (DDDG 100 / 100; GLLL-NoPE 98 / 70; G-NoPE 65 / 82).

Depth profile at 16K: G-RoPE finds only recent needles (1-6% in the first 60% of the
context, 88% in the last tenth); L only within its stacked receptive field (8 x 256);
every other model is flat across depths (GD slightly lower for the earliest needles,
91-93%). No recency bias in DeltaNet at this load.

Readings:
- **RoPE on global layers is what breaks extrapolation**, as in arith: G-RoPE collapses
  past 2x the training length, G-NoPE holds 97% at 16x. One RoPE global layer among
  NoPE ones (the 200M / 600M recipes) costs a lot past 16K (GLLL 90 / 61 vs GLLL-NoPE
  98 / 84 at 32K / 64K). Local layers with RoPE do not hurt: GLLL-NoPE >= G-NoPE.
- **The 3:1 DeltaNet hybrid with NoPE global layers is the best model here**: perfect
  to 64K on both seeds, and it learned fastest. Pure DeltaNet extrapolates well to 32K
  but fades at 64K; its short convolution helps retrieval (without it, one seed fails
  at 16K) — the opposite of arith, where the convolution destroyed length
  generalization (content lookup vs position-aligned computation).
- **GD (1:1) is worse than DDDG** past 16K although it has more global layers. One
  difference is the last layer (D in GD, global attention in DDDG); E4's GDDD vs DDDG
  tests placement directly.
- Caveats: light load (8 records); E1 (capacity) is where DeltaNet's fixed state should
  matter. RoPE + training-free dynamic YaRN (next section) also reaches 100% to 64K, so
  "RoPE vs NoPE" is really "plain RoPE vs NoPE or scaled RoPE".

## E4 results: hybrid design (2026-10-03)

`configs/lc/sweeps/e4-hybrid.yaml` (20 runs) and `configs/lc/sweeps/e4b-followup.yaml`
(14 runs), same task and protocol as E2. The written sweep varied ratio and placement
with RoPE on the first global layer, which E2 had already shown to be harmful, so the
follow-up repeats them with NoPE global layers, adds the global layer last at 1:1 and
1:7, tests the window-1024 failure without RoPE, and gives GDDD / DDDG a third seed.
Full tables: `runs/lc/e4-hybrid/summary.md`, `runs/lc/e4b-followup/summary.md`.

Layer pattern, ratio and placement (global layers listed by index, R = RoPE; mean of
seeds, per-seed 64K in brackets where the seeds disagree):

| pattern | global layers | 2K | 8K | 16K | 32K | 64K |
|---|---|---|---|---|---|---|
| **DG** (1:1) | 1 3 5 7 | 100 | 100 | 100 | 100 | 100 |
| **DDDG** (1:3, 3 seeds) | 3 7 | 100 | 100 | 100 | 100 | 100 |
| **DDDDDDDG** (1:7) | 7 | 100 | 100 | 100 | 100 | 100 |
| **LLLG** NoPE (1:3) | 3 7 | 100 | 100 | 100 | 100 | 100 |
| **LLLLLLLG** NoPE (1:7) | 7 | 100 | 100 | 100 | 100 | 100 |
| GLLL NoPE (E2) | 0 4 | 100 | 100 | 100 | 98 | 84 (98 / 70) |
| GLLL, RoPE first (E2; 600M recipe) | 0R 4 | 100 | 100 | 99 | 90 | 61 (89 / 32) |
| GL, RoPE first (E2; 200M recipe) | 0R 2 4 6 | 100 | 100 | 99 | 87 | 42 (76 / 8) |
| GDDD (3 seeds) | 0 4 | 100 | 99 | 84 | 69 | 65 (92 / 3 / 100) |
| GD (E2) | 0 2 4 6 | 100 | 99 | 97 | 68 | 22 |
| GDDDDDDD | 0 | 100 | 100 | 90 | 42 | 11 (22 / 0) |
| GLLLLLLL, RoPE first | 0R | 98 | 74 | 53 | 29 | 12 (0 / 24) |
| LLLG, RoPE first | 3R 7 | 100 | 61 | 27 | 12 | 6 |
| GLLLLLLL NoPE | 0 | 75 | 17 | 9 | 4 | 1 |

Window, RoPE and attention variants, all on GLLL (global layers 0 and 4):

| variant | 2K | 4K | 8K | 16K | 32K | 64K |
|---|---|---|---|---|---|---|
| window 256, RoPE first (E2 baseline) | 100 | 100 | 100 | 99 | 90 | 61 |
| window 64 | 100 | 100 | 99 | 85 | 24 | 2 |
| window 1024 | 100 | 72 (49 / 96) | 58 | 43 | 17 | 0 |
| window 1024, NoPE | 100 | 74 (100 / 49) | 62 | 56 | 51 | 44 (84 / 3) |
| RoPE on all global layers (theta 1e6) | 100 | 96 | 66 | 26 | 6 | 1 |
| RoPE on all, theta 1e4 | 100 | 56 | 25 | 9 | 2 | 0 |
| QK-norm + output gate | 100 | 100 | 98 | 69 | 3 | 0 |
| G NoPE + QK-norm + output gate (vs G NoPE in E2) | 100 | 100 | 100 | 99 (100) | 78 (97) | 44 (74); per seed 0 / 89 |

Every run learned the task in distribution (`learned_step` 500-3500) except GLLLLLLL
NoPE (seed 0 never, seed 1 at 6500).

Readings:
- **A NoPE global layer as the last layer solves the task at every ratio.** DG, DDDG,
  DDDDDDDG, LLLG and LLLLLLLG (NoPE) are all 100% at every length to 64K on every seed
  (2-3 each), flat across depths. With that placement, the ratio (1:1 to 1:7) and the
  other mixer (local attention vs DeltaNet) make no difference on this task; one global
  layer in 8 is enough. Same ratio, shifted by one layer: GD 22% at 64K, DG 100%.
- **A global layer at layer 0 does not retrieve.** GLLLLLLL NoPE, whose only global
  layer is layer 0, never learns the task even at 2K; it finds only needles inside the
  local layers' reach (0% for the first 10% of a 2K context). Plausible reason: at layer 0
  no earlier layer has written the key into the value tokens yet, so the induction
  pattern (match the key, read what follows) is impossible. GDDDDDDD (only layer 0
  global) behaves like pure DeltaNet (E2 D: 98 / 90 / 32), i.e. the DeltaNet layers do
  the retrieval. So in the GL / GLLL recipes, the RoPE layer 0 is mostly not the
  retrieval layer; this is why "RoPE on the first global layer" costs only moderately in
  E2. Moving that RoPE layer to where retrieval happens (LLLG with RoPE first: RoPE on
  layer 3) gives the plain-RoPE failure: 27% at 16K, finds only the most recent needles
  (0-4% in the first half, 96-100% in the last fifth), same as RoPE on all global layers.
- **Global layers that are not last cost accuracy past 16K** even with NoPE (GLLL NoPE
  84, G NoPE 74, GD 22, GDDD unstable: 92 / 3 / 100 at 64K, the failing seed with a
  recency profile). Hypothesis, not tested: a NoPE layer that does not attend sharply
  outputs an average over the whole context, which drifts with L, and every layer
  after it sees that drift; a last layer passes it only to the LM head.
- **RoPE on retrieving global layers fails as in E2**; theta 1e4 is worse than 1e6 (56%
  vs 96% at 4K), as expected from more dimensions completing full turns within 2K.
- **The window has to stay well below the training length.** With window 1024 the six
  local layers cover the 2K training context, and per seed the model learns either the
  global-layer solution (flat across depths, extrapolates: NoPE seed 0, 84% at 64K) or a
  local one that is bound to the training distance (at 4K: 0-1% for needles in the first
  half, 99-100% in the last 2K). This happens with and without RoPE on the global layers.
  Window 64 is worse than 256 too (with RoPE first; not tested with NoPE).
- **QK-norm + output gate did not help**: worse on GLLL on both seeds (69% vs 99% at 16K);
  on G NoPE the seeds disagree (0% and 89% at 64K vs 74% without). The norms carry
  learnable scales, so the reachable logit scale is learned; whether it gets large
  enough for sharp attention at 64K seems seed-dependent.
- Caveats: light load (8 records, 4 queried); the configurations that saturate here
  must be separated on capacity (E1) and multi-hop (E3) before the ratio is cut to 1:7.
  Tiny model (8 layers); the 200M / 600M stacks are 9 / 33 layers.

**Recommendation for the long-context stage** (to verify with E6 at 200M scale):
1. Global layers NoPE (`global_rope: none`). If RoPE stays (it is in the trained 200M
   checkpoints), keep it only on layer 0, which barely retrieves, and plan on dynamic
   YaRN for that layer (RoPE extension section below).
2. End the stack with a global layer. The 9-layer GL (globals 0, 2, ..., 8) and 33-layer
   GLLL (0, 4, ..., 32) patterns already do; the global layer 0 contributes little to
   retrieval, so a new design can put the local / DeltaNet layers first (LLLG, DDDG).
3. Ratio: 1:3 (DDDG / LLLG). Here even 1:7 was perfect, but E3 shows 1:7 cannot
   answer multi-hop questions directly (2 hops 57%, 3-4 hops at chance), while 1:3
   reaches 83-97% and 1:1 90-96%.
4. Window: do not raise it to 1K-4K for the long stage as training.md §4a suggests
   without testing it; here a window large enough for the local layers to cover the
   training context let some seeds learn a retrieval path that stops at the training
   length.
5. Local attention vs DeltaNet: no difference in retrieval at this load; decide on E1
   (state capacity) and throughput.
6. QK-norm + output gate: no gain for long-context retrieval; do not adopt for that reason.

For E5: the configurations above are already at 100% to 64K, so a long-training phase can
only show something on those that fail: the current recipes (GL / GLLL with RoPE first),
GD, D; or on a heavier task (more records).

## E3 results: multi-hop chains (2026-10-03)

`configs/lc/sweeps/e3-chain.yaml`: 8 chains of 1-4 hops (mixed) in noise, 4 queried,
trained on 256-2048 tokens, evaluated to 16K; answers direct (`v v v v`) or CoT (every
intermediate key, then the value). The written six architectures plus E4's best placement
(NoPE global layer last) at 1:1 / 1:3 / 1:7: DG, DDDDDDDG, LLLG, LLLLLLLG. 2 seeds,
8000 steps (40 runs). Some direct runs were still improving at 8000 steps, so
`configs/lc/sweeps/e3b-long.yaml` repeats the direct format for 6 architectures with
24000 steps (seed 0 only). Full tables: `runs/lc/e3-chain/summary.md`,
`runs/lc/e3b-long/summary.md`.

Accuracy per hop count (h = 1 / 2 / 3 / 4), mean of seeds:

| arch | CoT 2K | CoT 16K | direct 8K steps, 2K | direct 24K steps, 2K | direct 24K steps, 16K |
|---|---|---|---|---|---|
| DG (1:1) | 100 all | 100 all | 58 / 34 / 33 / 16 | 100 / 96 / 95 / 96 | 100 / 96 / 93 / 90 |
| DDDG (1:3) | 100 all | 100 all | 99 / 17 / 17 / 16 | 100 / 97 / 90 / 83 | 100 / 95 / 92 / 81 |
| LLLG NoPE (1:3) | 100 all | 100 all | 98 / 16 / 15 / 16 | 100 / 94 / 94 / 79 | 100 / 94 / 85 / 47 |
| DDDDDDDG (1:7) | 100 all | 100 all | 99 / 15 / 19 / 16 | 100 / 57 / 17 / 14 | 99 / 59 / 17 / 13 |
| LLLLLLLG NoPE (1:7) | 100 all | 100 all | 93 / 16 / 15 / 17 | | |
| GD | 100 all | 100 all | 99 / 19 / 19 / 18 | | |
| GLLL, RoPE first | 100 all | 94-96 | 99 / 16 / 15 / 17 | | |
| G NoPE | 100 all | 98 / 96 / 93 / 91 | 98 / 68 / 18 / 18 | 100 / 99 / 73 / 39 | 100 / 98 / 69 / 31 |
| G RoPE | 100 all | 34 / 20 / 10 / 6 | 99 / 89 / 89 / 24 | 100 / 94 / 72 / 57 | 19 / 3 / 4 / 3 |
| D | 100 / 99 / 99 / 98 | 99 / 98 / 96 / 96 | 79 / 17 / 15 / 15 | | |

Guessing one of the 8 values scores 12.5%, so 15-19% means the hop count is not solved.
CoT runs learned at 1000-2500 steps (D 4000-5000); in 24000 steps, direct runs learned
(>= 90% overall) at 13500 (DG), 14000 (LLLG), 21500 (DDDG), never (G NoPE, G RoPE,
DDDDDDDG).

Readings:
- **With CoT, multi-hop is as easy as single-hop.** Every architecture learns every hop
  count within 2500 steps (D: 5000), and accuracy at length follows E2 / E4: the NoPE
  global layer last stays at 100% to 16K at every ratio, G RoPE fails past 2K (72% at 8K
  over all hops), the rest hold 91-99% at 16K.
- **Direct multi-hop is learned late, not impossible for hybrids.** At 8000 steps
  every hybrid answered only 1-hop queries; with 24000 steps DG and DDDG reach 83-97% on
  2-4 hops, against the design's hypothesis that DDDG (2 global layers) fails at h >= 3.
  E3 at 8000 steps alone would have read as "hybrids cannot do multi-hop".
- **Direct multi-hop depends on the number of global layers among hybrids**: 1:1 (4
  global layers) > 1:3 (2) > 1:7 (1: 2 hops 57%, 3-4 at chance after 24000 steps). This is
  the first task where the 1:7 ratio falls short. The all-global models are not best:
  G NoPE (8 global layers) reaches only 73 / 39% on 3 / 4 hops and learns more slowly
  than DG; G RoPE does best on 4 hops in distribution (57%) but fails past 2K like
  every RoPE model.
- **Hybrids keep their direct multi-hop answers at length.** DG and DDDG lose little from
  2K to 16K (4-hop 96 -> 90, 83 -> 81); LLLG loses more on 4 hops (79 -> 47).
- Instability: DG direct seed 0 at 8000 steps learned nothing, not even 1 hop (17%
  overall); seed 1 was at 99 / 56 / 49% on 1-3 hops and rising.
- Caveats: the 24000-step runs are one seed each, and the cosine schedule stretches
  with the step count (so the learning rate stays high longer, not only more steps).
  Pure D is weaker on 1-hop in this mix (79% direct vs 100% in E2's needle task).

For the real model: real text rarely spells out intermediate steps, so the direct format
is the relevant one for multi-hop recall inside a forward pass. That argues against 1:7;
1:3 with the global layer last is the compromise (1:1 is better and costs more).

## RoPE extension on the G-rope checkpoint (2026-10-01)

The calibration G-rope model (8 global layers, RoPE base 1e6 on all, trained on <= 2048
tokens) with the RoPE tables rewritten at evaluation, s = L / 2048 at each length (scratch
script; not yet a config option). Needle task, 500 queries per length:

| method | 2K | 4K | 8K | 16K | 32K | 64K |
|---|---|---|---|---|---|---|
| plain RoPE | 100 | 97 | 63 | 5 | 0 | 0 |
| position interpolation (PI) | 100 | 0 | 0 | 0 | 0 | 0 |
| NTK-aware base | 100 | 99 | 96 | 65 | 8 | 0 |
| YaRN frequencies, no temperature | 100 | 100 | 99 | 85 | 46 | 9 |
| temperature only (0.1 ln s + 1 on q and k) | 100 | 98 | 77 | 40 | 14 | 5 |
| **YaRN** (frequencies + temperature) | 100 | 100 | 100 | 100 | 100 | 100 |

Rechecked with another seed and 2000 queries: YaRN 99.9% at 32K, 99.6% at 64K (97.6%
for needles in the first tenth of the context). Then 500 steps of fine-tuning at 8K
(30 s) with s = 4, evaluated with s fixed at 4 ("static") or s = max(4, L / 2048):

| fine-tuned with | 8K | 16K | 32K | 64K |
|---|---|---|---|---|
| plain RoPE | 99 | 84 | 10 | 0 |
| PI, static / dynamic | 99 / 99 | 42 / 36 | 0 / 0 | 0 / 0 |
| NTK, static / dynamic | 100 / 100 | 92 / 99 | 40 / 88 | 1 / 28 |
| YaRN, static / dynamic | 100 / 100 | 95 / 100 | 69 / 100 | 19 / 99 |

Readings: the extrapolation failure of RoPE has two parts, rotation angles never seen
in training (YaRN's frequency ramp fixes most of it) and attention spreading over more
tokens (the temperature fixes most of the rest); both are needed. PI breaks the
high-frequency dimensions and only works after fine-tuning, and then only up to the
fine-tuned length. Fine-tuning at 8K lets every method reach 8K, but beyond it only the
scaling matched to the actual length (dynamic) keeps working, and training-free dynamic
YaRN was already perfect. Caveats: one checkpoint and seed, and a synthetic task with
unique keys; YaRN usually needs some fine-tuning on real language models.

## Running

```bash
python -m myk_llm.lc.train configs/lc/needle.yaml model.layer_pattern=GD   # one run
python -m myk_llm.lc.sweep configs/lc/sweeps/e2-length.yaml                 # grid, resumable
python -m myk_llm.lc.sweep configs/lc/sweeps/e2-length.yaml --report --at 16384
```

Code: `src/myk_llm/lc/` (`tasks.py` generators, `data.py` packing, `eval.py` scoring,
`train.py`, `sweep.py` with the report; the grid runner is shared with arith). A run
writes `runs/lc/<sweep>/<run>/{config.yaml, metrics.jsonl, model.pt, results.json,
train.log}`; `results.json` has accuracy per length, per depth bucket and per hop count,
`accuracy_late` (mean of the second-half evaluations from `learned_step` on) and `learned_step`. Not implemented yet: E5's sweep, E6's
`scripts/eval_niah.py`.

## Pitfalls to watch

- Token-identity shortcuts: keys live in their own alphabet, so in the needle regime a
  model can find needles by token type alone. That is intended (RULER's basic NIAH);
  the capacity regime and the chain task remove the shortcut. If needed, add near-miss
  distractors (keys sharing their first token with the query).
- Packed rows use row-level position ids; with RoPE only relative distances matter, so
  training sees distances <= L_train, which is the extrapolation being tested.
- Copy examples train induction directly, which may favor some architectures (a
  recurrent state copies short spans fine; spans are 16-128 tokens). Keep the fraction
  fixed across compared runs.
