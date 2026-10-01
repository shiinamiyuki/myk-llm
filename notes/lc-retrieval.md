# Long-context retrieval testbed — experiment design

Status (2026-10-01): implemented (`src/myk_llm/lc/`, `configs/lc/`), calibrated (below);
sweeps E1-E4 written, not yet run. E5 is to be written from E2/E4's results; E6 needs
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
limit and the pure-D gap shrinks. 24 runs.

**E4 — hybrid design for the real model.** On E2's task: ratio of global layers
(1:1, 1:3, 1:7), placement of the global layer in the repeating group (first / last),
mixer of the non-global layers (L vs D), window size, global RoPE first / all / none,
theta. Output: a concrete recommendation for the long-context stage. 20 runs.

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
