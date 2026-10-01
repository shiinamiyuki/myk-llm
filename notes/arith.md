# Arithmetic testbed

Tiny character-level models (a few million parameters) trained only on generated
arithmetic problems, evaluated by exact match of the final answer. Code in
`src/myk_llm/arith/`, configs in `configs/arith/`.

## Why

A fast testbed with ground truth: a run takes minutes, so format, positional and
architecture questions can be answered with dozens of controlled runs. What
transfers to large models are mechanisms (how chain-of-thought (CoT) format, position
information and serial depth interact); absolute numbers and architecture rankings
often do not. Digits are single tokens in our Qwen-derived tokenizer too, so the
number representation matches the big model's.

## Task

Each example is `<prompt>=<completion>$`; the loss covers only the completion. Formats:

| format | example |
|---|---|
| direct | `123+989=1112$` |
| reverse | `123+989=2111$` (answer least-significant digit first) |
| cot (add) | `123+989=3+9+0:12,2+8+1:11,1+9+1:11\|1112$` (one step per digit: a+b+carry:sum) |
| cot (mul) | `37*24=841;047>888\|888$` (partial products and running sums, LSD first) |

`task.reverse_operands` writes the operands least-significant digit first as well.
Training operand lengths are uniform in `[min_digits, train_digits]` per operand;
evaluation uses both operands of length d for each d in `eval_digits` (default
1..2*train_digits), so lengths above `train_digits` measure length generalization.

Problems are packed into 1024-token rows with document masking (they never attend to
each other); data is generated on the fly by DataLoader workers.

## Knobs

- everything in `ModelConfig` (size, layer pattern incl. Gated DeltaNet, RoPE/NoPE,
  QK-norm, gated residuals, n-gram embeddings, ...), via `model.*` overrides;
- `digit_embedding.enabled`: Abacus-style learned index of each digit within its number,
  with a random training offset (McLeish et al., 2024);
- `task.*`: operation, format, operand order, lengths.

## Running

```bash
python -m myk_llm.arith.train configs/arith/add.yaml                       # one run (~1-3 min)
python -m myk_llm.arith.train configs/arith/add.yaml task.format=direct model.num_layers=4
python -m myk_llm.arith.sweep configs/arith/sweeps/format-position.yaml     # grid, resumable
python -m myk_llm.arith.sweep configs/arith/sweeps/format-position.yaml --report
```

A run writes `runs/arith/<run>/{config.yaml, metrics.jsonl, results.json, model.pt}`;
a sweep adds `train.log` per run and `summary.md` with accuracy per length.

Throughput: the 6.3M model trains at ~2.2M tokens/s on an RTX 5090.

## Results

### format-position (sweep 1, 2026-09-29; one seed per cell)

6.3M model, 5000 steps (add, ~3 min) / 10000 steps (mul, ~6 min). Full table:
`runs/arith/format-position/summary.md`. Addition accuracy (%) past the training length
(trained on 1-10 digits; all runs are ~100% on 1-10):

| run | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|
| any format + RoPE | 0 | 0 | 0 | 0 | 0 |
| direct + NoPE | 99 | 0 | 0 | 0 | 0 |
| direct + NoPE + reversed operands | 100 | 92 | 50 | 10 | 1 |
| reverse + NoPE | 100 | 70 | 13 | 1 | 0 |
| reverse + NoPE + reversed operands | 100 | 95 | 40 | 6 | 0 |
| reverse + Abacus (reversed operands) | 85 | 82 | 0 | 0 | 0 |
| cot + NoPE | 100 | 95 | 70 | 20 | 2 |
| cot + Abacus | 100 | 99 | 88 | 50 | 2 |

Multiplication (trained 1-5 digits): every setting is 0% at 6+ digits; in-distribution,
direct answers reach 92-96% at 5 digits while reverse and CoT reach 100%.

Tentative readings (single seed; length generalization is known to vary a lot across
seeds, so treat gaps of a few digits as noise until repeated):
- RoPE gives exactly zero extrapolation. A likely reason: the distance between an answer
  digit and the operand digits it reads grows with the operand length, so longer problems
  need rotation angles (relative distances) never seen in training. Every NoPE variant
  extrapolates by 1-4 digits, then collapses around 13-15.
- CoT helps extrapolation most, but nothing gets anywhere near the 5-10x length
  generalization reported for Abacus embeddings; our Abacus run (random offset up to
  50) is only marginally better than plain NoPE.
- In-distribution the task is too easy to separate settings; comparisons need a tighter
  budget (fewer steps / smaller models) or harder tasks.

### hybrid (sweep 2, 2026-10-01; addition, 2 seeds per cell)

`configs/arith/sweeps/hybrid.yaml`: 6 layers; G = global attention, L = sliding window
of 32 tokens (always RoPE), D = Gated DeltaNet (no positional encoding of its own).
Full table: `runs/arith/hybrid/summary.md`. All cells are 100% on 1-10 digits.

| arch | RoPE | format | +digits (>=50%) | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|
| G | none | revrev | 3.0 | 100 | 100 | 90 | 8 |
| G | none | cot | 2.5 | 99 | 89 | 51 | 9 |
| GD | global | cot | 0.5 | 31 | 0 | 0 | 0 |
| GD | none | cot | 0 | 18 | 0 | 0 | 0 |
| GD, D | any | revrev | 0 | 0 | 0 | 0 | 0 |
| D | - | cot | 0 | 0 | 0 | 0 | 0 |
| GL | none / first / all | both | 0 | 0 | 0 | 0 | 0 |
| G | all | both | 0 | 0 | 0 | 0 | 0 |

(revrev = reversed answer + reversed operands.) Training loss at step 300 (mean of
seeds), cot / revrev: D and GD 0.007-0.011 / 0.03; GL and G-RoPE 0.03-0.06 / 0.29-0.31;
G-NoPE 0.13 / 0.46. Throughput: D/GD 1.2-1.7M tok/s vs 1.8-2.2M for attention only.

Reading: any positional signal - RoPE (even only in the local layers) or the
recurrence / short convolution of DeltaNet - makes the task much faster to learn and
removes length extrapolation; only the fully position-free attention model
extrapolates (by ~3 digits). Adding DeltaNet layers to NoPE global layers removes the
extrapolation that NoPE alone has. Open questions: does this survive longer training
(NoPE learns slowest and may be under-trained at 5000 steps), and is DeltaNet's
positional signal coming from its short convolution?

### GD NoPE, long training (2026-10-01)

GD pattern, no RoPE anywhere, CoT format, 20000 steps (4x sweep 2), 2 seeds; accuracy
evaluated every 2000 steps (200 problems per length). Both runs finished training but
were stopped during the final evaluation (no `results.json`); numbers below are the
in-training evaluations. Accuracy (%) at 11 digits (trained on 1-10):

| seed | 2k | 4k | 6k | 8k | 10k | 12k | 14k | 16k | 18k |
|---|---|---|---|---|---|---|---|---|---|
| s0 | 17 | 7 | 1 | 3 | 1 | 0 | 0 | 1 | 0 |
| s1 | 82 | 51 | 26 | 6 | 24 | 14 | 12 | 26 | 12 |

12 digits: s1 33% at 2k, <= 3% afterwards; s0 <= 1% throughout. In-distribution
(10 digits) rises from 93-97% at 2k to 100%.

Reading: longer training does not unlock extrapolation for this architecture; it is
best early (while the in-distribution fit is still imperfect) and decays as the model
fits the training lengths, i.e. the length-specific solution wins over time. Whatever
generalizes at 2k steps is not what training converges to.

### GD NoPE without the DeltaNet short convolution (2026-10-01)

Same setup as above with `model.deltanet.use_short_conv=false` (no RoPE, no
convolution: the only order information left is the recurrence itself and causal
masking). Accuracy (%) past training, mean of the evaluations at steps 10k-18k
(200 problems per length) and the final evaluation (500 per length):

| run | 11 (10k-18k) | 12 | 13 | final 11 / 12 / 13 |
|---|---|---|---|---|
| with conv, s0 | 0 | 0 | 0 | (not saved) |
| with conv, s1 | 18 | 0 | 0 | (not saved) |
| no conv, s0 | 98 | 84 | 38 | 100 / 98 / 54 |
| no conv, s1 | 100 | 91 | 51 | 63 / 0 / 0 |

Reading: the short convolution is what destroys extrapolation in the GD hybrid.
Without it, extrapolation does not decay with training; over the second half of
training both seeds sit at roughly +2-3 digits, as good as the best pure-NoPE
attention model, while learning almost as fast as with the convolution (in
distribution at 100% by ~8k steps). The value at any single checkpoint is very
noisy (s1 went from 100/79 at 18k to 63/0 at the end), so compare runs by averages
over many checkpoints, and expect checkpoint choice to matter.

### filler augmentation (sweep 3, 2026-10-02; 3 seeds)

`configs/arith/sweeps/filler.yaml`: GD, no RoPE, CoT addition, 20k steps. "filler":
each gap between prompt characters gets 1-3 `_` with probability 0.2 (training and the
filler evaluation; completions are never padded). Mean of second-half evaluations
(`--metric accuracy_late`), accuracy % past the 1-10 digit training range:

| run | +digits | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|
| noconv, filler | 2.7 | 96 | 79 | 50 | 22 |
| noconv, clean | 2.3 | 98 | 82 | 45 | 18 |
| conv, filler | 1.3 | 91 | 36 | 6 | 0 |
| conv, clean | 0.3 | 19 | 1 | 0 | 0 |

Per seed, conv clean -> conv filler at 11 digits: 1 -> 98, 53 -> 82, 3 -> 94.

Reading: randomizing prompt offsets restores about one digit of extrapolation to the
model with the short convolution, consistently across seeds, so the convolution's
harm is (at least partly) a fixed-offset shortcut. It does not close the gap to the
model without the convolution: the CoT itself still has a rigid fixed-width layout
the model can count through, and p=0.2 is a mild perturbation. Fillers neither help
nor hurt the model without the convolution, and models trained with fillers are
equally accurate on clean and filler prompts.
