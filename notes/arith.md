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
