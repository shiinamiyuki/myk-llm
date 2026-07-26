# Pretraining pipeline & options

Reference for the bilingual (English + Chinese) pretraining pipeline and the
training-script options it adds. Companion to [training.md](training.md), which
holds the higher-level plan and rationale.

Pipeline at a glance:

```
prune tokenizer (248K -> 64K)  ->  warm-start embedding from Qwen rows
        |                                   |
        v                                   v
   stream EN+ZH corpus  ->  train (--stream)  ->  monitor (jsonl + tensorboard + EN/ZH val)
                                   |
                                   v
                    WSD schedule  ->  anneal from a stable checkpoint (--init-from)
```

---

## 1. Tokenizer pruning & embedding warm start

The Qwen3.5 vocab is 248K tokens; ~18% is languages we don't use. We prune to a
64K English+Chinese vocab and warm-start the embedding from the surviving Qwen
rows (free, since `hidden_size = 1024` matches).

Build the pruned tokenizer once (frequency-driven on a representative corpus):

```bash
# assemble a small EN+ZH corpus for frequency stats (token-budgeted, streamed)
python scripts/download_corpus.py --lang en --target-tokens 50_000_000 --output data/corpus_en.txt
python scripts/download_corpus.py --lang zh --target-tokens 50_000_000 --output data/corpus_zh.txt

# prune 248K -> 64K and emit the embedding row map
python scripts/prune_tokenizer.py \
    --output _download_models/Qwen3.5-0.8B-pruned64k \
    --corpus data/corpus_en.txt data/corpus_zh.txt \
    --target-vocab 65536
```

Outputs under `--output`: a loadable tokenizer (`tokenizer.json` etc.),
`embedding_row_map.npy` (`row_map[new_id] = old_id`), and `prune_report.json`.

Warm-start flags for `training.py`:

| Flag | Meaning |
|---|---|
| `--reuse-pretrained-embeddings` | enable embedding warm start |
| `--pretrained-embedding-source <dir>` | full Qwen checkpoint to copy rows from |
| `--pretrained-embedding-row-map <npy>` | gather only the pruned vocab's rows (sets `vocab_size` to 64K) |
| `--freeze-token-embedding` | optional: keep the embedding fixed during training |

Without `--pretrained-embedding-row-map`, the full Qwen embedding must match the
model vocab exactly; with it, the model is a 64K-vocab model warm-started from
the matching rows.

---

## 2. Streaming corpus (`--stream`)

Streams + tokenizes the corpus on the fly — no full local copy on disk, exact
resume. Implemented in [../myk-llm/streaming.py](../myk-llm/streaming.py).

| Flag | Default | Meaning |
|---|---|---|
| `--stream` | off | stream instead of a tokenized memmap |
| `--stream-langs` | `en,zh` | preset sources to interleave (see `PRESETS` in `streaming.py`) |
| `--stream-weights` | `0.5,0.5` | per-language **token** weights (token-balanced interleave) |
| `--stream-shuffle-buffer` | `10000` | per-source shuffle-buffer size |
| `--stream-num-workers` | `4` | DataLoader workers; keep `<=` the source's shard count |
| `--stream-seed` | `0` | base seed (decorrelated per worker) |
| `--stream-eos-token` | `<|endoftext|>` | document separator inserted between docs |
| `--stream-mask-seams` | off | mask cross-document next-token targets (for document-masked attention; see note) |

Notes:
- **Token balance** is exact (interleave draws from whichever language is behind
  on cumulative tokens), so `--stream-weights 0.45,0.55` shifts the *token* mix.
- **Workers**: HuggingFace streams auto-shard across workers. `sample-10BT` has
  14 shards; if a source has fewer shards than workers, the extras sit idle.
- **Presets** live in `PRESETS` in `streaming.py` (currently EN = FineWeb-Edu,
  ZH = chinese-fineweb-edu-v2). Switch EN to `sample-100BT` for budgets above
  ~20B total tokens so English never repeats.
- `--stream-mask-seams` already masks the seam *targets*, but cross-document
  *attention* masking is a later stage; leave it off until then.
- Streaming is pretrain-only (chat/SFT uses the tokenized-memmap path).

---

## 3. Monitoring

Always-on: a durable `metrics.jsonl` is written to `<save-dir>/metrics.jsonl`
(append, survives crashes/resumes). Implemented in
[../myk-llm/metrics.py](../myk-llm/metrics.py).

| Flag | Default | Meaning |
|---|---|---|
| `--tensorboard` | off | also write scalar curves to `<save-dir>/tb` |
| `--eval-interval` | `0` | held-out val loss every N steps (0 = off; streaming only) |
| `--eval-batches` | `8` | batches per language in the fixed eval set |
| `--eval-seed` | `999983` | seed for the (approximate) held-out eval set |

Logged keys:
- **train** (`--log-interval`): `loss`, `loss_ema`, `lr`, `grad_norm`,
  `tok_per_s`, `step_time_s`, `tokens`.
- **eval** (`--eval-interval`): `val_loss_en`, `val_ppl_en`, `val_loss_zh`,
  `val_ppl_zh`, and overall `val_loss` / `val_ppl`.

What to watch: train `loss_ema` + `val_loss` together (both descending, val
tracking train); `grad_norm` for instability (spikes early are fine, an upward
trend is not); and the `val_loss_zh - val_loss_en` gap to decide whether to
retune `--stream-weights`. View live with `tensorboard --logdir <save-dir>/tb`.

The eval set is an *approximate* held-out split (drawn from the stream with a
dedicated seed) — right for trend monitoring, not a benchmark number.

---

## 4. Learning-rate schedule

| Flag | Default | Meaning |
|---|---|---|
| `--lr-schedule` | `cosine` | `cosine` or `wsd` |
| `--warmup-ratio` | `0.01` | warmup fraction of `--steps` |
| `--min-lr-ratio` | `0.1` | LR floor as a fraction of peak |
| `--decay-ratio` | `0.2` | **wsd only**: fraction of steps spent decaying at the end |

- **cosine**: warmup -> cosine decay to the floor across the whole run. The
  schedule is tied to `--steps`, so the token budget is committed up front.
- **wsd** (Warmup-Stable-Decay): warmup -> **constant peak** -> cosine decay
  over the last `--decay-ratio` of the run. Lets you avoid pre-committing the
  budget (see workflows below).

WSD edge cases:
- `--decay-ratio 0` -> stay at peak forever (train open-ended, anneal later).
- `--warmup-ratio 0 --decay-ratio 1` -> pure decay-anneal from peak to floor.

---

## 5. Resume vs. init-from

| | `--resume <ckpt>` | `--init-from <ckpt>` |
|---|---|---|
| model weights | restored | restored |
| optimizer momentum | restored | only with `--init-from-optimizer` |
| LR schedule | restored (continues) | **fresh** |
| step counter | continues | **resets to 1** |
| data-stream position | restored (exact) | **fresh** |

`--resume` continues a run exactly (including the streaming position via the
StatefulDataLoader state in the checkpoint). `--init-from` warm-starts a *new*
run from a checkpoint's weights — used for WSD annealing, continued pretraining
with a new schedule/data, or future length-extension phases. The two are
mutually exclusive.

---

## 6. WSD workflows

**A. Single run, soft budget.** Warmup -> stable -> decay in one run; raise
`--steps` and `--resume` to extend (the decay window slides to the new tail).

```bash
python myk-llm/training.py --stream ... \
    --lr-schedule wsd --warmup-ratio 0.01 --decay-ratio 0.2 --steps <budget>
```

**B. Open-ended, anneal later (decouples the budget).** Train at constant peak,
checkpointing along the way; when you decide the budget, run a short decay-anneal
from the latest stable checkpoint.

```bash
# main run: constant LR, no decay, checkpoint periodically
python myk-llm/training.py --stream ... \
    --lr-schedule wsd --decay-ratio 0 --steps <large> --save-interval <N>

# finalize: short pure-anneal from a stable checkpoint (~10-20% of tokens seen),
# carrying optimizer momentum, fresh schedule from step 1
python myk-llm/training.py --stream ... \
    --init-from <save-dir>/last.pt --init-from-optimizer \
    --lr-schedule wsd --warmup-ratio 0 --decay-ratio 1.0 --steps <anneal-steps>
```

---

## 7. Example: end-to-end test config

See [../training-scripts/train-140M.sh](../training-scripts/train-140M.sh) for a
small (9-layer) streamed + warm-started run with monitoring enabled.
