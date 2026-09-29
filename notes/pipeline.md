# Training pipeline

End-to-end workflow: tokenizer → data → pretraining → context extension → supervised
fine-tuning (SFT) → evaluation. Every stage is one YAML config under `configs/` plus
command-line `key.path=value` overrides.

## 0. Environment

```bash
pip install -e .                       # or: uv pip install -e .
pip install -e '.[linear-attn,dev]'    # Gated DeltaNet layers (flash-linear-attention) + pytest
python -m pytest                       # unit tests (GPU tests are skipped without CUDA)
```

## 1. Tokenizer

The Qwen3.5 vocabulary (248K) is pruned to an English + Chinese vocabulary, and the
token embedding is warm-started from the surviving Qwen rows.

```bash
python scripts/download_corpus.py --lang en --target-tokens 50_000_000 --output data/corpus_en.txt
python scripts/download_corpus.py --lang zh --target-tokens 50_000_000 --output data/corpus_zh.txt
python scripts/prune_tokenizer.py --output _download_models/Qwen3.5-0.8B-pruned64k \
    --corpus data/corpus_en.txt data/corpus_zh.txt --target-vocab 65536
```

Outputs: a loadable tokenizer (with `assets/chatml_no_think.jinja` as its chat
template), `embedding_row_map.npy` (`row_map[new_id] = old_id`) and a report. For a
smaller vocabulary pass e.g. `--target-vocab 32768` (see `configs/experiments/vocab-32k.yaml`).

## 2. Pretraining data

**Pre-tokenized shards (default; use these on servers).** Deterministic, no network
access needed during training, exact resume on any number of GPUs.

```bash
python scripts/pretokenize.py --preset en --target-tokens 5e9 --workers 16 --output data/shards/fineweb-edu-en
python scripts/pretokenize.py --preset zh --target-tokens 5e9 --workers 16 --output data/shards/fineweb-edu-zh
```

Each output has `train/` and `val/` (the first `--val-docs` documents). The shard
format is documented in `src/myk_llm/data/shards.py`; ~4 bytes per token on disk.
Copy the `data/shards/` tree to the server (or run the script there).

**Streaming (`data.kind: stream`)** tokenizes HuggingFace streams on the fly; handy
for quick local runs (`configs/pretrain/200m-stream.yaml`). Each rank reads a disjoint
part of each stream; keep `num_workers x GPUs <=` the source's file-shard count.

Sources are mixed by **token** weight (`data.sources[*].weight`). With
`data.document_masking: true` (default) packed documents never attend to each other
and targets that cross a document boundary are masked.

## 3. Training

```bash
python -m myk_llm.train configs/pretrain/200m.yaml                  # 1 GPU
scripts/launch.sh configs/pretrain/200m.yaml                        # all GPUs of this node
NNODES=2 MASTER_ADDR=node0 scripts/launch.sh configs/pretrain/200m.yaml   # on every node
```

- **Batch size is global.** `global_batch_size` (sequences per optimizer step) is fixed
  by the config; `data.micro_batch_size` is per GPU; gradient accumulation is derived.
  The same config therefore trains identically on 1 or 64 GPUs (only faster).
- **Loss normalization is exact:** every loss is divided by the number of supervised
  tokens in the whole step (all micro-batches, all ranks).
- **Parallelism:** FSDP2 (fully sharded data parallel, `torch.distributed.fsdp.fully_shard`)
  when more than one GPU is used; one FSDP unit per transformer block. Options under
  `parallel:` — `param_dtype: bf16` (less memory / communication), `activation_checkpointing`,
  `reshard_after_forward: false` (more memory, fewer all-gathers), `compile`.
- **Muon** works on sharded parameters (momentum per shard, orthogonalization on the
  gathered matrix; every rank does all matrices, which is fine up to ~1B params).
- **Resume:** re-run the same command; it continues from the newest complete checkpoint
  in `runs/<run_name>/checkpoints/` (model, optimizer, data position). `--resume none`
  starts over, `--resume <dir>` picks a specific checkpoint. Checkpoints use
  torch.distributed.checkpoint (DCP), so they can be resumed on a different GPU count
  (shard data resumes exactly; a streaming loader restarts its stream if the count changed).
- **Warm starts** (`init:`): `embedding_from` + `embedding_row_map` (Qwen rows),
  `weights_from` (an exported model dir, e.g. to continue with a new schedule or seq_len;
  new modules keep their fresh init), `checkpoint_from` (+`load_optimizer`) for a WSD
  (warmup-stable-decay) anneal carrying optimizer momentum.

Run directory: `config.yaml` (resolved config), `metrics.jsonl` (durable log),
`tb/` (TensorBoard), `checkpoints/step_XXXXXXX/` (DCP shards + `model.safetensors`,
`model.yaml`, tokenizer files).

### Learning-rate schedules

`schedule.kind`: `cosine` (warmup → cosine to `min_lr_ratio`), `wsd` (warmup → constant →
cosine decay over the last `decay_ratio`), `constant`. For an open-ended run use
`wsd` with `decay_ratio: 0`, then anneal later with `init.checkpoint_from` +
`init.load_optimizer: true`, `warmup_ratio: 0`, `decay_ratio: 1`.

## 4. Context extension and SFT

`configs/pretrain/200m-long.yaml` continues a finished run at 8K context;
`configs/sft/200m.yaml` fine-tunes on chat data:

```bash
python scripts/build_sft_data.py --out-dir data/sft --max-per-source 200000 --val-frac 0.005
python scripts/tokenize_sft.py --data data/sft/train.jsonl --output data/sft/train
python scripts/tokenize_sft.py --data data/sft/val.jsonl --output data/sft/val
scripts/launch.sh configs/sft/200m.yaml
```

## 5. Architecture experiments

`configs/experiments/ablation.yaml` is a shared ~2B-token recipe; each experiment file
changes one thing:

| Config | Change |
|---|---|
| `attn-qknorm-gate` | QK-norm + sigmoid output gate on attention |
| `attn-rope-all` | RoPE on every global layer (baseline: first only, others NoPE) |
| `hybrid-deltanet` | Gated DeltaNet linear attention instead of sliding-window layers |
| `ngram-embedding` | hashed 2-/3-gram input embeddings |
| `gated-residual` | input-dependent residual gates |
| `mtp` | multi-token prediction (one extra depth) |
| `z-loss` | z-loss on output logits |
| `distill-qwen` | logit distillation from Qwen3.5-0.8B |
| `vocab-32k` | 32K pruned vocabulary |

All model knobs are in `src/myk_llm/model/config.py`; a new variant usually means a
field there plus a branch in `model/mixers.py`, `model/layers.py` or
`model/embeddings.py`. Compare runs by **validation bits per byte** (`val/*/bpb`), which
is independent of the tokenizer. Check speed first with
`python scripts/benchmark_throughput.py configs/experiments/<x>.yaml`.

## 6. Evaluation and chat

```bash
python -m myk_llm.eval.bench --model runs/200m-pt/checkpoints/step_0016000
python -m myk_llm.eval.bench --hf-model HuggingFaceTB/SmolLM2-135M      # reference model
python -m myk_llm.eval.chat_eval runs/200m-sft/checkpoints/step_0006000
python -m myk_llm.chat runs/200m-sft/checkpoints/step_0006000
```

Checkpoints from the old code (`checkpoints/*.pt`) convert with
`python scripts/convert_legacy_checkpoint.py <old.pt> <out_dir>` (verified to match the
old model's logits).
