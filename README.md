# MykLLM

A small bilingual (English + Chinese) language model built for architecture
experiments, trained with PyTorch FSDP2 (fully sharded data parallel) on one GPU or many.

```
src/myk_llm/
  configuration.py   YAML -> typed dataclass configs (`base:` inheritance, key=value overrides)
  model/             config, embeddings (token + hashed n-gram), mixers (attention / Gated DeltaNet),
                     layers (norm, SwiGLU, RoPE, residual gates), transformer (+ MTP), generation, io
  ops/fused_ce.py    fused linear + cross-entropy (+ z-loss) Triton kernel
  ops/flash_attn.py  FlashAttention forward/backward Triton kernels (varlen, sliding window, GQA, KV cache)
  data/              pre-tokenized shards, streaming, SFT packing, document filter
  train/             trainer loop, FSDP2 setup, Muon/AdamW, schedules, objectives, DCP checkpoints
  eval/              zero-shot benchmarks, chat sanity eval
  chat.py            interactive chat
configs/             model sizes, pretraining, SFT and experiment configs
scripts/             tokenizer pruning, pre-tokenization, SFT data, launch, benchmarks, legacy conversion
tests/               unit tests (pytest)
notes/               design notes; notes/pipeline.md is the how-to
```

Quick start (see [notes/pipeline.md](notes/pipeline.md)):

```bash
pip install -e '.[linear-attn,dev]'
python scripts/pretokenize.py --preset en --target-tokens 5e9 --workers 16 --output data/shards/fineweb-edu-en
python scripts/pretokenize.py --preset zh --target-tokens 5e9 --workers 16 --output data/shards/fineweb-edu-zh
scripts/launch.sh configs/pretrain/200m.yaml
```
