# Training MykLLM — notes & plan

> **Note (2026-09):** the code moved to the `myk_llm` package (`src/myk_llm/`); file and flag names below refer to the old `myk-llm/` scripts. See [pipeline.md](pipeline.md) for the current workflow.

Target: ~0.6B-param LLM, Qwen3.5-0.8B tokenizer, trained on an RTX 5090 (32 GB).
Goal: conversational ability, not deep world knowledge. Stretch goal: 64K context.

References to current code use `myk-llm/model.py` and `myk-llm/training.py`
(both now live in the `myk-llm/` subdirectory). Specific line numbers below are
approximate — refer to symbols (functions/classes), not line numbers.

> **Status (updated 2026-06-29).** Most of this plan has landed. §1 (Qwen
> embedding reuse) and **all of §3** (LR schedule, fp32 master weights,
> grad-accum, param groups, memmap data, SFT masking, chunked/fused loss) are
> implemented, plus things not in the original plan: a **Muon optimizer**
> (now the default), **residual-projection scaling**, and **`rope_theta` as a
> config field** (default `1e7`). The still-open / aspirational parts are the
> dataset choices (§2), distillation (§1), the 64K curriculum (§4), and MTP
> (§5). See the per-section notes for what's done vs. outstanding.

---

## 1. Reuse Qwen3.5's embedding matrix — yes

The embedding is a large fraction of a 0.6B model. With `hidden_size = 1024`
(8 heads × 128 head_dim) and Qwen's ~152K vocab, the embedding is
~152K × 1024 ≈ 155M params — roughly a quarter to a third of the whole model.
Learning that from scratch wastes a lot of the token budget, and Qwen's
embeddings already encode token semantics for the *exact* tokenizer we use.

Because we tie embeddings (`MykLLM.__init__`, `lm_head.weight =
token_embedding.weight`), copying one matrix initializes both the input
embedding and the `lm_head`.

### Status: implemented

This is wired up via `--reuse-pretrained-embeddings`
(`load_pretrained_token_embedding` + `initialize_token_embedding_from_pretrained`
in `training.py`). The caveats below are all handled:

- **Vocab size mismatch — handled.** When reusing embeddings, `vocab_size` is set
  to the checkpoint's embedding row count (`pretrained_embedding_weight.shape[0]`),
  not `len(tokenizer)`. `len(tokenizer) > vocab_size` is rejected with an error.
- **Init order — handled.** The model is constructed first (so `_init_weights`
  runs), then `initialize_token_embedding_from_pretrained` copies the Qwen matrix
  over the freshly-initialized embedding. The copy asserts an exact shape match,
  so it only succeeds when **our `hidden_size` equals Qwen's hidden size** (with
  the default `num_attention_heads=8 × head_dim=128 = 1024`, this matches).
- **Freezing:** the original plan said "no need to freeze," but a
  `--freeze-token-embedding` flag now exists *and the current `train.sh` runs use
  it.* This is a live deviation from the plan — see the chat discussion.

### Bigger idea: distill from Qwen3.5-0.8B

Since we share the tokenizer, logit distillation is nearly free to wire up: run
the teacher (Qwen3.5-0.8B) in inference mode on the same batches and add a KL
term between teacher and student logits (typically `loss = α·CE + (1-α)·KL`,
α≈0.5, with temperature T≈2). At 0.6B scale, distillation reliably beats raw
cross-entropy for the same token count. Cost: the teacher forward pass roughly
halves throughput, but it's usually worth more than 2× the data.

---

## 2. Datasets & training plan

### Budget sanity check
A 5090 does very roughly 25–40K tokens/sec on a 0.6B model →
~2–3B tokens/day. Chinchilla-optimal for 0.6B is ~12B tokens; modern small
models (SmolLM2, Qwen3-0.6B) train *far* past that. Plan for "as many days as
tolerable," realistically 10–30B tokens.

### Stage 1 — pretraining (gives fluency)
- **FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`, subset `sample-10BT` or
  `sample-100BT`) — highest quality-per-token open web corpus, which matters
  most at this scale. Default choice.
- **SmolLM corpus** (`HuggingFaceTB/smollm-corpus`) — FineWeb-Edu-dedup +
  Cosmopedia-v2 synthetic textbooks, literally designed for sub-1B models. Good
  curated alternative.
- Since we only want chat (not world knowledge), 5–10B tokens of FineWeb-Edu
  already yields fluent English.

### Stage 2 — SFT for chat
- **SmolTalk** (`HuggingFaceTB/smoltalk`, ~1M conversations) — built to make
  small models chat well; used by SmolLM2-Instruct. Best fit.
- **UltraChat-200k** (`HuggingFaceH4/ultrachat_200k`) — lighter-weight
  alternative.
- Our `chat` mode already applies the Qwen chat template, so these drop in as
  JSONL after a small conversion script.

**Do not** SFT from scratch and skip Stage 1 — you'd get something that mimics
chat format but emits word salad. Fluency comes from the base model.

---

## 3. Training-loop fixes — all landed

Every item from the original priority list is now implemented. Kept here as a
record of what's in the loop and where.

1. **LR schedule — done.** `build_scheduler` does linear warmup
   (`--warmup-ratio`, default 0.01) → cosine decay to `--min-lr-ratio` (default
   0.1) of peak.

2. **fp32 master weights — done.** The model is no longer cast to bf16; it stays
   fp32 and `torch.autocast(bfloat16)` picks the compute dtype for matmuls.
   `move_optimizer_state_to_device` even forces optimizer moments to fp32.

3. **Effective batch size — done.** `--grad-accum-steps` exists; `--seq-len`
   defaults to 1024; `--sliding-window-size` default is now 512 (was 128).

4. **Param groups for weight decay — done.** `split_optimizer_parameters` decays
   only 2D non-embedding weights; RMSNorm weights (`ndim < 2`) and
   `token_embedding.weight` go to a `weight_decay=0` group.

5. **Pre-tokenize to a memmap — done.** `tokenization.py` writes a uint32
   `.tokens.bin` (+ `.spans.npy`, `.meta.json`, optional `.loss_mask.bin`);
   `load_tokenized_dataset` maps it with `np.memmap` and `get_batch` samples
   windows. uint32 as recommended.

6. **SFT loss masking — done.** Chat mode builds an assistant-only loss mask in
   `encode_chat_example` (monotonic-template check + `IGNORE_INDEX` on
   non-assistant tokens). `get_batch` resamples windows that contain no
   supervised tokens, and per-conversation `spans` keep windows from crossing
   conversation boundaries.

7. **Chunked loss at large vocab — done (and then some).** Replaced by a fused
   Triton linear+cross-entropy (`fused_linear_cross_entropy` /
   `_FusedLinearCrossEntropy`): the projection runs per chunk and the full
   `(N, V)` logits are never materialized — peak head memory is `O(chunk · V)`.
   `--loss-chunk-size` (default 256) tunes the chunk.

**Beyond the original plan (also landed):**
- **Muon optimizer**, now the *default* (`--optimizer muon`). 2D hidden weights
  go to Muon; embeddings and norms stay on AdamW via `CombinedOptimizer`.
  `--optimizer adamw` keeps the old behavior.
- **Residual-projection scaling** by `1/sqrt(2·num_layers)` on `o_proj` and
  `ffn.down_proj` at init (`_scale_residual_projections`).
- **RoPE applied in fp32**, and only on layer 0 + the sliding-window layers —
  the non-first full-attention layers run **without** positional encoding (NoPE).
- Resume, progress checkpoints, and old-checkpoint pruning.

**Left alone (good defaults at this scale), as planned:** tied embeddings, betas
(0.9, 0.95), grad clip 1.0, dropout 0.

---

## 4. Scaling sequence length toward 64K

Reaching 64K is a *separate phase* after a solid short-context base model. Do
**not** pretrain at 64K from the start — it's far more expensive per token and
most training docs are short, so the long positions barely get signal.

### 4a. The architecture already helps
The attention pattern is `[FULL] + [SWA, SWA, SWA, FULL] * N` (`model.py:84-94`),
so only ~1/4 of layers use full (global) attention; the rest are sliding-window.
At long context the SWA layers cost O(n · window) instead of O(n²), so the
quadratic blow-up only hits the minority of full-attention layers. This is
exactly the design that makes long context affordable — keep it.

- At 64K, bump `sliding_window_size` to something meaningful (Qwen/Gemma-style
  long-context configs use 1K–4K windows). Too small a window starves the SWA
  layers of local context.

### 4b. RoPE base — now a config field (partly done)
`RotaryEmbedding` takes `base` from `MykLLMConfig.rope_theta`, which **defaults
to `1e7`** (no longer the hardcoded `10000` this note originally described).
`rope_theta` is threaded through to every `RotaryEmbedding`. Two gaps remain:
it is **not** exposed as a CLI arg and `build_config` does not pass it, so every
freshly-built model uses `1e7` regardless of context length (resumed checkpoints
keep whatever was saved). Options, in order of preference:

- **Train long context with a high RoPE base ("ABF" — adjusted base
  frequency).** Models that natively reach long context use a much larger base:
  e.g. Qwen long-context variants use base ≈ 1,000,000. The config field already
  exists; the remaining work is just to **expose `rope_theta` on the CLI / in
  `build_config`** so short and long phases can differ by config. (Note the
  current `1e7` default is already high — see the chat discussion on whether
  that's the right base for the short-context base run.)
- **YaRN / NTK scaling at inference + light fine-tune.** YaRN lets you extend a
  short-context model's effective length with a modest fine-tune (often
  <1B tokens) rather than full retraining. This is the cheap path and what most
  open long-context models actually use. Worth implementing if renting GPUs for
  a full 64K pretrain is too costly.

  Practical rule of thumb: don't expect more than ~2–4× length extension from
  scaling tricks *without* any long fine-tuning; for a clean 32×
  (2K → 64K) jump, budget a real long-context training phase.

### 4c. Staged length curriculum (recommended recipe)
1. **Base pretrain** at seq_len 2K (base 10000), most of the token budget.
2. **Mid extension** to 8K–16K, base raised (e.g. 1e6), a few B tokens of
   long-ish documents.
3. **Final extension** to 64K, a smaller slice (often <5B tokens) of genuinely
   long documents.

Each stage reuses the previous checkpoint. Length extension is cheap relative to
base pretraining precisely because it needs few tokens — but those tokens must
actually be long.

### 4d. Data for long context
Web text is mostly short; you can't extend on FineWeb-Edu alone. Sources of long
sequences:
- Books (e.g. PG-19 / books corpora), long-form articles.
- Code repositories concatenated per-repo (naturally long, good structure).
- **Document packing**: concatenate many short docs to fill the window. Cheap and
  standard, but teaches "many short things in a row," not true long-range
  dependency. Mix some genuinely long docs so the model learns real long-range
  retrieval, not just packing.
- A small amount of synthetic long-context tasks (retrieval / "needle"-style)
  in the final stage sharpens long-range behavior.

### 4e. Memory & throughput at 64K (the part that forces GPU rental)
- Flash-attn keeps attention **memory** at O(n), so you won't OOM purely from the
  score matrix — but full-attention **compute** is still O(n²). At 64K the
  full-attention layers dominate runtime.
- The RoPE cos/sin cache is precomputed to `max_seq_len` (`model.py:137-141`):
  64K × 128 fp32 ≈ 32 MB — negligible.
- The real pressure is per-token activation memory × 64K sequence. Expect to:
  - drop `batch_size` to 1 per device and lean on gradient accumulation,
  - rely on `gradient_checkpointing` (already supported, `model.py:366-367`),
  - and chunk the LM-head/CE (see §3.7) — at 64K the logits tensor alone is
    enormous.
- This is the phase where renting a larger GPU (or multi-GPU) pays off. The 5090
  is fine for the base + mid stages; the 64K final stage likely wants more VRAM
  or model/sequence parallelism. The good news: it's a short phase token-wise, so
  rental time is bounded.

### 4f. Code changes needed for long context (summary)
- ~~Make RoPE `base` a `MykLLMConfig` field~~ — **done** (`rope_theta`); still
  needs a CLI / `build_config` hook to vary it per stage (see §4b).
- (Optional) implement YaRN/NTK scaling in `RotaryEmbedding` for cheap extension
  — **not done.**
- Make `sliding_window_size` scale with the target length per stage — it's a
  config field / `--sliding-window-size` arg; just set it per stage.
- ~~Chunked CE / fused loss becomes mandatory at 64K~~ — **done** (fused Triton
  linear+CE, §3.7).
- Verify `generate()`'s constraint `prompt + max_new_tokens <= max_seq_len`
  (in `MykLLM.generate`) is acceptable, and that the SWA KV-cache trimming
  (`FlashSelfAttention._trim_cache`) behaves at large windows.

---

## 5. Multi-token prediction (MTP) — considered and deferred

**Decision: do NOT add MTP to the core run. Revisit only as a late, optional
phase.**

### Why deferred
MTP (predicting tokens t+2, t+3… via extra heads/modules alongside the normal
t+1 head) has two distinct payoffs that are often conflated:
1. **Denser training signal** → better sample efficiency / representations.
2. **Inference speedup** → the extra heads act as a built-in draft model for
   self-speculative decoding (~1.5–2×).

Neither justifies it at 0.6B:
- The training-quality benefit is **scale-dependent**. The original Meta paper
  (Gloeckle et al., 2024) found MTP helps *larger* models and is
  **neutral-to-slightly-harmful for small models**; gains appeared around the
  multi-billion-param range, especially on code/reasoning. At 0.6B we're below
  that regime.
- The speedup is moot: a 0.6B forward pass is already cheap, so
  self-speculative decoding saves little.
- It adds real complexity (extra modules, loss weighting, separate eval) for a
  benefit the literature says we won't see at this size.

### Reuse note (differs from the embedding case)
Qwen's **embedding** transfers because it's a per-token lookup over a **shared
vocab** — token ID *k* means the same thing in both models. Qwen's **MTP module
does not transfer the same way**: it's a small transformer block conditioned on
*the main model's internal hidden states*, and our hidden states are entirely
different representations from Qwen's. Only the module's shared embedding /
output-head parts would line up; the MTP transformer block itself would have to
be trained from scratch. So "reuse Qwen's MTP" is **not** a free transplant the
way the embedding is.

### How to add it later (if a non-MTP baseline ever motivates it)
Treat this as a bolt-on phase that starts from a finished base/SFT checkpoint —
not a from-scratch architecture change.

1. **Pick the simple variant.** Use **Meta-style parallel heads**: n independent
   prediction heads sitting on the shared trunk, each predicting one of
   t+1 … t+n in parallel. This is simpler than DeepSeek-V3's *sequential* MTP
   modules (each a full transformer block in a causal chain). DeepSeek's
   sequential design mainly improves *speculative-decoding acceptance*, which we
   don't need — skip it.

2. **Depth 1 is enough.** Predict just t+2 (so two heads total: the existing
   `lm_head` for t+1, plus one extra head for t+2). More depth = more cost for
   diminishing return at this scale.

3. **Architecture sketch** (on top of current `model.py`):
   - The trunk runs as today, producing `final_norm(x)` → hidden states `h`
     of shape `[B, T, hidden]`.
   - Main head: existing tied `lm_head(h)` predicts t+1 (unchanged).
   - MTP head: a small module — at minimum one `RMSNorm` + one
     `TransformerBlock` (or just an extra norm + linear) consuming `h`, then a
     projection to vocab. **Tie its output projection to `token_embedding`**
     too, so the extra head adds few params.
   - Keep the MTP module in its own `nn.Module` so it can be attached/detached
     and excluded from inference cleanly.

4. **Loss.** `loss = CE_main(logits_t+1, target_t+1)
   + λ · CE_mtp(logits_t+2, target_t+2)`, with **λ ≈ 0.2–0.3**.
   - Targets: t+1 uses the usual shift-by-1; the MTP head uses **shift-by-2**
     (`targets[:, 2:]` aligned to positions `[:-2]`). Mind the boundary — the
     last valid MTP position is `T-2`.
   - Apply the same large-vocab chunked/fused CE as §3.7; now there are two
     logit tensors, so memory pressure is higher — chunk both.

5. **When to run it.** Attach the MTP head for a **late slice** of training
   (e.g. during SFT or a short dedicated phase), starting from the existing
   checkpoint. Don't train it from scratch alongside the base model.

6. **Inference.** For normal generation, **discard the MTP head** and run exactly
   as today (`generate()` unchanged) — the MTP head only ever existed to densify
   the training signal. Only keep it loaded if you specifically wire up
   self-speculative decoding later (probably not worth it for 0.6B).

7. **Gate on eval.** Keep a non-MTP baseline checkpoint. Only adopt MTP if it
   beats that baseline on the chat eval — do not assume it helps.

---

## Suggested next action
The §3 fixes and §1 embedding reuse are done, and a short-context training loop
runs end-to-end (see `train.sh` / `tokenize.sh`). The open decisions are now
about *data and goal* (§2): the current runs train on a code/Rust corpus, not the
chat corpora this note assumes. Decide whether the goal is still "conversational"
(then wire up SmolTalk-style SFT data) or has shifted to code, and bring §2's
dataset section in line with that before the 64K curriculum (§4).
