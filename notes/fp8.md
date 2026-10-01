# FP8 training

`parallel.float8: tensorwise` runs the block linears' matmuls in FP8 (8-bit floating
point, e4m3) through torchao's `Float8Linear`, with dynamic per-tensor scaling. Master
weights (fp32), optimizer (Muon / AdamW), attention, norms and the output head / loss
are unchanged. Investigated 2026-10-01 on one RTX 5090, torch 2.13 + CUDA 13.0,
torchao 0.9.0 (installed) and 0.18.0 (tried from a scratch directory).

```bash
pip install -e '.[fp8]'                     # torchao
python scripts/benchmark_throughput.py configs/experiments/ablation.yaml parallel.float8=tensorwise
scripts/launch.sh configs/experiments/fp8.yaml
```

## Why it pays on this GPU

GeForce Blackwell runs bf16 matmuls with fp32 accumulation at half rate (~210 TFLOPS),
while FP8 runs at full rate. So the FP8 GEMM (general matrix multiply) speedup is larger
than the ~1.5x usually quoted for H100. `torch._scaled_mm`, this model's shapes
(M = 16 x 2048 tokens), TFLOPS:

| GEMM (M x K x N) | bf16 | FP8 tensorwise | FP8 rowwise | MXFP8 (32-block) |
|---|---|---|---|---|
| qkv 32768x1024x1536 | 199 | 524 (2.6x) | 416 (2.1x) | 541 (2.7x) |
| o_proj 32768x1024x1024 | 211 | 496 (2.4x) | 400 (1.9x) | 525 (2.5x) |
| up_gate 32768x1024x8192 | 236 | 628 (2.7x) | 433 (1.8x) | 529 (2.2x) |
| down 32768x4096x1024 | 224 | 542 (2.4x) | 437 (2.0x) | 599 (2.7x) |
| weight grad 8192x32768x1024 | 189 | 651 (3.4x) | 373 (2.0x) | 541 (2.9x) |

All three scaling granularities have working kernels on sm_120 (compute capability 12.0).

## End-to-end throughput

`scripts/benchmark_throughput.py`, fp32 master weights + bf16 autocast (the project default):

| model | batch | bf16 | FP8 tensorwise | FP8 rowwise\* | rowwise_with_gw_hp\* |
|---|---|---|---|---|---|
| 200M `ablation` (GLGLGLGLG) | 16x2048 | 121.7K tok/s, 14.8 GB | **154.7K (+27%), 11.7 GB** | 124.3K (+2%), 12.1 GB | 129.3K (+6%), 14.6 GB |
| 600M (33 layers, GLLL) | 8x2048 | 39.5K, 27.9 GB | **51.7K (+31%), 21.8 GB** | — | 41.3K (+4%), 27.0 GB |
| 200M `hybrid-deltanet` (GD) | 16x2048 | 108.1K, 17.6 GB | **129.7K (+20%), 14.5 GB** | | |

\* rowwise needs torchao 0.18 (see below).

- **Tensorwise is the recipe to use.** Rowwise scaling's extra passes (per-row amax, its
  slower kernel) eat almost the whole GEMM gain at hidden size 1024.
- **Memory drops too** (−3 GB on 200M, −6 GB on 600M): the activations saved for
  backward are stored in FP8.
- The gain is below the GEMM speedup because attention, norms, Muon (~19% of a step,
  see throughput.md) and the output head + loss (~1/3 of the 200M model's matmul FLOPs,
  64K vocab) stay in bf16/fp32.

## Integration (`train/distributed.py: convert_float8`)

Conversion runs first in `parallelize()`, before activation checkpointing, compile and
FSDP2 (fully sharded data parallel), so both the trainer and the benchmark script use it.
Converted: every `nn.Linear` except

- **the output head**: it is tied to the embedding, and the fused cross-entropy reads
  `output_weight` directly (never calls `lm_head`), so converting it would do nothing;
- **linears inside Gated DeltaNet layers**: fla's layer runs under
  `torch.compiler.disable`, where torchao's amax/scale/cast ops are not fused. With
  them converted the hybrid model gained nothing (108.1K → 107.0K tok/s); skipped, +20%.
- linears with a dimension not divisible by 16 (FP8 GEMM requirement).

Parameter names and dtypes are unchanged, so checkpoints, `model.safetensors` exports
and warm starts are interchangeable between FP8 and bf16 runs. Checked end to end with
`configs/experiments/smoke.yaml`: training, validation, greedy sampling, DCP checkpoint
save + resume, single GPU and FSDP (`torchrun --nproc_per_node 1`, `parallel.strategy=fsdp`).

## Loss parity

200M `ablation` model on the smoke shards: 360 steps x 64 x 2048 = 47M tokens, WSD
(warmup-stable-decay) schedule, real trainer. Validation bits per byte (BPB) at the end:

| run | val/en/bpb | val/zh/bpb | train tok/s (median) | peak mem |
|---|---|---|---|---|
| bf16, seed 0 | 1.4558 | 1.5049 | 126.6K | 15.6 GB |
| bf16, seed 1 | 1.4529 | 1.4963 | 126.5K | 15.6 GB |
| FP8 tensorwise, seed 0 | 1.4593 (+0.0035) | 1.5088 (+0.0039) | **160.8K (+27%)** | 12.5 GB |
| FP8 tensorwise, seed 1 | 1.4554 (+0.0025) | 1.4994 (+0.0031) | 160.7K | 12.5 GB |
| FP8 rowwise (torchao 0.18), seed 0 | 1.4654 (+0.0096) | 1.5228 (+0.0179) | 128.7K | 12.8 GB |

- **Tensorwise costs ~0.003 BPB (~0.2%) at matched seed.** The gap has the same sign
  on both seeds and at every evaluation (steps 60–360), so it is probably real, but it is
  smaller than the seed-to-seed spread of bf16 itself (0.003 en, 0.009 zh). No loss
  spikes; gradient norms match. Whether it grows or vanishes over a long run is open:
  `configs/experiments/fp8.yaml` (2.1B tokens) vs `ablation-baseline` answers that.
- At equal wall-clock, +27% tokens more than pays for it: late in this run BPB falls
  ~0.045–0.057 per 60 steps, so 0.003 BPB is ~4 steps (~1% of the run).
- Rowwise was *worse* (one seed, torchao 0.18) despite its finer scaling. Unexplained;
  not investigated further since it is also barely faster than bf16 here.

## Library caveats

- **torchao 0.9 rowwise + autocast fails** (`aten._to_copy.default with axiswise scaling
  is not supported yet`) when parameters are fp32, i.e. the project default. Fixed in
  torchao 0.18 (`uv pip install` resolves 0.18.0 for torch 2.13).
- **Tensor-subclass linears bypass autocast's input cast.** Under autocast, `RMSNorm`'s
  `pow`/`rsqrt` run in fp32, so block inputs arrive as fp32; a plain `nn.Linear` gets
  them cast to bf16 by autocast, but torchao's MXFP8 weight wrapper intercepts the call
  first and asserts bf16 inputs. `Float8Linear` (tensorwise) handles it itself.
- **MXFP8** (torchao 0.18, `prototype.moe_training.MXFP8TrainingOpConfig`): works on
  sm_120 with bf16 parameters and no autocast; 200M: 130.3K (pure bf16) → 161.4K tok/s
  (+24%) vs tensorwise 168.3K (+29%) in the same setting. Finer scaling (one scale per
  32 elements) than tensorwise for slightly less speed. Would need a bf16-parameter
  single-GPU path (today `param_dtype: bf16` only takes effect under FSDP).

## Next steps

- **FP8 all-gather under FSDP** (`Float8LinearConfig(enable_fsdp_float8_all_gather=True)`
  + `precompute_float8_dynamic_scale_for_fsdp(model)` after each optimizer step): weights
  are all-gathered in FP8, 4x less traffic than fp32 parameters, which matters on
  PCIe-only multi-GPU nodes. Tried with 1 process: training, Muon and DCP checkpoints
  work; the safetensors export fails because the gathered weights are still torchao
  subclasses (needs unwrapping in `export_model`). Benefit only measurable on >1 GPU.
- Rowwise / MXFP8 once torchao is upgraded, if tensorwise ever shows a loss gap at
  longer runs (more outlier-robust scaling).
- Output head in FP8: largest remaining bf16 GEMM on small models, but it feeds the
  loss directly; usually kept in high precision.
