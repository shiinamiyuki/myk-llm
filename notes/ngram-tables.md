# Large n-gram tables: optimizer and host offload

Status (2026-11-16): lazy (sparse) Adam with catch-up and host offload implemented
(`src/myk_llm/train/ngram_tables.py`, `optimizer.ngram`); optimizer sweep at 1M rows and
tables up to 2 x 32M rows (8.6B parameters, 8x what dense AdamW fits on the GPU) done.

Question: how to train hashed n-gram tables (`model.ngram`) that are larger than GPU
memory. Two parts: (1) which optimizer (and weight decay (WD)) the table needs, and (2)
where its master weights and optimizer state live when the GPU cannot hold them.

## What the literature says

- Qwen3.8-Next architecture report (arXiv 2608.30320, §2.3, §3.1): one n-gram embedding
  layer at block 2, 51B table parameters "held off the accelerator"; "the n-gram
  embedding table runs on Adam with weight decay disabled" (only the n-gram key/value
  projection uses Muon). Host offload is possible because "deterministic addressing
  enables host-memory offloading and asynchronous prefetching"; block 2 lets the
  prefetch overlap block 1. No learning rate, precision, sparse-update or training-side
  offload details.
- Engram (DeepSeek, arXiv 2601.07372, §4.1, Table 5): the table uses Adam with a 5x
  learning rate and no WD, with Muon for the rest of the model. During *training* the tables are
  sharded across GPUs, with an all-to-all exchange of the active rows and their gradients
  (§2.5). Host offload (100B table, <3% overhead) is measured for *inference* only.
- Recommender-system practice (FBGEMM / TorchRec table-batched embeddings, HugeCTR):
  sparse optimizers fused into the backward pass, rowwise second moments (rowwise Adagrad,
  "partial rowwise Adam"), and unified virtual memory (UVM) with a GPU cache for the hot
  rows.

Neither paper says whether the table update is dense or sparse. That choice decides
whether host offload is feasible at all, so it comes first below.

## Summary

- **Recipe for a large table:** Adam with no WD and a 5-15x learning rate. Use lazy
  updates (only the rows a step looks up) with `catch_up`, a rowwise second moment, and
  `placement: host` with prefetch. In this setup that gives the same bpb as dense AdamW,
  runs faster than dense, uses ~12.5 GB of GPU memory whatever the table size, and needs
  8 bytes per table parameter of host RAM.
- **Plain lazy Adam is not dense Adam.** At the base learning rate it loses 0.0085 bpb.
  Dense Adam keeps moving a row along its momentum after the row's n-gram stops
  appearing; `catch_up` applies those skipped updates in closed form when the row comes
  back. A high table learning rate also hides the gap.
- **WD on the table hurts once lr x wd x steps is large**: +0.0053 bpb at ~8. It is
  harmless at this run's 0.16, but a production run is in the hundreds.
- **The table keeps scaling:** 2 x 16M rows on host is 0.0096 bpb better than 2 x 4M,
  the largest that dense AdamW fits on the 32 GB GPU.

## Dense vs lazy updates

The repo's default (`optimizer.ngram.update: dense`) treats the tables as ordinary AdamW
parameters: every step builds a full-size gradient (zeros except the looked-up rows)
and updates every row. That costs 16 bytes per table parameter on the GPU (fp32 weight,
gradient, two moments) and an update proportional to the table, not the batch. Host
offload with dense updates would have to stream the whole table through the GPU every
step, so it cannot scale.

A step only touches a small part of the table. Measured on the 32K-vocabulary data with
131K tokens per step and 2-/3-grams:

| rows per order | distinct rows per step | share of the table | overlap with the previous step |
|---|---|---|---|
| 1M | 194K (0.74 of 262K lookups) | 9.3% | 16% |
| 16M | 204K | 0.6% | 8% |

`update: lazy` gathers the step's distinct rows, runs forward/backward against that
compact buffer, applies Adam to it and writes it back. Rows not in the batch are left
alone, moments included (LazyAdam, as `torch.optim.SparseAdam`; bias correction by the
global step).

Lazy and dense Adam differ on rows that skip steps. Under dense Adam a row touched once
keeps moving for many steps: its momentum decays as beta1^k while the square root of its
second moment decays as beta2^(k/2), so each later step moves it
(0.9 / sqrt(0.95))^k = 0.923^k times the first step's update. Summed over all later
steps that is about 12x the first update, plus the first update itself. Lazy Adam moves a
rare row only when it is looked up, so per lookup it learns at a lower effective learning
rate. Frequent rows behave the same under both. Two fixes, both measured below: a higher
table learning rate, or `catch_up`. With `catch_up`, a row last updated at step s and
looked up again at step t first gets the t - 1 - s zero-gradient updates dense AdamW
would have made. Their closed form is
`w <- d^k (w - lr m / sqrt(v) sum_{j=1..k} (r/d)^j)` with `d = 1 - lr wd`; the moments
decay to `beta1^k m` and `beta2^k v`. That needs one int32 per row for the step it was
last updated. It is exact up to eps, bias correction and a learning-rate change inside
the gap (`tests/test_ngram_tables.py` checks it against dense AdamW). Evaluation and
export read caught-up rows.

Decoupled WD differs in the same way. Dense AdamW shrinks every row every step,
`w *= 1 - lr * wd`, including the many rows that get a gradient only rarely, so a rare
n-gram's row decays between lookups. With the lazy update, WD only acts when a row is
looked up (`optimizer.ngram.weight_decay`, default 0).

## Host offload design

`placement: host` keeps the stacked tables (orders x buckets rows) and both Adam moments
in host memory. That is 12 bytes per parameter in fp32, or 8 with
`second_moment: rowwise`. The tables are ordinary pageable memory; only the staging
buffers are pinned. The GPU holds two staging slots, each as large as one step's rows
(weights, moments, ids), plus the compact gradient. Per optimizer step t (slot t % 2):

1. Main thread, `begin_step`: wait for the prefetched rows of step t. Patch the rows
   that step t-1 also touched from step t-1's updated GPU slot, since the prefetch read
   them before step t-1's write-back. Then queue the gather of step t+1. The trainer
   draws data one step ahead for this, and the row ids are a pure function of the token
   ids.
2. Worker thread, gather(t+1): hash and dedupe the ids on the CPU, `index_select` the
   rows, moments and ids into the pinned slot, and copy them to the GPU on a separate
   stream after the patch event of step t.
3. Main thread: forward/backward against the compact rows (a leaf tensor). Clip the
   global gradient norm with the compact gradient included, which gives the same norm
   as the dense gradient.
4. Main thread, `step`: Adam on the GPU slot, then copy the result to the pinned slot on a
   second stream. Worker thread, scatter(t): `index_copy_` back into the host tables.

The worker's queue is first-in-first-out (FIFO):
`gather(t+1) | scatter(t) | gather(t+2) | ...`. A row that step t+1 shares with step
t-1 but not with step t was therefore written back before it is gathered again. A row
it shares with step t is patched. The result matches `placement: gpu` bit for bit
(`tests/test_ngram_tables.py`, which also checks that the test fails without the
patch). Checkpoints save the tables as `ngram_tables.pt` next to the DCP checkpoint
(`checkpoint.final: false` skips the final save for sweeps). `export_weights` writes
them under the model's own parameter names.

Bandwidth on this machine (RTX 5090, i9-13900KF, DDR5, PCIe 5.0 x16 measured at
23 GB/s each way): a CPU row gather or scatter runs at 15-20 GB/s. One step's 200K rows x 128 x
(weight + 2 moments) is about 300 MB each way: about 23 ms of gather and 19 ms of scatter
on the worker, and about 13 ms of copying each way. The step itself takes about 300 ms.

Throughput (30M-body model, 2 x 1M rows x 128, 80-step runs):

| mode | tok/s | peak GPU memory | main-thread wait for rows |
|---|---|---|---|
| dense AdamW | 437K | 15.7 GB | - |
| lazy, GPU | 445K | 15.8 GB | - |
| lazy, host, prefetch | 441K | 12.8 GB | 0.1 ms |
| lazy, host, no prefetch | 361K | 12.8 GB | 61 ms |

- **Prefetch hides the offload completely.** Without it the gather and copy sit on the
  critical path and cost 19%.
- **Pitfall: one GPU->CPU sync per step cost 4%.** The first version of the patch
  used `hit.nonzero()`, which has a data-dependent output size and so forces a sync. That
  stopped the CPU from queueing kernels ahead of the GPU: 427K tok/s even with the copy fully
  hidden. A `torch.where` over all rows fixed it.
- **Pitfall: the copy streams must wait for the allocating stream.** The GPU staging
  slots come from PyTorch's caching allocator on the compute stream. A recycled block
  can still be in use by kernels queued there, which is safe only for work on the same
  stream. The first host-to-device copy into a slot ran on a side stream and could
  overwrite memory those kernels were still reading. That corrupted *unrelated* GPU work
  in ~60% of trials of a unit test, and turning off the caching allocator or
  `CUDA_LAUNCH_BLOCKING=1` hid it. Fix: `side.wait_stream(compute)` once after
  allocation, plus `record_stream` on the slots for freeing.
- The worker needs about 0.3 us per token at dim 128 with 2 orders, a ceiling of about
  3M tok/s. Compute per token grows with the model while the offload work per token does
  not, so offload gets cheaper relative to compute at larger model sizes.

## Hot rows

Over 400 steps at 16M rows per order, 62% of the table is touched, the median touched row
is looked up in 2 steps, and the hottest 1% / 5% / 10% of rows cover 25% / 45% / 57% of
all row lookups. At 1M rows per order, hash collisions flatten this to 7% / 19% / 28%.
A GPU cache of the hottest ~10% of a large table would roughly halve the PCIe and CPU
traffic, the same idea as FBGEMM's UVM caching and Engram's proposed multi-level cache.
Not needed at this model size, where the traffic is already hidden.

## Results: optimizer of the table (1M rows per order)

`configs/ngram-small/sweeps/optim.yaml`: the ngram-small 32K setup (30M-body model, 3210
steps x 131K tokens), 2 x 1M rows x 128 at block 1, mean en / zh validation bpb at the
final step, one seed (seed spread in this setup: 0.0007-0.0009). Table lr =
`optimizer.lr` (5e-4) x scale; everything else is unchanged (Muon 5e-4 for the body).

| table optimizer | lr 1x | lr 5x | lr 15x |
|---|---|---|---|
| dense AdamW, wd 0 | 1.1962 | 1.1894 | **1.1886** |
| dense AdamW, wd 0.1 | 1.1955 | | |
| lazy Adam | 1.2047 | 1.1910 | 1.1888 |
| lazy Adam + catch_up | 1.1964 | 1.1899 | |

- **The table learning rate matters most.** 5x gives -0.007 bpb, about as much as 4x
  more rows did in the table-size scan; 15x gives another 0.0008, which is near noise.
  That is consistent with Engram's 5x, and the curve is flat between 5x and 15x.
- **Plain lazy Adam loses to dense at a low learning rate**: +0.0085 at 1x, as much as
  the 1x to 5x gain. The gap shrinks as the learning rate rises (+0.0016 at 5x, +0.0002
  at 15x). Dense Adam's drift on skipped rows acts as an extra learning rate for rare
  rows, and a higher learning rate supplies the same.
- **`catch_up` makes lazy match dense** at every learning rate tested: +0.0002 at 1x,
  +0.0005 at 5x. It costs 4 bytes per row, which records the step each row was last
  updated, and some elementwise work on the staged rows (437K tok/s vs 438K for plain lazy).
- **WD 0.1 does not hurt at this horizon** (-0.0007, noise). Over the whole run it
  shrinks a never-looked-up row by exp(-lr x wd x steps) = exp(-0.16). A production run
  (Qwen3.8: lr 1.76e-3, about 1.7M steps) gives lr x wd x steps in the hundreds, enough
  to erase every row between lookups.
- **Strong WD does hurt.** Dense AdamW at 5x with wd 1.0 (lr x wd x steps ~ 8, closer
  to a long run) ends at 1.1947 vs 1.1894 without WD: +0.0053, most of the 5x gain gone.
  The gap grows during training (+0.003 / +0.006 / +0.009 at steps 750 / 1250 / 1750)
  before the learning-rate decay. This supports the Qwen3.8 / Engram "no WD on the
  table", and the reason is the horizon: WD acts every step on every row, while a row's
  gradient arrives only when its n-gram occurs.
- **The rowwise second moment is free**: lazy + catch_up at 5x, 1.1890 rowwise vs 1.1899
  elementwise. One v per row instead of one per element halves the optimizer state
  (8 instead of 12 bytes per parameter with the fp32 weight).
- Lazy runs are 2% faster than dense here (438K vs 429K tok/s): no full-size gradient,
  and no dense update of 2M rows.

| more runs at 1M rows | bpb |
|---|---|
| lazy + catch_up, 5x, rowwise second moment | 1.1890 |
| dense AdamW, 5x, wd 1.0 | 1.1947 |

## Results: a table larger than GPU memory

`configs/ngram-small/sweeps/host.yaml`: same setup, table lr 15x, lazy runs with
`catch_up`. Dense AdamW at 2 x 4M rows already needs 27.7 GB on the 32 GB card. The
next size up, 2 x 8M rows, would need ~40 GB dense and ~38 GB with lazy updates on the
GPU.

| rows per order | table params | optimizer | where | bpb | tok/s | GPU peak | host state |
|---|---|---|---|---|---|---|---|
| 1M | 268M | dense | GPU | 1.1886 | 429K | 15.7 GB | - |
| 1M | 268M | lazy + catch_up | GPU | 1.1885 | 438K | 15.8 GB | - |
| 4M | 1.07B | dense | GPU | 1.1757 | 392K | 27.7 GB | - |
| 4M | 1.07B | lazy + catch_up | host | 1.1759 | 433K | 12.8 GB | 12.0 GiB |
| 16M | 4.3B | lazy + catch_up | host | **1.1661** | 433K | 12.8 GB | 48.1 GiB |
| 32M | 8.6B | lazy + catch_up, rowwise v | host | **1.1612** | 436K | 12.5 GB | 64.5 GiB |

(No n-gram tables: 1.2215. The 1M dense row is from optim.yaml.)

- **Host offload matches dense quality and is faster.** At 4M rows: 1.1759 vs 1.1757,
  433K vs 392K tok/s, 12.8 vs 27.7 GB of GPU memory. Dense AdamW pays for a full-size
  gradient and an update of all 8M rows every step.
- **Throughput does not depend on the table size**: 433-440K tok/s from 4M to 32M rows.
  The main thread waits 0.04 ms per step for the rows. The worker's gather (27 ms) and
  scatter (22 ms) per step run in the background of a ~300 ms step.
- **Bigger tables keep paying off.** From 1M to 4M rows bpb drops by 0.0128, and from
  4M to 16M rows by another 0.0096. 16M rows is 0.0096 better than the largest table
  that dense AdamW fits on the GPU. 32M rows (rowwise v,
  which was within noise of elementwise at 1M) adds -0.0049 more, almost all on zh
  (1.1637 -> 1.1544; en 1.1685 -> 1.1679). Chinese has many more distinct 2-/3-grams
  for the hash to separate. On 400M training tokens, 2 x 32M rows is ~6 tokens per row.
- The table learning rate matters more for larger tables. At 4M rows 15x gives 1.1757
  against 1.1881 at 1x (tables.yaml): -0.0124, compared with -0.0076 at 1M rows. Larger
  tables have rarer rows, and rare rows need the larger step.

## Multi-GPU (not implemented)

`update: lazy` raises under multiple GPUs. Options:
- Every rank gathers its own rows from one shared host table per node. The step's
  gradients then have to be summed across ranks for the union of their rows: an
  all-gather of ids and gradients, or a reduce-scatter by row owner.
- Shard the rows by id across ranks, with an all-to-all for rows and gradients as in
  Engram's training setup; each shard can still live in its rank's host memory.
