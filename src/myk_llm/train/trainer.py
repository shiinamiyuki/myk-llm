"""Training entry point (pretraining and SFT share this loop).

    python -m myk_llm.train configs/pretrain/200m.yaml [key.path=value ...]
    torchrun --nproc_per_node 8 -m myk_llm.train configs/pretrain/200m.yaml

A run lives in <output_dir>/<run_name>/ (config.yaml, metrics.jsonl, tb/, checkpoints/).
Re-running the same command resumes from the newest complete checkpoint.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from ..configuration import dump_yaml, load_config
from ..model import MykLLM, generate, load_model
from .checkpoint import latest_checkpoint, load_checkpoint, prune_checkpoints, save_checkpoint, step_dir
from .config import InitConfig, TrainConfig
from .distributed import DistInfo, all_reduce_sum, init_distributed, parallelize, shutdown
from .optim import build_optimizers, lr_multiplier, set_lr


def log0(info: DistInfo, *args: Any) -> None:
    if info.is_main:
        print(*args, flush=True)


# -- initialization --------------------------------------------------------------------


def load_pretrained_embedding(source: str, row_map_path: str) -> torch.Tensor:
    """Token-embedding matrix of a HuggingFace safetensors checkpoint, optionally
    gathered down to a pruned vocab (row_map[new_id] = old_id)."""
    import json

    import numpy as np
    from safetensors import safe_open

    keys = ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight")
    root = Path(source)
    files = sorted(root.glob("*.safetensors")) if root.is_dir() else [root]
    index = root / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        files = [root / weight_map[k] for k in keys if k in weight_map][:1]
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            for key in keys:
                if key in handle.keys():
                    weight = handle.get_tensor(key)
                    if row_map_path:
                        weight = weight.index_select(0, torch.from_numpy(np.load(row_map_path).astype(np.int64)))
                    return weight
    raise KeyError(f"no embedding tensor ({keys}) found under {source}")


def initialize_weights(model: MykLLM, config: InitConfig) -> None:
    """Warm starts applied to the full (unsharded) model before parallelization."""
    if config.weights_from:
        source = load_model(config.weights_from, device="cpu")
        missing, unexpected = model.load_state_dict(source.state_dict(), strict=False)
        if unexpected:
            raise KeyError(f"unexpected weights in {config.weights_from}: {unexpected}")
        if missing:
            print(f"warm start: {len(missing)} tensors keep their fresh init (new modules): {missing[:8]}...")
    if config.embedding_from:
        weight = load_pretrained_embedding(config.embedding_from, config.embedding_row_map)
        if weight.shape != model.embed.token.weight.shape:
            raise ValueError(f"pretrained embedding {tuple(weight.shape)} != model {tuple(model.embed.token.weight.shape)}")
        with torch.no_grad():
            model.embed.token.weight.copy_(weight)
    if config.freeze_embedding:
        model.embed.token.weight.requires_grad_(False)


# -- evaluation ------------------------------------------------------------------------


def token_byte_lengths(tokenizer) -> torch.Tensor:
    """UTF-8 byte length of every token id (0 for special tokens), for bits-per-byte.
    Assumes a byte-level BPE vocabulary, where each character of a token's string form
    stands for exactly one byte."""
    special = set(tokenizer.added_tokens_decoder)
    tokens = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    return torch.tensor([0 if i in special else len(t) for i, t in enumerate(tokens)], dtype=torch.float64)


@torch.no_grad()
def validate(model, objective, val_sets: dict[str, list], token_bytes: torch.Tensor, info: DistInfo) -> dict[str, float]:
    """Per-token loss and bits per byte (BPB, comparable across tokenizers) of each fixed
    validation set. Batches are split across ranks; every rank runs the same number of
    forward passes (FSDP needs them in lockstep)."""
    model.eval()
    results = {}
    for name, batches in val_sets.items():
        loss_sum = token_count = byte_count = 0.0
        per_rank = math.ceil(len(batches) / info.world_size)
        for i in range(per_rank):
            index = info.rank + i * info.world_size
            batch = batches[index % len(batches)].to(info.device)
            weight = 1.0 if index < len(batches) else 0.0  # padding pass on uneven splits
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch.input_ids, seqlens=batch.seqlens, head=objective.head(batch, 1.0))
            loss_sum += weight * float(out.sums["ce"])
            token_count += weight * batch.num_targets
            targets = batch.targets[batch.targets >= 0].cpu()
            byte_count += weight * float(token_bytes[targets].sum())
        loss_sum = all_reduce_sum(loss_sum, info.device)
        loss = loss_sum / max(1.0, all_reduce_sum(token_count, info.device))
        results[f"val/{name}/loss"] = loss
        results[f"val/{name}/bpb"] = loss_sum / math.log(2) / max(1.0, all_reduce_sum(byte_count, info.device))
    model.train()
    return results


def sample_text(model, tokenizer, config: TrainConfig, info: DistInfo) -> list[str]:
    model.eval()
    texts = []
    for prompt in config.eval.sample_prompts:
        if config.eval.sample_chat:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True
            )["input_ids"]
        else:
            ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Greedy: identical on every rank, so FSDP ranks stay in lockstep.
            out = generate(model, ids.to(info.device), config.eval.sample_tokens, temperature=0.0,
                           eos_token_id=tokenizer.eos_token_id)
        texts.append(tokenizer.decode(out[0].tolist()))
    model.train()
    return texts


# -- main loop -------------------------------------------------------------------------


def train(config: TrainConfig, resume: str = "auto") -> None:
    info = init_distributed()
    run_dir = Path(config.output_dir) / config.run_name
    torch.manual_seed(config.seed)  # identical init on every rank before sharding
    torch.set_float32_matmul_precision("high")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer)
    if len(tokenizer) > config.model.vocab_size:
        raise ValueError(f"tokenizer has {len(tokenizer)} tokens > model.vocab_size {config.model.vocab_size}")
    per_step = config.data.micro_batch_size * info.world_size
    if config.global_batch_size % per_step:
        raise ValueError(f"global_batch_size {config.global_batch_size} is not a multiple of micro_batch_size x GPUs = {per_step}")
    grad_accum = config.global_batch_size // per_step
    tokens_per_step = config.global_batch_size * config.data.seq_len

    model = MykLLM(config.model).to(info.device)
    initialize_weights(model, config.init)
    log0(info, f"model: {model.num_parameters() / 1e6:.1f}M params ({model.num_parameters(exclude_embeddings=True) / 1e6:.1f}M non-embedding), layers {config.model.layout}")
    model = parallelize(model, config.parallel, info)
    optimizers = build_optimizers(model, config.optimizer)
    base_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]

    from ..data import build_fixed_batches, build_loader

    loader = build_loader(config.data, info.rank, info.world_size)
    from .objectives import Objective

    objective = Objective(config.objective, info.device)

    start_step, tokens_seen = 1, 0
    checkpoint = latest_checkpoint(run_dir) if resume == "auto" else (Path(resume) if resume != "none" else None)
    if checkpoint is not None:
        trainer_state, data_state = load_checkpoint(checkpoint, model, optimizers, info)
        start_step, tokens_seen = trainer_state["step"] + 1, trainer_state["tokens"]
        if data_state is not None:
            loader.load_state_dict(data_state)
        else:
            log0(info, "warning: no data-loader state for this rank; the data stream restarts")
        log0(info, f"resumed from {checkpoint} at step {start_step}")
    elif config.init.checkpoint_from:
        load_checkpoint(Path(config.init.checkpoint_from), model, optimizers if config.init.load_optimizer else None, info)
        log0(info, f"warm start from {config.init.checkpoint_from} (optimizer {'carried' if config.init.load_optimizer else 'fresh'})")

    val_sets = {
        vs.name: build_fixed_batches(replace(config.data, kind=vs.kind or config.data.kind, sources=vs.sources), config.eval.batches, config.eval.seed)
        for vs in config.eval.val_sets
    } if config.eval.interval > 0 else {}
    token_bytes = token_byte_lengths(tokenizer)

    metrics = None
    if info.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        name = "config.yaml" if checkpoint is None else f"config.resume-{start_step:07d}.yaml"
        (run_dir / name).write_text(dump_yaml(config), encoding="utf-8")
        from .metrics import MetricsLogger

        metrics = MetricsLogger(run_dir / "metrics.jsonl", run_dir / "tb" if config.logging.tensorboard else None)
    log0(info, f"run {run_dir}: {info.world_size} GPU(s), micro batch {config.data.micro_batch_size} x accum {grad_accum} "
               f"= {config.global_batch_size} seqs = {tokens_per_step:,} tokens/step, steps {start_step}..{config.steps}")

    params = [p for p in model.parameters() if p.requires_grad]
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if config.parallel.param_dtype == "fp32" else nullcontext()
    data_iter = iter(loader)
    model.train()
    window_start, window_tokens = time.perf_counter(), 0
    sums: dict[str, float] = {}
    step_tokens_total = 0.0

    def run_validation(step: int) -> None:
        record = validate(model, objective, val_sets, token_bytes, info)
        if metrics is not None:
            metrics.log(step, record)
        log0(info, f"step {step:6d} [val] " + " ".join(f"{k}={v:.4f}" for k, v in record.items()))

    for step in range(start_step, config.steps + 1):
        set_lr(optimizers, base_lrs, lr_multiplier(step, config.steps, config.schedule))
        micro_batches = [next(data_iter) for _ in range(grad_accum)]
        normalizer = all_reduce_sum(sum(b.num_targets for b in micro_batches), info.device)
        for batch in micro_batches:
            batch = batch.to(info.device)
            with autocast:
                out = model(batch.input_ids, seqlens=batch.seqlens, head=objective.head(batch, normalizer))
            out.loss.backward()
            for key, value in out.sums.items():
                sums[key] = sums.get(key, 0.0) + float(value)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, config.optimizer.grad_clip or float("inf"))
        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        tokens_seen += tokens_per_step
        window_tokens += tokens_per_step
        step_tokens_total += normalizer

        if step % config.logging.interval == 0 or step == config.steps:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - window_start
            record = {f"train/{k}": all_reduce_sum(v, info.device) / max(1.0, step_tokens_total) for k, v in sums.items()}
            record["train/loss"] = record.get("train/ce", float("nan"))
            record.update({
                "train/grad_norm": float(grad_norm),
                "train/lr": optimizers[-1].param_groups[0]["lr"],
                "perf/tokens_per_s": window_tokens / elapsed,
                "perf/tokens_per_s_per_gpu": window_tokens / elapsed / info.world_size,
                "perf/peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                "tokens": tokens_seen,
            })
            if metrics is not None:
                metrics.log(step, record)
            log0(info, f"step {step:6d} loss {record['train/loss']:.4f} lr {record['train/lr']:.2e} "
                       f"gnorm {record['train/grad_norm']:.2f} {record['perf/tokens_per_s']:,.0f} tok/s "
                       f"mem {record['perf/peak_mem_gb']:.1f}GB")
            window_start, window_tokens, sums, step_tokens_total = time.perf_counter(), 0, {}, 0.0

        if val_sets and (step % config.eval.interval == 0 or step == config.steps):
            run_validation(step)

        if config.eval.sample_interval and step % config.eval.sample_interval == 0:
            for text in sample_text(model, tokenizer, config, info):
                log0(info, f"--- sample ---\n{text}")

        if step == config.steps or (config.checkpoint.interval and step % config.checkpoint.interval == 0):
            target = step_dir(run_dir, step)
            trainer_state = {"step": step, "tokens": tokens_seen, "world_size": info.world_size}
            save_checkpoint(
                target, model, optimizers, trainer_state, loader.state_dict(), info,
                config.checkpoint.export_weights, config.data.tokenizer,
            )
            if info.is_main:
                prune_checkpoints(run_dir, config.checkpoint.keep_last, config.checkpoint.keep_every)
            log0(info, f"saved {target}")

    if val_sets and start_step > config.steps:  # resumed a finished run: validate its final checkpoint
        run_validation(config.steps)
    if metrics is not None:
        metrics.close()
    loader.close()
    shutdown()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="YAML run config")
    parser.add_argument("overrides", nargs="*", help="key.path=value overrides")
    parser.add_argument("--resume", default="auto", help="'auto' (newest checkpoint of the run), 'none', or a checkpoint dir")
    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HOME", str(Path.cwd() / ".cache" / "hf"))
    train(load_config(TrainConfig, args.config, args.overrides), resume=args.resume)
