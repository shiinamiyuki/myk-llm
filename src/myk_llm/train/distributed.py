"""Process-group setup and model sharding (FSDP2: torch's fully_shard)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

from ..model import MykLLM
from .config import ParallelConfig


@dataclass
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


def init_distributed() -> DistInfo:
    """Initialize from torchrun's environment variables; plain `python` runs single-GPU."""
    if not torch.cuda.is_available():
        raise SystemExit("training is CUDA-only; no GPU is available")
    if "RANK" not in os.environ:
        torch.cuda.set_device(0)
        return DistInfo(0, 1, 0, torch.device("cuda", 0))
    rank, world_size, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=device)
    return DistInfo(rank, world_size, local_rank, device)


def shutdown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_sum(value: float | int, device: torch.device) -> float:
    if not dist.is_initialized():
        return float(value)
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(tensor)
    return float(tensor)


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def uses_fsdp(config: ParallelConfig, info: DistInfo) -> bool:
    if config.strategy == "auto":
        return info.enabled
    return config.strategy == "fsdp"


def parallelize(model: MykLLM, config: ParallelConfig, info: DistInfo) -> MykLLM:
    """Activation checkpointing, torch.compile and FSDP2 sharding, in that order.

    Each block is its own FSDP unit (all-gathered just before it runs); the root unit
    holds the embeddings, final norm and output head. Gradients are reduced as a SUM:
    the objectives already normalize by the global number of supervised tokens.
    """
    if config.activation_checkpointing:
        model.set_activation_checkpointing(True)
    if config.compile:
        model.compile_blocks()
    if not uses_fsdp(config, info):
        return model
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16 if config.param_dtype == "bf16" else None, reduce_dtype=torch.float32
    )
    for block in model.blocks:
        fully_shard(block, mp_policy=policy, reshard_after_forward=config.reshard_after_forward)
    fully_shard(model, mp_policy=policy, reshard_after_forward=config.reshard_after_forward)
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)
    return model
