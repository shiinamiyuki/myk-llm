"""Optimizers (Muon + AdamW) and learning-rate schedules."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from .config import OptimizerConfig, ScheduleConfig

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def newton_schulz_orthogonalize(grad: torch.Tensor, steps: int, eps: float = 1e-7) -> torch.Tensor:
    """Approximate U V^T of grad = U S V^T with a quintic Newton-Schulz iteration in
    Brain Floating Point 16-bit (BF16)."""
    a, b, c = NS_COEFFICIENTS
    x = grad.bfloat16()
    transposed = grad.size(0) > grad.size(1)
    if transposed:
        x = x.T
    x = x / x.norm().clamp(min=eps)
    for _ in range(steps):
        gram = x @ x.T
        x = torch.addmm(x, torch.addmm(gram, gram, gram, beta=b, alpha=c), x, beta=a)
    return x.T if transposed else x


def _can_fast_cuda_muon(p: torch.Tensor, g: torch.Tensor, m: torch.Tensor) -> bool:
    supported = (torch.float32, torch.bfloat16)
    return (
        not isinstance(p, DTensor)
        and not isinstance(g, DTensor)
        and p.is_cuda
        and p.ndim == 2
        and p.is_contiguous()
        and g.is_contiguous()
        and m.is_contiguous()
        and p.dtype in supported
        and g.dtype in supported
        and m.dtype in supported
    )


def _can_owner_sharded_fsdp_muon(p: torch.Tensor, g: torch.Tensor) -> bool:
    if not (isinstance(p, DTensor) and isinstance(g, DTensor) and p.is_cuda and p.ndim == 2):
        return False
    mesh = p.device_mesh
    if mesh.ndim != 1 or len(p.placements) != 1 or p.placements != g.placements:
        return False
    placement = p.placements[0]
    if not (isinstance(placement, Shard) and placement.dim == 0):
        return False
    world = mesh.size()
    if p.shape[0] % world != 0:
        return False
    if world > 1 and (not dist.is_initialized() or dist.get_backend(mesh.get_group()) != "nccl"):
        return False
    return p.to_local().is_contiguous() and g.to_local().is_contiguous()


class Muon(torch.optim.Optimizer):
    """Muon (momentum + orthogonalized updates) for 2-D weight matrices.

    Numerically matches `torch.optim.Muon`. On Compute Unified Device Architecture (CUDA)
    tensors, parameters are grouped by shape and updated with fused Triton pack / unpack
    kernels and symmetric Gram products (`ops.muon`). Under Fully Sharded Data Parallel
    (FSDP) (`DTensor`), momentum stays on each rank's local shard and full matrices are
    distributed across ranks via BF16 all-to-all so each rank orthogonalizes `1 / world_size`
    of the matrices (falling back to `full_tensor()` on Central Processing Unit (CPU) or
    uneven shards).
    """

    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adjust_lr: Literal["original", "match_rms_adamw", "none"] = "match_rms_adamw",
    ) -> None:
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, adjust_lr=adjust_lr)
        super().__init__(params, defaults)

    @staticmethod
    def _lr_ratio(shape: torch.Size, mode: str) -> float:
        rows, cols = shape[0], shape[1]
        if mode == "original":
            return math.sqrt(max(1.0, rows / cols))
        if mode == "match_rms_adamw":
            return 0.2 * math.sqrt(max(rows, cols))
        return 1.0

    def _step_fsdp_bucket(
        self,
        items: list[tuple[DTensor, DTensor, DTensor]],
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool,
        ns_steps: int,
        adjust_lr: str,
    ) -> None:
        from ..ops.muon import batched_newton_schulz_orthogonalize

        sample_p = items[0][0]
        shape = sample_p.shape
        rows, cols = shape
        mesh = sample_p.device_mesh
        world = mesh.size()
        shard_rows = rows // world
        device = sample_p.to_local().device

        local_params = [p.to_local() for p, _, _ in items]
        local_grads = [g.to_local() for _, g, _ in items]
        local_bufs = [b.to_local() for _, _, b in items]
        torch._foreach_lerp_(local_bufs, local_grads, 1.0 - momentum)
        local_updates = torch._foreach_lerp(local_grads, local_bufs, momentum) if nesterov else local_bufs

        if world == 1:
            packed = torch.stack([u.bfloat16() for u in local_updates], dim=0)
            orth_shards = batched_newton_schulz_orthogonalize(packed, ns_steps, transposed=(rows > cols))
        else:
            pg = mesh.get_group()
            pad = (-len(items)) % world
            per_rank = (len(items) + pad) // world
            bf16_list = [u.bfloat16() for u in local_updates]
            if pad:
                bf16_list.extend(torch.zeros((shard_rows, cols), device=device, dtype=torch.bfloat16) for _ in range(pad))
            send = torch.stack(bf16_list, dim=0).view(world, per_rank, shard_rows, cols).contiguous()
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=pg)
            owned = recv.permute(1, 0, 2, 3).contiguous().view(per_rank, rows, cols)
            orth_owned = batched_newton_schulz_orthogonalize(owned, ns_steps, transposed=(rows > cols))
            send_back = orth_owned.view(per_rank, world, shard_rows, cols).permute(1, 0, 2, 3).contiguous()
            dist.all_to_all_single(recv, send_back, group=pg)
            orth_shards = recv.view(per_rank * world, shard_rows, cols)[: len(items)]

        if weight_decay != 0.0:
            torch._foreach_mul_(local_params, 1.0 - lr * weight_decay)
        step_scale = -lr * self._lr_ratio(shape, adjust_lr)
        torch._foreach_add_(local_params, [orth_shards[i].to(local_params[i].dtype) for i in range(len(items))], alpha=step_scale)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        for group in self.param_groups:
            lr: float = group["lr"]
            wd: float = group["weight_decay"]
            mom: float = group["momentum"]
            nesterov: bool = group["nesterov"]
            ns_steps: int = group["ns_steps"]
            adjust_lr: str = group["adjust_lr"]

            cuda_triples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            fsdp_buckets: dict[tuple[torch.Size, torch.dtype], list[tuple[DTensor, DTensor, DTensor]]] = defaultdict(list)
            fallback: list[tuple[nn.Parameter, torch.Tensor, torch.Tensor]] = []

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                if _can_fast_cuda_muon(param, grad, buf):
                    cuda_triples.append((param, grad, buf))
                elif _can_owner_sharded_fsdp_muon(param, grad) and isinstance(buf, DTensor):
                    fsdp_buckets[(param.shape, param.dtype)].append((param, grad, buf))  # type: ignore[arg-type]
                else:
                    fallback.append((param, grad, buf))

            if cuda_triples:
                from ..ops.muon import muon_step_cuda

                muon_step_cuda(cuda_triples, lr, wd, mom, nesterov, ns_steps, adjust_lr, self._lr_ratio)

            for items in fsdp_buckets.values():
                self._step_fsdp_bucket(items, lr, wd, mom, nesterov, ns_steps, adjust_lr)

            for param, grad, buf in fallback:
                buf.lerp_(grad, 1 - mom)
                update = grad.lerp(buf, mom) if nesterov else buf
                if isinstance(update, DTensor):
                    full = newton_schulz_orthogonalize(update.bfloat16().full_tensor(), ns_steps)
                    update = distribute_tensor(full, update.device_mesh, update.placements, src_data_rank=None)
                else:
                    update = newton_schulz_orthogonalize(update, ns_steps)
                param.mul_(1 - lr * wd)
                param.add_(update.to(param.dtype), alpha=-lr * self._lr_ratio(param.shape, adjust_lr))


@dataclass
class ParameterGroups:
    muon: list[nn.Parameter]
    decay: list[nn.Parameter]
    no_decay: list[nn.Parameter]
    ngram: list[nn.Parameter]


def split_parameters(model: nn.Module, use_muon: bool) -> ParameterGroups:
    """Muon: 2-D weights inside blocks. AdamW without decay: the token embedding and
    1-D parameters (norms, gates). The n-gram tables: their own AdamW group
    (optimizer.ngram). AdamW with decay: everything else."""
    groups = ParameterGroups([], [], [], [])
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if name.startswith("embed.ngram.tables."):
            groups.ngram.append(param)
            continue
        is_embedding = name.startswith("embed.") and "proj" not in name or name == "lm_head.weight"
        if is_embedding or param.ndim < 2:
            groups.no_decay.append(param)
        elif use_muon and param.ndim == 2:
            groups.muon.append(param)
        else:
            groups.decay.append(param)
    return groups


def build_optimizers(model: nn.Module, config: OptimizerConfig) -> list[torch.optim.Optimizer]:
    groups = split_parameters(model, use_muon=config.name == "muon")
    optimizers: list[torch.optim.Optimizer] = []
    if groups.muon:
        optimizers.append(
            Muon(
                groups.muon,
                lr=config.muon_lr if config.muon_lr is not None else config.lr,
                weight_decay=config.muon_weight_decay if config.muon_weight_decay is not None else config.weight_decay,
                momentum=config.muon_momentum,
                nesterov=config.muon_nesterov,
                ns_steps=config.muon_ns_steps,
                adjust_lr=config.muon_adjust_lr,
            )
        )
    adam_groups = [
        {"params": groups.decay, "weight_decay": config.weight_decay},
        {"params": groups.no_decay, "weight_decay": 0.0},
        {"params": groups.ngram, "weight_decay": config.ngram.weight_decay, "lr": config.lr * config.ngram.lr_scale},
    ]
    optimizers.append(
        torch.optim.AdamW(
            [g for g in adam_groups if g["params"]], lr=config.lr, betas=tuple(config.betas), eps=config.eps, fused=True
        )
    )
    return optimizers


def lr_multiplier(step: int, total_steps: int, config: ScheduleConfig) -> float:
    """LR multiplier for 1-based `step`: linear warmup, then cosine / WSD / constant."""
    warmup = min(total_steps, int(total_steps * config.warmup_ratio))
    if warmup and step <= warmup:
        return step / warmup

    def cosine(progress: float) -> float:
        return config.min_lr_ratio + (1 - config.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    if config.kind == "constant":
        return 1.0
    if config.kind == "cosine":
        return cosine((step - warmup) / max(1, total_steps - warmup))
    # Warmup-Stable-Decay (WSD): hold the peak, then cosine-decay over the last decay_ratio.
    decay_steps = min(total_steps - warmup, int(total_steps * config.decay_ratio))
    decay_start = total_steps - decay_steps
    if decay_steps == 0 or step <= decay_start:
        return 1.0
    return cosine((step - decay_start) / decay_steps)


def set_lr(optimizers: list[torch.optim.Optimizer], base_lrs: list[list[float]], multiplier: float) -> None:
    for optimizer, lrs in zip(optimizers, base_lrs):
        for group, base in zip(optimizer.param_groups, lrs):
            group["lr"] = base * multiplier
