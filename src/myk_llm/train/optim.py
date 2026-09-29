"""Optimizers (Muon + AdamW) and learning-rate schedules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, distribute_tensor

from .config import OptimizerConfig, ScheduleConfig

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def newton_schulz_orthogonalize(grad: torch.Tensor, steps: int, eps: float = 1e-7) -> torch.Tensor:
    """Approximate U V^T of grad = U S V^T with a quintic Newton-Schulz iteration (bf16)."""
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


class Muon(torch.optim.Optimizer):
    """Muon (momentum + orthogonalized updates) for 2-D weight matrices.

    Numerically matches `torch.optim.Muon`, but also accepts FSDP-sharded parameters
    (DTensor): momentum is kept on the local shard, and the orthogonalization, which
    needs the whole matrix, runs on an all-gathered copy. Every rank orthogonalizes
    every matrix (redundant but simple; fine up to ~1B parameters).
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

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1 - group["momentum"])
                update = grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                if isinstance(update, DTensor):
                    full = newton_schulz_orthogonalize(update.full_tensor(), group["ns_steps"])
                    update = distribute_tensor(full, update.device_mesh, update.placements, src_data_rank=None)
                else:
                    update = newton_schulz_orthogonalize(update, group["ns_steps"])
                lr = group["lr"]
                param.mul_(1 - lr * group["weight_decay"])
                param.add_(update.to(param.dtype), alpha=-lr * self._lr_ratio(param.shape, group["adjust_lr"]))


@dataclass
class ParameterGroups:
    muon: list[nn.Parameter]
    decay: list[nn.Parameter]
    no_decay: list[nn.Parameter]


def split_parameters(model: nn.Module, use_muon: bool) -> ParameterGroups:
    """Muon: 2-D weights inside blocks. AdamW without decay: embeddings (token, n-gram)
    and 1-D parameters (norms, gates). AdamW with decay: everything else."""
    groups = ParameterGroups([], [], [])
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
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
