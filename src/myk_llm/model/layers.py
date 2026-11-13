"""Small building blocks shared by the mixers and the transformer block."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, ResidualConfig


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (RMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * self.weight


class SwiGLU(nn.Module):
    """Gated feed-forward network with a SiLU (sigmoid linear unit) gate."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.up_gate_proj = nn.Linear(config.hidden_size, 2 * config.ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.up_gate_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class ResidualGate(nn.Module):
    """Scales a sublayer's output before it joins the residual stream (see ResidualConfig)."""

    def __init__(self, config: ResidualConfig, hidden_size: int) -> None:
        super().__init__()
        self.kind = config.gate
        if self.kind == "scalar":
            self.scale = nn.Parameter(torch.full((1,), config.init))
        elif self.kind == "channel":
            self.scale = nn.Parameter(torch.full((hidden_size,), config.init))
        elif self.kind == "input":
            self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def init_weights(self) -> None:
        if self.kind == "input":
            nn.init.zeros_(self.gate_proj.weight)

    def forward(self, residual: torch.Tensor, update: torch.Tensor, normed_input: torch.Tensor) -> torch.Tensor:
        if self.kind == "none":
            return residual + update
        if self.kind == "input":
            return residual + 2.0 * torch.sigmoid(self.gate_proj(normed_input)) * update
        return residual + self.scale * update


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # Interleaved-pair convention (dims 2i, 2i+1 rotate together).
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE), applied in fp32."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        self.head_dim, self.base = head_dim, base
        self._build(max_seq_len)

    def _build(self, length: int, device: torch.device | None = None) -> None:
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim))
        angles = torch.outer(torch.arange(length, dtype=torch.float32, device=device), inv_freq)
        angles = torch.repeat_interleave(angles, repeats=2, dim=-1)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def _apply(self, fn, recurse: bool = True):  # keep the tables fp32 under .to(bf16)
        super()._apply(fn, recurse=recurse)
        self.cos_cached = cast(torch.Tensor, self.cos_cached).float()
        self.sin_cached = cast(torch.Tensor, self.sin_cached).float()
        return self

    def rotate(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Rotate (batch, seq, heads, head_dim) queries or keys to their positions."""
        cos = cast(torch.Tensor, self.cos_cached)[position_ids].unsqueeze(2)
        sin = cast(torch.Tensor, self.sin_cached)[position_ids].unsqueeze(2)
        x32 = x.float()
        return (x32 * cos + rotate_half(x32) * sin).to(x.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotate(q, position_ids), self.rotate(k, position_ids)


def init_linear_(module: nn.Module, std: float) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


def residual_init_scale(num_layers: int) -> float:
    return 1.0 / math.sqrt(2 * num_layers)
