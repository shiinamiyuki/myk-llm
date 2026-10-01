"""Token mixers: softmax attention (global / sliding-window) and Gated DeltaNet.

Every mixer has the same call signature so the transformer block does not care which
one it holds:

    mixer(x, position_ids, varlen, state, use_cache) -> (output, new_state)

`varlen` carries packed-document boundaries for document-masked training (tokens never
mix across documents); `state` is the mixer's own inference cache (KV cache for
attention, recurrent + convolution state for DeltaNet).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_varlen_func

from .config import DELTANET, GLOBAL_ATTENTION, LOCAL_ATTENTION, ModelConfig
from .layers import RMSNorm, RotaryEmbedding


@dataclass
class Varlen:
    """Packed-document layout of a (batch, seq_len) block, flattened row-major."""

    cu_seqlens: torch.Tensor  # int32, (num_docs + 1,)
    max_seqlen: int


@dataclass
class KVCache:
    keys: torch.Tensor  # (batch, cached_len, num_kv_heads, head_dim)
    values: torch.Tensor


class Attention(nn.Module):
    """Grouped-query softmax attention via FlashAttention, global or sliding-window."""

    def __init__(self, config: ModelConfig, local: bool, use_rope: bool) -> None:
        super().__init__()
        attn = config.attention
        self.num_heads, self.num_kv_heads, self.head_dim = attn.num_heads, attn.num_kv_heads, attn.head_dim
        self.local = local
        self.window = attn.sliding_window if local else None
        self.max_seq_len = config.max_seq_len
        q_size, kv_size = attn.num_heads * attn.head_dim, attn.num_kv_heads * attn.head_dim
        self.split = (q_size, kv_size, kv_size)
        self.qkv_proj = nn.Linear(config.hidden_size, q_size + 2 * kv_size, bias=False)
        self.o_proj = nn.Linear(q_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(attn.head_dim, config.norm_eps) if attn.qk_norm else None
        self.k_norm = RMSNorm(attn.head_dim, config.norm_eps) if attn.qk_norm else None
        self.gate_proj = nn.Linear(config.hidden_size, q_size, bias=False) if attn.output_gate else None
        theta = attn.rope_theta_local if local else attn.rope_theta_global
        self.rope = RotaryEmbedding(attn.head_dim, config.max_seq_len, theta) if use_rope else None

    def residual_projections(self) -> list[nn.Linear]:
        return [self.o_proj]

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        varlen: Varlen | None = None,
        state: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        batch, seq_len, _ = x.shape
        q, k, v = self.qkv_proj(x).split(self.split, dim=-1)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        if self.q_norm is not None and self.k_norm is not None:
            # The fp32 norm weight would promote q/k to fp32; FlashAttention needs bf16.
            q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        if self.rope is not None:
            q, k = self.rope(q, k, position_ids)
        window = (self.window - 1, 0) if self.window is not None else (-1, -1)

        new_state = None
        if varlen is not None:
            out = flash_attn_varlen_func(
                q.reshape(-1, self.num_heads, self.head_dim),
                k.reshape(-1, self.num_kv_heads, self.head_dim),
                v.reshape(-1, self.num_kv_heads, self.head_dim),
                cu_seqlens_q=varlen.cu_seqlens,
                cu_seqlens_k=varlen.cu_seqlens,
                max_seqlen_q=varlen.max_seqlen,
                max_seqlen_k=varlen.max_seqlen,
                causal=True,
                window_size=window,
            )
        else:
            if state is not None:
                k = torch.cat([state.keys, k], dim=1)
                v = torch.cat([state.values, v], dim=1)
            # Attend over the full k/v; only the cache kept for the next step is trimmed
            # (early queries of a long prefill still need their own older keys).
            if use_cache:
                limit = self.window if self.window is not None else self.max_seq_len
                new_state = KVCache(k[:, -limit:], v[:, -limit:])
            out = flash_attn_func(q, k, v, causal=True, window_size=window)
        out = out.reshape(batch, seq_len, -1)
        if self.gate_proj is not None:
            out = out * torch.sigmoid(self.gate_proj(x))
        return self.o_proj(out), new_state


class _FlaLayerState:
    """Minimal stand-in for fla's `Cache`, holding one layer's recurrent state."""

    def __init__(self) -> None:
        self.state: dict[str, Any] | None = None

    def __len__(self) -> int:
        return 0 if self.state is None else 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        assert self.state is not None
        return self.state

    def update(self, layer_idx: int, recurrent_state=None, conv_state=None, offset: int = 1, **_) -> None:
        self.state = {"recurrent_state": recurrent_state, "conv_state": conv_state}


class DeltaNet(nn.Module):
    """Gated DeltaNet linear attention (Yang et al., 2024) from flash-linear-attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # fla's TileLang backward kernel (chunk_bwd_dqkwg) hits "misaligned address" on
        # RTX 5090 (fla 0.5.1, tilelang); fall back to its Triton kernels unless set.
        os.environ.setdefault("FLA_TILELANG", "0")
        try:
            from fla.layers import GatedDeltaNet
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError("'D' layers need flash-linear-attention: pip install flash-linear-attention") from error
        cfg = config.deltanet
        self.inner = GatedDeltaNet(
            hidden_size=config.hidden_size,
            expand_v=cfg.expand_v,
            head_dim=cfg.head_dim,
            num_heads=cfg.num_heads,
            use_gate=cfg.use_gate,
            use_short_conv=cfg.use_short_conv,
            conv_size=cfg.conv_size,
            layer_idx=0,  # each layer owns its own single-entry state object
            mode="chunk",
        )

    def residual_projections(self) -> list[nn.Linear]:
        return [self.inner.o_proj]

    # fla's layers are hand-written Triton kernels; tracing them into a compiled block
    # gains nothing, and Inductor fails on the inference-mode (fused_recurrent) path.
    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        varlen: Varlen | None = None,
        state: _FlaLayerState | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, _FlaLayerState | None]:
        if varlen is not None:
            batch, seq_len, hidden = x.shape
            out, _, _ = self.inner(x.reshape(1, batch * seq_len, hidden), cu_seqlens=varlen.cu_seqlens)
            return out.reshape(batch, seq_len, hidden), None
        if use_cache and state is None:
            state = _FlaLayerState()
        out, _, _ = self.inner(x, past_key_values=state if use_cache else None, use_cache=use_cache)
        return out, state if use_cache else None


def build_mixer(config: ModelConfig, layer_type: str, first_global: bool) -> nn.Module:
    if layer_type == LOCAL_ATTENTION:
        return Attention(config, local=True, use_rope=True)
    if layer_type == GLOBAL_ATTENTION:
        rope = config.attention.global_rope
        return Attention(config, local=False, use_rope=rope == "all" or (rope == "first" and first_global))
    if layer_type == DELTANET:
        return DeltaNet(config)
    raise ValueError(f"unknown layer type {layer_type!r}")
