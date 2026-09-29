from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from flash_attn import flash_attn_func, flash_attn_varlen_func
import triton
import triton.language as tl


FULL_ATTENTION = "full_attention"
SLIDING_WINDOW_ATTENTION = "sliding_window_attention"

# The compiled TransformerBlock.forward legitimately specializes on a handful of
# structural branches — sliding vs full attention, rope present vs absent, and
# prefill (no KV cache) vs cached decode — which combine to more distinct graphs
# than dynamo's default per-frame recompile limit of 8. Each compiles once and is
# then reused (shape churn is handled separately via _mark_seq_dynamic during
# generation), so raise the ceiling to give those variants room instead of falling
# back to slow eager once the limit is hit.
for _limit_attr in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _limit_attr):
        setattr(torch._dynamo.config, _limit_attr, max(64, getattr(torch._dynamo.config, _limit_attr)))


@triton.jit
def _fused_ce_kernel(
    logits_ptr,  # (rows, n_cols) fp32, overwritten in-place with d_loss/d_logits
    targets_ptr,  # (rows,) int64
    loss_ptr,  # (rows,) fp32, per-row negative log likelihood
    inv_valid,  # 1 / num_supervised_tokens, folded into the gradient scale
    n_cols,  # vocab size
    row_stride,
    ignore_index,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_start = logits_ptr + row * row_stride
    target = tl.load(targets_ptr + row).to(tl.int32)

    if target == ignore_index:
        for off in range(0, n_cols, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            tl.store(row_start + cols, tl.zeros([BLOCK], tl.float32), mask=cols < n_cols)
        tl.store(loss_ptr + row, 0.0)
        return

    # Pass 1: streaming max + sum-of-exp (online softmax) to get the logsumexp.
    m = -float("inf")
    d = 0.0
    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        vals = tl.load(row_start + cols, mask=cols < n_cols, other=-float("inf")).to(tl.float32)
        block_max = tl.max(vals, axis=0)
        new_m = tl.maximum(m, block_max)
        d = d * tl.exp(m - new_m) + tl.sum(tl.exp(vals - new_m), axis=0)
        m = new_m
    lse = m + tl.log(d)

    target_logit = tl.load(row_start + target).to(tl.float32)
    tl.store(loss_ptr + row, lse - target_logit)

    # Pass 2: grad = (softmax - onehot) * inv_valid, written over the logits.
    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < n_cols
        vals = tl.load(row_start + cols, mask=mask, other=0.0).to(tl.float32)
        grad = tl.exp(vals - lse) * inv_valid
        grad = tl.where(cols == target, grad - inv_valid, grad)
        tl.store(row_start + cols, grad, mask=mask)


class _FusedLinearCrossEntropy(torch.autograd.Function):
    """Linear projection + cross entropy that never materializes the full (N, V) logits.

    Tokens are processed in chunks; each chunk's logits live only long enough for the
    Triton kernel to turn them into gradients in-place, which are then folded into
    grad_hidden / grad_weight. Peak head memory is O(chunk * V) instead of O(N * V).
    """

    @staticmethod
    def forward(ctx, hidden, weight, targets, ignore_index, chunk_size):
        if not hidden.is_cuda:
            raise RuntimeError("fused cross entropy is CUDA-only")
        num_tokens, hidden_size = hidden.shape
        vocab_size = weight.shape[0]

        n_valid = int((targets != ignore_index).sum().item())
        if n_valid == 0:
            raise ValueError("targets did not contain any supervised positions")
        inv_valid = 1.0 / n_valid

        # The projection and gradient GEMMs run in bf16 (matching the rest of the model's
        # mixed-precision matmuls); tensor cores still accumulate in fp32. The numerically
        # sensitive softmax/logsumexp is done in fp32 inside the kernel regardless.
        compute_dtype = torch.bfloat16
        weight_c = weight.to(compute_dtype)
        grad_hidden = torch.empty(num_tokens, hidden_size, device=hidden.device, dtype=compute_dtype)
        grad_weight = torch.zeros(vocab_size, hidden_size, device=hidden.device, dtype=torch.float32)
        loss_total = torch.zeros((), device=hidden.device, dtype=torch.float32)

        with torch.autocast(device_type="cuda", enabled=False):
            for start in range(0, num_tokens, chunk_size):
                end = min(start + chunk_size, num_tokens)
                rows = end - start
                hidden_chunk = hidden[start:end].to(compute_dtype)
                logits = hidden_chunk @ weight_c.t()  # (rows, V) bf16
                losses = torch.empty(rows, device=hidden.device, dtype=torch.float32)
                _fused_ce_kernel[(rows,)](
                    logits,
                    targets[start:end],
                    losses,
                    inv_valid,
                    vocab_size,
                    logits.stride(0),
                    ignore_index,
                    BLOCK=4096,
                    num_warps=8,
                )
                # logits now holds d_loss/d_logits in bf16 (already scaled by inv_valid).
                torch.mm(logits, weight_c, out=grad_hidden[start:end])
                grad_weight.add_(logits.t() @ hidden_chunk)  # bf16 product upcast into fp32 accumulator
                loss_total += losses.sum()

        ctx.save_for_backward(grad_hidden, grad_weight)
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = weight.dtype
        return loss_total * inv_valid

    @staticmethod
    def backward(ctx, grad_output):  # pyright: ignore[reportIncompatibleMethodOverride]
        grad_hidden, grad_weight = ctx.saved_tensors
        grad_hidden = (grad_hidden * grad_output).to(ctx.hidden_dtype)
        grad_weight = (grad_weight * grad_output).to(ctx.weight_dtype)
        return grad_hidden, grad_weight, None, None, None


def fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    chunk_size: int = 1024,
) -> torch.Tensor:
    return cast(torch.Tensor, _FusedLinearCrossEntropy.apply(hidden, weight, targets, ignore_index, chunk_size))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def build_cu_seqlens(seqlens: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Cumulative document boundaries for flash-attn varlen over a packed batch.

    `seqlens` is the per-sample list of document lengths carried by the streaming collate
    (each sample's lengths sum to seq_len). Tokens are laid out (batch, seq_len) row-major,
    so concatenating the per-sample lengths across the batch and prefixing a 0 gives the
    block-diagonal boundaries for the whole flattened batch: cu[i]..cu[i+1] is one document.
    """
    lengths = torch.cat([segment.to(device=device, dtype=torch.int32) for segment in seqlens])
    return F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))


def _mark_seq_dynamic(tensor: torch.Tensor, dim: int) -> None:
    """Hint dynamo that `tensor`'s `dim` varies, so it compiles one reusable dynamic
    graph instead of specializing (and recompiling) per length. `maybe_mark_dynamic`
    is a soft hint that tolerates size-1 dims (e.g. the single decode token)."""
    marker = getattr(torch._dynamo, "maybe_mark_dynamic", None)
    if marker is None:  # very old torch: fall back to the hard marker, ignore if it rejects size-1
        marker = torch._dynamo.mark_dynamic
    try:
        marker(tensor, dim)
    except Exception:
        pass


@dataclass
class MykLLMConfig:
    vocab_size: int = 65536
    max_seq_len: int = 4096
    head_dim: int = 128
    num_kv_heads: int = 4
    num_attention_heads: int = 16
    num_layers: int = 13
    sliding_window_size: int = 512
    ffn_multiplier: float = 4.0
    ffn_multiple_of: int = 256
    dropout: float = 0.0
    initializer_range: float = 0.02
    attn_types: list[str] = field(default_factory=list)
    full_attn_period: int = 4
    tie_embeddings: bool = True
    gradient_checkpointing: bool = False
    sliding_window_rope_theta: float = 10000.0
    full_attention_rope_theta: float = 1000000.0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_attention_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_kv_heads for GQA")
        if self.sliding_window_size <= 0:
            raise ValueError("sliding_window_size must be positive")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.attn_types and len(self.attn_types) != self.num_layers:
            raise ValueError("attn_types must have the same length as num_layers")
        if self.full_attn_period <= 0:
            raise ValueError("full_attn_period must be positive")

    @property
    def hidden_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_hidden_size(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def ff_hidden_size(self) -> int:
        raw_hidden_size = int(self.hidden_size * self.ffn_multiplier)
        return self.ffn_multiple_of * ((raw_hidden_size + self.ffn_multiple_of - 1) // self.ffn_multiple_of)

    @property
    def attention_pattern(self) -> list[str]:
        if self.attn_types:
            allowed = {FULL_ATTENTION, SLIDING_WINDOW_ATTENTION}
            invalid = set(self.attn_types) - allowed
            if invalid:
                raise ValueError(f"unsupported attention types: {sorted(invalid)}")
            return list(self.attn_types)

        # Full attention every `full_attn_period` layers (layer 0, P, 2P, ...);
        # all other layers use sliding-window attention. P=4 reproduces the
        # original [FULL] + [SWA, SWA, SWA, FULL] * N schedule.
        return [
            FULL_ATTENTION if layer_idx % self.full_attn_period == 0 else SLIDING_WINDOW_ATTENTION
            for layer_idx in range(self.num_layers)
        ]


@dataclass
class LayerKVCache:
    keys: torch.Tensor
    values: torch.Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * self.weight


class SwiGLU(nn.Module):
    def __init__(self, config: MykLLMConfig) -> None:
        super().__init__()
        self.up_gate_proj = nn.Linear(config.hidden_size, 2 * config.ff_hidden_size, bias=False)
        self.down_proj = nn.Linear(config.ff_hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.up_gate_proj(x).chunk(2, dim=-1)
        x = F.silu(gate) * up
        x = self.down_proj(x)
        return self.dropout(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        angles = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self.cos_cached = cast(torch.Tensor, self.cos_cached).to(dtype=torch.float32)
        self.sin_cached = cast(torch.Tensor, self.sin_cached).to(dtype=torch.float32)
        return self

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        if position_ids.size(-1) != q.size(1):
            raise ValueError("position_ids must align with the query sequence length")

        cos = cast(torch.Tensor, self.cos_cached)[position_ids].unsqueeze(2)
        sin = cast(torch.Tensor, self.sin_cached)[position_ids].unsqueeze(2)

        q_fp32 = q.float()
        k_fp32 = k.float()
        q_out = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
        k_out = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
        return q_out.to(dtype=q.dtype), k_out.to(dtype=k.dtype)


class FlashSelfAttention(nn.Module):
    def __init__(self, config: MykLLMConfig, attn_type: str, use_rope: bool) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.attn_type = attn_type
        self.use_rope = use_rope
        self.dropout_p = config.dropout
        self.sliding_window_size = config.sliding_window_size
        self.max_seq_len = config.max_seq_len
        self.qkv_split = (
            config.hidden_size,
            config.kv_hidden_size,
            config.kv_hidden_size,
        )

        self.qkv_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size + (2 * config.kv_hidden_size),
            bias=False,
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        rope_theta = config.sliding_window_rope_theta if attn_type == SLIDING_WINDOW_ATTENTION else config.full_attention_rope_theta    
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, rope_theta) if use_rope else None

    def _trim_cache(self, keys: torch.Tensor, values: torch.Tensor) -> LayerKVCache:
        cache_limit = self.sliding_window_size if self.attn_type == SLIDING_WINDOW_ATTENTION else self.max_seq_len
        if keys.size(1) > cache_limit:
            keys = keys[:, -cache_limit:, :, :]
            values = values[:, -cache_limit:, :, :]
        return LayerKVCache(keys=keys, values=values)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: LayerKVCache | None = None,
        use_cache: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int = 0,
    ) -> tuple[torch.Tensor, LayerKVCache | None]:
        if not x.is_cuda:
            raise RuntimeError("MykLLM is CUDA-only and requires flash-attn on GPU tensors")

        batch_size, seq_len, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {hidden_size}")

        q, k, v = self.qkv_proj(x).split(self.qkv_split, dim=-1)
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        if self.rope is not None:
            q, k = self.rope(q, k, position_ids)

        window_size = (-1, -1)
        if self.attn_type == SLIDING_WINDOW_ATTENTION:
            window_size = (self.sliding_window_size - 1, 0)

        if cu_seqlens is not None:
            # Document-masked (block-diagonal) attention: each packed document is its own
            # varlen sequence, so tokens never attend across the EOS seams. Causal and the
            # sliding window still apply, but only *within* a document. This is a training
            # path only, so KV caching is never active here (guarded in MykLLM.forward).
            attn_output = flash_attn_varlen_func(
                q=q.reshape(-1, self.num_attention_heads, self.head_dim),
                k=k.reshape(-1, self.num_kv_heads, self.head_dim),
                v=v.reshape(-1, self.num_kv_heads, self.head_dim),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.dropout_p if self.training else 0.0,
                causal=True,
                window_size=window_size,
            )
            attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
            return self.resid_dropout(self.o_proj(attn_output)), None

        if kv_cache is not None:
            k = torch.cat([kv_cache.keys, k], dim=1)
            v = torch.cat([kv_cache.values, v], dim=1)

        # Attend over the untrimmed k/v and only trim what is kept for the next step:
        # during a prefill longer than the sliding window, the early queries still need
        # their own (older) keys, which the trimmed cache no longer holds.
        next_cache = self._trim_cache(k, v) if use_cache else None

        attn_output = flash_attn_func(
            q=q,
            k=k,
            v=v,
            dropout_p=self.dropout_p if self.training else 0.0,
            causal=True,
            window_size=window_size,
        )
        attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
        attn_output = self.o_proj(attn_output)
        return self.resid_dropout(attn_output), next_cache


class TransformerBlock(nn.Module):
    def __init__(self, config: MykLLMConfig, attn_type: str, use_rope: bool) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = FlashSelfAttention(config, attn_type, use_rope)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.ffn = SwiGLU(config)

    @torch.compile
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: LayerKVCache | None = None,
        use_cache: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int = 0,
    ) -> tuple[torch.Tensor, LayerKVCache | None]:
        attn_out, next_cache = self.attn(
            self.attn_norm(x),
            position_ids,
            kv_cache=kv_cache,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, next_cache


class MykLLM(nn.Module):
    def __init__(self, config: MykLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        attention_pattern = config.attention_pattern
        self.layers = nn.ModuleList(
            [
                TransformerBlock(config, attn_type, use_rope=(layer_idx == 0 or attn_type == SLIDING_WINDOW_ATTENTION))
                for layer_idx, attn_type in enumerate(attention_pattern)
            ]
        )
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        self._scale_residual_projections()

    @property
    def device(self) -> torch.device:
        return self.token_embedding.weight.device

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _scale_residual_projections(self) -> None:
        scale = (2 * self.config.num_layers) ** -0.5
        for layer in cast(list[TransformerBlock], list(self.layers)):
            layer.attn.o_proj.weight.data.mul_(scale)
            layer.ffn.down_proj.weight.data.mul_(scale)

    def gradient_checkpointing_enabled(self) -> bool:
        return self.gradient_checkpointing

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    def _checkpointed_layer_forward(
        self,
        layer: TransformerBlock,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int,
    ) -> torch.Tensor:
        def custom_forward(hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            hidden_states, _ = layer(
                hidden_states,
                positions,
                kv_cache=None,
                use_cache=False,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            return hidden_states

        return cast(torch.Tensor, checkpoint(custom_forward, x, position_ids, use_reentrant=False))

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        kv_caches: list[LayerKVCache | None] | None = None,
        use_cache: bool = False,
        seqlens: list[torch.Tensor] | None = None,
        loss_chunk_size: int = 0,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list[LayerKVCache | None] | None]:
        if not input_ids.is_cuda:
            raise RuntimeError("MykLLM is CUDA-only and expects CUDA input tensors")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must use dtype torch.long")
        if loss_chunk_size < 0:
            raise ValueError("loss_chunk_size must be non-negative")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}")
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        if position_ids.shape != (batch_size, seq_len):
            raise ValueError("position_ids must have shape (batch_size, seq_len) or (seq_len,)")
        if kv_caches is not None and len(kv_caches) != len(self.layers):
            raise ValueError("kv_caches must align with the number of transformer layers")
        if self.gradient_checkpointing and self.training and use_cache:
            raise ValueError("gradient checkpointing cannot be combined with KV caching")

        cu_seqlens: torch.Tensor | None = None
        max_seqlen = 0
        if seqlens is not None:
            if use_cache:
                raise ValueError("document-masked attention (seqlens) is incompatible with KV caching")
            cu_seqlens = build_cu_seqlens(seqlens, input_ids.device)
            if int(cu_seqlens[-1]) != batch_size * seq_len:
                raise ValueError("seqlens must sum to batch_size * seq_len (packed windows must be full)")
            # Upper bound on any document length is the window length; keeping it a constant
            # (not the data-dependent max) avoids per-batch recompiles of the compiled block.
            max_seqlen = seq_len
            torch._dynamo.mark_dynamic(cu_seqlens, 0)

        x = self.token_embedding(input_ids)
        x = self.embedding_dropout(x)

        if use_cache:
            # Autoregressive decode grows the KV cache by one token per step, and prompt
            # lengths vary per turn, so the compiled TransformerBlock would otherwise
            # recompile for every distinct sequence length and blow the dynamo recompile
            # limit. Marking the sequence dims dynamic lets a single dynamic-shape graph
            # be reused across steps/turns. Only done on the generation path; training
            # keeps fully static shapes for the fast compiled kernels.
            _mark_seq_dynamic(x, 1)
            _mark_seq_dynamic(position_ids, position_ids.dim() - 1)
            if kv_caches is not None:
                for cache in kv_caches:
                    if cache is not None:
                        _mark_seq_dynamic(cache.keys, 1)
                        _mark_seq_dynamic(cache.values, 1)

        next_caches: list[LayerKVCache | None] | None = [] if use_cache else None

        for layer_idx, layer in enumerate(self.layers):
            layer = cast(TransformerBlock, layer)
            layer_cache = None if kv_caches is None else kv_caches[layer_idx]
            if self.gradient_checkpointing and self.training:
                x = self._checkpointed_layer_forward(layer, x, position_ids, cu_seqlens, max_seqlen)
                next_cache = None
            else:
                x, next_cache = layer(
                    x,
                    position_ids,
                    kv_cache=layer_cache,
                    use_cache=use_cache,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
            if next_caches is not None:
                next_caches.append(next_cache)

        hidden = self.final_norm(x)
        loss = None
        logits = None
        if targets is not None:
            if targets.shape != (batch_size, seq_len):
                raise ValueError("targets must have the same shape as input_ids")
            # Fused linear + cross entropy: never materializes the full (N, V) logits.
            # When targets are supplied we only need the loss, so logits stays None.
            token_chunk = loss_chunk_size if loss_chunk_size > 0 else 1024
            loss = fused_linear_cross_entropy(
                hidden.reshape(-1, hidden.size(-1)),
                self.lm_head.weight,
                targets.reshape(-1),
                ignore_index=-100,
                chunk_size=token_chunk,
            )
        else:
            logits = self.lm_head(hidden)
        return logits, loss, next_caches

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        if max_new_tokens <= 0:
            return input_ids
        if not input_ids.is_cuda:
            raise RuntimeError("MykLLM is CUDA-only and expects CUDA input tensors")
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (batch_size, seq_len)")
        if input_ids.size(1) == 0:
            raise ValueError("input_ids must contain at least one token for generation")

        prompt = input_ids[:, -self.config.max_seq_len :]
        prompt_positions = torch.arange(prompt.size(1), device=prompt.device)
        logits, _, kv_caches = self(prompt, position_ids=prompt_positions, use_cache=True)
        total_len = prompt.size(1)
        finished = torch.zeros(prompt.size(0), dtype=torch.bool, device=prompt.device)

        for _ in range(max_new_tokens):
            next_token_logits = logits[:, -1, :].float()

            if repetition_penalty != 1.0:
                # CTRL-style penalty: down-weight tokens already in the sequence so the
                # model can't lock onto a degenerate loop. Positive logits are divided,
                # negative ones multiplied, so the sign (relative order) is preserved.
                seen_scores = next_token_logits.gather(1, prompt)
                seen_scores = torch.where(
                    seen_scores < 0, seen_scores * repetition_penalty, seen_scores / repetition_penalty
                )
                next_token_logits.scatter_(1, prompt, seen_scores)

            if temperature <= 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                if top_k is not None:
                    top_k = min(top_k, next_token_logits.size(-1))
                    values, _ = torch.topk(next_token_logits, k=top_k, dim=-1)
                    cutoff = values[:, [-1]]
                    next_token_logits = next_token_logits.masked_fill(next_token_logits < cutoff, float("-inf"))
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None and bool(finished.any()):
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )

            prompt = torch.cat([prompt, next_token], dim=1)
            total_len += 1
            if total_len > self.config.max_seq_len:
                raise ValueError("cache-aware generation currently requires prompt + max_new_tokens <= max_seq_len")

            if eos_token_id is not None:
                finished = finished | next_token.squeeze(1).eq(eos_token_id)
                if bool(finished.all()):
                    break

            decode_positions = torch.full(
                (prompt.size(0), 1),
                total_len - 1,
                device=prompt.device,
                dtype=torch.long,
            )
            logits, _, kv_caches = self(
                next_token,
                position_ids=decode_positions,
                kv_caches=kv_caches,
                use_cache=True,
            )

        return prompt


    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def print_linear_layer_shapes(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight
                print(
                    f"{name}: in_features={module.in_features} out_features={module.out_features} "
                    f"weight_shape={tuple(weight.shape)}"
                )