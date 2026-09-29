"""The MykLLM decoder: embeddings -> blocks -> final norm, plus optional MTP modules.

The model returns hidden states, not a loss. Objectives (cross entropy, z-loss,
multi-token prediction, distillation) live in `myk_llm.train.objectives` and use
`output_weight` directly, so the full (tokens, vocab) logits never need to exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import GLOBAL_ATTENTION, ModelConfig
from .embeddings import Embeddings
from .layers import ResidualGate, RMSNorm, SwiGLU, init_linear_, residual_init_scale
from .mixers import Varlen, build_mixer


class Block(nn.Module):
    """Pre-norm block: x + g1 * mixer(norm(x)), then x + g2 * ffn(norm(x))."""

    def __init__(self, config: ModelConfig, layer_type: str, first_global: bool) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.mixer_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.mixer = build_mixer(config, layer_type, first_global)
        self.mixer_gate = ResidualGate(config.residual, config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn = SwiGLU(config)
        self.ffn_gate = ResidualGate(config.residual, config.hidden_size)
        self.activation_checkpointing = False  # recompute this block in backward

    def residual_projections(self) -> list[nn.Linear]:
        return [*self.mixer.residual_projections(), self.ffn.down_proj]

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        varlen: Varlen | None = None,
        state: Any = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Any]:
        if self.activation_checkpointing and torch.is_grad_enabled() and not use_cache:
            return checkpoint(self._forward, x, position_ids, varlen, None, False, use_reentrant=False)
        return self._forward(x, position_ids, varlen, state, use_cache)

    def _forward(
        self, x: torch.Tensor, position_ids: torch.Tensor, varlen: Varlen | None, state: Any, use_cache: bool
    ) -> tuple[torch.Tensor, Any]:
        h = self.mixer_norm(x)
        update, state = self.mixer(h, position_ids, varlen, state, use_cache)
        x = self.mixer_gate(x, update, h)
        h = self.ffn_norm(x)
        x = self.ffn_gate(x, self.ffn(h), h)
        return x, state


class MTPModule(nn.Module):
    """One multi-token-prediction depth: combine the previous depth's hidden state with
    the embedding of the next known token, then run one transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.embed_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.block = Block(config, config.mtp.layer, first_global=True)
        self.final_norm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(
        self, hidden: torch.Tensor, next_embed: torch.Tensor, position_ids: torch.Tensor, varlen: Varlen | None
    ) -> torch.Tensor:
        x = self.proj(torch.cat([self.hidden_norm(hidden), self.embed_norm(next_embed)], dim=-1))
        x, _ = self.block(x, position_ids, varlen)
        return x


@dataclass
class ModelOutput:
    hidden: torch.Tensor  # final-normed hidden states, ready for the output head
    residual: torch.Tensor  # pre-norm residual stream (input to MTP modules)
    position_ids: torch.Tensor
    varlen: Varlen | None
    cache: list[Any] | None = None


def build_varlen(seqlens: list[torch.Tensor], batch: int, seq_len: int, device: torch.device) -> Varlen:
    """Block-diagonal layout from per-sample document lengths (each sums to seq_len)."""
    lengths = torch.cat([s.to(device=device, dtype=torch.int32) for s in seqlens])
    cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
    # A constant upper bound (not the data-dependent max) avoids recompiles.
    return Varlen(cu_seqlens=cu_seqlens, max_seqlen=seq_len)


class MykLLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = Embeddings(config)
        global_layers = [i for i, t in enumerate(config.layer_types) if t == GLOBAL_ATTENTION]
        first_global = global_layers[0] if global_layers else -1
        self.layers = nn.ModuleList(
            Block(config, layer_type, first_global=(i == first_global))
            for i, layer_type in enumerate(config.layer_types)
        )
        self.final_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed.token.weight
        self.mtp = nn.ModuleList(MTPModule(config) for _ in range(config.mtp.depth))
        self.init_weights()

    # -- initialization ---------------------------------------------------------------
    @torch.no_grad()
    def init_weights(self) -> None:
        std = self.config.initializer_range
        self.apply(lambda module: init_linear_(module, std))
        scale = residual_init_scale(self.config.num_layers)
        for block in self.blocks:
            for proj in block.residual_projections():
                proj.weight.mul_(scale)
            block.mixer_gate.init_weights()
            block.ffn_gate.init_weights()

    # -- introspection ----------------------------------------------------------------
    @property
    def output_weight(self) -> torch.Tensor:
        return self.lm_head.weight

    @property
    def device(self) -> torch.device:
        return self.embed.token.weight.device

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        params = {id(p): p for p in self.parameters()}
        if exclude_embeddings:  # token and n-gram tables, output head
            for p in [*self.embed.parameters(), self.lm_head.weight]:
                params.pop(id(p), None)
        return sum(p.numel() for p in params.values())

    @property
    def blocks(self) -> list[Block]:
        """All transformer blocks, including those inside MTP modules."""
        return [*self.layers, *(m.block for m in self.mtp)]

    def compile_blocks(self, **kwargs: Any) -> None:
        """torch.compile each block in place (parameter names are unchanged)."""
        for block in self.blocks:
            block.compile(**kwargs)

    def set_activation_checkpointing(self, enabled: bool) -> None:
        for block in self.blocks:
            block.activation_checkpointing = enabled

    # -- forward ----------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        seqlens: list[torch.Tensor] | None = None,
        cache: list[Any] | None = None,
        use_cache: bool = False,
        prefix_ids: torch.Tensor | None = None,
        head: Callable[["MykLLM", ModelOutput], Any] | None = None,
    ) -> Any:
        """`seqlens` enables document masking (training); `cache`/`use_cache` drive
        incremental decoding; `prefix_ids` are the tokens preceding `input_ids`
        (needed by n-gram embeddings when decoding with a cache).

        Returns a ModelOutput, or `head(self, output)` when `head` is given. Losses run
        as a head so that they execute inside forward: under FSDP (fully sharded data
        parallel) the root parameters (embedding / output head) are only gathered there.
        """
        batch, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).expand(batch, -1)
        varlen = None
        if seqlens is not None:
            if use_cache:
                raise ValueError("document masking (seqlens) cannot be combined with a cache")
            varlen = build_varlen(seqlens, batch, seq_len, input_ids.device)
        x = self.embed(input_ids, prefix_ids)
        states = cache if cache is not None else [None] * len(self.layers)
        new_cache: list[Any] | None = [] if use_cache else None
        for block, state in zip(self.layers, states):
            x, state = block(x, position_ids, varlen, state, use_cache)
            if new_cache is not None:
                new_cache.append(state)
        output = ModelOutput(self.final_norm(x), x, position_ids, varlen, new_cache)
        return head(self, output) if head is not None else output

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)

    def mtp_hidden(self, output: ModelOutput, input_ids: torch.Tensor) -> list[torch.Tensor]:
        """Final-normed hidden states of each MTP depth; depth k (1-based) at position i
        predicts token i + 1 + k. The last k positions see a padded (masked) token."""
        hidden, results = output.residual, []
        for k, module in enumerate(self.mtp, start=1):
            shifted = F.pad(input_ids[:, k:], (0, k), value=0)
            hidden = module(hidden, self.embed.token(shifted), output.position_ids, output.varlen)
            results.append(module.final_norm(hidden))
        return results
