"""The MykLLM decoder: embeddings -> blocks -> final norm, plus optional MTP modules.

With `loop.core` set, the blocks are prelude -> core (looped, shared weights) -> coda.
With trailing "X" blocks (decoder-decoder, YOCO) they are self-decoder -> shared KV ->
cross-decoder.

The model returns hidden states, not a loss. Objectives (cross entropy, z-loss,
multi-token prediction, distillation) live in `myk_llm.train.objectives` and use
`output_weight` directly, so the full (tokens, vocab) logits never need to exist.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import GLOBAL_ATTENTION, ModelConfig
from .embeddings import Embeddings
from .layers import ResidualGate, RMSNorm, SwiGLU, init_linear_, residual_init_scale
from .mixers import Attention, SharedKV, Varlen, build_mixer


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
            # `state` is None here except for a cross-decoder block (the shared KV).
            return checkpoint(self._forward, x, position_ids, varlen, state, False, use_reentrant=False)
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


class Injection(nn.Module):
    """Starts each loop iteration by combining the loop state with the core's input."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.kind = config.loop.injection
        if self.kind == "concat":
            self.state_norm = RMSNorm(config.hidden_size, config.norm_eps)
            self.input_norm = RMSNorm(config.hidden_size, config.norm_eps)
            self.proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

    def forward(self, state: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        if self.kind == "add":
            return state + inputs
        return self.proj(torch.cat([self.state_norm(state), self.input_norm(inputs)], dim=-1))


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
        self.injection = Injection(config) if config.loop.core else None
        self.cross_start, self.cross_kv_layer = config.cross_start, config.cross_kv_layer
        has_cross = self.cross_start < len(self.layers)
        self.shared_kv = SharedKV(config) if has_cross and config.cross.kv == "project" else None
        if self.cross_kv_layer is not None:
            source = self.layers[self.cross_kv_layer].mixer
            assert isinstance(source, Attention)
            source.shares_kv = True
        # Training draws iteration counts from its own generator, seeded like the model
        # init, so that every FSDP rank runs the same blocks in the same order.
        self._iteration_rng = random.Random(torch.initial_seed())
        self.init_weights()

    # -- initialization ---------------------------------------------------------------
    @torch.no_grad()
    def init_weights(self) -> None:
        std = self.config.initializer_range
        self.apply(lambda module: init_linear_(module, std))
        scale = residual_init_scale(self.config.effective_layers)
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
        # All blocks share Block.forward, so every (layer type, grad mode, shape) variant
        # is one more compiled entry of the same code; past the limit, blocks run eagerly.
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 16)
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
        embed_offset: torch.Tensor | None = None,
        iterations: int | None = None,
        keep_last: int = 0,
    ) -> Any:
        """`seqlens` enables document masking (training); `cache`/`use_cache` drive
        incremental decoding; `prefix_ids` are the tokens preceding `input_ids`
        (needed by n-gram embeddings when decoding with a cache).

        `embed_offset` (batch, seq_len, hidden) is added to the input embeddings (e.g. a
        task-specific positional signal computed outside the model).

        `iterations` sets the core iteration count of a looped model (test-time depth;
        default: see `num_iterations`). Every block call keeps its own cache entry, so a
        cache only fits passes with the iteration count it was built with.

        `keep_last` > 0: only the outputs of the last `keep_last` positions are needed
        (e.g. a prefill), so the cross-decoder of a decoder-decoder model and the final norm
        run on those positions only; `hidden`, `residual` and `position_ids` of the output
        then cover just them. The cache is complete either way.

        Returns a ModelOutput, or `head(self, output)` when `head` is given. Losses run
        as a head so that they execute inside forward: under FSDP (fully sharded data
        parallel) the root parameters (embedding / output head) are only gathered there.
        """
        batch, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).expand(batch, -1)
        varlen = None
        if seqlens is not None:
            if use_cache or keep_last:
                raise ValueError("document masking (seqlens) cannot be combined with a cache or keep_last")
            varlen = build_varlen(seqlens, batch, seq_len, input_ids.device)
        x, ngram_updates = self.embed(input_ids, prefix_ids)
        if embed_offset is not None:
            x = x + embed_offset
        loop = self.config.loop
        iterations = self.num_iterations(iterations)
        # One cache entry per block call, plus the shared KV (decoder-decoder, project).
        num_calls = len(self.layers) + loop.core * (iterations - 1) + (self.shared_kv is not None)
        if cache is not None and len(cache) != num_calls:
            raise ValueError(f"cache has {len(cache)} entries but this pass runs {num_calls} blocks (iteration count changed?)")
        states = iter(cache if cache is not None else [None] * num_calls)
        new_cache: list[Any] | None = [] if use_cache else None

        shared: Any = None  # the cross-decoder's keys / values

        def run(index: int, x: torch.Tensor) -> torch.Tensor:
            nonlocal shared
            x, state = self.layers[index](x, position_ids, varlen, next(states), use_cache)
            if new_cache is not None:
                new_cache.append(state)
            if index == self.cross_kv_layer:
                shared = state
            return x

        def enter(index: int, x: torch.Tensor) -> torch.Tensor:
            return x + ngram_updates[index][:, -x.size(1) :] if index in ngram_updates else x

        core_start = loop.prelude if loop.core else self.cross_start
        for index in range(core_start):
            x = run(index, enter(index, x))
        if loop.core:
            x = self._run_core(enter(core_start, x), run, iterations)
        for index in range(core_start + loop.core, self.cross_start):
            x = run(index, enter(index, x))
        self_output, self_positions = x, position_ids
        if keep_last:
            x, position_ids = x[:, -keep_last:], position_ids[:, -keep_last:]
        if self.cross_start < len(self.layers):
            if self.shared_kv is not None:
                # From the self-decoder output at every position, not only the kept ones.
                shared = self.shared_kv(self_output, self_positions, next(states))
                if new_cache is not None:
                    new_cache.append(shared)
            for index in range(self.cross_start, len(self.layers)):
                x, _ = self.layers[index](enter(index, x), position_ids, varlen, shared)
                next(states)
                if new_cache is not None:
                    new_cache.append(None)
        output = ModelOutput(self.final_norm(x), x, position_ids, varlen, new_cache)
        return head(self, output) if head is not None else output

    def num_iterations(self, iterations: int | None = None) -> int:
        """Core iterations of one pass: `iterations` if given, else a draw from
        loop.train_iterations in training mode, else loop.iterations."""
        loop = self.config.loop
        if not loop.core:
            if iterations is not None:
                raise ValueError("iterations needs a looped model (model.loop.core > 0)")
            return 1
        if iterations is not None:
            if iterations < 1:
                raise ValueError(f"iterations must be >= 1, got {iterations}")
            return iterations
        if self.training and loop.train_iterations:
            return self._iteration_rng.randint(*loop.train_iterations)
        return loop.iterations

    def _run_core(self, inputs: torch.Tensor, run: Callable[[int, torch.Tensor], torch.Tensor], iterations: int) -> torch.Tensor:
        """The core blocks, `iterations` times over a loop state that starts at zero and
        takes in `inputs` at the start of every iteration. With truncated backpropagation
        the earlier iterations run without autograd (gradients still reach `inputs`
        through the later ones)."""
        loop, injection = self.config.loop, self.injection
        assert injection is not None
        grad = torch.is_grad_enabled()
        first_tracked = iterations - loop.backprop_iterations if grad and loop.backprop_iterations else 0
        state = torch.zeros_like(inputs)
        for iteration in range(iterations):
            with torch.set_grad_enabled(grad and iteration >= first_tracked):
                state = injection(state, inputs)
                for index in range(loop.prelude, loop.prelude + loop.core):
                    state = run(index, state)
        return state

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
