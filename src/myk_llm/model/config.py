"""Model hyper-parameters. Every architecture experiment is a field here, so a YAML
config fully describes a model and checkpoints are self-describing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Layer-pattern codes (one character per layer, repeated cyclically to num_layers):
GLOBAL_ATTENTION = "G"  # full causal softmax attention
LOCAL_ATTENTION = "L"  # sliding-window softmax attention
DELTANET = "D"  # Gated DeltaNet linear attention (needs flash-linear-attention)
CROSS_ATTENTION = "X"  # cross-decoder: causal attention over the shared KV (see CrossDecoderConfig)
LAYER_CODES = (GLOBAL_ATTENTION, LOCAL_ATTENTION, DELTANET, CROSS_ATTENTION)


@dataclass
class AttentionConfig:
    num_heads: int = 8
    num_kv_heads: int = 2  # grouped-query attention (GQA) when < num_heads
    head_dim: int = 128
    sliding_window: int = 512  # for "L" layers
    rope_theta_local: float = 10_000.0
    rope_theta_global: float = 1_000_000.0
    # Which global ("G") layers get rotary position embedding (RoPE); the rest run
    # without positional encoding (NoPE). Local layers always use RoPE.
    #   first: only the first global layer (the original MykLLM design)
    global_rope: Literal["first", "all", "none"] = "first"
    qk_norm: bool = False  # RMSNorm on per-head queries/keys before RoPE
    # Elementwise sigmoid gate on the attention output, computed from the layer input
    # ("gated attention", as in Qwen3-Next); curbs attention sinks / massive activations.
    output_gate: bool = False


@dataclass
class DeltaNetConfig:
    # "gdn": Gated DeltaNet (scalar decay per head); "kda": Kimi Delta Attention (Kimi
    # Linear; per-channel decay, low-rank decay / output-gate projections, always gated).
    kind: Literal["gdn", "kda"] = "gdn"
    num_heads: int = 8
    head_dim: int = 128
    expand_v: float = 1.0
    conv_size: int = 4
    # Causal depthwise convolution on q/k/v before the delta rule. Its kernel sees
    # relative offsets directly, so it is also a source of positional information.
    use_short_conv: bool = True
    use_gate: bool = True  # output gate (gdn only; kda always has one)


@dataclass
class FFNConfig:
    multiplier: float = 4.0  # SwiGLU hidden = multiplier * hidden_size, rounded up
    multiple_of: int = 256


@dataclass
class ResidualConfig:
    """How a sublayer output is added to the residual stream: x + gate * f(norm(x)).

    none:    gate = 1 (standard pre-norm residual)
    scalar:  one learnable scalar per sublayer, initialized to `init`
    channel: one learnable value per channel (LayerScale), initialized to `init`
    input:   gate = 2 * sigmoid(W norm(x)), per channel and per token; W starts at 0,
             so the gate starts at exactly 1 and the model starts as the baseline.
    """

    gate: Literal["none", "scalar", "channel", "input"] = "none"
    init: float = 1.0


@dataclass
class NgramEmbeddingConfig:
    """Hashed n-gram embeddings added to the residual stream.

    For each order n, the n tokens ending at position i are hashed into one of
    `num_buckets` rows of a dedicated table (input side only; the output vocab is
    unchanged). The tables are looked up once; each layer listed in `layers` gets its
    own projection dim -> hidden_size, added to that block's input (layer 0 = on top of
    the token embedding; a middle layer lets the model combine the n-gram lookup with
    context it has already built). With the single layer 0 and dim == hidden_size
    there is no projection.
    """

    orders: list[int] = field(default_factory=list)  # e.g. [2, 3]; empty disables
    num_buckets: int = 131_072
    dim: int = 256
    layers: list[int] = field(default_factory=lambda: [0])  # block indices to inject into


@dataclass
class MTPConfig:
    """Multi-token prediction (MTP), DeepSeek-V3 style: `depth` sequential modules,
    module k predicts token t+1+k from the previous hidden state and the embedding
    of token t+k, sharing the embedding and output head with the main model."""

    depth: int = 0
    layer: str = GLOBAL_ATTENTION  # mixer type of each module's transformer block


@dataclass
class LoopConfig:
    """Looped (recurrent-depth) transformer: the `core` blocks after the first `prelude`
    blocks share their weights across `iterations` passes, then the remaining blocks
    (the coda) run once. Layer types still come from `layer_pattern` over the
    num_layers distinct blocks.

    The loop state starts at zero; every iteration first injects the core's input (the
    stream after the prelude) into it:
      add:    state + input (Yang et al., 2024; with one iteration this is exactly the
              unlooped model)
      concat: Linear([norm(state), norm(input)]) (Geiping et al., 2025)

    Training can draw the iteration count per step, uniformly from `train_iterations`,
    and backpropagate through only the last `backprop_iterations` of them (truncated
    backpropagation through time (BPTT)). Evaluation uses `iterations` unless forward
    is given a count (test-time depth).
    """

    core: int = 0  # blocks in the shared core; 0 disables looping
    prelude: int = 0
    iterations: int = 1  # default count; the residual init is scaled for this depth
    train_iterations: list[int] = field(default_factory=list)  # [min, max]; empty: always `iterations`
    backprop_iterations: int = 0  # 0: through every iteration
    injection: Literal["add", "concat"] = "add"


@dataclass
class CrossDecoderConfig:
    """Decoder-decoder (YOCO, "You Only Cache Once", Sun et al., 2024). The trailing "X"
    blocks of `layer_pattern` (written out in full, e.g. "LLLLXXXX") form the
    cross-decoder; the blocks before them, the self-decoder. One set of keys / values
    (the shared KV) is computed once from the self-decoder; every cross-decoder block
    attends to it causally with its own queries and has no keys / values of its own.

      project: K, V = Linear(norm(self-decoder output)) (YOCO)
      layer:   the K, V of the self-decoder's last global ("G") layer, i.e. the second half
               reuses that layer's attention cache

    So the cache holds one global KV plus the self-decoder's states, and a prefill only
    needs the cross-decoder on the positions whose outputs are used (`keep_last`).
    Queries get RoPE exactly when the shared keys do: under global_rope "all", or "first"
    when the shared KV is the first global attention of the stack.
    """

    kv: Literal["project", "layer"] = "project"


@dataclass
class ModelConfig:
    vocab_size: int = 65536
    hidden_size: int = 1024
    num_layers: int = 9
    layer_pattern: str = "GL"
    max_seq_len: int = 4096
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    initializer_range: float = 0.02
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    deltanet: DeltaNetConfig = field(default_factory=DeltaNetConfig)
    ffn: FFNConfig = field(default_factory=FFNConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    ngram: NgramEmbeddingConfig = field(default_factory=NgramEmbeddingConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    cross: CrossDecoderConfig = field(default_factory=CrossDecoderConfig)

    def __post_init__(self) -> None:
        if not self.layer_pattern or set(self.layer_pattern) - set(LAYER_CODES):
            raise ValueError(f"layer_pattern must use only {LAYER_CODES}, got {self.layer_pattern!r}")
        attention = self.attention
        if attention.num_heads % attention.num_kv_heads:
            raise ValueError("attention.num_heads must be divisible by attention.num_kv_heads")
        if any(order < 2 for order in self.ngram.orders):
            raise ValueError("ngram.orders must be >= 2 (order 1 is the token embedding)")
        if self.ngram.orders:
            layers = self.ngram.layers
            if not layers or len(set(layers)) != len(layers) or not all(0 <= i < self.num_layers for i in layers):
                raise ValueError(f"ngram.layers must be distinct block indices in [0, {self.num_layers}), got {layers}")
        if self.mtp.layer not in LAYER_CODES or self.mtp.layer == CROSS_ATTENTION:
            raise ValueError(f"mtp.layer must be one of {LAYER_CODES[:-1]}")
        types, start = self.layer_types, self.cross_start
        if start < self.num_layers:
            if start == 0 or CROSS_ATTENTION in types[:start] or set(types[start:]) != {CROSS_ATTENTION}:
                raise ValueError(f"cross-decoder ('X') blocks must be the last blocks after at least one other, "
                                 f"got {''.join(types)} (write the pattern out in full, e.g. LLLLXXXX)")
            if self.loop.core:
                raise ValueError("a cross-decoder cannot be combined with a looped core")
            if self.cross.kv == "layer" and GLOBAL_ATTENTION not in types[:start]:
                raise ValueError("cross.kv=layer needs a global ('G') layer in the self-decoder")
        loop = self.loop
        if loop.core < 0 or loop.prelude < 0 or loop.prelude + loop.core > self.num_layers:
            raise ValueError(f"need loop.core, loop.prelude >= 0 and prelude + core <= num_layers ({self.num_layers})")
        if loop.core:
            if loop.iterations < 1 or loop.backprop_iterations < 0:
                raise ValueError("need loop.iterations >= 1 and loop.backprop_iterations >= 0")
            if loop.train_iterations and (len(loop.train_iterations) != 2 or not 1 <= loop.train_iterations[0] <= loop.train_iterations[1]):
                raise ValueError(f"loop.train_iterations must be [min, max] with 1 <= min <= max, got {loop.train_iterations}")
            # A core block after the first runs on the loop state, which has no single
            # "input" to add the update to; the first block's update joins the injected input.
            if self.ngram.orders and any(loop.prelude < i < loop.prelude + loop.core for i in self.ngram.layers):
                raise ValueError("ngram.layers cannot point inside the loop core (except at its first block)")

    @property
    def layer_types(self) -> list[str]:
        return [self.layer_pattern[i % len(self.layer_pattern)] for i in range(self.num_layers)]

    @property
    def cross_start(self) -> int:
        """Index of the first cross-decoder block (num_layers without one)."""
        types = self.layer_types
        return types.index(CROSS_ATTENTION) if CROSS_ATTENTION in types else self.num_layers

    @property
    def cross_kv_layer(self) -> int | None:
        """Block whose attention keys / values the cross-decoder reuses (cross.kv=layer)."""
        if self.cross_start == self.num_layers or self.cross.kv != "layer":
            return None
        return max(i for i, t in enumerate(self.layer_types[: self.cross_start]) if t == GLOBAL_ATTENTION)

    @property
    def cross_rope(self) -> bool:
        """Whether the shared keys (hence the cross-decoder queries) are rotated."""
        rope = self.attention.global_rope
        global_layers = [i for i, t in enumerate(self.layer_types) if t == GLOBAL_ATTENTION]
        source = self.cross_kv_layer
        first = (source == global_layers[0]) if source is not None else not global_layers
        return rope == "all" or (rope == "first" and first)

    @property
    def effective_layers(self) -> int:
        """Blocks run per forward pass at the default iteration count."""
        return self.num_layers + self.loop.core * (self.loop.iterations - 1)

    @property
    def layout(self) -> str:
        """Layer types with the loop marked, e.g. "D(GD)x4G": prelude D, core GD run 4 times, coda G."""
        types, loop = "".join(self.layer_types), self.loop
        if not loop.core:
            return types
        end = loop.prelude + loop.core
        return f"{types[: loop.prelude]}({types[loop.prelude : end]})x{loop.iterations}{types[end:]}"

    @property
    def ffn_hidden_size(self) -> int:
        raw = int(self.hidden_size * self.ffn.multiplier)
        return self.ffn.multiple_of * -(-raw // self.ffn.multiple_of)
