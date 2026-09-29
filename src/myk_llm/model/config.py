"""Model hyper-parameters. Every architecture experiment is a field here, so a YAML
config fully describes a model and checkpoints are self-describing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Layer-pattern codes (one character per layer, repeated cyclically to num_layers):
GLOBAL_ATTENTION = "G"  # full causal softmax attention
LOCAL_ATTENTION = "L"  # sliding-window softmax attention
DELTANET = "D"  # Gated DeltaNet linear attention (needs flash-linear-attention)
LAYER_CODES = (GLOBAL_ATTENTION, LOCAL_ATTENTION, DELTANET)


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
    num_heads: int = 8
    head_dim: int = 128
    expand_v: float = 1.0
    conv_size: int = 4
    use_gate: bool = True


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
    """Hashed n-gram input embeddings added to the token embedding.

    For each order n, the n tokens ending at position i are hashed into one of
    `num_buckets` rows of a dedicated table (input side only; the output vocab is
    unchanged). `dim` < hidden_size stores a narrower table plus a projection.
    """

    orders: list[int] = field(default_factory=list)  # e.g. [2, 3]; empty disables
    num_buckets: int = 131_072
    dim: int = 256


@dataclass
class MTPConfig:
    """Multi-token prediction (MTP), DeepSeek-V3 style: `depth` sequential modules,
    module k predicts token t+1+k from the previous hidden state and the embedding
    of token t+k, sharing the embedding and output head with the main model."""

    depth: int = 0
    layer: str = GLOBAL_ATTENTION  # mixer type of each module's transformer block


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

    def __post_init__(self) -> None:
        if not self.layer_pattern or set(self.layer_pattern) - set(LAYER_CODES):
            raise ValueError(f"layer_pattern must use only {LAYER_CODES}, got {self.layer_pattern!r}")
        attention = self.attention
        if attention.num_heads % attention.num_kv_heads:
            raise ValueError("attention.num_heads must be divisible by attention.num_kv_heads")
        if any(order < 2 for order in self.ngram.orders):
            raise ValueError("ngram.orders must be >= 2 (order 1 is the token embedding)")
        if self.mtp.layer not in LAYER_CODES:
            raise ValueError(f"mtp.layer must be one of {LAYER_CODES}")

    @property
    def layer_types(self) -> list[str]:
        return [self.layer_pattern[i % len(self.layer_pattern)] for i in range(self.num_layers)]

    @property
    def ffn_hidden_size(self) -> int:
        raw = int(self.hidden_size * self.ffn.multiplier)
        return self.ffn.multiple_of * -(-raw // self.ffn.multiple_of)
