"""Training run configuration (one YAML file per run; see configs/)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..data.config import DataConfig, SourceConfig
from ..model.config import ModelConfig


@dataclass
class OptimizerConfig:
    name: Literal["muon", "adamw"] = "muon"
    lr: float = 3e-4  # AdamW parameters (embeddings, norms; everything when name=adamw)
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-8
    weight_decay: float = 0.1
    muon_lr: float | None = None  # defaults to lr
    muon_weight_decay: float | None = None  # defaults to weight_decay
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_adjust_lr: Literal["original", "match_rms_adamw", "none"] = "match_rms_adamw"
    grad_clip: float = 1.0  # max global gradient norm; 0 disables


@dataclass
class ScheduleConfig:
    kind: Literal["cosine", "wsd", "constant"] = "cosine"
    warmup_ratio: float = 0.01
    min_lr_ratio: float = 0.1
    decay_ratio: float = 0.2  # wsd: fraction of steps spent decaying at the end


@dataclass
class DistillConfig:
    """Logit distillation from a HuggingFace teacher sharing the tokenizer family."""

    teacher: str = ""  # HF model dir or id; empty disables distillation
    row_map: str = ""  # student id -> teacher id (embedding_row_map.npy of a pruned tokenizer)
    weight: float = 0.5  # loss = (1 - weight) * CE + weight * KL
    temperature: float = 1.0
    chunk_size: int = 1024


@dataclass
class ObjectiveConfig:
    z_loss: float = 0.0  # coefficient on logsumexp^2 (keeps output logits from drifting)
    mtp_weight: float = 0.3  # weight of each multi-token-prediction depth (model.mtp.depth > 0)
    loss_chunk_size: int = 2048
    distill: DistillConfig = field(default_factory=DistillConfig)


@dataclass
class InitConfig:
    # Warm-start the token embedding from a HuggingFace checkpoint (e.g. Qwen), gathering
    # the rows of a pruned vocabulary through `embedding_row_map`.
    embedding_from: str = ""
    embedding_row_map: str = ""
    freeze_embedding: bool = False
    # Warm-start all weights from an exported model directory (model.safetensors).
    weights_from: str = ""
    # Warm-start from a training checkpoint directory (fresh step counter and schedule),
    # optionally carrying the optimizer state (e.g. for a WSD decay/anneal run).
    checkpoint_from: str = ""
    load_optimizer: bool = False


@dataclass
class ParallelConfig:
    strategy: Literal["auto", "fsdp", "none"] = "auto"  # auto: fsdp when world_size > 1
    # fp32 keeps fp32 master weights and runs matmuls in bf16 via autocast (the residual
    # stream stays fp32). bf16 casts parameters for the forward/backward pass (FSDP
    # mixed precision): less memory and communication, slightly less precise.
    param_dtype: Literal["fp32", "bf16"] = "fp32"
    reshard_after_forward: bool = True  # False trades memory for less all-gather traffic
    compile: bool = True
    activation_checkpointing: bool = False
    # FP8 (8-bit floating point) matmuls in the block linears via torchao's Float8Linear,
    # with dynamic scaling; master weights, optimizer and output head are unchanged.
    # A torchao recipe name; "tensorwise" is the fast one (see notes/fp8.md). The
    # rowwise recipes fail under autocast with fp32 parameters in torchao 0.9.
    float8: Literal["none", "tensorwise", "rowwise", "rowwise_with_gw_hp"] = "none"


@dataclass
class CheckpointConfig:
    interval: int = 500  # steps; 0 disables periodic checkpoints (the final one is still saved)
    keep_last: int = 3  # rolling checkpoints to keep (0 keeps all)
    keep_every: int = 0  # additionally keep every N-th step forever (0 disables)
    export_weights: bool = True  # also write model.safetensors (for eval/chat) with each checkpoint


@dataclass
class ValSetConfig:
    name: str
    sources: list[SourceConfig]
    kind: Literal["shards", "stream", "sft"] | None = None  # defaults to data.kind


@dataclass
class EvalConfig:
    interval: int = 0  # steps between validation passes; 0 disables
    # Batches of data.micro_batch_size rows per validation set: the set grows or shrinks
    # with the micro-batch size, so keep batches x micro_batch_size fixed across runs.
    batches: int = 8
    seed: int = 999_983
    val_sets: list[ValSetConfig] = field(default_factory=list)
    sample_interval: int = 0  # steps between text samples; 0 disables
    sample_prompts: list[str] = field(default_factory=lambda: ["The meaning of life is"])
    sample_chat: bool = False  # wrap prompts in the chat template
    sample_tokens: int = 64


@dataclass
class LoggingConfig:
    interval: int = 10
    tensorboard: bool = True


@dataclass
class TrainConfig:
    run_name: str = "debug"
    output_dir: str = "runs"
    steps: int = 1000
    # Sequences per optimizer step across all GPUs; gradient accumulation is derived as
    # global_batch_size / (data.micro_batch_size * world_size), so a config keeps its
    # meaning when moved to a different number of GPUs.
    global_batch_size: int = 32
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    init: InitConfig = field(default_factory=InitConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
