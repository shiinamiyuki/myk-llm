"""Portable model directory: `model.safetensors` + `model.yaml` (+ the tokenizer files).

This is the format for evaluation, chat and warm starts. Training checkpoints
(sharded, with optimizer state) are separate; see `myk_llm.train.checkpoint`.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

from ..configuration import from_dict, to_dict
from .config import ModelConfig
from .transformer import MykLLM

WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "model.yaml"


def save_model(
    directory: str | Path, config: ModelConfig, state_dict: dict[str, torch.Tensor], tokenizer: str | None = None
) -> None:
    """Write weights and config; `tokenizer` (a path) is copied in so the directory is
    self-contained for evaluation and chat."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {name: t.detach().to("cpu").contiguous() for name, t in state_dict.items()}
    if config.tie_embeddings:
        tensors.pop("lm_head.weight", None)  # shared with embed.token.weight
    save_file(tensors, str(directory / WEIGHTS_FILE))
    (directory / CONFIG_FILE).write_text(yaml.safe_dump(to_dict(config), sort_keys=False), encoding="utf-8")
    if tokenizer:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(tokenizer).save_pretrained(str(directory))


def load_model_config(directory: str | Path) -> ModelConfig:
    tree = yaml.safe_load((Path(directory) / CONFIG_FILE).read_text(encoding="utf-8"))
    return from_dict(ModelConfig, tree)


def load_state_dict(directory: str | Path) -> dict[str, torch.Tensor]:
    return load_file(str(Path(directory) / WEIGHTS_FILE))


def load_model(
    directory: str | Path, device: str | torch.device = "cuda", max_seq_len: int | None = None
) -> MykLLM:
    config = load_model_config(directory)
    if max_seq_len is not None:
        config.max_seq_len = max_seq_len
    model = MykLLM(config).to(device)
    missing, unexpected = model.load_state_dict(load_state_dict(directory), strict=False)
    missing = [name for name in missing if not (config.tie_embeddings and name == "lm_head.weight")]
    if missing or unexpected:
        raise KeyError(f"weight mismatch loading {directory}: missing={missing} unexpected={unexpected}")
    return model.eval()
