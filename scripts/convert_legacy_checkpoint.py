"""Convert a pre-0.2 training checkpoint (`myk-llm/training.py` .pt) to the portable
`model.safetensors` + `model.yaml` format.

    python scripts/convert_legacy_checkpoint.py checkpoints/200M-v5-sft/last.pt out/200M-v5-sft
"""

from __future__ import annotations

import argparse
import re

import torch

from myk_llm.model import AttentionConfig, FFNConfig, ModelConfig, save_model


def convert_config(old: dict) -> ModelConfig:
    if old.get("attn_types"):
        codes = {"full_attention": "G", "sliding_window_attention": "L"}
        pattern = "".join(codes[t] for t in old["attn_types"])
    else:
        period = old.get("full_attn_period", 4)
        pattern = "G" + "L" * (period - 1)
    if old.get("dropout", 0.0):
        raise ValueError("dropout is no longer supported")
    return ModelConfig(
        vocab_size=old["vocab_size"],
        hidden_size=old["num_attention_heads"] * old["head_dim"],
        num_layers=old["num_layers"],
        layer_pattern=pattern,
        max_seq_len=old["max_seq_len"],
        tie_embeddings=old.get("tie_embeddings", True),
        initializer_range=old.get("initializer_range", 0.02),
        attention=AttentionConfig(
            num_heads=old["num_attention_heads"],
            num_kv_heads=old["num_kv_heads"],
            head_dim=old["head_dim"],
            sliding_window=old["sliding_window_size"],
            rope_theta_local=old.get("sliding_window_rope_theta", 10_000.0),
            rope_theta_global=old.get("full_attention_rope_theta", 1_000_000.0),
            # Old rule: RoPE on layer 0 and on sliding-window layers only.
            global_rope="first" if pattern[0] == "G" else "none",
        ),
        ffn=FFNConfig(multiplier=old.get("ffn_multiplier", 4.0), multiple_of=old.get("ffn_multiple_of", 256)),
    )


RENAMES = [
    (r"^token_embedding\.weight$", "embed.token.weight"),
    (r"^layers\.(\d+)\.attn_norm\.", r"layers.\1.mixer_norm."),
    (r"^layers\.(\d+)\.attn\.", r"layers.\1.mixer."),
]


def convert_state_dict(old: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    new = {}
    for name, tensor in old.items():
        for pattern, replacement in RENAMES:
            name = re.sub(pattern, replacement, name)
        new[name] = tensor
    return new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("output_dir")
    parser.add_argument("--tokenizer", default=None, help="defaults to the tokenizer recorded in the checkpoint")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = convert_config(checkpoint["model_config"])
    tokenizer = args.tokenizer or checkpoint.get("training_args", {}).get("tokenizer")
    save_model(args.output_dir, config, convert_state_dict(checkpoint["model_state_dict"]), tokenizer)
    print(f"wrote {args.output_dir} ({config.num_layers} layers, pattern {config.layer_pattern})")


if __name__ == "__main__":
    main()
