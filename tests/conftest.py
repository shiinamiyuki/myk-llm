import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def tiny_model_config(**overrides):
    from myk_llm.configuration import apply_overrides, from_dict, to_dict
    from myk_llm.model import ModelConfig

    base = {
        "vocab_size": 512, "hidden_size": 128, "num_layers": 4, "layer_pattern": "GL", "max_seq_len": 1024,
        "attention": {"num_heads": 2, "num_kv_heads": 1, "head_dim": 64, "sliding_window": 64},
        "deltanet": {"num_heads": 2, "head_dim": 64},
        "ffn": {"multiplier": 2.0, "multiple_of": 64},
    }
    tree = apply_overrides(base, [f"{k}={v}" for k, v in overrides.items()])
    return from_dict(ModelConfig, to_dict(tree))
