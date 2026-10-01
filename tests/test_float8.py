import pytest
import torch
import torch.nn as nn

from conftest import requires_cuda, tiny_model_config
from myk_llm.model import MykLLM
from myk_llm.train.distributed import convert_float8

pytestmark = requires_cuda
pytest.importorskip("torchao")


def build(**overrides) -> MykLLM:
    torch.manual_seed(0)
    return MykLLM(tiny_model_config(**overrides)).cuda()


def float8_names(model: nn.Module) -> set[str]:
    return {name for name, m in model.named_modules() if type(m).__name__ == "Float8Linear"}


def test_conversion_skips_head_deltanet_and_unaligned_linears():
    model = build(layer_pattern="GD", **{"residual.gate": "input"})
    keys = set(model.state_dict())
    count = convert_float8(model, "tensorwise")
    converted = float8_names(model)
    assert count == len(converted) > 0
    assert "lm_head" not in converted
    assert not any(".inner." in name for name in converted)  # DeltaNet runs eagerly
    assert {"layers.0.mixer.qkv_proj", "layers.1.ffn.down_proj", "layers.0.mixer_gate.gate_proj"} <= converted
    assert model.lm_head.weight is model.embed.token.weight  # tying survives
    assert set(model.state_dict()) == keys  # checkpoints stay interchangeable with bf16 runs


def test_float8_training_step_tracks_bf16():
    from myk_llm.data import Batch
    from myk_llm.train.config import ObjectiveConfig
    from myk_llm.train.objectives import Objective

    torch.manual_seed(1)
    ids = torch.randint(0, 500, (4, 257), device="cuda")
    batch = Batch(ids[:, :-1], ids[:, 1:], [torch.tensor([100, 156])] * 4)
    objective = Objective(ObjectiveConfig(), torch.device("cuda"))
    results = []
    for recipe in ("none", "tensorwise"):
        model = build().train()
        if recipe != "none":
            convert_float8(model, recipe)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch.input_ids, seqlens=batch.seqlens, head=objective.head(batch, float(batch.num_targets)))
        out.loss.backward()
        grad = model.layers[0].ffn.up_gate_proj.weight.grad
        assert torch.isfinite(out.loss) and torch.isfinite(grad).all()
        results.append((float(out.loss.detach()), grad.float()))
    (loss_ref, grad_ref), (loss_fp8, grad_fp8) = results
    assert abs(loss_fp8 - loss_ref) < 2e-3 * loss_ref
    assert torch.nn.functional.cosine_similarity(grad_fp8.flatten(), grad_ref.flatten(), dim=0) > 0.98
