import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from conftest import requires_cuda, tiny_model_config
from myk_llm.model import MykLLM
from myk_llm.ops import MXFP8Linear
from myk_llm.train.distributed import convert_float8

pytestmark = requires_cuda
pytest.importorskip("torchao")

requires_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="Microscaling 8-bit Floating Point (MXFP8) needs NVIDIA Blackwell compute capability >= 10.0",
)


def build(**overrides) -> MykLLM:
    torch.manual_seed(0)
    return MykLLM(tiny_model_config(**overrides)).cuda()


def swapped_names(model: nn.Module) -> set[str]:
    return {name for name, m in model.named_modules() if type(m).__name__ in {"Float8Linear", "MXFP8Linear"}}


def test_conversion_converts_deltanet_projections_and_skips_head_and_gates():
    # Use 16 heads so a_proj / b_proj have out_features=16 (divisible by 16) and verify
    # recurrent decay / step-size gates are still explicitly kept in high precision.
    model = build(layer_pattern="GD", **{"residual.gate": "input", "deltanet.num_heads": "16", "deltanet.head_dim": "16"})
    keys = set(model.state_dict())
    count = convert_float8(model, "tensorwise")
    converted = swapped_names(model)
    assert count == len(converted) > 0
    assert "lm_head" not in converted
    assert not any(name.endswith((".a_proj", ".b_proj")) for name in converted)
    assert {
        "layers.0.mixer.qkv_proj",
        "layers.1.mixer.inner.q_proj",
        "layers.1.mixer.inner.o_proj",
        "layers.1.ffn.down_proj",
        "layers.0.mixer_gate.gate_proj",
    } <= converted
    assert model.lm_head.weight is model.embed.token.weight  # tying survives
    assert set(model.state_dict()) == keys  # checkpoints stay interchangeable with Brain Floating Point 16-bit (BF16) runs


def test_skip_down_proj_keeps_ffn_down_in_high_precision():
    model = build(layer_pattern="GD")
    count = convert_float8(model, "tensorwise", skip_down_proj=True)
    converted = swapped_names(model)
    assert count == len(converted) > 0
    assert not any(name.endswith(".down_proj") for name in converted)
    assert "layers.0.ffn.up_gate_proj" in converted


@pytest.mark.parametrize("pattern", ["GL", "GD"])
@pytest.mark.parametrize("recipe", ["tensorwise", "mxfp8"])
def test_float8_training_step_tracks_bf16(pattern, recipe):
    if recipe == "mxfp8" and torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("MXFP8 needs compute capability >= 10.0")
    from myk_llm.data import Batch
    from myk_llm.train.config import ObjectiveConfig
    from myk_llm.train.objectives import Objective

    torch.manual_seed(1)
    ids = torch.randint(0, 500, (4, 257), device="cuda")
    batch = Batch(ids[:, :-1], ids[:, 1:], [torch.tensor([100, 156])] * 4)
    objective = Objective(ObjectiveConfig(), torch.device("cuda"))
    results = []
    for mode in ("none", recipe):
        model = build(layer_pattern=pattern).train()
        if mode != "none":
            convert_float8(model, mode)
        model.compile_blocks()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch.input_ids, seqlens=batch.seqlens, head=objective.head(batch, float(batch.num_targets)))
        out.loss.backward()
        grad = model.layers[1].ffn.up_gate_proj.weight.grad
        assert torch.isfinite(out.loss) and torch.isfinite(grad).all()
        results.append((float(out.loss.detach()), grad.float()))
    (loss_ref, grad_ref), (loss_fp8, grad_fp8) = results
    assert abs(loss_fp8 - loss_ref) < 2e-3 * loss_ref
    assert F.cosine_similarity(grad_fp8.flatten(), grad_ref.flatten(), dim=0) > 0.98


@requires_blackwell
def test_mxfp8_handles_unaligned_lengths_and_extreme_outliers():
    torch.manual_seed(0)
    ref = nn.Linear(256, 128, bias=False, device="cuda", dtype=torch.float32)
    mx = MXFP8Linear.from_float(nn.Linear(256, 128, bias=False, device="cuda", dtype=torch.float32))
    mx.weight.data.copy_(ref.weight.data)
    compiled_mx = torch.compile(mx)

    # Unaligned row counts: greedy decode (M=1), short prompt (M=7), and odd training length (M=33).
    for rows, grad in ((1, False), (7, False), (33, True)):
        mx.zero_grad(set_to_none=True)
        x = torch.randn(1, rows, 256, device="cuda", requires_grad=grad)
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
            y = compiled_mx(x)
        assert y.shape == (1, rows, 128) and torch.isfinite(y).all()
        if grad:
            y.sum().backward()
            assert torch.isfinite(x.grad).all() and torch.isfinite(mx.weight.grad).all()

    # Single-token massive outlier (`2e5`) plus zero-grad masked rows: non-outlier tokens stay > 0.999 accurate.
    x = torch.randn(64, 256, device="cuda")
    x[0, 0] = 2e5
    dy = torch.randn(64, 128, device="cuda")
    dy[8:24] = 0.0

    x_ref = x.clone().requires_grad_(True)
    ref.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y_ref = ref(x_ref)
    y_ref.backward(dy)

    x_mx = x.clone().requires_grad_(True)
    mx.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y_mx = compiled_mx(x_mx)
    y_mx.backward(dy)

    assert F.cosine_similarity(y_mx[1:].float().flatten(), y_ref[1:].float().flatten(), dim=0) > 0.999
    assert F.cosine_similarity(x_mx.grad[1:].float().flatten(), x_ref.grad[1:].float().flatten(), dim=0) > 0.999
    assert F.cosine_similarity(mx.weight.grad[:, 1:].float().flatten(), ref.weight.grad[:, 1:].float().flatten(), dim=0) > 0.999
