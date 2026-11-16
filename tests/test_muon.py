import pytest
import torch
import torch.nn as nn

from conftest import requires_cuda
from myk_llm.ops import batched_newton_schulz_orthogonalize, batched_syrk
from myk_llm.train.optim import Muon, newton_schulz_orthogonalize

pytestmark = requires_cuda


@pytest.mark.parametrize(("m", "k", "transposed"), [(100, 150, False), (256, 512, False), (192, 320, True)])
def test_batched_syrk_matches_bmm_and_baddbmm(m: int, k: int, transposed: bool):
    """Batched Symmetric Rank-K Update (SYRK) matches full Brain Floating Point 16-bit (BF16)
    batched matmul on aligned, unaligned, and transposed-view inputs."""
    torch.manual_seed(0)
    raw = torch.randn((5, k, m) if transposed else (5, m, k), device="cuda", dtype=torch.bfloat16)
    x = raw.transpose(-2, -1) if transposed else raw
    expected = torch.bmm(x, x.transpose(-2, -1))
    actual = batched_syrk(x)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    bias = (expected + expected.transpose(-2, -1)) * 0.5
    expected_biased = torch.baddbmm(bias, expected, expected, beta=-4.775, alpha=2.0315)
    actual_biased = batched_syrk(expected, bias=bias, alpha=2.0315, beta=-4.775)
    torch.testing.assert_close(actual_biased, expected_biased, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("nesterov", [True, False])
def test_cuda_muon_matches_reference_across_mixed_shapes(dtype: torch.dtype, nesterov: bool):
    """Fused multi-tensor CUDA Muon tracks per-matrix reference updates across narrow
    (`M < 64`), square, wide (`rows < cols`), and tall (`rows > cols`) matrices."""
    torch.manual_seed(7)
    shapes = [(8, 256), (8, 256), (128, 128), (128, 128), (128, 256), (384, 128), (384, 128)]
    params_fast = [nn.Parameter(torch.randn(s, device="cuda", dtype=dtype) * 0.02) for s in shapes]
    params_ref = [nn.Parameter(p.detach().clone()) for p in params_fast]

    lr, wd, mom, ns_steps = 1e-3, 0.1, 0.95, 5
    opt = Muon(params_fast, lr=lr, weight_decay=wd, momentum=mom, nesterov=nesterov, ns_steps=ns_steps)
    bufs_ref = [torch.zeros_like(p) for p in params_ref]

    for step in range(3):
        torch.manual_seed(100 + step)
        for pf, pr in zip(params_fast, params_ref):
            g = (torch.randn_like(pf) * 0.05).to(dtype)
            pf.grad = g.clone()
            pr.grad = g.clone()
        opt.step()
        with torch.no_grad():
            for pr, buf in zip(params_ref, bufs_ref):
                assert pr.grad is not None
                buf.lerp_(pr.grad, 1.0 - mom)
                u = pr.grad.lerp(buf, mom) if nesterov else buf
                orth = newton_schulz_orthogonalize(u, ns_steps)
                pr.mul_(1.0 - lr * wd)
                pr.add_(orth.to(pr.dtype), alpha=-lr * Muon._lr_ratio(pr.shape, "match_rms_adamw"))

    for pf, pr in zip(params_fast, params_ref):
        cos = torch.nn.functional.cosine_similarity(pf.detach().float().flatten(), pr.detach().float().flatten(), dim=0)
        assert float(cos) > 0.9999
        tol = 2e-3 if dtype == torch.bfloat16 else 2e-4
        torch.testing.assert_close(pf, pr, rtol=tol, atol=tol)


def test_batched_newton_schulz_matches_single():
    torch.manual_seed(3)
    mats = torch.randn(4, 320, 192, device="cuda", dtype=torch.bfloat16)
    batched = batched_newton_schulz_orthogonalize(mats.clone(), steps=5)
    single = torch.stack([newton_schulz_orthogonalize(mats[i], steps=5) for i in range(4)], dim=0)
    torch.testing.assert_close(batched, single, rtol=2e-2, atol=1e-2)
