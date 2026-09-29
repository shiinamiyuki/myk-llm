import pytest
import torch
import torch.nn.functional as F

from conftest import requires_cuda
from myk_llm.ops import fused_linear_cross_entropy


@requires_cuda
@pytest.mark.parametrize("z_loss", [0.0, 1e-3])
def test_matches_reference_loss_and_gradients(z_loss):
    torch.manual_seed(0)
    n, h, v = 300, 64, 1000
    hidden = torch.randn(n, h, device="cuda", requires_grad=True)
    weight = (0.1 * torch.randn(v, h, device="cuda")).requires_grad_()
    targets = torch.randint(0, v, (n,), device="cuda")
    targets[::7] = -100
    normalizer = 123.0

    loss, ce_sum, z_sum = fused_linear_cross_entropy(hidden, weight, targets, normalizer, z_loss, chunk_size=128)
    loss.backward()
    grads = hidden.grad.clone(), weight.grad.clone()
    hidden.grad = weight.grad = None

    logits = (hidden.bfloat16() @ weight.bfloat16().t()).float()
    valid = targets != -100
    ce = F.cross_entropy(logits, targets, ignore_index=-100, reduction="sum")
    lse = torch.logsumexp(logits, -1)[valid]
    ref = (ce + z_loss * lse.square().sum()) / normalizer
    ref.backward()
    torch.testing.assert_close(loss, ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(ce_sum, ce.detach(), rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(grads[0], hidden.grad, rtol=2e-2, atol=2e-4)
    torch.testing.assert_close(grads[1], weight.grad, rtol=2e-2, atol=2e-4)
