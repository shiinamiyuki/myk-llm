import math

import pytest
import torch
import torch.nn.functional as F

from conftest import requires_cuda
from myk_llm.ops import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache

pytestmark = requires_cuda


def reference(q, k, v, causal, window):
    """fp32 attention with flash-attn's masking: queries aligned to the end of the keys."""
    seqlen_q, seqlen_k, heads = q.shape[1], k.shape[1], q.shape[2]
    k = k.float().repeat_interleave(heads // k.shape[2], 2)
    v = v.float().repeat_interleave(heads // v.shape[2], 2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k) / math.sqrt(q.shape[-1])
    pos = torch.arange(seqlen_q, device=q.device)[:, None] + seqlen_k - seqlen_q
    key = torch.arange(seqlen_k, device=q.device)[None, :]
    left, right = window[0], 0 if causal else window[1]
    visible = torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device)
    if left >= 0:
        visible &= key >= pos - left
    if right >= 0:
        visible &= key <= pos + right
    probs = torch.softmax(scores.masked_fill(~visible, float("-inf")), -1).nan_to_num(0.0)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v)


def random_qkv(batch, seqlen_q, seqlen_k, heads, kv_heads, head_dim, packed=False):
    lead_q, lead_k = ((seqlen_q,), (seqlen_k,)) if packed else ((batch, seqlen_q), (batch, seqlen_k))
    make = lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    return make(*lead_q, heads, head_dim), make(*lead_k, kv_heads, head_dim), make(*lead_k, kv_heads, head_dim)


def assert_close(actual, expected):
    # bf16 inputs/outputs: compare relative to the largest reference value.
    tol = 2e-2 * expected.abs().max().item() + 1e-3
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=tol)


@pytest.mark.parametrize(
    "seqlen_q, seqlen_k, heads, kv_heads, head_dim, causal, window",
    [
        (128, 128, 4, 4, 64, True, (-1, -1)),
        (200, 200, 4, 2, 64, True, (-1, -1)),  # partial blocks, GQA
        (300, 300, 8, 2, 128, True, (63, 0)),  # sliding window
        (257, 257, 2, 1, 64, False, (-1, -1)),
        (257, 257, 2, 1, 64, False, (31, 17)),
        (5, 300, 4, 2, 128, True, (63, 0)),  # cached decoding: queries at the end of the keys
        (100, 37, 4, 2, 64, True, (-1, -1)),  # more queries than keys: early rows see nothing
        (130, 130, 2, 1, 256, True, (-1, -1)),
        (130, 130, 2, 1, 16, True, (7, 0)),
    ],
)
def test_matches_reference_output_and_gradients(seqlen_q, seqlen_k, heads, kv_heads, head_dim, causal, window):
    torch.manual_seed(0)
    q, k, v = random_qkv(2, seqlen_q, seqlen_k, heads, kv_heads, head_dim)
    out = flash_attn_func(q, k, v, causal=causal, window_size=window)
    grad_out = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), grad_out)
    expected = reference(q, k, v, causal, window)
    expected_grads = torch.autograd.grad(expected, (q, k, v), grad_out.float())
    assert_close(out, expected)
    for grad, expected_grad in zip(grads, expected_grads):
        assert_close(grad, expected_grad)


@pytest.mark.parametrize("window", [(-1, -1), (15, 0)])
def test_varlen_keeps_documents_separate(window):
    torch.manual_seed(0)
    lengths = [1, 17, 128, 300, 5, 64, 3]
    cu_seqlens = F.pad(torch.tensor(lengths).cumsum(0), (1, 0)).to(torch.int32).cuda()
    q, k, v = random_qkv(1, sum(lengths), sum(lengths), 4, 2, 64, packed=True)
    out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, 512, 512, causal=True, window_size=window)
    grad_out = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), grad_out)
    bounds = cu_seqlens.tolist()
    expected = torch.cat([reference(q[None, a:b], k[None, a:b], v[None, a:b], True, window)[0]
                          for a, b in zip(bounds, bounds[1:])])
    expected_grads = torch.autograd.grad(expected, (q, k, v), grad_out.float())
    assert_close(out, expected)
    for grad, expected_grad in zip(grads, expected_grads):
        assert_close(grad, expected_grad)


@pytest.mark.parametrize("window", [(-1, -1), (15, 0)])
def test_kvcache_matches_full_attention(window):
    torch.manual_seed(0)
    batch, heads, kv_heads, head_dim = 3, 8, 2, 128
    with torch.no_grad():
        q, k, v = (x.detach() for x in random_qkv(batch, 41, 41, heads, kv_heads, head_dim))
        k_cache = torch.zeros(batch, 64, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        v_cache = torch.zeros_like(k_cache)
        cache_seqlens = torch.zeros(batch, dtype=torch.int32, device="cuda")
        outs = []
        for a, b in [(0, 37), (37, 38), (38, 41)]:  # prefill, then steps of 1 and 3 tokens
            outs.append(flash_attn_with_kvcache(q[:, a:b], k_cache, v_cache, k[:, a:b], v[:, a:b],
                                                cache_seqlens=cache_seqlens, causal=True, window_size=window))
            cache_seqlens += b - a
        torch.testing.assert_close(k_cache[:, :41], k, rtol=0, atol=0)
        assert_close(torch.cat(outs, 1), reference(q, k, v, True, window))

        # Ragged caches: each sequence attends over its own length only.
        cache_seqlens = torch.tensor([5, 20, 37], dtype=torch.int32, device="cuda")
        q_new, k_new, v_new = (x.detach() for x in random_qkv(batch, 2, 2, heads, kv_heads, head_dim))
        out = flash_attn_with_kvcache(q_new, k_cache, v_cache, k_new, v_new, cache_seqlens=cache_seqlens,
                                      causal=True, window_size=window)
        for i, length in enumerate((cache_seqlens + 2).tolist()):
            assert_close(out[i : i + 1], reference(q_new[i : i + 1], k_cache[i : i + 1, :length],
                                                   v_cache[i : i + 1, :length], True, window))


def test_compiles_without_graph_breaks():
    torch.manual_seed(0)
    q, k, v = random_qkv(2, 96, 96, 4, 2, 64)

    def attend(q, k, v):
        return flash_attn_func(q * 1.0, k, v, causal=True, window_size=(31, 0)).float().square().sum()

    compiled = torch.compile(attend, fullgraph=True)
    loss = compiled(q, k, v)
    grads = torch.autograd.grad(loss, (q, k, v))
    expected = attend(q, k, v)
    expected_grads = torch.autograd.grad(expected, (q, k, v))
    torch.testing.assert_close(loss, expected)
    for grad, expected_grad in zip(grads, expected_grads):
        torch.testing.assert_close(grad, expected_grad)


def test_rejects_unsupported_inputs():
    q = torch.randn(1, 8, 2, 48, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="head_dim"):
        flash_attn_func(q, q, q)
    with pytest.raises(TypeError, match="bf16"):
        flash_attn_func(q.float(), q.float(), q.float())
