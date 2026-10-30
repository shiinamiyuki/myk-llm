"""Speed of our Triton FlashAttention (`myk_llm.ops.flash_attn`) against PyTorch's
scaled_dot_product_attention (SDPA) backends and FlexAttention, forward and
forward + backward, on the shapes this project trains with (bf16, one GPU).

SDPA only covers plain causal attention; sliding windows and packed documents are
compared with FlexAttention alone. TFLOPS count only the visible (query, key) pairs;
the backward pass is counted as 2.5x the forward.

    python scripts/benchmark_attention.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from myk_llm.ops import flash_attn_func, flash_attn_varlen_func

# (batch, seq_len, heads, kv_heads, head_dim, sliding window or None, document lengths or None)
SHAPES = [
    (8, 2048, 8, 2, 128, None, None),  # 200M global layer
    (8, 2048, 8, 2, 128, 512, None),  # 200M local layer
    (8, 2048, 8, 2, 128, None, [300, 700, 48, 1000]),  # document masking
    (2, 8192, 8, 2, 128, None, None),
    (16, 1024, 4, 4, 64, None, None),  # lc testbed
    (16, 1024, 4, 4, 64, 256, None),
]


def milliseconds(fn, repeats: int = 30) -> float:
    for _ in range(3):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def compare(batch, seq_len, heads, kv_heads, head_dim, window, docs) -> None:
    make = lambda h: torch.randn(batch, seq_len, h, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    q, k, v = make(heads), make(kv_heads), make(kv_heads)
    grad_out = torch.randn_like(q)
    lengths = docs or [seq_len]
    pairs = batch * sum(min(i + 1, window or seq_len) for n in lengths for i in range(n))
    flops = 4 * pairs * heads * head_dim

    results = {}
    if docs is None:
        ours = lambda: flash_attn_func(q, k, v, causal=True, window_size=(window - 1, 0) if window else (-1, -1))
        inputs = (q, k, v)
    else:
        cu_seqlens = F.pad(torch.tensor(docs * batch).cumsum(0), (1, 0)).to(torch.int32).cuda()
        inputs = tuple(x.detach().flatten(0, 1).requires_grad_() for x in (q, k, v))
        ours = lambda: flash_attn_varlen_func(*inputs, cu_seqlens, cu_seqlens, seq_len, seq_len, causal=True,
                                              window_size=(window - 1, 0) if window else (-1, -1))
    flat_grad = grad_out if docs is None else grad_out.flatten(0, 1)
    results["triton (ours)"] = (milliseconds(ours), milliseconds(lambda: torch.autograd.grad(ours(), inputs, flat_grad)))

    qt, kt, vt, grad_t = (x.detach().transpose(1, 2).requires_grad_() for x in (q, k, v, grad_out))
    if window is None and docs is None:
        for name, backend in [("sdpa flash", SDPBackend.FLASH_ATTENTION), ("sdpa cudnn", SDPBackend.CUDNN_ATTENTION)]:
            with sdpa_kernel(backend):
                sdpa = lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=True, enable_gqa=True)
                results[name] = (milliseconds(sdpa), milliseconds(lambda: torch.autograd.grad(sdpa(), (qt, kt, vt), grad_t)))

    doc_ids = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths)).cuda()

    def mask_mod(b, h, q_idx, kv_idx):
        visible = (q_idx >= kv_idx) & (doc_ids[q_idx] == doc_ids[kv_idx])
        return visible & (q_idx - kv_idx < window) if window else visible

    block_mask = create_block_mask(mask_mod, None, None, seq_len, seq_len)
    compiled = torch.compile(flex_attention, dynamic=False)
    flex = lambda: compiled(qt, kt, vt, block_mask=block_mask, enable_gqa=True)
    results["flex"] = (milliseconds(flex), milliseconds(lambda: torch.autograd.grad(flex(), (qt, kt, vt), grad_t)))

    print(f"\nbatch {batch} x {seq_len}, heads {heads}/{kv_heads} x {head_dim}, window {window}, "
          f"documents {docs}")
    print("| kernel | fwd ms | fwd TFLOPS | fwd+bwd ms | fwd+bwd TFLOPS |\n|---|---|---|---|---|")
    for name, (forward, both) in results.items():
        print(f"| {name} | {forward:.3f} | {flops / forward / 1e9:.0f} | {both:.3f} | {3.5 * flops / both / 1e9:.0f} |")


def main() -> None:
    torch.manual_seed(0)
    for shape in SHAPES:
        compare(*shape)


if __name__ == "__main__":
    main()
