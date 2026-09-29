"""Fused linear + cross-entropy (CE) with optional z-loss, never materializing (N, V) logits.

Tokens are processed in chunks; each chunk's logits live only long enough for a Triton
kernel to overwrite them in place with d(loss)/d(logits), which are immediately folded
into the hidden-state and weight gradients. Peak memory is O(chunk * V).

The gradient is computed during the forward pass, so the result must be backpropagated
exactly once (the normal training pattern).
"""

from __future__ import annotations

from typing import cast

import torch
import triton
import triton.language as tl


@triton.jit
def _ce_kernel(
    logits_ptr,  # (rows, n_cols), overwritten with d loss / d logits
    targets_ptr,  # (rows,) int64
    ce_ptr,  # (rows,) fp32 out: -log p(target)
    lse_ptr,  # (rows,) fp32 out: logsumexp
    scale,  # gradient scale (1 / normalizer)
    z_coef,  # z-loss coefficient: loss += z_coef * lse^2
    n_cols,
    row_stride,
    ignore_index,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_start = logits_ptr + row * row_stride
    target = tl.load(targets_ptr + row).to(tl.int32)

    if target == ignore_index:
        for off in range(0, n_cols, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            tl.store(row_start + cols, tl.zeros([BLOCK], tl.float32), mask=cols < n_cols)
        tl.store(ce_ptr + row, 0.0)
        tl.store(lse_ptr + row, 0.0)
        return

    # Pass 1: online softmax -> logsumexp.
    m = -float("inf")
    d = 0.0
    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        vals = tl.load(row_start + cols, mask=cols < n_cols, other=-float("inf")).to(tl.float32)
        new_m = tl.maximum(m, tl.max(vals, axis=0))
        d = d * tl.exp(m - new_m) + tl.sum(tl.exp(vals - new_m), axis=0)
        m = new_m
    lse = m + tl.log(d)
    target_logit = tl.load(row_start + target).to(tl.float32)
    tl.store(ce_ptr + row, lse - target_logit)
    tl.store(lse_ptr + row, lse)

    # Pass 2: d/dlogits [ce + z * lse^2] = softmax * (1 + 2 z lse) - onehot.
    softmax_scale = scale * (1.0 + 2.0 * z_coef * lse)
    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < n_cols
        vals = tl.load(row_start + cols, mask=mask, other=0.0).to(tl.float32)
        grad = tl.exp(vals - lse) * softmax_scale
        grad = tl.where(cols == target, grad - scale, grad)
        tl.store(row_start + cols, grad, mask=mask)


class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, normalizer, z_coef, ignore_index, chunk_size):
        num_tokens, hidden_size = hidden.shape
        vocab_size = weight.shape[0]
        scale = 1.0 / normalizer
        compute_dtype = torch.bfloat16  # GEMMs in bf16 (fp32 accumulate); softmax in fp32
        weight_c = weight.to(compute_dtype)
        grad_hidden = torch.empty(num_tokens, hidden_size, device=hidden.device, dtype=compute_dtype)
        grad_weight = torch.zeros(vocab_size, hidden_size, device=hidden.device, dtype=torch.float32)
        ce = torch.empty(num_tokens, device=hidden.device, dtype=torch.float32)
        lse = torch.empty(num_tokens, device=hidden.device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", enabled=False):
            for start in range(0, num_tokens, chunk_size):
                end = min(start + chunk_size, num_tokens)
                hidden_chunk = hidden[start:end].to(compute_dtype)
                logits = hidden_chunk @ weight_c.t()
                _ce_kernel[(end - start,)](
                    logits, targets[start:end], ce[start:end], lse[start:end],
                    scale, z_coef, vocab_size, logits.stride(0), ignore_index,
                    BLOCK=4096, num_warps=8,
                )
                torch.mm(logits, weight_c, out=grad_hidden[start:end])
                grad_weight.add_(logits.t() @ hidden_chunk)
        ctx.save_for_backward(grad_hidden, grad_weight)
        ctx.hidden_dtype, ctx.weight_dtype = hidden.dtype, weight.dtype
        ce_sum, z_sum = ce.sum(), lse.square().sum()
        loss = (ce_sum + z_coef * z_sum) * scale
        ctx.mark_non_differentiable(ce_sum, z_sum)
        return loss, ce_sum, z_sum

    @staticmethod
    def backward(ctx, grad_loss, _grad_ce, _grad_z):  # pyright: ignore[reportIncompatibleMethodOverride]
        grad_hidden, grad_weight = ctx.saved_tensors
        return (
            (grad_hidden * grad_loss).to(ctx.hidden_dtype),
            (grad_weight * grad_loss).to(ctx.weight_dtype),
            None, None, None, None, None,
        )


def fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    normalizer: float,
    z_loss: float = 0.0,
    ignore_index: int = -100,
    chunk_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (loss, ce_sum, z_sum): loss = (sum CE + z_loss * sum lse^2) / normalizer.

    Passing the normalizer explicitly (e.g. supervised tokens across all micro-batches
    and ranks) makes gradient accumulation and data parallelism token-exact.
    `ce_sum` / `z_sum` are detached sums over supervised tokens, for logging.
    """
    if not hidden.is_cuda:
        raise RuntimeError("fused cross entropy is CUDA-only")
    result = _FusedLinearCE.apply(
        hidden.reshape(-1, hidden.size(-1)), weight, targets.reshape(-1), float(normalizer), float(z_loss),
        ignore_index, chunk_size,
    )
    return cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor], result)
