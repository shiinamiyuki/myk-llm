"""Fused Triton kernels for the Muon optimizer (batched symmetric Gram products and
multi-tensor momentum / parameter updates).

Each quintic Newton-Schulz iteration computes:
    G  = X @ X^T                (symmetric: M x K times K x M -> M x M)
    G2 = b * G + c * (G @ G)    (symmetric: G == G^T implies G @ G is symmetric)
    X' = a * X + G2 @ X         (general batched matmul via cuBLAS baddbmm)

Evaluating only the lower-triangular tiles of G and G2 and mirroring them across the
diagonal (a batched Symmetric Rank-K Update (SYRK)) cuts the two symmetric matmuls'
floating-point operations in half while staying numerically identical to full matmuls.
Multi-tensor pointer-array kernels pack Brain Floating Point 16-bit (BF16) inputs (while
updating 32-bit Floating Point (FP32) or BF16 momentum buffers and accumulating norms in
one pass) and apply decoupled weight decay + orthogonalized updates back in one pass.
"""

from __future__ import annotations

from typing import Callable

import torch
import triton
import triton.language as tl

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


@triton.autotune(
    configs=[
        triton.Config({"BM": 128, "BK": 64}, num_stages=3, num_warps=8),
        triton.Config({"BM": 128, "BK": 32}, num_stages=4, num_warps=8),
        triton.Config({"BM": 64, "BK": 64}, num_stages=4, num_warps=4),
    ],
    key=["M", "K", "STRIDED_X", "HAS_BIAS"],
)
@triton.jit
def _batched_syrk_kernel(
    X_ptr,
    Bias_ptr,
    Out_ptr,
    M,
    K,
    stride_xb,
    stride_xm,
    stride_xk,
    stride_bb,
    stride_bm,
    stride_bn,
    stride_ob,
    stride_om,
    stride_on,
    alpha,
    beta,
    STRIDED_X: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    """Out[b] = alpha * (X[b] @ X[b]^T) + beta * Bias[b] via lower-triangular tile mapping."""
    pid = tl.program_id(0)
    b = tl.program_id(1)
    pid_m = ((tl.math.sqrt(8.0 * pid.to(tl.float32) + 1.0) - 1.0) * 0.5).to(tl.int32)
    if pid_m * (pid_m + 1) // 2 > pid:
        pid_m -= 1
    if (pid_m + 1) * (pid_m + 2) // 2 <= pid:
        pid_m += 1
    pid_n = pid - pid_m * (pid_m + 1) // 2

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BM + tl.arange(0, BM)
    offs_k = tl.arange(0, BK)

    x_base = X_ptr + b.to(tl.int64) * stride_xb
    a_ptrs = x_base + offs_m[:, None].to(tl.int64) * stride_xm + offs_k[None, :].to(tl.int64) * stride_xk
    b_ptrs = x_base + offs_n[:, None].to(tl.int64) * stride_xm + offs_k[None, :].to(tl.int64) * stride_xk

    acc = tl.zeros((BM, BM), dtype=tl.float32)
    for k0 in range(0, K, BK):
        k_mask = (k0 + offs_k) < K
        a_tile = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b_tile = tl.load(b_ptrs, mask=(offs_n[:, None] < M) & k_mask[None, :], other=0.0)
        acc = tl.dot(a_tile, tl.trans(b_tile), acc)
        a_ptrs += BK * stride_xk
        b_ptrs += BK * stride_xk

    if HAS_BIAS:
        bias_base = Bias_ptr + b.to(tl.int64) * stride_bb
        bias_tile = tl.load(
            bias_base + offs_m[:, None].to(tl.int64) * stride_bm + offs_n[None, :].to(tl.int64) * stride_bn,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < M),
            other=0.0,
        ).to(tl.float32)
        acc = alpha * acc + beta * bias_tile
    elif alpha != 1.0:
        acc = alpha * acc

    out_tile = acc.to(tl.bfloat16)
    out_base = Out_ptr + b.to(tl.int64) * stride_ob
    mask_mn = (offs_m[:, None] < M) & (offs_n[None, :] < M)
    tl.store(out_base + offs_m[:, None].to(tl.int64) * stride_om + offs_n[None, :].to(tl.int64) * stride_on, out_tile, mask=mask_mn)

    if pid_m != pid_n:
        mask_nm = (offs_n[:, None] < M) & (offs_m[None, :] < M)
        tl.store(out_base + offs_n[:, None].to(tl.int64) * stride_om + offs_m[None, :].to(tl.int64) * stride_on, tl.trans(out_tile), mask=mask_nm)


def batched_syrk(
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute `alpha * (x @ x.transpose(-2, -1)) + beta * bias` for BF16 `x` of shape (B, M, K)."""
    B, M, K = x.shape
    if out is None:
        out = torch.empty((B, M, M), device=x.device, dtype=torch.bfloat16)

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        blocks = triton.cdiv(M, meta["BM"])
        return (blocks * (blocks + 1) // 2, B)

    _batched_syrk_kernel[grid](
        x,
        bias if bias is not None else x,
        out,
        M,
        K,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        bias.stride(0) if bias is not None else 0,
        bias.stride(1) if bias is not None else 0,
        bias.stride(2) if bias is not None else 0,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        alpha,
        beta,
        STRIDED_X=(x.stride(2) != 1),
        HAS_BIAS=bias is not None,
    )
    return out


@triton.jit
def _fused_pre_pack_norm_kernel(
    grad_ptrs,
    buf_ptrs,
    out_ptr,
    norm_sq_ptr,
    N,
    mom,
    GRAD_BF16: tl.constexpr,
    BUF_BF16: tl.constexpr,
    NESTEROV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    g_ty = tl.bfloat16 if GRAD_BF16 else tl.float32
    m_ty = tl.bfloat16 if BUF_BF16 else tl.float32
    g_ptr = tl.load(grad_ptrs + b).to(tl.pointer_type(g_ty))
    m_ptr = tl.load(buf_ptrs + b).to(tl.pointer_type(m_ty))
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m_new = m + (1.0 - mom) * (g - m)
    tl.store(m_ptr + offs, m_new.to(m_ty), mask=mask)
    u = g + mom * (m_new - g) if NESTEROV else m_new
    u_bf16 = u.to(tl.bfloat16)
    tl.store(out_ptr + b * N + offs, u_bf16, mask=mask)
    uf = u_bf16.to(tl.float32)
    sq = tl.sum(tl.where(mask, uf * uf, 0.0), axis=0)
    tl.atomic_add(norm_sq_ptr + b, sq)


@triton.jit
def _scale_bf16_kernel(
    x_ptr,
    norm_sq_ptr,
    N,
    eps,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    inv = 1.0 / tl.maximum(tl.sqrt(tl.load(norm_sq_ptr + b)), eps)
    val = tl.load(x_ptr + b * N + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(x_ptr + b * N + offs, (val * inv).to(tl.bfloat16), mask=mask)


@triton.jit
def _fused_post_unpack_kernel(
    param_ptrs,
    orth_ptr,
    N,
    decay_mul,
    step_scale,
    PARAM_BF16: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    p_ty = tl.bfloat16 if PARAM_BF16 else tl.float32
    p_ptr = tl.load(param_ptrs + b).to(tl.pointer_type(p_ty))
    p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(orth_ptr + b * N + offs, mask=mask, other=0.0).to(tl.float32)
    p_new = p * decay_mul + step_scale * u
    tl.store(p_ptr + offs, p_new.to(p_ty), mask=mask)


def batched_newton_schulz_orthogonalize(
    packed: torch.Tensor,
    steps: int,
    transposed: bool | None = None,
    norm_sq: torch.Tensor | None = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Batched quintic Newton-Schulz on `packed` of shape (B, rows, cols) in BF16.

    When `norm_sq` (per-matrix squared Frobenius norm of `packed`) is supplied, `packed`
    is normalized in place via `_scale_bf16_kernel`; otherwise its norm is computed first.
    Uses zero-copy transpose views when `rows > cols` so the returned tensor is always
    contiguous in `(B, rows, cols)` layout.
    """
    a, b, c = NS_COEFFICIENTS
    if packed.dtype != torch.bfloat16:
        packed = packed.bfloat16()
    if not packed.is_contiguous():
        packed = packed.contiguous()
    B, rows, cols = packed.shape
    if transposed is None:
        transposed = rows > cols
    numel = rows * cols
    BLOCK = 1024
    if norm_sq is not None:
        _scale_bf16_kernel[(triton.cdiv(numel, BLOCK), B)](packed, norm_sq, numel, eps, BLOCK=BLOCK, num_warps=8)
    else:
        norm = torch.linalg.vector_norm(packed, dim=(-2, -1), keepdim=True, dtype=torch.float32).clamp_(min=eps)
        packed = (packed / norm).bfloat16()

    x = packed.transpose(-2, -1) if transposed else packed
    M = x.size(1)
    if M < 64:
        for _ in range(steps):
            gram = torch.bmm(x, x.transpose(-2, -1))
            gram2 = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
            x = torch.baddbmm(x, gram2, x, beta=a, alpha=1.0)
        return x.transpose(-2, -1).contiguous() if transposed else x

    gram = torch.empty((B, M, M), device=x.device, dtype=torch.bfloat16)
    gram2 = torch.empty((B, M, M), device=x.device, dtype=torch.bfloat16)
    buf2 = torch.empty_like(packed)
    x_next = buf2.transpose(-2, -1) if transposed else buf2
    cur_buf, nxt_buf = packed, buf2
    for _ in range(steps):
        batched_syrk(x, out=gram)
        batched_syrk(gram, bias=gram, alpha=c, beta=b, out=gram2)
        torch.baddbmm(x, gram2, x, beta=a, alpha=1.0, out=x_next)
        x, x_next = x_next, x
        cur_buf, nxt_buf = nxt_buf, cur_buf
    return cur_buf


def muon_step_cuda(
    triples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_steps: int,
    adjust_lr: str,
    lr_ratio_fn: Callable[[torch.Size, str], float],
    max_chunk_bytes: int = 128 * 1024 * 1024,
) -> None:
    """Batched Compute Unified Device Architecture (CUDA) step for `(param, grad, buf)` triples."""
    if not triples:
        return
    device = triples[0][0].device
    # Group by (shape, param_dtype, grad_dtype, buf_dtype) so pointer types are uniform per bucket.
    buckets: dict[tuple[torch.Size, torch.dtype, torch.dtype, torch.dtype], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    for p, g, m in triples:
        key = (p.shape, p.dtype, g.dtype, m.dtype)
        buckets.setdefault(key, []).append((p, g, m))

    # Build one flat Central Processing Unit (CPU) pointer table across all chunks for a
    # single host-to-device copy (avoiding per-chunk stream syncs).
    chunks: list[tuple[torch.Size, torch.dtype, torch.dtype, torch.dtype, int, int]] = []
    flat_ptrs: list[int] = []
    for (shape, p_dt, g_dt, m_dt), items in buckets.items():
        numel = shape[0] * shape[1]
        chunk_size = max(1, max_chunk_bytes // max(1, numel * 2))
        for start in range(0, len(items), chunk_size):
            sub = items[start : start + chunk_size]
            offset = len(flat_ptrs)
            flat_ptrs.extend(g.data_ptr() for _, g, _ in sub)
            flat_ptrs.extend(m.data_ptr() for _, _, m in sub)
            flat_ptrs.extend(p.data_ptr() for p, _, _ in sub)
            chunks.append((shape, p_dt, g_dt, m_dt, offset, len(sub)))

    ptrs_cpu = torch.tensor(flat_ptrs, dtype=torch.int64, pin_memory=True)
    ptrs_gpu = ptrs_cpu.to(device=device, non_blocking=True)
    norm_buf = torch.empty((max(c[5] for c in chunks),), device=device, dtype=torch.float32)

    BLOCK = 1024
    decay_mul = 1.0 - lr * weight_decay
    for shape, p_dt, g_dt, m_dt, offset, count in chunks:
        rows, cols = shape
        numel = rows * cols
        g_ptrs = ptrs_gpu[offset : offset + count]
        m_ptrs = ptrs_gpu[offset + count : offset + 2 * count]
        p_ptrs = ptrs_gpu[offset + 2 * count : offset + 3 * count]

        packed = torch.empty((count, rows, cols), device=device, dtype=torch.bfloat16)
        norm_sq = norm_buf[:count]
        norm_sq.zero_()
        grid = (triton.cdiv(numel, BLOCK), count)
        _fused_pre_pack_norm_kernel[grid](
            g_ptrs,
            m_ptrs,
            packed,
            norm_sq,
            numel,
            momentum,
            GRAD_BF16=(g_dt == torch.bfloat16),
            BUF_BF16=(m_dt == torch.bfloat16),
            NESTEROV=nesterov,
            BLOCK=BLOCK,
            num_warps=8,
        )
        orth = batched_newton_schulz_orthogonalize(packed, ns_steps, transposed=(rows > cols), norm_sq=norm_sq)
        step_scale = -lr * lr_ratio_fn(shape, adjust_lr)
        _fused_post_unpack_kernel[grid](
            p_ptrs,
            orth,
            numel,
            decay_mul,
            step_scale,
            PARAM_BF16=(p_dt == torch.bfloat16),
            BLOCK=BLOCK,
            num_warps=8,
        )
