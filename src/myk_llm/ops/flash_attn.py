"""FlashAttention (Dao et al., 2022) forward and backward as Triton kernels.

Replaces the flash-attn package, which has no sm_120 (RTX 50xx) / Python 3.14 builds.
The three entry points take the same arguments as their flash-attn namesakes:

    flash_attn_func          (batch, seq, heads, head_dim) tensors
    flash_attn_varlen_func   packed (tokens, heads, head_dim) documents with cu_seqlens
    flash_attn_with_kvcache  writes new keys/values into a static cache, then attends
                             (forward only; capturable in a CUDA graph)

Supported: causal or bidirectional attention, a sliding window, grouped-query attention
(GQA), bf16/fp16 inputs, head_dim a power of two in [16, 256]. Softmax runs in fp32 with
base-2 exponentials. As in flash-attn, queries are aligned to the *end* of the keys:
query i of Lq sits at key position i + Lk - Lq and sees keys j with

    i + Lk - Lq - window_left  <=  j  <=  i + Lk - Lq + window_right

where a window side of -1 is unbounded and causal forces window_right = 0.

The backward pass runs two kernels without atomics (deterministic): one per query block
for dQ, one per key block accumulating dK/dV over every query head of its GQA group.
torch.compile traces through the wrappers (no graph break).
"""

import math

import torch
import triton
import triton.language as tl
from torch import Tensor

_UNBOUNDED = 1 << 30  # a window side of -1, as a finite offset (int32-safe)
_LOG2E = 1.4426950408889634


# --- kernels ---------------------------------------------------------------------------


@triton.jit
def _seq_info(b, cu_seqlens, seqused, seqlen, IS_VARLEN: tl.constexpr, HAS_SEQUSED: tl.constexpr):
    """Sequence b's first token in the packed tensor (0 when dense), its first row in the
    (heads, rows) log-sum-exp (lse, base 2) and delta buffers, and its length."""
    if IS_VARLEN:
        start = tl.load(cu_seqlens + b)
        length = (tl.load(cu_seqlens + b + 1) - start).to(tl.int32)
        tok0 = start.to(tl.int64)
        row0 = tok0
    else:
        tok0 = tl.zeros([], tl.int64)
        row0 = b.to(tl.int64) * seqlen
        if HAS_SEQUSED:
            length = tl.load(seqused + b).to(tl.int32)
        else:
            length = tl.zeros([], tl.int32) + seqlen
    return tok0, row0, length


@triton.jit
def _block_ranges(s0, lo_off, hi_off, t_len, BLOCK_S: tl.constexpr, BLOCK_T: tl.constexpr):
    """For the block of "self" indices [s0, s0 + BLOCK_S), each seeing "other" indices t
    with lo_off <= t - s <= hi_off and 0 <= t < t_len: the BLOCK_T-aligned range
    [lo, hi) of t to visit, split as [lo, mid_lo) masked, [mid_lo, mid_hi) seen in full by
    every s (no mask needed), [mid_hi, hi) masked."""
    lo = tl.maximum(s0 + lo_off, 0) // BLOCK_T * BLOCK_T
    hi = tl.maximum(tl.minimum(s0 + BLOCK_S + hi_off, t_len), lo)
    mid_lo = tl.minimum(tl.cdiv(tl.maximum(s0 + BLOCK_S - 1 + lo_off, lo), BLOCK_T) * BLOCK_T, hi)
    mid_hi = tl.maximum(tl.minimum(s0 + hi_off + 1, t_len), 0) // BLOCK_T * BLOCK_T
    return lo, mid_lo, tl.maximum(mid_hi, mid_lo), hi


@triton.jit
def _band(s_idx, t_idx, lo_off, hi_off, t_len):
    rel = t_idx[None, :] - s_idx[:, None]
    return (rel >= lo_off) & (rel <= hi_off) & (t_idx[None, :] < t_len)


@triton.jit
def _fwd_inner(
    acc, m_i, l_i, q, K, V, stride_kt, stride_vt, qk_scale, offs_m, offs_n, offs_d,
    start, end, lo_off, hi_off, k_len, MASK: tl.constexpr, BLOCK_N: tl.constexpr,
):
    for n0 in range(start, end, BLOCK_N):
        cols = n0 + offs_n
        k_ptrs = K + cols[:, None] * stride_kt + offs_d[None, :]
        v_ptrs = V + cols[:, None] * stride_vt + offs_d[None, :]
        if MASK:
            k = tl.load(k_ptrs, mask=cols[:, None] < k_len, other=0.0)
        else:
            k = tl.load(k_ptrs)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        if MASK:
            s = tl.where(_band(offs_m, cols, lo_off, hi_off, k_len), s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        if MASK:
            v = tl.load(v_ptrs, mask=cols[:, None] < k_len, other=0.0)
        else:
            v = tl.load(v_ptrs)
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
        m_i = m_new
    return acc, m_i, l_i


@triton.jit(do_not_specialize=["seqlen_q", "seqlen_k", "window_left", "window_right"])
def _fwd_kernel(
    Q, K, V, Out, Lse, cu_seqlens_q, cu_seqlens_k, seqused_k,
    qk_scale, seqlen_q, seqlen_k, window_left, window_right,
    stride_qb, stride_qt, stride_qh, stride_kb, stride_kt, stride_kh,
    stride_vb, stride_vt, stride_vh, stride_ob, stride_ot, stride_oh, stride_lse,
    GROUP: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_VARLEN: tl.constexpr, HAS_SEQUSED: tl.constexpr,
):
    m0 = (tl.num_programs(0) - 1 - tl.program_id(0)) * BLOCK_M  # longest causal rows first
    b = tl.program_id(1)
    h = tl.program_id(2)
    q_tok, q_row, q_len = _seq_info(b, cu_seqlens_q, seqused_k, seqlen_q, IS_VARLEN, False)
    if m0 >= q_len:
        return
    k_tok, _, k_len = _seq_info(b, cu_seqlens_k, seqused_k, seqlen_k, IS_VARLEN, HAS_SEQUSED)
    b64, h64, kv_h64 = b.to(tl.int64), h.to(tl.int64), (h // GROUP).to(tl.int64)
    Q += b64 * stride_qb + q_tok * stride_qt + h64 * stride_qh
    K += b64 * stride_kb + k_tok * stride_kt + kv_h64 * stride_kh
    V += b64 * stride_vb + k_tok * stride_vt + kv_h64 * stride_vh
    Out += b64 * stride_ob + q_tok * stride_ot + h64 * stride_oh

    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + offs_m[:, None] * stride_qt + offs_d[None, :], mask=offs_m[:, None] < q_len, other=0.0)

    lo_off = k_len - q_len - window_left
    hi_off = k_len - q_len + window_right
    lo, mid_lo, mid_hi, hi = _block_ranges(m0, lo_off, hi_off, k_len, BLOCK_M, BLOCK_N)
    # A finite floor keeps exp2(m_i - m_new) NaN-free while a row has seen only masked keys.
    m_i = tl.full([BLOCK_M], -1e30, tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    acc, m_i, l_i = _fwd_inner(acc, m_i, l_i, q, K, V, stride_kt, stride_vt, qk_scale, offs_m, offs_n, offs_d,
                               lo, mid_lo, lo_off, hi_off, k_len, True, BLOCK_N)
    acc, m_i, l_i = _fwd_inner(acc, m_i, l_i, q, K, V, stride_kt, stride_vt, qk_scale, offs_m, offs_n, offs_d,
                               mid_lo, mid_hi, lo_off, hi_off, k_len, False, BLOCK_N)
    acc, m_i, l_i = _fwd_inner(acc, m_i, l_i, q, K, V, stride_kt, stride_vt, qk_scale, offs_m, offs_n, offs_d,
                               mid_hi, hi, lo_off, hi_off, k_len, True, BLOCK_N)

    # Rows that see no key at all (only possible when Lq > Lk) output 0; an infinite lse
    # zeroes their probabilities in the backward pass.
    empty = l_i == 0.0
    acc = acc / tl.where(empty, 1.0, l_i)[:, None]
    lse = tl.where(empty, float("inf"), m_i + tl.math.log2(l_i))
    row_ok = offs_m < q_len
    tl.store(Lse + h64 * stride_lse + q_row + offs_m, lse, mask=row_ok)
    tl.store(Out + offs_m[:, None] * stride_ot + offs_d[None, :], acc.to(Out.dtype.element_ty), mask=row_ok[:, None])


@triton.jit
def _dq_inner(
    dq, q, do, lse, delta, K, V, stride_kt, stride_vt, qk_scale, offs_m, offs_n, offs_d,
    start, end, lo_off, hi_off, k_len, MASK: tl.constexpr, BLOCK_N: tl.constexpr,
):
    for n0 in range(start, end, BLOCK_N):
        cols = n0 + offs_n
        k_ptrs = K + cols[:, None] * stride_kt + offs_d[None, :]
        v_ptrs = V + cols[:, None] * stride_vt + offs_d[None, :]
        if MASK:
            k = tl.load(k_ptrs, mask=cols[:, None] < k_len, other=0.0)
            v = tl.load(v_ptrs, mask=cols[:, None] < k_len, other=0.0)
        else:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        p = tl.math.exp2(tl.dot(q, tl.trans(k)) * qk_scale - lse[:, None])
        if MASK:
            p = tl.where(_band(offs_m, cols, lo_off, hi_off, k_len), p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq = tl.dot(ds.to(k.dtype), k, dq)
    return dq


@triton.jit(do_not_specialize=["seqlen_q", "seqlen_k", "window_left", "window_right"])
def _bwd_dq_kernel(
    Q, K, V, Out, dO, dQ, Lse, Delta, cu_seqlens_q, cu_seqlens_k,
    qk_scale, sm_scale, seqlen_q, seqlen_k, window_left, window_right,
    stride_qb, stride_qt, stride_qh, stride_kb, stride_kt, stride_kh, stride_vb, stride_vt, stride_vh,
    stride_ob, stride_ot, stride_oh, stride_dob, stride_dot, stride_doh, stride_dqb, stride_dqt, stride_dqh,
    stride_lse,
    GROUP: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """dQ for one query block; also writes delta = rowsum(dO * O), which the dK/dV
    kernel (launched after this one) needs for every query row."""
    m0 = (tl.num_programs(0) - 1 - tl.program_id(0)) * BLOCK_M
    b = tl.program_id(1)
    h = tl.program_id(2)
    q_tok, q_row, q_len = _seq_info(b, cu_seqlens_q, None, seqlen_q, IS_VARLEN, False)
    if m0 >= q_len:
        return
    k_tok, _, k_len = _seq_info(b, cu_seqlens_k, None, seqlen_k, IS_VARLEN, False)
    b64, h64, kv_h64 = b.to(tl.int64), h.to(tl.int64), (h // GROUP).to(tl.int64)
    Q += b64 * stride_qb + q_tok * stride_qt + h64 * stride_qh
    Out += b64 * stride_ob + q_tok * stride_ot + h64 * stride_oh
    dO += b64 * stride_dob + q_tok * stride_dot + h64 * stride_doh
    dQ += b64 * stride_dqb + q_tok * stride_dqt + h64 * stride_dqh
    K += b64 * stride_kb + k_tok * stride_kt + kv_h64 * stride_kh
    V += b64 * stride_vb + k_tok * stride_vt + kv_h64 * stride_vh

    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = offs_m < q_len
    q = tl.load(Q + offs_m[:, None] * stride_qt + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    do = tl.load(dO + offs_m[:, None] * stride_dot + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    o = tl.load(Out + offs_m[:, None] * stride_ot + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
    tl.store(Delta + h64 * stride_lse + q_row + offs_m, delta, mask=row_ok)
    lse = tl.load(Lse + h64 * stride_lse + q_row + offs_m, mask=row_ok, other=0.0)

    lo_off = k_len - q_len - window_left
    hi_off = k_len - q_len + window_right
    lo, mid_lo, mid_hi, hi = _block_ranges(m0, lo_off, hi_off, k_len, BLOCK_M, BLOCK_N)
    dq = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    dq = _dq_inner(dq, q, do, lse, delta, K, V, stride_kt, stride_vt, qk_scale, offs_m, offs_n, offs_d,
                   lo, mid_lo, lo_off, hi_off, k_len, True, BLOCK_N)
    dq = _dq_inner(dq, q, do, lse, delta, K, V, stride_kt, stride_vt, qk_scale, offs_m, offs_n, offs_d,
                   mid_lo, mid_hi, lo_off, hi_off, k_len, False, BLOCK_N)
    dq = _dq_inner(dq, q, do, lse, delta, K, V, stride_kt, stride_vt, qk_scale, offs_m, offs_n, offs_d,
                   mid_hi, hi, lo_off, hi_off, k_len, True, BLOCK_N)
    tl.store(dQ + offs_m[:, None] * stride_dqt + offs_d[None, :], (dq * sm_scale).to(dQ.dtype.element_ty),
             mask=row_ok[:, None])


@triton.jit
def _dkdv_inner(
    dk, dv, k, v, Q, dO, Lse, Delta, stride_qt, stride_dot, qk_scale, offs_n, offs_m, offs_d,
    start, end, lo_off, hi_off, q_len, MASK: tl.constexpr, BLOCK_M: tl.constexpr,
):
    # Transposed tiles (keys x queries), so each key block's dK/dV stay in registers.
    for m0 in range(start, end, BLOCK_M):
        rows = m0 + offs_m
        q_ptrs = Q + rows[:, None] * stride_qt + offs_d[None, :]
        do_ptrs = dO + rows[:, None] * stride_dot + offs_d[None, :]
        if MASK:
            row_ok = rows < q_len
            q = tl.load(q_ptrs, mask=row_ok[:, None], other=0.0)
            do = tl.load(do_ptrs, mask=row_ok[:, None], other=0.0)
            lse = tl.load(Lse + rows, mask=row_ok, other=0.0)
            delta = tl.load(Delta + rows, mask=row_ok, other=0.0)
        else:
            q = tl.load(q_ptrs)
            do = tl.load(do_ptrs)
            lse = tl.load(Lse + rows)
            delta = tl.load(Delta + rows)
        p_t = tl.math.exp2(tl.dot(k, tl.trans(q)) * qk_scale - lse[None, :])
        if MASK:
            p_t = tl.where(_band(offs_n, rows, lo_off, hi_off, q_len), p_t, 0.0)
        dv = tl.dot(p_t.to(do.dtype), do, dv)
        dp_t = tl.dot(v, tl.trans(do))
        ds_t = p_t * (dp_t - delta[None, :])
        dk = tl.dot(ds_t.to(q.dtype), q, dk)
    return dk, dv


@triton.jit(do_not_specialize=["seqlen_q", "seqlen_k", "window_left", "window_right"])
def _bwd_dkdv_kernel(
    Q, K, V, dO, dK, dV, Lse, Delta, cu_seqlens_q, cu_seqlens_k,
    qk_scale, sm_scale, seqlen_q, seqlen_k, window_left, window_right,
    stride_qb, stride_qt, stride_qh, stride_kb, stride_kt, stride_kh, stride_vb, stride_vt, stride_vh,
    stride_dob, stride_dot, stride_doh, stride_dkb, stride_dkt, stride_dkh,
    stride_dvb, stride_dvt, stride_dvh, stride_lse,
    GROUP: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    n0 = tl.program_id(0) * BLOCK_N  # causal: the first key blocks see the most queries
    b = tl.program_id(1)
    kv_h = tl.program_id(2)
    k_tok, _, k_len = _seq_info(b, cu_seqlens_k, None, seqlen_k, IS_VARLEN, False)
    if n0 >= k_len:
        return
    q_tok, q_row, q_len = _seq_info(b, cu_seqlens_q, None, seqlen_q, IS_VARLEN, False)
    b64, kv_h64 = b.to(tl.int64), kv_h.to(tl.int64)
    K += b64 * stride_kb + k_tok * stride_kt + kv_h64 * stride_kh
    V += b64 * stride_vb + k_tok * stride_vt + kv_h64 * stride_vh
    dK += b64 * stride_dkb + k_tok * stride_dkt + kv_h64 * stride_dkh
    dV += b64 * stride_dvb + k_tok * stride_dvt + kv_h64 * stride_dvh

    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    key_ok = offs_n < k_len
    k = tl.load(K + offs_n[:, None] * stride_kt + offs_d[None, :], mask=key_ok[:, None], other=0.0)
    v = tl.load(V + offs_n[:, None] * stride_vt + offs_d[None, :], mask=key_ok[:, None], other=0.0)

    # Query i sees key j iff (i - j) in [Lq - Lk - window_right, Lq - Lk + window_left].
    lo_off = q_len - k_len - window_right
    hi_off = q_len - k_len + window_left
    lo, mid_lo, mid_hi, hi = _block_ranges(n0, lo_off, hi_off, q_len, BLOCK_N, BLOCK_M)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    for g in range(GROUP):
        h64 = kv_h64 * GROUP + g
        Qh = Q + b64 * stride_qb + q_tok * stride_qt + h64 * stride_qh
        dOh = dO + b64 * stride_dob + q_tok * stride_dot + h64 * stride_doh
        Lse_h = Lse + h64 * stride_lse + q_row
        Delta_h = Delta + h64 * stride_lse + q_row
        dk, dv = _dkdv_inner(dk, dv, k, v, Qh, dOh, Lse_h, Delta_h, stride_qt, stride_dot, qk_scale,
                             offs_n, offs_m, offs_d, lo, mid_lo, lo_off, hi_off, q_len, True, BLOCK_M)
        dk, dv = _dkdv_inner(dk, dv, k, v, Qh, dOh, Lse_h, Delta_h, stride_qt, stride_dot, qk_scale,
                             offs_n, offs_m, offs_d, mid_lo, mid_hi, lo_off, hi_off, q_len, False, BLOCK_M)
        dk, dv = _dkdv_inner(dk, dv, k, v, Qh, dOh, Lse_h, Delta_h, stride_qt, stride_dot, qk_scale,
                             offs_n, offs_m, offs_d, mid_hi, hi, lo_off, hi_off, q_len, True, BLOCK_M)
    tl.store(dK + offs_n[:, None] * stride_dkt + offs_d[None, :], (dk * sm_scale).to(dK.dtype.element_ty),
             mask=key_ok[:, None])
    tl.store(dV + offs_n[:, None] * stride_dvt + offs_d[None, :], dv.to(dV.dtype.element_ty), mask=key_ok[:, None])


# --- launchers -------------------------------------------------------------------------

# (BLOCK_M, BLOCK_N, num_warps, num_stages) by head_dim, tuned on an RTX 5090 (sm_120,
# 99 KB of shared memory per block). Small backward tiles win there: bigger ones spill or
# drop to one stage.
_FWD_CONFIGS = {16: (64, 32, 4, 3), 32: (64, 32, 4, 3), 64: (64, 32, 4, 3), 128: (128, 32, 8, 3),
                256: (64, 32, 4, 2)}
_DECODE_CONFIG = (16, 64, 4, 3)  # few queries per sequence: one 16-row block
_DKDV_CONFIGS = {16: (32, 32, 4, 2), 32: (32, 32, 4, 2), 64: (32, 32, 4, 2), 128: (32, 32, 4, 2),
                 256: (32, 32, 4, 1)}
_DQ_CONFIGS = {16: (32, 32, 4, 2), 32: (32, 32, 4, 2), 64: (32, 32, 4, 2), 128: (32, 32, 4, 2),
               256: (32, 32, 4, 1)}


def _strides(x: Tensor) -> tuple[int, int, int]:
    """(batch, token, head) strides; batch is 0 for packed varlen tensors."""
    return (x.stride(0), x.stride(1), x.stride(2)) if x.dim() == 4 else (0, x.stride(0), x.stride(1))


def _check(q: Tensor, k: Tensor, v: Tensor) -> None:
    head_dim = q.shape[-1]
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise RuntimeError("flash attention is CUDA-only")
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError(f"flash attention needs matching bf16/fp16 q, k, v; got {q.dtype}, {k.dtype}, {v.dtype}")
    if head_dim not in _FWD_CONFIGS or k.shape[-1] != head_dim or v.shape[-1] != head_dim:
        raise ValueError(f"head_dim must be one of {sorted(_FWD_CONFIGS)} for q, k and v; got "
                         f"{q.shape[-1]}, {k.shape[-1]}, {v.shape[-1]}")
    if q.shape[-2] % k.shape[-2] or k.shape[-2] != v.shape[-2]:
        raise ValueError(f"query heads ({q.shape[-2]}) must be a multiple of key/value heads ({k.shape[-2]})")


def _last_dim_contiguous(x: Tensor) -> Tensor:
    return x if x.stride(-1) == 1 else x.contiguous()


def _forward(
    q: Tensor, k: Tensor, v: Tensor, cu_seqlens_q: Tensor | None, cu_seqlens_k: Tensor | None,
    seqused_k: Tensor | None, max_seqlen_q: int, max_seqlen_k: int, softmax_scale: float,
    window_left: int, window_right: int,
) -> tuple[Tensor, Tensor]:
    varlen = cu_seqlens_q is not None
    num_heads, head_dim = q.shape[-2], q.shape[-1]
    batch = cu_seqlens_q.numel() - 1 if cu_seqlens_q is not None else q.shape[0]
    rows = q.shape[0] if varlen else q.shape[0] * q.shape[1]
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    lse = torch.empty((num_heads, rows), dtype=torch.float32, device=q.device)
    block_m, block_n, warps, stages = _DECODE_CONFIG if max_seqlen_q <= 16 else _FWD_CONFIGS[head_dim]
    grid = (triton.cdiv(max_seqlen_q, block_m), batch, num_heads)
    _fwd_kernel[grid](
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, seqused_k,
        softmax_scale * _LOG2E, max_seqlen_q, max_seqlen_k, window_left, window_right,
        *_strides(q), *_strides(k), *_strides(v), *_strides(out), lse.stride(0),
        GROUP=num_heads // k.shape[-2], HEAD_DIM=head_dim, BLOCK_M=block_m, BLOCK_N=block_n,
        IS_VARLEN=varlen, HAS_SEQUSED=seqused_k is not None, num_warps=warps, num_stages=stages,
    )
    return out, lse


def _backward(
    dout: Tensor, q: Tensor, k: Tensor, v: Tensor, out: Tensor, lse: Tensor,
    cu_seqlens_q: Tensor | None, cu_seqlens_k: Tensor | None, max_seqlen_q: int, max_seqlen_k: int,
    softmax_scale: float, window_left: int, window_right: int,
) -> tuple[Tensor, Tensor, Tensor]:
    varlen = cu_seqlens_q is not None
    dout = _last_dim_contiguous(dout)
    num_heads, num_kv_heads, head_dim = q.shape[-2], k.shape[-2], q.shape[-1]
    batch = cu_seqlens_q.numel() - 1 if cu_seqlens_q is not None else q.shape[0]
    dq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    delta = torch.empty_like(lse)
    qk_scale = softmax_scale * _LOG2E
    seq_args = (max_seqlen_q, max_seqlen_k, window_left, window_right)

    block_m, block_n, warps, stages = _DQ_CONFIGS[head_dim]
    _bwd_dq_kernel[(triton.cdiv(max_seqlen_q, block_m), batch, num_heads)](
        q, k, v, out, dout, dq, lse, delta, cu_seqlens_q, cu_seqlens_k, qk_scale, softmax_scale, *seq_args,
        *_strides(q), *_strides(k), *_strides(v), *_strides(out), *_strides(dout), *_strides(dq), lse.stride(0),
        GROUP=num_heads // num_kv_heads, HEAD_DIM=head_dim, BLOCK_M=block_m, BLOCK_N=block_n,
        IS_VARLEN=varlen, num_warps=warps, num_stages=stages,
    )
    block_m, block_n, warps, stages = _DKDV_CONFIGS[head_dim]
    _bwd_dkdv_kernel[(triton.cdiv(max_seqlen_k, block_n), batch, num_kv_heads)](
        q, k, v, dout, dk, dv, lse, delta, cu_seqlens_q, cu_seqlens_k, qk_scale, softmax_scale, *seq_args,
        *_strides(q), *_strides(k), *_strides(v), *_strides(dout), *_strides(dk), *_strides(dv), lse.stride(0),
        GROUP=num_heads // num_kv_heads, HEAD_DIM=head_dim, BLOCK_M=block_m, BLOCK_N=block_n,
        IS_VARLEN=varlen, num_warps=warps, num_stages=stages,
    )
    return dq, dk, dv


# --- autograd --------------------------------------------------------------------------


class _FlashAttn(torch.autograd.Function):
    # A plain autograd.Function (not a torch.library op): torch.compile traces through it
    # and launches the Triton kernels from the compiled graph, and eager calls skip the
    # custom-op dispatch, which costs more than the kernels at small sizes.
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale,
                window_left, window_right):
        out, lse = _forward(q, k, v, cu_seqlens_q, cu_seqlens_k, None, max_seqlen_q, max_seqlen_k, softmax_scale,
                            window_left, window_right)
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.args = (max_seqlen_q, max_seqlen_k, softmax_scale, window_left, window_right)
        return out

    @staticmethod
    def backward(ctx, dout):  # pyright: ignore[reportIncompatibleMethodOverride]
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = _backward(dout, q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, *ctx.args)
        return dq, dk, dv, None, None, None, None, None, None, None


# --- public API ------------------------------------------------------------------------


def _window(causal: bool, window_size: tuple[int, int]) -> tuple[int, int]:
    left, right = window_size
    left = _UNBOUNDED if left < 0 else left
    right = 0 if causal else (_UNBOUNDED if right < 0 else right)
    return left, right


def flash_attn_func(
    q: Tensor, k: Tensor, v: Tensor, causal: bool = False, window_size: tuple[int, int] = (-1, -1),
    softmax_scale: float | None = None,
) -> Tensor:
    """q (batch, seqlen_q, heads, head_dim), k/v (batch, seqlen_k, kv_heads, head_dim)."""
    _check(q, k, v)
    q, k, v = (_last_dim_contiguous(x) for x in (q, k, v))
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    return _FlashAttn.apply(q, k, v, None, None, q.shape[1], k.shape[1], scale, *_window(causal, window_size))


def flash_attn_varlen_func(
    q: Tensor, k: Tensor, v: Tensor, cu_seqlens_q: Tensor, cu_seqlens_k: Tensor, max_seqlen_q: int,
    max_seqlen_k: int, causal: bool = False, window_size: tuple[int, int] = (-1, -1),
    softmax_scale: float | None = None,
) -> Tensor:
    """Packed q (total_q, heads, head_dim), k/v (total_k, kv_heads, head_dim); sequence b
    spans rows cu_seqlens[b]:cu_seqlens[b + 1] (int32)."""
    _check(q, k, v)
    q, k, v = (_last_dim_contiguous(x) for x in (q, k, v))
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    return _FlashAttn.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale,
                            *_window(causal, window_size))


def flash_attn_with_kvcache(
    q: Tensor, k_cache: Tensor, v_cache: Tensor, k: Tensor | None = None, v: Tensor | None = None, *,
    cache_seqlens: Tensor, causal: bool = False, window_size: tuple[int, int] = (-1, -1),
    softmax_scale: float | None = None,
) -> Tensor:
    """Writes k/v (batch, new, kv_heads, head_dim) into the caches (batch, capacity,
    kv_heads, head_dim) at cache_seqlens[b] onward, then attends q over each sequence's
    cache_seqlens[b] + new keys. The caller advances cache_seqlens. Forward only."""
    _check(q, k_cache, v_cache)
    if k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        raise ValueError("the caches need a contiguous last dimension")
    q = _last_dim_contiguous(q)
    seqused = cache_seqlens.to(torch.int32)
    if k is not None and v is not None:
        batch, new = k.shape[:2]
        rows = torch.arange(batch, device=k.device)[:, None]
        cols = cache_seqlens.long()[:, None] + torch.arange(new, device=k.device)
        k_cache[rows, cols] = k  # integer-index writes: no host sync, so graph-capturable
        v_cache[rows, cols] = v
        seqused = seqused + new
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    out, _ = _forward(q, k_cache, v_cache, None, None, seqused, q.shape[1], k_cache.shape[1], scale,
                      *_window(causal, window_size))
    return out
