"""Open Compute Project (OCP) Microscaling 8-bit Floating Point (MXFP8) linear layer.

Uses 32-element block scaling (`float8_e4m3fn` elements with `float8_e8m0fnu` power-of-two
scales computed via ceiling ratio `ScaleCalculationMode.RCEIL`) for all three General
Matrix Multiply (GEMM) operations of a linear layer on NVIDIA Blackwell Graphics
Processing Units (GPUs) (compute capability >= 10.0).

Unlike `torchao.prototype.moe_training`, this module:
- Keeps `self.weight` as a plain 32-bit Floating Point (FP32) or Brain Floating Point
  16-bit (BF16) `nn.Parameter` (interchangeable checkpoints, Muon, and Fully Sharded Data
  Parallel 2 (FSDP2) without tensor-subclass weight wrappers).
- Casts activations outside the custom autograd boundary so TorchInductor Common
  Subexpression Elimination (CSE) shares one BF16 cast across sibling projections.
- Handles arbitrary token counts (`rows < 32` or `rows % 32 != 0`) for greedy decoding
  and Multi-Token Prediction (MTP) modules without compilation or alignment errors.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_torchao_mx():
    try:
        from torchao.prototype.mx_formats.config import (
            MXFP8Dim0CastKernelChoice,
            MXFP8Dim1CastKernelChoice,
            ScaleCalculationMode,
        )
        from torchao.prototype.mx_formats.mx_tensor import MXTensor
        from torchao.prototype.mx_formats.utils import _to_mxfp8_dim1_kernel_wrapper
        from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("parallel.float8='mxfp8' needs torchao >= 0.18: pip install torchao") from error
    return (
        MXTensor,
        _to_mxfp8_dim1_kernel_wrapper,
        MXFP8Dim0CastKernelChoice,
        MXFP8Dim1CastKernelChoice,
        ScaleCalculationMode,
        KernelPreference,
    )


@torch._dynamo.allow_in_graph
class _MXFP8Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_2d: torch.Tensor, weight_hp: torch.Tensor) -> torch.Tensor:
        (
            MXTensor,
            _,
            MXFP8Dim0CastKernelChoice,
            _,
            ScaleCalculationMode,
            KernelPreference,
        ) = _load_torchao_mx()

        target_dtype = x_2d.dtype
        w_bf16 = (
            weight_hp
            if (weight_hp.dtype == target_dtype and weight_hp.is_contiguous())
            else weight_hp.to(target_dtype).contiguous()
        )
        # Save the original parameter reference (`weight_hp`) rather than `w_bf16` so
        # forward does not hold duplicate BF16 weight copies across layers.
        ctx.save_for_backward(x_2d, weight_hp)
        ctx.target_dtype = target_dtype
        ctx.weight_dtype = weight_hp.dtype

        # `TORCH` compiles on Blackwell to inline Parallel Thread Execution (PTX)
        # `cvt.rp.satfinite.ue8m0x2.f32` and falls back cleanly in eager mode.
        x_mx = MXTensor.to_mx(
            x_2d,
            torch.float8_e4m3fn,
            32,
            ScaleCalculationMode.RCEIL,
            KernelPreference.AUTO,
            mxfp8_dim0_cast_kernel_choice=MXFP8Dim0CastKernelChoice.TORCH,
        )
        w_mx = MXTensor.to_mx(
            w_bf16,
            torch.float8_e4m3fn,
            32,
            ScaleCalculationMode.RCEIL,
            KernelPreference.AUTO,
            mxfp8_dim0_cast_kernel_choice=MXFP8Dim0CastKernelChoice.TORCH,
        )
        return torch.mm(x_mx, w_mx.t())

    @staticmethod
    def backward(ctx, grad_out_2d: torch.Tensor):
        (
            MXTensor,
            _to_mxfp8_dim1_kernel_wrapper,
            MXFP8Dim0CastKernelChoice,
            MXFP8Dim1CastKernelChoice,
            ScaleCalculationMode,
            KernelPreference,
        ) = _load_torchao_mx()

        x_2d, weight_hp = ctx.saved_tensors
        target_dtype = ctx.target_dtype
        w_bf16 = (
            weight_hp
            if (weight_hp.dtype == target_dtype and weight_hp.is_contiguous())
            else weight_hp.to(target_dtype).contiguous()
        )
        dy_2d = (
            grad_out_2d
            if (grad_out_2d.dtype == target_dtype and grad_out_2d.is_contiguous())
            else grad_out_2d.to(target_dtype).contiguous()
        )

        dy_mx_dim0 = MXTensor.to_mx(
            dy_2d,
            torch.float8_e4m3fn,
            32,
            ScaleCalculationMode.RCEIL,
            KernelPreference.AUTO,
            mxfp8_dim0_cast_kernel_choice=MXFP8Dim0CastKernelChoice.TORCH,
        )
        # Compute Unified Device Architecture (CUDA) kernel quantizes along dim 1 without
        # materializing an explicit high-precision transpose.
        w_mx_dim1 = _to_mxfp8_dim1_kernel_wrapper(
            w_bf16,
            32,
            torch.float8_e4m3fn,
            target_dtype,
            KernelPreference.AUTO,
            MXFP8Dim1CastKernelChoice.CUDA,
            ScaleCalculationMode.RCEIL,
        )
        grad_input = torch.mm(dy_mx_dim0, w_mx_dim1.t())

        dy_mx_dim1 = _to_mxfp8_dim1_kernel_wrapper(
            dy_2d,
            32,
            torch.float8_e4m3fn,
            target_dtype,
            KernelPreference.AUTO,
            MXFP8Dim1CastKernelChoice.CUDA,
            ScaleCalculationMode.RCEIL,
        )
        x_t_mx_dim0 = _to_mxfp8_dim1_kernel_wrapper(
            x_2d,
            32,
            torch.float8_e4m3fn,
            target_dtype,
            KernelPreference.AUTO,
            MXFP8Dim1CastKernelChoice.CUDA,
            ScaleCalculationMode.RCEIL,
        ).t()
        grad_weight = torch.mm(dy_mx_dim1, x_t_mx_dim0)
        if grad_weight.dtype != ctx.weight_dtype:
            grad_weight = grad_weight.to(ctx.weight_dtype)
        return grad_input, grad_weight


class MXFP8Linear(nn.Linear):
    """Drop-in `nn.Linear` replacement running 32-element block-scaled MXFP8 GEMMs."""

    @classmethod
    def from_float(cls, mod: nn.Linear) -> MXFP8Linear:
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=mod.bias is not None)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        rows = x_2d.shape[0]
        # Short decode / prompt prefills (`rows < 32`) are memory-bound; run standard
        # linear directly and avoid 32-row MX block alignment requirements.
        if rows < 32 or (not torch.is_grad_enabled() and rows % 32 != 0):
            return F.linear(x, self.weight, self.bias)

        target_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        if target_dtype == torch.float32:
            target_dtype = torch.bfloat16
        x_2d = x_2d.to(target_dtype).contiguous()

        pad_rows = (-rows) % 32
        if pad_rows:
            x_2d = F.pad(x_2d, (0, 0, 0, pad_rows))
        out = _MXFP8Matmul.apply(x_2d, self.weight)
        if pad_rows:
            out = out[:rows]
        out = out.reshape(*orig_shape[:-1], out.shape[-1])
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out


def convert_to_mxfp8_training(
    module: nn.Module,
    *,
    module_filter_fn: Callable[[nn.Module, str], bool] | None = None,
    _prefix: str = "",
) -> int:
    """Swap eligible `nn.Linear` modules in place for `MXFP8Linear`; returns count."""
    _load_torchao_mx()
    if torch.cuda.is_available() and torch.cuda.get_device_capability() < (10, 0):
        major, minor = torch.cuda.get_device_capability()
        raise RuntimeError(
            f"parallel.float8='mxfp8' requires NVIDIA Blackwell compute capability >= 10.0, got {major}.{minor}"
        )
    swapped = 0
    for name, child in list(module.named_children()):
        fqn = f"{_prefix}.{name}" if _prefix else name
        if (
            type(child) is nn.Linear
            and child.in_features % 32 == 0
            and child.out_features % 32 == 0
            and (module_filter_fn is None or module_filter_fn(child, fqn))
        ):
            setattr(module, name, MXFP8Linear.from_float(child))
            swapped += 1
        else:
            swapped += convert_to_mxfp8_training(child, module_filter_fn=module_filter_fn, _prefix=fqn)
    return swapped
