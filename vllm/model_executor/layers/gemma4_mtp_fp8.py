# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-row FP8 GEMVs for the Gemma4 assistant on Hopper."""

import torch
from torch import nn

from vllm import envs
from vllm.model_executor.layers.fusion.quant_activation import get_input_quant_key
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8Dynamic128Sym
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _raw_fp8_quant(values):
    maximum = tl.max(tl.abs(values), 1)
    inverse = tl.where(maximum != 0.0, 448.0 / maximum, 1.0)
    # Match FlashInfer's reciprocal-multiply rounding, including FP8 ties.
    scale = tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[inverse],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    quant = (values * inverse[:, None]).to(tl.float8e4nv).to(tl.float32)
    return quant, scale


@triton.jit
def _fp8_gemv_kernel(
    X,
    Gamma,
    W,
    S,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    NORM: tl.constexpr,
    EPS: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    ki = tl.arange(0, K)
    x = tl.load(X + row * K + ki).to(tl.float32)
    if NORM:
        gamma = tl.load(Gamma + ki).to(tl.float32)
        inverse_rms = tl.rsqrt(tl.sum(x * x, 0) / K + EPS)
        x = ((x * inverse_rms) * gamma).to(tl.bfloat16).to(tl.float32)
    grouped = tl.reshape(x, (K // 128, 128))
    q, scales = _raw_fp8_quant(grouped)
    weight = tl.load(W + ni[:, None] * K + ki[None, :], ni[:, None] < N, 0.0)
    weight = tl.reshape(weight.to(tl.float32), (BN, K // 128, 128))
    dots = tl.sum(weight * q[None, :, :], 2)
    kg = tl.arange(0, K // 128)
    ws = tl.load(
        S + (ni[:, None] // 128) * SS0 + kg[None, :] * SS1,
        ni[:, None] < N,
        0.0,
    )
    coefficient = ws * scales[None, :]
    accum = tl.full((BN,), 0.0, tl.float32)
    # Preserve native K128 scale-promotion order without a split-K reduction.
    for group in tl.static_range(K // 128):
        index = tl.full((BN, 1), group, tl.int32)
        dot = tl.reshape(tl.gather(dots, index, axis=1), (BN,))
        scale = tl.reshape(tl.gather(coefficient, index, axis=1), (BN,))
        accum = tl.fma(dot, scale, accum)
    tl.store(Y + row * N + ni, accum, ni < N)


def fp8_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    norm_weight: torch.Tensor | None,
    epsilon: float,
) -> torch.Tensor:
    m, k = x.shape
    n = weight.shape[0]
    assert m > 0 and k in (1024, 2048) and n in (1024, 2048)
    assert x.is_cuda and x.is_contiguous() and x.dtype == torch.bfloat16
    assert weight.shape == (n, k) and weight.is_contiguous()
    assert weight.dtype == torch.float8_e4m3fn
    assert scales.shape == (n // 128, k // 128) and scales.dtype == torch.float32
    if norm_weight is not None:
        assert norm_weight.shape == (k,) and norm_weight.is_contiguous()
        assert norm_weight.dtype == x.dtype
    assert all(t.device == x.device for t in (weight, scales))
    if norm_weight is not None:
        assert norm_weight.device == x.device
    if m > 8:
        if norm_weight is None:
            return torch.ops.vllm.dynamic_flashinfer_deepgemm_blockscale_gemm(
                x, weight, scales, 128, True
            )
        data, activation_scales = torch.ops.vllm.gemma4_hybrid_rms_quant(
            x, norm_weight, epsilon
        )
        return torch.ops.vllm.flashinfer_deepgemm_blockscale_gemm_fp8_input(
            data, activation_scales, weight, scales
        )
    small_tile = 4 if m <= 5 else 8
    wide_tile = 4 if m == 1 else 8 if m <= 3 else 16
    bn = small_tile if n == 1024 else wide_tile
    warps = 8 if m <= 3 else 4
    output = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _fp8_gemv_kernel[(m, triton.cdiv(n, bn))](
        x,
        norm_weight if norm_weight is not None else x,
        weight,
        scales,
        output,
        n,
        k,
        *scales.stride(),
        norm_weight is not None,
        epsilon,
        bn,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return output


def _fp8_gemv_fake(x, weight, scales, norm_weight, epsilon):
    return x.new_empty((x.shape[0], weight.shape[0]))


direct_register_custom_op("gemma4_mtp_fp8_gemv", fp8_gemv, fake_impl=_fp8_gemv_fake)


def maybe_mtp_fp8_gemv(
    linear: nn.Module, x: torch.Tensor, norm: RMSNorm | None = None
) -> torch.Tensor | None:
    if (
        envs.VLLM_BATCH_INVARIANT
        or getattr(linear, "bias", None) is not None
        or getattr(linear, "gather_output", False)
        or get_input_quant_key(linear) != kFp8Dynamic128Sym
        or not getattr(linear, "_flashinfer_fp8_input_ue8m0", False)
        or x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[1] not in (1024, 2048)
    ):
        return None
    from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
        FP8BlockParams,
    )

    params = FP8BlockParams.from_layer(linear)
    weight, scales = params.weight, params.block_scale
    if (
        weight.shape[0] not in (1024, 2048)
        or weight.shape[1] != x.shape[1]
        or weight.dtype != torch.float8_e4m3fn
        or not weight.is_contiguous()
        or scales.dtype != torch.float32
        or scales.shape != (weight.shape[0] // 128, weight.shape[1] // 128)
    ):
        return None
    if norm is not None and (
        x.shape[1] != 1024
        or type(norm) is not RMSNorm
        or not norm.has_weight
        or not norm.pass_weight
        or norm.variance_size_override is not None
        or norm.weight.dtype != x.dtype
        or not norm.weight.is_contiguous()
        or norm.weight.shape != (x.shape[1],)
    ):
        return None
    return torch.ops.vllm.gemma4_mtp_fp8_gemv(
        x,
        weight,
        scales,
        norm.weight if norm is not None else None,
        norm.variance_epsilon if norm is not None else 1e-6,
    )


@triton.jit
def _concat_fp8_gemm_kernel(
    X,
    Z,
    W,
    S,
    Partial,
    M: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
):
    mi = tl.program_id(0) * 16 + tl.arange(0, 16)
    ni = tl.program_id(1) * 64 + tl.arange(0, 64)
    split = tl.program_id(2)
    accum = tl.full((16, 64), 0.0, tl.float32)
    for iteration in range(6):
        group = split * 6 + iteration
        ki = group * 128 + tl.arange(0, 128)
        first = tl.load(
            X + mi[:, None] * 5376 + ki[None, :],
            (mi[:, None] < M) & (ki[None, :] < 5376),
            0.0,
        ).to(tl.float32)
        second = tl.load(
            Z + mi[:, None] * 5376 + ki[None, :] - 5376,
            (mi[:, None] < M) & (ki[None, :] >= 5376),
            0.0,
        ).to(tl.float32)
        values = tl.where(ki[None, :] < 5376, first, second)
        q, scale = _raw_fp8_quant(values)
        weight = tl.load(W + ni[None, :] * 10752 + ki[:, None])
        ws = tl.load(S + (ni // 128) * SS0 + group * SS1)
        dot = tl.dot(
            q.to(tl.float8e4nv), weight, out_dtype=tl.float32, max_num_imprecise_acc=0
        )
        accum = tl.fma(scale[:, None] * ws[None, :], dot, accum)
    tl.store(
        Partial + split * M * 1024 + mi[:, None] * 1024 + ni[None, :],
        accum,
        mi[:, None] < M,
    )


@triton.jit
def _concat_fp8_gemm_reduce(Partial, Output, SIZE: tl.constexpr):
    index = tl.program_id(0) * 256 + tl.arange(0, 256)
    accum = tl.full((256,), 0.0, tl.float32)
    for split in range(14):
        accum += tl.load(Partial + split * SIZE + index, index < SIZE, 0.0)
    tl.store(Output + index, accum, index < SIZE)


def concat_fp8_gemm(
    x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Fuse concatenation/quantization/projection with FP32 split-K partials."""
    m, h = x.shape
    assert 0 < m <= 20 and h == 5376
    assert x.shape == z.shape and x.dtype == z.dtype == torch.bfloat16
    assert x.is_contiguous() and z.is_contiguous() and x.is_cuda
    assert weight.shape == (1024, 10752) and weight.is_contiguous()
    assert weight.dtype == torch.float8_e4m3fn
    assert scales.shape == (8, 84) and scales.dtype == torch.float32
    assert all(t.device == x.device for t in (z, weight, scales))
    partial = torch.empty((14, m, 1024), device=x.device, dtype=torch.float32)
    output = x.new_empty((m, 1024))
    _concat_fp8_gemm_kernel[(triton.cdiv(m, 16), 16, 14)](
        x,
        z,
        weight,
        scales,
        partial,
        m,
        *scales.stride(),
        num_warps=4,
        enable_fp_fusion=False,
    )
    _concat_fp8_gemm_reduce[(triton.cdiv(m * 1024, 256),)](
        partial,
        output,
        m * 1024,
    )
    return output


def _concat_fp8_gemm_fake(x, z, weight, scales):
    return x.new_empty((x.shape[0], 1024))


direct_register_custom_op(
    "gemma4_mtp_concat_fp8_gemm", concat_fp8_gemm, fake_impl=_concat_fp8_gemm_fake
)


def pre_projection(
    x: torch.Tensor,
    z: torch.Tensor,
    full_weight: torch.Tensor,
    full_scales: torch.Tensor,
    local_weight: torch.Tensor,
    local_scales: torch.Tensor,
) -> torch.Tensor:
    # Dispatch here: model warmup traces a large batch before small decode graphs.
    if x.shape[0] <= 20:
        return concat_fp8_gemm(x, z, full_weight, full_scales)
    from vllm.distributed import tensor_model_parallel_all_gather

    combined = torch.cat([x, z], dim=-1)
    partial = torch.ops.vllm.dynamic_flashinfer_deepgemm_blockscale_gemm(
        combined, local_weight, local_scales, 128, True
    )
    return tensor_model_parallel_all_gather(partial, dim=-1)


def _pre_projection_fake(x, z, full_weight, full_scales, local_weight, local_scales):
    return x.new_empty((x.shape[0], full_weight.shape[0]))


direct_register_custom_op(
    "gemma4_mtp_pre_projection", pre_projection, fake_impl=_pre_projection_fake
)
