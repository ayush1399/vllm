# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 normalization and FP8 activation producers."""

import torch

from vllm import envs
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8Dynamic128Sym
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import direct_register_custom_op


def preserve_bf16_intermediates() -> bool:
    """Match native execution or Inductor's optional precision casts."""
    if not torch.compiler.is_compiling():
        return True
    from vllm.config import get_current_vllm_config

    config = get_current_vllm_config().compilation_config
    return config.backend != "inductor" or bool(
        config.inductor_compile_config.get("emulate_precision_casts", False)
    )


def _gemma4_hybrid_rms_quant(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    m, hidden = x.shape
    result = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale_storage = torch.empty(
        (hidden // 128, round_up(m, 4)), device=x.device, dtype=torch.float32
    )
    scales = scale_storage[:, :m].t()
    if m < 32:
        from flashinfer.gemm.gemm_base import get_fp8_blockscale_gemm_runner_sm90

        normalized = torch.empty_like(x)
        torch.ops._C.rms_norm(normalized, x, weight, epsilon)
        get_fp8_blockscale_gemm_runner_sm90().fp8_quantize_1x128(
            normalized, result, scale_storage, False
        )
    else:
        torch.ops._C.rms_norm_per_block_quant(
            result, x, weight, scales, epsilon, None, None, 128, True, True
        )
    return result, scales


def _gemma4_hybrid_rms_quant_fake(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    m, hidden = x.shape
    result = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale_storage = torch.empty(
        (hidden // 128, round_up(m, 4)), device=x.device, dtype=torch.float32
    )
    return result, scale_storage[:, :m].t()


direct_register_custom_op(
    "gemma4_hybrid_rms_quant",
    _gemma4_hybrid_rms_quant,
    fake_impl=_gemma4_hybrid_rms_quant_fake,
    tags=(torch.Tag.needs_exact_strides,),
)


def _maybe_gemma4_hybrid_norm_quant(
    norm: RMSNorm, x: torch.Tensor, linear: torch.nn.Module
) -> torch.Tensor | QuantizedActivation:
    key = get_input_quant_key(linear)
    if (
        key != kFp8Dynamic128Sym
        or not getattr(linear, "_flashinfer_fp8_input_ue8m0", False)
        or envs.VLLM_BATCH_INVARIANT
        or type(norm) is not RMSNorm
        or not norm.has_weight
        or norm.variance_size_override is not None
        or x.ndim != 2
        or x.dtype != torch.bfloat16
        or norm.weight.dtype != x.dtype
        or not x.is_contiguous()
        or x.shape[-1] % 128 != 0
    ):
        return norm(x)
    data, scale = torch.ops.vllm.gemma4_hybrid_rms_quant(
        x, norm.weight, norm.variance_epsilon
    )
    return QuantizedActivation(
        data=data,
        scale=scale,
        orig_dtype=x.dtype,
        orig_shape=x.shape,
        quant_key=key,
    )
