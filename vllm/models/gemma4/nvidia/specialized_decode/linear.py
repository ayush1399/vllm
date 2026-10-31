# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8-input GEMM consumer and assistant GEMV dispatch for Gemma4."""

import torch

from vllm import envs
from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
    FlashInferFp8DeepGEMMDynamicBlockScaledKernel,
)
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
)
from vllm.utils.deep_gemm import fp8_gemm_nt
from vllm.utils.flashinfer import flashinfer_fp8_blockscale_gemm
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import direct_register_custom_op


class SpecializedFp8BlockScaledKernel(FlashInferFp8DeepGEMMDynamicBlockScaledKernel):
    def input_quant_key(self) -> QuantKey:
        return kFp8Dynamic128Sym

    def process_weights_after_loading(self, layer: torch.nn.Module):
        super().process_weights_after_loading(layer)
        layer._flashinfer_fp8_input_ue8m0 = self.fallback.use_deep_gemm_e8m0

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if getattr(layer, "_gemma4_mtp_fp8_gemv", False) and isinstance(
            x, torch.Tensor
        ):
            from vllm.models.gemma4.nvidia.specialized_decode.mtp_kernels import (
                maybe_mtp_fp8_gemv,
            )

            output = maybe_mtp_fp8_gemv(layer, x)
            if output is not None:
                return output if bias is None else output + bias
        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is None:
            assert isinstance(x, torch.Tensor)
            return super().apply_weights(layer, x, bias, **kwargs)

        params = self._get_layer_params(layer)
        weight = params.weight
        k = qa.data.shape[-1]
        output = torch.ops.vllm.flashinfer_deepgemm_blockscale_gemm_fp8_input(
            qa.data.view(-1, k),
            qa.scale.view(-1, k // self.weight_group_shape.col),
            weight,
            params.block_scale,
        )
        if bias is not None:
            output = output + bias
        output_shape = [*qa.orig_shape[:-1], weight.shape[0]]
        return output.to(dtype=self.config.out_dtype).view(*output_shape)


def _check_fp8_input_layout(input: torch.Tensor, input_scale: torch.Tensor) -> None:
    """Both GEMMs read input_scale through a raw pointer / TMA descriptor, so
    reject anything but the (M, K // 128) fp32 column-major TMA-aligned layout."""
    m, k = input.shape
    if input.dtype != torch.float8_e4m3fn or not input.is_contiguous():
        raise ValueError(
            f"FP8 input must be contiguous float8_e4m3fn, got {input.dtype} "
            f"with strides {input.stride()}"
        )
    if (
        input_scale.dtype != torch.float32
        or input_scale.shape != (m, k // 128)
        or input_scale.stride(1) != round_up(m, 4)
        or (m > 1 and input_scale.stride(0) != 1)
    ):
        raise ValueError(
            "FP8 input scale must be float32 with shape "
            f"{(m, k // 128)} and strides (1, {round_up(m, 4)}), got "
            f"{input_scale.dtype} {tuple(input_scale.shape)} "
            f"strides {input_scale.stride()}"
        )


def _flashinfer_deepgemm_blockscale_gemm_fp8_input_impl(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """FP8-input variant of dynamic_flashinfer_deepgemm_blockscale_gemm.

    Takes activations already quantized to FP8 with 1x128 fp32 scales in the
    column-major TMA-aligned layout both kernels consume directly.
    """
    _check_fp8_input_layout(input, input_scale)

    def run_flashinfer_deepgemm_swapAB(
        input: torch.Tensor,
        input_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        return flashinfer_fp8_blockscale_gemm(
            input=input,
            weight=weight,
            input_scale=input_scale,
            weight_scale=weight_scale,
            out_dtype=torch.bfloat16,
        )

    def run_deepgemm(
        input: torch.Tensor,
        input_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.empty(
            (input.shape[0], weight.shape[0]),
            dtype=torch.bfloat16,
            device=input.device,
        )
        fp8_gemm_nt(
            (input, input_scale),
            (weight, weight_scale),
            output,
            is_deep_gemm_e8m0_used=False,
        )
        return output

    if envs.VLLM_BATCH_INVARIANT:
        return run_deepgemm(input, input_scale, weight, weight_scale)

    return torch.cond(
        input.shape[0] < 32,
        run_flashinfer_deepgemm_swapAB,
        run_deepgemm,
        (input, input_scale, weight, weight_scale),
    )


def _flashinfer_deepgemm_blockscale_gemm_fp8_input_fake(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        input.shape[0], weight.shape[0], dtype=torch.bfloat16, device=input.device
    )


direct_register_custom_op(
    "flashinfer_deepgemm_blockscale_gemm_fp8_input",
    _flashinfer_deepgemm_blockscale_gemm_fp8_input_impl,
    fake_impl=_flashinfer_deepgemm_blockscale_gemm_fp8_input_fake,
    # Inductor must not repack the TMA-aligned input_scale strides.
    tags=(torch.Tag.needs_exact_strides,),
)
