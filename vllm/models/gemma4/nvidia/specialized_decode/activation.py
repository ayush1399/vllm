# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GELU/gating/FP8 production for the target's small verification batches."""

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    FP8BlockParams,
)
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.fusion.quant_activation import (
    get_input_quant_key,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8Dynamic128Sym
from vllm.model_executor.models.gemma4 import Gemma4MLP
from vllm.utils.torch_utils import direct_register_custom_op

from . import linear as _linear  # noqa: F401
from .norm import preserve_bf16_intermediates


def gelu_quant(
    x: torch.Tensor, round_gate: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    q, storage = allocate(x)
    torch.ops._C.gemma4_gelu_quant_out(x, q, storage, round_gate)
    return q, storage[:, : x.shape[0]].t()


def allocate(x):
    rows, width = x.shape
    width //= 2
    q = x.new_empty((rows, width), dtype=torch.float8_e4m3fn)
    storage = torch.empty(
        (width // 128, (rows + 3) // 4 * 4), device=x.device, dtype=torch.float32
    )
    return q, storage


def gelu_quant_fake(x, round_gate=True):
    q, storage = allocate(x)
    return q, storage[:, : x.shape[0]].t()


direct_register_custom_op(
    "gemma4_gelu_quant",
    gelu_quant,
    fake_impl=gelu_quant_fake,
    tags=(torch.Tag.needs_exact_strides,),
)


def gelu_down_projection(
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    round_gate: bool = True,
) -> torch.Tensor:
    rows = gate_up.shape[0]
    if rows == 0:
        return gate_up.new_empty((0, weight.shape[0]))
    data, scale = gelu_quant(gate_up, round_gate)
    return torch.ops.vllm.flashinfer_deepgemm_blockscale_gemm_fp8_input(
        data, scale, weight, weight_scale
    )


def gelu_down_projection_fake(gate_up, weight, weight_scale, round_gate=True):
    return gate_up.new_empty((gate_up.shape[0], weight.shape[0]))


direct_register_custom_op(
    "gemma4_gelu_down_projection",
    gelu_down_projection,
    fake_impl=gelu_down_projection_fake,
)


class SpecializedGemma4MLP(Gemma4MLP):
    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        return self.forward_gate_up(gate_up)

    def forward_gate_up(self, gate_up):
        if (
            gate_up.ndim == 2
            and gate_up.dtype == torch.bfloat16
            and gate_up.is_contiguous()
            and gate_up.shape[-1] % 256 == 0
            and type(self.act_fn) is GeluAndMul
            and self.act_fn.approximate == "tanh"
            and get_input_quant_key(self.down_proj) == kFp8Dynamic128Sym
            and getattr(self.down_proj, "_flashinfer_fp8_input_ue8m0", False)
            and self.down_proj.bias is None
            and self.down_proj.input_is_parallel
        ):
            params = FP8BlockParams.from_layer(self.down_proj)
            output = torch.ops.vllm.gemma4_gelu_down_projection(
                gate_up,
                params.weight,
                params.block_scale,
                self.act_fn.enabled()
                or self.act_fn._enforce_enable
                or preserve_bf16_intermediates(),
            )
            if self.down_proj.reduce_results and self.down_proj.tp_size > 1:
                output = tensor_model_parallel_all_reduce(output)
            return output
        return self.down_proj(self.act_fn(gate_up))[0]
