# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Specialized target execution over the stock Gemma4 backbone."""

from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8Dynamic128Sym
from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer

from . import full_chain  # noqa: F401
from .norm import _maybe_gemma4_hybrid_norm_quant

if TYPE_CHECKING:
    from vllm.model_executor.models.gemma4 import Gemma4Model


class SpecializedGemma4DecoderLayer(Gemma4DecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fuse_norm_quant = (
            not self.enable_moe_block and not self.self_attn.is_kv_shared_layer
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        per_layer_input: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = _maybe_gemma4_hybrid_norm_quant(
            self.input_layernorm, residual, self.self_attn.qkv_proj
        )
        hidden_states = self.self_attn(
            positions=positions, hidden_states=hidden_states, **kwargs
        )
        hidden_states = self.post_attention_layernorm(hidden_states) + residual
        residual = hidden_states
        hidden_states = _maybe_gemma4_hybrid_norm_quant(
            self.pre_feedforward_layernorm, hidden_states, self.mlp.gate_up_proj
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states) + residual
        return hidden_states * self.layer_scalar, None


def _gemma4_full_chain_norm_supported(norm: RMSNorm, device) -> bool:
    return (
        type(norm) is RMSNorm
        and norm.has_weight
        and norm.pass_weight
        and norm.variance_size_override is None
        and norm.hidden_size == 5376
        and norm.variance_epsilon == 1e-6
        and norm.weight.dtype == torch.bfloat16
        and norm.weight.shape == (5376,)
        and norm.weight.is_contiguous()
        and norm.weight.device == device
    )


def _gemma4_full_chain_linear_supported(linear: nn.Module) -> bool:
    return (
        get_input_quant_key(linear) == kFp8Dynamic128Sym
        and getattr(linear, "_flashinfer_fp8_input_ue8m0", False) is True
    )


def _gemma4_can_run_full_chain(
    model: "Gemma4Model",
    hidden_states: torch.Tensor,
    per_layer_inputs: torch.Tensor | None,
) -> bool:
    from vllm.model_executor.models.gemma4 import Gemma4Attention, Gemma4MLP

    if (
        not model._full_chain_config_supported
        or model.fast_prefill_enabled
        or model.aux_hidden_state_layers
        or per_layer_inputs is not None
        or model.start_layer != 0
        or model.end_layer != 60
        or len(model.layers) != 60
        or hidden_states.ndim != 2
        or hidden_states.shape[-1] != 5376
        or hidden_states.shape[0] == 0
        or hidden_states.dtype != torch.bfloat16
        or hidden_states.device.type != "cuda"
        or not hidden_states.is_contiguous()
        or not _gemma4_full_chain_norm_supported(model.norm, hidden_states.device)
    ):
        return False
    for layer in model.layers:
        if (
            type(layer) is not SpecializedGemma4DecoderLayer
            or type(layer.self_attn) is not Gemma4Attention
            or type(layer.mlp) is not Gemma4MLP
            or not layer._fuse_norm_quant
            or layer.enable_moe_block
            or layer.self_attn.is_kv_shared_layer
            or layer.per_layer_input_gate is not None
            or layer.per_layer_projection is not None
            or layer.post_per_layer_input_norm is not None
            or not _gemma4_full_chain_linear_supported(layer.self_attn.qkv_proj)
            or not _gemma4_full_chain_linear_supported(layer.mlp.gate_up_proj)
            or layer.layer_scalar.dtype != torch.bfloat16
            or layer.layer_scalar.numel() != 1
            or not layer.layer_scalar.is_contiguous()
            or layer.layer_scalar.device != hidden_states.device
        ):
            return False
        for norm in (
            layer.input_layernorm,
            layer.post_attention_layernorm,
            layer.pre_feedforward_layernorm,
            layer.post_feedforward_layernorm,
        ):
            if not _gemma4_full_chain_norm_supported(norm, hidden_states.device):
                return False
    return True


def _gemma4_full_chain_activation(
    data: torch.Tensor, scale: torch.Tensor
) -> QuantizedActivation:
    return QuantizedActivation(
        data=data,
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=data.shape,
        quant_key=kFp8Dynamic128Sym,
    )


def _gemma4_run_full_chain(
    model: "Gemma4Model",
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    first = model.layers[0]
    activation = _maybe_gemma4_hybrid_norm_quant(
        first.input_layernorm, hidden_states, first.self_attn.qkv_proj
    )
    assert isinstance(activation, QuantizedActivation)
    carry = hidden_states
    for index, layer in enumerate(model.layers):
        attention_out = layer.self_attn(
            positions=positions, hidden_states=activation, **kwargs
        )
        residual, data, scale = torch.ops.vllm.gemma4_full_chain(
            attention_out,
            carry,
            layer.post_attention_layernorm.weight,
            layer.pre_feedforward_layernorm.weight,
            layer.layer_scalar,
            0 if index == 0 else 1,
            1e-6,
        )
        feedforward_out = layer.mlp(_gemma4_full_chain_activation(data, scale))
        if index + 1 == len(model.layers):
            _, normalized, _ = torch.ops.vllm.gemma4_full_chain(
                feedforward_out,
                residual,
                layer.post_feedforward_layernorm.weight,
                model.norm.weight,
                layer.layer_scalar,
                5,
                1e-6,
            )
            return normalized
        next_layer = model.layers[index + 1]
        carry, data, scale = torch.ops.vllm.gemma4_full_chain(
            feedforward_out,
            residual,
            layer.post_feedforward_layernorm.weight,
            next_layer.input_layernorm.weight,
            layer.layer_scalar,
            2,
            1e-6,
        )
        activation = _gemma4_full_chain_activation(data, scale)
    raise AssertionError("Full-chain eligibility requires 60 decoder layers")
