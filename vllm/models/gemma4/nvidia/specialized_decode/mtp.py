# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Assistant fusions and replicated pre-projection preparation."""

import torch

from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    FP8BlockParams,
)
from vllm.model_executor.layers.fusion.quant_activation import get_input_quant_key
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8Dynamic128Sym
from vllm.model_executor.models.gemma4_mtp import (
    Gemma4MTPDecoderLayer,
    Gemma4MultiTokenPredictor,
)
from vllm.sequence import IntermediateTensors

from .mtp_kernels import maybe_mtp_fp8_gemv
from .norm import _maybe_gemma4_hybrid_norm_quant


class SpecializedGemma4MTPDecoderLayer(Gemma4MTPDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fuse_norm_quant = True
        self.self_attn.o_proj._gemma4_mtp_fp8_gemv = True
        self.mlp.down_proj._gemma4_mtp_fp8_gemv = True

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        projected_q = None
        if self._fuse_norm_quant:
            projected_q = maybe_mtp_fp8_gemv(
                self.self_attn.q_proj, residual, self.input_layernorm
            )
            if projected_q is None:
                hidden_states = _maybe_gemma4_hybrid_norm_quant(
                    self.input_layernorm, residual, self.self_attn.q_proj
                )
        else:
            hidden_states = self.input_layernorm(residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            projected_q=projected_q,
            **kwargs,
        )

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states

        gate_up = None
        if self._fuse_norm_quant:
            gate_up = maybe_mtp_fp8_gemv(
                self.mlp.gate_up_proj, hidden_states, self.pre_feedforward_layernorm
            )
            if gate_up is None:
                hidden_states = _maybe_gemma4_hybrid_norm_quant(
                    self.pre_feedforward_layernorm, hidden_states, self.mlp.gate_up_proj
                )
        else:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        if gate_up is None:
            hidden_states = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp.act_fn(gate_up)
            hidden_states, _ = self.mlp.down_proj(hidden_states)

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states, None


def prepare_pre_projection(self) -> None:
    self.model._mtp_pre_weight = None
    self.model._mtp_pre_scales = None
    linear = self.model.pre_projection
    if (
        not self._fuse_pre_projection
        or get_input_quant_key(linear) != kFp8Dynamic128Sym
        or not getattr(linear, "_flashinfer_fp8_input_ue8m0", False)
        or self.model.backbone_hidden_size != 5376
        or self.model.hidden_size != 1024
    ):
        return
    params = FP8BlockParams.from_layer(linear)
    if (
        params.weight.shape != (128, 10752)
        or not params.weight.is_cuda
        or not params.weight.is_contiguous()
        or not params.block_scale.is_cuda
        or params.weight.dtype != torch.float8_e4m3fn
        or params.block_scale.shape != (1, 84)
        or params.block_scale.dtype != torch.float32
    ):
        return
    self.model._mtp_pre_scale_name = params.block_scale_attr
    # Replicate the small projection once, removing its per-draft all-gather.
    self.model._mtp_pre_weight = tensor_model_parallel_all_gather(
        params.weight.view(torch.uint8).contiguous(), dim=0
    ).view(torch.float8_e4m3fn)
    self.model._mtp_pre_scales = tensor_model_parallel_all_gather(
        params.block_scale.contiguous(), dim=0
    )


class SpecializedGemma4MultiTokenPredictor(Gemma4MultiTokenPredictor):
    decoder_layer_cls = SpecializedGemma4MTPDecoderLayer

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.register_buffer("_mtp_pre_weight", None, persistent=False)
        self.register_buffer("_mtp_pre_scales", None, persistent=False)
        self._mtp_pre_scale_name = "weight_scale"

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (draft_hidden_states, backbone_hidden_states).

        draft_hidden_states: draft-dim, used by compute_logits via lm_head.
        backbone_hidden_states: backbone-dim, stored in the proposer's
            hidden-state buffer and fed back as input to the next step.
        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)

        if (
            self._mtp_pre_weight is not None
            and hidden_states.dtype == inputs_embeds.dtype == torch.bfloat16
            and hidden_states.shape == inputs_embeds.shape
            and hidden_states.is_contiguous()
            and inputs_embeds.is_contiguous()
        ):
            hidden_states = torch.ops.vllm.gemma4_mtp_pre_projection(
                inputs_embeds,
                hidden_states,
                self._mtp_pre_weight,
                self._mtp_pre_scales,
                self.pre_projection.weight,
                getattr(self.pre_projection, self._mtp_pre_scale_name),
            )
        else:
            combined = torch.cat([inputs_embeds, hidden_states], dim=-1)
            hidden_states, _ = self.pre_projection(combined)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        draft_hidden_states = self.norm(hidden_states)

        backbone_hidden_states, _ = self.post_projection(draft_hidden_states)
        return draft_hidden_states, backbone_hidden_states
