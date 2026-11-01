# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused per-head normalization and RoPE for Gemma4 target and assistant."""

from typing import cast

import torch

from vllm import ir
from vllm.model_executor.models.gemma4 import Gemma4Attention
from vllm.model_executor.models.gemma4_mtp import Gemma4MTPAttention
from vllm.utils.torch_utils import direct_register_custom_op

from .norm import preserve_bf16_intermediates


def prepare(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    round_before_weight: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = prepare_fake(
        qkv,
        q_weight,
        k_weight,
        cache,
        positions,
        q_heads,
        kv_heads,
        head_dim,
        round_before_weight,
    )
    if qkv.shape[0] == 0:
        return q, k, v
    torch.ops._C.gemma4_attention_prep_out(
        qkv,
        q_weight,
        k_weight,
        cache,
        positions,
        q,
        k,
        v,
        q_heads,
        kv_heads,
        head_dim,
        1e-6,
        round_before_weight,
    )
    return q, k, v


def prepare_fake(
    qkv,
    q_weight,
    k_weight,
    cache,
    positions,
    q_heads,
    kv_heads,
    head_dim,
    round_before_weight=False,
):
    rows = qkv.shape[0]
    q = qkv.new_empty((rows, q_heads * head_dim), dtype=torch.bfloat16)
    k = qkv.new_empty((rows, kv_heads * head_dim))
    return q, k, torch.empty_like(k)


direct_register_custom_op("gemma4_attention_prep", prepare, fake_impl=prepare_fake)


def supported(attention, positions):
    cache = attention.rotary_emb.cos_sin_cache
    priority = ir.ops.rms_norm.get_priority()
    return (
        bool(priority)
        and priority[0] in ("native", "vllm_c")
        and attention.head_dim in (256, 512)
        and attention.q_norm.variance_epsilon == 1e-6
        and attention.rotary_emb.is_neox_style
        and cache.dtype == torch.bfloat16
        and cache.shape[1] == attention.head_dim
        and cache.is_contiguous()
        and positions.ndim == 1
        and positions.dtype == torch.int64
        and positions.is_contiguous()
    )


def prepare_attention(attention, qkv, positions, kv_heads):
    return torch.ops.vllm.gemma4_attention_prep(
        qkv,
        attention.q_norm.weight,
        attention.k_norm.weight if kv_heads else attention.q_norm.weight,
        attention.rotary_emb.cos_sin_cache,
        positions,
        attention.num_heads,
        kv_heads,
        attention.head_dim,
        ir.ops.rms_norm.get_priority()[0] == "native" and preserve_bf16_intermediates(),
    )


def prepare_cached(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    round_before_weight: bool,
    layer_name: str,
) -> torch.Tensor:
    from vllm.model_executor.layers.attention.attention import get_attention_context

    _, _, kv_cache, slots = get_attention_context(layer_name)
    rows = qkv.shape[1] if qkv.ndim == 3 else qkv.shape[0]
    q = qkv.new_empty((rows, q_heads * head_dim), dtype=torch.bfloat16)
    if slots is None or kv_cache.numel() == 0:
        if qkv.ndim == 3:
            qkv = qkv.sum(dim=0).bfloat16()
        return prepare(
            qkv,
            q_weight,
            k_weight,
            cache,
            positions,
            q_heads,
            kv_heads,
            head_dim,
            round_before_weight,
        )[0]
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)
    torch.ops._C.gemma4_attention_prep_cache_out(
        qkv,
        q_weight,
        k_weight,
        cache,
        positions,
        slots,
        q,
        key_cache,
        value_cache,
        q_heads,
        kv_heads,
        head_dim,
        round_before_weight,
    )
    return q


def prepare_cached_fake(
    qkv,
    q_weight,
    k_weight,
    cache,
    positions,
    q_heads,
    kv_heads,
    head_dim,
    round_before_weight,
    layer_name,
):
    rows = qkv.shape[1] if qkv.ndim == 3 else qkv.shape[0]
    return qkv.new_empty((rows, q_heads * head_dim), dtype=torch.bfloat16)


direct_register_custom_op(
    "gemma4_attention_prep_cached", prepare_cached, fake_impl=prepare_cached_fake
)


def attention_fp8(
    query: torch.Tensor, layer_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.attention.attention import get_attention_context
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    metadata, layer, kv_cache, _ = get_attention_context(layer_name)
    rows, width = query.shape
    data = query.new_empty(query.shape, dtype=torch.float8_e4m3fn)
    storage = torch.empty(
        (width // 128, (rows + 3) // 4 * 4), device=query.device, dtype=torch.float32
    )
    if metadata is None:
        return data, storage[:, :rows].t()
    impl = cast(FlashAttentionImpl, layer.impl)
    q = query.view(rows, layer.num_heads, layer.head_size)
    if (
        rows < 32
        and metadata.max_seq_len >= 1024
        and impl.vllm_flash_attn_version == 4
        and isinstance(metadata.causal, bool)
    ):
        from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

        k, v = kv_cache.transpose(1, 2).split(layer.head_size, dim=-1)
        window = impl.sliding_window or (-1, -1)
        actual = metadata.num_actual_tokens
        _flash_attn_fwd(
            q[:actual],
            k,
            v,
            out=data[:actual].view(actual, layer.num_heads, layer.head_size),
            output_block_scale=storage,
            cu_seqlens_q=metadata.query_start_loc,
            max_seqlen_q=metadata.max_query_len,
            max_seqlen_k=metadata.max_seq_len,
            seqused_k=metadata.seq_lens,
            page_table=metadata.block_table,
            softmax_scale=impl.scale,
            causal=metadata.causal,
            window_size_left=window[0] if window[0] >= 0 else None,
            window_size_right=window[1] if window[1] >= 0 else None,
            num_splits=-1,
        )
    else:
        output = torch.empty_like(q)
        impl.forward(layer, q, None, None, kv_cache, metadata, output)
        output = output.view(rows, width)
        if rows < 32:
            from flashinfer.gemm.gemm_base import get_fp8_blockscale_gemm_runner_sm90

            get_fp8_blockscale_gemm_runner_sm90().fp8_quantize_1x128(
                output, data, storage, False
            )
        else:
            data, scales = per_token_group_quant_fp8(
                output,
                128,
                column_major_scales=True,
                tma_aligned_scales=True,
                use_ue8m0=True,
            )
            return data, scales
    return data, storage[:, :rows].t()


def attention_fp8_fake(query, layer_name):
    rows, width = query.shape
    data = query.new_empty(query.shape, dtype=torch.float8_e4m3fn)
    storage = torch.empty(
        (width // 128, (rows + 3) // 4 * 4), device=query.device, dtype=torch.float32
    )
    return data, storage[:, :rows].t()


direct_register_custom_op(
    "gemma4_attention_fp8",
    attention_fp8,
    fake_impl=attention_fp8_fake,
    tags=(torch.Tag.needs_exact_strides,),
)


def norm_qkv_prep(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    pre: torch.Tensor,
    scalar: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope: torch.Tensor,
    positions: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    round_before_weight: bool,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .pdl import _norm_gemm

    carry, qkv = _norm_gemm(
        x,
        residual,
        post,
        pre,
        scalar,
        weight,
        weight_scale,
        2,
        True,
        6 if x.shape[0] <= 8 else 1,
    )
    q = prepare_cached(
        qkv,
        q_weight,
        k_weight,
        rope,
        positions,
        q_heads,
        kv_heads,
        head_dim,
        round_before_weight,
        layer_name,
    )
    return carry, q


def norm_qkv_prep_fake(
    x,
    residual,
    post,
    pre,
    scalar,
    weight,
    weight_scale,
    q_weight,
    k_weight,
    rope,
    positions,
    q_heads,
    kv_heads,
    head_dim,
    round_before_weight,
    layer_name,
):
    return x.new_empty(x.shape), x.new_empty((x.shape[0], q_heads * head_dim))


direct_register_custom_op(
    "gemma4_norm_qkv_prep", norm_qkv_prep, fake_impl=norm_qkv_prep_fake
)


def can_prepare_cached(attention):
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    return (
        type(attention.attn.impl) is FlashAttentionImpl
        and attention.attn.impl.dcp_world_size == 1
        and attention.attn.kv_cache_dtype in ("auto", "bfloat16")
    )


class SpecializedGemma4Attention(Gemma4Attention):
    def forward(
        self, positions, hidden_states, projected_qkv=None, prepared_q=None, **kwargs
    ):
        if prepared_q is not None:
            return self.project_attention(prepared_q)
        if self.is_kv_shared_layer or not supported(self, positions):
            assert projected_qkv is None
            return super().forward(positions, hidden_states, **kwargs)
        qkv = (
            self.qkv_proj(hidden_states)[0] if projected_qkv is None else projected_qkv
        )
        if can_prepare_cached(self):
            q = torch.ops.vllm.gemma4_attention_prep_cached(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rotary_emb.cos_sin_cache,
                positions,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                ir.ops.rms_norm.get_priority()[0] == "native"
                and preserve_bf16_intermediates(),
                self.attn.layer_name,
            )
            return self.project_attention(q)
        q, k, v = prepare_attention(self, qkv, positions, self.num_kv_heads)
        return self.o_proj(self.attn(q, k, v))[0]

    def project_attention(self, q):
        from vllm.model_executor.layers.fusion.quant_activation import (
            QuantizedActivation,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8Dynamic128Sym,
        )

        if getattr(self.o_proj, "_flashinfer_fp8_input_ue8m0", False):
            data, scales = torch.ops.vllm.gemma4_attention_fp8(q, self.attn.layer_name)
            activation = QuantizedActivation(
                data=data,
                scale=scales,
                orig_dtype=torch.bfloat16,
                orig_shape=data.shape,
                quant_key=kFp8Dynamic128Sym,
            )
            return self.o_proj(activation)[0]
        return self.o_proj(self.attn(q, None, None))[0]


class SpecializedGemma4MTPAttention(Gemma4MTPAttention):
    def forward(self, positions, hidden_states, projected_q=None, **kwargs):
        if not supported(self, positions):
            return super().forward(positions, hidden_states, projected_q, **kwargs)
        qkv = self.q_proj(hidden_states)[0] if projected_q is None else projected_q
        q, _, _ = prepare_attention(self, qkv, positions, 0)
        return self.o_proj(self.attn(q, None, None))[0]
