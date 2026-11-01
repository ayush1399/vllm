# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.bfloat16, torch.float16]
IS_NEOX = [True, False]
EPS_VALUES = [1e-5, 1e-6]
SEEDS = [13]
PARTIAL_ROPE = [True, False]
CUDA_DEVICES = ["cuda:0"]


def _apply_qk_norm_rope(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    rope: RotaryEmbedding,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
) -> torch.Tensor:
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim

    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_dim, head_dim)
    q_by_head = q_norm.forward_native(q_by_head)
    assert isinstance(q_by_head, torch.Tensor)
    q = q_by_head.view(q.shape)

    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // head_dim, head_dim)
    k_by_head = k_norm.forward_native(k_by_head)
    assert isinstance(k_by_head, torch.Tensor)
    k = k_by_head.view(k.shape)

    q, k = rope.forward_native(positions, q, k)
    return torch.cat([q, k, v], dim=-1)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_qk_norm_rope custom op requires cuda and rocm platform",
)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_neox", IS_NEOX)
@pytest.mark.parametrize("eps", EPS_VALUES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("rotary_ratio", [1.0, 0.5, 0.25])
@torch.inference_mode()
def test_fused_qk_norm_rope_matches_reference(
    default_vllm_config,
    device: str,
    dtype: torch.dtype,
    is_neox: bool,
    eps: float,
    seed: int,
    rotary_ratio: float,
):
    torch.set_default_device(device)
    set_random_seed(seed)
    num_heads, num_kv_heads, head_dim = 16, 4, 128
    num_tokens = 4

    total_dim = (num_heads + 2 * num_kv_heads) * head_dim
    qkv_base = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
    qkv_fused = qkv_base.clone()
    positions = torch.arange(num_tokens, dtype=torch.long, device=device)

    q_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    k_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    q_norm.weight.data.normal_(mean=1.0, std=0.1)
    k_norm.weight.data.normal_(mean=1.0, std=0.1)
    q_weight = q_norm.weight.data
    k_weight = k_norm.weight.data
    rotary_dim = int(head_dim * rotary_ratio)
    rope = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=rotary_dim,
        max_position_embeddings=4096,
        base=10000.0,
        is_neox_style=is_neox,
        dtype=dtype,
    ).to(device)

    ref_result = _apply_qk_norm_rope(
        qkv=qkv_base,
        positions=positions,
        q_norm=q_norm,
        k_norm=k_norm,
        rope=rope,
        num_heads_q=num_heads,
        num_heads_kv=num_kv_heads,
        head_dim=head_dim,
    )

    opcheck(
        torch.ops._C.fused_qk_norm_rope,
        (
            qkv_fused.clone(),
            num_heads,
            num_kv_heads,
            num_kv_heads,
            head_dim,
            eps,
            q_weight,
            k_weight,
            rope.cos_sin_cache,
            is_neox,
            positions.view(-1),
        ),
    )

    torch.ops._C.fused_qk_norm_rope(
        qkv_fused,
        num_heads,
        num_kv_heads,
        num_kv_heads,
        head_dim,
        eps,
        q_weight,
        k_weight,
        rope.cos_sin_cache,
        is_neox,
        positions.view(-1),
    )

    if dtype == torch.float16:
        ATOL, RTOL = (2e-3, 2e-3)
    else:
        ATOL, RTOL = (1e-2, 1e-2)

    torch.testing.assert_close(
        qkv_fused,
        ref_result,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.skipif(
    not current_platform.is_device_capability(90),
    reason="Gemma4 preparation is SM90-only",
)
@pytest.mark.parametrize("num_tokens", [1, 5, 20, 128, 4096])
@pytest.mark.parametrize(
    "head_dim,num_kv_heads", [(256, 0), (256, 2), (512, 0), (512, 1)]
)
@pytest.mark.parametrize("round_before_weight", [False, True])
@torch.inference_mode()
def test_gemma4_preparation_preserves_native_qkv(
    default_vllm_config, num_tokens, head_dim, num_kv_heads, round_before_weight
):
    """Fuse Q/K/V norm and padded proportional RoPE without changing BF16 outputs."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.rotary_embedding.gemma4_rope import (
        Gemma4RotaryEmbedding,
    )
    from vllm.models.gemma4.nvidia.specialized_decode.attention import prepare

    set_random_seed(123)
    num_heads = 4
    dtype = torch.bfloat16
    qkv = torch.randn(
        (num_tokens, (num_heads + 2 * num_kv_heads) * head_dim),
        device="cuda",
        dtype=dtype,
    )
    q_weight = torch.randn(head_dim, device="cuda", dtype=dtype)
    k_weight = torch.randn_like(q_weight)
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int64) + 17
    rope = Gemma4RotaryEmbedding(
        head_dim,
        head_dim if head_dim == 256 else 128,
        max(1024, num_tokens + 32),
        10000,
        True,
        dtype,
    ).to("cuda")
    q, k, v = qkv.split(
        [num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1
    )
    expected_q = torch.empty_like(q)
    expected_k = torch.empty_like(k)
    expected_v = torch.empty_like(v)
    ops.rms_norm(
        expected_q.view(num_tokens, num_heads, head_dim),
        q.view(num_tokens, num_heads, head_dim),
        torch.ones_like(q_weight) if round_before_weight else q_weight,
        1e-6,
    )
    if round_before_weight:
        expected_q.view(num_tokens, num_heads, head_dim).mul_(q_weight)
    if num_kv_heads:
        ops.rms_norm(
            expected_k.view(num_tokens, num_kv_heads, head_dim),
            k.view(num_tokens, num_kv_heads, head_dim),
            torch.ones_like(k_weight) if round_before_weight else k_weight,
            1e-6,
        )
        if round_before_weight:
            expected_k.view(num_tokens, num_kv_heads, head_dim).mul_(k_weight)
        ops.rms_norm(
            expected_v.view(num_tokens, num_kv_heads, head_dim),
            v.view(num_tokens, num_kv_heads, head_dim),
            None,
            1e-6,
        )
    ops.rotary_embedding(
        positions,
        expected_q,
        expected_k if num_kv_heads else None,
        head_dim,
        rope.cos_sin_cache,
        True,
    )
    actual = prepare(
        qkv,
        q_weight,
        k_weight,
        rope.cos_sin_cache,
        positions,
        num_heads,
        num_kv_heads,
        head_dim,
        round_before_weight,
    )
    for output, expected in zip(actual, (expected_q, expected_k, expected_v)):
        assert torch.equal(output, expected)


@pytest.mark.skipif(
    not current_platform.is_device_capability(90),
    reason="Gemma4 cache fusion is SM90-only",
)
@pytest.mark.parametrize("num_tokens", [1, 5, 20, 128, 4096])
@pytest.mark.parametrize("head_dim,kv_heads", [(256, 2), (512, 1)])
@pytest.mark.parametrize("layout", ["NHD", "HND", "packed"])
@pytest.mark.parametrize("partitions", [1, 6])
@torch.inference_mode()
def test_gemma4_preparation_writes_paged_cache_after_graph_replay(
    num_tokens, head_dim, kv_heads, layout, partitions
):
    """Preserve Q and cached KV under changed inputs, slots, and padded rows."""
    from vllm import _custom_ops as ops
    from vllm.models.gemma4.nvidia.specialized_decode.attention import prepare

    torch.manual_seed(714)
    q_heads, page_size = 4, 16
    pages = (num_tokens + 15) // 16 + 2
    x = torch.randn(
        num_tokens,
        (q_heads + 2 * kv_heads) * head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    if partitions == 6:
        x = torch.randn(6, *x.shape, device="cuda", dtype=torch.float32)
    qw = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    kw = torch.randn_like(qw)
    rope = torch.randn(num_tokens + 64, head_dim, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int64) + 17
    slots = torch.randperm(pages * page_size, device="cuda", dtype=torch.int64)[
        :num_tokens
    ].contiguous()
    slots[::3] = -1
    if num_tokens > 1:
        slots = slots[:-1]
    shape = (pages, page_size, kv_heads, head_dim)
    if layout == "HND":
        cache = torch.randn(
            pages, kv_heads, page_size, head_dim, device="cuda", dtype=torch.bfloat16
        ).transpose(1, 2)
    else:
        cache = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    packed = None
    if layout == "packed":
        packed = torch.randn(
            pages,
            kv_heads,
            page_size,
            2 * head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        key_cache, value_cache = packed.transpose(1, 2).split(head_dim, dim=-1)
    else:
        key_cache, value_cache = cache.clone(), cache.clone()
    q = x.new_empty((num_tokens, q_heads * head_dim), dtype=torch.bfloat16)
    scale = torch.ones(1, device="cuda")
    args = (
        x,
        qw,
        kw,
        rope,
        positions,
        slots,
        q,
        key_cache,
        value_cache,
        q_heads,
        kv_heads,
        head_dim,
        False,
    )
    torch.ops._C.gemma4_attention_prep_cache_out(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops._C.gemma4_attention_prep_cache_out(*args)
    for _ in range(3):
        x.normal_()
        slots.copy_(slots.roll(1))
        if packed is not None:
            expected_packed = packed.clone()
            expected_k, expected_v = expected_packed.transpose(1, 2).split(
                head_dim, dim=-1
            )
        else:
            expected_k, expected_v = key_cache.clone(), value_cache.clone()
        reference = x
        if partitions == 6:
            reference = torch.zeros_like(x[0])
            for part in x.unbind(0):
                reference.add_(part)
            reference = reference.bfloat16()
        expected_q, k, v = prepare(
            reference, qw, kw, rope, positions, q_heads, kv_heads, head_dim
        )
        ops.reshape_and_cache_flash(
            k[: slots.numel()].view(-1, kv_heads, head_dim),
            v[: slots.numel()].view(-1, kv_heads, head_dim),
            expected_k,
            expected_v,
            slots,
            "auto",
            scale,
            scale,
        )
        graph.replay()
        assert torch.equal(q, expected_q)
        assert torch.equal(key_cache, expected_k)
        assert torch.equal(value_cache, expected_v)
