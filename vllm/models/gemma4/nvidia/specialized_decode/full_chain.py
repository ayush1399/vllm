# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guarded SM90 Gemma4 full-chain producer."""

import torch

from vllm.utils.torch_utils import direct_register_custom_op


def native_available() -> bool:
    try:
        return all(
            torch._C._dispatch_has_kernel_for_dispatch_key(name, "CUDA")
            for name in (
                "_C::gemma4_full_chain_out",
                "_C::gemma4_attention_prep_out",
                "_C::gemma4_attention_prep_cache_out",
                "_C::gemma4_combine_quant_out",
                "_C::gemma4_gelu_quant_out",
                "_C::gemma4_gelu_mul_out",
            )
        )
    except RuntimeError:
        return False


def allocate(x, mode, compact_carry=False):
    m, h = x.shape
    if m <= 0 or h != 5376 or mode not in (0, 1, 2, 5):
        raise ValueError("Gemma4 full chain requires positive M, H5376 and mode0/1/2/5")
    carry_shape = (0,) if mode == 5 else (m, h)
    carry = torch.empty(
        carry_shape,
        device=x.device,
        dtype=torch.bfloat16 if compact_carry and mode == 2 else torch.float32,
    )
    output = torch.empty(
        (m, h),
        device=x.device,
        dtype=torch.bfloat16 if mode == 5 else torch.float8_e4m3fn,
    )
    storage = torch.empty(
        (h // 128, (m + 3) // 4 * 4), device=x.device, dtype=torch.float32
    )
    return carry, output, storage


def full_chain(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_weight: torch.Tensor,
    pre_weight: torch.Tensor,
    scalar: torch.Tensor,
    mode: int,
    epsilon: float,
    compact_carry: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.device.type != "cuda":
        raise ValueError("Gemma4 full chain requires CUDA tensors")
    m, h = x.shape
    assert h == 5376 and mode in (0, 1, 2, 5) and epsilon == 1e-6
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    assert residual.shape == x.shape and residual.is_contiguous()
    assert residual.dtype == (
        torch.bfloat16 if mode == 0 or (compact_carry and mode == 1) else torch.float32
    )
    assert post_weight.shape == pre_weight.shape == (h,)
    assert post_weight.dtype == pre_weight.dtype == scalar.dtype == torch.bfloat16
    assert post_weight.is_contiguous() and pre_weight.is_contiguous()
    assert scalar.numel() == 1 and scalar.is_contiguous()
    assert all(
        t.device == x.device for t in (residual, post_weight, pre_weight, scalar)
    )
    carry, output, storage = allocate(x, mode, compact_carry)
    small = m < 32 and mode != 5
    native_mode = mode + 7 if small else mode
    if compact_carry and mode in (1, 2):
        native_mode += 10
    with torch.accelerator.device_index(x.device.index):
        torch.ops._C.gemma4_full_chain_out(
            x,
            residual,
            post_weight,
            pre_weight,
            scalar,
            carry,
            output,
            storage,
            native_mode,
            epsilon,
        )
    return carry, output, storage[:, :m].t()


def fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_weight: torch.Tensor,
    pre_weight: torch.Tensor,
    scalar: torch.Tensor,
    mode: int,
    epsilon: float,
    compact_carry: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    carry, output, storage = allocate(x, mode, compact_carry)
    return carry, output, storage[:, : x.shape[0]].t()


direct_register_custom_op(
    "gemma4_full_chain",
    full_chain,
    fake_impl=fake,
    tags=(torch.Tag.needs_exact_strides,),
)
