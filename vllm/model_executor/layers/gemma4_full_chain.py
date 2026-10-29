# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guarded SM90 Gemma4 full-chain producer."""

import torch

from vllm.utils.torch_utils import direct_register_custom_op


def native_available() -> bool:
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(
            "_C::gemma4_full_chain_out", "CUDA"
        )
    except RuntimeError:
        return False


def allocate(x, mode):
    m, h = x.shape
    if m <= 0 or h != 5376 or mode not in (0, 1, 2, 5):
        raise ValueError("Gemma4 full chain requires positive M, H5376 and mode0/1/2/5")
    carry_shape = (0,) if mode == 5 else (m, h)
    carry = torch.empty(carry_shape, device=x.device, dtype=torch.float32)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.device.type != "cuda":
        raise ValueError("Gemma4 full chain requires CUDA tensors")
    m, h = x.shape
    assert h == 5376 and mode in (0, 1, 2, 5) and epsilon == 1e-6
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    assert residual.shape == x.shape and residual.is_contiguous()
    assert residual.dtype == (torch.bfloat16 if mode == 0 else torch.float32)
    assert post_weight.shape == pre_weight.shape == (h,)
    assert post_weight.dtype == pre_weight.dtype == scalar.dtype == torch.bfloat16
    assert post_weight.is_contiguous() and pre_weight.is_contiguous()
    assert scalar.numel() == 1 and scalar.is_contiguous()
    assert all(
        t.device == x.device for t in (residual, post_weight, pre_weight, scalar)
    )
    carry, output, storage = allocate(x, mode)
    small = m < 32 and mode != 5
    norm = torch.empty_like(x) if small else output
    native_mode = (6 if mode == 2 else mode + 3) if small else mode
    with torch.accelerator.device_index(x.device.index):
        torch.ops._C.gemma4_full_chain_out(
            x,
            residual,
            post_weight,
            pre_weight,
            scalar,
            carry,
            norm,
            storage,
            native_mode,
            epsilon,
        )
        if small:
            from flashinfer.gemm.gemm_base import get_fp8_blockscale_gemm_runner_sm90

            get_fp8_blockscale_gemm_runner_sm90().fp8_quantize_1x128(
                norm, output, storage, False
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    carry, output, storage = allocate(x, mode)
    return carry, output, storage[:, : x.shape[0]].t()


direct_register_custom_op(
    "gemma4_full_chain",
    full_chain,
    fake_impl=fake,
    tags=(torch.Tag.needs_exact_strides,),
)
