# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM90 normalization/GEMM chains with independent weight staging."""

import ctypes
import functools
import hashlib
import importlib.util
from pathlib import Path

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


@functools.cache
def _library():
    from torch.utils.cpp_extension import load

    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("FlashInfer headers are unavailable")
    data = Path(next(iter(spec.submodule_search_locations))) / "data"
    headers = data / "csrc/nv_internal/tensorrt_llm/deep_gemm"
    source = Path(__file__).parent
    paths = [
        source / name for name in ("pdl_driver.cu", "pdl_gemm.cuh", "full_chain.cuh")
    ]
    paths += [
        headers / name
        for name in ("mma_utils.cuh", "scheduler.cuh", "tma_utils.cuh", "utils.cuh")
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    digest.update(torch.__version__.encode())
    name = "gemma4_pdl_" + digest.hexdigest()[:16]
    build = Path(envs.VLLM_CACHE_ROOT) / "gemma4_pdl" / name
    build.mkdir(parents=True, exist_ok=True)
    library = load(
        name=name,
        sources=[str(source / "pdl_driver.cu")],
        extra_include_paths=[str(headers), str(data / "cutlass/include")],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-gencode=arch=compute_90a,code=sm_90a",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ],
        extra_ldflags=["-lcuda"],
        build_directory=str(build),
        is_python_module=False,
    )
    lib = ctypes.CDLL(library)
    pointer, integer = ctypes.c_void_p, ctypes.c_int
    lib.initialize_pdl.argtypes = []
    lib.initialize_pdl.restype = integer
    lib.launch_norm_gemm.argtypes = [pointer] * 11 + [integer] * 4 + [pointer]
    lib.launch_norm_gemm.restype = integer
    lib.launch_norm_gemm_split.argtypes = lib.launch_norm_gemm.argtypes
    lib.launch_norm_gemm_split.restype = integer
    return lib


@functools.cache
def initialize(device: torch.device) -> bool:
    if not current_platform.is_device_capability(90):
        return False
    try:
        with torch.accelerator.device_index(device.index):
            rc = _library().initialize_pdl()
        if rc:
            raise RuntimeError(f"CUDA kernel initialization failed ({rc})")
        _sms(device)
    except (RuntimeError, OSError) as exc:
        logger.warning("Gemma4 PDL unavailable; using existing GEMMs: %s", exc)
        return False
    return True


@functools.cache
def _sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _norm_gemm(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    pre: torch.Tensor,
    scalar: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    mode: int,
    compact_carry: bool = False,
    split_k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .full_chain import allocate, full_chain

    rows, hidden = x.shape
    if split_k not in (1, 6):
        raise ValueError("Gemma4 split K must be one or six")
    if rows <= 0 or hidden != 5376 or mode not in (0, 1, 2):
        raise ValueError("Gemma4 PDL requires positive rows, H5376 and mode 0/1/2")
    if weight.shape not in ((5376, 5376), (2048, 5376)):
        raise ValueError("Unsupported Gemma4 PDL projection")
    tensors = (x, residual, post, pre, scalar, weight, weight_scale)
    if any(t.device != x.device or not t.is_contiguous() for t in tensors):
        raise ValueError("Gemma4 PDL inputs must be contiguous on one device")
    if any(t.data_ptr() % 16 for t in (x, residual, post, pre, weight)):
        raise ValueError("Gemma4 PDL vector inputs require 16-byte alignment")
    if (
        x.dtype != torch.bfloat16
        or post.dtype != torch.bfloat16
        or pre.dtype != torch.bfloat16
        or scalar.dtype != torch.bfloat16
        or residual.dtype
        != (
            torch.bfloat16
            if mode == 0 or (compact_carry and mode == 1)
            else torch.float32
        )
        or residual.shape != x.shape
        or post.shape != (hidden,)
        or pre.shape != (hidden,)
        or scalar.numel() != 1
        or weight.dtype != torch.float8_e4m3fn
        or weight_scale.dtype != torch.float32
        or weight_scale.shape != (weight.shape[0] // 128, hidden // 128)
    ):
        raise ValueError("Invalid Gemma4 PDL normalization or block-scale inputs")
    if rows >= 32:
        carry, data, scales = full_chain(
            x, residual, post, pre, scalar, mode, 1e-6, compact_carry
        )
        out = torch.ops.vllm.flashinfer_deepgemm_blockscale_gemm_fp8_input(
            data,
            scales,
            weight,
            weight_scale,
        )
        return carry, out
    if not initialize(x.device):
        raise RuntimeError("Gemma4 PDL was not initialized")
    carry, data, scales = allocate(x, mode, compact_carry)
    use_split = split_k == 6 and rows <= 8 and weight.shape[0] == 2048
    out = (
        torch.empty((6, rows, 2048), device=x.device, dtype=torch.float32)
        if use_split
        else torch.empty((rows, weight.shape[0]), device=x.device, dtype=x.dtype)
    )
    launch = (
        _library().launch_norm_gemm_split if use_split else _library().launch_norm_gemm
    )
    with torch.accelerator.device_index(x.device.index):
        rc = launch(
            *(
                t.data_ptr()
                for t in (
                    x,
                    residual,
                    post,
                    pre,
                    scalar,
                    carry,
                    data,
                    scales,
                    weight,
                    weight_scale,
                    out,
                )
            ),
            rows,
            weight.shape[0],
            mode + 10 if compact_carry and mode in (1, 2) else mode,
            _sms(x.device),
            torch.cuda.current_stream().cuda_stream,
        )
    if rc:
        raise RuntimeError(f"Gemma4 PDL launch failed ({rc})")
    return carry, out


def norm_gemm(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    pre: torch.Tensor,
    scalar: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    mode: int,
    compact_carry: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _norm_gemm(
        x, residual, post, pre, scalar, weight, weight_scale, mode, compact_carry
    )


def norm_gemm_fake(
    x, residual, post, pre, scalar, weight, weight_scale, mode, compact_carry=False
):
    from .full_chain import allocate

    if mode not in (0, 1, 2) or weight.shape not in ((5376, 5376), (2048, 5376)):
        raise ValueError("Unsupported Gemma4 PDL mode or projection")
    carry, _, _ = allocate(x, mode, compact_carry)
    return carry, x.new_empty((x.shape[0], weight.shape[0]))


direct_register_custom_op("gemma4_norm_gemm", norm_gemm, fake_impl=norm_gemm_fake)


def norm_qkv_partials(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    pre: torch.Tensor,
    scalar: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    mode: int,
    compact_carry: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[0] > 8 or weight.shape != (2048, 5376):
        raise ValueError("QKV partitions require M1-8, N2048, K5376")
    return _norm_gemm(
        x, residual, post, pre, scalar, weight, weight_scale, mode, compact_carry, 6
    )


def norm_qkv_partials_fake(
    x, residual, post, pre, scalar, weight, weight_scale, mode, compact_carry=False
):
    from .full_chain import allocate

    carry, _, _ = allocate(x, mode, compact_carry)
    return carry, x.new_empty((6, x.shape[0], 2048), dtype=torch.float32)


direct_register_custom_op(
    "gemma4_norm_qkv_partials", norm_qkv_partials, fake_impl=norm_qkv_partials_fake
)
