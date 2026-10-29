# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import itertools

import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.utils import fp8_allclose, fp8_ulp_distance, opcheck
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_group_quant_int8,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

ON_GFX950 = False
if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx950

    ON_GFX950 = on_gfx950()

DTYPES = [torch.bfloat16, torch.float]
QUANT_DTYPES = [torch.int8, current_platform.fp8_dtype()]
VEC_HIDDEN_SIZES = [1024, 1025, 1027, 1029]
# Avoid combinatorial explosion with full Cartesian product
NUM_TOKENS_HIDDEN_SIZES = [
    *[(1, i) for i in [1, 64, 128, *VEC_HIDDEN_SIZES, 4096, 5120, 5137, 8192]],
    *[(7, i) for i in [128, 4096, 8192]],
    *[(2048, i) for i in [1, 64, *VEC_HIDDEN_SIZES, 4096, 5137]],
    *[(4096, i) for i in [1, 64, 4096, 5137]],
]

ADD_RESIDUAL = [False, True]
SCALE_UBS = [True, False]
GROUP_SIZES = [[1, 64], [1, 128]]
TMA_ALIGNMENTS = [0, 4]
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]

EPS = 1e-6


def _is_valid_config(
    num_tokens: int,
    hidden_size: int,
    has_scale_ub: bool,
    quant_dtype: torch.dtype,
    group_size: list[int] | None,
    tma_alignment: int,
) -> bool:
    if group_size is not None and hidden_size % group_size[1] != 0:
        return False
    if group_size is not None and has_scale_ub:
        return False
    if (
        group_size is None or quant_dtype != current_platform.fp8_dtype()
    ) and tma_alignment != 0:
        return False
    if (
        group_size is not None
        and tma_alignment != 0
        and num_tokens % tma_alignment == 0
    ):
        return False
    return not (has_scale_ub and quant_dtype != current_platform.fp8_dtype())


def _config_id(
    num_tokens: int,
    hidden_size: int,
    has_scale_ub: bool,
    quant_dtype: torch.dtype,
    group_size: list[int] | None,
    tma_alignment: int,
) -> str:
    quant = str(quant_dtype).removeprefix("torch.")
    group = "per-token" if group_size is None else f"group-{group_size[1]}"
    return (
        f"{num_tokens}x{hidden_size}-{quant}-{group}-"
        f"tma-{tma_alignment}-scale-ub-{has_scale_ub}"
    )


# Filter unsupported combinations during collection. Letting each case call
# pytest.skip still runs the global per-test teardown and dominates this suite.
RMS_NORM_CONFIGS = [
    pytest.param(
        num_tokens,
        hidden_size,
        has_scale_ub,
        quant_dtype,
        group_size,
        tma_alignment,
        id=_config_id(
            num_tokens,
            hidden_size,
            has_scale_ub,
            quant_dtype,
            group_size,
            tma_alignment,
        ),
    )
    for (
        (num_tokens, hidden_size),
        has_scale_ub,
        quant_dtype,
        (group_size, tma_alignment),
    ) in itertools.product(
        NUM_TOKENS_HIDDEN_SIZES,
        SCALE_UBS,
        QUANT_DTYPES,
        [(None, 0), *itertools.product(GROUP_SIZES, TMA_ALIGNMENTS)],
    )
    if _is_valid_config(
        num_tokens,
        hidden_size,
        has_scale_ub,
        quant_dtype,
        group_size,
        tma_alignment,
    )
]

## Helpers


def as_float32_tensor(x: float | torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device="cuda")


def ref_rms_norm(
    rms_norm_layer: RMSNorm, x: torch.Tensor, residual: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if residual is not None:
        residual = residual.clone()
        out, residual = rms_norm_layer.forward_native(x, residual)
    else:
        out = rms_norm_layer.forward_native(x)

    return out, residual


def ref_dynamic_per_token_or_block_quant(
    rms_norm_layer: RMSNorm,
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    residual: torch.Tensor | None,
    scale_ub: torch.Tensor | None,
    group_size: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if scale_ub is not None:
        assert quant_dtype == current_platform.fp8_dtype()

    # Norm
    torch_out, residual = ref_rms_norm(rms_norm_layer, x, residual)

    # Quant
    if group_size is not None:
        if quant_dtype == current_platform.fp8_dtype():
            torch_out, scales = per_token_group_quant_fp8(
                torch_out, group_size=group_size[1], use_ue8m0=False
            )
        else:
            assert quant_dtype == torch.int8
            torch_out, scales = per_token_group_quant_int8(
                torch_out, group_size=group_size[1]
            )
    else:
        if quant_dtype == current_platform.fp8_dtype():
            torch_out, scales = ops.scaled_fp8_quant(
                torch_out, scale_ub=scale_ub, use_per_token_if_dynamic=True
            )
        else:
            assert quant_dtype == torch.int8
            torch_out, scales, _ = ops.scaled_int8_quant(torch_out)

    return torch_out, scales, residual


def ref_impl(
    rms_norm_layer: RMSNorm,
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    residual: torch.Tensor | None,
    scale_ub: torch.Tensor | None,
    group_size: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    return ref_dynamic_per_token_or_block_quant(
        rms_norm_layer, x, quant_dtype, residual, scale_ub, group_size
    )


def ops_dynamic_per_token_or_block_quant(
    weight: torch.Tensor,
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    residual: torch.Tensor | None,
    scale_ub: torch.Tensor | None,
    group_size: list[int] | None,
    tma_alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if residual is not None:
        residual = residual.clone()
    if group_size is not None:
        out, scales = ops.rms_norm_per_block_quant(
            x,
            weight,
            EPS,
            quant_dtype,
            group_size,
            scale_ub,
            residual,
            True,
            tma_alignment,
        )
        scales = scales.contiguous()
    else:
        out, scales = ops.rms_norm_dynamic_per_token_quant(
            x, weight, EPS, quant_dtype, scale_ub, residual
        )
    return out, scales, residual


def ops_impl(
    weight: torch.Tensor,
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    residual: torch.Tensor | None,
    scale_ub: torch.Tensor | None,
    group_size: list[int] | None,
    tma_alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    return ops_dynamic_per_token_or_block_quant(
        weight, x, quant_dtype, residual, scale_ub, group_size, tma_alignment
    )


@pytest.mark.parametrize(
    (
        "num_tokens",
        "hidden_size",
        "has_scale_ub",
        "quant_dtype",
        "group_size",
        "tma_alignment",
    ),
    RMS_NORM_CONFIGS,
)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
@torch.inference_mode()
def test_rms_norm(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    has_scale_ub: bool,
    dtype: torch.dtype,
    quant_dtype: torch.dtype,
    group_size: list[int] | None,
    tma_alignment: int,
    seed: int,
    device: str,
    strided_input: bool,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)

    layer = RMSNorm(hidden_size, EPS).to(dtype=dtype)

    # Make weights
    layer.weight.data.normal_(mean=1.0, std=0.1)

    # Make inputs: use a wider tensor and slice to create a non-contiguous
    # (strided) input when strided_input=True. The last dimension stride
    # remains 1, which the kernel requires.
    scale = 1 / (hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x = torch.randn(num_tokens, last_dim, dtype=dtype) * scale
    x = x[:, :hidden_size]

    # dim 1 gets special-cased
    x_is_strided = strided_input and num_tokens != 1
    # check that the input is strided iff we expect it to be
    assert x.is_contiguous() != x_is_strided

    # Residual must still be contiguous
    residual = (
        torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
        if add_residual
        else None
    )
    if has_scale_ub:
        rms_x, _ = ref_rms_norm(layer, x, residual)
        scale_ub = torch.mean(rms_x).to(dtype=torch.float32, device="cuda")
    else:
        scale_ub = None

    ref_out, ref_scales, ref_residual = ref_impl(
        layer, x, quant_dtype, residual, scale_ub, group_size
    )
    ops_out, ops_scales, ops_residual = ops_impl(
        layer.weight, x, quant_dtype, residual, scale_ub, group_size, tma_alignment
    )

    assert ref_out.dtype == quant_dtype
    assert ops_out.dtype == quant_dtype

    # Per-block bf16 scales: allow a small relative tolerance for a few groups
    # whose abs-max flips by one ULP between the fused and reference paths. The
    # per-token and fp32 paths stay strict.
    # The same one-ULP group-scale flip also occurs on CUDA (H200), so extend
    # the block relaxation there too — the fused groupwise reduction rounds
    # differently than the reference for isolated groups.
    relax_block = (
        group_size is not None
        and dtype == torch.bfloat16
        and (current_platform.is_rocm() or current_platform.is_cuda())
    )
    use_gfx950_fp8_allclose = (
        current_platform.is_rocm()
        and ON_GFX950
        and group_size is None
        and dtype == torch.bfloat16
        and quant_dtype == current_platform.fp8_dtype()
    )
    allow_cuda_fp8_rounding_outliers = (
        current_platform.is_cuda()
        and group_size is None
        and dtype == torch.bfloat16
        and quant_dtype == current_platform.fp8_dtype()
    )

    def scales_close(rtol: float, atol: float) -> bool:
        if torch.allclose(ref_scales, ops_scales, rtol=rtol, atol=atol):
            return True
        return relax_block and torch.allclose(
            ref_scales, ops_scales, rtol=1e-2, atol=atol
        )

    if quant_dtype == torch.int8:
        assert scales_close(rtol=1e-5, atol=1e-6)
        # big atol to account for round-off errors.
        assert torch.allclose(ref_out, ops_out, atol=1)
    else:
        assert scales_close(rtol=1e-5, atol=1e-8)
        a = ref_out.to(dtype=torch.float32)
        b = ops_out.to(dtype=torch.float32)
        ok = torch.allclose(a, b, atol=1e-6)
        if not ok:
            if relax_block:
                # ULP-flipped group scale can cross an E4M3 tie; tolerate a
                # bounded count of isolated fp8 outliers.
                ulp = fp8_ulp_distance(ref_out, ops_out)
                max_outliers = ulp.numel() // 100_000 + 8
                ok = int((ulp > 0).sum().item()) <= max_outliers
            elif use_gfx950_fp8_allclose:
                # Valid gfx950 reduction trees can straddle an E4M3 boundary.
                ok = fp8_allclose(ops_out, ref_out, rtol=0.125, atol=2e-3)
                ok = ok and int(fp8_ulp_distance(ops_out, ref_out).max()) <= 1
            elif allow_cuda_fp8_rounding_outliers:
                # A valid BF16 reduction can cross an E4M3 boundary for
                # isolated values.
                ulp = fp8_ulp_distance(ref_out, ops_out)
                max_outliers = ulp.numel() // 100_000 + 8
                ok = int(ulp.max()) <= 1
                ok = ok and int((ulp > 0).sum().item()) <= max_outliers
            else:
                # Compare dequantized values with relaxed tolerance.
                if group_size is None:
                    a_deq = a * ref_scales.view(-1, 1)
                    b_deq = b * ops_scales.view(-1, 1)
                else:
                    a_deq = a * ref_scales.repeat_interleave(group_size[1], dim=1)
                    b_deq = b * ops_scales.repeat_interleave(group_size[1], dim=1)
                # NOTE: It is possible that some future test cases trigger this
                # max diff due to precision issues. If such an error is
                # encountered, it's recommended to inspect the differences between
                # all corresponding elements from each tensor (e.g. by looping over
                # them) and checking how many the max diff error shows up on (just
                # a few bad elements should still be considered acceptable).
                ok = torch.allclose(a_deq, b_deq, rtol=5e-2, atol=5e-2)
        assert ok
    if add_residual:
        assert torch.allclose(ref_residual, ops_residual)

    output = torch.empty(x.shape, dtype=quant_dtype, device=x.device)
    if group_size is None:
        scales = torch.empty(
            (x.numel() // x.shape[-1], 1), device=x.device, dtype=torch.float32
        )
        opcheck(
            torch.ops._C.rms_norm_dynamic_per_token_quant,
            (output, x, layer.weight, scales, 1e-5, scale_ub, residual),
        )
    else:
        assert hidden_size % group_size[1] == 0
        num_groups = hidden_size // group_size[1]
        scales = torch.empty(
            (num_groups, num_tokens),
            device=x.device,
            dtype=torch.float32,
        ).transpose(0, 1)
        opcheck(
            torch.ops._C.rms_norm_per_block_quant,
            (
                output,
                x,
                layer.weight,
                scales,
                1e-5,
                scale_ub,
                residual,
                group_size[1],
                True,  # is_scale_transposed
            ),
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA UE8 contract")
@pytest.mark.parametrize("num_tokens", [33, 80, 800, 8320])
@pytest.mark.parametrize("add_residual", [False, True])
@pytest.mark.parametrize("tma_alignment", [0, 4])
@pytest.mark.parametrize(
    "input_case", ["zero", "tiny", "power2-boundaries", "rounding"]
)
@torch.inference_mode()
def test_rms_group128_ue8_cuda_contract(
    default_vllm_config, num_tokens, add_residual, tma_alignment, input_case
):
    """Check rounded-value scales, real TMA pitch, residual output and guards.

    Structured norm inputs isolate the UE8 quant contract. They do not establish
    equality of arbitrary RMS reduction trees or model quality.
    """
    device = "cuda:0"
    hidden_size, groups, guard = 5376, 42, 16
    layer = RMSNorm(hidden_size, EPS).to(device=device, dtype=torch.bfloat16)
    x = torch.full(
        (num_tokens, hidden_size),
        0.0 if input_case == "zero" else 0.25 if add_residual else 1.0,
        dtype=torch.bfloat16,
        device=device,
    )
    residual = None
    if add_residual:
        residual = torch.full_like(x, 0.0 if input_case == "zero" else 0.75)
    if input_case == "rounding":
        pattern = torch.tensor([0.5, 1.0, 2.0, 4.0], device=device, dtype=x.dtype)
        pattern = pattern.repeat(hidden_size // 4)
        x.copy_(pattern * (0.25 if add_residual else 1.0))
        if add_residual:
            assert residual is not None
            residual.copy_(pattern * 0.75)
    if input_case == "tiny":
        weights = torch.full((groups,), 1e-12, device=device, dtype=x.dtype)
    elif input_case == "power2-boundaries":
        base = torch.tensor([112.0, 224.0, 448.0], device=device, dtype=x.dtype)
        down = torch.nextafter(base, torch.zeros_like(base))
        up = torch.nextafter(base, torch.full_like(base, float("inf")))
        weights = torch.stack((down, base, up)).flatten().repeat(5)[:groups]
        weights[1::2].neg_()
    elif input_case == "rounding":
        weights = torch.tensor(
            [0.51171875, -1.0078125, 1.015625], device=device, dtype=x.dtype
        ).repeat(groups // 3)
    else:
        weights = torch.ones(groups, device=device, dtype=x.dtype)
    layer.weight.copy_(weights.repeat_interleave(128))
    original_x, original_weight = x.clone(), layer.weight.clone()
    expected, expected_residual = ref_rms_norm(layer, x, residual)
    ref_q, ref_s = per_token_group_quant_fp8(
        expected,
        128,
        column_major_scales=True,
        tma_aligned_scales=bool(tma_alignment),
        use_ue8m0=True,
    )
    # Independent scale reference starts AFTER both native BF16 roundings.
    amax = expected.float().view(num_tokens, groups, 128).abs().amax(-1)
    raw = amax.clamp_min(1e-10) / 448.0
    formula_s = torch.exp2(torch.ceil(torch.log2(raw.abs().clamp_min(1e-10))))
    assert torch.equal(ref_s, formula_s)
    pitch = num_tokens if tma_alignment == 0 else (num_tokens + 3) // 4 * 4
    scale_storage = torch.full(
        (groups * pitch + 2 * guard,),
        -12345.0,
        device=device,
        dtype=torch.float32,
    )
    scales = torch.as_strided(
        scale_storage,
        (num_tokens, groups),
        (1, pitch),
        guard,
    )
    quant_storage = torch.full(
        (num_tokens * hidden_size + 2 * guard,),
        37,
        device=device,
        dtype=torch.uint8,
    )
    out = quant_storage[guard:-guard].view(current_platform.fp8_dtype()).view_as(x)
    residual_out = residual.clone() if residual is not None else None
    torch.ops._C.rms_norm_per_block_quant(
        out,
        x,
        layer.weight,
        scales,
        EPS,
        None,
        residual_out,
        128,
        True,
        True,
    )
    assert scales.stride() == (1, pitch)
    assert torch.equal(scales, ref_s)
    assert torch.equal(out.view(torch.uint8), ref_q.view(torch.uint8))
    actual_deq = out.float() * scales.repeat_interleave(128, -1)
    expected_deq = ref_q.float() * ref_s.repeat_interleave(128, -1)
    assert torch.equal(actual_deq, expected_deq)
    assert torch.equal(x, original_x)
    assert torch.equal(layer.weight, original_weight)
    if add_residual:
        assert torch.equal(residual_out, expected_residual)
    assert (quant_storage[:guard] == 37).all()
    assert (quant_storage[-guard:] == 37).all()
    assert (scale_storage[:guard] == -12345.0).all()
    assert (scale_storage[-guard:] == -12345.0).all()
    if pitch != num_tokens:
        physical = scale_storage[guard:-guard].view(groups, pitch)
        assert (physical[:, num_tokens:] == -12345.0).all()
    wrapped_q, wrapped_s = ops.rms_norm_per_block_quant(
        x,
        layer.weight,
        EPS,
        current_platform.fp8_dtype(),
        [1, 128],
        residual=residual.clone() if residual is not None else None,
        is_scale_transposed=True,
        tma_alignment=tma_alignment,
        scale_ue8m0=True,
    )
    assert wrapped_s.stride() == (1, pitch)
    assert torch.equal(wrapped_s, scales)
    assert torch.equal(wrapped_q.view(torch.uint8), out.view(torch.uint8))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA UE8 contract")
@torch.inference_mode()
def test_rms_group128_ue8_default_preserves_legacy(default_vllm_config):
    """Omitted schema/wrapper argument and explicit False preserve legacy values."""
    x = torch.ones((33, 5376), device="cuda:0", dtype=torch.bfloat16)
    weight = torch.full_like(x[0], 1e-12)
    legacy_q, legacy_s = ops.rms_norm_per_block_quant(
        x,
        weight,
        EPS,
        current_platform.fp8_dtype(),
        [1, 128],
        is_scale_transposed=True,
        tma_alignment=4,
    )
    false_q, false_s = ops.rms_norm_per_block_quant(
        x,
        weight,
        EPS,
        current_platform.fp8_dtype(),
        [1, 128],
        is_scale_transposed=True,
        tma_alignment=4,
        scale_ue8m0=False,
    )
    assert torch.equal(legacy_q.view(torch.uint8), false_q.view(torch.uint8))
    assert torch.equal(legacy_s, false_s)
    output = torch.empty_like(legacy_q)
    scales = torch.empty_strided(
        legacy_s.shape, legacy_s.stride(), device=x.device, dtype=torch.float32
    )
    torch.ops._C.rms_norm_per_block_quant(
        output,
        x,
        weight,
        scales,
        EPS,
        None,
        None,
        128,
        True,
    )
    assert torch.equal(output.view(torch.uint8), legacy_q.view(torch.uint8))
    assert torch.equal(scales, legacy_s)


# Contract: SM90 H5376 postnorm/add/prenorm with distinct live carry and
# group128 scales. Exact constant inputs expose mode, scalar and layout errors;
# direct native calls also exercise host guards before any vectorized launch.
@pytest.mark.skipif(
    not current_platform.is_device_capability(90), reason="Gemma4 chain is SM90-only"
)
@pytest.mark.parametrize("num_tokens", [31, 32, 33, 128, 136, 800])
@pytest.mark.parametrize("mode", [0, 1, 2, 5])
def test_gemma4_full_chain_native_stream_and_outputs(num_tokens, mode):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.zeros((num_tokens, 5376), device="cuda", dtype=torch.bfloat16)
        residual = torch.full_like(
            x, 2, dtype=torch.bfloat16 if mode == 0 else torch.float32
        )
        weight = torch.ones(5376, device="cuda", dtype=torch.bfloat16)
        scalar = torch.tensor([0.5], device="cuda", dtype=torch.bfloat16)
        carry = torch.empty(
            (0,) if mode == 5 else x.shape, device="cuda", dtype=torch.float32
        )
        output = torch.empty_like(
            x, dtype=torch.bfloat16 if mode == 5 else torch.float8_e4m3fn
        )
        pitch = (num_tokens + 3) // 4 * 4
        scales = torch.full((42, pitch), -17, device="cuda", dtype=torch.float32)
        torch.ops._C.gemma4_full_chain_out(
            x, residual, weight, weight, scalar, carry, output, scales, mode, EPS
        )
    stream.synchronize()
    if mode == 5:
        assert carry.shape == (0,)
        assert torch.equal(output, torch.ones_like(x))
        assert torch.all(scales == -17)
    else:
        assert torch.all(carry == (1 if mode == 2 else 2))
        assert torch.all(output.float() == 256)
        assert torch.all(scales[:, :num_tokens] == 1 / 256)
        assert torch.all(scales[:, num_tokens:] == -17)
    assert torch.count_nonzero(x) == 0
    assert torch.all(residual == 2)
    assert torch.all(weight == 1)


@pytest.mark.skipif(
    not current_platform.is_device_capability(90), reason="Gemma4 chain is SM90-only"
)
@pytest.mark.parametrize("num_tokens", [31, 32])
@pytest.mark.parametrize("mode", [0, 1, 2, 5])
def test_gemma4_full_chain_adapter_raw_boundary_and_fake(num_tokens, mode):
    from vllm.model_executor.layers.gemma4_full_chain import fake, full_chain

    x = torch.zeros((num_tokens, 5376), device="cuda", dtype=torch.bfloat16)
    residual = torch.full_like(
        x, 2, dtype=torch.bfloat16 if mode == 0 else torch.float32
    )
    weight = torch.ones(5376, device="cuda", dtype=torch.bfloat16)
    scalar = torch.tensor([0.5], device="cuda", dtype=torch.bfloat16)
    args = (x, residual, weight, weight, scalar, mode, EPS)
    carry, output, scales = full_chain(*args)
    fake_outputs = fake(*args)
    for actual, abstract in zip((carry, output, scales), fake_outputs):
        assert (actual.shape, actual.dtype, actual.stride()) == (
            abstract.shape,
            abstract.dtype,
            abstract.stride(),
        )
    assert scales.shape == (num_tokens, 42)
    assert scales.stride() == (1, (num_tokens + 3) // 4 * 4)
    if mode == 5:
        assert carry.shape == (0,)
        assert torch.equal(output, torch.ones_like(x))
    else:
        assert torch.all(carry == (1 if mode == 2 else 2))
        expected_scale = 1 / 448 if num_tokens < 32 else 1 / 256
        expected_q = 448 if num_tokens < 32 else 256
        assert torch.all(scales == expected_scale)
        assert torch.all(output.float() == expected_q)


@pytest.mark.skipif(
    not current_platform.is_device_capability(90), reason="Gemma4 chain is SM90-only"
)
@pytest.mark.parametrize(
    "invalid",
    [
        "unaligned_input",
        "unaligned_weight",
        "unaligned_output",
        "wrong_residual",
        "wrong_pitch",
        "zero_rows",
        "alias",
        "mode",
    ],
)
def test_gemma4_full_chain_rejects_unsafe_native_buffers(invalid):
    m, mode = (0, 0) if invalid == "zero_rows" else (32, 3)
    x = torch.zeros((m, 5376), device="cuda", dtype=torch.bfloat16)
    residual = torch.ones_like(x)
    weight = torch.ones(5376, device="cuda", dtype=torch.bfloat16)
    scalar = torch.ones(1, device="cuda", dtype=torch.bfloat16)
    carry = torch.empty(x.shape, device="cuda", dtype=torch.float32)
    output = torch.empty_like(x)
    scales = torch.empty((42, (m + 3) // 4 * 4), device="cuda", dtype=torch.float32)
    if invalid == "unaligned_input":
        x = torch.empty(x.numel() + 1, device="cuda", dtype=x.dtype)[1:].view(x.shape)
    elif invalid == "unaligned_weight":
        weight = torch.empty(5377, device="cuda", dtype=weight.dtype)[1:]
    elif invalid == "unaligned_output":
        output = torch.empty(output.numel() + 1, device="cuda", dtype=output.dtype)[
            1:
        ].view(output.shape)
    elif invalid == "wrong_residual":
        residual = residual.float()
    elif invalid == "wrong_pitch":
        scales = torch.empty((42, 33), device="cuda", dtype=torch.float32)
    elif invalid == "alias":
        output = x
    elif invalid == "mode":
        mode = 7
    with pytest.raises(RuntimeError):
        torch.ops._C.gemma4_full_chain_out(
            x, residual, weight, weight, scalar, carry, output, scales, mode, EPS
        )


def test_gemma4_full_chain_rejects_zero_rows_in_fake():
    from vllm.model_executor.layers.gemma4_full_chain import fake

    x = torch.empty((0, 5376), dtype=torch.bfloat16)
    weight = torch.empty(5376, dtype=torch.bfloat16)
    scalar = torch.empty(1, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="positive M"):
        fake(x, x, weight, weight, scalar, 0, EPS)
