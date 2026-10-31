# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-checkpoint Gemma4 MTP FP8 fused-kernel correctness and CUPTI timing."""

import argparse
import hashlib
import json
import statistics
import warnings
from pathlib import Path

import torch
from flashinfer.gemm import fp8_blockscale_gemm_sm90
from flashinfer.testing import bench_gpu_time_with_cupti
from safetensors import safe_open

import vllm._custom_ops  # noqa: F401
import vllm.model_executor.kernels.linear.scaled_mm.flashinfer  # noqa: F401
import vllm.model_executor.models.gemma4  # noqa: F401
from vllm.model_executor.layers.gemma4_mtp_fp8 import concat_fp8_gemm, fp8_gemv
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
)


def load_tensor(model, name):
    index = model / "model.safetensors.index.json"
    shard = (
        model / json.loads(index.read_text())["weight_map"][name]
        if index.exists()
        else next(model.glob("*.safetensors"))
    )
    with safe_open(shard, framework="pt", device="cpu") as reader:
        return reader.get_tensor(name)


def load_projection(model, kind, rank):
    def parameter(prefix, suffix, n=None, k=None):
        n = slice(None) if n is None else n
        k = slice(None) if k is None else k
        group = 128 if suffix == "weight_scale" else 1
        ns = slice(n.start // group, n.stop // group) if n.start is not None else n
        ks = slice(k.start // group, k.stop // group) if k.start is not None else k
        return load_tensor(model, f"{prefix}.{suffix}")[ns, ks]

    gamma = None
    if kind == "pre":
        prefix = "pre_projection"
        weight = load_tensor(model, prefix + ".weight")
        scales = load_tensor(model, prefix + ".weight_scale")
    else:
        layer = 3 if kind.startswith("global") else 0
        base = f"model.layers.{layer}"
        if kind == "gate_up":
            ns = slice(rank * 1024, (rank + 1) * 1024)
            weight = torch.cat(
                [
                    parameter(f"{base}.mlp.{p}_proj", "weight", n=ns)
                    for p in ("gate", "up")
                ]
            )
            scales = torch.cat(
                [
                    parameter(f"{base}.mlp.{p}_proj", "weight_scale", n=ns)
                    for p in ("gate", "up")
                ]
            )
            gamma = load_tensor(model, base + ".pre_feedforward_layernorm.weight")
        else:
            q = kind.endswith("q")
            width = 2048 if kind.startswith("global") else 1024
            prefix = f"{base}.self_attn.{'q' if q else 'o'}_proj"
            if kind == "down":
                prefix = base + ".mlp.down_proj"
            shard = slice(rank * width, (rank + 1) * width)
            ns, ks = (shard, slice(None)) if q else (slice(None), shard)
            weight = parameter(prefix, "weight", n=ns, k=ks)
            scales = parameter(prefix, "weight_scale", n=ns, k=ks)
            if q:
                gamma = load_tensor(model, base + ".input_layernorm.weight")
    weight, scales = deepgemm_post_process_fp8_weight_block(
        weight.cuda().contiguous(),
        scales.cuda().float().contiguous(),
        (128, 128),
        use_e8m0=True,
    )
    return weight, scales, gamma.cuda() if gamma is not None else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 3, 4, 5, 8, 20])
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 7])
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=["q", "global_q", "gate_up", "o", "global_o", "down", "pre"],
        choices=["q", "global_q", "gate_up", "o", "global_o", "down", "pre"],
    )
    args = parser.parse_args()
    if any(not 0 <= rank < 8 for rank in args.ranks):
        parser.error("TP8 shard ranks must be in 0..7")
    warnings.filterwarnings("error", message=".*Falling back.*")
    torch.manual_seed(0)
    source = Path(__import__("vllm").__file__).resolve().parent
    kernel = source / "model_executor/layers/gemma4_mtp_fp8.py"
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "kernel_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
        "boundary": (
            "BF16 input (plus RMSNorm or concatenation) through local FP8 GEMM; "
            "TP collectives excluded"
        ),
        "timing": "CUPTI, CUDA graph, cold L2; alternating order, two rounds",
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for kind in args.kinds:
        for rank in [0] if kind == "pre" else args.ranks:
            weight, scales, gamma = load_projection(args.model, kind, rank)
            n, k = weight.shape
            for m in args.rows:
                if not 0 < m <= 20:
                    continue
                x = torch.randn(
                    (m, k // 2 if kind == "pre" else k),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                z = torch.randn_like(x) if kind == "pre" else None

                def native(x=x, z=z, gamma=gamma, weight=weight, scales=scales):
                    values = torch.cat([x, z], dim=-1) if z is not None else x
                    if gamma is not None:
                        values = torch.empty_like(x)
                        torch.ops._C.rms_norm(values, x, gamma, 1e-6)
                    return fp8_blockscale_gemm_sm90(
                        values, weight, weight_scale=scales, out_dtype=x.dtype
                    )

                def fused(x=x, z=z, gamma=gamma, weight=weight, scales=scales):
                    if z is not None:
                        return concat_fp8_gemm(x, z, weight, scales)
                    return fp8_gemv(x, weight, scales, gamma, 1e-6)

                expected, actual = native(), fused()
                relative = (actual.float() - expected.float()).abs().sum(
                    -1
                ) / expected.float().abs().sum(-1).clamp_min(1e-12)
                assert torch.isfinite(actual).all() and relative.max().item() <= 1e-3
                samples = {"native": [], "fused": []}
                for order in [("native", "fused"), ("fused", "native")]:
                    for arm in order:
                        fn = native if arm == "native" else fused
                        samples[arm].extend(
                            t * 1000
                            for t in bench_gpu_time_with_cupti(
                                fn,
                                dry_run_iters=10,
                                repeat_iters=50,
                                use_cuda_graph=True,
                                cold_l2_cache=True,
                            )
                        )
                native_us, fused_us = (
                    statistics.median(samples[a]) for a in ("native", "fused")
                )
                best_bytes = (
                    weight.numel()
                    + 4 * scales.numel()
                    + 2 * (m * k + m * n)
                    + (2 * gamma.numel() if gamma is not None else 0)
                )
                case = {
                    "kind": kind,
                    "tp_shard": rank,
                    "m": m,
                    "n": n,
                    "k": k,
                    "relative_l1_max": relative.max().item(),
                    "native_us": native_us,
                    "fused_us": fused_us,
                    "speedup": native_us / fused_us,
                    "fused_gbps": best_bytes / (fused_us * 1000),
                    "fused_tflops": 2 * m * n * k / (fused_us * 1e6),
                    "samples_us": samples,
                }
                result["cases"].append(case)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(
                    {key: value for key, value in case.items() if key != "samples_us"},
                    flush=True,
                )


if __name__ == "__main__":
    main()
