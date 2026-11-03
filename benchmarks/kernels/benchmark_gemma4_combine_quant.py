# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare combine/quantize against a Git revision using paired CUPTI samples.

Run from the repository root with .venv/bin/python. Both kernels are built in
one library with identical flags. Only the CUDA kernel is timed; input creation,
compilation, correctness checks, and cache flushing are outside its duration.
"""

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti
from torch.utils.cpp_extension import load

SOURCE = Path("vllm/models/gemma4/nvidia/specialized_decode/attention_prep.cu")


def kernel(source: str, name: str) -> str:
    begin = source.index("void combine_quant(")
    begin = source.rfind("__global__", 0, begin)
    helper = source.rfind("template <int Threads>", 0, begin)
    if helper >= 0:
        begin = helper
    end = source.index("void gemma4_combine_quant_out(", begin)
    return (
        source[begin:end]
        .replace("combine_quant(", f"{name}(")
        .replace("combine_weights_warp", f"{name}_weights")
        .replace("combine_partials", f"{name}_partials")
        .replace("combine_single_row", f"{name}_single_row")
    )


def build(baseline: str, candidate: str, root: Path):
    code = """
#include "cub_helpers.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cub/block/block_reduce.cuh>
"""
    for name, source in (("baseline", baseline), ("candidate", candidate)):
        code += kernel(source, f"combine_{name}")
        code += f"""
extern "C" void launch_{name}(
    const float* p, const float* l, const int* cu, const int* splits,
    void* out, float* scales, int rows, int heads, int dim, int batches,
    int pitch, int64_t ls0, int64_t ls1, int64_t ls2, void* stream) {{
  combine_{name}<<<dim3(rows, heads * dim / 128), 128, 0,
                    (cudaStream_t)stream>>>(
      p, l, cu, splits, (__nv_fp8_e4m3*)out, scales, rows, heads, dim,
      batches, pitch, ls0, ls1, ls2);
}}
"""
    name = "gemma4_combine_compare_" + hashlib.sha256(code.encode()).hexdigest()[:16]
    build_dir = root / name
    build_dir.mkdir(parents=True, exist_ok=True)
    path = build_dir / "compare.cu"
    if not path.exists() or path.read_text() != code:
        temporary = path.with_name(f"compare.{os.getpid()}.cu.tmp")
        temporary.write_text(code)
        temporary.replace(path)
    library = ctypes.CDLL(
        load(
            name=name,
            sources=[str(path)],
            extra_include_paths=[str(Path.cwd() / "csrc")],
            extra_cuda_cflags=[
                "-O3",
                "-lineinfo",
                "-gencode=arch=compute_90,code=sm_90",
                "--expt-relaxed-constexpr",
            ],
            build_directory=str(build_dir),
            is_python_module=False,
        )
    )
    for name in ("baseline", "candidate"):
        fn = getattr(library, f"launch_{name}")
        fn.argtypes = (
            [ctypes.c_void_p] * 6
            + [ctypes.c_int] * 5
            + [ctypes.c_int64] * 3
            + [ctypes.c_void_p]
        )
        fn.restype = None
    return library


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warm-l2", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--split-counts", type=int, nargs="+")
    args = parser.parse_args()
    baseline = subprocess.check_output(
        ["git", "show", f"{args.baseline_ref}:{SOURCE}"], text=True
    )
    candidate = SOURCE.read_text()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lib = build(baseline, candidate, args.output.parent)
    torch.manual_seed(714)
    burn = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    began = time.monotonic()
    while time.monotonic() - began < 2:
        torch.mm(burn, burn)
        torch.accelerator.synchronize()
    results = []
    counts = args.split_counts or ((4, 128) if args.quick else (4, 9, 32, 128))
    if any(count < 1 or count > 128 for count in counts):
        raise ValueError("Split counts must be in [1, 128]")
    cases = (
        ((1, 1), (4, 3), (20, 20))
        if args.quick
        else ((1, 1), (4, 4), (4, 3), (5, 5), (8, 8), (16, 15), (20, 20), (31, 31))
    )
    for dim in (256, 512):
        for rows, actual in cases:
            for count in counts:
                heads = 4
                partial = torch.randn(count, rows, heads, dim, device="cuda")
                lse = torch.randn(count, heads, rows, device="cuda")
                cu = torch.arange(rows + 1, device="cuda", dtype=torch.int32).clamp(
                    max=actual
                )
                # Baseline padded counts must be bounded to permit timing it.
                splits = torch.full((rows,), count, device="cuda", dtype=torch.int32)
                pitch = (rows + 3) // 4 * 4
                output = [
                    torch.empty(
                        rows, heads, dim, device="cuda", dtype=torch.float8_e4m3fn
                    )
                    for _ in range(2)
                ]
                scales = [
                    torch.empty(heads * dim // 128, pitch, device="cuda")
                    for _ in range(2)
                ]
                calls = []
                for index, name in enumerate(("baseline", "candidate")):
                    fn = getattr(lib, f"launch_{name}")
                    parameters = (
                        *(
                            t.data_ptr()
                            for t in (
                                partial,
                                lse,
                                cu,
                                splits,
                                output[index],
                                scales[index],
                            )
                        ),
                        rows,
                        heads,
                        dim,
                        rows,
                        pitch,
                        *lse.stride(),
                    )
                    calls.append(
                        lambda fn=fn, parameters=parameters: fn(
                            *parameters,
                            torch.accelerator.current_stream().native_handle,
                        )
                    )
                calls[0]()
                splits[actual:].fill_(1 << 30)
                calls[1]()
                torch.accelerator.synchronize()
                assert torch.equal(
                    output[0][:actual].view(torch.uint8),
                    output[1][:actual].view(torch.uint8),
                )
                assert torch.equal(scales[0][:, :actual], scales[1][:, :actual])
                assert torch.count_nonzero(output[1][actual:].float()) == 0
                assert torch.all(scales[1][:, actual:rows] == 1)
                splits.fill_(count)
                torch.accelerator.synchronize()
                timings = [[], []]
                for repeat in range(args.repeats):
                    order = (0, 1) if repeat % 2 == 0 else (1, 0)
                    for arm in order:
                        samples = bench_gpu_time_with_cupti(
                            calls[arm],
                            dry_run_iters=10,
                            repeat_iters=args.iterations,
                            use_cuda_graph=True,
                            cold_l2_cache=not args.warm_l2,
                        )
                        timings[arm].append(statistics.median(samples) * 1000)
                us = [statistics.median(values) for values in timings]
                logical_bytes = actual * heads * (4 * count * (dim + 1) + dim)
                logical_bytes += 4 * actual * heads * dim // 128
                record = {
                    "rows": rows,
                    "actual": actual,
                    "heads": heads,
                    "dim": dim,
                    "splits": count,
                    "baseline_us": us[0],
                    "candidate_us": us[1],
                    "ratio": us[1] / us[0],
                    "logical_bytes": logical_bytes,
                    "baseline_effective_gbps": logical_bytes / (us[0] * 1000),
                    "candidate_effective_gbps": logical_bytes / (us[1] * 1000),
                    "baseline_pass_us": timings[0],
                    "candidate_pass_us": timings[1],
                }
                results.append(record)
                print(json.dumps(record), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "baseline_ref": args.baseline_ref,
                "baseline_sha256": hashlib.sha256(baseline.encode()).hexdigest(),
                "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
                "gpu": torch.cuda.get_device_properties().name,
                "torch": torch.__version__,
                "cold_l2": not args.warm_l2,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
