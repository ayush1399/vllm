# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare logits-statistics JIT specialization with paired CUPTI timing.

Check bitwise outputs before timing. Measure only the kernel with alternating
arm order; allocation, compilation, and cache flushing are excluded. The default
candidate removes alignment specialization for scalar index loads and row count,
while retaining the single-row bonus-only specialization.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.triton_utils import triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_local_logits_stats_kernel as original,
)

ARMS = {
    "baseline": {},
    "row_alignment": {"do_not_specialize_on_alignment": ["num_logits"]},
    "indices": {
        "do_not_specialize_on_alignment": [
            "expanded_idx_mapping_ptr",
            "expanded_local_pos_ptr",
        ]
    },
    "alignment_only": {
        "do_not_specialize_on_alignment": [
            "num_logits",
            "expanded_idx_mapping_ptr",
            "expanded_local_pos_ptr",
        ]
    },
    "no_scalar": {
        "do_not_specialize": ["num_logits"],
        "do_not_specialize_on_alignment": [
            "expanded_idx_mapping_ptr",
            "expanded_local_pos_ptr",
        ],
    },
}


def setup(n, v, dtype, mode, offset):
    blocks = triton.cdiv(v, 8192)
    logits = torch.randn(n, v, dtype=dtype, device="cuda")
    mapping = torch.arange(n + 4, device="cuda", dtype=torch.int32)[offset : offset + n]
    positions = torch.empty(n + 4, device="cuda", dtype=torch.int32)[
        offset : offset + n
    ]
    positions.copy_(torch.arange(n, device="cuda") % 5)
    temps = torch.full((n + 4,), 0.7 if mode != "greedy" else 0.0, device="cuda")
    if mode == "mixed":
        temps[::2] = 0.0
    has_draft = mode in ("draft", "mixed")
    draft = torch.randn(n + 4, 4, v, dtype=dtype, device="cuda") if has_draft else None
    outputs = [
        torch.full(
            (n, blocks),
            -12345,
            dtype=torch.int64 if i == 0 else torch.float32,
            device="cuda",
        )
        for i in range(5)
    ]
    arguments = []
    for tensor in outputs:
        arguments.extend([tensor, tensor.stride(0)])
    arguments.extend(
        [
            logits,
            logits.stride(0),
            draft,
            draft.stride(0) if has_draft else 0,
            draft.stride(1) if has_draft else 0,
            mapping,
            positions,
            temps,
            v,
            n,
        ]
    )

    def launch(kernel, warps=4):
        return kernel[(n, blocks)](
            *arguments,
            BLOCK_SIZE=8192,
            HAS_DRAFT_LOGITS=has_draft,
            DRAFT_LOGITS_NUM_SHARDS=1,
            num_warps=warps,
        )

    return launch, outputs, (logits, draft, mapping, positions, temps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[1, 5, 16, 20, 64, 256, 512]
    )
    parser.add_argument(
        "--vocab-sizes", type=int, nargs="+", default=[151936, 262144, 262147]
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["bfloat16", "float32"],
        default=["bfloat16", "float32"],
    )
    parser.add_argument(
        "--arms", nargs="+", choices=list(ARMS), default=["baseline", "alignment_only"]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--warm-l2", action="store_true")
    parser.add_argument("--dump-code", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if "baseline" not in args.arms:
        parser.error("--arms must include baseline")
    torch.manual_seed(1234)
    burn = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    started = time.monotonic()
    while time.monotonic() - started < 2:
        torch.mm(burn, burn)
        torch.accelerator.synchronize()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    kernels = {
        name: triton.jit(original.fn, **settings)
        for name, settings in ARMS.items()
        if name in args.arms
    }
    results = []
    cases = [
        (n, v, dtype)
        for n in args.rows
        for v in args.vocab_sizes
        for dtype in args.dtypes
    ]
    for n, vocab, dtype in cases:
        for mode in ("greedy", "random", "draft", "mixed"):
            for offset in (0, 1):
                launch, outputs, backing = setup(
                    n, vocab, getattr(torch, dtype), mode, offset
                )
                baseline = launch(kernels["baseline"], args.num_warps)
                reference = [t.clone() for t in outputs]
                valid = torch.zeros(n, device="cuda", dtype=torch.bool)
                valid[:-1] = backing[3][1:] != 0
                draft_rows = (valid & (backing[4][backing[2].long()] != 0)).sum().item()
                logits_bytes = (
                    (valid.sum().item() + (draft_rows if backing[1] is not None else 0))
                    * vocab
                    * backing[0].element_size()
                )
                code = baseline.asm["sass"]
                row = dict(
                    n=n,
                    vocab=vocab,
                    dtype=dtype,
                    num_warps=args.num_warps,
                    mode=mode,
                    offset=offset,
                    logits_bytes=logits_bytes,
                    arms={},
                )
                for name, kernel in kernels.items():
                    for output in outputs:
                        output.fill_(-12345)
                    compiled = launch(kernel, args.num_warps)
                    for actual, expected in zip(outputs, reference):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    asm = compiled.asm["sass"]
                    row["arms"][name] = dict(
                        registers=compiled.n_regs,
                        cubin_equal_to_baseline=compiled.asm["cubin"]
                        == baseline.asm["cubin"],
                        cubin_hash=hashlib.sha256(compiled.asm["cubin"]).hexdigest(),
                        asm_equal_to_baseline=asm == code,
                        asm_hash=hashlib.sha256(asm.encode()).hexdigest(),
                        us=[],
                    )
                    if args.dump_code:
                        stem = f"{n}-{vocab}-{dtype}-{mode}-{offset}-{name}"
                        (args.output.parent / f"{stem}.ptx").write_text(
                            compiled.asm["ptx"]
                        )
                        if "sass" in compiled.asm:
                            (args.output.parent / f"{stem}.sass").write_text(
                                compiled.asm["sass"]
                            )
                torch.accelerator.synchronize()
                for repeat in range(args.repeats):
                    names = (
                        list(kernels) if repeat % 2 == 0 else list(reversed(kernels))
                    )
                    for name in names:
                        fn = lambda launch=launch, kernel=kernels[name]: launch(
                            kernel, args.num_warps
                        )
                        times = bench_gpu_time_with_cupti(
                            fn,
                            dry_run_iters=3,
                            repeat_iters=args.iterations,
                            use_cuda_graph=True,
                            cold_l2_cache=not args.warm_l2,
                        )
                        row["arms"][name]["us"].append(statistics.median(times) * 1000)
                for arm in row["arms"].values():
                    if arm["us"]:
                        arm["median_us"] = statistics.median(arm["us"])
                        arm["logits_gbps"] = logits_bytes / (arm["median_us"] * 1000)
                results.append(row)
                print(json.dumps(row), flush=True)
                args.output.write_text(
                    json.dumps(
                        dict(
                            gpu=torch.cuda.get_device_name(),
                            revision=revision,
                            assembly="sass",
                            torch=torch.__version__,
                            triton=triton.__version__,
                            timing=dict(
                                method="CUPTI",
                                cuda_graph=True,
                                cold_l2=not args.warm_l2,
                                repeats=args.repeats,
                                iterations=args.iterations,
                            ),
                            settings={name: ARMS[name] for name in args.arms},
                            results=results,
                        ),
                        indent=2,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
