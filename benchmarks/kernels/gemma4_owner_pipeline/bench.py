# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP8 projection, owner normalization/publication, and following GEMM."""

import argparse
import ctypes
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import statistics
import subprocess
from datetime import timedelta
from pathlib import Path

import pynvml
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
from flashinfer.comm import AllReduceFusionPattern, allreduce_fusion
from flashinfer.gemm import fp8_blockscale_gemm_sm90 as gemm
from flashinfer.gemm.gemm_base import get_fp8_blockscale_gemm_runner_sm90

from vllm.distributed.device_communicators.flashinfer_all_reduce import (
    get_fi_ar_workspace,
)
from vllm.models.gemma4.nvidia.specialized_decode import full_chain

parser = argparse.ArgumentParser()
parser.add_argument("--build-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--rows", type=int, nargs="+", default=[5, 20])
parser.add_argument("--ks", type=int, nargs="+", default=[1024, 2048, 2688])
parser.add_argument("--samples", type=int, default=64)
parser.add_argument("--warmup", type=int, default=128)
parser.add_argument("--replays", type=int, default=8)
parser.add_argument("--tune", action="store_true")
parser.add_argument("--trace", action="store_true")
parser.add_argument("--stress", type=int, default=4)
parser.add_argument("--configs-from", type=Path)
parser.add_argument(
    "--arms",
    nargs="+",
    default=[
        "fi",
        "matched",
        "owner_barrier",
        "owner_ready",
        "owner_pdl",
        "owner_uc",
        "owner_uc_pdl",
        "owner_writer_pdl",
        "owner_grid",
        "owner_grid_pdl",
        "owner_grid_prefetch_pdl",
        "local_plain",
        "local_pdl",
        "replicated_plain",
        "replicated_pdl",
    ],
)
args = parser.parse_args()

rank = int(os.environ["LOCAL_RANK"])
torch.accelerator.set_device_index(rank)
device = torch.device("cuda", rank)
dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=180))
assert dist.get_world_size() == 8
cpu = dist.new_group(backend="gloo", timeout=timedelta(seconds=180))
pynvml.nvmlInit()
nvml = pynvml.nvmlDeviceGetHandleByIndex(rank)
workspace = get_fi_ar_workspace(8, rank, 4096, 5376, torch.bfloat16, cpu)
pointer, integer, uint64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
projection = ctypes.CDLL(str(args.build_dir / "projection.so"))
projection.prepare.argtypes = [pointer] * 5 + [integer] * 5 + [ctypes.POINTER(pointer)]
projection.prepare.restype = integer
projection.launch_projection.argtypes = (
    [pointer, uint64] + [integer] * 3 + [pointer, integer]
)
projection.launch_projection.restype = integer
projection.release.argtypes = [pointer]
following = ctypes.CDLL(str(args.build_dir / "next_gemm.so"))
following.prepare.argtypes = (
    [pointer] * 5
    + [integer] * 5
    + [pointer] * 3
    + [integer, pointer, integer, ctypes.POINTER(pointer)]
)
following.prepare.restype = integer
following.launch_next.argtypes = [pointer, pointer, integer, integer]
following.launch_next.restype = integer
following.release.argtypes = [pointer]
owner = ctypes.CDLL(str(args.build_dir / "owner_norm.so"))
owner.launch_owner.argtypes = (
    [pointer]
    + [integer] * 3
    + [pointer] * 5
    + [uint64] * 3
    + [pointer, integer, pointer, pointer, integer, integer]
)
owner.launch_owner.restype = integer
replicated = ctypes.CDLL(str(args.build_dir / "replicated_norm.so"))
replicated.launch_local.argtypes = [pointer] * 7 + [integer] * 2 + [pointer] * 2
replicated.launch_local.restype = integer
replicated.launch_replicated.argtypes = (
    [pointer] * 3 + [uint64] + [integer] * 3 + [pointer] * 7 + [integer] + [pointer] * 2
)
replicated.launch_replicated.restype = integer
implementation = {
    "git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "torch": torch.__version__,
    "flashinfer": importlib.metadata.version("flashinfer-python"),
    "libraries_sha256": {
        name: hashlib.sha256((args.build_dir / (name + ".so")).read_bytes()).hexdigest()
        for name in ("projection", "owner_norm", "replicated_norm", "next_gemm")
    },
    "source_sha256": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(__file__).parent.iterdir()
        if path.suffix in (".py", ".cu", ".cuh", ".patch")
    },
}

arms = tuple(args.arms)
assert arms[:2] == ("fi", "matched")
results, retained = [], []
H = 5376
DIRTY = -2147483648


def check(rc):
    assert rc == 0, ("CUDA launch/setup error", rank, rc)


def symmetric(shape, dtype):
    tensor = symm.empty(shape, device=device, dtype=dtype)
    tensor.zero_()
    handle = symm.rendezvous(tensor, dist.group.WORLD)
    assert handle.multicast_ptr
    return tensor, handle


def benchmark_case(k, rows):
    next_n = 2048 if k == 2688 else 5376
    mode = 2 if k == 2688 else 1
    pitch = (rows + 3) // 4 * 4
    owned = (rows + 7) // 8
    valid_owned = len(range(rank, rows, 8))
    torch.manual_seed(701 + rank * 31 + rows + k)
    scatter, scatter_handle = symmetric((2, 8, owned, H), torch.bfloat16)
    scatter.view(torch.int32).fill_(DIRTY)
    peers = torch.tensor(scatter_handle.buffer_ptrs, device=device, dtype=torch.int64)
    error = torch.zeros(1, device=device, dtype=torch.int32)
    part_box, part_handle = symmetric((2, 8, rows, H), torch.bfloat16)
    part_box.view(torch.int32).fill_(DIRTY)
    sum_box, sum_handle = symmetric((2, rows, H), torch.bfloat16)
    sum_box.view(torch.int32).fill_(DIRTY)
    communications = {}
    for name in (n for n in arms if n.startswith("owner_")):
        q, qh = symmetric((2, rows, H), torch.float8_e4m3fn)
        scales, sh = symmetric((2, H // 128, pitch), torch.float32)
        flags, fh = symmetric((2, max(rows, 8)), torch.int32)
        ready_peers = torch.tensor(
            [
                [ptr + slot * max(rows, 8) * 4 for ptr in fh.buffer_ptrs]
                for slot in range(2)
            ],
            device=device,
            dtype=torch.int64,
        )
        q_peers = torch.tensor(qh.buffer_ptrs, device=device, dtype=torch.int64)
        scale_peers = torch.tensor(sh.buffer_ptrs, device=device, dtype=torch.int64)
        generations = torch.zeros(2, owned, device=device, dtype=torch.int32)
        expected = torch.zeros(2, 132, device=device, dtype=torch.int32)
        communications[name] = dict(
            q=q,
            qh=qh,
            scales=scales,
            sh=sh,
            ready_peers=ready_peers,
            q_peers=q_peers,
            scale_peers=scale_peers,
            flags=flags,
            fh=fh,
            generations=generations,
            expected=expected,
        )
    # Each graph streams more than 80 MiB of distinct weights on every rank.
    count = max(3, math.ceil(80 * 1024**2 / (2 * H * (k + next_n))))
    data = []
    quantizer = get_fp8_blockscale_gemm_runner_sm90()
    for j in range(count):
        pair = []
        for slot in range(2):
            x = torch.randn(rows, k, device=device, dtype=torch.bfloat16)
            q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
            xs = torch.empty(k // 128, pitch, device=device)
            quantizer.fp8_quantize_1x128(x, q, xs, False)
            w = torch.randn(H, k, device=device).to(torch.float8_e4m3fn)
            ws = (0.5 + torch.rand(H // 128, k // 128, device=device)) / k**0.5
            nw = torch.randn(next_n, H, device=device).to(torch.float8_e4m3fn)
            nws = (0.5 + torch.rand(next_n // 128, H // 128, device=device)) / H**0.5
            g = torch.Generator(device=device).manual_seed(900 + j * 2 + slot)
            residual = torch.randn(rows, H, device=device, generator=g)
            post = (1 + 0.1 * torch.randn(H, device=device, generator=g)).bfloat16()
            pre = (1 + 0.1 * torch.randn(H, device=device, generator=g)).bfloat16()
            scalar = torch.full((1,), 0.875, device=device, dtype=torch.bfloat16)
            output = torch.empty(rows, H, device=device, dtype=torch.bfloat16)
            reduced = torch.empty_like(output)
            reference = {
                n: full_chain.allocate(output, mode)
                for n in arms
                if not n.startswith("owner_")
            }
            carries = {
                n: torch.full((owned, H), float("nan"), device=device)
                for n in arms
                if n.startswith("owner_")
            }
            owned_residual = torch.full((owned, H), float("nan"), device=device)
            owned_residual[:valid_owned].copy_(residual[rank::8])
            next_out = {
                n: torch.empty(rows, next_n, device=device, dtype=torch.bfloat16)
                for n in arms
            }
            pair.append(
                dict(
                    q=q,
                    xs=xs,
                    w=w,
                    ws=ws,
                    nw=nw,
                    nws=nws,
                    residual=residual,
                    owned_residual=owned_residual,
                    post=post,
                    pre=pre,
                    scalar=scalar,
                    output=output,
                    reduced=reduced,
                    reference=reference,
                    carries=carries,
                    next_out=next_out,
                )
            )
        data.append(pair)

    # First-GEMM geometry from the previous full-chain sweep, retuned below
    # together with the following GEMM when --tune is requested.
    first_config = (
        (64, 32, 8)
        if rows == 20 and k != 1024
        else ((128, 16, 8) if rows == 20 else (64, 16, 8))
    )

    def prepare(first, second):
        for pair in data:
            for slot, d in enumerate(pair):
                params = pointer()
                check(
                    projection.prepare(
                        d["q"].data_ptr(),
                        d["w"].data_ptr(),
                        d["xs"].data_ptr(),
                        d["ws"].data_ptr(),
                        d["output"].data_ptr(),
                        rows,
                        k,
                        *first,
                        ctypes.byref(params),
                    )
                )
                d["first"] = params
                d["next"] = {}
                for name in arms:
                    if not name.startswith("owner_"):
                        _, q, scales = d["reference"][name]
                        ready = expected = 0
                    else:
                        c = communications[name]
                        q, scales = c["q"][slot], c["scales"][slot]
                        ready = c["flags"][slot].data_ptr()
                        expected = c["expected"][slot].data_ptr()
                    params = pointer()
                    check(
                        following.prepare(
                            q.data_ptr(),
                            d["nw"].data_ptr(),
                            scales.data_ptr(),
                            d["nws"].data_ptr(),
                            d["next_out"][name].data_ptr(),
                            rows,
                            next_n,
                            *second,
                            ready,
                            expected,
                            error.data_ptr(),
                            -1
                            if name.startswith(("local_", "replicated_"))
                            else 2
                            if name.startswith("owner_grid")
                            else int(name in ("owner_ready", "owner_pdl")),
                            communications[name]["ready_peers"][slot].data_ptr()
                            if name.startswith("owner_")
                            else 0,
                            rank,
                            ctypes.byref(params),
                        )
                    )
                    d["next"][name] = params

    def release():
        for pair in data:
            for d in pair:
                projection.release(d["first"])
                for params in d["next"].values():
                    following.release(params)

    def run(index, name):
        stream = torch.cuda.current_stream().cuda_stream
        for slot, d in enumerate(data[index]):
            if name == "fi":
                gemm(d["q"], d["w"], d["xs"][:, :rows].t(), d["ws"], out=d["output"])
            else:
                check(
                    projection.launch_projection(
                        d["first"],
                        part_handle.multicast_ptr
                        if name.startswith("replicated_") and rows <= 16
                        else peers.data_ptr(),
                        rows,
                        slot,
                        rank,
                        stream,
                        0
                        if name == "matched" or name.startswith("local_")
                        else 1
                        if name.startswith("replicated_") and rows <= 16
                        else 2,
                    )
                )
            if name in arms[:2] or name.startswith("local_"):
                a = d["reference"][name]
                allreduce_fusion(
                    d["output"],
                    workspace,
                    AllReduceFusionPattern.kAllReduce,
                    output=d["reduced"],
                    launch_with_pdl=True,
                    trigger_completion_at_end=rows > 16,
                )
                if name.startswith("local_"):
                    check(
                        replicated.launch_local(
                            d["reduced"].data_ptr(),
                            d["residual"].data_ptr(),
                            d["post"].data_ptr(),
                            d["pre"].data_ptr(),
                            a[0].data_ptr(),
                            a[1].data_ptr(),
                            a[2].data_ptr(),
                            rows,
                            mode,
                            d["scalar"].data_ptr(),
                            stream,
                        )
                    )
                else:
                    torch.ops._C.gemma4_full_chain_out(
                        d["reduced"],
                        d["residual"],
                        d["post"],
                        d["pre"],
                        d["scalar"],
                        *a,
                        mode + 7,
                        1e-6,
                    )
                if name == "fi":
                    gemm(
                        a[1],
                        d["nw"],
                        a[2][:, :rows].t(),
                        d["nws"],
                        out=d["next_out"][name],
                    )
                    continue
            elif name.startswith("replicated_"):
                a = d["reference"][name]
                check(
                    replicated.launch_replicated(
                        scatter.data_ptr(),
                        part_box.data_ptr(),
                        sum_box.data_ptr(),
                        sum_handle.multicast_ptr,
                        rows,
                        slot,
                        rank,
                        error.data_ptr(),
                        d["residual"].data_ptr(),
                        d["post"].data_ptr(),
                        d["pre"].data_ptr(),
                        a[0].data_ptr(),
                        a[1].data_ptr(),
                        a[2].data_ptr(),
                        mode,
                        d["scalar"].data_ptr(),
                        stream,
                    )
                )
            else:
                c = communications[name]
                check(
                    owner.launch_owner(
                        scatter.data_ptr(),
                        rows,
                        slot,
                        rank,
                        error.data_ptr(),
                        d["owned_residual"].data_ptr(),
                        d["post"].data_ptr(),
                        d["pre"].data_ptr(),
                        d["carries"][name].data_ptr(),
                        c["q_peers"].data_ptr()
                        if name.startswith("owner_uc")
                        else c["qh"].multicast_ptr + slot * rows * H,
                        c["scale_peers"].data_ptr()
                        if name.startswith("owner_uc")
                        else c["sh"].multicast_ptr + slot * 42 * pitch * 4,
                        c["ready_peers"][slot].data_ptr(),
                        c["generations"][slot].data_ptr(),
                        mode,
                        d["scalar"].data_ptr(),
                        stream,
                        2 if "_prefetch_" in name else int(name.endswith("pdl")),
                        1
                        if name.startswith("owner_uc")
                        else 3
                        if name.startswith("owner_grid")
                        else 2
                        if name == "owner_writer_pdl"
                        else 0,
                    )
                )
                if name == "owner_barrier":
                    c["qh"].barrier(channel=0, timeout_ms=10000)
            check(
                following.launch_next(
                    d["next"][name],
                    stream,
                    int(
                        name != "owner_barrier"
                        and name not in arms[:2]
                        and not name.endswith("plain")
                    ),
                    int(name.endswith("pdl")),
                )
            )

    def validate():
        torch.accelerator.synchronize()
        assert error.item() == 0, ("readiness timeout", rank, error.item())
        for j, pair in enumerate(data):
            for slot, d in enumerate(pair):
                ref = d["reference"]["fi"]
                for field, (a, b) in enumerate(zip(ref, d["reference"]["matched"])):
                    if field == 1:
                        a, b = a.view(torch.uint8), b.view(torch.uint8)
                    if field == 2:
                        a, b = a[:, :rows], b[:, :rows]
                    assert torch.equal(a, b), ("matched intermediate", rank, field)
                for name in arms:
                    assert torch.equal(d["next_out"]["fi"], d["next_out"][name]), (
                        "following GEMM mismatch",
                        rank,
                        rows,
                        k,
                        j,
                        slot,
                        name,
                        (d["next_out"]["fi"].float() - d["next_out"][name].float())
                        .abs()
                        .max()
                        .item(),
                    )
                    if not name.startswith("owner_"):
                        for field, (a, b) in enumerate(zip(ref, d["reference"][name])):
                            if field == 1:
                                a, b = a.view(torch.uint8), b.view(torch.uint8)
                            if field == 2:
                                a, b = a[:, :rows], b[:, :rows]
                            assert torch.equal(a, b), (
                                "local intermediate",
                                rank,
                                name,
                                field,
                            )
                        continue
                    carry = d["carries"][name]
                    assert torch.equal(ref[0][rank::8], carry[:valid_owned]), (
                        "owner carry",
                        rank,
                        name,
                    )
                    assert torch.isnan(carry[valid_owned:]).all().item()
                    if j == count - 1:
                        c = communications[name]
                        assert torch.equal(
                            ref[1].view(torch.uint8), c["q"][slot].view(torch.uint8)
                        ), (
                            "published FP8",
                            rank,
                            name,
                        )
                        assert torch.equal(
                            ref[2][:, :rows], c["scales"][slot, :, :rows]
                        ), (
                            "published scales",
                            rank,
                            name,
                        )
        assert (scatter.view(torch.int32) == DIRTY).all().item()
        assert (part_box.view(torch.int32) == DIRTY).all().item()
        assert (sum_box.view(torch.int32) == DIRTY).all().item()
        dist.barrier(group=cpu)

    def capture_time(first, second, samples):
        prepare(first, second)
        graphs = {}
        for name in arms:
            for j in range(count):
                run(j, name)
            torch.accelerator.synchronize()
            dist.barrier(group=cpu)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for j in range(count):
                    run(j, name)
            graphs[name] = graph
        validate()
        # Changing rank-specific FP8 inputs after capture catches stale payloads
        # and readiness epochs that only work for identical graph replays.
        for _ in range(args.stress):
            for pair in data:
                for d in pair:
                    quantizer.fp8_quantize_1x128(
                        torch.randn(rows, k, device=device, dtype=torch.bfloat16),
                        d["q"],
                        d["xs"],
                        False,
                    )
            for name in arms:
                scatter_handle.barrier(channel=1, timeout_ms=10000)
                graphs[name].replay()
            validate()
        for j in range(args.warmup):
            for offset in range(len(arms)):
                name = arms[(j + offset) % len(arms)]
                scatter_handle.barrier(channel=1, timeout_ms=10000)
                for _ in range(args.replays):
                    graphs[name].replay()
        torch.accelerator.synchronize()
        before = pynvml.nvmlDeviceGetClockInfo(nvml, pynvml.NVML_CLOCK_SM)
        timing = {name: [] for name in arms}
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for j in range(samples):
            for offset in range(len(arms)):
                name = arms[(j + offset) % len(arms)]
                scatter_handle.barrier(channel=1, timeout_ms=10000)
                start.record()
                for _ in range(args.replays):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                elapsed = torch.tensor(
                    start.elapsed_time(end) * 1000 / (2 * count * args.replays),
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                timing[name].append(elapsed.item())
        after = pynvml.nvmlDeviceGetClockInfo(nvml, pynvml.NVML_CLOCK_SM)
        clocks = [None] * 8
        dist.all_gather_object(
            clocks, dict(before_mhz=before, after_mhz=after), group=cpu
        )
        validate()
        if args.trace:
            dist.barrier(group=cpu)
            torch.cuda.cudart().cudaProfilerStart()
            for name in arms:
                scatter_handle.barrier(channel=1, timeout_ms=10000)
                torch.cuda.nvtx.range_push(f"rows{rows}_k{k}_{name}")
                for _ in range(4):
                    graphs[name].replay()
                torch.accelerator.synchronize()
                torch.cuda.nvtx.range_pop()
                dist.barrier(group=cpu)
            torch.cuda.cudart().cudaProfilerStop()
        release()
        return dict(
            first_config=list(first),
            next_config=list(second),
            median_us={n: statistics.median(v) for n, v in timing.items()},
            samples_us=timing,
            clocks=clocks,
            exact_all_ranks=True,
        )

    configs = (
        list(itertools.product((64, 128), (16, 32), (4, 8)))
        + [(64, 16, 16), (64, 16, 20), (64, 32, 16), (128, 16, 10), (128, 32, 10)]
        if args.tune
        else [(64, 16, 8) if rows == 5 else (64, 32, 8)]
    )
    if args.configs_from:
        previous = json.loads(args.configs_from.read_text())
        configs = [
            next(r for r in previous if r["rows"] == rows and r["k"] == k)["selected"][
                "next_config"
            ]
        ]
    tuning = []
    for config in configs:
        record = capture_time(first_config, config, 16 if args.tune else args.samples)
        tuning.append(record)
        if rank == 0:
            print("TUNE", rows, k, config, record["median_us"], flush=True)
    best = min(tuning, key=lambda r: min(r["median_us"][n] for n in arms[2:]))
    chosen = (
        capture_time(first_config, best["next_config"], args.samples)
        if args.tune
        else best
    )
    if rank == 0:
        result = dict(
            implementation=implementation,
            rows=rows,
            k=k,
            next_n=next_n,
            next_k=H,
            selected=chosen,
            tuning=tuning,
            fixed_work="k4 B1/B4 verification rows; no sampling/acceptance variation",
            boundary=(
                "prequantized projection input -> GEMM -> TP8 sum -> "
                "postnorm/residual/scalar/prenorm/FP8 -> next GEMM BF16 output; "
                "owner-private FP32 carry"
            ),
            protocol=dict(
                ranks=8,
                gpu=torch.cuda.get_device_name(),
                samples=args.samples,
                warmup_rounds=args.warmup,
                replays=args.replays,
                operations_per_graph=2 * count,
                weights_bytes=2 * count * H * (k + next_n),
                input_scale_layout="FP32 column-major, M padded to multiple of four",
                weight_scale_layout="FP32 [N/128, K/128] blocks",
                collective=(
                    "FlashInfer mnnvl; candidate multicast within one NVSwitch domain"
                ),
                environment={
                    key: os.environ.get(key)
                    for key in ("OMP_NUM_THREADS", "VLLM_FLASHINFER_ALLREDUCE_BACKEND")
                },
                rank_reduction="MAX per sample",
                statistic="median",
                order="rotating arms",
                input_changes=args.stress,
            ),
        )
        results.append(result)
        args.output.write_text(json.dumps(results, indent=2))
        print("FINAL", rows, k, chosen["median_us"], flush=True)
    retained.append(
        (
            data,
            scatter,
            scatter_handle,
            peers,
            communications,
            error,
            part_box,
            part_handle,
            sum_box,
            sum_handle,
        )
    )


for k, rows in itertools.product(args.ks, args.rows):
    benchmark_case(k, rows)

workspace.destroy()
dist.destroy_process_group()
