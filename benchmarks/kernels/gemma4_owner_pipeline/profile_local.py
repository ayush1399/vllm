# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NCU control for local normalization/GEMM overlap; excludes TP reduction."""

import argparse
import ctypes
import json
from pathlib import Path

import torch
from flashinfer.gemm import fp8_blockscale_gemm_sm90 as gemm

from vllm.models.gemma4.nvidia.specialized_decode import full_chain

parser = argparse.ArgumentParser()
parser.add_argument("--build-dir", type=Path, required=True)
parser.add_argument("--configs-from", type=Path, required=True)
args = parser.parse_args()
torch.accelerator.set_device_index(0)
p, i = ctypes.c_void_p, ctypes.c_int
norm = ctypes.CDLL(str(args.build_dir / "replicated_norm.so"))
norm.launch_local.argtypes = [p] * 7 + [i] * 2 + [p] * 2
norm.launch_local.restype = i
following = ctypes.CDLL(str(args.build_dir / "next_gemm.so"))
following.prepare.argtypes = [p] * 5 + [i] * 5 + [p] * 3 + [i, p, i, ctypes.POINTER(p)]
following.prepare.restype = i
following.launch_next.argtypes = [p, p, i, i]
following.launch_next.restype = i
following.release.argtypes = [p]
configs = json.loads(args.configs_from.read_text())
retained = []

for rows in (5, 20):
    for n in (5376, 2048):
        torch.manual_seed(761 + rows + n)
        mode = 2 if n == 2048 else 1
        k = 2688 if mode == 2 else 1024
        config = next(r for r in configs if r["rows"] == rows and r["k"] == k)
        tile = config["selected"]["next_config"]
        x = torch.randn(rows, 5376, device="cuda").bfloat16()
        res = torch.randn(rows, 5376, device="cuda")
        post = torch.randn(5376, device="cuda").bfloat16()
        pre = torch.randn_like(post)
        scalar = torch.full((1,), 0.875, device="cuda", dtype=torch.bfloat16)
        carry, q, scales = full_chain.allocate(x, mode)
        weight = torch.randn(n, 5376, device="cuda").to(torch.float8_e4m3fn)
        ws = (0.5 + torch.rand(n // 128, 42, device="cuda")) / 5376**0.5
        error = torch.zeros(1, device="cuda", dtype=torch.int32)
        outputs = {
            name: torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
            for name in ("plain", "pdl")
        }
        graphs, handles = {}, []
        for name, out in outputs.items():
            params = p()
            rc = following.prepare(
                q.data_ptr(),
                weight.data_ptr(),
                scales.data_ptr(),
                ws.data_ptr(),
                out.data_ptr(),
                rows,
                n,
                *tile,
                0,
                0,
                error.data_ptr(),
                -1,
                0,
                0,
                ctypes.byref(params),
            )
            assert rc == 0, rc
            handles.append(params)

            norm_args = (
                x.data_ptr(),
                res.data_ptr(),
                post.data_ptr(),
                pre.data_ptr(),
                carry.data_ptr(),
                q.data_ptr(),
                scales.data_ptr(),
                rows,
                mode,
                scalar.data_ptr(),
            )

            def run(params=params, name=name, norm_args=norm_args):
                stream = torch.cuda.current_stream().cuda_stream
                rc = norm.launch_local(*norm_args, stream)
                assert rc == 0, rc
                rc = following.launch_next(params, stream, name == "pdl", name == "pdl")
                assert rc == 0, rc

            run()
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            graphs[name] = graph
            graph.replay()
            torch.accelerator.synchronize()
        expected = gemm(q, weight, scales[:, :rows].t(), ws)
        assert all(torch.equal(expected, out) for out in outputs.values())
        for name, graph in graphs.items():
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push(f"rows{rows}_n{n}_{name}")
            graph.replay()
            torch.accelerator.synchronize()
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
        assert all(torch.equal(expected, out) for out in outputs.values())
        assert error.item() == 0
        print("NCU local chain exact", rows, n, tile, flush=True)
        retained.append(
            (
                graphs,
                handles,
                outputs,
                x,
                res,
                post,
                pre,
                scalar,
                carry,
                q,
                scales,
                weight,
                ws,
                error,
            )
        )
