# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile owner publication with ready local partials and idle peers."""

import argparse
import ctypes
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm

from vllm.models.gemma4.nvidia.specialized_decode import full_chain

parser = argparse.ArgumentParser()
parser.add_argument("--build-dir", type=Path, required=True)
parser.add_argument("--publication", type=int, choices=(0, 2, 3), default=3)
args = parser.parse_args()
rank = int(os.environ["LOCAL_RANK"])
torch.accelerator.set_device_index(rank)
device = torch.device("cuda", rank)
dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=600))
cpu = dist.new_group(backend="gloo", timeout=timedelta(seconds=600))
pointer, integer, uint64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
lib = ctypes.CDLL(str(args.build_dir / "owner_norm.so"))
lib.launch_owner.argtypes = (
    [pointer]
    + [integer] * 3
    + [pointer] * 5
    + [uint64] * 3
    + [pointer, integer, pointer, pointer, integer, integer]
)
lib.launch_owner.restype = integer
retained = []

for rows in (5, 20):
    h, pitch, owned = 5376, (rows + 3) // 4 * 4, (rows + 7) // 8
    shared = []
    for shape, dtype in (
        ((rows, h), torch.float8_e4m3fn),
        ((42, pitch), torch.float32),
        ((rows,), torch.int32),
    ):
        t = symm.empty(shape, device=device, dtype=dtype)
        t.zero_()
        handle = symm.rendezvous(t, dist.group.WORLD)
        shared.append((t, handle))
    ready_peers = torch.tensor(
        shared[2][1].buffer_ptrs, device=device, dtype=torch.int64
    )
    if rank == 0:
        torch.manual_seed(491)
        scatter = torch.randn(2, 8, owned, h, device=device).bfloat16()
        snapshot = scatter.clone()
        reduced = torch.zeros(owned, h, device=device)
        for src in range(8):
            reduced.add_(scatter[0, src].float())
        x = torch.zeros(rows, h, device=device, dtype=torch.bfloat16)
        x[::8].copy_(reduced.bfloat16())
        residual = torch.randn(rows, h, device=device)
        owned_residual = residual[::8].contiguous()
        post = torch.randn(h, device=device).bfloat16()
        pre = torch.randn_like(post)
        scalar = torch.full((1,), 0.875, device=device, dtype=torch.bfloat16)
        carry = torch.empty(owned, h, device=device)
        generations = torch.zeros(owned, device=device, dtype=torch.int32)
        error = torch.zeros(1, device=device, dtype=torch.int32)
        ref = full_chain.allocate(x, 2)
        for name in ("native", "owner"):
            scatter.copy_(snapshot)
            torch.accelerator.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push(f"rows{rows}_{name}")
            if name == "native":
                torch.ops._C.gemma4_full_chain_out(
                    x,
                    residual,
                    post,
                    pre,
                    scalar,
                    *ref,
                    9,
                    1e-6,
                )
            else:
                rc = lib.launch_owner(
                    scatter.data_ptr(),
                    rows,
                    0,
                    0,
                    error.data_ptr(),
                    owned_residual.data_ptr(),
                    post.data_ptr(),
                    pre.data_ptr(),
                    carry.data_ptr(),
                    *(handle.multicast_ptr for _, handle in shared[:2]),
                    ready_peers.data_ptr(),
                    generations.data_ptr(),
                    2,
                    scalar.data_ptr(),
                    torch.cuda.current_stream().cuda_stream,
                    0,
                    args.publication,
                )
                assert rc == 0, rc
            torch.accelerator.synchronize()
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
        assert error.item() == 0
        assert torch.equal(carry, ref[0][::8])
        assert torch.equal(
            shared[0][0][::8].view(torch.uint8), ref[1][::8].view(torch.uint8)
        )
        assert torch.equal(shared[1][0][:, ::8][:, :owned], ref[2][:, :rows][:, ::8])
        print("NCU ready-input owner exact", rows, flush=True)
    dist.barrier(group=cpu)
    retained.append(shared)

dist.destroy_process_group()
