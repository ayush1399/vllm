# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build isolated SM90 experiments against the installed FlashInfer headers."""

import argparse
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--build-dir", type=Path, required=True)
parser.add_argument(
    "--only",
    nargs="+",
    default=["owner_norm", "projection", "next_gemm", "replicated_norm"],
)
args = parser.parse_args()
build = args.build_dir.resolve()
build.mkdir(parents=True, exist_ok=True)
source = Path(__file__).resolve().parent
spec = importlib.util.find_spec("flashinfer")
assert spec and spec.submodule_search_locations
fi = Path(next(iter(spec.submodule_search_locations))) / "data"
vendor = fi / "csrc/nv_internal/tensorrt_llm/deep_gemm"
for name in ("deep_gemm", "next_deep_gemm"):
    shutil.copytree(vendor, build / name, dirs_exist_ok=True)

for patch, target in (("projection", "deep_gemm"), ("following", "next_deep_gemm")):
    subprocess.run(
        [
            "patch",
            "--batch",
            "--forward",
            "-p1",
            "-i",
            str(source / (patch + ".patch")),
        ],
        cwd=build / target,
        check=True,
    )

for name in args.only:
    command = [
        "/usr/local/cuda/bin/nvcc",
        "-std=c++17",
        "-O3",
        "-lineinfo",
        "-gencode=arch=compute_90a,code=sm_90a",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--threads",
        "4",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-Xptxas",
        "-v",
        "-I",
        str(build),
        "-I",
        str(fi / "cutlass/include"),
        "-I",
        str(fi / "csrc"),
        str(source / (name + ".cu")),
        "-L/usr/local/cuda/lib64",
        "-lcuda",
        "-lcudart",
        "-o",
        str(build / (name + ".pending.so")),
    ]
    print("Building", name, flush=True)
    subprocess.run(command, check=True)
    os.replace(build / (name + ".pending.so"), build / (name + ".so"))
