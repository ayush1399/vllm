# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched Gemma4 FP8 MTP generation timing and optional offline GSM8K evaluation.

Run once per source tree and k, with identical flags and checkpoints. Generation
latency includes prefill, drafting, verification, sampling and CPU overhead.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def inspect_worker(worker):
    import torch

    from vllm.model_executor.models.gemma4 import (
        Gemma4Model,
    )
    from vllm.models.gemma4.nvidia.specialized_decode.model import (
        _gemma4_can_run_full_chain,
    )

    runner = worker.model_runner
    target = next(m for m in runner.model.modules() if isinstance(m, Gemma4Model))
    rows = runner.vllm_config.speculative_config.num_speculative_tokens + 1
    x = torch.empty((rows, 5376), dtype=torch.bfloat16, device=worker.device)
    drafter = getattr(runner, "speculator", None) or runner.drafter
    draft = drafter.model.model
    return {
        "model_runner": type(runner).__module__ + "." + type(runner).__name__,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(worker.device),
        "target_full_chain": _gemma4_can_run_full_chain(target, x, None),
        "replicated_pre_projection": getattr(draft, "_mtp_pre_weight", None)
        is not None,
        "draft_raw_gemv": [
            getattr(layer.self_attn.o_proj, "_gemma4_mtp_fp8_gemv", False)
            for layer in draft.layers
        ],
        "draft_norm_quant": [
            getattr(layer, "_fuse_norm_quant", False) for layer in draft.layers
        ],
        "draft_q_fp8_input": [
            getattr(layer.self_attn.q_proj, "_flashinfer_fp8_input_ue8m0", False)
            for layer in draft.layers
        ],
        "draft_mlp_fp8_input": [
            getattr(layer.mlp.gate_up_proj, "_flashinfer_fp8_input_ue8m0", False)
            for layer in draft.layers
        ],
    }


class WorkerExtension:
    def inspect_gemma4_mtp(self):
        return inspect_worker(self)


def token_hash(output):
    return hashlib.sha256(
        bytes(json.dumps(output.outputs[0].token_ids), "utf-8")
    ).hexdigest()


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--k", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--input-lens", type=int, nargs="+", default=[128, 2048, 20000])
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--gsm8k-train", type=Path)
    parser.add_argument("--gsm8k-test", type=Path)
    parser.add_argument("--num-questions", type=int, default=32)
    parser.add_argument("--quality-max-tokens", type=int, default=1024)
    parser.add_argument(
        "--decode-backend", choices=["auto", "latency_specialized"], default="auto"
    )
    args = parser.parse_args()
    if (args.gsm8k_train is None) != (args.gsm8k_test is None):
        parser.error("Pass both --gsm8k-train and --gsm8k-test")
    if args.repeat < 1 or args.warmup < 0:
        parser.error("--repeat must be positive and --warmup nonnegative")
    llm = LLM(
        model=args.model,
        additional_config={"decode_backend": args.decode_backend},
        profiler_config=(
            {
                "profiler": "torch",
                "torch_profiler_dir": str(args.profile_dir),
                "torch_profiler_with_stack": False,
                "torch_profiler_record_shapes": True,
            }
            if args.profile_dir
            else {}
        ),
        worker_extension_cls=(
            "benchmarks.multi_turn.benchmark_gemma4_mtp.WorkerExtension"
        ),
        tensor_parallel_size=8,
        dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        max_model_len=max(4096, max(args.input_lens) + args.output_len + 32),
        max_num_seqs=max(args.batch_sizes),
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.6,
        language_model_only=True,
        enable_prefix_caching=False,
        seed=0,
        enforce_eager=args.eager,
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 4},
        compilation_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": list(range(1, 33)),
        },
        hf_overrides={"text_config": {"use_bidirectional_attention": None}},
        speculative_config={
            "method": "mtp",
            "model": args.draft_model,
            "num_speculative_tokens": args.k,
        },
    )
    package = Path(__import__("vllm").__file__).resolve().parent
    source_paths = [
        "model_executor/models/gemma4.py",
        "model_executor/models/gemma4_mtp.py",
        "models/gemma4/nvidia/specialized_decode/config.py",
        "models/gemma4/nvidia/specialized_decode/linear.py",
        "models/gemma4/nvidia/specialized_decode/model.py",
        "models/gemma4/nvidia/specialized_decode/mtp.py",
        "models/gemma4/nvidia/specialized_decode/norm.py",
        "models/gemma4/nvidia/specialized_decode/full_chain.py",
        "models/gemma4/nvidia/specialized_decode/mtp_kernels.py",
        "model_executor/kernels/linear/__init__.py",
        "model_executor/kernels/linear/scaled_mm/flashinfer.py",
        "_C_stable_libtorch.abi3.so",
    ]
    result = {
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "source": str(package),
        "source_sha256": {
            p: hashlib.sha256((package / p).read_bytes()).hexdigest()
            for p in source_paths
            if (package / p).exists()
        },
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k
            in {
                "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER",
                "VLLM_USE_DEEP_GEMM_E8M0",
                "VLLM_DEEP_GEMM_WARMUP",
                "VLLM_USE_V2_MODEL_RUNNER",
                "VLLM_BATCH_INVARIANT",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                "TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION",
                "OMP_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
            }
        },
        "workers": llm.collective_rpc("inspect_gemma4_mtp"),
        "timing_boundary": "LLM.generate wall time, prefix caching disabled",
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    tokenizer = llm.get_tokenizer()
    seed_ids = tokenizer.encode("The quick brown fox jumps over the lazy dog. ")
    params = SamplingParams(
        temperature=0, max_tokens=args.output_len, ignore_eos=True, detokenize=False
    )
    for length in args.input_lens:
        ids = (seed_ids * (length // len(seed_ids) + 1))[:length]
        for batch in args.batch_sizes:
            prompts = [{"prompt_token_ids": ids} for _ in range(batch)]
            samples = []
            for iteration in range(args.warmup + args.repeat):
                start = time.perf_counter()
                outputs = llm.generate(prompts, params, use_tqdm=False)
                elapsed = time.perf_counter() - start
                assert all(
                    len(o.outputs[0].token_ids) == args.output_len for o in outputs
                )
                if iteration >= args.warmup:
                    samples.append(elapsed)
            result["cases"].append(
                {
                    "input_len": length,
                    "batch_size": batch,
                    "samples_s": samples,
                    "median_s": statistics.median(samples),
                    "output_hashes": [token_hash(o) for o in outputs],
                }
            )
            save()
    if args.gsm8k_test:
        from benchmarks.multi_turn.evaluate_gemma4_quality import load_protocol

        train = [json.loads(s) for s in args.gsm8k_train.read_text().splitlines()]
        test = [json.loads(s) for s in args.gsm8k_test.read_text().splitlines()]
        source = Path(__file__).resolve().parents[2] / "tests/evals/gsm8k/gsm8k_eval.py"
        protocol, protocol_sha = load_protocol(source, train, test)
        prompts, labels = protocol["_build_gsm8k_prompts"](args.num_questions, 5)
        start = time.perf_counter()
        outputs = llm.chat(
            [[{"role": "user", "content": p}] for p in prompts],
            SamplingParams(temperature=0, max_tokens=args.quality_max_tokens),
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - start
        scores = protocol["_score_gsm8k"](
            [o.outputs[0].text for o in outputs],
            [len(o.outputs[0].token_ids) for o in outputs],
            labels,
            5,
            args.quality_max_tokens,
            elapsed,
        )
        result["quality"] = {
            **scores,
            "protocol_sha256": protocol_sha,
            "train_sha256": hashlib.sha256(args.gsm8k_train.read_bytes()).hexdigest(),
            "test_sha256": hashlib.sha256(args.gsm8k_test.read_bytes()).hexdigest(),
            "truncated": sum(o.outputs[0].finish_reason == "length" for o in outputs),
            "correct": [
                protocol["get_answer_value"](o.outputs[0].text) == label
                for o, label in zip(outputs, labels)
            ],
            "output_hashes": [token_hash(o) for o in outputs],
        }
        save()

    if args.profile_dir:
        length = args.input_lens[0]
        ids = (seed_ids * (length // len(seed_ids) + 1))[:length]
        profile_params = SamplingParams(
            temperature=0,
            max_tokens=min(16, args.output_len),
            ignore_eos=True,
            detokenize=False,
        )
        for batch in args.batch_sizes:
            prompts = [{"prompt_token_ids": ids} for _ in range(batch)]
            for _ in range(args.warmup):
                llm.generate(prompts, profile_params, use_tqdm=False)
            llm.start_profile(f"mtp-k{args.k}-q{length}-c{batch}")
            llm.generate(prompts, profile_params, use_tqdm=False)
            llm.stop_profile()


if __name__ == "__main__":
    main()
