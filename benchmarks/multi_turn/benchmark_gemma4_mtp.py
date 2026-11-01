# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched Gemma4 FP8 MTP generation timing and optional offline GSM8K evaluation.

Run once per source tree and k, with identical flags and checkpoints. Generation
latency includes prefill, drafting, verification, sampling and CPU overhead.
Queue each complete batch while scheduling is paused, then time resume to finish.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from vllm import LLM, RequestOutput, SamplingParams


def inspect_worker(worker):
    import inspect

    import torch

    import vllm
    from vllm.model_executor.models.gemma4 import (
        Gemma4Model,
    )

    runner = worker.model_runner
    config = runner.vllm_config
    compilation = config.compilation_config
    target = next(m for m in runner.model.modules() if isinstance(m, Gemma4Model))
    rows = runner.vllm_config.speculative_config.num_speculative_tokens + 1
    x = torch.empty((rows, 5376), dtype=torch.bfloat16, device=worker.device)
    drafter = getattr(runner, "speculator", None) or runner.drafter
    draft = drafter.model.model
    full_chain = False
    if getattr(target, "_full_chain_config_supported", False):
        from vllm.models.gemma4.nvidia.specialized_decode.model import (
            _gemma4_can_run_full_chain,
        )

        full_chain = _gemma4_can_run_full_chain(target, x, None)
    return {
        "runtime_config": {
            "activation_dtype": str(config.model_config.dtype),
            "kv_cache_dtype": config.cache_config.cache_dtype,
            "max_model_len": config.model_config.max_model_len,
            "tensor_parallel_size": config.parallel_config.tensor_parallel_size,
            "pipeline_parallel_size": config.parallel_config.pipeline_parallel_size,
            "max_num_seqs": config.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
            "enable_prefix_caching": config.cache_config.enable_prefix_caching,
            "compile_mode": str(compilation.mode),
            "capture_sizes": compilation.cudagraph_capture_sizes,
            "custom_ops": sorted(compilation.custom_ops),
            "inductor_compile_config": compilation.inductor_compile_config,
            "ir_op_priority": str(config.kernel_config.ir_op_priority),
        },
        "package_root": str(Path(vllm.__file__).resolve().parent),
        "target_source": inspect.getfile(type(target)),
        "foreign_vllm_modules": {
            name: str(Path(module.__file__).resolve())
            for name, module in sys.modules.copy().items()
            if name.startswith("vllm.")
            and getattr(module, "__file__", "")
            and not Path(module.__file__)
            .resolve()
            .is_relative_to(Path(vllm.__file__).resolve().parent)
        },
        "native_modules": {
            name: str(Path(module.__file__).resolve())
            for name, module in sys.modules.copy().items()
            if name.startswith("vllm.")
            and getattr(module, "__file__", "")
            and str(module.__file__).endswith(".so")
        },
        "model_runner": type(runner).__module__ + "." + type(runner).__name__,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(worker.device),
        "cuda_graph_mode": str(runner.vllm_config.compilation_config.cudagraph_mode),
        "cuda_graph_max_splits": (
            runner.vllm_config.attention_config.flash_attn_max_num_splits_for_cuda_graph
        ),
        "target_attention_backends": sorted(
            {
                type(layer.self_attn.attn.impl).__module__
                + "."
                + type(layer.self_attn.attn.impl).__name__
                for layer in target.layers
            }
        ),
        "draft_attention_backends": sorted(
            {
                type(layer.self_attn.attn.impl).__module__
                + "."
                + type(layer.self_attn.attn.impl).__name__
                for layer in draft.layers
            }
        ),
        "target_full_chain": full_chain,
        "target_pdl_gate_up": [
            getattr(layer.mlp.gate_up_proj, "_gemma4_pdl", False)
            for layer in target.layers
        ],
        "target_pdl_qkv": [
            getattr(layer.self_attn.qkv_proj, "_gemma4_pdl", False)
            for layer in target.layers
        ],
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

    def gemma4_jit_events(self):
        from vllm.utils import jit_monitor

        if not hasattr(self, "_gemma4_jit_events"):
            self._gemma4_jit_events = []
            handle = jit_monitor._handle_jit_event

            def record(**event):
                self._gemma4_jit_events.append(event)
                return handle(**event)

            jit_monitor._handle_jit_event = record
        events, self._gemma4_jit_events = self._gemma4_jit_events, []
        return events

    def gemma4_trace_stages(self):
        import functools

        import torch

        if getattr(self, "_gemma4_trace_stages", False):
            return
        self._gemma4_trace_stages = True
        use_torch = self.vllm_config.profiler_config.profiler == "torch"

        def wrap(obj, name, label):
            if not hasattr(obj, name):
                return
            original = getattr(obj, name)

            @functools.wraps(original)
            def marked(*args, **kwargs):
                context = (
                    torch.profiler.record_function(label)
                    if use_torch
                    else torch.cuda.nvtx.range(label)
                )
                with context:
                    return original(*args, **kwargs)

            setattr(obj, name, marked)

        runner = self.model_runner
        for name in (
            "execute_model",
            "prepare_inputs",
            "prepare_attn",
            "sample",
            "sample_tokens",
            "postprocess_sampled",
        ):
            wrap(runner, name, f"gemma4/{name}")
        if getattr(runner, "speculator", None) is not None:
            for name in ("propose", "_run_model", "compute_logits"):
                wrap(runner.speculator, name, f"gemma4/draft/{name}")


def token_hash(output):
    return hashlib.sha256(
        bytes(json.dumps(output.outputs[0].token_ids), "utf-8")
    ).hexdigest()


def generate_prequeued(llm, prompts, params):
    llm.sleep(level=0, mode="keep")
    llm.enqueue(prompts, params, use_tqdm=False)
    start = time.perf_counter()
    llm.wake_up(tags=["scheduling"])
    outputs = llm.wait_for_completion(output_type=RequestOutput, use_tqdm=False)
    return outputs, time.perf_counter() - start


def generation_timings(outputs: list[RequestOutput]) -> dict:
    """Use engine timestamps; batched requests can overlap prefill and decode."""
    requests = []
    for output in outputs:
        metrics = output.metrics
        if metrics is None:
            raise RuntimeError("Request timings require disable_log_stats=False")
        if not (
            0
            < metrics.queued_ts
            <= metrics.scheduled_ts
            <= metrics.first_token_ts
            <= metrics.last_token_ts
        ):
            raise RuntimeError(f"Invalid request timestamps: {metrics}")
        spec = output.outputs[0].spec_decode_metrics
        requests.append(
            {
                "request_id": output.request_id,
                "prompt_tokens": len(output.prompt_token_ids),
                "output_tokens": len(output.outputs[0].token_ids),
                "cached_tokens": output.num_cached_tokens,
                "preemptions": metrics.num_preemptions,
                "queue_s": metrics.scheduled_ts - metrics.queued_ts,
                "prefill_s": metrics.first_token_ts - metrics.scheduled_ts,
                "decode_s": metrics.last_token_ts - metrics.first_token_ts,
                "ttft_s": metrics.first_token_latency,
                "scheduled_ts": metrics.scheduled_ts,
                "first_token_ts": metrics.first_token_ts,
                "last_token_ts": metrics.last_token_ts,
                "speculative_decoding": spec.to_dict() if spec else None,
            }
        )
    scheduled = min(r["scheduled_ts"] for r in requests)
    first = min(r["first_token_ts"] for r in requests)
    last_first = max(r["first_token_ts"] for r in requests)
    last = max(r["last_token_ts"] for r in requests)
    return {
        "requests": requests,
        "engine_span_s": last - scheduled,
        "initial_prefill_s": first - scheduled,
        "mixed_prefill_decode_s": last_first - first,
        "decode_tail_s": last - last_first,
    }


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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
    parser.add_argument("--profile-tool", choices=["torch", "cuda"], default="torch")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--profile-output-len", type=int, default=16)
    parser.add_argument("--profile-stack", action="store_true")
    parser.add_argument("--gsm8k-train", type=Path)
    parser.add_argument("--gsm8k-test", type=Path)
    parser.add_argument("--num-questions", type=int, default=32)
    parser.add_argument("--quality-max-tokens", type=int, default=1024)
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--accepted-draft-tokens", type=int)
    parser.add_argument(
        "--gemma4-pdl", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--decode-backend", choices=["auto", "latency_specialized"], default="auto"
    )
    args = parser.parse_args()
    if (args.gsm8k_train is None) != (args.gsm8k_test is None):
        parser.error("Pass both --gsm8k-train and --gsm8k-test")
    if args.repeat < 1 or args.warmup < 0:
        parser.error("--repeat must be positive and --warmup nonnegative")
    if args.quality_only != (args.gsm8k_test is not None):
        parser.error("Use --quality-only with GSM8K files; run performance separately")
    if args.quality_only and args.profile_dir:
        parser.error("Run performance profiling separately with controlled acceptance")
    if args.profile_only and not args.profile_dir:
        parser.error("--profile-only requires --profile-dir")
    if args.profile_output_len < 1:
        parser.error("--profile-output-len must be positive")
    accepted = (
        args.k if args.accepted_draft_tokens is None else args.accepted_draft_tokens
    )
    if not 0 <= accepted <= args.k:
        parser.error("--accepted-draft-tokens must be between zero and k")
    speculative_config = {
        "method": "mtp",
        "model": args.draft_model,
        "num_speculative_tokens": args.k,
    }
    if not args.quality_only:
        speculative_config.update(
            rejection_sample_method="synthetic",
            synthetic_acceptance_rates=[1.0] * accepted + [0.0] * (args.k - accepted),
        )
    llm = LLM(
        model=args.model,
        additional_config=(
            {
                "decode_backend": args.decode_backend,
                "gemma4_pdl": args.gemma4_pdl,
            }
            if args.decode_backend != "auto"
            else {}
        ),
        profiler_config=(
            (
                {"profiler": "cuda"}
                if args.profile_tool == "cuda"
                else {
                    "profiler": "torch",
                    "torch_profiler_dir": str(args.profile_dir),
                    "torch_profiler_with_stack": args.profile_stack,
                    "torch_profiler_record_shapes": True,
                }
            )
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
        disable_log_stats=False,
        per_request_spec_decode_metrics="summary",
        enforce_eager=args.eager,
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 4},
        compilation_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": list(range(1, 33)),
        },
        hf_overrides={"text_config": {"use_bidirectional_attention": None}},
        speculative_config=speculative_config,
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
        "models/gemma4/nvidia/specialized_decode/pdl.py",
        "models/gemma4/nvidia/specialized_decode/pdl_driver.cu",
        "models/gemma4/nvidia/specialized_decode/pdl_gemm.cuh",
        "models/gemma4/nvidia/specialized_decode/full_chain.cuh",
        "models/gemma4/nvidia/specialized_decode/activation.py",
        "models/gemma4/nvidia/specialized_decode/attention.py",
        "models/gemma4/nvidia/specialized_decode/attention_prep.cu",
        "models/gemma4/nvidia/specialized_decode/gemma4_full_chain.cu",
        "model_executor/kernels/linear/__init__.py",
        "model_executor/kernels/linear/scaled_mm/flashinfer.py",
        "_C_stable_libtorch.abi3.so",
        "third_party/deep_gemm/_C_extension.abi3.so",
        "third_party/deep_gemm/_C.py",
        "vllm_flash_attn/cute/flash_fwd_sm90.py",
        "vllm_flash_attn/cute/interface.py",
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
        "timing_boundary": (
            "Prequeued batch: scheduler resume through completion, prefix caching "
            "disabled. Pause/enqueue excluded; resume RPC and CPU overhead included."
        ),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "acceptance_policy": speculative_config,
        "phase_timing_boundary": {
            "clock": "engine monotonic timestamps; TTFT uses frontend wall clock",
            "request_prefill": "first scheduled to first token, including sampling",
            "request_decode": "first token to last token",
            "batch_initial_prefill": "first scheduled to earliest first token",
            "batch_mixed": "earliest to latest first token; prefill/decode may overlap",
            "batch_decode_tail": "latest first token to latest last token",
            "overhead": "engine timestamps include scheduling and CPU overhead",
        },
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
    for length in [] if args.quality_only or args.profile_only else args.input_lens:
        ids = (seed_ids * (length // len(seed_ids) + 1))[:length]
        for batch in args.batch_sizes:
            prompts = [{"prompt_token_ids": ids} for _ in range(batch)]
            samples = []
            phase_samples = []
            discarded = []
            llm.collective_rpc("gemma4_jit_events")
            for attempt in range(5):
                for _ in range(args.warmup):
                    generate_prequeued(llm, prompts, params)
                llm.collective_rpc("gemma4_jit_events")
                samples, phase_samples = [], []
                for _ in range(args.repeat):
                    outputs, elapsed = generate_prequeued(llm, prompts, params)
                    assert all(
                        len(o.outputs[0].token_ids) == args.output_len for o in outputs
                    )
                    samples.append(elapsed)
                    phase_samples.append(generation_timings(outputs))
                events = llm.collective_rpc("gemma4_jit_events")
                if not any(events):
                    break
                discarded.append({"samples_s": samples, "jit_events": events})
                print(
                    f"Repeating entire q{length}/b{batch} block after runtime JIT",
                    flush=True,
                )
            else:
                raise RuntimeError(f"Runtime JIT persisted for q{length}/b{batch}")
            result["cases"].append(
                {
                    "input_len": length,
                    "batch_size": batch,
                    "samples_s": samples,
                    "median_s": statistics.median(samples),
                    "discarded_jit_blocks": discarded,
                    "timed_jit_events": events,
                    "output_hashes": [token_hash(o) for o in outputs],
                    "phase_samples": phase_samples,
                    "phase_median_s": {
                        name: statistics.median(s[name] for s in phase_samples)
                        for name in (
                            "engine_span_s",
                            "initial_prefill_s",
                            "mixed_prefill_decode_s",
                            "decode_tail_s",
                        )
                    },
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
        profile_params = SamplingParams(
            temperature=0,
            max_tokens=min(args.profile_output_len, args.output_len),
            ignore_eos=True,
            detokenize=False,
        )
        result["profiles"] = []
        for length in args.input_lens:
            ids = (seed_ids * (length // len(seed_ids) + 1))[:length]
            for batch in args.batch_sizes:
                prompts = [{"prompt_token_ids": ids} for _ in range(batch)]
                llm.collective_rpc("gemma4_jit_events")
                for _ in range(args.warmup):
                    generate_prequeued(llm, prompts, profile_params)
                warmup_jit = llm.collective_rpc("gemma4_jit_events")
                llm.collective_rpc("gemma4_trace_stages")
                llm.start_profile(f"mtp-k{args.k}-q{length}-c{batch}")
                outputs, elapsed = generate_prequeued(llm, prompts, profile_params)
                llm.stop_profile()
                events = llm.collective_rpc("gemma4_jit_events")
                result["profiles"].append(
                    {
                        "input_len": length,
                        "batch_size": batch,
                        "output_len": profile_params.max_tokens,
                        "profiled_wall_s": elapsed,
                        "phase_sample": generation_timings(outputs),
                        "warmup_jit_events": warmup_jit,
                        "profiled_jit_events": events,
                    }
                )
                save()


if __name__ == "__main__":
    main()
