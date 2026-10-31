# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure Gemma4 MTP acceptance on a fixed JSONL chat prompt manifest.

Each line contains dataset, id and messages. Run from both source checkouts
with identical checkpoints, manifest, sampling and execution settings.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from benchmarks.multi_turn.benchmark_gemma4_mtp import token_hash
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector


def snapshot(llm: LLM, k: int) -> dict:
    metrics = llm.get_metrics()
    result = {}
    for field in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
        selected = [m for m in metrics if m.name == f"vllm:spec_decode_{field}"]
        assert len(selected) == 1 and isinstance(selected[0], Counter)
        result[field] = selected[0].value
    selected = [
        m for m in metrics if m.name == "vllm:spec_decode_num_accepted_tokens_per_pos"
    ]
    assert len(selected) == 1 and isinstance(selected[0], Vector)
    assert len(selected[0].values) == k
    result["accepted_per_position"] = selected[0].values
    return result


def difference(before: dict, after: dict) -> dict:
    delta = {
        field: after[field] - before[field]
        for field in ("num_drafts", "num_draft_tokens", "num_accepted_tokens")
    }
    delta["accepted_per_position"] = [
        a - b
        for a, b in zip(after["accepted_per_position"], before["accepted_per_position"])
    ]
    assert delta["num_drafts"] > 0
    assert 0 <= delta["num_accepted_tokens"] <= delta["num_draft_tokens"]
    assert sum(delta["accepted_per_position"]) == delta["num_accepted_tokens"]
    assert all(v >= 0 for v in delta["accepted_per_position"])
    return delta


def summarize(groups: list[dict], k: int) -> dict:
    totals = {
        field: sum(g["counters"][field] for g in groups)
        for field in ("num_drafts", "num_draft_tokens", "num_accepted_tokens")
    }
    totals["accepted_per_position"] = [
        sum(g["counters"]["accepted_per_position"][i] for g in groups) for i in range(k)
    ]
    return {
        **totals,
        "acceptance_rate": totals["num_accepted_tokens"] / totals["num_draft_tokens"],
        "mean_acceptance_length": 1
        + totals["num_accepted_tokens"] / totals["num_drafts"],
        "accepted_per_position_per_draft": [
            v / totals["num_drafts"] for v in totals["accepted_per_position"]
        ],
        "output_tokens": sum(sum(g["output_lengths"]) for g in groups),
        "truncated": sum(g["finish_reasons"].count("length") for g in groups),
    }


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    records = [json.loads(s) for s in args.prompts.read_text().splitlines()]
    datasets = sorted({r["dataset"] for r in records})
    llm = LLM(
        model=args.model,
        worker_extension_cls=(
            "benchmarks.multi_turn.benchmark_gemma4_mtp.WorkerExtension"
        ),
        tensor_parallel_size=8,
        dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        max_model_len=8192,
        max_num_seqs=max(args.batch_sizes),
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.6,
        language_model_only=True,
        enable_prefix_caching=False,
        disable_log_stats=False,
        seed=0,
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
    paths = [
        "model_executor/models/gemma4.py",
        "model_executor/models/gemma4_mtp.py",
        "model_executor/layers/gemma4_full_chain.py",
        "model_executor/layers/gemma4_mtp_fp8.py",
        "model_executor/kernels/linear/scaled_mm/flashinfer.py",
        "_C_stable_libtorch.abi3.so",
    ]
    result = {
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source": str(package),
        "source_sha256": {
            p: hashlib.sha256((package / p).read_bytes()).hexdigest()
            for p in paths
            if (package / p).exists()
        },
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            k: os.environ.get(k)
            for k in (
                "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER_FP8_INPUT",
                "VLLM_USE_DEEP_GEMM_E8M0",
                "VLLM_DEEP_GEMM_WARMUP",
                "VLLM_BATCH_INVARIANT",
                "VLLM_USE_V2_MODEL_RUNNER",
                "OMP_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                "TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION",
            )
        },
        "prompt_manifest_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "workers": llm.collective_rpc("inspect_gemma4_mtp"),
        "definitions": {
            "acceptance_rate": "accepted draft tokens / proposed draft tokens",
            "mean_acceptance_length": "1 + accepted draft tokens / draft rounds",
            "accepted_per_position_per_draft": (
                "accepted at position / draft rounds; unconditional, not conditional"
            ),
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    tokenizer = llm.get_tokenizer()
    for r in records:
        r["tokens"] = tokenizer.apply_chat_template(
            r["messages"],
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        assert isinstance(r["tokens"], list) and all(
            isinstance(t, int) for t in r["tokens"]
        )
        assert len(r["tokens"]) + args.max_tokens <= 8192
    params = SamplingParams(
        temperature=0, seed=0, max_tokens=args.max_tokens, detokenize=False
    )
    save()
    for dataset in datasets:
        selected = [r for r in records if r["dataset"] == dataset]
        for batch_size in args.batch_sizes:
            warmup = [{"prompt_token_ids": r["tokens"]} for r in selected[:batch_size]]
            llm.generate(warmup, params, use_tqdm=False)
            previous = snapshot(llm, args.k)
            case = {"dataset": dataset, "batch_size": batch_size, "groups": []}
            result["cases"].append(case)
            for start in range(0, len(selected), batch_size):
                group = selected[start : start + batch_size]
                outputs = llm.generate(
                    [{"prompt_token_ids": r["tokens"]} for r in group],
                    params,
                    use_tqdm=False,
                )
                current = snapshot(llm, args.k)
                case["groups"].append(
                    {
                        "ids": [r["id"] for r in group],
                        "prompt_token_hashes": [
                            hashlib.sha256(json.dumps(r["tokens"]).encode()).hexdigest()
                            for r in group
                        ],
                        "counters": difference(previous, current),
                        "output_hashes": [token_hash(o) for o in outputs],
                        "output_lengths": [
                            len(o.outputs[0].token_ids) for o in outputs
                        ],
                        "finish_reasons": [o.outputs[0].finish_reason for o in outputs],
                    }
                )
                previous = current
                case["summary"] = summarize(case["groups"], args.k)
                save()
            print(dataset, batch_size, case["summary"], flush=True)


if __name__ == "__main__":
    main()
