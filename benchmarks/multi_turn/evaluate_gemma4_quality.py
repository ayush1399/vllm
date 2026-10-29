# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline-data GSM8K five-shot serving evaluation with per-item checkpoints."""

import argparse
import ast
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import regex as re

NUM_SHOTS = 5
MAX_TOKENS = 1024
CONCURRENCY = 4
TOTAL_QUESTIONS = 1319


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def utc_now():
    return datetime.now(UTC).isoformat()


def count_arg(value):
    count = TOTAL_QUESTIONS if value == "all" else int(value)
    if not 1 <= count <= TOTAL_QUESTIONS:
        raise argparse.ArgumentTypeError("count must be 1..1319 or all")
    return count


def load_protocol(path, train, test):
    """Load the exact existing builder/scorer without importing vLLM or CUDA."""
    import numpy as np
    import regex

    source = path.read_text()
    wanted = {"get_answer_value", "_build_gsm8k_prompts", "_score_gsm8k"}
    selected = []
    found = set()
    for node in ast.parse(source, filename=str(path)).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            selected.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "INVALID"
            for target in node.targets
        ):
            selected.append(node)
    if found != wanted:
        raise RuntimeError(f"Evaluation source is missing functions: {wanted - found}")
    namespace = {
        "ast": ast,
        "np": np,
        "re": regex,
        "time": time,
        "load_gsm8k_data": lambda: (train, test),
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    if "INVALID" not in namespace:
        raise RuntimeError("Evaluation source is missing INVALID")
    return namespace, sha256_bytes(source.encode())


def strict_answer(text):
    """Require a final #### marker followed only by a decimal and punctuation."""
    if "####" not in text:
        return None
    final = text.rsplit("####", 1)[1].strip()
    # Retain the C0 separators accepted by the measured Python whitespace class.
    match = re.fullmatch(
        r"\$?[\s\x1c-\x1f]*"
        r"([-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))"
        r"[\s\x1c-\x1f]*\.?",
        final,
    )
    if not match:
        return None
    try:
        return Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None


def atomic_checkpoint(path, checkpoint):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(checkpoint, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_rows(path, expected_count):
    raw = path.read_bytes()
    rows = [
        json.loads(line)
        for line in raw.splitlines()
        if line.strip() and not line.startswith(b"#")
    ]
    if len(rows) != expected_count:
        raise ValueError(f"{path}: expected {expected_count} rows, got {len(rows)}")
    for index, row in enumerate(rows):
        if not isinstance(row.get("question"), str) or not isinstance(
            row.get("answer"), str
        ):
            raise ValueError(f"{path}: invalid row {index}")
        if strict_answer(row["answer"]) is None:
            raise ValueError(f"{path}: invalid reference answer at row {index}")
    return rows, {
        "path": str(path.resolve()),
        "rows": len(rows),
        "sha256": sha256_bytes(raw),
    }


def make_payload(prompt, model):
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "seed": 42,
        "top_p": 1,
        "top_k": -1,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def validate_response(result):
    if not isinstance(result, dict) or result.get("error") is not None:
        raise ValueError("Non-object response or API error response")
    choices = result.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Expected exactly one response choice")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Missing or empty final answer content")
    if not isinstance(choice.get("finish_reason"), str):
        raise ValueError("Missing finish_reason")
    usage = result.get("usage")
    if (
        not isinstance(usage, dict)
        or not isinstance(usage.get("completion_tokens"), int)
        or usage["completion_tokens"] < 0
    ):
        raise ValueError("Missing or invalid completion_tokens usage")
    return content, choice["finish_reason"], usage


async def evaluate(args):
    import aiohttp

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not args.output.parent.is_dir():
        raise ValueError(f"Output directory does not exist: {args.output.parent}")
    train, train_info = load_rows(args.data_dir / "train.jsonl", 7473)
    test, test_info = load_rows(args.data_dir / "test.jsonl", TOTAL_QUESTIONS)
    source = args.repo / "tests/evals/gsm8k/gsm8k_eval.py"
    protocol, source_hash = load_protocol(source, train, test)
    prompts, labels = protocol["_build_gsm8k_prompts"](
        num_questions=args.count, num_shots=NUM_SHOTS, gen_prefix=""
    )
    endpoint = args.host.rstrip("/") + f":{args.port}/v1/chat/completions"
    payloads = [make_payload(prompt, args.model) for prompt in prompts]
    payload_bytes = [json_bytes(payload) for payload in payloads]
    checkpoint = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "config": {
            "label": args.label or args.output.stem,
            "endpoint": endpoint,
            "model": args.model,
            "num_questions": len(prompts),
            "num_shots": NUM_SHOTS,
            "gen_prefix": "",
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "seed": 42,
            "top_p": 1,
            "top_k": -1,
            "chat_template_kwargs": {"enable_thinking": False},
            "stop": None,
            "concurrency": CONCURRENCY,
            "request_timeout_seconds": args.request_timeout_seconds,
            "datasets": {"train": train_info, "test": test_info},
            "evaluation_source": str(source.resolve()),
            "evaluation_source_sha256": source_hash,
            "wrapper_sha256": sha256_bytes(Path(__file__).read_bytes()),
            "python": sys.executable,
            "prompt_hash_encoding": "UTF-8",
            "payload_hash_encoding": "canonical sorted UTF-8 JSON; exact sent bytes",
            "plan_sha256": sha256_bytes(json_bytes(payloads)),
            "strict_scoring": (
                "Last #### marker followed only by decimal, optional dollar "
                "sign and final period; Decimal equality"
            ),
            "client_instrumentation": "none",
            "prefix_cache_reset": False,
            "server_instrumentation": (
                "not verified; consult external launch provenance"
            ),
            "server_provenance": (
                "External launch provenance must identify BF16/original "
                "FP8/candidate and no-spec TP8 configuration"
            ),
        },
        "completed_items": 0,
        "failed_items": 0,
        "aggregate": None,
        "results": [None] * len(prompts),
    }
    atomic_checkpoint(args.output, checkpoint)
    tic = time.perf_counter()
    queue = asyncio.Queue()
    for index in range(len(prompts)):
        queue.put_nowait(index)

    async def worker(session):
        while not queue.empty():
            index = queue.get_nowait()
            started_at = utc_now()
            started = time.perf_counter()
            base = {
                "index": index,
                "question": test[index]["question"],
                "prompt": prompts[index],
                "prompt_sha256": sha256_bytes(prompts[index].encode("utf-8")),
                "payload_sha256": sha256_bytes(payload_bytes[index]),
                "reference_answer": test[index]["answer"],
                "established_label": labels[index],
                "strict_label": str(strict_answer(test[index]["answer"])),
                "started_at": started_at,
            }
            try:
                async with session.post(
                    endpoint,
                    data=payload_bytes[index],
                    headers={"Content-Type": "application/json"},
                ) as response:
                    body = await response.text()
                    if response.status < 200 or response.status >= 300:
                        raise RuntimeError(f"HTTP {response.status}: {body[:500]}")
                    result = json.loads(body)
                content, finish_reason, usage = validate_response(result)
                established_pred = protocol["get_answer_value"](content)
                strict_pred = strict_answer(content)
                base.update(
                    status="complete",
                    text=content,
                    finish_reason=finish_reason,
                    usage=usage,
                    response_id=result.get("id"),
                    response_model=result.get("model"),
                    system_fingerprint=result.get("system_fingerprint"),
                    established_predicted=established_pred,
                    established_correct=bool(established_pred == labels[index]),
                    strict_predicted=str(strict_pred)
                    if strict_pred is not None
                    else None,
                    strict_correct=bool(
                        strict_pred is not None
                        and strict_pred == strict_answer(test[index]["answer"])
                    ),
                    truncated=finish_reason == "length",
                )
                checkpoint["completed_items"] += 1
            except asyncio.CancelledError:
                base.update(
                    status="cancelled", error="Cancelled after another worker failed"
                )
                raise
            except Exception as error:
                base.update(
                    status="error", error_type=type(error).__name__, error=str(error)
                )
                checkpoint["failed_items"] += 1
                raise
            finally:
                base["completed_at"] = utc_now()
                base["elapsed_s"] = time.perf_counter() - started
                checkpoint["results"][index] = base
                atomic_checkpoint(args.output, checkpoint)
                queue.task_done()

    timeout = aiohttp.ClientTimeout(total=args.request_timeout_seconds)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    tasks = []
    try:
        async with aiohttp.ClientSession(
            timeout=timeout, connector=connector
        ) as session:
            reset_url = args.host.rstrip("/") + f":{args.port}/reset_prefix_cache"
            async with session.post(reset_url) as response:
                response.raise_for_status()
                reset = await response.json()
                if not reset.get("success"):
                    raise RuntimeError("Prefix cache reset failed before evaluation")
            checkpoint["config"]["prefix_cache_reset"] = True
            atomic_checkpoint(args.output, checkpoint)
            tasks = [asyncio.create_task(worker(session)) for _ in range(CONCURRENCY)]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        latency = time.perf_counter() - tic
        rows = checkpoint["results"]
        if any(row is None or row["status"] != "complete" for row in rows):
            raise RuntimeError("Evaluation did not complete every requested item")
        established = protocol["_score_gsm8k"](
            [row["text"] for row in rows],
            [row["usage"]["completion_tokens"] for row in rows],
            labels,
            NUM_SHOTS,
            MAX_TOKENS,
            latency,
        )
        # NumPy scalars need conversion before strict JSON checkpointing.
        established = {
            key: value.item() if hasattr(value, "item") else value
            for key, value in established.items()
        }
        checkpoint["aggregate"] = {
            "established": established,
            "established_correct": sum(row["established_correct"] for row in rows),
            "strict_accuracy": sum(row["strict_correct"] for row in rows) / len(rows),
            "strict_correct": sum(row["strict_correct"] for row in rows),
            "strict_invalid_count": sum(
                row["strict_predicted"] is None for row in rows
            ),
            "truncation_count": sum(row["truncated"] for row in rows),
            "finish_reasons": {
                reason: sum(row["finish_reason"] == reason for row in rows)
                for reason in sorted({row["finish_reason"] for row in rows})
            },
        }
        checkpoint["status"] = "complete"
    except BaseException as error:
        checkpoint["status"] = "failed"
        checkpoint["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        checkpoint["completed_at"] = utc_now()
        checkpoint["elapsed_s"] = time.perf_counter() - tic
        atomic_checkpoint(args.output, checkpoint)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": checkpoint["status"],
                "aggregate": checkpoint["aggregate"],
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--count",
        type=count_arg,
        default=TOTAL_QUESTIONS,
        help="Number of GSM8K test questions: 1..1319 or all (default 1319)",
    )
    parser.add_argument("--host", default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=600)
    args = parser.parse_args()
    if not args.request_timeout_seconds > 0:
        parser.error("request-timeout-seconds must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
