# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare cached-prefix serving with sustained staggered or burst clients.

The server must expose prompt-token usage details and streamed token IDs.
Input is a JSON array of natural-text IDs from the server's tokenizer. Reserved
marker IDs 10000..189999 must also be valid ordinary tokens for that tokenizer.
This is a synthetic closed-loop workload, not a production arrival-time replay.
"""

import argparse
import asyncio
import hashlib
import inspect
import json
import random
import time
from pathlib import Path

import aiohttp
import numpy as np

CASE_IDS = {"burst": 0, "warm-fixed": 1, "warm": 2, "mixed": 3}
MAX_CLIENTS = 64
NAMESPACES = {
    "warmup": (64, 128, 10000),
    "measured": (10000, 1024, 0),
    "background_before": (80000, 512, 20000),
    "background_after": (120000, 512, 40000),
}
BACKGROUND_REASONS = {
    "background_before": "awaiting_measurement_start",
    "background_after": "awaiting_other_clients_measured_quota",
}


def quantiles(values):
    values = [v for v in values if v is not None]
    return (
        {f"p{p}": float(np.percentile(values, p)) for p in [50, 95, 99]}
        if values
        else None
    )


def prompt_hash(prompt):
    return hashlib.sha256(np.asarray(prompt, dtype="<u4").tobytes()).hexdigest()


def marker_token(name, repeat, slot):
    # Addition modulo the full token namespace is injective for reserved slots.
    return 10000 + (CASE_IDS[name] * 8191 + repeat * 31337 + slot) % 180000


def request_spec(args, name, repeat, client_id, namespace, ordinal):
    start, width, sequence_offset = NAMESPACES[namespace]
    if not 0 <= ordinal < width:
        raise ValueError(f"Exhausted {namespace} namespace for client {client_id}")
    logical_index = sequence_offset + ordinal * MAX_CLIENTS + client_id
    output = (
        110
        if name in ["burst", "warm-fixed"]
        else random.Random(args.seed + logical_index).randint(
            args.min_output, args.max_output
        )
    )
    cold = name == "mixed" and logical_index % 33 == 0
    query = 8192 if cold else 800 if name == "mixed" and logical_index % 10 == 0 else 80
    phase = "background" if namespace.startswith("background") else namespace
    return dict(
        request_id=f"{name}-r{repeat}-c{client_id}-{namespace}-t{ordinal}",
        client_id=client_id,
        ordinal=ordinal,
        namespace=namespace,
        phase=phase,
        measured=phase == "measured",
        # Keep legacy ``not warmup`` consumers from counting background traffic.
        warmup=phase != "measured",
        background_reason=BACKGROUND_REASONS.get(namespace),
        measured_turn=ordinal if phase == "measured" else None,
        turn=args.warmup_turns + ordinal if phase == "measured" else ordinal,
        marker_token=marker_token(name, repeat, start + client_id * width + ordinal),
        request_type="cold" if cold else "warm",
        planned_query=query,
        planned_output=output,
    )


async def request(session, args, prompt, output, request_id, prompt_sha256=None):
    digest = prompt_hash(prompt)
    if prompt_sha256 is not None:
        assert digest == prompt_sha256, "Request differs from its measured plan"
    payload = dict(
        model=args.model,
        prompt=prompt,
        max_tokens=output,
        temperature=0,
        seed=42,
        ignore_eos=True,
        stream=True,
        return_token_ids=True,
        request_id=request_id,
        stream_options={"include_usage": True},
    )
    start = time.perf_counter_ns()
    first = last = None
    usage = metrics = None
    output_ids = []
    async with session.post(args.base_url + "/v1/completions", json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError((resp.status, (await resp.text())[:2000]))
        async for line in resp.content:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                continue
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            for choice in chunk.get("choices", []):
                ids = choice.get("token_ids") or []
                if ids:
                    now = time.perf_counter_ns()
                    first = first or now
                    last = now
                    output_ids.extend(ids)
            usage = chunk.get("usage") or usage
            metrics = chunk.get("metrics") or metrics
    end = time.perf_counter_ns()
    if first is None or usage is None:
        raise RuntimeError("Missing first token or usage")
    assert len(output_ids) == usage["completion_tokens"] == output
    assert usage["prompt_tokens"] == len(prompt)
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    return dict(
        request_id=request_id,
        prompt_sha256=digest,
        arrival_ns=start,
        first_token_ns=first,
        finish_ns=end,
        ttft_ms=(first - start) / 1e6,
        latency_ms=(end - start) / 1e6,
        tpot_ms=(last - first) / 1e6 / (output - 1) if output > 1 else None,
        prompt_tokens=len(prompt),
        cached_tokens=cached,
        computed_prompt_tokens=len(prompt) - cached,
        completion_tokens=output,
        output_sha256=hashlib.sha256(json.dumps(output_ids).encode()).hexdigest(),
        server_metrics=metrics,
    )


async def invoke_hook(hook, *args):
    value = hook(*args)
    while inspect.isawaitable(value):
        value = await value


async def run_case(session, args, corpus, name, repeat, case_index, hooks):
    # case_index is retained for callers; canonical case IDs determine inputs.
    del case_index
    prefix = corpus[: args.prefix_tokens]
    issued_markers = set()

    def make_prompt(query, marker, cold=False):
        if marker in issued_markers:
            raise RuntimeError(f"Duplicate marker in reset-cache case: {marker}")
        issued_markers.add(marker)
        suffix = [marker] + corpus[args.suffix_offset : args.suffix_offset + query - 1]
        return ([] if cold else prefix) + suffix

    measured_turns = args.turns - args.warmup_turns
    plan = []
    for client_id in range(args.concurrency):
        for turn in range(measured_turns):
            spec = request_spec(args, name, repeat, client_id, "measured", turn)
            suffix = [spec["marker_token"]] + corpus[
                args.suffix_offset : args.suffix_offset + spec["planned_query"] - 1
            ]
            tokens = ([] if spec["request_type"] == "cold" else prefix) + suffix
            spec["prompt_sha256"] = prompt_hash(tokens)
            plan.append(spec)
    plan_hash = hashlib.sha256(
        json.dumps(
            dict(
                case=name,
                repeat=repeat,
                corpus_sha256=args.corpus_sha256,
                prefix_tokens=args.prefix_tokens,
                suffix_offset=args.suffix_offset,
                requests=plan,
            ),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    plans = {(r["client_id"], r["ordinal"]): r for r in plan}
    async with session.post(args.base_url + "/reset_prefix_cache") as resp:
        resp.raise_for_status()
        assert (await resp.json())["success"]
    setup = []
    for slot, query in enumerate([16, 80, 800]):
        row = await request(
            session,
            args,
            make_prompt(query, marker_token(name, repeat, slot)),
            1 if slot == 0 else 2,
            f"{name}-r{repeat}-setup-{slot}",
        )
        assert row["cached_tokens"] == (0 if slot == 0 else args.prefix_tokens)
        row.update(
            phase="setup",
            measured=False,
            warmup=False,
            background_reason=None,
            namespace="setup",
            ordinal=slot,
            planned_query=row["computed_prompt_tokens"],
            planned_output=1 if slot == 0 else 2,
            request_type="cold" if slot == 0 else "warm",
            marker_token=marker_token(name, repeat, slot),
        )
        setup.append(row)

    started = time.perf_counter_ns()
    rows, hook_tasks = [], []
    warmup_complete, measured_complete = {}, {}
    all_warm = asyncio.Event()
    measuring = asyncio.Event()
    all_measured = asyncio.Event()
    stop = asyncio.Event()
    window = dict(ready_ns=None, start_ns=None, end_ns=None)
    flight = peak_flight = 0

    async def issue(spec):
        nonlocal flight, peak_flight
        tokens = make_prompt(
            spec["planned_query"], spec["marker_token"], spec["request_type"] == "cold"
        )
        flight += 1
        peak_flight = max(peak_flight, flight)
        active = flight
        try:
            row = await request(
                session,
                args,
                tokens,
                spec["planned_output"],
                spec["request_id"],
                spec.get("prompt_sha256"),
            )
        finally:
            flight -= 1
        expected = 0 if spec["request_type"] == "cold" else args.prefix_tokens
        assert row["cached_tokens"] == expected, (spec["request_id"], row)
        assert row["computed_prompt_tokens"] == spec["planned_query"]
        row.update(spec, active_on_arrival=active)
        rows.append(row)

    async def client(client_id):
        delay = (
            0
            if name == "burst"
            else args.stagger_seconds * client_id / args.concurrency
        )
        await asyncio.sleep(delay)
        for turn in range(args.warmup_turns):
            await issue(request_spec(args, name, repeat, client_id, "warmup", turn))
        warmup_complete[client_id] = time.perf_counter_ns()
        if len(warmup_complete) == args.concurrency:
            window["ready_ns"] = max(warmup_complete.values())
            all_warm.set()
        background = 0
        while not measuring.is_set():
            await issue(
                request_spec(
                    args, name, repeat, client_id, "background_before", background
                )
            )
            background += 1
        for turn in range(measured_turns):
            await issue(plans[(client_id, turn)])
        measured_complete[client_id] = time.perf_counter_ns()
        if len(measured_complete) == args.concurrency:
            window["end_ns"] = max(measured_complete.values())
            stop.set()
            all_measured.set()
        background = 0
        while not stop.is_set():
            await issue(
                request_spec(
                    args, name, repeat, client_id, "background_after", background
                )
            )
            background += 1

    async def coordinate_measurement():
        await all_warm.wait()
        window["start_ns"] = time.perf_counter_ns()
        measuring.set()
        if hooks and "measurement_ready" in hooks:
            hook_tasks.append(
                asyncio.create_task(
                    invoke_hook(
                        hooks["measurement_ready"],
                        session,
                        name,
                        repeat,
                        dict(
                            window,
                            clients_ready=len(warmup_complete),
                            warmup_complete_ns=dict(warmup_complete),
                            measured_plan_sha256=plan_hash,
                            corpus_sha256=args.corpus_sha256,
                        ),
                    )
                )
            )
        await all_measured.wait()
        if hook_tasks:
            await asyncio.gather(*hook_tasks)

    if hooks and "case_start" in hooks:
        hook_tasks.append(
            asyncio.create_task(invoke_hook(hooks["case_start"], session, name, repeat))
        )
    tasks = [asyncio.create_task(client(i)) for i in range(args.concurrency)]
    tasks.append(asyncio.create_task(coordinate_measurement()))
    try:
        await asyncio.gather(*tasks)
    finally:
        stop.set()
        for task in tasks + hook_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, *hook_tasks, return_exceptions=True)
    rows.sort(key=lambda r: r["arrival_ns"])
    measured = [r for r in rows if r["measured"]]
    assert len(measured) == args.concurrency * measured_turns
    assert all(
        window["start_ns"] <= r["arrival_ns"] <= r["finish_ns"] <= window["end_ns"]
        for r in measured
    )
    groups = {
        "all": measured,
        "warm": [r for r in measured if r["request_type"] == "warm"],
        "cold": [r for r in measured if r["request_type"] == "cold"],
    }
    groups.update(
        {
            f"query{q}": [r for r in measured if r["planned_query"] == q]
            for q in [80, 800, 8192]
        }
    )
    summary = {
        kind: dict(
            count=len(selected),
            **{
                metric: quantiles([r.get(metric) for r in selected])
                for metric in ["ttft_ms", "tpot_ms", "latency_ms"]
            },
        )
        for kind, selected in groups.items()
    }
    elapsed = (window["end_ns"] - window["start_ns"]) / 1e9
    return dict(
        name=name,
        repeat=repeat,
        requests=rows,
        setup_requests=setup,
        measured_plan=plan,
        measured_plan_sha256=plan_hash,
        corpus_sha256=args.corpus_sha256,
        summary=summary,
        measurement_window=dict(
            window,
            warmup_complete_ns=warmup_complete,
            measured_complete_ns=measured_complete,
        ),
        measurement_elapsed_s=elapsed,
        measured_request_count=len(measured),
        measured_requests_per_second=len(measured) / elapsed,
        elapsed_s=(time.perf_counter_ns() - started) / 1e9,
        phase_counts={
            phase: sum(r["phase"] == phase for r in rows)
            for phase in ["warmup", "measured", "background"]
        },
        peak_flight=peak_flight,
        cache_validation=True,
    )


async def run(args, hooks=None):
    raw = Path(args.token_ids).read_bytes()
    corpus = json.loads(raw)
    args.corpus_sha256 = hashlib.sha256(raw).hexdigest()
    assert len(corpus) >= max(args.prefix_tokens, args.suffix_offset + 8192)
    assert min(corpus) >= 0 and max(corpus) < 2**32
    if not 1 <= args.concurrency <= MAX_CLIENTS:
        raise ValueError(f"Concurrency must be in [1,{MAX_CLIENTS}]")
    if not 0 <= args.warmup_turns <= NAMESPACES["warmup"][1]:
        raise ValueError("Warmup quota exceeds the reserved request namespace")
    if not 1 <= args.turns - args.warmup_turns <= NAMESPACES["measured"][1]:
        raise ValueError("Measured quota must be in [1,1024] per client")
    assert 2 <= args.min_output <= args.max_output
    assert args.repeats > 0 and args.stagger_seconds >= 0
    assert all(name in CASE_IDS for name in args.cases)
    configuration = dict(
        vars(args),
        schema_version=2,
        prompt_hash_encoding="uint32-le",
        hook_names=sorted(hooks or {}),
        hooks_present=bool(hooks),
        instrumented=bool(hooks),
        profiling="hook-controlled" if hooks else "not requested by client",
    )
    result = dict(configuration=configuration, cases=[])
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=900),
        connector=aiohttp.TCPConnector(
            limit=args.concurrency + 8, keepalive_timeout=30
        ),
    ) as session:
        for repeat in range(args.repeats):
            order = list(enumerate(args.cases))
            if repeat % 2:
                order.reverse()
            for i, name in order:
                case = await run_case(session, args, corpus, name, repeat, i, hooks)
                result["cases"].append(case)
                output = Path(args.output)
                temporary = output.with_suffix(output.suffix + ".tmp")
                temporary.write_text(json.dumps(result, indent=2) + "\n")
                temporary.replace(output)
                print(name, repeat, case["summary"], flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--token-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix-tokens", type=int, default=8192)
    parser.add_argument("--suffix-offset", type=int, default=90000)
    parser.add_argument(
        "--concurrency", type=int, default=13, help="Sustained client count, at most 64"
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=20,
        help="Warmup plus measured quota per client; excludes background",
    )
    parser.add_argument(
        "--warmup-turns",
        type=int,
        default=3,
        help="Warmup quota per client, at most 128",
    )
    parser.add_argument("--stagger-seconds", type=float, default=1.2)
    parser.add_argument("--min-output", type=int, default=80)
    parser.add_argument("--max-output", type=int, default=140)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument(
        "--cases", nargs="+", default=["burst", "warm", "mixed"], choices=list(CASE_IDS)
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
