# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run cached-prefill controls with explicit warmup and measured quotas."""

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp
import benchmark_prefix_cache_staggered as replay


async def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    corpus = json.loads(args.token_ids.read_bytes())
    if len(corpus) < 90000 + 18304:
        raise ValueError("Token corpus is too short for the longest query")
    records = []
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=900)
    ) as session:
        for query in [800, 8192, 18304]:
            async with session.post(args.base_url + "/reset_prefix_cache") as response:
                response.raise_for_status()
                assert (await response.json())["success"]
            prefix = corpus[:8192]
            setup = await replay.request(
                session, args, prefix + [175000], 1, "control-setup"
            )
            assert setup["cached_tokens"] == 0
            for index in range(args.warmup_requests + args.measured_requests):
                prompt = prefix + [175100 + index] + corpus[90000 : 90000 + query - 1]
                row = await replay.request(
                    session, args, prompt, 1, f"control-q{query}-i{index}"
                )
                assert row["cached_tokens"] == 8192
                assert row["computed_prompt_tokens"] == query
                records.append(
                    dict(row, query=query, warmup=index < args.warmup_requests)
                )
    result = {
        "protocol": (
            "cached8192, queries800/8192/18304, output1, "
            f"warmup{args.warmup_requests}/measured{args.measured_requests}"
        ),
        "records": records,
        "summary": {
            str(query): replay.quantiles(
                [
                    row["ttft_ms"]
                    for row in records
                    if row["query"] == query and not row["warmup"]
                ]
            )
            for query in [800, 8192, 18304]
        },
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--token-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--measured-requests", type=int, default=12)
    args = parser.parse_args()
    if args.warmup_requests < 0 or args.measured_requests <= 0:
        parser.error(
            "warmup-requests must be nonnegative and measured-requests positive"
        )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
