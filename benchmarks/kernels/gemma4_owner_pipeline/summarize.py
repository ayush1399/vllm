# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize an independent repeat without selecting winners on that repeat."""

import argparse
import json
import random
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repeat", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
data = json.loads(args.repeat.read_text())
rows = []
rng = random.Random(42)


def paired_interval(baseline, candidate):
    ratios = [a / b for a, b in zip(baseline, candidate)]
    boot = sorted(
        statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000)
    )
    return [(boot[50] - 1) * 100, (boot[1949] - 1) * 100]


for case in data:
    selected = case["selected"]
    assert selected["exact_all_ranks"]
    medians = selected["median_us"]
    # Chosen from the tuning run before the independent repeat.
    variant = "replicated_pdl" if case["k"] == 2688 else "local_pdl"
    plain = variant.replace("pdl", "plain")
    rows.append(
        dict(
            rows=case["rows"],
            k=case["k"],
            next_n=case["next_n"],
            config=selected["next_config"],
            variant=variant,
            median_us=medians,
            latency_reduction_pct=(1 - medians[variant] / medians["fi"]) * 100,
            speedup_pct=(medians["fi"] / medians[variant] - 1) * 100,
            pdl_saved_us=medians[plain] - medians[variant],
            paired_speedup_bootstrap_95_pct=paired_interval(
                selected["samples_us"]["fi"],
                selected["samples_us"][variant],
            ),
            weight_only_effective_GB_s=(5376 * (case["k"] + case["next_n"]))
            / medians[variant]
            / 1000,
        )
    )
summary = dict(
    implementation=data[0].get("implementation"),
    boundary=data[0]["boundary"],
    baseline="Current optimized branch, not stock vLLM",
    scope="Kernel-chain experiments; model runtime not enabled",
    rows=rows,
)
args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
protocol = data[0]["protocol"]
clocks = sorted(
    {
        clock[key]
        for case in data
        for clock in case["selected"]["clocks"]
        for key in ("before_mhz", "after_mhz")
    }
)
clock_text = ", ".join(str(c) for c in clocks)
lines = [
    "# Gemma4 SM90 two-GEMM chain results",
    "",
    "Independent repeat on TP8 H100. Baseline is the current optimized branch.",
    "The chosen variants and geometries were fixed by the preceding tuning run.",
    "",
    (
        "| Rows | First K | Baseline µs | Matched µs | Local PDL µs | "
        "Fused reduction/norm PDL µs | Chosen latency reduction |"
    ),
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    v = row["median_us"]
    lines.append(
        f"| {row['rows']} | {row['k']} | {v['fi']:.2f} | {v['matched']:.2f} | "
        f"{v['local_pdl']:.2f} | {v['replicated_pdl']:.2f} | "
        f"{row['latency_reduction_pct']:.1f}% |"
    )
lines += [
    "",
    (
        f"Each result is the median of {protocol['samples']} samples, reduced with "
        "MAX over eight ranks. Arm order rotates. Graphs stream distinct weights "
        f"beyond L2 capacity. Warmup is {protocol['warmup_rounds']} rounds; "
        f"each timed sample uses {protocol['replays']} replays. NVML clock readings "
        f"before/after timing are {clock_text} MHz. No clock settings were changed."
    ),
    "",
    (
        "Correctness checks compare valid FP8 bytes, scales, FP32 carry, "
        "and the following BF16 GEMM output exactly on all ranks. The repeat "
        f"runs {protocol['input_changes']} changed-input replays after capture. "
        "Mailbox resets and unused owner-carry rows are checked too."
    ),
    "",
    (
        "PDL savings are isolated against the same local/replicated plain "
        "kernel chain. Overall gains also include GEMM geometry and any "
        "collective/norm fusion changes. Weight-only effective bandwidth omits "
        "activation, scale, residual, and collective traffic; it is not a measured "
        "HBM utilization percentage."
    ),
    "",
    (
        "Owner-only FP8 publication remains an experiment: its extra TP "
        "readiness exchange prevents a consistent win against the matched control. "
        "These results do not establish a model-level or serving speedup."
    ),
]
args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
print("\n".join(lines[:12]))
