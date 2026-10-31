# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare paired outputs from benchmark_gemma4_mtp_acceptance.py."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(baseline, candidate, rng, samples):
    bg, cg = baseline["groups"], candidate["groups"]
    assert len(bg) == len(cg)
    for b, c in zip(bg, cg):
        assert b["ids"] == c["ids"]
        assert b["prompt_token_hashes"] == c["prompt_token_hashes"]
    fields = ("num_accepted_tokens", "num_draft_tokens", "num_drafts")
    bcounts = np.array([[g["counters"][f] for f in fields] for g in bg])
    ccounts = np.array([[g["counters"][f] for f in fields] for g in cg])
    deltas = []
    for start in range(0, samples, 5000):
        indices = rng.integers(0, len(bg), (min(5000, samples - start), len(bg)))
        b, c = bcounts[indices].sum(1), ccounts[indices].sum(1)
        rate = 100 * (c[:, 0] / c[:, 1] - b[:, 0] / b[:, 1])
        length = 100 * ((1 + c[:, 0] / c[:, 2]) / (1 + b[:, 0] / b[:, 2]) - 1)
        deltas.append(np.stack([rate, length], axis=1))
    deltas = np.concatenate(deltas)
    bs, cs = baseline["summary"], candidate["summary"]
    estimates = [
        100 * (cs["acceptance_rate"] - bs["acceptance_rate"]),
        100 * (cs["mean_acceptance_length"] / bs["mean_acceptance_length"] - 1),
    ]
    result = {
        "dataset": baseline["dataset"],
        "batch_size": baseline["batch_size"],
        "paired_groups": len(bg),
        "prompts": sum(len(g["ids"]) for g in bg),
        "baseline": bs,
        "candidate": cs,
        "per_position_delta_percentage_points": [
            100 * (c - b)
            for b, c in zip(
                bs["accepted_per_position_per_draft"],
                cs["accepted_per_position_per_draft"],
            )
        ],
        "identical_output_hashes": sum(
            b == c
            for bgroup, cgroup in zip(bg, cg)
            for b, c in zip(bgroup["output_hashes"], cgroup["output_hashes"])
        ),
    }
    complete = np.array(
        [
            "length" not in b["finish_reasons"] and "length" not in c["finish_reasons"]
            for b, c in zip(bg, cg)
        ]
    )
    if complete.any():
        b, c = bcounts[complete].sum(0), ccounts[complete].sum(0)
        result["untruncated_groups_sensitivity"] = {
            "paired_groups": int(complete.sum()),
            "acceptance_rate_delta_pp": float(100 * (c[0] / c[1] - b[0] / b[1])),
            "acceptance_length_delta_pct": float(
                100 * ((1 + c[0] / c[2]) / (1 + b[0] / b[2]) - 1)
            ),
        }
    for i, name in enumerate(
        ("acceptance_rate_delta_pp", "acceptance_length_delta_pct")
    ):
        lower, upper = np.quantile(deltas[:, i], [0.025, 0.975])
        corrected_lower = float(np.quantile(deltas[:, i], 0.05 / 32))
        result[name] = {
            "estimate": estimates[i],
            "paired_bootstrap_95pct_interval": [float(lower), float(upper)],
            "familywise_95pct_lower_bound": corrected_lower,
            "margin": 1.0,
            "within_screening_margin": corrected_lower > -1.0,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=200000)
    parser.add_argument(
        "--depths", type=int, nargs="+", choices=[1, 2, 3, 4], default=[1, 2, 3, 4]
    )
    args = parser.parse_args()
    rng = np.random.default_rng(20261101)
    result = {
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": 20261101,
        "depths": args.depths,
        "protocol": json.loads((args.results_dir / "protocol.json").read_text()),
        "dataset_provenance": json.loads(
            (args.results_dir / "dataset-provenance.json").read_text()
        ),
        "runs": {},
        "comparisons": [],
    }
    for k in args.depths:
        runs = {}
        for arm in ("baseline", "candidate"):
            path = args.results_dir / f"{arm}-k{k}.json"
            data = json.loads(path.read_text())
            assert len(data["cases"]) == 4 and len(data["workers"]) == 8
            assert all(
                w["target_full_chain"] == (arm == "candidate") for w in data["workers"]
            )
            runs[arm] = data
            result["runs"][f"{arm}-k{k}"] = {
                **data,
                "raw_result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        b, c = runs["baseline"], runs["candidate"]
        for field in ("environment", "prompt_manifest_sha256", "benchmark_sha256"):
            assert b[field] == c[field], (k, field)
        for field in b["args"]:
            if field not in ("output", "decode_backend"):
                assert b["args"][field] == c["args"][field], (k, field)
        for field in (
            "_C_stable_libtorch.abi3.so",
            "model_executor/layers/gemma4_full_chain.py",
        ):
            source_field = field
            if field not in b["source_sha256"]:
                source_field = "models/gemma4/nvidia/specialized_decode/full_chain.py"
            assert b["source_sha256"][source_field] == c["source_sha256"][source_field]
        for bc, cc in zip(b["cases"], c["cases"]):
            assert (bc["dataset"], bc["batch_size"]) == (
                cc["dataset"],
                cc["batch_size"],
            )
            comparison = {"k": k, **compare(bc, cc, rng, args.bootstrap_samples)}
            result["comparisons"].append(comparison)
            print(
                k,
                bc["dataset"],
                bc["batch_size"],
                comparison["acceptance_rate_delta_pp"],
                comparison["acceptance_length_delta_pct"],
                flush=True,
            )
    result["all_within_screening_margins"] = all(
        c[field]["within_screening_margin"]
        for c in result["comparisons"]
        for field in ("acceptance_rate_delta_pp", "acceptance_length_delta_pct")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
