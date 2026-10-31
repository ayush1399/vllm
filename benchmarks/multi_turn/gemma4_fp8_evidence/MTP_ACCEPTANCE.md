# Gemma4 FP8 MTP acceptance validation

This comparison measures actual speculative-decoding acceptance through `LLM`
on the original serving branch and the optimized MTP working tree. The source
baseline is `d30ed80158a493fdb193ae5cc1b368c4189520e4`. It uses the same FP8
target and FP8 assistant as the latency matrix, on eight H100 80GB GPUs with
TP8/PP1, FA4 and BF16 activations/KV cache.

## Completed k=4 results

| Workload | Batch | Baseline acceptance | Optimized acceptance | Change | Baseline mean length | Optimized mean length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gsm8k | 1 | 85.295% | 85.175% | -0.121 pp | 4.4118 | 4.4070 |
| gsm8k | 4 | 84.903% | 84.667% | -0.237 pp | 4.3961 | 4.3867 |
| mt-bench | 1 | 57.400% | 57.182% | -0.218 pp | 3.2960 | 3.2873 |
| mt-bench | 4 | 57.411% | 56.770% | -0.641 pp | 3.2964 | 3.2708 |

The observed token acceptance changes are -0.121 to -0.641 percentage points;
mean acceptance length changes are approximately -0.11% to -0.78%. The user
accepted these observed differences and cancelled further depth testing and the
additional k=4 repeat. This is practical acceptance of the measured changes,
rather than a claim of exact equality.

The largest per-position decrease is 1.051 percentage points at the third
draft position on MT-Bench batch 4. Excluding groups truncated in either arm
changes the MT-Bench batch-4 token acceptance delta from -0.641 to -0.621 pp.
Both arms have zero GSM8K truncations and three MT-Bench truncations per batch
setting.

All four paired 95% acceptance-rate intervals include zero. The conservative
predeclared 1-pp/1% confidence screen is inconclusive for three of the four
cases: its corrected lower bounds extend beyond the screening margins. The
finite prompt sample therefore does not establish strict statistical
noninferiority within those margins. All observed primary metric changes are
within the margins, and the uncertainty is retained in the evidence.

Full integer counters, paired confidence bounds, per-position changes and
output hashes are in `mtp-acceptance.json`. Raw files and the cancelled run logs
remain under `/jobtmp/gemma4-mtp-acceptance/` on the measurement machine.

## Protocol

The completed comparison uses speculative depth 4 on each source arm, with the
first 64 GSM8K test questions and all 80 MT-Bench first turns at batch sizes 1
and 4: four matched cases and 576 measured prompt generations across both arms.
The original plan included depths 1–3, but the user explicitly cancelled those
runs after accepting the completed k=4 differences. The queued additional k=4
repeat was also cancelled. Partial k=1 data is excluded from this report.
GSM8K uses the existing acceptance test's step-by-step/boxed-answer instruction;
MT-Bench uses its original first user turn. There are no few-shot examples.
Prompt manifests and tokenized prompt groups must match between arms.

Sampling is greedy, seed 0, thinking disabled, natural EOS stopping and a
1,024-token generation limit. Prefix caching is disabled. Each dataset/batch
case has one complete warmup group, excluded from the measured counter deltas.
The graph policy is `FULL_AND_PIECEWISE` with capture sizes 1–32, a 4,096-token
batch budget, maximum four sequences, maximum model length 8,192 and 0.6 GPU
memory utilization. GPU runs are serial. No profiling is active.

Metrics come from `LLM.get_metrics()` with statistics enabled:

| Metric | Definition |
| --- | --- |
| Draft acceptance rate | Accepted draft tokens / proposed draft tokens |
| Mean acceptance length | 1 + accepted draft tokens / draft rounds |
| Acceptance at draft position | Accepted tokens at that position / draft rounds |

The last definition is unconditional: later-position rates include rounds that
rejected an earlier position. It is not conditional accuracy among rounds that
reached that position. The acceptance-length definition follows vLLM's existing
convention of including a bonus token. Aggregate ratios weight by token or draft
counts, respectively; they are not unweighted averages of per-prompt ratios.

Every prompt group records integer counter deltas, prompt IDs/token hashes,
generated token hashes, output lengths and finish reasons. The per-position
counts must sum to the accepted-token count. This captures acceptance directly,
rather than inferring it from output accuracy or throughput.

## Uncertainty and screening criteria

The analysis pairs identical prompt groups between source arms and resamples
those pairs 200,000 times with seed 20261101. Each resample recomputes both
aggregate ratios. The resampling unit is one prompt for batch size 1 and one
four-prompt group for batch size 4, preserving dependence within each batch.

The practical screening margins, recorded before inspecting comparisons, are:

- At most a 1 percentage-point absolute reduction in token acceptance rate.
- At most a 1% relative reduction in mean acceptance length.

The report includes observed differences and paired percentile-bootstrap 95%
intervals. Screening uses one-sided lower bounds with Bonferroni correction
across the originally planned 16 cases times two primary metrics, providing
a nominal familywise 95% screen. The original correction is retained after the
user narrowed the scope to k=4, making it conservative for these four cases. Each corrected lower bound must exceed minus its corresponding margin.
Failure to meet this criterion can be an inconclusive interval or an observed
regression; neither is silently converted into a passing result.

These margins are practical tolerances, not exact numerical equality. Bootstrap
intervals describe variation over the sampled prompt groups and do not cover
every source of inference nondeterminism or generalize to all workloads.

## Reproduction

Prepare a JSONL manifest with `dataset`, `id` and `messages` on each line. Use
the same file for both source checkouts. Dataset formatting follows
`tests/v1/e2e/spec_decode/acceptance_rates/utils.py`; this run uses locally
cached public datasets and records the input-file and manifest hashes.

```bash
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER_FP8_INPUT=1
export VLLM_USE_DEEP_GEMM_E8M0=1
export VLLM_DEEP_GEMM_WARMUP=skip
export OMP_NUM_THREADS=1
.venv/bin/python benchmarks/multi_turn/benchmark_gemma4_mtp_acceptance.py \
  --model /path/to/gemma4-31B-it-FP8-block \
  --draft-model /path/to/gemma4-assistant-FP8-block \
  --prompts /path/to/prompts.jsonl --k 4 \
  --output /path/to/results/candidate-k4.json
```

Run from each source arm, naming outputs `baseline-k4.json` and
`candidate-k4.json`. Preserve the measurement protocol and dataset provenance
alongside the raw files.

```bash
.venv/bin/python benchmarks/multi_turn/analyze_gemma4_mtp_acceptance.py \
  --results-dir /path/to/results --depths 4 \
  --output /path/to/mtp-acceptance.json
```

The evidence retains all counter groups, per-position rates, output lengths,
truncations, source/native-library hashes and paired uncertainty estimates.
The initial benchmark startup required an explicit `return_dict=False` for the
installed tokenizer's chat-template API; no acceptance samples from that failed
startup enter this comparison.

## Limits

This evaluation covers greedy, thinking-disabled text generation on GSM8K and
MT-Bench first turns at batch sizes 1 and 4. It does not establish acceptance
equivalence for stochastic sampling, thinking-enabled generation, longer
contexts, other batching policies, other checkpoints or other hardware.
Floating-point changes can alter generated trajectories, so token hashes and
truncations remain visible alongside acceptance metrics.
