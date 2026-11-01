# Gemma4 SM90 projection/normalization/GEMM experiments

These are isolated kernel-chain experiments.
They compare the current optimized branch with owner-only FP8 publication and
local-input PDL pipelines on eight H100s in one NVSwitch domain.
The local-input PDL design is integrated separately in
`vllm/models/gemma4/nvidia/specialized_decode/pdl.py`, `pdl_driver.cu`, and
`pdl_gemm.cuh`. The model connects post-attention normalization to gate/up and
post-MLP normalization to the next local QKV projection. The multicast and
owner-only variants in this folder remain experiments.

The timed boundary is prequantized FP8 input → projection GEMM → TP8 sum →
post-RMSNorm/residual/scalar/pre-RMSNorm/raw FP8 quantization → following GEMM
with BF16 output. Allocation, input quantization, compilation, and sampling are
outside timing. First-GEMM output width and following-GEMM input width are 5376.

| First local K | Following local N | Boundary represented |
| ---: | ---: | --- |
| 1024 | 5376 | Local-attention output projection → gate/up |
| 2048 | 5376 | Global-attention output projection → gate/up |
| 2688 | 2048 | MLP down projection → next local QKV |

The default 5/20 rows represent B1/B4 verification with four draft tokens.
Work is fixed: no sampling or acceptance variation occurs in this benchmark.
Model performance tests must use identical fixed acceptance schedules; quality
tests must use actual drafting and rejection sampling.

## Implementations

- `fi`: existing FlashInfer GEMMs/all-reduce and the branch's native full chain.
- `matched`: private GEMMs with the candidate geometry and existing all-reduce
  and normalization. This distinguishes GEMM tuning from fusion/overlap gains.
- `local_plain` / `local_pdl`: existing all-reduce followed by identical local
  normalization and GEMM kernels, with ordinary launch ordering or PDL.
- `replicated_plain` / `replicated_pdl`: the first GEMM publishes BF16 partials
  from its epilogue. At 5 rows it multicasts all partials; at 20 rows it scatters
  to row owners, which multicast BF16 sums. Each rank normalizes and quantizes
  locally. The next GEMM stages weights before its local grid-dependency wait.
- `owner_grid` / `owner_grid_pdl`: row owners alone reduce, normalize, retain
  FP32 residuals, and multicast FP8/scales. After staging weights, the following
  GEMM waits for local completion. One signaling thread per rank bridges aliases
  and publishes TP readiness. The owner kernel itself has no system fences.
- `owner_grid_prefetch_pdl`: also loads normalization operands and partials
  before triggering the following GEMM.
- Other `owner_*` arms retain row-readiness, multicast/unicast, and fence-placement
  ablations that explain the initial regression.

The local PDL ordering follows
[#64893's Hopper dense GEMM](https://github.com/vllm-project/vllm/blob/382b4f475f2d8d7ad0adddae0f7a6e4f3313ed80/vllm/models/deepseek_v41/nvidia/specialized_decode/hopper/dense_gemm_cute.py):
stage independent weights, wait for the producer grid, then load activations.
Local FP8 outputs need no separate TP readiness flags or per-thread system
fences. Owner-only publication requires an additional TP readiness exchange.

All kernels preserve BF16 projection rounding before TP summation, the native
normalization reduction tree, scalar/interior rounding, raw FP8 conversion, and
the TMA-compatible column-major scale layout.

Double-buffered mailboxes and monotonic readiness epochs remain live across
graph replays. Reusing a slot requires contributions from every rank; those
contributions follow each rank's preceding GEMM consumption. A slot cannot be
overwritten while a lagging rank still consumes its preceding generation.

## Build and run

Use the repository environment with FlashInfer, its installed CUTLASS/DeepGEMM
headers, CUDA, and NVLink multicast support. Build output is separate from vLLM's
installed native library; completed shared libraries are replaced atomically.
Header patches fail closed if the installed FlashInfer source no longer matches.

```bash
.venv/bin/python benchmarks/kernels/gemma4_owner_pipeline/build.py \
  --build-dir /jobtmp/gemma4-owner/build

OMP_NUM_THREADS=1 VLLM_FLASHINFER_ALLREDUCE_BACKEND=mnnvl \
  .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 \
  benchmarks/kernels/gemma4_owner_pipeline/bench.py \
  --build-dir /jobtmp/gemma4-owner/build \
  --output /jobtmp/gemma4-owner/tuned.json --tune \
  --arms fi matched local_plain local_pdl replicated_plain replicated_pdl

OMP_NUM_THREADS=1 VLLM_FLASHINFER_ALLREDUCE_BACKEND=mnnvl \
  .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 \
  benchmarks/kernels/gemma4_owner_pipeline/bench.py \
  --build-dir /jobtmp/gemma4-owner/build \
  --output /jobtmp/gemma4-owner/repeat.json \
  --configs-from /jobtmp/gemma4-owner/tuned.json --stress 32 \
  --arms fi matched local_plain local_pdl replicated_plain replicated_pdl \
    owner_grid_prefetch_pdl

.venv/bin/python benchmarks/kernels/gemma4_owner_pipeline/summarize.py \
  --repeat /jobtmp/gemma4-owner/repeat.json \
  --output /jobtmp/gemma4-owner/RESULTS
```

Tuning covers 13 following-GEMM configurations: 64/128 output columns,
16/32 rows, 4/8 stages, plus larger 10/16/20-stage buffers that fit SM90 shared
memory. Exact distributed correctness runs before timing and after replay.
The repeat changes BF16 inputs and requantizes them after capture. Valid FP8
bytes, scales, FP32 carry, following BF16 outputs, mailbox resets, and unused
owner-carry rows are checked. Scale-storage padding is not an output.

Each sample is MAX over eight ranks; the reported statistic is the median of
64 samples. Arm order rotates. Graphs stream pointer-distinct weights beyond
L2 capacity. Warmup is 128 rounds with eight replays per arm; timed samples
also use eight replays. The common device barrier is outside timing; all
synchronization required by a candidate remains inside timing. NVML clocks,
source/library hashes, software versions, and operation boundaries are recorded.

## Profiling

Add `--trace` and run the benchmark under Nsight Systems with
`--trace=cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi
--capture-range-end=repeat`. Profiler timings are not benchmark results.

`profile_local.py` isolates the local normalization/following-GEMM pair for
Nsight Compute. It excludes TP reduction and runs on one GPU:

```bash
ncu --profile-from-start off --replay-mode kernel --graph-profiling graph \
  --clock-control none --cache-control all \
  --section LaunchStats --section Occupancy --section SchedulerStats \
  --section WarpStateStats --section MemoryWorkloadAnalysis \
  --section SpeedOfLight -o /jobtmp/gemma4-owner/local-chain \
  .venv/bin/python benchmarks/kernels/gemma4_owner_pipeline/profile_local.py \
  --build-dir /jobtmp/gemma4-owner/build \
  --configs-from /jobtmp/gemma4-owner/tuned.json
```

`profile_owner.py` profiles owner publication with ready local partials and
idle peers. Launch eight ranks, with only rank 0 under NCU. Its default
`--publication 3` selects the owner variant without publication fences.
These isolated counters do not establish the live eight-GPU critical path.

## Recorded run: 2026-11-01

Independent repeat, all ranks at 1980 MHz before/after timing, no clock-setting
changes. Values are microseconds for the complete two-GEMM chain.

| Rows | First K | Current branch | Local PDL | Fused reduction/norm + PDL |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 1024 | 26.02 | 24.46 | 24.65 |
| 20 | 1024 | 28.35 | 26.71 | 26.79 |
| 5 | 2048 | 27.83 | 26.63 | 26.84 |
| 20 | 2048 | 30.29 | 29.00 | 28.79 |
| 5 | 2688 | 25.72 | 23.02 | 23.21 |
| 20 | 2688 | 28.02 | 24.85 | 24.79 |

PDL saves about 1.5–2.2 µs against otherwise identical plain chains. Total gains
also include GEMM tuning and, for the fused variant, collective/norm changes.
The owner-only path does not consistently beat the matched GEMM control.

Live eight-GPU Nsight Systems traces show overlap in all 768 measured
norm/GEMM pairs for each PDL variant and none for the corresponding plain
variants. NCU reports zero memory-barrier stalls in the isolated local path;
its aggregate scoreboard counters do not fall under PDL. NCU substantially
perturbs durations, so speed claims use the independent timing run above.

Raw timing, correctness metadata, NCU reports/CSV, disassembly, live traces,
and per-rank overlap analysis are under `/jobtmp/gemma4-owner-20261101/`.
The baseline is this optimized branch, not stock vLLM. These kernel measurements
do not establish model-level or serving speedups. Use
`benchmarks/multi_turn/benchmark_gemma4_mtp.py` with `--gemma4-pdl` and
`--no-gemma4-pdl` for the matched runtime comparison; use the same
`--accepted-draft-tokens` value in both arms.

## Model-runtime validation: 2026-11-01

The runtime connects 60 attention-to-gate/up boundaries and 49 MLP-to-local-QKV
boundaries per target forward. Global QKV projections retain their existing
GEMMs. Small-row launches use the tuned local PDL geometry above; batches of
32 or more rows retain the existing normalization/GEMM implementation.

Validation passed 54 normalization/GEMM cases across three residual modes,
both projection widths, graph replay with changing inputs, and the prefill
fallback. Another 77 existing native normalization cases passed after factoring
the common CUDA header. The compiled TP8 MTP k4 regression passed against the
ordinary backend and confirmed the PDL kernels in its model trace. Across all
eight ranks, 2,495 of 2,616 traced normalization/GEMM pairs overlapped; median
overlap was 2.46–3.62 µs per rank. These profiler durations are separate from
performance measurements.

Separate five-shot GSM8K runs used real drafting and standard rejection sampling.
Both arms scored 31/32, with identical token hashes for all 32 answers and no
truncation. This is a small quality screen. Its elapsed times include cold-shape
JIT and are not used for performance comparisons.

The performance workload uses TP8 H100, BF16 activation/KV, FA4 (the configured graph split limit of 32 applies to FA3, not FA4), FULL_AND_PIECEWISE graphs, k4, and 128 output
tokens. Both arms accept exactly two draft tokens per verification step. There
are three warmup generations and nine timed generations per case. Prefix
caching is disabled. Monitored JIT events invalidate the whole timed block;
ordinary slow samples remain in the data.

An initial four-restart campaign exposed a request-admission race: ordinary
`LLM.generate` sometimes scheduled part of the short B4 batch before the remaining
requests arrived, adding variable mixed-prefill work. Those short-B4 wall times
are diagnostic only. The final comparison pauses scheduling, enqueues the full
batch, then times scheduler resume through completion. Pause/enqueue work is
excluded; the resume RPC and CPU work remain inside timing. Every retained
short-prompt sample has zero mixed-prefill interval.

Final medians below compare this optimized branch with PDL disabled/enabled.
Intervals include engine scheduling and CPU work; decode tail is the interval
after every request has produced its first token.

| Prompt tokens | Batch | Generation ms, before → PDL | Latency reduction | Decode tail ms, before → PDL | Tail reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 343.47 → 334.02 | 2.75% | 271.79 → 264.93 | 2.53% |
| 128 | 4 | 365.66 → 356.00 | 2.64% | 295.54 → 286.83 | 2.95% |
| 20000 | 1 | 727.97 → 716.94 | 1.51% | 288.55 → 277.70 | 3.76% |
| 20000 | 4 | 2164.42 → 2152.03 | 0.57% | 328.09 → 316.98 | 3.39% |

For 20k-token prompts, initial prefill was 437.77 → 437.76 ms at B1 and
448.29 → 448.51 ms at B4. B4's mixed-prefill/decode interval was
1387.20 → 1385.80 ms. Prefill therefore dominates the long-B4 total despite
the lower decode latency. The earlier roughly 5% short-generation estimate
from the admission-racing campaign is superseded by this comparison.

Raw runtime results, source hashes, test logs, quality hashes, and model overlap
analysis are under `/jobtmp/gemma4-runtime-20261101/`. The final timing artifact
is `queued-comparison.json`; individual samples and worker metadata are in
`perf-queued-baseline.json` and `perf-queued-candidate.json`.

## Matched upstream-main comparison: 2026-11-01

This campaign predates the residual/QKV/attention follow-up linked below.

This compares the complete current branch against upstream main
`5d55ef1590c92213e3893dc8e8712f7f2e0bc508`, fetched for this run. The branch
is HEAD `7fd4e3584ec4fdc3acd59c3aef9ab2efeff1dd38` plus the recorded working-tree
changes. The preceding runtime table isolates PDL within the optimized branch.

Both arms use the same FP8 target and assistant, TP8 H100, BF16 activations/KV,
FA4, FULL_AND_PIECEWISE graphs through 32 rows, maximum batch size
four, 4096 batched tokens, and disabled prefix caching. Synthetic token-ID prompts
contain 128 or 20,000 tokens; output length is fixed at 128. MTP uses
k4 with exactly two accepted draft tokens per step. Every measured request
recorded 43 verification steps, 172 drafted tokens and 86 accepted tokens.
The entire batch is admitted before scheduling resumes.

Torch 2.14.0+cu132, FlashInfer 0.7.2, CUTLASS DSL 4.8.0, Triton 3.8.0 and the
installed DeepGEMM provider are held constant. Main's core native extension was
built from its pinned source. Its FA4 files come from the clean upstream
dependency; the branch retains its split-selection patch. The configured FA3 graph split setting does not force FA4 splits.
FA4 uses its own selection heuristic; the branch patch preserves zero-valued
sliding-window bounds. See the profiling follow-up for measured launch grids. Native-library and Python-module
paths were checked on all ranks, with no imports from the other checkout.
Custom-op enable lists are compared as effective sets; repeated enable entries
are equivalent. GPU clock settings were left unchanged.

Restart order was main → branch → branch → main. Each case has three warmup
generations followed by nine timed generations per restart. The table uses the
mean of the two process medians per arm. All retained timed blocks had zero
monitored JIT events; all short-prompt samples had zero mixed-prefill interval.

| Prompt tokens | Batch | Main generation ms | Branch generation ms | Main / branch | Latency reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 389.61 | 334.30 | 1.165× | 14.20% |
| 128 | 4 | 391.57 | 357.04 | 1.097× | 8.82% |
| 20000 | 1 | 791.92 | 716.97 | 1.105× | 9.46% |
| 20000 | 4 | 2229.48 | 2153.60 | 1.035× | 3.40% |

Separate engine intervals, main → branch in milliseconds:

| Prompt tokens | Batch | Initial prefill | Mixed prefill/decode | Decode tail |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 57.36 → 67.46 | 0 → 0 | 331.31 → 265.71 |
| 128 | 4 | 56.34 → 66.29 | 0 → 0 | 334.25 → 289.69 |
| 20000 | 1 | 443.54 → 437.88 | 0 → 0 | 346.85 → 277.52 |
| 20000 | 4 | 454.32 → 448.97 | 1405.56 → 1386.52 | 368.42 → 317.06 |

Decode tail is 13.3–20.0% lower. Short-prompt initial prefill regresses by about
17.6%, offsetting part of the decode gain. Long-prompt initial prefill improves
by about 1.2%; its larger share of total time limits the long-B4 generation gain.
These engine intervals include CPU work and are not isolated GPU kernel times.

Separate five-shot GSM8K runs used real drafting and standard rejection sampling.
Main and branch both scored 31/32, with the same questions correct and no
truncation. Token IDs matched exactly for 19/32 answers; the remaining outputs
differed. This is a small accuracy screen, rather than full output parity.
Quality-run elapsed times include cold-shape JIT and are excluded from speedups.

Artifacts are under `/jobtmp/gemma4-main-comparison-20261101/`.
`comparison.json` contains the checked timing and quality comparison;
`performance-comparison.json` contains the timing comparison and all
process medians; the individual `perf-*.json` files retain samples and worker
metadata. `provenance.json` records package and checkpoint identities. Build,
installation, pre-commit and runtime logs are retained alongside them.

## Profiling follow-up

The [2026-11-01 profiling report](PROFILE_20261101.md) ranks remaining costs
using all-rank Nsight Systems traces, Nsight Compute counters, Python/operator
traces, and clean controls with fixed speculative acceptance.

The [residual/QKV/attention optimization follow-up](OPTIMIZATION_20261101.md)
records the next five changes and their matched generation comparison against
this optimized branch.
