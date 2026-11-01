# Gemma4 FP8/MTP: residual, QKV and attention follow-up

These five changes reduce generation latency by **3.9–5.6% against the previous
optimized branch**, with the same configuration and controlled acceptance.
This comparison does not use upstream main as its baseline. The previous
[profiling report](PROFILE_20261101.md) identified the costs addressed here.

## Implementation

| Requested area | Implemented change | Scope |
| --- | --- | --- |
| FP32 residual traffic | Store the inter-layer carry as BF16 at its existing BF16 rounding boundary; keep FP32 statistics and post-attention accumulation | Target full-chain and PDL producers |
| Small-GEMM geometry | Six K partitions of 896 elements, with 128 registers/thread, producing FP32 QKV partials | Local QKV, at most eight rows |
| Separate QKV preparation/cache writes | Consume projection partials, normalize Q/K/V, apply RoPE and write paged K/V in one kernel | Supported target FlashAttention BF16 caches, including packed layout |
| Remaining FP8 quantization | GELU/gating directly produces FP8 at large as well as small row counts; attention combine directly produces FP8 and group128 scales | Raw scales below 32 rows, UE8-rounded scales for larger GELU batches |
| Excessive static FA4 splits | Select effective split counts from device-side request lengths and local windows; skip inactive split work | SM90 paged short-query FA4 |

The captured FA4 maximum grid stays fixed. The optimization skips inactive
splits and limits combine to the active count; it does not resize CUDA graph
launches. The configured `flash_attn_max_num_splits_for_cuda_graph=32` belongs
to the FA3 AOT path and does not force FA4 to use 32 splits.

QKV split-K only pays off when preparation consumes its partials directly.
A standalone partition-reduction launch erased the gain. Stage/register sweeps
and narrower MMA tiles mostly regressed and were not retained. Adaptive FA4
splits alone were roughly neutral or slower in the measured complete boundary;
fusing attention combine with quantization made that boundary faster.

The compiled model sees a fixed-output norm/QKV/preparation custom operation.
Its internal small-row split-K implementation returns FP32 partials, while its
larger-row implementation returns BF16 QKV. Hiding that difference inside the
operation is necessary for compilation ranges that cover both row regimes.

## Matched generation measurements

Eight H100 SXM 80GB GPUs, TP8/PP1, target
`RedHatAI/gemma-4-31B-it-FP8-block`, assistant
`google/gemma-4-31B-it-assistant-FP8-block`, BF16 activation/KV, FA4,
FULL_AND_PIECEWISE graphs through 32 rows, maximum 4096 batched tokens and four
sequences, prefix caching disabled. Both arms use the same Torch 2.14.0+cu132,
FlashInfer 0.7.2, CUTLASS DSL 4.8.0 and Triton 3.8.0 providers.

The baseline snapshots the optimized branch before these five changes.
Its closing restart rebuilds the original Gemma CUDA translation units against
the same common native objects as the candidate. Module-path checks found no
imports from the other checkout on any worker.

Restart order: baseline, candidate, candidate, baseline. Each case uses three
warmup generations and nine timed generations per restart. Values below are
the means of the two process medians. Scheduling is paused while the complete
batch is queued, then resume-to-completion is timed, including CPU work.
All retained timed blocks have zero monitored JIT events.

**Acceptance is fixed at two of four draft tokens in both arms.** Every measured
request has 43 verification steps, 172 drafted tokens and 86 accepted tokens,
with acceptance histogram `[0, 0, 43, 0, 0]`. All worker runtime configurations
and acceptance records were checked by `audit_comparison.py`.

| Prompt tokens | Batch | Previous branch ms | Candidate ms | Latency reduction |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 333.76 | 315.16 | 5.57% |
| 128 | 4 | 357.26 | 343.41 | 3.88% |
| 20000 | 1 | 717.83 | 683.19 | 4.83% |
| 20000 | 4 | 2154.75 | 2061.52 | 4.33% |

The two short-B1 baseline process medians were 332.97/334.54 ms; candidate
medians were 312.22/318.10 ms. This restart variation limits precision on that
case. Both candidate restarts improve every case.

Engine phase medians, averaged across restarts, in milliseconds:

| Prompt tokens | Batch | Initial prefill, before → after | Mixed prefill/decode | Decode tail |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 68.97 → 66.57 | 0 → 0 | 264.30 → 247.85 |
| 128 | 4 | 69.47 → 67.77 | 0 → 0 | 287.23 → 274.48 |
| 20000 | 1 | 438.85 → 417.43 | 0 → 0 | 277.52 → 264.07 |
| 20000 | 4 | 449.76 → 427.59 | 1387.49 → 1326.73 | 316.68 → 306.32 |

These independently computed interval medians need not sum to generation
medians. They include engine/CPU work and are not isolated GPU kernel times.
The long-B4 improvement includes prefill and mixed-phase gains; it is not
explained solely by decode kernels.

Compared with the earlier upstream-main campaign at commit `5d55ef1590c9`,
the latest branch numbers imply generation-latency reductions of 19.11%,
12.30%, 13.73% and 7.53%, respectively, in the table's case order. Those main
medians were 389.61, 391.57, 791.92 and 2229.48 ms. Configuration and two-of-four
acceptance match, but main was not rerun alongside these latest changes.
These overall comparisons are provisional; the paired measurement above
establishes the incremental gain against the previous optimized branch.

## Kernel measurements

CUPTI, graph replay, cold L2, 64 samples, inputs/allocation/compilation outside
timing. Figures describe complete producer/consumer boundaries where stated.

| Boundary | Shape | Before µs | After µs |
| --- | --- | ---: | ---: |
| Post-attention normalization | M4096 H5376 | 95.68 | 91.07 |
| Inter-layer normalization | M4096 H5376 | 97.66 | 77.86 |
| GELU/gating plus quantization | M4096, gate/up width 5376 | 43.36 | 29.15 |
| Local attention plus quantization | B1 short context | 14.18 | 11.55 |
| Global attention plus quantization | B1 short context | 19.55 | 12.27 |
| Local attention plus quantization | B4 long context | 18.11 | 16.61 |
| Global attention plus quantization | B4 long context | 80.13 | 78.19 |
| Norm → local QKV → preparation/cache | M1, packed cache | 16.19 | 14.18 |
| Norm → local QKV → preparation/cache | M5, packed cache | 16.58 | 14.37 |
| Norm → local QKV → preparation/cache | M8, packed cache | 15.86 | 13.78 |
| Norm → local QKV → preparation/cache | M20 fallback | 16.32 | 16.30 |
| QKV preparation plus cache insertion | M4096 D256, packed cache | 56.64 | 31.65 |
| QKV preparation plus cache insertion | M4096 D512, packed cache | 76.06 | 44.99 |

Paired Nsight Compute captures explain the retained geometry. The original QKV
kernel launches 132 CTAs but only 32 have useful output tiles. Six K partitions
launch 192 useful CTAs. Registers/thread fall from 232 to 128 and shared memory
per CTA falls from 169,384 to 75,928 bytes. HBM throughput rises from 0.986 to
1.388 TB/s; active-warp occupancy rises from 8.87% to 11.73%. Eligible warps per
active scheduler cycle fall from 0.19 to 0.10, so the gain is not evidence of
better scheduler eligibility. Both kernels have zero measured local/shared
spilling requests. Isolated NCU duration falls from 11.26 to 7.97 µs; the CUPTI
complete-boundary measurements above establish the useful gain after consuming
the partials.

For M4096 inter-layer normalization, paired NCU durations are 94.82 → 74.30 µs
and HBM bandwidth is 2.405 → 2.528 TB/s. This corresponds to approximately
228 → 188 MB of measured DRAM traffic. Both variants use 78 registers/thread,
4096 CTAs and zero local spilling requests. The reduction in carry bytes,
rather than an occupancy change, explains most of the improvement. NCU replay
durations are diagnostic and separate from the unprofiled model measurements.

Numerical correctness uses explicit error bounds and model quality. Bitwise
identity and token-hash equality are not acceptance gates.

Separate five-shot GSM8K evaluations with actual drafting and standard rejection
sampling scored **126/128 in both arms**, with the same questions correct,
zero invalid answers and no truncation. This is a 128-question quality screen,
not the full GSM8K evaluation.

## Validation

The combined native normalization, activation, QKV and adaptive-attention
selection passed 317 cases. After correcting integration with the packed cache
layout, another 20 graph-replay cache cases passed. The fixed-output custom-op
interface then passed 57 normalization/GEMM and projection-partial cases.
These selections overlap; their counts should not be summed as unique tests.
Graph tests change inputs, request lengths and slot mappings after capture,
including negative/padded cache slots. Numerical tests compare against the
existing operations with explicit tolerances.

The TP8 `test_fp8_mtp_producer_outputs[4]` runtime regression passed. It runs
both specialized and ordinary backends, checks arithmetic answers, and checks
the specialized trace for PDL GEMMs and fused attention combine with no separate
`reshape_and_cache_flash_kernel`. It does not require matching token IDs.

An isolated sanitizer selection exposed a missing explicit import of the
projection reference's custom-op registration in the new partition test.
The broader suite had registered it through an earlier test. The test now
imports its dependency directly.

The corrected four-case Compute Sanitizer memcheck selection passed with
**zero errors**, covering partitioned QKV, packed local/global cache writes,
and adaptive attention combine/quantization under graph replay. The initial
run also reported 14 CUDA API capability-probe errors: the Python CUDA bindings
request version 13041 while the driver exposes 13040. The final run uses
`--report-api-errors no` to exclude these API probes; device memory checking
remains enabled. Both logs are retained.

Live all-rank Nsight Systems captures cover the four benchmark cases with
16 output tokens and the same two-of-four acceptance schedule. All 409,872
kernels map to their CPU launches, and all captures have zero monitored JIT
events. Every rank records six pure-decode target forwards per capture, each
with 60 fused attention-combine/FP8 calls. There are no separate
`reshape_and_cache_flash_kernel` launches in any phase or rank.

Both B1 captures use 49 six-partition QKV GEMMs per decode forward, with grid
`(32, 6, 1)`. Short B4 uses the original QKV geometry. Long B4's tail also
selects the partitioned path when its active batch shrinks: five of six
pure-decode forwards use it. The local/global FA4 maximum grids remain
`(2, 18, 1)` / `(1, 128, 1)` at B1 and `(8, 16, 1)` / `(5, 33, 1)` at short B4.
The active-split logic works inside those launches.

Two target quantization kernels per pure-decode forward remain, versus the
removed per-layer attention-output passes. Large-prefill attention output also
retains its BF16-to-FP8 pass; the fused attention combine currently targets
short queries. Trace durations are diagnostic, not performance measurements.

Formatting, typing, forbidden-import and test-wiring checks passed after adding
an explicit attention implementation type cast and using the repository's
preferred `regex` import. The cast changes neither tensor values nor kernel
dispatch.

## Artifacts and reproduction

Raw artifacts: `/jobtmp/gemma4-opt-20261101/`.

- `run_model.sh`, `perf-{baseline,candidate}-r{1,2}.json`,
  `audit_comparison.py`, `comparison.json`: configuration, source/library hashes,
  worker provenance, all samples, phase intervals and acceptance audit.
- `run_quality.sh`, `quality-{baseline,candidate}-128.json`: separate five-shot
  GSM8K with real drafting and standard rejection sampling, 1024-token cap.
  Quality-run elapsed times are excluded from performance claims.
- `producer-bench.json`, `producer-packed-bench.json`, `qkv-chain-bench.json`,
  `attention-quant-bench.json`: producer and complete boundary timings.
- `pdl-tuning.json`, `splitk-tuning.json`: geometry and negative controls.
- `ncu-{qkv,norm}.ncu-repz`, matching CSV/logs: paired counters.
- `candidate-live*.nsys-rep`: live model trace with node-level CUDA graph tracing.
- `candidate-live.*.analysis.json`, `audit_live.py`,
  `live-integration-audit.json`: all-rank kernel mapping and integration checks.
- `test-combined.log`, `test-fixed-schema.log`, `test-runtime.log`,
  `memcheck-final.log`, `precommit*.log`: validation logs. `memcheck.log` retains
  the initial API-probe reports and isolated-test import failure.

The adaptive FA4 changes ship as
`cmake/patches/flash_attn_sm90_adaptive.patch`, applied after the zero-window
patch by the dependency build. Both patch application and reapplication were
checked. A fresh build must include this dependency patch and the native
Gemma extension. The specialized configuration remains opt-in.
