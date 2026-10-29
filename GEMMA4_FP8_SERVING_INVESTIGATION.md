# Gemma4 FP8 serving: regressions, Nsight findings, and optimization work

Updated: 2026-10-29. This investigation compares native BF16 Gemma4 31B with
RedHatAI FP8-block on TP8 H100s. Speculation is disabled; no MTP models are used.
The goal is to identify and remove FP8 regressions against BF16. A universal
FP8 win has **not** been established.

Latest validated snapshot (2026-10-29):

- Final campaign is complete:12/12 arms validate, with160/162 blocking speed
  points passing and no pending points. Two C57 mixed warm comparisons against
  same-policy BF16 B1 fail: R1 p95=92.947/87.979ms (+5.65%), and R3
  median=48.056/45.177ms (+6.37%). The full speed goal remains open.
- All36 warm p50/p95 points versus original-native BF16 B0 pass. Versus main
  FP8 F0, warm median improves21.2–36.7% and q800 median improves51.3–53.6%.
  All72 decode and18 isolated-prefill checks pass. Ranges are per-case/restart
  ratios, not pooled percentiles. Cold points remain separate/nonblocking.
- Integrated full1319 GSM8K quality passes:1277 established/1257 strict versus
  saved FP8 reference1276/1256, with no errors/truncations. Native tests and
  the delivery adapter's64 device/stream/compile/graph plus2 exception checks
  pass. No model bitwise-equivalence gate is used.
- The complete metrics are in
  `benchmarks/multi_turn/gemma4_fp8_evidence/final_three_restart_metrics.json`.
  The internal review branch target is `ayush1399/vllm:perf/gemma4-fp8-serving`.
  Packaging and the remaining warm-median investigation continue; the full
  speed goal is not achieved. Full details and preserved history follow below.

Final36-file pre-commit validation passed, including Ruff, clang-format,
Markdown, typos, mypy, forbidden imports/APIs and CI test tethering. Log:
`/jobtmp/gemma4-staggered/gemma4-internal-delivery-precommit-final-v2.log`.
The internal branch retains both warm failures; source correctness and quality
passing do not mark the complete speed goal achieved.

The source-only completion-readiness audit is recorded at
`/jobtmp/gemma4-staggered/gemma4-completion-readiness-audit-v1/report-v2.md`.
It maps the requested profiling, workload, native correctness, quality and
numerical-explanation evidence to actual artifacts. Its requested compact
final per-restart/source/config/gate summary has now been added to the branch;
the private frozen plan and local result files alone are insufficient reporting.
Exact historical replay still requires the explicitly documented external
corpus. No extra model evaluation or tail-only campaign was introduced.

The compact evidence exporter is prepared at
`/jobtmp/gemma4-staggered/gemma4-portable-final-summary-v1/export_summary.py`,
SHA256 `27a2f2361c086182cec2caa01ed8b28bb81cfba6b8ac7ea6feaf514c0e63cf1f`.
Root independently reran its CPU checks and reproduced the labeled seven-arm
private export byte for byte (SHA256
`9eb09b740e47e23dd25c94d5c1f905d477cb2aef9b60360dbdc244def18c3770`).
It reuses the pinned evaluator to preserve all180 point outcomes, includes
source/config/quality identities and cached-prefix/new-token labels, and
excludes raw payloads/answers/identity rows. Default final mode rejects
incomplete arms or unevaluable points; actual failed outcomes remain exported.
These were preparation checks, not final results; the private partial export
has not been added to the branch. The complete export is linked above.
Exporter releasev2 subsequently clarifies the prefix scope: warm serving and
all isolated controls cache8192 tokens; cold serving requests omit that prefix
and have zero cached tokens. Current exporter SHA256 is
`34dfce235fb2fe8f5a12cabb4b75b4cce8e9c2cccaa4ab1d7ce3a79c08f08ece`.
Root's second CPU rerun passes and its new private partial export is SHA256
`b2ca1d65178ca26ffecfb171c2a76e42b4c493707ffd861bf0afa62e053720bf`.
Independent JSON comparison confirms that only this scope string and exporter
source identity changed; all metrics/gates/outcomes remain identical. Earlier
release/proof/output artifacts are preserved.

**C57 cold-tail latency is a separate, non-blocking issue.** It already occurs
on genuine main under both original and current serving configurations. Track
it separately from this FP8 optimization; it does not block the speedup work,
quality validation, or delivery. Preserve its measurements as evidence without
treating it as a regression introduced by this change.

Current direction after user review: close the shared tail-variance investigation
as a blocker and assess the speedup independently of that variance. The extra
nine-launch FP8 tail-only campaign is stopped; its reviewed drafts/CPU proofs
are preserved, with nothing frozen, queued or launched. Native integration,
correctness, task quality and three-restart collection are now complete. Two
warm comparisons against same-policy BF16 remain outside the3% band. Prepare
the internal branch with those limits visible and investigate the new warm
median gap; do not dismiss it as the separate cold-tail issue.

Delivery instruction: once validation is complete, prepare and push an isolated
clean validated branch to the user's fork, `https://github.com/ayush1399/vllm.git`,
then provide its link. Preserve concurrent root-worktree changes and include
test/model-evaluation/performance evidence and attribution. The user clarified
that this is for internal use: stop/defer upstream overlap review, and do not
make upstream duplicate checks or upstream submission readiness a blocker.
Existing attribution/overlap notes are retained without further investigation.

## Final integrated campaign and remaining work

The controller and all servers terminated cleanly. The final archive reports
12 clean arms, no error, no model processes/GPU contexts and the API port closed.
Strict final reader:
`/jobtmp/gemma4-staggered/g4-fs2-root-final-v1.json`, SHA256
`692ad62954f6e6ecb32bc9c3ebfb4768432c8da00a6dd27c251d656411d2702c`.
Independent all12-arm/raw NumPy reconstruction matches all180 decisions:
`/jobtmp/gemma4-staggered/g4-fs2-final-independent-v1/report-v2.json`
with SHA256 `ca8d953eda237f90e5f4578cc6fe243d4f068786e53ee7364fad14ca90cbf675`.

| Restart | Blocking results | Retained warm miss against B1 |
| --- | ---: | --- |
| 1 | 53/54 pass | C57 mixed p95:92.947/87.979ms,ratio1.056472 |
| 2 | 54/54 pass | None |
| 3 | 53/54 pass | C57 mixed median:48.056/45.177ms,ratio1.063736 |

Every warm p50/p95 comparison with original-native BF16 B0 passes, with warm
medians20.6–31.3% lower and p9519.8–59.2% lower. Against main FP8 F0, warm
medians are21.2–36.7% lower and p9524.2–55.9% lower. Decode medians improve
1.6–10.4% versus F0 and6.45–13.8% versus B0. All q800 and long-prefill
guards pass. Q800 median mainFP8→F1 is78.088→36.640ms,74.656→36.384ms,
and78.467→36.406ms across the three restarts. These are cumulative
source/configuration comparisons; native capacity1024 differs from optimized64.

The final portable export preserves all162 blocking and18 descriptive outcomes,
including both warm failures and the separate R2 cold-p95 failure. Its SHA256
is `b10c5217d2df886d936f86a61a5db2b91d378d508a8a39b5ad4fb20c8c30d596`.
The initial export (`8767bc982766d7efe6e26c284196955c42a04d47db2c5da0cd4b38696ab5937c`)
is preserved. Packaging renamed only four FP8 input opt-in report keys to
`fp8_input_opt_in` for the spelling hook and updated exporter source identity.
The isolated exporter revision is
`/jobtmp/gemma4-staggered/gemma4-portable-final-summary-v3/export_summary.py`;
its CPU/delta proof confirms all values, counts, gates, outcomes and quality
are unchanged. No measured source or benchmark data changed.
Cold results are17 PASS/1 FAIL and remain nonblocking per user direction.
Task quality remains1277/1257 versus1276/1256 with zero errors/truncations.

The postcampaign adapter GPU check completed under the campaign ownership lock:
64 exact-old-wrapper route cases on two H100s,2 real native-error device
restorations, CPU rejection/fake metadata and post-test GPU idle all pass.
Root also rechecked the21 delivery source hashes against the enumerated lint
proof. GPU proof:
`/jobtmp/gemma4-staggered/gemma4-delivery-adapter-gpu-root-proof-v1.json`, SHA256
`b93e790a964b7a5ff15a880c8e3e7cc965ee8179038fea208d1edeb697d5d89c`.

The remaining R3 warm-median audit is
`/jobtmp/gemma4-staggered/g4-fs2-r3-warm-median-counterexample-v1/report.md`.
For the actual q80 median contributors, the2.879ms F1-minus-B1 gap comprises
-0.197ms scheduled interval,+0.002ms queue and+3.074ms unallocated remainder.
This is same-row bracket arithmetic, not subtraction of marginal percentiles.
The remainder mixes pre-admission, transport, frontend and client parsing;
absolute engine/worker/GPU timestamps are absent, so no frontend-only or
kernel cause is established. Cohort q80 median is only1.51% higher and q800
median0.074% higher; the aggregate median lies near q80's55.6th percentile.
All117 q800 samples exceed both median brackets. Original payloads match,
but dynamic arrivals/send order differ. The failure remains; the next causal
step is request-boundary correlation, rather than assuming a kernel shape issue.

## Validation history

Current validation status (supersedes the historical build-in-progress entries
below): the integrated stable CUDA extension is built and installed. Its SHA256
is `05868591290baf1097a6affae8604ae14c55f24e9318bcaa0bdd239987715db2`.
Fresh tests pass:41 chain cases,65 UE8 cases,1160 existing RMSNorm cases,
64 hybrid/TMA cases,32 graph-policy cases and44 offline detokenizer cases.
One pre-existing detokenizer test requiring an unavailable external tokenizer
was deselected. Additional native proofs pass72 selected structured cases,
9 selected final cases,64 adapter cases (including12 raw-reference checks),
and M32768 compiled/CUDA-graph checks. These selected cases do not repeat all
216/27 independent prototype launch configurations. The independent audit is
`/jobtmp/gemma4-staggered/gemma4-v7-native-validation-independent-v1/audit-v1.json`.
The original build driver reported a bookkeeping failure after successful
compilation/install because it expected a different manifest filename; the
additive `build-install-verified-v2.json` retains and resolves that failure.

Final matched serving validation has started, with controller PID2276942 and
first genuine-main FP8 API PID2277150. The frozen plan is
`/jobtmp/gemma4-staggered/g4-fs1-frozen-plan-v1.json`, SHA256
`290c886d91cad9eb61af73a444ecf0c20c32b408b1aa1cc518681b815ca82dc3`.
It compares original-main and integrated FP8/BF16 configurations over three
fresh restart blocks with reversed order, using the existing C4/C13/C57
workloads and isolated prefills. The integrated native build receives its own
1319-question quality evaluation on the first integrated FP8 arm; the saved
quality baseline is reused. Startup is observed, but no final serving or
integrated-quality acceptance result is available yet. Live status is
`/jobtmp/gemma4-staggered/g4-fs1-source-archive/status.json`.

The first F0 server subsequently became ready and began benchmark controls.
Independent replay of its saved `eligibility-before.json` and
`expected-before.json` passes the frozen metadata contract: all8 ranks use the
genuine native extension, FA4, BF16 KV and original graph policy, without private
fusion operators or timing hooks. Its saved `dispatch.json` confirms limit512
and no graph coverage for q800/1040/2048. This establishes first-arm eligibility,
not a completed performance comparison. Two delivery-only follow-ups are staged
outside the sealed runtimes: an explicit local reset-endpoint environment
export in the reproduction README and the FlashInfer source path in H100 CI
test-selection dependencies. Apply them after campaign completion.

Restart1 F0 completed with clean exit0 and is valid under the strict offline
reader (`g4-fs1-root-partial-after-f0-v1.json`). All4188 measured serving requests,
isolated protocols, first-token/eager controls, restored dispatch and all-rank
before/after identity pass. Its warm-request TTFT p50/p95 in ms is:

| Case | Main FP8 p50 / p95 |
| --- | ---: |
| C4 warm | 34.18 / 43.34 |
| C4 mixed | 34.31 / 84.26 |
| C13 warm | 35.35 / 50.84 |
| C13 mixed | 35.58 / 98.85 |
| C57 warm | 61.51 / 97.87 |
| C57 mixed | 71.34 / 205.64 |

Isolated q800/q8192/q18304 medians are78.088/194.210/423.348ms,
with400 measured requests each. The controller has started restart1 B0 BF16
(API PID2346184). Only1/12 arms is validated; comparisons remain unevaluable.
The partial reader's missing-status issue for the live B0 arm is expected,
not a failed completed run or an acceptance result.
Independent first-arm replay also passes. This unprofiled genuine-main arm
retains a C57 mixed cold TTFT p95 of308.361ms (37 cold requests) and warm q800
p95 of252.188ms (117 requests), providing another main counterexample to
fusion-specific attribution. These are descriptive results under the user's
separate/nonblocking cold-tail instruction; no tail-only campaign is reopened.
Restart1 B0 has now become ready. Independent replay of its saved metadata
confirms all8 ranks use240 unquantized BF16 linear sites each, tied BF16
embedding/head weights,241 weighted norms,180 head norms, FA4 and BF16 KV.
Private FP8 operators/flags/guards and FX calls/operator references are absent;
original512-token graph dispatch matches the frozen B0 contract. Its performance
arm subsequently completed with clean exit0. Root strict replay in
`g4-fs1-root-partial-after-b0-v1.json` validates both F0 and B0, including equal
measured-plan hashes/counts across all six cells. Integrated F1 has started
(API PID2351907);2/12 arms are validated, with no final speedup acceptance yet.

First matched baseline pair: main FP8 relative to BF16 (positive is slower).
TTFT uses warm requests; decode uses all measured requests in each case.

| Case | TTFT p50 change | TTFT p95 change | Decode p50 change |
| --- | ---: | ---: | ---: |
| C4 warm | +4.42% | +5.83% | -9.19% |
| C4 mixed | +2.76% | +43.67% | -9.23% |
| C13 warm | +1.26% | +3.92% | -6.51% |
| C13 mixed | +0.30% | +57.54% | -1.08% |
| C57 warm | +4.86% | +11.66% | -3.92% |
| C57 mixed | +8.50% | -6.37% | +0.83% |

Thus the fresh-main pair reproduces the warm-TTFT regression. Preserve the C57
mixed decode counterexample rather than claiming a uniform baseline decode
advantage. Isolated q800 median is78.088ms main FP8 versus55.570ms BF16
(+40.52%); q8192 is194.210 versus221.807ms, and q18304 is423.348 versus
484.804ms. These first-restart baselines are not candidate results or repeated
validation. That partial snapshot preceded the F1 preflight failure below.

**Final campaign stopped at first F1 preflight; candidate results remain
unmeasured.** After342 graph captures, the private validator raised
`ValueError('Actual saved FX path/operator counts')`. Server exit was0, the
controller terminated and its final idle check passed. F1 served zero completion
or chat requests; no candidate timing or quality result is being discarded.
The failed directory, original frozen sources and both valid baselines are
preserved.

All-eight-rank CPU AST inspection identifies a semantic observer error: every
backbone contains120 full-chain calls and1 initial producer,120 FP8-input GEMMs
(60 QKV/60 gate-up), and120 dynamic GEMMs (60 output projections/60 down
projections). Direct-call and reference counts agree; no consumer is unclassified.
The validator incorrectly expected zero dynamic GEMMs. This is not evidence of
missing fusion. The new private-validator version must require exactly120
dynamic calls, save raw metadata before checking it, and revalidate full runtime
eligibility. It must preserve every performance/quality threshold. Continuation
work reuses the two baselines only through their original frozen reader and
provenance, with fresh directories/caches for all remaining launches. No kernel
or runtime source change is indicated by this failure.

The reviewed V3 continuation is now frozen and executing under controller
PID2421754. Plan: `/jobtmp/gemma4-staggered/g4-fs2-frozen-plan-v1.json`, SHA256
`b6a94f066f4e322d12c87e72141cedc21b4aec3a17aa0862dd4cd608a6a8afcc`.
Its source handoff SHA256 is
`f201865d50703faac7f976b8714780cc98f02459d9f6566c6e7f36fbbc26e2cb`;
the independently reproduced CPU proof SHA256 is
`784ebc6917c96b4c08f07fd2f88150502b190cd8e0369687be7b71c6db8e264a`.
The continuation pins all original evidence plus16 failed-launch FX/factor
files, revalidates F0/B0 in an isolated original-reader subprocess, and starts
only10 fresh servers. All180 point definitions remain unchanged:162 blocking
speed points and18 descriptive cold points. First-F1 quality remains the same
single1319-question evaluation. The observer now requires exactly120 remaining
dynamic GEMM calls/references and saves raw RPC metadata before validation;
48 missing/extra count mutations are rejected in CPU checks. No candidate
performance or integrated-quality result is available yet. Live continuation
status is `/jobtmp/gemma4-staggered/g4-fs2-source-archive/status.json`.

The fresh F1 API PID2422402 subsequently passed full saved eligibility and
independent replay on all8 ranks: actual native05868591, CUDA out-op schema,
whole/config guards,120 supported FP8 sites,120 chain calls+1 initial producer,
120 ordinary output/down GEMMs, FA4, BF16 KV and no timing hooks. Actual captured
coverage is342 descriptors over331 sizes, with q800/1040/2048 dispatchFULL.

First-restart isolated results below have400 measured samples after10 warmups
per query/arm. All410 row identities, prompt hashes, exact8192 cached-prefix
counts and new-token counts match across the three arms. Query length is the
new suffix, rather than total context length. Values are TTFT p50/p95 in ms.

| New tokens | BF16 B0 | Main FP8 F0 | Integrated FP8 F1 |
| --- | ---: | ---: | ---: |
| 80 | 30.044 / 31.654 | 31.618 / 36.190 | 23.471 / 24.169 |
| 800 | 55.570 / 59.101 | 78.088 / 81.777 | 36.640 / 37.411 |
| 8192 | 221.807 / 224.523 | 194.210 / 197.651 | 177.384 / 179.806 |
| 18304 | 484.804 / 488.563 | 423.348 / 427.260 | 395.172 / 397.782 |

F1 q800 median improves53.08% versus main FP8 and34.06% versus BF16; p95
improves54.25% and36.70%, respectively. Its q80 median improves25.77% versus
main FP8 and21.88% versus BF16. These are preliminary first-restart shape
results, not full-arm or repeated acceptance. Long-prefill q8192/q18304 medians
improve8.66%/6.66% versus main FP8, with p95 improving9.03%/6.90%. All1230
q800/8192/18304 rows and summaries replay correctly; first-token/eager control
grids and all8 before/restored identities pass. The candidate has moved to
serving/decode measurements. After-traffic identity, integrated quality and later
restart checks remain outstanding. The earlier failed pretraffic F1 remains
preserved separately.

First F1 serving traffic subsequently finished. All4188 measured requests pass
the original six plan/hash/cache/timestamp checks, and all8 before/restored/
after-serving metadata snapshots match. The native integrated quality run is
in progress; full-arm teardown and repeated acceptance remain pending. Warm
TTFT p50/p95 below is in ms; decode is F1 all-request median ms/token.

| Case | BF16 B0 TTFT | Main FP8 F0 TTFT | Integrated FP8 F1 TTFT | F1 decode |
| --- | ---: | ---: | ---: | ---: |
| C4 warm | 32.732 / 40.956 | 34.178 / 43.343 | 25.473 / 32.845 | 5.947 |
| C4 mixed | 33.392 / 58.646 | 34.314 / 84.255 | 25.618 / 38.109 | 5.999 |
| C13 warm | 34.912 / 48.924 | 35.353 / 50.841 | 27.695 / 38.455 | 7.298 |
| C13 mixed | 35.477 / 62.746 | 35.585 / 98.847 | 28.036 / 44.659 | 7.612 |
| C57 warm | 58.658 / 87.649 | 61.508 / 97.870 | 43.111 / 56.241 | 12.243 |
| C57 mixed | 65.757 / 219.627 | 71.344 / 205.639 | 45.176 / 92.947 | 14.976 |

Warm TTFT medians improve20.67–31.30% versus BF16 and21.21–36.68% versus
main FP8; every listed warm p95 improves too. Decode p50/p95 improves in all
six cases versus both references: median gains are6.45–13.45% versus BF16
and2.26–9.55% versus main FP8. These results cover the first restart only;
the integrated BF16 B1 control and reversed restart blocks remain outstanding.
C57 mixed cold p95 is302.189ms (37 requests), and warm q800 p95 is185.099ms
(117 requests), versus308.361/252.188ms on main FP8. Outliers remain;
no general tail fix or fusion-specific tail attribution is claimed, and the
cold measurements retain their separate descriptive/nonblocking status.

First integrated F1 then completed quality and exited cleanly. Exact frozen
V3 `read_arm` passes independently for all4188 performance rows, controls,
source/provenance, four all8 identity snapshots and the quality artifact:
`/jobtmp/gemma4-staggered/g4-fs2-root-f1-arm-read-v1.json`.
Scores are1277 established/1257 strict out of1319 versus the saved required
flag-off FP8 reference1276/1256, with zero errors/truncations and all stop
finishes. Paired wins831/951 and loss1016 were privately reviewed as reasoning
differences. A saved-only descriptive comparison with the earlier full FP8
evaluation1276/1257 gives one established gain in aggregate and unchanged strict
score. Protocol/prompt/payload/dataset/scorer identities match; no baseline was
rerun and no statistical gain/equivalence is claimed. Independent quality audit:
`/jobtmp/gemma4-staggered/g4-fs2-r1-f1-quality-independent-v1/review.json`.
Raw discordant answers remain outside the branch.

The frozen V3 aggregate reader exposed a separate offline-only representation
bug: original-reader subprocess JSON carries isolated identity rows as lists,
while fresh `read_arm` returns tuples. Values/order/counts match. A reviewed
external adapter at
`/jobtmp/gemma4-staggered/gemma4-fs2-offline-reader-v4/read_campaign.py`
(SHA256 `efbb528f0d1e2a7402cc9ff16bca987724ced3ef8a9fbc41d49cefa8ca0bf037`)
restores only tuple representation, proves unchanged JSON values, and preserves
all frozen source and thresholds. Independent CPU proof reproduces the original
failure and rejects changed identity/hash/warmup/order/count data. Its first
partial report validates F0/B0/F1:36 blocking points and6 descriptive cold points
pass, with126 blocking and12 cold points still unevaluable. No point fails;
this is3/12 valid arms, not full goal acceptance. Future aggregate reads use
this external adapter; the active controller and measured data are unchanged.

First B1 (API PID2492686) has also passed independent initial eligibility:
all8 workers use ordinary BF16 methods/weights and FA4/BF16 KV, with whole-chain
guard false, zero private FX calls/references and342 FULL descriptors. Its
performance arm is running. B1 and the later reversed restart blocks remain
required before completion and fork push.
First B1 isolated controls pass the same410-row identity/cache checks. Its
q80/q800/q8192 medians are24.241/36.906/207.219ms; F1 is3.18%/0.72%/14.40%
faster at those shapes. Thus the first isolated comparison remains favorable
against BF16 with the same FULL policy, with a narrow q800 margin. These are
preliminary first-block results, not repeated or full-arm acceptance.

First B1 subsequently completed and independently passed full-arm validation:
all4188 measured rows, controls, all8 before/restored/after identities, source
archives and clean exit0. No quality baseline was rerun. The complete first
block has4/12 valid arms and **53/54 blocking speed points pass, with one fail**.
All6 descriptive cold points pass. Report:
`/jobtmp/gemma4-staggered/g4-fs2-root-after-block1-v1.json`.

The retained failure is **C57 mixed warm TTFT p95 versus B1** (BF16 with the
same FULL policy): F1 92.9473769ms versus B1 87.9790385ms, ratio1.0564718424,
or+5.65% against the frozen+3% limit. F1 median is3.32% faster in this cell;
the other five B1 warm p95 comparisons improve3.35–6.44%. Comparisons with
original-main BF16 B0 and main FP8 F0 remain favorable as recorded above.
This warm-request failure is preserved and keeps the full goal open; it is not
reclassified as a descriptive cold point or pooled away. A single pair does not
establish a kernel-caused regression. The frozen reversed-order blocks2/3
continue unchanged, starting with block2 B1 (API PID2501044). A narrow CPU-only
review of the existing percentile-bracket rows is underway; no new GPU trace or
cold-tail campaign is being introduced.
That bounded CPU audit reproduces all1160 warm records (1043 q80/117 q800)
with exact plan/cache/output-length identities. F1 q80 p95 is89.040ms versus
B1 81.882ms, while q800 p95 improves185.099 versus221.007ms. F1 has56 warm
requests above150ms versus49 for B1, but only7 above200ms versus49: the
distributions cross. The first-F1 closed-loop window is38.810s versus42.153s;
only138/1197 request positions match in observed arrival order. These are
descriptive observations, not causal attribution or a replacement acceptance
metric. Per-request p95 brackets include substantial scheduled-to-output time,
so an all-frontend explanation is unsupported. Report:
`/jobtmp/gemma4-staggered/g4-fs2-r1-warm-p95-counterexample-v1/report-v2.json`.

Delivery-only updates are prepared outside the sealed runtimes. Apply portable
reproduction v2, then v3's reset-environment README correction, v4's completed
native-validation record and v5's quality-reference clarification; packages are
`/jobtmp/gemma4-staggered/gemma4-internal-branch-evidence-v{2,3,4,5}/`.
The H100 CI dependency follow-up is separately staged at
`/jobtmp/gemma4-staggered/gemma4-ci-source-dependency-fix-v1/`.
All were checked in package-owned scratch; none has been applied to campaign
sources. F1 quality uses the saved corrected-v2 flag-off FP8 reference, rather
than B0/B1 BF16 or genuine main; BF16 fallback is verified separately. Final
integrated quality/performance/restart fields remain pending in the staged
record, while completed native tests and bounded numerical findings are filled.
The separate delivery checkout is now
`/jobtmp/gemma4-fp8-internal-delivery-v1`, on local branch
`perf/gemma4-fp8-serving` at base274. It contains the exact21 validated source
files plus the reviewed benchmark/documentation patches and CI dependency fix.
The campaign runtimes remain sealed and unchanged, including their Git HEADs;
delivery commits will be made in this separate checkout. The supplied author
identity and fork remote are configured locally. No commit or push has occurred;
final model-quality, performance and restart results still gate delivery.
Packaging subsequently applied Ruff/clang-format and Markdown table formatting
in the delivery checkout only. The model's normalized AST/import order and
C++ lexical content (including concatenated schema strings) remain unchanged;
the only non-format source difference from the validated runtime is the reviewed
CI dependency entry. All21 source paths are checked in
`/jobtmp/gemma4-staggered/gemma4-internal-delivery-format-proof-v1.json`.
All15 packaged Python files pass Ruff check/format; clang-format, Markdown lint,
three benchmark CLI help commands and `git diff --check` pass. No model or
native tests were rerun for formatting. Campaign source/binary seals remain
unchanged.

Delivery lint follow-up (2026-10-29): the first full pre-commit pass found
technical UE8 spelling, two test type-narrowing omissions, a forbidden stdlib
regex import in the portable quality helper, and forbidden CUDA convenience
API calls. The delivery checkout now has narrowly scoped UE8 identifier
handling, explicit non-None assertions in the two tests, generic accelerator
synchronization in the TMA test, and a generic accelerator device context in
the full-chain adapter. An explicit CUDA input guard preserves its device
restriction. These are additional changes beyond the earlier formatting/CI
record; the measured runtimes and native binaries remain unchanged.
The enumerated source-diff proof is
`/jobtmp/gemma4-staggered/gemma4-internal-delivery-lint-source-proof-v1.json`,
SHA256 `ee620fd9d8a29ae29bbaaffaa509c47e173f84a1a17f1882ad22a578e2ec957c`.
The complete35-file pre-commit rerun now passes, including Ruff, clang-format,
Markdown lint, typos, mypy, forbidden imports/APIs and CI test tethering. Log:
`/jobtmp/gemma4-staggered/gemma4-internal-delivery-precommit-v2.log`.
The portable helper now uses project regex with explicit C0 whitespace. Its
separate CPU proof checks3957 saved answers and their golds,4807 edgecases and
all1114112 Unicode property codepoints; strict acceptance/Decimal results agree.
The unchanged inference-function ASTs and changed scorer AST are distinguished
in the portable evidence, rather than claiming all scorer code is identical.
Proof: `/jobtmp/gemma4-staggered/gemma4-quality-regex-compat-proof-v1/proof.json`.
No baseline/model evaluation was repeated for this packaging correction.
Actual delivery
adapter device/stream/graph checks must run after the campaign releases GPUs;
the source proof alone does not discharge that validation.

The postcampaign delivery-adapter verifier is now prepared at
`/jobtmp/gemma4-staggered/gemma4-delivery-adapter-check-v1/README.md`.
Its manifest SHA256 is
`dacf69e06e57ddb52da7e17e92c1970bfdad6965f912d29ee4832c24b97a9398`.
Root independently reran its standard-library-only source check successfully
to `/tmp/delivery-adapter-root-source-review-v1.json`. The future GPU check
compares64 two-device eager/compile/graph cases with the exact frozen wrapper
function bodies, without importing/registering the old adapter module, plus
two real native-error device-restoration cases. This tests an API-only change
using the same native provider, rather than adding a model bitwise gate.
No GPU execution or acceptance is claimed by the CPU preparation proof.
The campaign holds
`/jobtmp/gemma4-staggered/quality-baseline-source-transition.lock` exclusively.
After all12 arms complete and the controller is terminal, acquire that same
lock and verify the API port and GPUs are idle before the adapter GPU check.
An idle gap between campaign arms does not authorize a competing test launch.

The second-block B1 arm has completed cleanly and passed the strict partial
reader, bringing the valid count to5/12. Report:
`/jobtmp/gemma4-staggered/g4-fs2-root-after-r2-b1-v1.json`.
The second-block F1 subsequently completed cleanly and passed the strict reader,
bringing the valid count to6/12. The cumulative blocking speed-point counts are
71 PASS,1 FAIL,90 NOT_EVALUABLE; the single failure remains the first-block
C57 mixed warm p95 against B1. No acceptance gate was changed or pooled away.
Report: `/jobtmp/gemma4-staggered/g4-fs2-root-after-r2-f1-v1.json`, SHA256
`0845c3301b002db9d8588b98464ba4b61f3ab0f62df883cdd3f01ab68936d76a`.

All12 second-pair warm TTFT p50/p95 comparisons against B1 pass the1.03 limit.
Values below are milliseconds; B1 is the common-policy optimized BF16 control,
not the original B0 policy or a newly completed main-FP8 F0 comparison.

| Case | B1 p50 / p95 | F1 p50 / p95 |
| --- | ---: | ---: |
| C4 warm | 28.097 /34.444 | 25.318 /32.241 |
| C4 mixed | 28.368 /40.031 | 25.357 /37.839 |
| C13 warm | 30.526 /40.501 | 27.630 /38.962 |
| C13 mixed | 30.747 /46.017 | 28.038 /45.965 |
| C57 warm | 44.111 /56.658 | 43.746 /55.460 |
| C57 mixed | 45.650 /203.420 | 46.619 /102.019 |

Thus the first-pair C57 mixed warm p95 miss did not repeat: first-pair F1/B1
was92.947/87.979ms, second-pair102.019/203.420ms. This is descriptive evidence
of variation between runs, especially in the BF16 control; it does not establish
kernel causality, waive the first failed point, or prove three-restart success.
Second-pair F1 decode p50 beats B1 in all six cases by7.0–14.5%. Block2 B0 is
now live (API PID2582138); block2 F0 and all of block3 remain pending.
Second-pair isolated q800 p50/p95 is36.384/37.172ms F1 versus
37.561/38.448ms B1; q8192 is176.263/178.583 versus207.753/210.138ms,
and q18304 is393.849/397.486 versus464.713/467.405ms. These are
descriptive B1 comparisons; second-block main-FP8 q800/long-prefill gates
are not yet evaluable because its F0 arm has not run.

Independent raw audit:
`/jobtmp/gemma4-staggered/g4-fs2-r2-optimized-pair-independent-v1/README.md`.
It replays the unmodified frozen arm/raw/metadata/control checks and separately
reproduces warm/decode percentiles with NumPy. All18 evaluable second-pair
points pass. It also retains a descriptive subset counterexample: C13 mixed
q800-cohort p95 is57.5365ms F1 versus48.7451ms B1 (+18.04%,49 measured
requests each), even though aggregate warm p95 is45.965/46.017ms. That cohort's
median improves40.371/42.016ms. C57 mixed q800 median is60.252/59.298ms
(+1.61%,117 each), while its p95 improves. These observations are not new
acceptance gates or causal claims about actual batched GEMM shapes; the
request cohort does not determine the executed batch shape.

Second-block B0 subsequently completed cleanly. Strict replay validates7/12
arms with89 PASS,1 FAIL and72 NOT_EVALUABLE blocking speed points. All18
new F1/B0 comparisons pass: warm TTFT p50 improves20.6–26.6%, p95
improves20.0–54.2%, and decode p50 improves6.7–13.5% over the six cases.
These gains are against native BF16's original graph/capacity policy; they
do not replace the separately reported B1 common-policy comparisons.
Report: `/jobtmp/gemma4-staggered/g4-fs2-root-after-r2-b0-v1.json`, SHA256
`cd963f8ec2a71e2153720603009a9daad04ecde3e3d6606387668ee8b5604fde`.
Second-block F0 is now live (API PID2588551). Its results are still needed
to evaluate that block's FP8-baseline q800, decode and long-prefill guards.

Second restart block is now complete: F0 exited cleanly and all four arms pass
the strict source/workload/control checks. All54 blocking speed points for
this block pass. Cumulative counts are107 PASS,1 FAIL,54 NOT_EVALUABLE;
the first-block warm-p95 miss remains unchanged. Report:
`/jobtmp/gemma4-staggered/g4-fs2-root-after-block2-v1.json`, SHA256
`db7442c4072e9f4053c506c607610e81a2de238e01ccbf5bbd69f99e822dd3c1`.

Second-block warm TTFT p50/p95, in milliseconds:

| Case | Native BF16 B0 | Main FP8 F0 | Optimized FP8 F1 |
| --- | ---: | ---: | ---: |
| C4 warm | 32.840 /41.134 | 33.295 /44.280 | 25.318 /32.241 |
| C4 mixed | 33.411 /57.846 | 34.361 /83.323 | 25.357 /37.839 |
| C13 warm | 34.878 /48.699 | 35.470 /52.515 | 27.630 /38.962 |
| C13 mixed | 35.302 /62.107 | 36.273 /92.750 | 28.038 /45.965 |
| C57 warm | 59.584 /88.878 | 63.072 /103.678 | 43.746 /55.460 |
| C57 mixed | 63.193 /222.531 | 67.560 /187.483 | 46.619 /102.019 |

Relative to main FP8, F1 warm median improves22.1–31.0%, p95 improves
25.8–54.6%, and decode median improves1.6–10.4% across the six cases.
Isolated q800 p50/p95 is36.384/37.172ms F1 versus74.656/78.943ms F0,
improving51.26%/52.91%. Long q8192 improves8.33%/8.57% and q18304
improves6.32%/6.74%. These remain separate per-restart comparisons.

Counterexamples remain visible. Native F0 C57 mixed decode median is16.680ms
versus16.416ms B0 (+1.61%), while F1 is14.950ms. C13 mixed warm TTFT p99
is162.792ms F1 versus152.787ms F0; p99 is descriptive here. C57 mixed cold
p95 is310.673ms F1 versus289.153ms F0 (+7.44%); that point remains
descriptive/nonblocking under the user's cold-tail instruction. The12 evaluable
cold points now comprise11 PASS and1 FAIL, with6 pending. No kernel cause or
new tail-only investigation is inferred. Third-block B0 is live (API PID2657589).

Independent complete block2 audit:
`/jobtmp/gemma4-staggered/g4-fs2-block2-independent-v1/README.md`.
It revalidates all four frozen contracts and directly reproduces raw
p50/p95/p99 summaries; all54 blocking decisions pass. Its machine report
retains all unfavorable comparisons, including C13 mixed warm p99 versus
B0/B1 (162.792 versus122.527/54.210ms) and C57 warm p99 versus B1 (+3.27%).
These remain descriptive, without new gates or causal conclusions. Root's
eight-arm private portable export also preserves the warm failure and the
separate cold failure, with SHA256
`e33c99ca696ff9a9900ce69e3532fb07f0ae2e8552da64c2712e9685a7c9603f`.
It remains labeled partial and has not been promoted to final branch evidence.
The portable reproduction README now explicitly records the frozen model and
workload orders: block1 F0/B0/F1/B1 at4/13/57, warm/mixed; block2 B1/F1/B0/F0
at57/13/4, mixed/warm; block3 B0/F0/B1/F1 at13/4/57, warm/mixed. These were
read from the unchanged frozen plan, not chosen after observing results.

Third-block B0 completed cleanly and passes the strict arm reader, bringing
the validated count to9/12. All original native-BF16 source/role, workload,
cache/count, isolated/control and restoration checks pass. There are no newly
evaluable block3 F1 comparisons yet, so blocking counts remain107 PASS,
1 FAIL and54 pending. Report:
`/jobtmp/gemma4-staggered/g4-fs2-root-after-r3-b0-v1.json`, SHA256
`91fcd10c4d2a5e9d60f17474e82f7ee5c48b7f77f7d5e1132e52a82d47c11f9f`.
Third-block F0 is live (API PID2664692); B1 and F1 follow in the frozen order.

Third-block F0 subsequently completed cleanly and passes the strict reader,
bringing the validated count to10/12. Both original native baselines now have
three validated restarts. There are still no third-block F1 comparisons, so
blocking counts remain107 PASS,1 FAIL,54 pending. Report:
`/jobtmp/gemma4-staggered/g4-fs2-root-after-r3-f0-v1.json`, SHA256
`e9cd407bd78ecc076c4457a48f7817fcbc7a5bdc764e9920d6ae5c101fdbb990`.
Third-block B1 is live (API PID2730152), followed by the final F1 arm.

Independent three-original-baseline-pair audit:
`/jobtmp/gemma4-staggered/g4-fs2-native-three-pair-independent-v1/README.md`.
All six native arms pass their original/recovered frozen contracts and direct
raw NumPy checks. Positive percentages below mean native FP8 is slower than
native BF16; entries are separate restart1/restart2/restart3 ratios.

| Case | Warm TTFT p50 change R1 /R2 /R3 | Warm TTFT p95 change R1 /R2 /R3 |
| --- | ---: | ---: |
| C4 warm | +4.42% /+1.39% /+3.34% | +5.83% /+7.65% /+4.67% |
| C4 mixed | +2.76% /+2.84% /+6.24% | +43.67% /+44.04% /+47.87% |
| C13 warm | +1.26% /+1.70% /+0.85% | +3.92% /+7.84% /+4.59% |
| C13 mixed | +0.30% /+2.75% /-1.23% | +57.54% /+49.34% /+39.31% |
| C57 warm | +4.86% /+5.85% /+8.89% | +11.66% /+16.65% /+11.25% |
| C57 mixed | +8.50% /+6.91% /-4.67% | -6.37% /-15.75% /-14.71% |

Thus warm p95 is worse in native FP8 in five of six cases in all three pairs;
C57 mixed p95 is better in all three, and mixed C13/C57 medians reverse in R3.
The isolated q800 regression repeats: FP8 p50 is34.02–40.52% slower and p95
is32.46–38.37% slower. Isolated long-prefill FP8 medians remain12.4–13.6%
faster. Native C57 mixed decode p50 is slower in all three pairs by0.829%,
1.611% and1.662%; all other decode p50/p95 cells improve in every pair.
These are serving observations, not standalone kernel timing or causal proof.

All-rank native metadata confirms effective max_num_seqs1024 in the original
profile, versus64 in F1/B1. The reproduction README now names that capacity
tradeoff explicitly; cumulative main-to-optimized gains combine code and
configuration, while B1 preserves the same-policy BF16 control. No third-F1
acceptance is inferred from this completed baseline-only audit.

Third-block B1 completed cleanly and passes the strict source/fallback,
workload/cache, isolated/control and graph-restoration checks. This validates
11/12 arms; all54 third-block candidate comparisons remain pending. Report:
`/jobtmp/gemma4-staggered/g4-fs2-root-after-r3-b1-v1.json`, SHA256
`df641ac044193beec3626f68216f8106bf22d5b4944984b3964ad3bb0ee5b246`.
The same-policy BF16 C57 mixed warm-p95 values across the three restarts are
87.979,203.420 and96.777ms. They remain separate run observations, without
pooling or a causal conclusion about the first failed comparison. Final F1
is now live (API PID2735909), with no new quality evaluation scheduled.

Frozen V7 prototype quality validation completed against the saved corrected-v2
flag-off reference; the baseline was not rerun. The exact1319-question protocol
and all baseline prompts/labels/payloads/scoring were independently audited.
V7 scores1278 established and1258 strict, versus1276/1256, with zero errors
and zero truncations. Both nondecrease gates pass. There are three paired wins
(157,831,951) and one loss(1016); do not claim a statistical quality gain.
No bitwise or identical-text gate is used. Frozen plan:
`/jobtmp/gemma4-staggered/g4-v7q1-frozen-plan-v1.json`, SHA256
`576d1281f891fe9b7cb61be73ae382aed4d958ff8a2488d87cc3ac2b3d286bb2`.
Root session27707 completed with clean server exit and final GPU idle. All-eight
rank identity/FX/nohook checks bracketed the evaluation. This is prototype V7
quality evidence, not evidence for the subsequently rebuilt native binding.
No performance/tail benchmark ran in this evaluation.

The exact-source review package is preserved at
`/jobtmp/gemma4-staggered/gemma4-v7-review-package-v1/`; release SHA256
`18e9d3a92de2f74589e48b3beeaea2d76bfaef2c55b5f966a402ac7068d956a8`.
Its three-patch stack and combined patch reproduce14 source files against
main274 in independent scratch apply checks. Both native dependencies are
included as source: the existing stable extension's fused producer and the
prototype full-chain CUDA kernel. No binary-only optimization is proposed.
Build integration source is complete separately in
`/jobtmp/gemma4-v7-review-integration-v1`, a detached main274 clone with the
exact package applied. It registers the chain through vLLM's existing
TorchStable/CMake extension, replacing the prototype's manual ctypes library;
this uses existing CUDA build/install support. The measured/quality V7 runtime
and the user's concurrent root-worktree changes remain untouched.

CPU-only integration staging completed in the new checkout/build root: a fresh
uv-managed `.venv`,1003 original provider/vendor byte copies (excluding the old
primary stable extension), and10 privately copied dependency trees. Both the
old `_C_stable_libtorch` and manual `_gemma4_full_chain.so` are absent. The
vllm-project FA4 interface remains SHA256
`0d00d4b96c9cd616eae488ae394d6c50ab22e403a01df5a896995599fd62b2e3`.
Staging script: `/jobtmp/gemma4-staggered/stage_gemma4_v7_review_integration_v1.sh`,
SHA256 `c11128575c844c541a2eab221d769387faeae7bec5450f95cf8ece5b1218b7b0`.
No native build, kernel test or GPU import ran during staging. Source integration
and review are complete; root is building after the quality run and idle gate.
The stopped nine-launch tail-only campaign remains stopped. Final performance
validation preparation concerns the original warm/decode/isolated speedup goal;
shared tail variance is retained as descriptive evidence per user direction.

Native integration source is complete and independently reviewed. The sealed
release `/jobtmp/gemma4-staggered/gemma4-v7-native-integration-release-v1.json`
(SHA256 `dfc7e89887dff4dcf1b318ab432f92ac148e64167b83eb1494e865738729bd56`)
binds17 cumulative source files and9 integration edits. The complete CUDA
template/launch body matches frozen V7 exactly (body SHA256
`7303d79a5fa9f9ef3059e63b791fa3522655d804b234692463e4d8df4e8f4e44`).
The stable out-op validates dtype/device/shape/mode/epsilon/alignment/nonalias
contracts and uses the current stream. Allocation/fake/scale-stride/raw-small-M
and final empty-carry behavior remain preserved. This source proof is not a
bitwise output requirement or a substitute for GPU correctness. The existing
kernel suite now contains41 new source-derived cases, not yet run.

Root queued configure-only session41914 behind the exact quality controller's
PID/create-time and successful completed quality status. It rechecks HEAD,
all17 changed/new source paths and hashes, then takes the shared exclusive lock
and idle gate before running the isolated CMake configure. It does not build,
install, run GPU tests or launch another model automatically. Independent queue
review: `/jobtmp/gemma4-staggered/g4-v7-native-configure-queue-independent-review-v1.json`.
Queue status: `g4-v7-native-configure-queue-status-v1.json` in the same directory.
Configure session41914 completed successfully. Root inspected the actual chain
compile command: SM90, O3, C++20 and no fast-math/changed division flags. Cached
source/install/Python/fetch paths and all17 seals pass. The fresh stable-only
build/install is running in session13338 (driver2236855), under the same lock.
Status: `/jobtmp/gemma4-v7-review-integration-build-v1/build-install-status-v1.json`.
GPU kernel/proof execution remains pending build/install. The prepared
stable-op proof scope is72 selected structured and9 selected final cases,
64 adapter cases/12 raw FlashInfer checks and M32768; do not label that a rerun
of all216/27 independent prototype launch configurations.

Latest tail evidence: genuine pinned-main FP8 with the original serving CLI
also exhibits large C57 mixed tails: cold TTFT p95 is325.875ms and q800 p95
is303.714ms. Main with the current serving CLI also has cold p95342.650ms
and q800 p95206.773ms; both main profiles have6/37 cold requests above300ms.
Both controls completed cleanly and left all GPUs idle.
Strict source/native/all-eight-rank identity and the1197-request
workload pass independent review. This is an instrumented diagnostic, not a
matched unprofiled comparison. Fusion-specific tail regression is unproven.
The user closed further tail-only investigation after both native controls
reproduced large tails. No further kernel/scheduler optimization is required
to demonstrate the recorded speedups. Existing failed sample gates remain
recorded.

Latest completed performance screen: V7 preserves the isolated gains and
improves historical BF16 warm TTFT/decode in all six recorded serving cases,
but C57 mixed cold p95 is314.497ms versus258.344ms historical BF16 and
229.823ms historical native-main FP8. V7 q800 isolated median is36.340ms,
52.28% below historical native-main FP8. Prototype V7 quality now passes
1278/1258; integrated-source and fresh restart validation remain outstanding.
Independent review
recomputes all4188 measured serving rows and1600 measured isolated requests:
`/jobtmp/gemma4-staggered/g4-carry-v1-independent-review-v1.md`.
Serving peak falls9.98→9.81GiB versus V6, while graph memory remains4.96GiB;
KV capacity rises455840→457274 tokens. The672MiB tensor saving is not a
672MiB serving-memory recovery. The C57 mixed q800 p95 improves versus V6,
but cold p95 regresses37.85%; do not infer a general tail fix.

Earlier completed quality diagnostic: corrected CUDA V6 passes the paired quality gate,
1276→1277 established and 1256→1257 strict out of1319, with zero truncations.
Both servers exit cleanly; the final reader verifies sources, request plans,
all-rank routing, raw metrics and final GPU idle. Decode medians improve
2.4–4.6% versus the fusion-disabled control. Incremental regressions remain:
q800 median +1.02%, C57 warm-only median +0.64%, C57 mixed warm median +2.34%.
Graph memory rises4.34→4.96GiB. This is a passing quality diagnostic with mixed
performance, not full goal acceptance. Final artifact:
`/jobtmp/gemma4-staggered/g4-corrected-v2-root-final-v1.json`.
Independent raw review also identifies an important subgroup regression:
C57 mixed warm q800 TTFT p50 rises 58.141→63.380ms (+9.01%), and p95 rises
113.606→199.519ms (+75.62%), with117 measured requests per arm. The nearly
flat aggregate mixed-warm p95 masks this change. C4 mixed all-request p95 also
rises41.126→43.370ms (+5.45%). Both C57 arms retain two cold requests above
300ms; cold p99 rises318.028→319.876ms. Preserve these counterexamples without
assigning their cause to CUDA or scheduling from a single pair.

Follow-up decomposition of C57 mixed q800 (117 measured warm requests/arm):
client TTFT p95 rises113.606→199.519ms, while the server metric measuring
first-token timestamp minus first-scheduled timestamp is65.628→65.772ms.
Engine queue p95 is0.0138→0.0196ms. Per-request client-minus-server TTFT p95
rises31.906→64.751ms; this residual includes time before engine queueing and
after engine first-token accounting, plus client/transport delays. It is not
an isolated CPU or scheduler measurement. Six→ten q800 requests exceed150ms;
two→six have client-minus-server gaps above100ms. With117 samples, this moves
the p95 across a sparse slow cluster. Warm q80 has similar long outliers
(48→40 above150ms among1043 requests), but its p95 improves86.969→81.171ms.
Thus the observed 75.62% is not evidence of a q800 kernel running75% slower.
Isolated q800 regresses only1.02%; launch specialization remains possible but
does not explain the magnitude without an interaction outside that measurement.
Correction after per-request review: the flat marginal server p95 does not
localize the client p95 regression outside scheduled work. The two candidate
requests bracketing client p95 have client/scheduled/residual intervals of
198.232/171.796/26.436ms and204.666/179.599/25.068ms, predominantly inside the
scheduled interval. Among ten candidate requests above150ms, five have
scheduled intervals above100ms and six have residuals above100ms (one has
both); the control's six slow requests split four/two. Additional residual-tail
events matter, but an outside-only diagnosis is unsupported. Trace both engine
execution and frontend/client paths. Marginal percentiles cannot be subtracted
or treated as if they identify the same requests.
The deterministic workload assigns q800 by logical-index modulo10, so arrival
phase and request shape are correlated. Next causal work must separate these
factors with timing/trace evidence rather than assuming shape-only causality.
Raw-bound decomposition:
`/jobtmp/gemma4-staggered/g4-corrected-v2-c57-tail-decomposition-v1.json`.

Next memory experiment: CUDA V7 removes only final mode5's unused carry store;
the staged adapter returns an empty carry for that mode. The standalone build
passes 216 structured cases and 27 final-mode checks, including M32768 with
an empty/null carry pointer. Final BF16 output is unchanged in these sampled
checks. The actual adapter now passes all 64 eager/compiled/CUDA-graph cases
through M800, including empty final carry outputs. An additional M32768 check
passes eager, compiled and CUDA-graph execution with finite final outputs and
confirms removal of the 704,643,072-byte (672MiB) carry allocation. The remaining
adapter outputs allocate 357,826,560 bytes in that check. Model peak memory,
KV capacity and serving latency remain unmeasured for V7; the allocation saving
does not establish a fix for the C57 tail regression. Evidence:
`/jobtmp/gemma4-staggered/gemma4-carry-elision-op-proof-v1/results-v1/results.json`
and `max-allocation-v1.json` in the same proof directory.
No frozen serving runtime has been modified. Artifacts:
`/jobtmp/gemma4-staggered/gemma4-final-carry-elision-stage-v1/` and
`gemma4-full-chain-cuda-prototype-v7/` under the same directory.

V7 serving screen is frozen in
`/jobtmp/gemma4-staggered/gemma4-carry-screen-v1/prepared-plan-v1.json`.
The isolated runtime `/jobtmp/gemma4-carry-elision-runtime-v1` differs from V6
in exactly the CUDA source, matching shared library and adapter (9499 files
checked). This single-candidate screen retains C4/C13/C57 warm/mixed plans and
q80/q800/q8192/q18304 controls; comparisons reuse the completed V6 results.
Full quality evaluation is omitted for this memory-only performance screen;
quality acceptance is explicitly unassessed, and all final validation gates
remain open. Controller log: `/jobtmp/gemma4-staggered/g4-carry-v1-launch.log`.
The measured adapter allocation delta above is live output allocation, not
model peak memory or CUDA-graph pool size.
The read-only comparator `/jobtmp/gemma4-staggered/read_g4_carry_screen_v1.py`
checks completed request identities and timing data before calculating changes
against retained V6 candidate/control and historical native-main results.
Its preflight records all missing cases explicitly; pending results are not
performance evidence. A replay against completed V6 raw requests reproduces
all four isolated and six serving cases, zero deltas versus V6 itself, and the
C57 mixed q800 p95 regression of75.62% versus the fusion-disabled control:
`/jobtmp/gemma4-staggered/g4-carry-v1-reader-v6-replay-v1.json`.
This replay is a reader check, not V7 performance evidence.
Independent CPU review found no V7 screen source/routing/workload mismatch.
Provenance caveat: copied README/adaptation and archived op/boundary proof files
retain their V6 history. New V7 proofs are identified by `v7_proofs` in the
prerequisites and pinned directly in the frozen plan. Those inherited V6 files
must not be reported as V7 validation or acceptance. All eight V7 workers have
completed compilation; measured results are pending.
Startup KV profiling now reports52.34GiB /457,274 tokens for V7 versus
52.18GiB /455,840 tokens for V6: +1,434 tokens (+0.315%). This is a smaller net
budget improvement than the672MiB tensor-allocation saving; those quantities
measure different things. Initial profiling captures report3.62GiB in both
runs. Full342-graph capture and serving latency remain pending while fresh
DeepGEMM warmup runs. Evidence, including observed log-prefix identities:
`/jobtmp/gemma4-staggered/g4-carry-v1-startup-memory-v1.json`.

Current cumulative progress versus the historical native-main FP8 reference:
isolated q800 median TTFT improves 76.158→36.668ms (**51.85% lower**), and
q8192 improves 192.077→177.057ms (**7.82% lower**).
At q18304 the cumulative reduction is 418.204→394.615ms (**5.64% lower**).
These include the earlier serving changes as well as the current custom kernel.
Warm TTFT medians across warm-only/mixed workloads are approximately 19–20%
lower at C13 and 27–30% lower at C57 than the historical native-main reference.
Even the C57 mixed warm q800 subgroup improves cumulatively: native-main
112.049/233.167ms p50/p95 versus current 63.380/199.519ms, reductions of
43.44%/14.43%. Its incremental regression versus the optimized control above
still requires investigation. Historical C4/C13 serving windows differ from the
current windows; these comparisons do not prove matched-restart acceptance.
Against the fresh
fusion-disabled control, the custom kernel alone changes those medians by
+1.02%, -0.47% and -0.44% respectively. Both views matter: cumulative improvements
must not be attributed entirely to the kernel. The historical main controls
have 12 measured requests per query versus 400 currently; these are progress
indicators, not fresh matched-restart acceptance. Both use an 8192-token cached
prefix. Raw reference and current result hashes are recorded in
`/jobtmp/gemma4-staggered/g4-corrected-v2-isolated-pair-check-v1.json`.

Token-count terminology: isolated q80/q800/q8192/q18304 denotes newly computed
prefill tokens after an 8192-token cached prefix. Total prompt lengths are
8272/8992/16384/26496 respectively; q80 is not an 80-token context window.
The server's configured maximum model length is 65536.

Acceptance clarification from the user: bitwise equivalence is not required.
Byte/scale comparisons below are diagnostic tools for identifying arithmetic
changes, not acceptance gates. Remaining byte differences alone do not block
timing, integration experiments or model evaluation. Acceptance rests on
numerical stability, model quality and the unchanged serving performance goals.
Historical failed quality/performance results remain failures of their stated
gates; numerical differences have not been proven to cause those failures.

Earlier result (2026-10-28): the original full-chain CUDA candidate was **rejected under
the unchanged quality gate**. Both arms completed cleanly, but full1319 scores
fell from 1276 to 1274 established and 1257 to 1254 strict, with zero truncations.
Independent raw rescoring agrees. Decode improves about 2.35–4.96%, but isolated
q800 median increases 0.34%, and C57 mixed warm median increases 3.52%.
This is a diagnostic FP8 off/on pair, not BF16 or three-restart acceptance.
Evidence under `/jobtmp/gemma4-staggered/`:
`gemma4-full-chain-modelpair-v1-root-final-v1.json` and
`gemma4-full-chain-modelpair-v1-independent-review-v1.{json,md}`.

The subsequent source audit corrects an important earlier interpretation of
the microbenchmark below. A cached autotuning body is not necessarily the
selected serving implementation. The serialized compile-range mapping and
actual wrapper AST show that 59 interior large-range Stage-B boundaries use
BF16-staged numerators with unrounded FP32 variance statistics. The earlier
`c6xus` FP32-staging microbenchmark does not reproduce those boundaries.
The small range (1–48 rows) additionally uses 120 FlashInfer fused
all-reduce + RMSNorm calls, absent from our full-chain path. Thus the candidate
changes both fusion and arithmetic; uniform FP32 carry is not equivalent to
the selected baseline. These differences are established source facts, not
proof of which difference caused individual quality losses.

The next investigation at that stage was a numerical and performance comparison against those exact
selected implementations, separately for initial, interior and final layers
and for both compile ranges. Correctness precedes further serving runs.
Then hold arithmetic and operation boundaries fixed while sweeping CUDA
elements per thread and Triton reduction tiles at actual padded row counts.
Report kernel-only and complete producer-chain costs separately; include the
collective wrapper in TP8 comparisons. This separates intrinsic kernel cost,
shape specialization and lost fusion instead of attributing all three to CUDA.
Retain all previous failures; do not rerun unchanged arithmetic for a green
quality result. The goal remains open.

The selected interior Stage-B numerical ablation now completes all 18
shape/tile cases (M64/80/128/136/800/8320; three reduction tiles). Preserving
FP32 statistics and rounding only the numerator exactly reproduces the selected
Triton normalized outputs, FP8 bytes and scales on these synthetic inputs.
Rounding both statistics and numerator does not. This is a controlled boundary
ablation, not a model-quality explanation. Evidence:
`/jobtmp/gemma4-staggered/gemma4-selected-stageb-rounding-v1/`.

An isolated CUDA V5 implements that interior rounding boundary, leaving final
mode5 unchanged. Its selected-reference comparison also completes 18 cells,
with three CUDA schedules each. At Triton1024/8 versus CUDAE16, differing FP8
bytes fall from 8071 to 1 at M80, 78369 to 6 at M800, and 816742 to 95 at M8320;
scale differences become zero in these comparisons. BF16-rounded carry still
differs at 15/123/1501 elements respectively. This is not bitwise equivalence.
The small-M BF16-output interior path still requires a distinct mode from final
norm; V5 is not ready for serving integration. No new timing or model-quality
result is claimed. Native source/build and comparisons are retained in
`gemma4-full-chain-cuda-prototype-v5/` and `gemma4-selected-stageb-cuda-v1/`
under the same experiment directory. Old runtimes and failures remain intact.

Selected-chain timing now completes ten shapes with both implementations tuned,
forward/reverse order, CUDA graphs and cold-L2 CUPTI timing. Against the best
tested Triton tile, V5 CUDA reduces A+B producer time by 16.7–21.4% at M64–128,
1.6% at M136, 0% at M176, 2.1–4.9% at M800–864 and 6.7% at M8320.
Both arms include an identical input-reset copy required to replay the selected
baseline's in-place StageB; these are not copy-free production timings. Actual
attention, GEMMs and collectives are excluded. This is narrower than the earlier
FP32-staging comparison and does not prove serving improvement. Evidence:
`/jobtmp/gemma4-staggered/gemma4-selected-chain-timing-v1/`.

CUDA V6 adds a separate BF16-output interior mode6 so small-M raw quantization
can use the correction without changing final mode5. Its 216 structured kernel
cases complete successfully, checking finite/bounded arithmetic, memory guards,
padding and input immutability. Paired-output quantization consistency is an
internal contract check, not an Inductor bitwise-equivalence requirement.
No serving adapter/runtime has been updated yet; raw FlashInfer consumption,
model quality and serving performance remain to be tested for this candidate.
Evidence: `gemma4-full-chain-cuda-prototype-v6/` under the experiment directory.

The corrected adapter now passes 64 eager/compiled/CUDA-graph cases, including
12 actual FlashInfer raw-quant checks. Another 16 cases pass against the adapter
relocated into `/jobtmp/gemma4-corrected-chain-runtime-v1`. Its 9499-file source
manifest is verified; exactly three files differ from the prior full-chain
runtime: CUDA source, native binary, and small-interior mode selection. Model
code and the pinned main snapshot remain unchanged. These are execution-path
checks, not baseline bitwise or model-quality acceptance. Artifacts under
`/jobtmp/gemma4-staggered/`: `gemma4-corrected-chain-op-proof-v1/`,
`gemma4-corrected-chain-runtime-op-boundary-proof-v1/`, and
`gemma4-corrected-chain-runtime-setup-v1.json`. The next model-pair package is
prepared with the original workloads and gates. The corrected comparison is
initially launched from sealed package `gemma4-corrected-chain-modelpair-v1/`, plan
SHA256 `57a2e194a1620189227ea6bbe96de3c5bf409b7dd903596ca2af0274729be220`.
Root verified unchanged workload plans, all 60 inherited point gates, isolated
controls, quality gate, execution functions and passive workers. Initial GPU
idle state was verified. That startup was stopped before measurement after
`AF_UNIX path too long` disabled baseline FlashInfer collective/norm fusion.
The controller terminated with Signal15 and verified final GPU idle; the
attempt and frozen inputs remain preserved. This is an invalid startup, not a
quality/performance result.

The replacement uses package `gemma4-corrected-chain-modelpair-v2/`, short tag
`g4-corrected-v2`, and plan SHA256
`90404bd6dd9fcd51d3679ce9204bf78df2a727990742f724107797061c0f0f67`.
Its private TMPDIR is 20 characters shorter. Root verified unchanged runtime,
weights, protected inputs, workloads and gates; only experiment paths changed.
The replacement is live. Its baseline initializes the MNNVL workspace, and an
independent selected-wrapper census confirms 120 fused all-reduce/norm sites
in the small compile range on every rank, zero in the large range, and no
candidate full-chain calls. Evidence:
`/jobtmp/gemma4-staggered/g4-corrected-v2-off-fusion-check-v1.json`.
This verifies compiled source selection, not measured launch counts. No measured
candidate result or quality conclusion exists yet. The final reader is prepared at
`read_gemma4_corrected_chain_modelpair_v2.py` under the experiment directory.

The replacement off arm has now completed startup: graph capture took 104s
and 4.34GiB. Root verified all eight ranks' actual eligibility, source hashes,
saved FX and disabled candidate guard. Its first isolated control passes
canonical validation (10 warmup +400 measured, query80, prefix8192 cached):
TTFT p50/p95/p99 is 23.841/24.329/24.676ms. This is baseline-only evidence,
not an optimization result. Source and raw-data pins:
`/jobtmp/gemma4-staggered/g4-corrected-v2-off-q80-check-v1.json`.
The q800 and q8192 controls also pass canonical validation with 400 measured
requests each: p50/p95 is 36.297/37.128ms and 177.890/179.978ms respectively.
The still-growing control file was copied to an immutable snapshot for this
partial check; evidence is `g4-corrected-v2-off-q800-q8192-check-v1.json` and its
pinned snapshot under the same directory. No candidate comparison is implied.
All isolated controls are now complete and canonically validated. The q18304
baseline measures p50/p95 396.363/399.536ms over 400 requests, with q800/q8192
unchanged from the partial snapshot. Evidence:
`g4-corrected-v2-off-long-controls-v1.json`. Concurrent serving and full quality
remain pending for this arm, followed by the candidate arm.
Both C4 cases now pass full canonical validation: 416 measured requests each,
with 14 cold requests in mixed. Warm-only TTFT p50/p95 is 26.434/33.181ms;
mixed warm is 26.644/38.589ms and mixed cold is 156.739/159.842ms. Median TPOT
is 6.239ms for warm-only and 6.290ms across mixed requests. These are off-arm
baseline values, not a candidate gain. Evidence:
`g4-corrected-v2-off-c4-check-v1.json`.
C13 also passes canonical validation: 481 measured requests per case and
17 mixed cold requests. Warm-only TTFT p50/p95 is 28.612/39.412ms; mixed warm
is 29.150/46.703ms, mixed cold 159.722/173.057ms. Warm-only median TPOT is
7.492ms. Evidence: `g4-corrected-v2-off-c13-check-v1.json`. C57 and quality
remain outstanding before the candidate comparison.
C57 is now complete and canonically validated: 1197 measured requests per
case, including 37 mixed cold requests. Warm-only TTFT p50/p95 is
44.151/56.617ms; mixed warm is 47.793/88.354ms and mixed cold is
183.967/231.167ms. The latter differs from the prior off-arm cold p95 of
309.011ms; retain both launches, without pooling or treating either as stable
tail performance. All six baseline serving cases are complete; full1319
quality is collecting. Evidence: `g4-corrected-v2-off-c57-check-v1.json`.
The fusion-disabled FP8 control now finishes cleanly (server exit0). Canonical
full1319 rescoring validates 1276 established /1256 strict, all stop finishes,
zero truncations. The established score matches the prior off arm; strict is
one lower. No causal attribution to compiler tiling is established. Evidence:
`g4-corrected-v2-off-quality-check-v1.json`. The paired harness is proceeding
to the corrected CUDA arm; there is no candidate quality verdict yet.
The corrected arm now passes actual eligibility/source checks on all eight
ranks; saved FX contains 120 full-chain calls plus one initial norm-quant call.
Graph capture takes 106s and 4.96GiB versus control104s/4.34GiB, preserving the
memory tradeoff. The first matched q80 control validates identical canonical
requests (400 measured): median TTFT 23.841→23.390ms (-1.89%), p95
24.329→24.175ms (-0.63%), p99 24.676→24.868ms (+0.78%). This is an isolated
single-pair result, not serving/quality acceptance. Evidence:
`g4-corrected-v2-q80-pair-check-v1.json`.
The next paired controls also pass canonical identity/cache validation. At
q800, median/p95 TTFT changes 36.297/37.128→36.668/37.495ms (+1.02/+0.99%);
at q8192 it changes 177.890/179.978→177.057/179.613ms (-0.47/-0.20%). Retain
the q800 regression: the isolated results do not show a broad additional
prefill gain from the custom chain. Evidence and immutable partial candidate
snapshot: `g4-corrected-v2-q800-q8192-pair-check-v1.json`. Concurrent serving,
longest control and candidate quality remain pending.
The corrected C4 pair now validates against the frozen request plans. The
kernel's incremental warm TTFT median changes -3.77% in both warm-only and
mixed cases; p95 changes -1.91% and -0.29%. All-request median TPOT improves
4.65%/4.63%; mixed cold p95 rises 0.39%. Cumulative warm median TTFT reductions
versus historical native-main FP8 are 22.83%/22.44%. That main reference used
24 turns (84 measured per case) versus 107 turns (416 measured) now, so the
historical windows/plans are not matched. Preserve that limitation separately
from the fully matched current off/on pair. Evidence:
`g4-corrected-v2-c4-pair-check-v1.json`.
Both C13 cases are validated: incremental warm TTFT median improves
2.51%/2.28%, p95 improves 0.82%/2.11%, and median TPOT improves 2.70%/2.41%.
Mixed cold p95 improves 2.75%. Historical native-main warm median reductions
are 19.71%/19.04%, with the shorter historical C13 window limitation retained.
Both C57 cases also validate. Incremental warm TTFT median rises 0.64%/2.34%,
while p95 improves 0.77%/0.04%; median TPOT improves 3.18%/2.64%. Mixed cold
p95 improves 1.31%. Historical native-main warm median reductions are
26.85%/30.32%. These C57 medians retain a mixed incremental result rather than
a universal kernel win. All serving cases are complete; candidate quality
remains pending. Evidence: `g4-corrected-v2-c13-pair-check-v1.json` and
`g4-corrected-v2-c57-pair-check-v1.json`.

A source-only comparison of the prior and replacement off arms now covers all
eight ranks, 976 selected range/submodule records and 1936 H5376 kernel sites.
Selected arithmetic ASTs, signatures, store types and range mappings are
unchanged. Final serialized tuner bindings differ at 502 static sites,
including initial RMS tiles on four ranks; these are saved restoration
configurations, not observed GPU launch selections. No quality difference is
causally attributed to them. This narrows the cross-launch investigation without
introducing bitwise acceptance. Pinned report:
`/jobtmp/gemma4-staggered/g4-off-selected-crosslaunch-audit-v2.md`.
The passive-process follow-up binds all 11616 site/snapshot observations to
selected wrapper attributes using exact module aliases, arithmetic ASTs and
pointer signatures. Configurations stay stable across before-controls,
after-controls and after-serving snapshots within each launch. Live configured
objects differ at 602 static sites between launches, rather than the 502
saved-restoration differences above; some small-interior-B live configurations
disagree with the saved per-artifact setting. These counts describe different
evidence layers. Configured reachability still does not prove execution for a
particular measured request or quality causality. Supplemental report:
`/jobtmp/gemma4-staggered/g4-off-passive-selected-bindings-v1.md`.

Terminology clarification: the current pair's "baseline" is the
**fusion-disabled FP8 control**, with the same earlier serving fixes and graph
configuration as the candidate. It is not untouched main, native BF16, or the
original native-FP8 reference. Its fresh quality run is intended to investigate
the previous fusion candidate's quality failure amid observed cross-launch
variation. Repeating a control evaluation is not inherently required for every
screening iteration; historical validated controls can support screening, while
the current paired diagnostic and final validation retain their stated gates.
Untouched-main/BF16 references and the three-restart campaign remain outstanding.

The entries below retain the chronological experiment history and are
superseded by the current result and source correction above where noted.

Earlier CUDA fusion diagnostic (2026-10-28): the isolated FP8 off/on pair
`gemma4-hybrid-modelpair-v3` completed both arms cleanly. The independent reader
and raw-data review validate matching workloads, source identities, final GPU
idle state, and unchanged runtime. Full1319 quality improves from1276/1256 to
1279/1259 established/strict, with three paired wins, no losses and no truncations.
Isolated TTFT improves only about0.5–1.1%. This is **not a selected serving win**:
C13 mixed warm p95 rises44.605→46.615 ms (+4.51%), and C57 mixed cold p95
rises210.659→314.281 ms (+49.19%,37 cold requests per arm), despite a better
cold median184.509→177.803 ms. This single ordered FP8 pair does not qualify
BF16 comparisons or the three-restart goal. Evidence:
`/jobtmp/gemma4-staggered/gemma4-hybrid-modelpair-v3-root-final-v1.json` and
`gemma4-hybrid-modelpair-v3-independent-review-v2.{json,md}`.

The next CUDA experiment follows the user's request to preserve complete
Inductor fusion boundaries. Actual generated code fuses post-attention RMSNorm,
residual addition and pre-FF RMSNorm; another boundary includes post-FF norm,
residual reconstruction/addition, layer scalar and next-layer input norm.
The small custom RMSNorm+quant producer splits these existing chains. The
standalone17–27% producer improvement therefore does not imply the same model
improvement. This topology is demonstrated by generated bodies, but does not
prove the cause of the cold-tail regression.

Priority is now a full-chain CUDA prototype, checking live residual outputs,
FP32 intermediates, cast elimination and FMA/reduction differences before timing.
Stage A will fuse post-attention norm + residual add + pre-FF norm + group128
FP8 quantization. Stage B must preserve the downstream post-FF/residual/scalar/
next-input-norm chain, including cross-layer liveness. A deliberate FP32 residual
output is a proposed new interface, **not** a claim that the baseline stores a
persistent FP32 residual. Merely covering the same operators cannot guarantee
performance or bitwise equivalence. Full-chain timing and serving validation
remain required, and all previous counterexamples and acceptance gates remain.

The Stage-A feasibility probe has now run (both processes exit0). Scalar
loads/stores in prototypeV1 lose at all four shapes. The vectorized V2 coarse
8/16/32-elements-per-thread sweep gives approximately27%,8%,20% lower latency
at80/800/8320 rows against a fresh compiled full-chain reference plus CUDA
quantization. Both sides expose the proposed FP32 residual output; this is not
an actual captured serving-chain comparison. All observed scales match; Q bytes
match at32/80, with about1–2 differing bytes per million at800/8320. Guards and
input immutability pass, but full correctness/model acceptance remains open.
Stage B and cross-layer integration are not implemented. Raw results and the
negative V1 are retained under `/jobtmp/gemma4-staggered/` in
`gemma4-full-chain-cuda-prototype-v{1,2}`. See V2 `README.md` for boundaries,
schedules, timing method and limitations. These are preliminary kernel gains.

Full-chain follow-up: V3 adds FP32 residual-input and post-FF scalar modes;
all15 isolated timing cells complete. StageB's best sampled schedule takes
4.912/23.216/190.207 us at80/800/8320 rows versus7.584/26.336/246.671 us for
its fresh explicit-FP32 compiled reference plus quantization. This comparison
still is not the actual serving graph. The V3 baseline includes multiplication
by1 in nonscalar modes, so it must not be silently conflated with V2.

V4 adds BF16-output specializations needed for final norm and small-M raw-quant
integration. Its216 structured paired-output cases pass, including padding,
input/scalar immutability, zero/tiny inputs and scalar edge cases. FP8 bytes/
scales exactly match the existing CUDA quantizer of the paired BF16 output;
this does not prove bitwise equality to baseline model arithmetic. The maximum
BF16 norm difference against the explicit eager reference is0.03125. A separate
16-case synthetic two-layer composition (64 stage checks) passes using actual
FP32 carries and a BF16 final norm. Actual attention/MLP and the small-M raw
FlashInfer branch are not covered. All processes exit0. Source design and
review: `/jobtmp/gemma4-staggered/gemma4-full-chain-cuda-design-v1.md`;
experiments: `gemma4-full-chain-cuda-prototype-v{3,4}` under the same directory.
The next step remains guarded model integration and actual serving validation,
not claiming acceptance from these isolated tests.

Specialization audit requested by the user: a captured-body probe replays
cached `c2hyh` + `c6xus` arithmetic and both CUDA quantizers. The later selected
wrapper audit above shows this does not reproduce the 59 interior Stage-B
serving boundaries; its timings apply only to the tested arithmetic. It compares
the recorded Triton1024/8 configuration, alternatives2048/16 and8192/16, and
CUDA8/16/32-elements-per-thread schedules on11 row counts including96/136/176/
816/864 from padded serving dispatch. Both chains' carry dependencies are kept;
external attention/MLP outputs are synthetic. The completed probe exits0.
CUDA gains versus the **best tested** Triton configuration are24.7–29.5% at
32/64/80/96/128 rows,6.0% at136,2.7% at176,6.3–7.1% at800/816/864, and13.3%
at8320. The captured recorded config is not always the best small-shape choice.
Our adapter's M128 E16 choice is also suboptimal: about11.95 us versus10.592 us
E8. A separate unapplied adapter stage changes the experimental boundary to
E8 through128. This does not attribute serving-tail regressions to specialization.

Source and results: `/jobtmp/gemma4-staggered/gemma4-captured-chain-shape-probe-v1/`.
Only the captured FP32-staging family is covered; the BF16-staging family remains
outstanding. Numerical comparison uses recordedTriton1024/8 versusCUDAE16 and
retains StageB scale differences atM80 andM8320. Other timed configurations need
separate numerical checks. No serving/quality acceptance follows from this probe.

The custom-op proof also finishes48 cases across eager/compile/CUDA graphs,
including12 actual FlashInfer raw-scale reference checks atM1/31. Zero-input
raw scales equal1. Exact output/carry checks, input immutability and fake/output
scale strides pass. This verifies plumbing, not model equivalence. Evidence:
`gemma4-full-chain-op-proof-v1/results-v1/results.json`; all processes exit0.

Full-chain model integration is now staged in a separate runtime,
`/jobtmp/gemma4-full-chain-runtime-v1`. The9496-file copied source tree is
verified against the prior runtime; exactlyfour files differ: Gemma model,
custom-op adapter, standalone CUDA source and its shared library. Independent
Git metadata retains baseHEAD274a5e19. The prior measured runtime is untouched.
The standalone C-ABI library is experimental; production CMake integration and
reviewable native-op cleanup are still required if the candidate is selected.

The guarded model loop consumes StageB's already-normalized activation at the
next attention, and returns the final BF16 norm once. Whole-model eligibility
checks all60layers after actual provider selection, rejecting PP/aux/PLE/MoE/
KV-sharing/speculation/LoRA and incompatible norm/linear contracts. Source/AST
and CPU fixture checks pass12 rejection cases and120 handoffs. Actual runtime
import passes; an earlier attempt failed an import-path assertion because cwd
was the user workspace, corrected without code changes. The revised runtime
adapter passes16 eager/compile/CUDA-graph cases atM128/136. Runtime setup/import
and boundary proof artifacts are under `/jobtmp/gemma4-staggered/` with the
`gemma4-full-chain-runtime-*` prefix. These do not establish full-model tracing
or serving success. The new matched controller is being prepared to require
positive whole-model eligibility and actualFX120full-chain+1initial normquant
calls before timing; original C4/C13/C57, full1319quality and all gates remain.

The full-chain paired TP8 run has now launched from the sealed package
`/jobtmp/gemma4-staggered/gemma4-full-chain-modelpair-v1/`, planSHA
`da5600bcc6cbdd6c45ed5aa6047e67f7463c9d288da3010fa5eb0efa8d3a69f6`.
It compares FP8 flag0 then1 in the same new runtime, preserving original server
settings, every measured workload plan/turn count, full1319quality and all60
point gates. Root independent static plan comparison passes. Before timing,
the passive worker requires actual model eligibility and savedFX0/0 calls for
off versus120full-chain/1initial normquant for on. Startup/full-model tracing
and serving outcomes are pending; no result or acceptance is claimed yet.

A separate numerical sweep now characterizes all66 shape/schedule combinations
from the captured-body comparison (`gemma4-captured-chain-numerics-v1`), without
rerunning timings. All complete. Retiling Triton itself changes some Q bytes
(up to60 across both outputs at8320 rows) relative to its recorded config.
CUDA's M80 case changes one scale group; E16/E32 also change one at8320, while
E8 does not there. These scale discontinuities and all per-variant differences
remain explicit. Numerical sanity is not a model-quality acceptance test.

While the full-chain pair runs, the independent offline reader is prepared at
`/jobtmp/gemma4-staggered/read_gemma4_full_chain_modelpair_v1.py` with a separate
CPU proof. It retains exact workload identities, canonical isolated requests,
quality scoring and zero-truncation checks, and additionally reparses the saved
FX source to validate the new operator counts. Full-pair execution is pending.
The live baseline has loaded all8ranks and completed initial compilation; fresh
DeepGEMM cache warmup is still in progress. No latency results are available yet.

A source-only resource audit (`gemma4-full-chain-register-analysis-v1.{json,md}`)
records all18 compiled variants. Quantized E8 uses54 registers/thread at672
threads, permitting at most one CTA/SM by the recorded65,536-register budget.
E16 uses78×352, a register-only upper bound of two. This supports investigating
wave/occupancy effects near132 rows but is not a runtime occupancy measurement
or proof of the observed128→136 timing change. BF16-output E8 uses47 registers,
so its bound differs. The live kernel and dispatch remain frozen.

The full-chain pair's off arm has reached measurement. Graph capture completes
in103 seconds at4.34 GiB/rank. Root independently reparses all8 saved baseline
FX graphs and confirms whole-path guardFalse and0full-chain/0hybrid normquant
calls. The400 measured q80 requests validate their canonical IDs, prompt hashes,
cache counts and timestamps: p50/p95 TTFT23.500/24.035 ms. This is a baseline-only
checkpoint, not evidence of candidate improvement. Evidence:
`gemma4-full-chain-modelpair-off-fx-check-v1.json` and
`gemma4-full-chain-modelpair-off-q80-check-v1.json` under the experiment root.
The same controller remains live; the candidate arm has not been evaluated.

The full-chain pair's off-arm isolated controls are now all complete and
independently validated (400 measured requests each): q800 p50/p95
36.272/37.031 ms; q8192 177.874/179.600 ms; q18304
396.480/399.096 ms. Canonical prompt hashes, IDs, cache/computed-token counts
and timestamps pass. The candidate and serving comparison remain pending.
Evidence: `gemma4-full-chain-modelpair-off-long-controls-v1.json`.

All six off-arm serving cases in the full-chain pair are complete; root
independent checks validate C4/C13/C57 request plans and counts. C57 mixed
warm TTFT p50/p95 is48.251/116.652 ms; cold is178.399/309.011 ms (37 cold).
The full-chain kernel is disabled and independently confirmed absent from all
8FXgraphs. Thus a high cold tail occurs without this new fusion on the existing
S3 profile. This does not exonerate the earlier combined S3 changes versus
native main, nor establish a candidate benefit. The previous simple-producer
pair's off210.659/on314.281 ms cold-p95 counterexample stays intact. Reversed
restart validation remains necessary; no tail failure is waived. Full1319
baseline quality is now collecting, with the candidate arm still pending.
Artifacts: `gemma4-full-chain-modelpair-off-c{4,13,57}-check-v1.json`.

The full-chain pair's off arm finishes cleanly (exit0). Canonical full1319
quality is1276 established/1257 strict, with0failures/0truncations and allstop
finish reasons. Root recomputation matches the controller's archived identity
and scores (`gemma4-full-chain-modelpair-off-quality-check-v1.json`). The one
strict-score difference versus the earlier flag0 baseline is retained for
cross-launch analysis. The same paired controller is proceeding to flag1;
no candidate timing/quality result is claimed yet.

The full-chain on arm passes actual whole-model eligibility on all8ranks;
root independently verifies savedFX120full-chain +1initial hybrid normquant
calls after startup. Its first400 measured q80 requests validate at23.294/23.929
ms p50/p95 versus off23.500/24.035 ms. These small isolated differences do not
establish a serving win or model quality. The controller remains live.

A material memory tradeoff is measured: on graph capture4.96 GiB/rank versus
off4.34; KV capacity455,840 versus473,391 tokens (52.18 versus54.19 GiB).
Reported peak activation is9.98 versus7.97 GiB, **including** estimated graph
memory7.22 versus5.88 GiB; these quantities must not be added independently.
Warm requests share one8192-token prefix, so57×8192 is not their live-prefix
footprint and does not prove cache pressure. No capacity-caused slowdown is
established. A source-only audit is checking unnecessary final FP32 carry
allocation and later-stage carry lifetimes; actual model effects require a
separate controlled variant after this frozen pair. Evidence is in both server
logs and `gemma4-full-chain-modelpair-on-fx-check-v1.json`.

The complete isolated full-chain pair is independently validated. On versus
off median TTFT changes are q80−0.88%, q800+0.34% (slower), q8192−0.51%,
q18304−0.46%; the prefill result is small despite kernel-level gains. The
matched C4/C13 cases also validate: warm TTFT p50 changes−3.32/−3.79%
(C4 warm/mixed) and−3.38/−3.02% (C13); warm p95 changes−0.31/−1.36% and
−4.13/+0.25%. All-request decode p50 improves4.96/4.91% atC4 and2.81/2.35%
atC13. Per-query tails remain in raw summaries, including worse mixed q800 p95;
no broad all-metrics win is asserted. C57 and full quality are still pending.
Artifacts: `gemma4-full-chain-modelpair-isolated-pair-check-v1.json` and
`gemma4-full-chain-modelpair-c4-c13-pair-check-v1.json`.

The completed baseline cross-launch audit verifies identical1319 prompts,
payloads/golds and all8FX ASTs plus232 observed kernel source bodies.278 answers
change, with1 established gain/1loss; the netstrict+1 additionally comes from
format-only example473 (trailingpercent removed).14 reduction-block selections
differ and remain stable within each launch, but are not proven to cause the
specific answers. Evidence: `full-chain-off-crosslaunch-quality-audit-v1.{json,md}`.
The memory audit prioritizes removing final mode5's unused672MiB carry atM32768
before introducing mutable carry operations. No live inputs are changed;
`gemma4-full-chain-memory-audit-v1.md` records the accounting and compiler
functionalization caveats for a later controlled ablation.

C57 full-chain serving is now complete and independently matched. Warm-only
TTFT p50/p95 improves1.10/1.41% and decode p50 improves3.29%. In the mixed case,
**warm TTFT p50 worsens3.52%** (48.251→49.952 ms), while warm p95 falls26.70%
(116.652→85.505 ms) and decode p50 improves2.40%. Cold p95 is309.011→192.897 ms,
but prior restart variability precludes a causal/final tail-improvement claim.
This is not an unqualified win, and no fresh BF16/native or three-restart
acceptance has been established. Candidate full1319quality is still running.
Evidence: `gemma4-full-chain-modelpair-c57-pair-check-v1.json`.

An additional source audit finds actual
`flashinfer_trtllm_fused_allreduce_norm` calls in baseline generated code and
none in the candidate search so far. Exact selected wrapper/range mapping is
in progress; the recorded local c2hyh/c6xus two-chain microbenchmark does not
cover every compiler fusion. Treat this as a concrete missing comparison, not
proof that this fusion loss causes a particular latency regression. The live
run and candidate remain frozen while that audit completes.

Earlier paired checkpoint: the threshold0/FULL32768/maxseq64 pilot completes
BF16 then FP8 with the same three runtime patches. All six C4/C13/C57 warm/mixed
TTFT p50/p95 cells meet the within3% target against equally configured BF16.
C57 mixed is46.32/88.52 ms FP8 versus46.65/200.74 ms BF16. Isolated FP8 q800 is
36.18 ms median, versus native FP8's76.16 ms. Full1319 quality is1277/1259
established/strict FP8 and1278/1257 BF16, with zero failures/truncations.

The goal is still unmet: **C57 mixed cold FP8 p95 is309.95 ms versus229.82 ms
native FP8, a34.9% regression**, despite its improved178.60 ms median. Four
cold requests above300 ms form two overlapping pairs with equal native TTFT
durations within each pair. Shared batch membership is a hypothesis until
current worker/admission tracing proves it. The earlier same-config FP8 screen
had cold p95215.45 ms; both outcomes are retained. The three-restart campaign
has not launched. A separate numerical-attribution diagnostic completed with
exact restoration. The two-arm scheduling diagnostic completes with zero quota
deferrals in either arm; no serialization benefit is established. Collective
screens reject the tested cutoff/protocol/backend changes. The existing2048
cap on this exact FULL32768/maxseq64 profile also regresses warm/cold latency
and is rejected. The larger-sample diagnostic's first attempt stops during
long-case setup; its completed short case and failure are retained, and a
private preparation-order correction passes CPU transport checks. A fresh
six-cohort diagnostic now has three clean matched restart pairs. The first passes;
the second fails the cold-p95 guard in both sample sizes. Its long-case optimized
p95 is309.10 ms versus273.91 ms native, a12.85% regression. The original short
case is306.82 versus252.62 ms, a21.45% regression. The third pair passes;
the full BF16/FP8 acceptance matrix has not started. The cold regression cannot
be dismissed as a small-sample artifact, and the old failure remains.

Final independent diagnostic checkpoint (`cold-repeat-v2-independent-final-v1.json`):

| Restart | Cold samples per arm | Optimized p50/p95, ms | Native p50/p95, ms | Cold guard |
| --- | ---: | ---: | ---: | --- |
| 1 | 37 | 175.64 / 227.25 | 189.64 / 263.57 | Pass |
| 1 | 546 | 181.05 / 309.22 | 200.24 / 327.36 | Pass |
| 2, reversed order | 37 | 183.73 / 306.82 | 201.18 / 252.62 | Fail p95 |
| 2, reversed order | 546 | 182.20 / 309.10 | 200.77 / 273.91 | Fail p95 |
| 3 | 37 | 178.17 / 201.85 | 196.70 / 275.69 | Pass |
| 3 | 546 | 182.87 / 210.41 | 201.42 / 337.64 | Pass |

These are separate cases and restarts, without pooling. All three pairs retain warm
TTFT and decode improvements against native FP8. This diagnostic has no BF16
arm or full quality evaluation and cannot establish acceptance. All six cohorts
exit cleanly; independent source restoration and final idle checks pass. The
controller and independent reader both finish with diagnostic completion only.

A provisional review bundle is exported separately at
`/jobtmp/gemma4-staggered/review-candidate-bundle-v1/`. Its seven-file patch
contains the three measured runtime changes and their tests/design documentation,
with exact baseline/candidate snapshots and a manifest. AST parsing and read-only
patch-application checks pass; this is packaging evidence, not goal acceptance.
The TMA alignment hunk overlaps open upstream PR49928 exactly, as recorded below.

The CPU-only transport proposal's five-case fixture also passes
(`staggered-client-thread-plan-v1/transport-results-v1.json`). Planning in a worker
thread preserves event-loop progress for fresh and pooled connections across a
controlled6.25-second setup delay; original and preparation-order-only clients
retain their expected failure cases. There are no model/GPU requests or measured
retries in this proof. The proposed client remains unapplied to frozen runs.

The independent request audit (`cold-repeat-r1r2-tail-analysis-v3.{json,md}`)
validates every measured plan, prompt/cache/computed-token count, output length
and saved percentile in the eight completed cases. Each optimized long case has
30/546 cold requests above300 ms; all overlap another cold arrival-to-first-token
interval. Native restart2 has27/546 above300 ms. Optimized long-case warm requests
above150 ms rise from868/17466 to882/17466, crossing the5% quantile boundary.
These observations describe tail sensitivity, not its cause or a waiver. The
native counterexample c28/t113 exceeds300 ms without another cold overlap.
Request-level durations cannot distinguish pre-admission waiting, shared batches
and GPU time; those repeats contain no admission/worker/GPU join.

The repeat has released the GPUs. The next experiments are an intermediate4096
per-request chunk threshold against the same optimized threshold0 profile, and
a small fresh-process collective algorithm comparison. Both diagnostics are
complete. **Reject the4096 cap:** it worsens cold and warm TTFT in both phases.
The former must prove actual chunk sequences and preserve cold median, warm
latency and decode; two queued chunks or multiple cold requests can still block.
The latter reuses the frozen full-wrapper collective benchmark and obtains
actual selection logs, rather than inferring algorithms from kernel names.
Neither is a selected speedup. Actual NCCL2.30.7 INFO/TUNING records in that
microbenchmark select NVLS/Simple for the8320/16512/32768-row BF16 payloads;
their PyNCCL latencies remain approximately415/750/1405 microseconds. This
identifies current microbenchmark task selection, not the older model trace's
algorithm. All five processes exit cleanly and the independent final reader
validates raw logs, correctness records, all rank samples, distributed medians,
source/environment identity and ordered bracketing. No forced algorithm is
selected. The completed2048-cap and collective negatives remain in the record.

NCCL PyNCCL full-wrapper latency at8320 rows, microseconds, separate sweeps:

| Fresh process | Sweep0 | Sweep1 |
| --- | ---: | ---: |
| Default before | 415.248 | 415.584 |
| Forced NVLS | 414.624 | 415.104 |
| Forced Tree | 932.960 | 933.280 |
| Forced Ring | 477.984 | 478.336 |
| Default after | 415.136 | 415.536 |

Large defaults already select NVLS/Simple; forcing NVLS has no material gain.
Ring and Tree lose. The64-row default selects Ring/LL, while forcing NVLS also
slows that synthetic small case. Small production routing remains a separate
question because TorchSymm/FlashInfer/custom paths can apply. Per-sweep complete
shape tables and selection evidence are retained in
`nccl-algo-root-final-v1.{json,md}` and the independently generated
`nccl-algo-final-independent-v1.{json,md}`. This is not a serving/model-quality
result or a new reconstruction of historical NCCL selection.

The threshold0-versus4096 serving pair starts from the reviewed
`threshold0-vs4096-full32768-diagnostic/release-v5.json`, with a single launch
argument difference and unchanged candidate source. Its real builtin
AsyncScheduler CPU proof shows two4096 schedules before either output completes,
correct partial-prefill completion, unchanged cached80/800/decode, isolated-query
bypass, and possible8192-token dual-cold batches. The short case and isolated
controls retain eight-rank chunk tracing; the separate546-cold case disables
that trace after drain. Both arms finish cleanly with source unchanged and all
GPUs idle. Independent and root readers validate the complete diagnostic.

| Separate phase | Metric | Threshold0 p50/p95, ms | Threshold4096 p50/p95, ms |
| --- | --- | ---: | ---: |
| 37 cold | Cold TTFT | 175.82 / 234.94 | 201.82 / 293.55 |
| Short case | Warm TTFT | 44.71 / 82.57 | 45.51 / 111.26 |
| Short case | Decode TPOT, all requests | 15.39 / 17.69 | 15.53 / 18.12 |
| 546 cold | Cold TTFT | 179.45 / 218.74 | 202.41 / 285.88 |
| Long case | Warm TTFT | 46.49 / 106.62 | 48.70 / 118.83 |
| Long case | Decode TPOT, all requests | 15.16 / 18.22 | 15.32 / 18.62 |

All37 short-case cold requests change from one8192 prefill to exactly
4096+4096, without cap bypasses. Isolated800/8192/18304 p50/p95 remain within
1.6% of the threshold0 control, and decode remains within3%; neither rescues the
cold/warm regressions. The larger case increases cold median12.8% and p9530.7%.
Full evidence is in `threshold-pair-root-final-v1.json` and the independent
`threshold-pair-independent-results-v1.{json,md}`. This is one ordered FP8 pair,
not acceptance or model-quality evidence; the old nativeF0 guard remains a
distinct unmatched reference, and the earlier restart2 failure remains.

Completed threshold0 short-case evidence: cold TTFT p50/p95 is
175.82/234.94 ms and warm is44.71/82.57 ms. All37 measured cold requests have
one8192-token prefill, and **none shares a worker batch with another cold
prefill**. The two above300 ms have distinct CPU timelines:
`c6/t3` reaches rank0's worker after157.15 ms and receives its first token
162.24 ms later; `c10/t5` reaches the worker after12.43 ms but receives its
first token296.74 ms later. The latter's step1127 follows c43/t5's cold
step1126 by15.74 ms in worker CPU entry time. Thus even similar client tails
can place their delay before or after worker entry. These are eight-rank
request/shape joins, not GPU spans or scheduler admission timestamps.
The clean-cohort source/hash/clock evidence and every cold membership are saved
in `threshold-short-t0-batch-tail-clean-v3.json`; earlier provisional outputs
remain unchanged. The4096 short case co-batches four measured cold requests
with another cold prefill, including its two above300 ms. The same analyzer
recovers four co-batched cold requests from the old real2048 trace, providing
an independent check of the membership logic. These CPU/shape joins do not
retroactively establish the old pilot's batch membership or identify GPU spans.

The next native-kernel option has a concrete upstream-reuse draft at
`rms-ue8-upstream62514-prototype-v1/`: the saved62514 single-pass producer files
are reconstructed exactly (all three upstream Git blob hashes match), then the
explicit UE8 interface/arithmetic is ported onto them. Source reversal, geometry,
tiny/zero/scale-boundary, AST/Ruff and patch-application checks pass. The narrow
patch changes no upstream load/reduction schedule; no provider/model wiring is
enabled. It remains unapplied and unbuilt, with no GPU correctness, speed or
quality claim. The prior three-read and competing compiler-pattern drafts remain
held; downstream integration should reuse the existing upstream consumer contract.

The source-only sequence-parallel reuse check also found no near-term flag:
generic reduce-scatter/all-gather helpers exist, but denseGemma needs its local
BF16 residual stream preserved, an FP8-input consumer, and explicit rank/group
scale reordering into column-major TMA layout. Existing custom all-gather rejects
FP8. Defer that broader model/dataflow change; see
`gemma4-sequence-parallel-reuse-assessment-v1.{md,json}`. The next concrete step is
the isolated native producer build,65 focused kernel cases and legacy raw tests,
followed by complete-boundary timing and consumer/model evaluation if they pass.

The first isolated native build reached CUDA compilation and failed on
upstream62514's `constexpr float qmax`: Torch2.14's Float8-to-float conversion
is not constexpr. The original failed log/status remain intact; protected
serving sources, native libraries and environments match their before hashes,
and cleanup reports all GPUs idle. A separate one-line `constexpr`→`const`
compatibility patch is recorded in `rms-ue8-torch214-compat-v2/` and applied only
inside `/jobtmp/gemma4-rms-ue8-isolated-build-v1/src`. Incremental attempt2
successfully compiles, links and installs the complete library. Its runner then
stops at an incorrect manifest filename assumption: this CMake setup writes
`install_local_manifest.txt`. Attempt3 validates that actual LAB-only install,
new native load path and defaultFalse schema, then test collection finds a
missing auxiliary FlashAttention dependency in the isolated environment. The
existing compatible FA3 library is copied byte-for-byte into the LAB only,
with source/destination hashes recorded.

Attempt4 completes successfully: **all 65 focused GPU cases and all 1160
existing eager/compiled legacy cases pass, with zero failures or skips**. The loaded
native digest is `c6e2d9f2d2d118ac4047240967557c19c8e810a169e48970b1f467e70dd21c1e`.
Every attempt has separate status/log/XML files; prior failures remain unchanged.
The serving environment and libraries match their protected before/after hashes,
and final attempt4 cleanup confirms all GPUs idle. The producer microbenchmark
was bound to these completed results before timing; the tests alone establish
no performance or model-quality result.

The completed producer microbenchmark V3 now records 17,600 CUPTI samples
across eleven shapes, using cold L2 and four alternating AB/BA rounds with
200 samples per arm per round. It compares compiled native RMSNorm followed
by the existing CUDA UE8 group quantizer against the new fused producer,
with BF16 hidden5376/group128 and FP32 column-major TMA scales:

| Rows | Separate producer, us | Fused producer, us | Lower latency |
| ---: | ---: | ---: | ---: |
| 33 | 4.592 | 3.584 | 22.0% |
| 80 | 4.800 | 3.680 | 23.3% |
| 800 | 14.384 | 11.840 | 17.7% |
| 2048 | 34.176 | 26.320 | 23.0% |
| 4096 | 66.056 | 48.160 | 27.1% |
| 8320 | 125.696 | 92.416 | 26.5% |
| 18304 | 266.991 | 197.183 | 26.1% |

These are producer-only diagnostic timings with inherited configurations,
not a measured model speedup or a tuning-optimum claim. All eleven arbitrary
input cases require numerical review: about1.8–2.0% of FP8 bytes and
0.2–0.4% of group scales differ. Changed scales include factors0.5 and2;
the complete histogram and error measurements are retained. The compiled
baseline RMS differs from eager RMS by up to0.0625. Generated Inductor code
appears to remove its intermediate BF16 rounding before weight multiplication;
direct GPU attribution against eager/native CUDA RMS is the next check.
No model-quality pass follows from these error measurements. The M31 UE8
case remains reference-only because production uses raw scales below32 rows.
Raw results are in `rms-ue8-producer-microbench-results-v3/`; independently
recomputed summary is `rms-ue8-producer-analysis-v1.{json,md}`. The benchmark
session exits0 with all cases collected; serving sources/libraries remain
unchanged. Direct prequantized GEMM consumer validation is still pending.

The subsequent untimed rounding diagnostic completes eleven exact-input cases.
It isolates intermediate rounding using two eager formulas with the same
materialized FP32 reciprocal RMS, and separately compares default/emulated
compilation and CUDA native RMS. Fused outputs reproduce their original
microbenchmark hashes in all eleven cases. Eager two-round RMS plus the current
CUDA quantizer matches fused bytes/scales exactly for the eight shapes through
4096 rows. At8192/8320/18304 rows it retains18/19/5 FP8-byte differences,
respectively, with all scales identical. Intermediate BF16 rounding therefore
accounts for nearly all observed FP8 differences and all scale differences in
this diagnostic; remaining reduction differences are retained, not waived.

The new default compile reproduces prior normalization and quantization hashes
in ten shapes. At8192 rows, normalization and FP8-byte hashes differ from the
earlier run despite matching input/weight/fused hashes; scale hashes match.
Attribution there applies directly to the new diagnostic, and does not establish
exact reconstruction of the original baseline. Precision-emulated compilation
is a confounded comparison because it also changes fusion/libdevice behavior;
it is not used alone to prove the rounding cause. No precision-emulation flag
is enabled in serving. Results are in `rms-ue8-rounding-diagnostic-results-v1/`
and `rms-ue8-rounding-analysis-v1.json`. These producer findings do not resolve
the separate historical147 PIECEWISE/index4 numerical counterexample or prove
model quality.

A separate forced-tile ablation then closes this new8192-row reproduction
counterexample. The two saved caches select different `R0_BLOCK` values,
2048 versus8192, for the identical generated RMS kernel. Replaying that exact
kernel body on hash-matched input/weight, with the same grid, XBLOCK1,
16 warps and one stage, reproduces **both normalization and quantized byte/scale
hashes** from the corresponding earlier run for each tile. This demonstrates
reduction-configuration sensitivity rather than an input or fused-kernel change.
The original non-reproduction flags remain intact; the new positive ablation
is separate evidence in `rms-m8192-forced-tile-results-v1/`, including PTX/TTIR.
No compiler configuration is forced in serving. The historical147
PIECEWISE/index4 counterexample remains separate and unresolved.

The isolated direct-consumer proof completes successfully from
`rms-ue8-direct-consumer-v1/prepared-plan-v2.json`. It extracts six unchanged
upstream consumer AST units into a fresh process, bypassing production kernel
selection through an explicit metadata fixture. It tests hidden5376,
N2048/3072/5376 and M31/32/33/80/800/8320, comparing QA eager/compiled/graph
outputs against the same direct GEMM with identical Q/scales and actual
UE8-postprocessed synthetic weights. A separate M31 raw FlashInfer control,
True/False DeepGEMM flag comparison and invalid odd-M scale layouts are included.
All identified JIT caches are isolated; existing auxiliary DeepGEMM sources and
native library are loaded read-only via an explicit package path. No source
stage is applied to either checkout. Results and controller status are in
`rms-ue8-direct-consumer-results-v1/`. All21 cases pass eager, compiled and
CUDA graph comparisons exactly, including the three raw M31 controls.
All15 large-M True/False DeepGEMM flag comparisons are bitwise equal with
the tested synthetic UE8-postprocessed weights. Invalid odd-M layouts are
rejected. The controller exits0, all15 protected files remain unchanged,
and postflight confirms all GPUs idle. These finite synthetic cases establish
neither checkpoint quality nor model/serving speed.

A new four-file Gemma integration stage is frozen in
`gemma4-hybrid-rms-integration-stage-v1/`, with exact patch roundtrip and
independent source review. It reuses upstream opt-in/QA/consumer code but
replaces the generic raw producer with a private Gemma opaque producer. The
actual selected consumer advertises a private UE8 marker after weight loading;
False/absent marker, missing/masked key, batch-invariant mode or unsupported
plain-norm inputs fall back to the original tensor path. Below32 actual padded
rows the producer uses CUDA RMS plus actual FlashInfer raw quantization;
at32 or more it uses the tested native fused UE8 producer. Full scale backing
storage covers every padded row, including the last group. No generic key
field or new user flag is introduced. The original two guarded call sites and
BF16 residual ordering remain intact.

This stage is copied into a separate main274a5e19 runtime,
`/jobtmp/gemma4-hybrid-rms-integration-runtime-v1`, with byte-identical native
library c6e2d9f2d2d118ac4047240967557c19c8e810a169e48970b1f467e70dd21c1e.
The prior serving checkout, user checkout and native test workspace are
unchanged. Actual whole-helper/selected-kernel proof completes successfully from
`gemma4-hybrid-helper-proof-v1/prepared-plan-v2.json`. It uses real kernel
construction and weight postprocessing with a local offline ModelConfig,
fourteen random M/N cases, four zero/tiny31/32 boundary cases and eight
key/marker/LoRA/batch-invariant fallbacks. It checks eager/compiled/CUDA graph
outputs, raw FP8 bytes and full scale-storage capacity. Status is in
`gemma4-hybrid-helper-proof-results-v1/`. All18 execution cases match eagerly,
compiled and in CUDA graphs, and all8 fallbacks pass. Actual post-load marker,
native large-branch reference, small CUDA-RMS/FlashInfer production consumer,
scale backing capacity and zero/tiny31/32 behavior pass. The controller exits0,
protected files are unchanged and postflight confirms idle GPUs. This clears
the prerequisite for TP8 model validation; no decoder/model-quality/serving pass
exists yet.

The TP8 model-pair diagnostic is now running from
`gemma4-hybrid-modelpair-fa4-v2/prepared-plan-v1.json`, output tag
`gemma4-hybrid-modelpair-v3`. It retains C4/C13/C57 warm and mixed workloads
(107/40/24 turns),400 measured isolated requests per query, first-token/eager/
restored controls and full1319 quality in each arm. The two arms share the new
native library, source and FULL32768/maxseq64/threshold0 profile; only the
existing fusion opt-in differs,0 then1. All8 workers must verify source/native
identities, FA4/BF16-KV/no-speculation and the60-layer/120-site eligibility
census before measurements. Original60 gates remain unchanged and UNASSESSED
by this FP8-only pair; BF16 references and three restarts remain required.

The first launch was interrupted during startup after a temporary user request
to exclude concurrency above8. The user immediately revoked that restriction
and explicitly restored optimization against the existing13/57 benchmark;
C4 remains included. The interrupted attempt has zero measurements/arms and
terminal130 with clean final idle and unchanged runtime. Its raw artifacts
remain intact. The new versioned plan changes only output/cache paths and tag,
with identical source inputs, command, workloads and gates. This is a
user-directed restart before measurement, not a latency-result reroll. Evidence:
`gemma4-hybrid-modelpair-v1-user-interruption-note.json` and both source archives.

The second launch stopped before measurement because the isolated runtime
omitted the gitignored FA4 CuTe Python package. Its startup1 status, traceback
and clean final idle remain intact. The dependency is corrected by copying
54 Python files byte-for-byte from the working checkout. Every file matches
the CMake namespace transformation of the pinned **vllm-project/flash-attention
fork**, revision `b273a02621d3c94bc747df67204f591b67f38fae`; the CMake repository
URL and commit are verified. Actual interface import and H100/SM90 FA4 support
preflight pass. No generic upstream FlashAttention package is installed.

Runtime setup-v2 preserves the old921 entries and adds those54 dependency
hashes. The separate corrected controller adds only setup/default-tag and
passive FA4 interface path/hash checks. Independent review verifies all975
runtime entries,38 source inputs and unchanged commands/workloads/gates/quality
protocol. The current modelpair-v3 starts with fusion disabled, then enabled;
neither interrupted nor failed startup produced benchmark measurements.
Evidence is in `gemma4-hybrid-fa4-fork-copy-v1.json`,
`gemma4-hybrid-fa4-import-preflight-v1/result.json` and
`gemma4-hybrid-modelpair-fa4-independent-review-v1.json`.

The fusion-disabled control finishes all six serving cells and full1319 quality
with clean exit0:1276 established correct,1256 strict correct, zero truncations
and1319 stop finishes. Its isolated q80/q800/q8192/q18304 medians are
23.888/36.494/177.351/396.065 ms (400 measured samples each). All8 ranks verify
FA4, BF16 KV,60 layers and120 inspected sites with fusion disabled. These are
unpaired control results and do not replace prior failures or establish a win.
The fusion-enabled arm is now collecting its measurements.

The user explicitly chooses to continue the CUDA approach. The bounded CuTE
source audit finds no drop-in matching producer: inspected CuTE FP8 norm uses
a supplied scalar scale rather than dynamic group128 output scales. The
existing group128/TMA producer uses CUDA but has a different raw-scale/rounding
contract and mandatory extra buffers. CuTE/CUDA optimality is not established;
no CuTE implementation or benchmark is prepared. The audit is recorded in
`gemma4-cute-cutlass-rms-group128-audit-v1.md`.

Actual generated off/on model code also shows a model-level fusion tradeoff:
the off path combines post-attention norm, residual add and pre-MLP norm, and
can combine post-MLP/add/scalar with the next input norm. The new opaque
producer separates some of that work. The standalone norm+quant comparison
does not capture those existing compiler combinations; serving results must
determine the net effect before accepting this kernel integration.

A separate source-only Gemma integration stage reuses upstream63202/61073 once,
with two guarded plain-norm call sites and the existing opt-in Hopper flag.
Its QA/layout/fallback/residual handling passes independent source review. The
stage preserves upstream raw-scale behavior, which differs from this campaign's
UE8True path; it remains unapplied and has no GPU/quality/performance evidence.
See `upstream63202-gemma4-source-stage-v3/release-manifest-v1.json` for exact
dependency/adaptation diffs and provenance limits.

A separate source review of upstream FA4 FP8-MMA work finds no ready option
within the fixed BF16-KV scope: PR60372 requires E4M3 paged KV storage and
head512/page64, while Gemma4's local head256 FP8 path remains on FA3. BF16
query/output support does not imply BF16 KV support. No attention flag, KV
precision or implementation is changed. Evidence is in
`fa4-fp8-mma-reuse-review-v1/assessment.md`.

A separate four-arm full pilot is prepared and independently reviewed
(`contemporaneous-full-pilot/release-v6.json`), but has not run. It retains all60
performance points and the original requirement that optimized FP8's established
and strict aggregate quality counts each equal or exceed native FP8. The actual
reader passes equality and rejects each independently lowered score in CPU
fixtures using the saved1319-question artifacts. That validates the reader's
quality gate, not a new model result. The separate546-cold case cannot replace
the original37-cold guard, and this pilot cannot replace three-restart acceptance.

The first collective microbenchmarks reject increasing the Torch symmetric-memory
cutoff: its full wrapper is about12% slower at padded8320 rows and21% slower at
16512 rows than PyNCCL. A single-stream repeat confirms this. Forcing
`NCCL_PROTO=Simple` gives essentially unchanged large-payload latency and worsens
the64-row PyNCCL case from about29 to43.5 microseconds. Neither change is selected.

This configuration reduces scheduler capacity from1024 to64, while all measured
concurrency levels remain at most57. Final FP8 graph capture costs104 seconds
and4.34 GiB/rank. It does not establish capacity equivalence or a universal win.
GPU Nsight correlation is complete: observed cold GPU work is faster in FP8,
captured warm work is near parity, and uncaptured medium batches have large
host/collective envelopes. The older2048-token prefill cap's cold-median penalty
is retained in the history below; it is no longer the current candidate.

One measured bottleneck is synchronous first-output detokenization: real CPU
objects spend about 5.7 ms priming an 8K prompt on the first update, versus
0.006 ms on subsequent updates. Batches with three new warm requests spend
about 16 ms in output processing. A guarded native-decoder context optimization
is now frozen and CPU-validated: 20,986 original/candidate pairs and 285,960
per-step comparisons pass. The actual constructor takes reduced context in
the eligible cases; this is not a fallback-only pass. Naive truncation and
state-cache substitutions failed byte-fallback parity and are not accepted.
The combined serving candidate improves FP8 latency, but the cross-launch
comparison does not isolate the detokenizer as its sole cause. A fresh CPU-only
run now measures final `47992...` on clean main: first-update p50/p95 is
0.00794/0.01225 ms. Earlier development timings remain separately labeled.

The `detok-window-full1024-pilot-v1` serving/evaluation pilot completed,
FP8 then BF16, from clean main `274a5e19` with the frozen graph-coverage,
TMA-alignment, and safe-detokenizer patches. It uses FULL1024 and the default
output chunk size 128, with C4/C13/C57 warm/mixed cases, one within-server
repeat, isolated q800/decode/long-prefill controls, and full 1,319-item quality
evaluation for both arms. All four serving references validate. Five of six
warm/mixed cells pass; the C57 mixed counterexample is retained. This pilot does
not replace the three-restart acceptance gate.

Full unmodified-main GSM8K baselines completed: FP8 1,276/1,319, BF16 1,278/1,319;
both strict scores are 1,257/1,319, with no request errors or truncations.
The FULL1024 candidate scores FP8 1,277/1,319 and BF16 1,274/1,319; strict scores
are1,258 and1,254. The threshold2048/FULL4096 pilot scores1,277 per precision,
strict1,258 FP8 and1,256 BF16. All evaluations have zero request failures or
truncations. FP8 improves by one answer against its native baseline; BF16 loses
four then one across these different pilots. Aggregate scores do not establish
per-question equivalence. Numerical attribution and repeated acceptance remain
open. Isolated reduction correctness
passes all 48 cases; this does not establish harmlessness of amplified outputs.

The fourth Claude CLI review challenged unmeasured output-backlog starvation.
The unbenchmarked batch-end-yield draft was withdrawn after the CPU probe
identified substantial within-batch work. No output-loop production change is
currently present. The separate mixed-FA4 GPU control completed all 839
validation gates. Forced splits help selected global-attention shapes but can
regress local attention by 97% and change outputs; no serving win or uniform
split policy is established.

A fifth Claude CLI review packet was prepared, but automatic approval review
rejected exporting its 61,557 bytes of source and benchmark evidence as
potentially sensitive. Approval for that exact packet and Anthropic destination
was requested; no fifth review ran. Local subagent reviews and diagnostic work
continue without that export.

## Approved active goal

Approved by the user on 2026-10-27 and registered as the active goal:

Deliver a reviewable optimization that removes the reproduced Gemma4 FP8
warm-TTFT regressions against BF16 while preserving FP8's decode advantage
and model quality. Use fresh main, RedHatAI FP8-block versus native BF16,
TP8 H100, FA4, BF16 KV, and speculation disabled.

Acceptance criteria:

- FP8 warm-serving p50/p95 TTFT within 3% of BF16 on repeatable warm80 and
  mixed workloads at concurrency 4, 13, and 57.
- At least 30% lower FP8 TTFT than its baseline for the 800-token regression
  case.
- Decode and long-prefill latency within 3% of their FP8 baselines.
- Explain cross-launch output differences and pass relevant correctness and
  model-quality evaluations.
- Validate across at least three server restarts with reversed comparison
  order. Keep workload, configuration, and measurement boundaries explicit.

Start by establishing reproducible staggered-arrival benchmarks and capturing
the concurrent warm13 regression with Nsight. Then optimize the measured
bottleneck in graph coverage, host submission, or scheduling. Deliver a focused
patch, reproducible benchmarks, and updates to this file. Retain counterexamples
and keep the goal open if any acceptance criterion remains unmet.

## Configuration and source provenance

- BF16: `/models/google/gemma-4-31b-it`.
- FP8: `/models/RedHatAI/gemma-4-31B-it-FP8-block`.
- Eight H100 80GB GPUs, tensor parallelism 8, FA4, BF16 activations and KV cache.
- Maximum model length 65,536; GPU memory utilization 0.85; maximum batched
  tokens 32,768; language-model-only mode; thinking disabled.
- Tool-call and reasoning parsers: `gemma4`. Bidirectional-attention override:
  `{"text_config":{"use_bidirectional_attention":null}}`.
- Initial isolated/Nsight campaign main:
  `94ad4dfbaec94e382b53983c3d8256c4fff3b24e`, in
  `/jobtmp/gemma4-fp8-regressions-main`. Its native core and DeepGEMM were
  built from that source. Those historical measurements retain this revision.
- Current V2 staggered/default/candidate/FULL2048 campaign main:
  `274a5e19a18460ed41d47117398494b721e3df16`, refreshed at campaign start.
  All four default/candidate arms and the FULL2048 pair record this HEAD;
  no fetch or HEAD mutation occurred during those runs. The native core was
  rebuilt at `e3883b7a7ec41f3117dfb7ca764746ec80a24d87`; the historical
  source audit records no native changes from that revision through
  `274a5e19`. Installed package/version text can still contain `94ad4dfb`;
  saved Git HEAD, source diffs, and the rebuilt native artifact identify
  the measured code.
- The separate development checkout contains existing experiments and is
  **not** the benchmark baseline.
- Seven later commits through `741584f8ca89c98cc6f18d2b4c6ec1e6d7e021b3` were
  audited; they do not change the selected H100/Gemma4/BF16-KV execution path.
  This is an audit, not a measurement relabeled to that newer revision.
- Runtime: PyTorch 2.14.0+cu132, Triton 3.8.0, FlashInfer 0.7.2.post1,
  CuTeDSL 4.8.0, Transformers 5.20.2, compressed-tensors 0.17.
- Nsight Systems: 2026.5.1.161-265138896106v0.

The current server uses an explicit `--attention-config.flash_attn_version 4`;
the supplied `VLLM_FLASH_ATTN_VERSION=4` is also retained in provenance.
Every run saves its complete launch command, environment, and source revision.

## Starting observations supplied by the user

These are the production observations motivating the investigation, not results
from the synthetic reproduction below:

| Metric | FP8 against BF16 | Conditions |
| --- | --- | --- |
| Decode p50 | 7–10% faster | All supplied load levels and request types |
| Whole-request p50 | 4–10% faster | About 110 output tokens/request |
| Cold TTFT | 4–9% faster | First request; about 7–9K prefill tokens at p50 |
| Warm TTFT p50 / p95 | 5–12% / 5–16% slower | About 80 uncached tokens at p50 |
| Warm TTFT p99 | 10–17% faster at 50–57 calls | Consistent with shorter decode queues |

For example, at 13 calls, warm TTFT was 64→67 ms and decode was
7.03→6.40 ms/token. At 57 calls, warm TTFT was 66→72 ms. Cold request p95
also regressed at 57 calls, 1,859→1,999 ms. Faster decode or p50 latency does
not imply all request tails improve.

## Reproduced regression map before Nsight

The paired serving study covered 45 cases, retaining 9,240 valid timed requests.
Precision order was reversed in confirmation. A transient HTTP disconnect in
the first screen was retained; the missing case was recovered in confirmation.

- With an 8,192-token cached prefix, isolated 512-token prefill was faster
  under FP8; at 513 tokens, FP8 TTFT increased from roughly 56.4 to 80.5 ms.
- Isolated 800–1,024-token prefills regressed by roughly 30–38%; 2,048 tokens
  regressed by about 9%. Cold 513–1,024-token cases also regressed.
- Long 4K–18K prefills favored FP8.
- Isolated 80-token warm requests were near parity or faster with FP8.
  Concurrent warm80 and mixed cases showed regressions, but scheduling and
  queueing make their attribution different from an isolated request.
- The important boundary is **total scheduled token rows per forward**,
  including mixed requests, not each request's uncached length alone.
- H100 FP8 peak tensor-core FLOPs do not guarantee faster small GEMMs, nor
  faster quantization + GEMM + host submission + collective synchronization.

Earlier PyTorch profiling suggested additional uncaptured host overhead. Those
inclusive operator scopes are not additive and are not precise CPU attribution.
Nsight was used to test that hypothesis.

## Nsight experiment and measurement boundaries

Six real serving captures: BF16 and FP8 for each of:

1. Detailed CUDA/NVTX/OS tracing with graph nodes and requested CPU sampling.
2. Lighter CUDA/NVTX tracing, opaque graph activities, all CUDA APIs.
3. The same lighter trace with only `max_cudagraph_capture_size` changed
   from its default 512 to 2,048.

All use an 8,192-token cached prefix, query lengths 80, 512, 513, 800, 2,048,
and 8,192, with one output token. Per shape: 12 measurements before collection,
5 during collection, and 12 after collection, with shape warmups. All 1,044
saved requests passed checks for exact cached/computed prompt token counts and
one generated/streamed token. All six captures completed with exit code zero.

There are 240 native worker forward NVTX ranges per capture: six shapes × five
requests × eight ranks. GPU operations are correlated through the launching
CUDA API enclosed by each worker range. GPU idle excludes recorded kernels,
memcpy, memset, and opaque graph activities. Report summaries use rank-zero
medians; all-rank records and a common-timebase timeline are retained.

Important limits:

- Before/after collection is stopped, but **Nsight remains attached**. These
  measurements are not fully unattached serving benchmarks.
- Active tracing perturbs latency. CPU and GPU work overlap; independent
  medians do not necessarily add.
- CPU time outside CUDA APIs includes Python/native work and possible
  descheduling. It is not a Python-only measurement.
- Requested CPU backtrace/context-switch collection was unavailable. Worker
  Python/native stack attribution was not obtained from the exports.
- Nsight reported possible CUDA/NVTX incompleteness, including workers. All
  expected forward ranges exist and per-shape kernel counts agree across
  ranks/repeats; these checks do not prove absence of event loss. Full warnings
  are preserved in the report.
- The isolated one-token-output workload does not reproduce the entire
  concurrent 110-output-token production workload. Five captured requests per
  shape do not establish p95/p99 behavior.

## Main Nsight finding: host submission and rank waiting

For an 800-token prefill, rank-zero median values from the lighter capture:

| Measurement | BF16 | FP8 |
| --- | ---: | ---: |
| GPU GEMMs | 9.43 ms | 8.18 ms |
| Activation quantization | 0 | 1.43 ms |
| Attention | 3.77 ms | 3.75 ms |
| CPU forward range | 50.76 ms | 66.54 ms |
| CPU outside CUDA APIs | 39.03 ms | 53.45 ms |
| GPU projected forward span | 50.35 ms | 72.20 ms |

The FP8 GEMMs are about 13% faster at this shape, but separate activation
quantization consumes that saving: about 9.61 ms combined versus 9.43 ms BF16.
The larger end-to-end regression is not explained by matrix multiplication.

The eight-rank timeline shows a late-submitting rank with large GPU idle gaps,
while earlier ranks spend much longer in all-reduce. Which rank is slow changes
between runs. In one matched 800-token sample, FP8 GPU2 has 51.5 ms idle inside
its forward span; the BF16 sample's most-idle rank has 27.2 ms. Start-of-forward
skew is small; delays recur across layers.

The detailed trace independently shows the same pattern. For its rank-zero
800-token case, the early-arrival wait proxy rises from about 28.5 to 52.6 ms,
while all-reduce kernel time rises from about 33.3 to 57.3 ms. The proxy ends
when the last rank enters each collective; it is not exact transport isolation.
**Do not count these waits as a separate additive penalty or interpret the
entire all-reduce duration as network transfer.**

At 8K new tokens, FP8 still has more host work, but longer GPU computation
largely hides it: GEMM time falls from about 80.4 to 45.3 ms, plus about 10.6 ms
quantization. This explains why long prefill can win while short uncaptured
prefill loses.

## Graph-coverage diagnostic

Only the compilation configuration changed:

```json
{"max_cudagraph_capture_size": 2048}
```

Client TTFT medians, collection stopped after the trace, 12 requests per cell:

| New tokens | Default BF16 | Default FP8 | Expanded BF16 | Expanded FP8 |
| --- | ---: | ---: | ---: | ---: |
| 80 | 32.74 ms | 32.72 ms | 32.95 ms | 34.31 ms |
| 512 | 34.39 ms | 34.08 ms | 35.64 ms | 35.54 ms |
| 513 | 59.92 ms | 85.84 ms | 35.50 ms | 35.74 ms |
| 800 | 61.09 ms | 78.10 ms | 43.26 ms | 41.28 ms |
| 2,048 | 72.45 ms | 84.22 ms | 71.84 ms | 65.45 ms |
| 8,192 | 220.56 ms | 192.36 ms | 222.14 ms | 190.61 ms |

At 800 tokens, FP8 TTFT falls 47.1%, becoming 4.6% faster than BF16 with the
same expanded coverage. At 2,048, FP8 falls 22.3% and is 8.9% faster than BF16.
At 513, FP8 remains about 0.7% slower. The 8K control outside coverage is
largely unchanged. Small-case variation remains; this is not an all-shapes win.

The captured 800-token CPU range falls to about 17 ms in both precisions;
projected GPU span is 25.35 ms BF16 and 24.79 ms FP8. This supports targeting
host submission/amortization first, without proving which individual host
function accounts for the uncaptured overhead.

### Output comparison: unresolved quality gate

Same-precision, same-prompt/phase/index first-token hashes:

- Detailed versus lighter tracing, unchanged graph configuration: **0/174
  differences per precision**.
- Default versus expanded graph coverage: at 2,048 new tokens, **9/29 BF16**
  and **2/29 FP8** first tokens differ. All other tested shapes match.
- Only output hashes were saved in this pass. Token identities, top-logprob
  margins, and full output-quality evaluation were not captured.

These differences are not automatically correctness failures, and they are
not cleared as harmless numerical changes. Investigate with deterministic
repeated prompts and actual token IDs/top logprobs before promotion.

### Memory and startup cost

| Precision | Default KV budget / capacity | Expanded KV budget / capacity | Final graph allocation | Final capture time |
| --- | --- | --- | --- | --- |
| BF16 | 53.19 GiB / 464,686 tokens | 52.10 GiB / 455,198 tokens | 0.29→0.61 GiB | 22→47 s |
| FP8 | 56.67 GiB / 495,109 tokens | 55.60 GiB / 485,775 tokens | 0.33→0.58 GiB | 25→55 s |

Values are per GPU where applicable. The approximately 1.1 GiB KV-budget
reduction includes conservative graph-memory estimation; it is larger than
the final graph-allocation increase. Startup logs also contain preliminary
profiling captures; the table uses final captures.

## Priorities

Current priorities after the three-pair cold-repeat diagnostic:

- **Resolve C57 mixed cold p95.** The FULL32768/maxseq64 pilot meets all six
  warm/mixed targets against equally configured BF16 and improves isolatedq800
  by52.5%, but fails its cold-p95 guard. Fresh restart2 repeats that failure in
  both37- and546-cold cases. Restarts1/3 pass; all six cohorts exit cleanly with
  source restoration. Preserve every case separately. Earlier admission tracing
  proves a pre-admission waiting mechanism;
  current client-only repeats cannot establish each event's internal cause.
- **Reduce actual forward cost after the negative policy/backend screens.**
  The existing full-wrapper screens reject the tested Torch cutoff/two-shot,
  forced Simple and registered-symmetric-memory changes. A short fresh-process
  algorithm screen now confirms large defaults already use NVLS/Simple, with
  no useful forced-algorithm gain against either bracketing default. The matched
  threshold0-versus4096 diagnostic also rejects4096: both37- and546-cold phases
  worsen cold and warm TTFT, despite exact two-chunk formation and preserved
  isolated/decode controls. The earlier2048 cap remains rejected. Neither a
  per-request cap nor earlier worker admission guarantees shorter TTFT. The
  next experiment is the upstream-based fused producer's isolated native build
  and correctness, then timing of the complete RMS+quant boundary.
- **Preserve the measured warm improvements and correctness.** Keep the safe
  detokenizer, narrow decode graph coverage and TMA alignment fix as the
  provisional candidate, with maxseq64's capacity cost explicit. Preserve
  q800, decode, isolated long-prefill, C4/C13/C57 cold/warm and quality guards.
  Coordinate upstream overlap before any PR: the TMA hunk duplicates49928;
  related detokenizer/graph work also exists.
- **Reuse upstream fusion work if further kernel work is needed.** The private
  RMS/UE8 drafts remain unapplied. Existing producer/consumer PRs provide a
  reuse path, but their rounding and scale contracts differ from the measured
  native path. They need direct correctness, complete-boundary performance and
  model evaluation before selection; no kernel speedup is established here.
- **Complete full qualification and numerical review.** The prepared four-arm
  pilot retains all60 performance points and both original FP8 aggregate quality
  gates. Do not run it merely to replace a failed cold comparison. Three full
  acceptance restart blocks remain required. Controlled norm/FA4 interventions
  reproduce the later FULL2048 outputs exactly; the older high-margin piecewise
  flip remains a separate unresolved target. Full aggregate quality scores do
  not imply per-question or bitwise equivalence. The old failed preflight and
  draft12-arm acceptance collector remain unchanged.

The original numbered priority list below records the initial Nsight decision
and the meaning of later references to “priority 1”:

1. **Graph coverage and host submission amortization.** Largest demonstrated
   opportunity. Resolve the 2,048-token output differences; test fully
   unattached serving with concurrent warm80 and mixed requests plus decode;
   compare matched BF16/FP8 configurations. Consider sparse additional capture
   sizes to reduce startup/memory cost. Keep production defaults unchanged
   until quality, concurrency, and resource tradeoffs are understood.
2. **Uncaptured FP8 dispatcher and launch preparation.** Investigate
   `vllm/model_executor/kernels/linear/scaled_mm/flashinfer.py`, particularly
   `_dynamic_flashinfer_deepgemm_blockscale_gemm_impl`. There are four
   quantized linear calls per layer over 60 layers. For M≥32, the selected
   path separately quantizes inputs and calls DeepGEMM. Isolate host costs
   before changing dispatch. `torch.cond` with a Python bool has a direct fast
   path; it has **not** been isolated as the culprit.
3. **Activation producer/quantization fusion, then residual small-M tuning.**
   FP8 adds 240 input-quantization launches per forward, about 1.43 ms at M800.
   Address this to preserve the GEMM saving and investigate residual 513-row
   regressions after host overhead is reduced.

Defer network-bandwidth tuning based solely on long all-reduce bars. Also
defer TMA encoding alone: 960 `cuTensorMapEncodeTiled` calls total about
0.24 ms at M800 in the lighter trace. Surrounding host construction may still
matter, but descriptor encoding itself does not explain tens of milliseconds.

## Work on priority 1

Status: initial investigation completed; production mitigation remains open.
The bounded follow-up was designed to:

1. Preserve the original baseline and use the isolated main worktree.
2. Repeat the 29 matched 2,048-token prompts with token IDs and top logprobs,
   including same-configuration repeatability controls.
3. Measure default and expanded graph coverage without an attached profiler,
   across BF16/FP8, isolated boundary cases, concurrent warm80, and the prior
   synthetic mixed workload with 110 generated tokens.
4. Validate cache hits and output counts, retain negative results, and record
   numerical/output differences separately from performance measurements.
5. Decide whether graph coverage is a useful bounded configuration mitigation
   or whether a code-level submission change is needed. No universal default
   increase or production source change has been made.

The supplied real request trace is unavailable; concurrent tests remain a
synthetic proxy. A repeatable speedup does not by itself clear model quality.

### Fresh-main follow-up setup

The priority-1 follow-up refreshes main to
`e3883b7a7ec41f3117dfb7ca764746ec80a24d87` (14 commits after the Nsight baseline).
The isolated checkout was advanced and its native `_C_stable_libtorch`
extension rebuilt and installed using the incremental CMake workflow. The
dependency pins and DeepGEMM build configuration are unchanged. Historical
measurements above remain attributed to `94ad4dfbae`.

The new main includes changes to bounded prefix-cache reads for prompt
logprobs, so this follow-up measures the refreshed source directly rather
than assuming every scheduler/cache path is identical. Diagnostic requests
use generated-token logprobs, and exact cache-hit checks remain mandatory.

The four profiler-free arms are default BF16, expanded FP8, default FP8,
expanded BF16. Each arm runs two passes with reversed case order: isolated
80/513/800/2,048-token cases; warm80 concurrency 4/13/57; mixed concurrency
13/57. Concurrent requests generate 110 tokens. Each server also repeats all
29 original 2,048-token prompts twice, saving actual token IDs and top-20
logprobs. These are within-server repeats, not independent restart replicas.

Follow-up harness and outputs: `/jobtmp/gemma4-graph-coverage/study.py`,
`analyze.py`, and `{default,dense2048}-{bf16,fp8}/`.

### Profiler-free four-arm results

All four arms completed with exit code zero. Each saved 1,328 timed serving
requests plus 58 boundary-token diagnostic requests. All serving cache checks
passed. The table gives the **range of the two per-pass p50 TTFTs**, in ms;
these ranges are not confidence intervals. Concurrent cases use closed-loop
bursts, making their scheduling behavior different from staggered production
calls. The two workload passes share a server instance.

| Case | Default BF16 | Default FP8 | Expanded BF16 | Expanded FP8 |
| --- | ---: | ---: | ---: | ---: |
| Isolated q80 | 30.4–30.8 | 30.2–30.6 | 29.8–30.5 | 30.1–31.5 |
| Isolated q513 | 55.3–56.8 | 78.1–79.6 | 33.1–34.5 | 34.2–36.2 |
| Isolated q800 | 55.5–56.7 | 72.1–73.3 | 41.8–42.6 | 40.5–42.3 |
| Isolated q2048 | 70.7–70.8 | 75.4–76.0 | 68.6–70.6 | 64.3–66.7 |
| Warm80 C4 | 55.8–57.5 | 56.9–58.8 | 53.0–56.0 | 55.0–58.6 |
| Warm80 C13 | 82.3–157.6 | 76.0–86.8 | 91.3–148.1 | 118.1–125.7 |
| Warm80 C57 | 476.9–560.5 | 492.7–541.5 | 389.5–436.5 | 288.2–479.2 |
| Mixed C13, warm subset | 167.4–184.7 | 139.0–194.3 | 148.5–197.0 | 106.2–115.8 |
| Mixed C57, warm subset | 965.6–966.1 | 572.7–732.8 | 880.8–956.1 | 572.4–582.8 |

The isolated FP8 q800 improvement reproduces without an attached profiler:
73.35→40.52 ms in pass 0 and 72.13→42.34 ms in pass 1, reductions of about
45% and 41%. Mixed C13 improves by about 45% and 17%. However, pure warm80
C13 **regresses by 65% and 36%**. Mixed C57 improves by 20% in one pass and is
essentially unchanged in the other. Large BF16 C13 variation also demonstrates
that scheduling phase matters. These results reject an unconditional 2,048
recommendation; they do not isolate every concurrent regression to graph replay.

Expanded FP8 remains slower than matched expanded BF16 at q513 and warm80 C4.
The all-cases target is still unmet.

Numerical controls:

- Within each configuration, all 29 prompts produce identical tokens **and
  identical saved top-20 logprobs** across the two repeats.
- Across default/expanded configurations on fresh main, BF16 changes 12/29
  first tokens and FP8 changes 2/29, identically in both repeats.
- One FP8 change switches token 37798→715 despite a default top-two logprob
  margin of about 2.594 nats; the expanded margin favors the other token by
  about 1.128 nats. This is not merely an argmax tie. Another flip has a much
  smaller default margin, about 0.057 nats.
- The fresh default BF16 run also differs from historical Nsight hashes on
  9/29 prompts. Those cross-run comparisons additionally differ in source,
  profiler attachment, request history, and logprob collection, so they cannot
  isolate graph coverage. Preserve them as diagnostics, not proof of a cause.

Fresh-main KV budgets are 53.20→52.12 GiB BF16 and 56.68→55.62 GiB FP8.
Final graph allocations are 0.29→0.61 GiB and 0.33→0.58 GiB respectively.
Final graph capture takes 23→46 seconds BF16 and 25→54 seconds FP8. The
package version string is generated metadata from the earlier installation;
the saved Git HEAD and rebuilt native artifact identify the actual source.

Full results: `/jobtmp/gemma4-graph-coverage/report.json`, `timings.csv`, and
`graph-coverage.png`. The harness, analyzer, and renderer pass Ruff checks.

### Same-process dispatch control

Status: completed on FP8, exit code zero. This diagnostic loads and captures the 2,048-row
configuration once, then filters MRV2's graph-dispatch candidate map at limits
512, 1,024, 2,048, 2,048, 1,024, 512. Weights, compiler state, and allocated
graph pool remain in the same process. All eight ranks must report the
expected graph/eager modes before each block. This changes no production
source and runs only in the isolated loopback benchmark server.

Each block repeats the 29 boundary prompts, times isolated q800, runs 24
arithmetic questions framed as exact 8K cached + 2K uncached Gemma4 chat
prompts, and tests warm80/mixed C13 with 104 requests each. The arithmetic
check is bounded model-quality evidence, not a comprehensive evaluation.

The 1,024-row cutoff is motivated by preserving coverage for q800 while
leaving a synchronized 13×80=1,040-token batch uncaptured. Intermediate
mixed batches can still differ; this hypothesis must be measured. Because
all graphs are already allocated, the control **does not measure** the lower
startup/memory cost of launching a genuine 1,024-row configuration.

Scripts: `/jobtmp/gemma4-graph-coverage/same_process.py` and
`graph_probe_worker.py`; outputs: `same-process-fp8/`.

Results, in execution order (TTFT p50 in ms):

| Dispatch limit | q800, 12 requests | Warm80 C13, 104 requests | Mixed C13 warm subset, 100 requests |
| --- | ---: | ---: | ---: |
| 512, forward | 75.66 | 96.58 | 212.08 |
| 1,024, forward | 42.63 | 104.49 | 136.22 |
| 2,048, forward | 42.62 | 103.98 | 192.66 |
| 2,048, reverse | 41.61 | 126.93 | 155.11 |
| 1,024, reverse | 41.59 | 164.20 | 232.09 |
| 512, reverse | 88.20 | 204.14 | 232.68 |

The q800 improvement persists in both directions: approximately 44–53%
lower TTFT at limit 1,024. The uncaptured path drifts considerably while
graph-replayed q800 remains around 42 ms. This supports submission overhead
as a useful target; it does not assign that drift to a particular CPU function.
The concurrent results are inconsistent: the default limit itself moves from
96.6 to 204.1 ms in warm80 C13. Neither expanded cutoff establishes a general
warm-request win in this synthetic workload. Concurrent decode p50 also varies
across blocks, roughly 6.2–7.0 ms/token.

Numerical and bounded quality results:

- All seven pairwise boundary comparisons, including same-limit repeats and
  512-versus-1,024/2,048 in both directions, have **0/29 token differences**
  and **zero difference in saved top-20 logprobs**.
- All corresponding q800 comparisons have **0/12 output-hash differences**.
- Each of the six blocks scores **24/24** on the long-prefill arithmetic
  diagnostic: 144 correct answers total. This does not establish broad quality.
- Across independently launched configurations in the earlier four-arm study,
  timed isolated q80/q513/q800 first-token hashes also match. Timed q2048
  changes 6/20 BF16 tokens per pass and 0/20 FP8 tokens per pass; the separate
  29-prompt FP8 diagnostic still has the two recorded changes.

The same-process result means CUDA graph replay **alone is not a sufficient
explanation** for the cross-launch output differences on these inputs.
Compilation/autotuning, allocation/workspace state, and request/prefix history
remain candidates to isolate; none has been established as the cause. Do not
remove the cross-launch quality gate based on this bounded control.

The control collected 174 boundary-token diagnostics, 72 timed q800 requests,
144 math responses, and 1,248 concurrent serving requests. All graph-mode,
cache-hit, and timed-output-count checks passed. Report:
`/jobtmp/gemma4-graph-coverage/same-process-report.json`. The clean main worktree
has no tracked changes, and all eight GPUs were released after the runs.

### Updated priority-1 decision and next experiments

**Do not make an unconditional default increase to 2,048.** Retain the clear
isolated 513–2,048-token speedups as evidence for graph/submission work, while
keeping the concurrent regressions and numerical differences in the record.
A narrower 1,024 dispatch cutoff preserves the q800 benefit but is not yet a
validated serving recommendation or a measured memory/startup mitigation.

Next within priority 1:

1. Replay staggered arrivals with varied output lengths as well as bursts;
   capture per-step request counts, query lengths, padded rows, graph mode,
   prefill/decode mix, and queue time. Use repeated server restarts and reverse
   configuration order. This will distinguish scheduling phase from a repeatable
   per-step regression and connect the result to the original warm80 workload.
2. Capture the regressing concurrent warm80 C13 path with Nsight under matched
   conditions. The isolated q800 trace cannot establish its complete cause.
3. Isolate the cross-launch output changes with matched prefix/request history
   and compilation/autotune/allocation controls; expand quality evaluation before
   recommending a configuration.
4. After those gates, test a genuine smaller or sparse capture profile and
   measure its actual KV budget/startup cost. If broad graph coverage cannot
   provide a consistent serving gain, target uncaptured host dispatch directly
   (priority 2), rather than widening the graph limit further.

No production code/defaults were changed in this iteration. The diagnostic
worker extension edits only an isolated process's graph-candidate map and is
not an implementation proposed for deployment.

## Staggered replay and the first focused candidate

The next baseline refresh uses main
`274a5e19a18460ed41d47117398494b721e3df16`. The four commits after the previous
follow-up change Python code and documentation; native kernel sources are
unchanged. Historical measurements above retain their original revisions.

The reusable client is
`benchmarks/multi_turn/benchmark_prefix_cache_staggered.py`. It maintains 13
closed-loop clients, staggers their initial starts over 1.2 seconds, and uses
seeded 80–140-token output lengths. Burst and staggered-fixed-110 controls
separate synchronization effects from output-length variability. The mixed
case retains approximately 3% cold 8K prompts and 10% longer warm suffixes.
Every request checks cache hits and exact requested output-token counts.

Each precision completed 2,080 requests across four cases and two passes in
reversed case order. The first three turns per client are retained as warmup
records and excluded from the reported quantiles. These are two passes within
one server launch per precision, not the three independent restart validations
required by the goal.

Warm-request TTFT p95, ms, pass 0 / pass 1:

| Workload | BF16 | FP8 | FP8 change |
| --- | ---: | ---: | ---: |
| Burst, fixed 110 output | 190.69 / 161.65 | 80.20 / 106.24 | −58% / −34% |
| Staggered, fixed 110 output | 40.75 / 40.41 | 38.09 / 38.07 | −6.5% / −5.8% |
| Staggered, varied output | 44.73 / 44.67 | 48.07 / 48.12 | +7.5% / +7.7% |
| Staggered mixed, varied output | 58.91 / 60.20 | 80.19 / 75.97 | +36% / +26% |

Staggered variable-output p50 is near parity, around 34 ms. Decode retains its
FP8 advantage. This is a stable tail regression under the synthetic replay;
the fixed-output burst results should not be substituted for it.

### Concurrent Nsight evidence

Completed matched four-second CUDA/NVTX captures of staggered mixed C13 on all
eight GPUs. A diagnostic worker extension records every forward's request IDs,
query lengths, context/decode split, padded rows, and graph mode. There are 417
complete cross-rank forward steps in FP8 and 403 in BF16; the analyzer reports
no missing GPU projections or incomplete cross-rank groups.

| Forward category | BF16 GPU span | FP8 GPU span | Samples BF16 / FP8 |
| --- | ---: | ---: | ---: |
| Decode | 7.34 ms | 6.68 ms | 356 / 371 |
| Warm80-containing steps | 15.87 ms | 17.22 ms | 40 / 40 |
| Exact M812: one q800 plus 12 decodes | 49.83 ms | 70.58 ms | 3 / 3 |
| Cold-containing steps | 166.60 ms | 146.10 ms | 2 / 2 |

For exact M812, rank-zero CPU time outside CUDA APIs is approximately
37.71 ms BF16 versus 49.01 ms FP8. M812 is outside default graph coverage and
is already scale-aligned. The extra submission time and rank waiting explain
why these steps can extend mixed-request tails. The q800 aggregate also has
different shapes in each precision, so use this exact-shape comparison rather
than treating all q800-containing steps as identical work.

Warm80 steps remain a separate investigation: rank-zero FP8 CPU time is lower
than BF16 in this capture, while its distributed GPU span is higher. Opaque
graph envelopes include internal waits and idle time. They do not establish
which individual kernels cause the short-prefill difference. Do not extend
the uncaptured-q800 host attribution to every warm80 regression.

An independent audit using a common CPU-entry-to-completion boundary further
qualifies this observation. Exact M92 (80 context + 12 decode tokens), padded
to 96, has a distributed forward envelope of **21.013 ms BF16 versus 20.988 ms
FP8**, essentially parity (29/32 samples). Its GPU-only envelope increases
15.895→17.155 ms while the CPU-entry-to-first-GPU interval falls
4.856→3.786 ms. The longer GPU envelope is therefore not demonstrated to be
a longer forward or the cause of the warm80 serving-tail regression.
Rank-zero CPU time also understates FP8's distributed work: rank zero returns
about 2.83 ms before the latest rank at the median.

Exact M812 retains a substantial regression under the corrected boundary:
the distributed forward envelope is **53.431→74.005 ms** (three samples each).
CPU-start skew falls 0.218→0.078 ms, whereas CPU-end skew grows
7.087→14.175 ms. This supports delay accumulating inside the forward rather
than an initial rank-start skew. These instrumented samples diagnose a
mechanism; they do not replace profiler-free serving acceptance evidence.

Collection/export pauses visibly affect instrumented client request tails.
Use the unattached runs for serving latency and these captures for attribution.
The host has no CPU cgroup quota or recorded cgroup throttling and permits
all 208 logical CPUs; CPU quota is not an established explanation for rank skew.

Artifacts: `/jobtmp/gemma4-staggered/baseline-{bf16,fp8}/`,
`nsys-mixed-{bf16,fp8}/serving.nsys-rep`, `serving.sqlite`, `steps.json`, and
`analysis.json`. Harness: `run.py`, `step_probe_worker.py`, and
`analyze_trace.py` in the same directory.

### Activation-scale layout candidate

The standalone DeepGEMM linear backend honors
`VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES` (default true), but the Hopper hybrid's
dynamic DeepGEMM branch omitted that argument when quantizing activations.
For row counts not divisible by four, DeepGEMM therefore allocates aligned
scale storage and copies the scale tensor again. The focused patch passes
the existing setting through to the quantizer. It changes scale storage and
preserves the quantization arithmetic and backend-selection threshold.

A kernel probe on H100 verifies bit-identical FP8 bytes, scale values, and
BF16 GEMM output for M=33/80/83/91/92/93/513/800, N=2,048, K=5,376.
CPU profiling shows one scale-layout copy removed at each unaligned M.
Cold-L2 CUPTI graph timings for quantization plus GEMM include:

| M | Original | Aligned scales | Copy count |
| --- | ---: | ---: | ---: |
| 33 | 15.20 µs | 13.60 µs | 1→0 |
| 83 | 15.97 µs | 13.76 µs | 1→0 |
| 91 | 16.06 µs | 13.89 µs | 1→0 |
| 513 | 31.87 µs | 28.83 µs | 1→0 |
| 800 | 38.62 µs | 38.62 µs | 0→0 |

Host wall measurements in this initial probe used baseline-first order and
remain diagnostic. Other projection shapes, compiled execution, and serving
impact still need validation. The dominant M812 case and padded M96 short
prefills are already aligned, so this patch alone does not establish the
serving target or the required q800 improvement.

The regression in `tests/kernels/quantization/test_block_fp8.py` compares
exact outputs and checks absence of the redundant copy when alignment is
enabled. The expanded suite covers two projection shapes, contiguous and
row-padded inputs, both scale representations, the disabled-setting control,
and CUDA graph replay with changed input. Against unmodified main it reports
**24 failed, 40 passed**; after the implementation change, **64 passed**.
The failures identify redundant copies; baseline and candidate numerical
outputs match under the suite's exact comparisons.
Logs: `/jobtmp/gemma4-staggered/alignment-test-v2-{before,after}.log`.

Independent subagent review identified a Blackwell portability issue in the
initial test and overly broad documentation. The test is now gated on actual
Hopper hybrid availability, and documentation is scoped to Hopper FP32 scales.
The disabled-setting control retains exact-output checks without prescribing
an external DeepGEMM implementation's copy count. The expanded 64-pass rerun
includes these refinements. The H100 DeepGEMM CI job also gains the scaled-MM
source dependency.

The unattached alignment-only FP8 serving run completed without an end-to-end
win: variable warm p95 was about 50.00/50.47 ms, versus the original FP8
48.07/48.12 ms. This single-restart comparison does not establish a causal
regression, but it does rule out claiming the serving goal is achieved by this
patch. Artifacts: `/jobtmp/gemma4-staggered/aligned-fp8/`.

Two subagents and a tool-disabled Claude CLI review independently examined the
patch and evidence. The completed Claude review is saved at
`/jobtmp/gemma4-staggered/review/claude-review.txt`. Reviews agree that the scale
layout fix cannot address the dominant already-aligned M812 case. Suggestions
about environment-lookup overhead require runtime verification; source shows
the multiprocess workers enable environment caching before inference.

The V2 staggered benchmark now keeps other clients active until all measured
requests finish and uses stable prompt identities across case subsets and
ordering. A CPU-only mock smoke covered C4/C13/C57, prompt hashes, and exclusion
of background requests. Its real-server graph-coverage comparison is running
in `/jobtmp/gemma4-staggered/coverage-v2-{fp8,bf16}/`. Each case has 273
measured requests at C13. FP8 coverage order is 512/1024/2048/2048/1024/512;
the captured 2048 graph pool and loaded weights stay fixed. This is a dispatch
control, not a measurement of a true lower-memory 1024 startup configuration.
Runtime RPC confirms environment caching enabled on all eight workers.

The completed FP8 arm retained 3,276 measured serving requests. Both orders
show a large q800 improvement, while the warm-only tail remains:

| Pass | Runtime cutoff | Isolated q800 p50 | Warm-only p95 | Mixed warm p95 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 512 | 77.34 ms | 48.87 ms | 84.21 ms |
| 1 | 1024 | 42.36 ms | 49.91 ms | 56.54 ms |
| 2 | 2048 | 43.01 ms | 54.61 ms | 59.65 ms |
| 3 | 2048 | 41.90 ms | 50.39 ms | 53.21 ms |
| 4 | 1024 | 42.11 ms | 52.08 ms | 61.10 ms |
| 5 | 512 | 75.65 ms | 48.98 ms | 93.75 ms |

The measured request-plan hashes match across limits. All 225 isolated
one-token outputs after the first pass match their corresponding fixed-input
baseline hashes, including q800 and long-prefill controls. These are
same-process numerical controls, not a substitute for task accuracy or
cross-launch investigation.

The BF16 reference completed with exit zero, adding 1,092 measured requests.
Its two default-512 passes have warm-only p95 **46.15/45.85 ms**, mixed warm
p95 **62.91/63.06 ms**, and isolated q800 p50 **57.17/57.22 ms**. All use the
same V2 measured plans. Thus 1024 piecewise coverage beats this BF16 mixed
reference while leaving warm-only p95 about 8–14% slower. Broad piecewise
coverage alone does not meet the goal.

A supported `cudagraph_mode=FULL`, capture-limit
1024 comparison is now running at C13 and C57; FA4 advertises
full mixed graph support, and this checkpoint has no KV-shared layers or
fast-prefill optimization to disable. FULL also changes decode attention
specialization, so decode remains a required guardrail.

The first FULL launch failed before serving measurements because the read-only
autotuner RPC returned PyTorch's `TorchVersion` string subclass, which vLLM's
message encoder rejects. The task-owned failed launch was stopped and retained
at `full1024-v2-fp8/`; the corrected run is `full1024-v2-retry-{fp8,bf16}/`.
The probe now converts the complete metadata payload to plain primitives.
An actual clean-main MsgpackEncoder/MsgpackDecoder smoke covers rich synthetic
autotuners, nested primitive subclasses, and eight-rank UtilityResult
round-trips; its raw-TorchVersion negative control reproduces the failure.
Evidence: `/jobtmp/gemma4-staggered/numerics-probe-encoder-smoke.json`.
Numerical-control labels `before/captured/after` identify historical prompt
namespaces; they do not mark a graph-state transition in this harness. A
separate zero-cutoff dispatch control explicitly compares replay with eager
execution in the same process and restores the configured cutoff before timing.

A second completed Claude CLI review is retained at
`/jobtmp/gemma4-staggered/review/claude-coverage-review.txt`. It emphasizes
that these single-launch diagnostics cannot clear the acceptance matrix,
restart variance, mixed-batch numerics, or model quality. It also raises a
load-dependence hypothesis: faster FP8 decode increases closed-loop arrival
rate. Quantify that using actual measurement-window arrivals, including
background traffic, before attributing the warm-only tail. A matched-arrival
control can help diagnose the mechanism; it does not replace or relax the
approved closed-loop C4/C13/C57 acceptance criteria. Relevant environment
settings should be recorded without dumping credentials or the full environment.

The completed load audit finds warm arrival rates of 15.334 FP8 versus 14.313
BF16 requests/s (+7.14%) at runtime cutoff 512, including background traffic.
The mixed difference is +3.41%; average in-flight concurrency is about 12.994.
This does not establish the tail mechanism: one same-config FP8 window has a
worse p95 at a lower achieved rate, and both Nsight captures contain exactly
one short+short M171 collision step. FP8's additional two-prefill observations
pair a short prefill with a cold 8192-token prefill. Full audit and joins:
`/jobtmp/gemma4-staggered/load-analysis.{md,json,py}`.

### FULL-mode diagnostic and decode-specialization candidate

The corrected FP8 FULL1024 arm completed with exit zero. Its C13 warm-only
p50/p95 are 35.16/46.83 ms and 34.95/47.48 ms; mixed warm p95 is
53.58/57.27 ms. Decode p50 stays at 7.41/7.44 ms in warm-only cases.
At C57, however, warm decode p50 reaches **13.94/13.97 ms/token**, with
warm p95 86.81/86.31 ms. Historical default-graph C57 burst measurements are
9.48/10.48 ms/token, but they have a different load boundary; a fresh matched
default C57 comparison is required. Plain FULL is not an accepted solution.
The matching BF16 FULL arm also completed with exit zero. Its C13 warm p95
is 45.86/49.85 ms, with decode 8.10/8.10 ms/token. At C57 it has warm p95
88.54/86.98 ms and decode 14.42/14.43 ms/token. Matching FULL configurations
retain FP8's advantage in these samples, but their broad decode captures still
need comparison against a fresh default baseline.

Source explains a concrete specialization risk. FA4's SM90 selection uses
`ceil(max_query_len * qheads_per_kvhead / tile_m)` to estimate M tiles and
select SplitKV. Gemma4's global head dimension is 512, with four Q heads per
KV head per TP rank and tile M=64. Decode graph bounds 1 and 16 both estimate
one M tile; bound 64 estimates four. Packed GQA remains enabled. FA4 uses
heuristic split selection here (`num_splits=0`); the graph split upper-bound
setting is initialized only for FA3's AOT scheduler. A standalone bound
ablation is being prepared to test this mechanism directly.

The standalone single-H100 ablation now confirms that bound effect on identical
BF16 Q/K/V and paged 8192-token shared-prefix data. Four alternating AB/BA rounds
use CUPTI, CUDA graphs, and cold L2; all variants pass the existing BF16
attention tolerance against an FP32 matmul/softmax reference.

| Actual requests / padded slots | Max-Q bound | Selected splits | Attention p50 |
| --- | ---: | ---: | ---: |
| 13 / 16 | 1 | 8 | 68.54 µs |
| 13 / 16 | 16 | 8 | 68.64 µs |
| 57 / 64 | 1 | 2 | 234.58 µs |
| 57 / 64 | 64 | unsplit (returned 0) | 438.69 µs |

At C57 the broader bound is **1.87× slower** for this operation. At C13 the
selected configuration and output hashes match, with effectively equal latency.
This measures one attention call with per-TP-rank dimensions, not distributed
model latency. It explains a concrete cost of plain FULL's broad decode bound;
it does not establish the original warm-tail mechanism or model-level quality.
Artifacts: `/jobtmp/gemma4-staggered/fa4-bounds-c13-c57.{json,log}` and
`fa4_bounds_probe.py`.

A root-only candidate now gives MRV2 FULL separate narrow decode captures and
mixed FULL captures. It prioritizes narrow compatible descriptors, preserves
independent padding ladders, and deduplicates overlapping DBO twins. It adds
no enum or configuration flag and leaves FULL_AND_PIECEWISE generation intact.
At ceiling 1024/max-sequences 1024, the expected capture count grows from 83
to 166; actual memory and startup costs remain GPU validation items.

CPU manager tests report **32 passed**, with CUDA uninitialized before and
after. New regressions first reproduced 8 failures against the original
implementation, and a separate DBO case reproduced duplicate descriptors.
An inherited CPU fixture also needed its existing CUDA capture mock to cover
the external stream call. Two independent reviews found no blocking new
correctness issue for the no-speculation path. An inherited dynamic-width
performance fallback is documented by review and is outside this fix's scope.
After the unmodified-manager FULL comparison finished, the clean serving
checkout's runtime was restored to unmodified main for fresh references.
Patch and logs: `/jobtmp/gemma4-staggered/full-specialization-reviewed.patch`,
`full-specialization-test-{before,after}.log`, and
`full-specialization-dbo-before.log`.

FULL versus eager one-token controls in the same FP8 process retain the first
token for all 58 q80 and all 58 q800 requests, but all compared top-20 logprob
records differ; maximum shared-token logprob deltas are 3.19 and 2.05 nats.
These changes still require explanation and task-quality evaluation. The live
probe also observes different embedding RMSnorm reduction blocks across TP
ranks: 8192 on ranks 0/3/4/7 and 2048 on ranks 1/2/5/6. This strengthens the
autotuning lead without proving the cause of the historical cross-launch flips.

Offline quality preparation checked all 1,319 GSM8K five-shot user prompts.
The checkpoints have different chat-template files, but every rendered prompt
matches with thinking disabled; tokenizer JSON and generation settings also
match. Evidence: `/jobtmp/gemma4-staggered/quality-template-parity.json`.
Full task-quality and numerical-quality acceptance remain pending.

The active validation sequence is
`/jobtmp/gemma4-staggered/run_candidate_sequence.py`:

1. Unmodified-main FP8 and BF16, FULL_AND_PIECEWISE/512, C4/C13/C57, and
   256-question five-shot GSM8K with the prefix cache reset before evaluation.
2. After both reference servers exit successfully, apply the reviewed scale
   layout and FULL-specialization patches.
3. Patched FULL1024 BF16 and FP8, reversed precision/concurrency order, same
   workload plans and GSM8K protocol.

Outputs are `default-v2-ref-{fp8,bf16}/` and
`specialized-full1024-v2-{bf16,fp8}/`. This is initial candidate validation,
not the required three-restart acceptance campaign or full 1,319-item evaluation.
The agreed goal remains active.

Both fresh default reference arms have now completed successfully. Their
single-launch, matched V2 measurements are:

| C | Workload | BF16 warm p50 / p95 | FP8 warm p50 / p95 | FP8 warm decode p50 |
| --- | --- | ---: | ---: | ---: |
| 4 | warm80 | 32.55 / 42.57 ms | 32.96 / 42.45 ms | 6.194 ms/token |
| 4 | mixed | 33.21 / 59.87 ms | 33.06 / 85.17 ms | 6.256 ms/token |
| 13 | warm80 | 35.02 / 47.03 ms | 34.74 / 49.23 ms | 7.426 ms/token |
| 13 | mixed | 35.06 / 64.69 ms | 35.18 / 89.38 ms | 8.475 ms/token |
| 57 | warm80 | 58.79 / 90.89 ms | 60.74 / 98.71 ms | 12.596 ms/token |
| 57 | mixed | 63.92 / 211.58 ms | 70.20 / 197.50 ms | 16.604 ms/token |

This supplies a matched C57 decode reference: plain FULL's 13.94 ms/token is
about 11% slower than default FP8's 12.60 ms/token, rather than the larger
difference suggested by the historical burst protocol. Preserve both protocols
and use the matched reference for the guardrail.

Default BF16 and FP8 each score **251/256** on five-shot GSM8K, with matching
correctness on every item; strict-marker scoring is 248/256 for each. Neither
has request errors or truncated responses. This is initial reference evidence,
not candidate accuracy or the full 1,319-question evaluation. Paired report:
`/jobtmp/gemma4-staggered/quality-default-bf16-vs-fp8.json`.

The patched BF16/FP8 GPU sequence is now active. The isolated patched checkout
also passes all **32 manager CPU tests**, with CUDA uninitialized throughout;
log: `/jobtmp/gemma4-staggered/full-specialization-clean-main-test.log`.
The comparator scripts `compare_serving.py` and `compare_quality.py` validate
workload/protocol identity before reporting paired evidence and do not decide
goal completion.

A separate numerical intervention in `run_reduction_control.py` and
`reduction_probe_worker.py` has completed successfully in one loaded model on
uncaptured q2048 requests. It compares original→2048→8192→original, counts actual
kernel calls, and verifies exact restoration. The causal output-sensitivity
result and its remaining correctness limits are recorded in the numerical
section below. Its first attempt stopped before activation because GC-only
discovery missed vLLM's frozen objects and a cached kernel hash was specific to
rank zero. The successful rerun uses module inventories and source-validated
per-rank identities. It ran separately from serving timing.

### Initial FULL1024 candidate: completed gates and counterexamples

All four arms of the reference/candidate sequence completed with exit zero.
The validated comparator retains exact per-case workload identities:
`/jobtmp/gemma4-staggered/serving-comparison-initial-candidate.{json,md}`.

| C | Workload | FP8 candidate warm p50 / p95 | Ratio to default BF16 p50 / p95 |
| --- | --- | ---: | ---: |
| 4 | warm80 | 33.19 / 39.69 ms | 1.0199 / 0.9324 |
| 4 | mixed | 33.80 / 47.29 ms | 1.0176 / 0.7899 |
| 13 | warm80 | 35.74 / 48.23 ms | 1.0206 / 1.0256 |
| 13 | mixed | 36.70 / 54.79 ms | **1.0468** / 0.8469 |
| 57 | warm80 | 57.36 / 87.59 ms | 0.9758 / 0.9637 |
| 57 | mixed | 61.95 / 186.18 ms | 0.9691 / 0.8799 |

The C13 mixed median exceeds the approved 3% band. An additional matched-profile
counterexample matters: C57 mixed p95 is 186.18 ms FP8 versus 116.72 ms BF16
when both use patched FULL1024 (59.5% slower). Keep both comparisons; a favorable
default-BF16 comparison alone cannot establish an all-cases win. Coverage above
1024 is the next targeted test for multiple-prefill batches.

FP8 C57 warm decode is 12.508 ms/token, recovering the 12.596 ms default
reference from plain FULL's 13.94 ms. All measured decode p50/p95 and isolated
long-prefill guard ratios stay within 3% of default FP8 in this initial run.
Isolated q800 TTFT ratio is 0.5746 at p50 and 0.5734 at p95, about 43% lower.
These are single-launch points; three-restart confidence evidence is pending.

Candidate FP8 scores 251/256 established and 248/256 strict on GSM8K, with
zero paired correctness changes, errors, or truncations versus default FP8.
Candidate BF16 scores 250/256 and 247/256, losing question 235 (10→11) versus
default BF16. Reports `quality-fp8-default-vs-specialized.json` and
`quality-bf16-default-vs-specialized.json` preserve per-item references and
paired uncertainty. The full 1,319-item evaluation remains required.

The independent FULL2048 comparison was started at the same C4/C13/C57
workloads and initial 256-item quality protocol, using unchanged candidate
source. Its subsequent FP8 results are recorded below. Its expected capture count is 230 (147 mixed plus 83 decode). Artifacts:
`/jobtmp/gemma4-staggered/specialized-full2048-v2-{fp8,bf16}/`.

Future Nsight captures must use the all-warm `measurement_ready` hook and
validate capture timestamps against the sustained measurement window.

### Remaining warm80 tail: serving timing boundaries

The exposed per-request `time_to_first_token_ms` measures SCHEDULED to the
core's first output-batch timestamp, not the complete client TTFT. In the first
V2 FP8 512-cutoff warm block, client p95 is 48.87 ms, this metric's p95 is
22.08 ms, and the per-request residual's p95 is 27.10 ms. These separately
computed percentiles are not additive. Scheduler queue p95 is about 0.011 ms.

Source audit identifies an important unmeasured wait: QUEUED is stamped only
when `Scheduler.add_request` admits the request. The core input thread can
receive/hash a request and leave it in `EngineCoreProc.input_queue` while the
main loop executes an earlier step or waits for its result. That pre-admission
wait is excluded from both reported scheduler queue time and scheduled-to-first
token time. The residual also includes HTTP/input work, output IPC/frontend
processing, transport, and client parsing; do not label it all frontend time.

This benchmark requests token IDs, and the first SSE contains the full prompt
ID list. Its client timestamp is taken after JSON parsing. That overhead is
common to the paired protocol but requires a separate control before extending
these absolute timings to production chat requests.

If FULL leaves warm tails, add buffered request-correlated measurements of
core input-enqueue to scheduler admission, frontend first-token latency,
SSE yield, and client receipt/JSON parsing under unchanged arrivals. Keep
clock domains and overlapping intervals explicit. The existing merged-prefill
M171 observation is only one step per precision (21.791→21.621 ms complete
forward); neither mixed capture contains M250, so they cannot resolve all
merged-prefill tail behavior.

### FULL1024 to FULL2048: completed pair and warm-TTFT decomposition

Completed snapshot: 2026-10-27. Both FULL2048 arms finished with exit code zero.
The comparison validator marks all four default/candidate arms READY for
descriptive comparison. This validates artifact/protocol completeness, not
performance or model-quality acceptance. Three independent restarts remain
pending.

These runs use the same candidate source, model, client plan, worker, helper,
and recorded environment. The saved four-arm campaign provenance records
HEAD `274a5e19a18460ed41d47117398494b721e3df16`, refreshed main at campaign
start. The earlier isolated/Nsight study retains its separate revision above;
the installed package version label is not the current source identity.
The harness change adds only isolated-primer cache assertions before serving. Concurrency order differs: C57/C13/C4 for FULL1024
and C4/C13/C57 for FULL2048; precision order also differs. Each configuration
has only one launch. Coverage, launch variation, autotuning, arrival phases,
and order effects are not isolated by this comparison.

FP8 warm-request summaries, FULL1024 → FULL2048; milliseconds throughout:

| Concurrency | Case | Client TTFT p50 | Client TTFT p95 | Decode p50/token |
| --- | --- | --- | --- | --- |
| 4 | warm | 33.194 → 33.645 | 39.689 → 44.610 | 6.184 → 6.131 |
| 4 | mixed, warm requests | 33.797 → 33.032 | 47.291 → 46.834 | 6.223 → 6.200 |
| 13 | warm | 35.745 → 35.056 | 48.231 → 47.093 | 7.435 → 7.388 |
| 13 | mixed, warm requests | 36.701 → 35.449 | 54.791 → 56.155 | 7.772 → 7.761 |
| 57 | warm | 57.364 → 61.735 | 87.586 → 105.976 | 12.508 → 12.483 |
| 57 | mixed, warm requests | 61.946 → 66.432 | 186.178 → 190.459 | 15.334 → 15.254 |

For each request, define the residual as client TTFT minus that request's
scheduled-to-first-output metric. The following percentiles are computed
separately and **must not be added**:

| Case | Measured warm requests | Scheduled-to-first p95 | Residual p95 | Scheduler queue p95 |
| --- | --- | --- | --- | --- |
| C4 warm | 84 per run | 19.474 → 19.807 | 20.041 → 25.000 | 0.0079 → 0.0071 |
| C13 warm | 273 per run | 23.488 → 26.210 | 27.353 → 25.970 | 0.0113 → 0.0108 |
| C57 warm | 1,197 per run | 33.577 → 37.804 | 55.561 → 68.379 | 0.0192 → 0.0196 |

C57 warm's matched request-plan **mean** client TTFT increases by 5.213 ms.
The same-request engine interval increases by 0.818 ms and the residual by
4.396 ms. This is an interval-mean identity, not independent-percentile
addition or a frontend attribution. In each run's own worst 5% of client
requests (60 requests), mean engine/residual intervals are 37.535/71.551 ms
and 40.342/79.989 ms respectively. The cohorts differ between runs.

C4 warm's mean increase is also mainly residual: +2.714 ms client, +0.183 ms
engine interval, +2.531 ms residual. C13 warm instead improves overall while
its engine interval rises slightly. C13 mixed query80 has a different pattern:
engine p95 rises from 22.395 to 26.643 ms while residual p95 falls from 28.716
to 27.332 ms. Therefore a single residual-only explanation does not cover all
remaining cases.

Actual arrivals include background work within each sustained measurement
window. C57 warm arrival rate decreases from 39.868 to 39.713 requests/s
(−0.39%), and average in-flight requests stay at 56.986/56.985. Mean requested
output lengths stay at 109.853/109.878 tokens. Higher aggregate arrival rate
does not explain this cross-run tail increase; arrival clustering and preceding
step shapes can still differ. C4/C13 warm rates rise only 0.52%/0.96%.

Client/residual correlations are high, but they are algebraically coupled
because residual = client − engine. They identify an interval to investigate,
not a CPU component or a causal mechanism. The existing QUEUED boundary
excludes core input-queue admission blocking; small scheduler queue times
cannot rule that out.

If common GPU shapes and modes remain unchanged, the next diagnostic should
record core ADD input enqueue → `Scheduler.add_request` admission, correlated
by request ID and overlapping engine-step ID. Record preprocessing separately;
buffer events and dump after the case. Both endpoints must use the same core
process clock. Only if that wait fails to explain the residual should the
experiment add frontend output and client receipt/parsing markers. Large
traces are not the first step.

FULL2048 FP8 also completed its 256-question five-shot GSM8K screen: 251/256
established answers, 248/256 strict final-marker answers, no truncations, all
responses stopped normally. This screen does not replace the full 1,319-item
paired evaluation or explain historical cross-launch numerical differences.

The completed FULL2048 precision comparison gives these FP8/BF16 client
warm-request TTFT ratios. A ratio above 1 is slower:

| Concurrency / case | Against default BF16 p50 / p95 | Against FULL2048 BF16 p50 / p95 |
| --- | --- | --- |
| C4 warm | 1.0337 / 1.0480 | 0.9502 / 0.9378 |
| C4 mixed, warm requests | 0.9946 / 0.7823 | 0.9468 / 0.9495 |
| C13 warm | 1.0009 / 1.0014 | 0.9572 / 1.0241 |
| C13 mixed, warm requests | 1.0111 / 0.8680 | 0.9460 / 0.9628 |
| C57 warm | 1.0501 / 1.1660 | 1.0209 / 1.2043 |
| C57 mixed, warm requests | 1.0393 / 0.9002 | 1.0471 / 0.8887 |

C4 warm, C57 warm, and C57 mixed median retain point estimates outside the
approved 3% band against default BF16. With equally configured FULL2048
servers, C57 warm p95 is 105.976 ms FP8 versus 87.995 ms BF16 (+20.43%).
C57 mixed p95 is 190.459 versus 214.305 ms, favoring FP8; this reverses the
equally configured FULL1024 mixed-tail counterexample. The reversal is a
reason to replicate and decompose the measurements, not to select a favorable
configuration or declare the tail problem solved.

Against default FP8, isolated q800 ratios are 0.5738/0.5673 at p50/p95,
about 43% lower. Long-prefill and decode point guards remain within 3%; these
single-launch observations do not complete the restart requirement.

The completed paired quality screens have zero observed correctness
discordances for default versus FULL2048 in both precisions: 251/256
established and 248/256 strict answers per arm, no truncations. Accuracy delta
is zero and exact McNemar p=1. The empirical paired bootstrap collapses to
[0,0] because there are no observed discordances; this does not establish
quality equivalence or exclude unseen failures. Full 1,319-question evaluation
and the historical numerical-difference explanation remain open.

A fixed-PID CPU observer collected 242 one-second samples during the BF16
arm and stopped automatically when its completion status appeared. It used
only the controller/client, API, EngineCore, and eight logged worker PIDs;
there was no process-tree scan, NVML, CUDA, or HTTP request. CPU percentages
below are process means; 100% represents one logical CPU:

| BF16 serving case | Samples | Client CPU mean | API CPU mean | EngineCore CPU mean |
| --- | --- | --- | --- | --- |
| C4 warm | 15 | 2.8% | 10.9% | 102.1% |
| C4 mixed | 19 | 2.5% | 10.2% | 101.8% |
| C13 warm | 21 | 6.4% | 21.9% | 101.7% |
| C13 mixed | 24 | 6.0% | 20.6% | 101.5% |
| C57 warm | 34 | 16.9% | 54.6% | 101.6% |
| C57 mixed | 42 | 14.0% | 44.3% | 101.3% |

C57 warm API/client maxima are 62%/19%. These averages do not show sustained
API/client single-core saturation. EngineCore and workers stay near one busy
CPU each; polling or waiting can contribute, so this alone does not establish
a compute bottleneck. One-second averages cannot rule out millisecond stalls,
event-loop bursts, or core pre-admission blocking. The first sample is 2.636 s
into C4 warm, leaving its initial segment unobserved; all subsequent serving
windows are covered. The observer makes this arm diagnostic evidence; future
acceptance should omit it unless its inclusion is justified.

Artifacts:

- `/jobtmp/gemma4-staggered/serving-comparison-full2048.json` and `.md`:
  validated four-arm point comparisons and provenance.
- `/jobtmp/gemma4-staggered/quality-{fp8,bf16}-default-vs-full2048.json`:
  paired screen statistics and original per-item references.
- `/jobtmp/CPU-samples2048-bf16.summary.json`: window-correlated CPU samples,
  coverage, and source hashes.
- `/jobtmp/gemma4-staggered/specialized-full1024-v2-fp8/` and
  `/jobtmp/gemma4-staggered/specialized-full2048-v2-{fp8,bf16}/`: saved requests,
  windows, launch/source provenance, statuses, and quality outputs.
- `/jobtmp/coverage-warm-decomposition.json`, `.md`, and `.py`: per-request
  residuals, conditional tails, paired plan checks, actual arrivals,
  clustering diagnostics, and source hashes.
- `/jobtmp/CPU-samples2048-bf16.jsonl` and `.observer.json`: a lightweight
  fixed-PID CPU observer ran from 23:11:11.836 to 23:15:14.837 UTC during
  the BF16 arm and stopped on completion. These are diagnostic measurements;
  future acceptance should omit the observer unless its inclusion is justified.

### Request-boundary diagnostic and output-loop control

Both arms of `admission-full1024-v1` completed on the same candidate FULL1024
source, with default output-processing chunk size 128, C13/C57 warm and mixed
workloads, and two repetitions. Fresh per-process identities, clock metadata,
client plans, and all surviving frontend events validate. However, **full core
analysis fails**: timeout-zero abort shutdown killed each EngineCore before its
buffered JSON serialization completed. Exit zero did not prove probe coverage.
Core enqueue/admission is unavailable; raw failure evidence is retained.

The explicit frontend-only report joins every one of 22,923 FP8 and 22,905
BF16 events into 7,641 and 7,635 request chains respectively, including setup,
warmup, and background traffic. Measured warm-only examples:

| C57 warm | FP8 client p50 / p95 | BF16 client p50 / p95 |
| --- | ---: | ---: |
| Repeat 0 | 70.54 / 120.70 ms | 69.13 / 106.53 ms |
| Repeat 1 | 67.20 / 115.85 ms | 68.29 / 113.57 ms |

For FP8 repeat 0, frontend first-output to first-token SSE handoff is
17.79 / 45.08 ms p50/p95. Native engine-output timestamp to frontend receipt is
3.50 / 14.53 ms; client arrival to frontend ADD is 6.85 / 29.43 ms. The probe's
extra server JSON parsing is only 0.512 / 0.575 ms, and client JSON parsing is
about 0.5 ms. The first-token SSE is about 46 KB because prompt IDs are returned.
The matching frontend-to-handoff means are similar in both precisions, about
20–22 ms: this is not an established FP8-specific mechanism.

These are independent component quantiles and must not be added. The JSON
contains exact same-request closure, additive means, and conditional worst-5%
means. Native output timestamps are copied from the frontend's received batch;
ADD-to-native-output combines the entire core path and does not isolate input
queueing. An inferred SCHEDULED stamp is explicitly not an observed admission
stamp. Instrumented TTFT shifted substantially from the uninstrumented FULL1024
run despite stable decode, so these logs cannot establish the original
regression's cause or a performance acceptance result.

Source inspection supplies a testable fairness hypothesis. At
`vllm/v1/engine/async_llm.py:_run_output_handler`, the default
`VLLM_V1_OUTPUT_PROC_CHUNK_SIZE=128` processes a 57-output batch without an
inter-chunk `asyncio.sleep(0)`. If the next output batch is already queued, its
async get need not suspend, so ready per-request streaming tasks can wait behind
more output processing. This possibility is not proof of its observed cost.

`admission-chunk8-v2-retry` now compares BF16 then FP8 with the existing chunk
setting at 8, keeping FULL1024, prompts, sampling, and other serving settings.
The new probe adds only a post-drain flush barrier: after all clients finish,
the controller creates `probe/flush.request`, awaits the existing cache-reset
RPC, and requires both completed log files before allowing shutdown. The
measured frontend hooks are AST-identical to V1. Five CPU lifecycle smokes pass;
actual complete-log and request-coverage validation is still required. The
initial metadata-path preflight failed before starting any server and is
retained separately; the corrected fresh-tag launch is live. No production
output-processing code or default has changed.

Artifacts: `admission-v1-frontend-paired.{json,md}`,
`admission-fp8-frontend-final.{json,md}`, `review/admission-probe-review.md`,
`admission_probe_v2/manifest.json`, `run_admission_v2.py`, and
`admission-chunk8-v2-retry-*` under `/jobtmp/gemma4-staggered/`.

### Completed chunk8 decomposition and first-output CPU cost

Both chunk8 arms passed complete clock/source/barrier/request checks. The
post-drain cache-reset barrier writes complete core and API logs before normal
shutdown: 7,636 FP8 and 7,638 BF16 requests join uniquely, with zero overflow and
no pending IDs. Unlike V1, core enqueue-to-admission is directly measured.
At C57 warm, BF16 median/p95 admission wait is about 4.3/11.6 ms and follows the
remaining core-step interval; it is not evidence of GPU, GIL, or GC attribution.

Matched warm TTFT examples, p50/p95 ms:

| Case | FP8 | BF16 | FP8/BF16 p50 / p95 |
| --- | ---: | ---: | ---: |
| C13 warm r0 | 36.80 / 50.03 | 37.49 / 56.06 | 0.981 / 0.893 |
| C13 warm r1 | 36.60 / 51.85 | 37.46 / 53.40 | 0.977 / 0.971 |
| C57 warm r0 | 67.46 / 96.06 | 65.18 / 89.49 | 1.035 / 1.074 |
| C57 warm r1 | 67.88 / 94.30 | 68.44 / 93.26 | 0.992 / 1.011 |

C57 mixed warm p95 favors FP8 in both repeats (202.38/201.33 versus
218.11/217.06 ms). These instrumented points are not acceptance results and do
not replace the default-BF16 reference. Output-count-conditioned native engine
means are mostly faster or similar under FP8. Groups with 3+ co-emitted first
tokens contain 24.4%/26.1% of FP8 warm requests versus 23.2%/20.6% under BF16,
and dominate client-tail cohorts. Frontend-to-handoff means rise from about
11 ms for one first output to 24–26 ms for 3+. These are output cohorts, not
complete scheduled compositions or GPU kernel measurements.

A separate CPU probe uses real CachedGemmaTokenizer, FastIncrementalDetokenizer,
EngineCoreRequest/Output, OutputProcessor, RequestOutputCollector, and response
objects. It performs no model spawn, HTTP request, or CUDA initialization.
Twelve measured batches per cell give 96 measured batches:

| Batch outputs | 0 / 1 / 2 / 3 fresh long-prompt requests: output-processing CPU p50 |
| --- | ---: |
| 13 | 0.090 / 5.423 / 10.859 / 16.305 ms |
| 57 | 0.332 / 5.841 / 11.353 / 16.002 ms |

First detokenizer update p50/p95 is 5.731/5.916 ms, constructor 0.107/0.120 ms,
and subsequent update 0.006/0.010 ms. Response construction/serialization is
about 0.4–1.3 ms per batch. cProfile attributes the dominant self time to native
`DecodeStream.step`. Shared-host CPU measurements establish a concrete
within-batch cost, not a serving win. Tokenizers 0.23.3 lazily decodes the supplied
prompt at the first step; the pinned [upstream stream implementation](https://raw.githubusercontent.com/huggingface/tokenizers/v0.23.3/tokenizers/src/tokenizer/mod.rs)
explains why a large cached model prefix still incurs decoder work.

Simple raw/visible tail truncation is unsafe. Skipped special tokens can hide
pending UTF8 bytes, and one invalid byte can invalidate an entire contiguous
ByteFallback run. A long run with the invalid byte outside a short window
bypasses a naive tail-end check. Native full-prefill behavior can re-emit prompt
text in such cases; a performance change must preserve that behavior rather
than silently fixing an unrelated decoder issue. Priming/discarding cached
prefix text also failed malformed-byte parity and is not accepted generically.

The conservative development candidate only shortens context for the exact
native decoder chain Replace(▁→space), ByteFallback, Fuse. It requires a known
nonempty raw tail of seven non-byte pieces and a nonempty native-decoded tail
not ending U+FFFD; unknown backends, chains, mappings, byte spellings, or unsafe
boundaries keep the full original prompt. All raw pieces starting `<0x` are
rejected, including valid Rust byte spellings such as `<0x+F>`. The original
request and output accounting stay intact. Independent review checks 9,744
native comparisons with zero per-step mismatches and 3,164 reset events.
Frozen implementation validation and the first full serving/quality pilot are
complete below; repeated acceptance and the C57 mixed regression remain open.

The fourth Claude CLI review is retained at
`review/claude-output-fairness-review.{json,txt}`. It correctly requires evidence
of queued output batches before attributing cost to starvation. The three-line
batch-end-yield draft was removed from the development source; its external
patch remains as a historical hypothesis, with no performance claim.

Artifacts under `/jobtmp/gemma4-staggered/`:
`admission-chunk8-paired-full-summary.{json,md}`,
`admission-chunk8-output-batches.{json,md}`, `cpu-frontend-cost.{json,md}`,
`cpu_frontend_cost_probe.py`, `detok-prefill-parity.{json,md,py}`, and
`detok-safe-context-review.md`. Unsafe candidate counterexamples remain in the
parity report. The reusable benchmark is also documented in the rendered
`docs/benchmarking/cli.md` and linked from `benchmarks/multi_turn/README.md`;
Markdown checks pass.

### Frozen detokenizer implementation: CPU validation complete

The final candidate requires the exact native Tokenizer backend and decoder
chain. It inspects the raw final seven IDs without a separate HF special-token
filter. Every retained mapping must be a known nonempty string not starting
`<0x`; native decoding with the actual skip-special flag must be nonempty and
not end in U+FFFD. Otherwise the full original prompt is retained. Native
factory replacements and unknown decoders also retain full context. Custom
decoder serialization can raise a generic exception; the fingerprint check
catches it and falls back, without broadening unrelated exception handling.

Frozen source hashes:

- Candidate `vllm/v1/engine/detokenizer.py`:
  `47992ba31ab7df1ce935f5683fb0abe8f3b1393dd6c7c77cdc25dc66d1f37430`.
- Targeted test file:
  `48d78aaeae84a465729a8a84feca2521d5e0dccd5de6a12b4e5de138f710f220`.
- Immutable original full-prefix implementation:
  `b9e23b5714b80fbccbff3149541f690bc3ff1d41b064f27bfa7fe85b2bc15476`,
  byte-identical to the clean benchmark original.

The clean-main test log explicitly records importing
`/jobtmp/gemma4-fp8-regressions-main/vllm/__init__.py`, with **44 passed,
1 deselected** and CUDA uninitialized before and after. The deselected legacy
test needs `google/gemma-3-1b-it`: that asset is unavailable locally and gated
access returned HTTP 401. Its offline failure also reproduces with the original
implementation. A supplemental replay changes only that asset literal to the
locally cached `google/gemma-3-27b-it`; both original and candidate pass the
remaining original test logic and exact expected French text. This supports
family-tokenizer behavior but does not replace the unavailable 1B-asset test.

Independent production integration imports the actual development module and
loads the immutable original under a distinct module name. The native
DecodeStream factory is never patched. A temporary CPU profile hook observes
the selector return inside each real constructor; source AST checks confirm
that this value is passed directly to `DecodeStream(ids=...)`.

| CPU validation scope | Original/candidate pairs | Per-step checks | Reduced-context constructors | Matching resets |
| --- | ---: | ---: | ---: | ---: |
| 1,427-case matrix plus adjacent-special bonus; both skip/spacing flags | 5,712 | 41,576 | 1,448 | 1,088 |
| All saved prompts; actual skip/spacing flags | 15,274 | 244,384 | 15,274 | 0 |
| Total | 20,986 | 285,960 | 16,722 | 1,088 |

Every emitted delta, complete text, generated-token ID/count, original request
prompt, and per-step reset count matches. All saved prompts are reconstructed
and their uint32-le hashes verified. Direct selector checks for both skip modes
also match the prototype in all 30,548 cases. The observed reduced contexts
rule out a pass caused by silently falling back on every request.

The saved records omit generated token IDs, so their streaming checks use a
deterministic 16-token corpus continuation, not the original model outputs.
Adversarial continuations and both special-token-spacing modes are covered by
the matrix. These finite CPU checks do not establish model quality, unseen
tokenizer behavior, or serving speed. Source/test/reference/prototype/client
hashes remain unchanged through the run. No GPU work or server requests occur
in this validation.

Artifacts under `/jobtmp/gemma4-staggered/`:
`detok-production-parity.{py,json,md}`, `detok-production-reference.py`,
`detok-window-clean-main-tests.log`, `detok-window-development-manifest.json`,
`detok-legacy-baseline-offline.log`, and
`detok-legacy-local-family-replay.json`. The completed serving pilot is launched
by `run_detok_window_pilot.py` and its verified BF16 continuation.

After both pilot servers exited, the unchanged real-object CPU probe ran on
actual clean-main sources with final detokenizer SHA `47992...`. It measures
96 batches, CUDA hidden and uninitialized before/after. Initialization p50/p95
is 0.01557/0.01717 ms; first update is 0.00794/0.01225 ms. C57 output-processing
CPU p50 is 0.262/0.311/0.329/0.338 ms with zero/one/two/three new warm requests.
These are shared-host CPU measurements, not additive serving intervals. They
close the earlier development-version timing caveat without relabeling the
older `e8d28...` report. Artifact: `cpu-frontend-final47992-clean-main.{json,md}`.

### Completed mixed-FA4 bounds control

The single-GPU H100 diagnostic passes **839 validation gates** across 12 cases,
48 reference-checked arms, 288 timing measurements, and 28,800 CUPTI samples.
It models one to three warm80 prefills mixed with decode at C13/C57, using
synthetic BF16 activations and shared physical KV pages. Timings include native
metadata and split/combine GPU kernels, CUDA graphs, and cold L2. Each pair uses
four AB/BA rounds; samples are graph replays, not independent server launches.

Tightening maxQ alone to 80 has negligible latency effect and is bitwise equal
to auto in all 12 cases. Forced splitting has shape-dependent results:

- Global C13 split4 improves p50 by 64–72%, up to 3.56×. Global C57 split2
  improves one/two-prefill cases by 43–44%, but regresses three-prefill latency
  by 3.7%; split4 improves that case by 17.3%.
- Local C13 with one prefill improves 16.9% under split2; the two-prefill case
  changes less than 1%. Other local split2 cases regress; split4 regressions
  reach 97%. A uniform fixed-split policy is not supported.

All arms pass the combined FP32-reference tolerance of atol 0.015 and rtol
0.01, with inputs, guards, and unused output tails unchanged. All 24 forced-split
arms differ bitwise from auto, with maximum absolute difference 0.015625.
No ULP bound, downstream-logit equivalence, model-quality result, or serving
benefit follows from this check. A later global-only model diagnostic would
need matched FP8/BF16 batch composition and evaluation of changed outputs;
the remaining mixed-tail attribution takes priority over new split policies.

Artifacts: `/jobtmp/gemma4-staggered/fa4-mixed-bounds-summary.{json,md}`,
`fa4-mixed-bounds-results-v1.json`, and `fa4-mixed-bounds-plan-v1.json`.

### Completed safe-detokenizer serving pilot

`detok-window-full1024-pilot-v1` uses clean main `274a5e19` plus the frozen TMA,
narrow-decode FULL-graph, and guarded detokenizer patches. Both arms use
FULL1024, batch budget 32768, output chunk size 128, C4/C13/C57, turns24,
warmup3, repeat1, identical request plans and no profiler. FP8 ran first. Its
controller terminated after successful FP8 shutdown because a GPU context
briefly lingered; BF16 continued only after verifying that controller terminal,
GPU clearance and every original frozen source hash. No FP8 data was repeated
or discarded. The continuation and its additional source are archived.

The completed four-arm comparator validates native-main FP8/BF16 references
and both optimized arms. READY means protocol validation, not goal acceptance.
Warm TTFT excludes cold requests but includes q800 in the mixed case.

| C | Case | FP8 TTFT p50 / p95 ms | Optimized BF16 p50 / p95 ms | FP8/BF16 p50 / p95 |
| --- | --- | ---: | ---: | ---: |
| 4 | Warm | 26.26 / 30.89 | 28.12 / 32.65 | 0.934 / 0.946 |
| 4 | Mixed warm | 26.49 / 38.74 | 28.04 / 40.31 | 0.945 / 0.961 |
| 13 | Warm | 28.35 / 36.26 | 30.17 / 37.11 | 0.940 / 0.977 |
| 13 | Mixed warm | 29.10 / 42.74 | 30.64 / 46.75 | 0.950 / 0.914 |
| 57 | Warm | 43.70 / 56.10 | 44.00 / 57.23 | 0.993 / 0.980 |
| 57 | Mixed warm | 47.58 / 174.75 | 45.52 / 90.57 | **1.045 / 1.930** |

All cells pass against native-main BF16. The equally optimized BF16 comparison
retains the C57 mixed failure, so the all-cases objective is unmet. C4 has only
84 warm-case and 80 mixed-warm observations; C13 has 273/262, C57 1197/1160.
These single-launch points do not replace repeated acceptance.

Against native-main FP8, isolated q800 p50/p95 is 36.36/37.02 ms versus
76.16/78.20 ms: reductions 52.26%/52.66%. Isolated q8192 improves 7.29%/8.96%,
q18304 improves 5.96%/6.11%. Decode TPOT p50/p95 passes the 3% no-regression
guard in every cell; the largest increase is 1.19%. Decode p50 remains
3.6–10.3% faster than equally optimized BF16. Isolated controls have only 12
measured observations per query, so their final tail-confidence gate is open.

The C57 mixed tail is not a uniform FP8 slowdown. Warm means are 58.34 ms FP8
and 57.72 ms BF16. Requests above 150 ms number **63/1160 (5.43%)** and
**56/1160 (4.83%)**, respectively: seven requests move the p95 across a sharp
boundary. All 56 slow BF16 requests exceed 200 ms; only five FP8 requests do.
Worst-5% conditional means are 186.34 ms FP8 and 214.51 ms BF16.

Every above-150-ms request overlaps a client-observed cold request's
arrival-to-first-token interval. This is association, not an observed GPU
execution interval. Both windows contain 38 cold arrivals, including one
background request, and at most two outstanding cold requests. FP8's window
is shorter, 40.11 versus 42.25 seconds, with 5.33% higher arrival throughput.
For q80, native server-TTFT p95 is nearly identical (44.25/44.12 ms), while
client p95 is 175.30/87.77 ms. Other slow requests have large native-engine
intervals too; the residual cannot all be labeled frontend cost. Scheduler
queue time still excludes waiting before admission. Exact step/admission
tracing is the next diagnostic.

The reversed-order `detok-shape-admission-v1` diagnostic completed
BF16 then FP8 on the unchanged FULL1024 candidate. It combines the unchanged
validated V2 admission plugin/client with the existing StepProbe and read-only
NumericsProbe, recording exact request/query/context/dispatch shapes on all
eight workers: 6,372 BF16 and 6,386 FP8 steps per rank match exactly. It uses
C57 warm/mixed, turns24, repeat1, chunk128 and the explicit flush barrier.
Both complete admission chains validate. These instrumented timings are
diagnostic, not final acceptance. Mixed p95 reverses the pilot's ordering:
FP8 103.52 ms versus BF16 190.45 ms; median FP8 remains 50.97 versus 48.50 ms.
The original failing pilot remains retained.

Exact internal-request-ID joins identify three paths:

- Warm first forwards share a q8192 cold batch: 19 FP8 and 28 BF16 measured
  warm requests. All such forwards are uncaptured. Native engine means are
  164.38 and 192.31 ms, respectively.
- Warm arrivals wait for admission during a core completion step tied by the
  cold request's output ID to its actual worker dispatch: 27 FP8 and 24 BF16
  requests. Every admission interval above 100 ms attaches to this group.
  Remaining core-step time and admission wait closely agree: FP8
  140.641/140.662 ms, BF16 165.794/165.817 ms. Core and worker counters are not
  equated; two FP8 requests overlap the first two groups.
- Seven requests per arm have no cold request in their own forward but spend
  over 100 ms between worker host exit and native output. Their immediately
  preceding nonempty worker dispatch contains q8192. This is consistent with
  waiting after host submission; it is not a measured GPU duration.

The diagnostic also observes uncaptured non-cold batches at M1090–1813.
For M1093 (53 decodes, three q80 and one q800), native means are 73.58 ms FP8
versus 51.94 ms BF16. These rare cohorts do not explain the whole median gap:
the no-cold FULL cohort remains 3.4% apart. Host execute and output intervals
must not be relabeled GPU spans. Exact reports are
`detok-shape-admission-v1-exact-join.{json,md}` and
`detok-shape-admission-v1-exact-join-findings.md`.

The bounded Nsight companion captured BF16 successfully, but its initial
validator rejected an overly strict profiling-RPC return bound before FP8
started. Native timestamps verify eight worker starts at callback+1.006 s and
first native stop at +8.014 s, about 7.008 seconds of capture. GPU activity ends
by +8.024 s, while the stop RPC returns at +19.389 s. That teardown pause is
observer overhead, not a 19-second GPU capture or model slowdown. An external
corrected validator and FP8-only continuation now complete successfully. FP8's
native capture is 7.037 seconds; its stop RPC returns at callback+21.079 seconds.
Both arms pass native capture, full context/request joins and eight-rank checks.
The frozen original package and failure log remain unchanged. Raw graph-trace
tables also contain pre-capture rows, so attribution requires actual NVTX/API
correlations. Reports: `nsys-v2-{bf16,fp8}-revalidated-v3.json`; FP8 continuation
status: `nsys-mixed-tail-fp8-resume-v3-status.json`. GPU correlation analysis is
separate from successful capture validation.

The next **unattached** pilot, `threshold2048-full4096-pilot-v1`, completed:
FP8 then BF16, with the original batch budget32768, a public per-request long
prefill threshold2048, adaptive threshold disabled, FULL4096 and chunk128.
Runtime source is still the same three patches. The copied harness only adds
the threshold flags; the new worker expands the existing graph-limit control
to4096 and verifies effective configuration on all eight ranks. No tracing
hooks or profiler run during serving. The plan retains isolated/eager controls,
C4/C13/C57 warm/mixed, full1319 per precision, and explicit mixed cold TTFT and
whole-request costs. Two simultaneous cold chunks can still exceed4096;
the resulting cold costs and median counterexample are recorded below.
Plan v4 and prior versions are retained
under `threshold2048-full4096-pilot/`. Independent review fixed both new
completion/readiness checks to require clean exit0 before execution. Comparisons
against older worker snapshots retain their mismatch flags; this is not final
acceptance or an assertion that all batch shapes are captured.

### Threshold2048 / FULL4096: completed tradeoff, goal not met

The two-arm pilot exits cleanly with unchanged runtime/client/worker sources.
Effective settings validate on all eight ranks. Actual final FP8 capture is
358 graphs, 62 seconds, 2.12 GiB/rank; the preliminary profiling capture is
separate. Available KV memory is 56.27 GiB (491,603 tokens). These are measured
resource costs, not the earlier static allocation estimate.

| C | Case | FP8 TTFT p50 / p95 ms | BF16 p50 / p95 ms | FP8/BF16 p50 / p95 |
| --- | --- | ---: | ---: | ---: |
| 4 | Warm | 26.29 / 32.41 | 28.07 / 32.47 | 0.936 / 0.998 |
| 4 | Mixed warm | 26.56 / 39.36 | 28.15 / 40.54 | 0.943 / 0.971 |
| 13 | Warm | 28.66 / 37.28 | 30.19 / 37.46 | 0.949 / 0.995 |
| 13 | Mixed warm | 28.93 / 47.52 | 30.49 / 46.78 | 0.949 / 1.016 |
| 57 | Warm | 44.28 / 57.53 | 44.38 / 58.38 | 0.998 / 0.985 |
| 57 | Mixed warm | 49.52 / 117.04 | 46.00 / 131.56 | **1.077** / 0.890 |

Isolated q800 FP8 remains 36.62/37.04 ms p50/p95. Isolated q8192 is 178.90/180.49
and q18304 is 395.50/396.48 ms, near the previous candidate's controls. However,
concurrent cold TTFT degrades despite the isolated-request exemption:

| C | Prior FP8 cold p50 / p95 ms | New FP8 cold p50 / p95 ms | New BF16 cold p50 / p95 ms |
| --- | ---: | ---: | ---: |
| 4 | 156.19 / 156.63 | 193.75 / 194.41 | 215.68 / 220.26 |
| 13 | 159.82 / 168.92 | 200.19 / 209.87 | 220.60 / 230.16 |
| 57 | 178.07 / 223.37 | 228.06 / 265.33 | 265.25 / 391.75 |

Cold samples are only 4/11/37. Retain both quantiles and whole-request metrics;
do not replace this tradeoff with a claim based only on isolated long-prefill
guards. Both precisions score 1277/1319 on GSM8K, strict 1258 FP8/1256 BF16,
without errors/truncations. FP8 aggregate quality remains one answer above its
native baseline; paired per-question review remains separate.

The completed saved-quality review validates all1319 prompt/payload hashes and
protocol records per arm. FP8 has exactly the prior FULL1024 candidate's
per-question correctness: two established wins and one loss against native FP8.
BF16 has one win and two losses against native BF16. All13 scored-discordant
questions/references and complete outputs across six arms were reviewed; visible
changes concern reasoning, quantities, units, conventions or answer formatting.
No visible corruption or truncation was found in those reviewed items. However,
265 FP8 and149 BF16 full texts change against the prior pilot, including226/131
completion-count changes. Unchanged scores do not establish numerical equivalence
or a sole cause. Artifacts: `threshold4096-quality-review.{json,md}` and
`quality-full1319-{fp8,bf16}-{native,detok1024}-vs-threshold2048-full4096.json`.

The expanded comparator validates actual worker snapshots. The new paired
configuration is comparable, but older native/control references retain
worker/common-command mismatch flags. A descriptive cross-launch table does not
waive those flags or the missing restart gate. Artifact:
`serving-comparison-threshold2048-full4096-pilot-v1.{json,md}`.

The subsequent `threshold2048-full4096-maxseq64-screen-v1` is **FP8 only**, with
`quality_count=0` and unchanged measured plans/controls. Its single serving-setting
delta is `--max-num-seqs 64`; it leaves the 57-client workload below the service
capacity and preserves decode57's padded64 shape. Mixed graph metadata now
pads at most 64 requests. The flag also reduces persistent state, dummy/profile
work, compile/autotune choices and known penalty-state VRAM by 990 MiB/rank;
it is not a pure padding-only intervention. Live LM-head/sampler rows remain
the real request count. The lower service capacity is an explicit tradeoff,
and paired BF16/full quality/restarts are still required. Plan/source proof and
the unchanged shared worker are frozen under
`threshold2048-full4096-maxseq64-screen/`. The controller completed with exit0,
unchanged sources and idle GPUs. Measured FP8 warm TTFT follows:

| C | Warm p50 / p95 ms | Mixed warm p50 / p95 ms | Mixed cold p50 / p95 ms |
| --- | ---: | ---: | ---: |
| 4 | 26.31 / 32.69 | 26.61 / 39.35 | 189.76 / 199.49 |
| 13 | 28.48 / 37.80 | 28.89 / 46.97 | 194.81 / 209.20 |
| 57 | 44.05 / 56.67 | 47.56 / 95.01 | 224.51 / 381.35 |

Isolated q800 is36.95/37.27 ms, q8192 is179.15/181.80 and q18304 is396.36/397.68.
The C57 mixed warm median/tail improve descriptively against the preceding FP8
launch, but cold costs remain and C57 cold p95 worsens. Separate launches with
one repeat cannot isolate padding causality or establish repeatability.

The next screen is under
`threshold0-full32768-maxseq64-screen/`. It removes the long-prefill cap,
retains maxseq64 and the32768-token batch budget, and explicitly captures331
mixed plus11 narrow decode graphs. Above1024 tokens, buckets are spaced128
tokens apart through32768. The intent is to cover large mixed batches without
splitting a cold prefill into repeated forwards. It launched from frozen plan v2,
SHA256 `928c2e7f862a817e012ef56501c631fa6f8b7cbdee7be4050de675274585c5d1`,
after an idle-GPU check and independent read-only source review. It retains
the serving/isolated controls
and adds a separately labeled same-loaded-model eager isolated comparison.
No BF16, full-quality or restart acceptance is implied by this FP8-only screen.

The screen completed with exit0, unchanged sources, idle GPUs and no arm/case
issues from the frozen expanded comparator. All eight ranks validate the331
mixed plus11 narrow descriptors, effective threshold0 and requested padding
boundaries. Actual graph capture takes104 seconds and4.34 GiB/rank; the separate
profiling capture takes2.94 GiB. Available KV memory is54.19 GiB,473391 tokens.
These measured costs exceed the earlier4096 pilot's2.12 GiB final graph pool.

| C | Warm p50 / p95 ms | Mixed warm p50 / p95 ms | Mixed cold p50 / p95 ms |
| --- | ---: | ---: | ---: |
| 4 | 26.31 / 33.70 | 26.59 / 38.74 | 159.20 / 163.20 |
| 13 | 28.63 / 39.55 | 29.17 / 46.77 | 159.89 / 163.67 |
| 57 | 44.42 / 57.21 | 47.23 / 88.53 | 177.92 / 215.45 |

Isolated FULL q800/8192/18304 medians are38.00/178.96/394.76 ms. Separately
ordered, same-loaded-model eager controls give68.41/180.38/400.79 ms. All45 paired
one-token output hashes match, including warmups; this is not full-sequence
quality proof or a randomized timing comparison. Per-arm validation artifact:
`threshold0-full32768-maxseq64-screen-v1-validation.json`. Older launch/config
mismatch flags remain applicable to cross-configuration comparisons.

The matched BF16-then-FP8 pilot has now launched under
`threshold0-full32768-maxseq64-pilot/`, from plan v3 SHA256
`3943686d54bd2e44b13756397196ba4ddb56c35eea987221218fdea961a709e1`.
It retains the identical FP8 command and changes only model path for BF16, with
full1319 quality per arm and unchanged serving/isolated/eager controls. Independent
review caught a duplicate archive basename before launch; v3 removes it and
passes the actual harness's24-file uniqueness check. Earlier plans remain
superseded. No paired result or restart acceptance is available yet.

The BF16 arm subsequently completes with exit0 and full1319 quality:1278
established/1257 strict, matching native BF16's aggregate scores. All1319 finish
with `stop`, with zero failures/truncations. The controller is moving to FP8;
per-question review and the matched precision comparison remain pending.

Preliminary BF16 serving measurements in that still-running pilot give C57 mixed
warm46.65/200.74 ms and isolated q80036.50 ms median. The earlier FP8-only
screen's47.23/88.53 ms C57 mixed result is promising, but isolated q80038.00 ms
is about4.1% slower than this BF16 point. These are cross-launch preliminary
observations, not a completed paired comparison; the fresh FP8 arm is pending.
This isolated precision counterexample remains visible even though the approved
q800 baseline-improvement criterion is separately met by the FP8-only screen.

The fresh FP8 arm's isolated controls subsequently measure q80036.18/36.98 ms
p50/p95, versus BF16's36.50/36.79 ms. Thus this fresh arm does not reproduce
the earlier q800 median regression; the38.00 ms screen remains recorded as
cross-launch variability, and no repeatability claim is made yet. Fresh FP8
q8192/18304 medians are176.53/392.29 ms, versus BF16's206.13/462.14 ms.

The new final-campaign tools are being prepared in
`full32768-final-campaign/`; they have not launched or been frozen executable.
They retain the twelve-arm, three-restart, reversed-order design. In addition to
the existing162 performance points, eighteen explicit mixed-cold TTFT points
apply the same1.03 long-prefill bound against native FP8 per concurrency,
quantile and restart. Whole-request cold latency remains descriptive. This
prevents an isolated long-prefill pass from hiding the observed concurrent cold
tradeoff. Exact profile-aware preflight must validate intentional configuration
differences without rewriting older generic comparator mismatch flags.

A read-only comparison of the4096 and32768 maxseq64 FP8 live Inductor
inventories finds the same recorded kernel hashes/names on every rank, but15
selected reduction configurations differ across ranks. Changes span the first
RMS, residual RMS and interlayer RMS families, including2048/4096/8192 reduction
blocks. This is additional evidence that identical generated source does not
imply identical selected reductions across launches. The inventory is best
effort, not an executed-call census or proof of latency/output causality.
Artifact: `full4096-vs32768-maxseq64-inductor-selection.json`.

By contrast, the same-profile32768 FP8 screen and fresh paired pilot have zero
selected-config differences among common recorded kernels. Each screen rank
has one additional `Placeholder.DESCRIPTIVE_NAME` tuner. This inventory does
not explain the screen's38.00 ms versus the pilot's36.18 ms q800 median; no
RMS-autotuning latency cause is inferred. Other backends and runtime conditions
are outside this best-effort inventory. Artifact:
`full32768-screen-vs-pilot-inductor-selection.json`.

Source assessment of a possible later quantization-fusion improvement confirms
that the active dynamic GEMM custom op hides its group128 quantizer from the
outer FX graph and existing RMS-quant fusion pass. Gemma4's standard RMS IR is
otherwise compatible. However, archived active FX passes `use_deep_gemm_e8m0=True`;
the existing fused RMS group-quant replacement has no equivalent explicit
power-of-two scale-rounding argument. Exposing the quantizer must not silently
activate a replacement with different scale semantics. No fusion change or
latency saving is claimed. Artifact:
`gemma4-rms-group128-fusion-source-assessment.md`. This is a source-only
assessment; implementation depends on confirming residual measurements and
validating scale semantics.

Further source work narrows a possible implementation without changing the
running candidate. Archived FX has120 eligible plain-RMS-adjacent dynamic GEMMs
(60 qkv and60 gate_up), rather than all240 activation quantizers. Current
effective compile endpoints are `[48,32768]`. A pattern can leave the entire
dynamic `[1,48]` range untouched and replace only proven-large matched rows
using the existing fused RMS/group128 kernel and existing prequantized
`vllm.fp8_gemm_nt_op`. A compile range alone is insufficient: the matched row
symbol must be the dispatched symbol, or a concrete supported large dimension.
Indexed/MoE rows, extra norm consumers, unsupported layouts and residual fusions
remain outside the proposed initial contract.

The kernel needs an explicit UE8 mode, including the active contiguous CUDA
quantizer's two floors and power-of-two ceiling. Its old fused FP8 minimum scale
differs from that formula; applying a ceiling to the old scale is incorrect for
zero/tiny groups. Both the amax and quantized values must use the same BF16
normalized/weighted value after both rounding steps. The CUDA and fallback Triton
tiny-value formulas differ, so the active CUDA path is the target reference.

An external six-file kernel/schema/wrapper/test prototype is prepared, not
applied: `rms-ue8-cuda-prototype-v1/kernel-interface.patch`, SHA256
`72d798b16f43e0b04fcb08db6ce5ba512775f245d4f7eb988fc15c5ebab4c7d6`.
It proposes default-false UE8 support,64 structured arithmetic/layout cases and
one default-compatibility case in the existing kernel suite. AST/Ruff and
read-only apply checks pass. Native build, all GPU tests, numerical/model
evaluation and timing remain unperformed. The old false-path arithmetic stays
unchanged; the proposed Python wrapper requires the matching native schema.

Source/design artifacts are `rms-ue8-group128-design-v1.md` and
`gemma4-rms-group128-integration-design-v1.{json,md}`. A separate external
compiler-pattern prototype is prepared and held as research. No production or live runtime
file is changed, and no build/GPU work overlaps the restart diagnostic. The
fused CUDA kernel rereads input in three phases and may lose surrounding
Inductor producer fusion;120 potential pairs are not a latency or bandwidth
saving claim. Only measured kernel/forward/serving gains with correctness and
quality validation can select this change.

The upstream overlap check changes the implementation priority. Open
[#63202](https://github.com/vllm-project/vllm/pull/63202) already provides an H100
FP8-input consumer stacked on the normalization contract in
[#61073](https://github.com/vllm-project/vllm/pull/61073). Apply that stack once;
its producer commits must not be duplicated. Open
[#62514](https://github.com/vllm-project/vllm/pull/62514) already provides the
single-pass alternative to the three-read fused kernel, while
[#51477](https://github.com/vllm-project/vllm/pull/51477) disables UE8-unaware
compiler matches. Our independent kernel/compiler prototypes remain unapplied;
the next integration assessment reuses existing work instead of proposing a
competing producer/consumer PR.

Actual RedHatAI compressed-tensors execution already forwards
`QuantizedActivation`, so the candidate reuse path needs only the two Gemma4
plain norm→qkv/gate_up call sites, with the residual highway unchanged. There
are still numerical and performance gates: the upstream consumer's exact test
uses raw-scale mode, whereas our active graph uses UE8; the installed FlashInfer
provider changes rounding/floors, requires residual scratch and materialized norm
output, and may JIT-build. Its H100 shape/layout support does not establish a
drop-in replacement. The upstream author's Qwen results are not our Gemma4/TP8
measurements. See `upstream-fusion-reuse-assessment-v1.{json,md}` and
`upstream-consumer-gemma4-reuse-review-v1.{json,md}` under the artifact directory.

### Reviewability and upstream overlap

A bounded public REST review, completed2026-10-28 04:53 UTC, finds overlap with
all three current production changes. Source snapshots, PR heads, discussion,
reviews and inline comments are preserved in
`public-duplicate-review-v1.{json,md}`. This is not an exhaustive absence-of-
duplicates claim, and no public proposal or comment is posted.

| Current change | Existing work | Disposition |
| --- | --- | --- |
| TMA scale-alignment argument | [#49928](https://github.com/vllm-project/vllm/pull/49928) contains the exact same argument at the same call | Treat this hunk as overlapping upstream work; do not submit a standalone duplicate. |
| Guarded DecodeStream prompt tail | [#51281](https://github.com/vllm-project/vllm/pull/51281), [#57330](https://github.com/vllm-project/vllm/pull/57330) address priming cost | Local native-class/decoder/byte-boundary guards differ materially from32/growing or unconditional128-token tails. Assess contributing safety evidence to the existing effort before any proposal. |
| FULL narrow-decode/mixed ladders | [#61882](https://github.com/vllm-project/vllm/pull/61882), [#63406](https://github.com/vllm-project/vllm/pull/63406), [#56136](https://github.com/vllm-project/vllm/pull/56136) modify capture planning | Inspected peer changes retain different FULL policy; explain composition and the distinct additional narrow-decode routine before any proposal. |

The review follows the repository `pr-checklist` skill's design, validation,
style and contribution criteria. The local workload validates no-spec Gemma4,
not every planner compatibility path. Existing correctness/model/performance
evidence remains useful, but a clean text merge or similar PR title proves
neither general correctness nor novelty. The earlier exact TMA hunk, bundled
serving attribution limits and retained cold-p95 failure remain explicit.
No unmeasured fusion prototype is selected from this source review.

### FULL32768 / maxseq64: six warm cells pass; cold tail still fails

The paired pilot completes both arms with exit0, unchanged sources and idle
GPUs. The frozen generic reader validates all four arm artifacts; comparisons
against historical helpers/configurations retain their generic mismatch flags.
A separate exact-profile preflight validates planned command differences,
runtime reconstruction, helper lineage, request plans, cache accounting and full
quality protocol. Its actual second run passes59 of60 points and fails only
C57 mixed cold p95. A reader manifest-key bug in the first preflight run is
retained with its failure; the fix derives all three runtime hashes from the
approved pilot's pinned absolute source paths rather than the detokenizer-only
development manifest. No performance guard was removed.

| C | Case | FP8 TTFT p50 / p95 ms | BF16 p50 / p95 ms | FP8/BF16 p50 / p95 |
| --- | --- | ---: | ---: | ---: |
| 4 | Warm | 26.38 / 31.71 | 27.77 / 34.03 | 0.950 / 0.932 |
| 4 | Mixed warm | 26.67 / 38.14 | 28.11 / 39.73 | 0.949 / 0.960 |
| 13 | Warm | 28.45 / 38.75 | 30.39 / 38.01 | 0.936 / 1.019 |
| 13 | Mixed warm | 29.10 / 44.48 | 30.90 / 50.57 | 0.942 / 0.880 |
| 57 | Warm | 43.00 / 55.44 | 44.56 / 58.27 | 0.965 / 0.951 |
| 57 | Mixed warm | 46.32 / 88.52 | 46.65 / 200.74 | 0.993 / 0.441 |

All existing54 warm/decode/isolated points pass, including decode advantage
against BF16, decode guard against native FP8, q800's30% baseline-improvement
target and isolated long-prefill guard. Five of six additional mixed-cold
points pass. The remaining counterexample is explicit:

| C | FP8 mixed cold p50 / p95 ms | Native FP8 p50 / p95 ms | Matched BF16 p50 / p95 ms |
| --- | ---: | ---: | ---: |
| 4 | 156.11 / 156.78 | 162.91 / 163.09 | 188.66 / 191.84 |
| 13 | 159.51 / 172.77 | 167.13 / 202.86 | 189.64 / 200.02 |
| 57 | 178.60 / **309.95** | 196.85 / **229.82** | 210.00 / 235.21 |

The C57 cold cohort has37 measured requests. Four exceed300 ms and form two
pairs: `c39t3/c6t3` arrive1.452 ms apart, first tokens0.706 ms apart, and share
287.367801 ms native TTFT; `c10t5/c43t5` arrive5.880 ms apart, first tokens0.450
ms apart, and share305.151650 ms native TTFT. Recorded core queue times are
5–13 microseconds. All57 in-flight requests at those arrivals are measured;
there is no overlapping background prefill in these intervals. The preceding
same-config screen has two cold requests above300 ms, and its p95 is215.45 ms.
Quantile endpoints and timeline differences explain the reported p95 movement,
not the GPU cause or permission to waive the guard. Current client artifacts
lack exact admission/step joins and native clock mapping; earlier Nsight batch
membership cannot be transferred to this run.

Full quality is FP8 1277/1259 established/strict and BF16 1278/1257, all1319 stop
without failures/truncations. FP8 is one/two answers above its native aggregates;
BF16 matches its native aggregates. Per-question and numerical review remain
separate. Artifacts: `serving-comparison-threshold0-full32768-maxseq64-pilot-v1`
JSON/Markdown, `full32768-final-campaign/pilot-preflight-v2.json`, and
`full32768-pilot-c57-cold-tail-v2.{json,md}`. The final restart campaign stays
unstarted while the cold-tail guard fails.

The final reviewed preflight evaluator is SHA256
`34c250c395d3a3435b9a39b748df84f5bc0582c38d457a51b7583f33f17b5d2e`;
`pilot-preflight-v3.json` reproduces the same59/60 failure with unchanged gates.
Real native/candidate reconstruction fixtures validate all three runtime hashes
and reject missing entries or corrupt patch contexts. The final campaign stays
`DRAFT_NOT_EXECUTABLE`; no freeze or launch bypasses this failure.

Saved-quality content review is complete for all14 scored-discordant questions,
full references and all six output variants,84 rows/51 unique texts. It retains
FP8's native-baseline item406 loss and its prior-pilot item590 host-seat loss,
alongside corrected item340 and strict-only percentage formatting gains473/1030.
No gold is relabeled. Against native FP8,424 full texts and363 completion counts
change; equal/better aggregates do not establish equivalent outputs. Artifacts:
`full32768-pilot-quality-review-v1/review-findings.md`, `review-closure.json` and
the complete discordant-text bundle. This closes this pilot's content review,
not launch repeatability or numerical attribution for every changed answer.

The ID-bound scheduling diagnostic completes under
`long-prefill-quota-diagnostic/`. The clean274 checkout has neither historical
partial-prefill CLI knob; cadence throttles all prefills, a smaller scheduled
budget can force chunks, and disabling chunking fails the32768/65536 budget
constraint. Source assessment: `partial-prefill-source-assessment.{json,md}`.

The experimental class inherits the real AsyncScheduler and uses its existing
zero-token defer/continue path to limit delegated proposals above1024 uncached
tokens to one per schedule call. It leaves the first whole8192-token prefill,
short cached80/800 requests and decode work intact. Real CPU Scheduler/cache
fixtures verify warm requests behind the second cold request proceed, and that
the deferred cold request progresses on the next step. The quota is claimed
before KV allocation: failures can consume it and are explicitly recorded.
It is not a successful-allocation count, global in-flight cap or production
compatibility claim. No speculation/lookahead/MM/external-transfer/prefix-replay
support is inferred from this scoped diagnostic.

Two fresh FP8 arms use the same observers and differ only in scheduler class:
an observational AsyncScheduler control and the one-long-prefill subclass.
Both retain FULL32768/maxseq64, threshold0, batch32768 and the identical C57
mixed24-turn plan,1197 measured requests/37 cold. Quality is zero and results
will be diagnostic, not acceptance. Collection checks actual clocks/config,
all eight worker shape streams, flush completeness and clean exit; exact
scheduler/request/admission/worker joins run separately on retained raw files.
Executed plan v3 SHA256:
`447f6dc90ff694e0ddf0942ae9f6d33e4a50fa1c36aa63691927cf0c3c1f0293`.
The controller starts from an idle host, control first then prototype; initial
control API PID1196567. Its source/helper/prototype inputs remain frozen across
both arms, both exit0, and final GPU/server clearance passes.
No production scheduler change or final-campaign freeze is selected here.

### Scheduling diagnostic: no quota intervention was exercised

Both frozen posthoc readers validate1551 client requests,1197 measured requests,
host clocks and all eight worker streams. Control has3172 positive worker
steps and prototype3171; both have46 long-prefill steps, at most one long
prefill per step, zero quota deferrals and zero claimed-but-unscheduled requests.
Thus neither latency change below demonstrates serialization benefit.

| Instrumented C57 mixed | Control p50 / p95 ms | Prototype p50 / p95 ms |
| --- | ---: | ---: |
| Warm TTFT | 51.67 / 88.32 | 50.98 / 174.60 |
| Cold TTFT | 185.32 / 308.51 | 180.24 / 199.25 |

The control's three cold requests above300 ms are already serial: `c0t0` waits
behind `c33t0`, `c6t3` behind `c39t3`, and `c43t5` behind `c10t5`. Exact
enqueue-to-admission waits are137.469/138.483/140.690 ms; own native TTFT is
157.669/156.973/158.933 ms. The predecessor cold's first output occurs within
0.060–0.249 ms of the overlapping core-step end, and admission follows13–18
microseconds later. Own worker batches contain one8192-token cold request plus
54/55 decodes, optionally one80-token prefill, FULL8320/8448 with64 request slots.
This is pre-admission head-of-line waiting plus own service, not a dual-long
batch or pure GPU-time measurement. The untraced pilot's equal287/305 ms native
duration pairs remain separate evidence and are not retroactively reassigned.

Prototype cold pre-admission waits above100 ms drop3→0, while warm requests
overlapping long-prefill service rise50→63 of1160; these account for every warm
TTFT above150 ms and cross the5% tail boundary. Other warm TTFTs remain below97
ms. Both arms share exact measured inputs/output lengths,131407 measured output
tokens, yet closed-loop arrival/overlap placement differs. Their compiler
inventories also contain unresolved duplicate RMS launcher configurations; this
best-effort inventory cannot identify the executed captured launcher.887 measured
output-token hashes change, with no quality evaluation in this diagnostic.
No quota policy is promoted from these two launches, and neither the cold gain
nor warm regression is attributed to deferral.

Artifacts: `quota-{control,prototype}-posthoc-v1.json`,
`quota-control-cold-tail-findings-v1.{json,md}` and
`quota-paired-analysis-v1.{json,md}`. All4091 frozen inputs remain intact.

The C57 mixed cold percentile has only37 measured observations per launch.
With linear quantile interpolation, p95 lies20% of the way from sorted item35
to item36. The same-config screen has two cold requests above300 ms, yielding
p95215.45; the paired FP8 pilot has four, yielding309.95. Both are retained.
Equal payload plans do not impose equal wall-clock arrivals in this closed-loop
benchmark: against native FP8, the candidate's matched request IDs differ in
relative arrival by a median1684 ms and maximum5598 ms. This is an observed
workload-timing difference, not evidence that the failed gate can be discarded.

Even under an independent-sample assumption,37 samples leave a14.99% probability
that their maximum is below the population95th percentile (`0.95**37`). Actual
closed-loop requests are correlated, so this calculation is illustrative and
is not a confidence interval for these runs. More cold observations and
independent restarts are needed to distinguish a stable distributional
regression from phase-sensitive overlaps; the point counterexample and existing
acceptance limits remain unchanged. Exact five-arm source hashes, largest cold
requests and relative-arrival comparison are saved in
`cold-tail-sample-size-assessment-v1.json` under `/jobtmp/gemma4-staggered/`.

### Current cold-tail experiments

The same-profile threshold2048 screen completes with exit0, with the exact current
FULL32768/331+11 graph ladder, max sequences64, max batch32768, three runtime
patches and original client. Only the existing prefill-cap value and a private
read-only worker extension change. The real builtin AsyncScheduler/cache CPU
proof shows cold8192→2048×4 while warm80/800 and decode allocations stay below
the cap; lone8192/18304 requests remain whole. Two consecutive schedules can
queue4096 cold tokens before completion, so the cap is not a wall-clock bound.

The GPU screen collects the existing45 isolated records, then C57 mixed24
turns/3 warmup/one repeat and all-eight-rank raw step shapes. It has no quality
evaluation and is not acceptance evidence. Effective dtype/KV/TP8/FA4 command
lineage, builtin source/configuration, graph counts and worker/clock identities
pass their guards. The separate exact-ID posthoc join now passes across1550
client rows and all eight ranks: every one of37 measured cold requests executes
exactly `[2048,2048,2048,2048]`, summing8192. All1043 warm80 and117 warm800
context sums match their computed prompt tokens. Each of45 isolated queries
executes one whole context step; three reused-ID setup requests are explicitly
audited separately. Raw rank agreement alone would not prove these claims.
Artifacts: `cap-full32768-posthoc-v1.{json,md}` under `/jobtmp/gemma4-staggered/`.
Package: `threshold2048-full32768-maxseq64-diagnostic/`; active plan
`prepared-plan-v3.json`, SHA256
`7b3387ab3d6b696761b01e382d6313f56f9b3184e1391e77c787f81754eb05f6`.
No scheduler or engine production patch is introduced. The earlier FULL4096
threshold2048 cold-median penalty remains a counterexample.

This screen is rejected: C57 mixed cold p50/p95 is235.034/351.147 ms, versus
178.599/309.954 for the current threshold0 pilot and196.849/229.823 native FP8.
Warm p50/p95 is48.033/114.917 versus46.324/88.524 in the threshold0 pilot.
All-request decode p50/p95 is15.552/18.035 ms, versus15.333/17.637. Isolated
q800/8192/18304 remains36.684/176.940/393.465 ms. Raw trace agreement passes
for144 isolated and3179 serving steps on each of eight ranks. These are
separate-launch instrumented timings, not a sole-cause attribution or quality
evaluation. The measured cap does not solve the retained cold-tail failure.

The actual cap trace contains142 distinct steps with measured cold chunks;
six carry two such chunks, so the per-request2048 cap permits4096 cold tokens
in one batch. Every cold request's four chunks occur in consecutive worker
steps. Median first-to-fourth worker CPU-entry span is106.925 ms, maximum
226.152 ms; these are not GPU or admission timings. This establishes a real
intervention, unlike the earlier zero-deferral quota, but not an aggregate or
in-flight prefill bound. Details: `cap-full32768-findings-v1.{json,md}`.

A separate larger-sample diagnostic starts for the current threshold0
candidate versus native FP8: three independent restart pairs, with profile
order optimized/native, native/optimized, optimized/native. Each fresh server
runs a separately saved original24-turn case and319-turn case. The actual
client sampler gives18012 measured requests including546 cold observations for
the longer case. Commands, source variants, corpus, model-shard identity and
environment are pinned, and source restoration uses the reviewed idle/atomic
transition protocol. This is a distinct diagnostic entry point, not an override
of the failed acceptance preflight. Each pair retains its own cold p50/p95
ratios and failures; rows are not pooled to rescue a bad restart. See
`cold-repeat-design-v1.md` and `cold-tail-repeat-diagnostic/`. The cap server
exited and the host passed idle checks before this diagnostic launched. Frozen
plan `frozen-plan-v3.json` has SHA256
`f661b877401e63d69e9fbad607ee36b73b495096176e5c8f3d8dd9a4e7884d46`;
controller SHA256 is
`0a44f61277ebc1a5b221839e878bdaf82d7b62fa73816ae322d3412425478117`.
Source archives/status are in `cold-tail-repeat-v1-source-archive/`; all six
cohorts must finish and the candidate must be restored before interpreting
completion. This run does not replace the full BF16/FP8 acceptance campaign.

The first attempt now terminates with `ServerDisconnectedError` during the
long case's first primer, before any long measured requests. Its selected short
case completed and validates: warm p50/p9546.588/80.937 ms, cold177.561/214.449.
This is one retained short case from an incomplete cohort, not a completed
restart pair or a replacement for the earlier failed pilot. Server teardown
exits0; diagnostic status remains incomplete. Candidate restoration and final
host-idle checks both pass in `cold-tail-repeat-v1-source-archive/status.json`.

The long-case reset succeeds, then the original client builds18012 prompt/hash
records synchronously before its first primer. The server's default HTTP
keepalive is5 seconds; the observed reset→disconnect interval is about5 seconds.
An idle-connection race during that CPU pause is the current hypothesis,
not a proved model failure. The failed controller, frozen plan, raw short rows
and logs remain unchanged.

A controlled CPU transport reproduction now supports that mechanism. With a
separate Uvicorn server and the same5-second keepalive, uncontented original and
private clients both succeed with4.64/4.60-second preparation. Under controlled
CPU contention, the original client's9.28-second preparation causes
`ServerDisconnectedError`; the private client's9.22-second preparation reaches
the first streamed primer successfully. One primer attempt, no retries, and no
GPU/model requests are used. This demonstrates the idle-block mechanism under
the tested condition; it is not independent proof of every real-server cause.

The private copy moves only the three-line cache-reset HTTP block from before
CPU plan construction to immediately after it, before the same three primers.
Payload generation, cache protocol, sampler hashes, measured request timing,
hooks and server settings are unchanged. The original repository client stays
frozen. Actual sources/results are in `cold-tail-repeat-diagnostic/`, including
`staggered_reset_after_plan.py`, `private-client-reset-order.patch` and
`keepalive-repro-v{1,2}.json`. Independent review confirms the exact file change
and unchanged24/319 sampler hashes, server commands/environment, source variants,
profile order and diagnostic thresholds. Actual timing-client provenance is
recorded separately from the unchanged server harness's static original-client
descriptor. The failed v1 artifacts are not rewritten or resumed.

The new `cold-tail-repeat-v2` diagnostic now starts from an idle host with
controller `run_cold_repeat_v2.py` SHA256
`d1c7fb202aa46f90111e6bf3f954846b2db9e746cd773e4b12c4bdaed1061857`,
private client SHA256
`9bf702312620429d927cc7fcdfc46c95460b60db394ec973bb948ec91537a678`,
and `frozen-plan-v4.json` SHA256
`2995f612437ba0a50ff29420b3587472c835fca400da33ca5f479b0ee0800288`.
`cold-tail-repeat-v2-source-archive/` records its progress/restoration status.
All six fresh cohorts, both cases per cohort, structural checks and source
restoration remain required. No result or goal acceptance follows from launch.

The new first optimized cohort now validates its short24 case:1197 measured,
37 cold, warm p50/p9545.298/84.328 ms and cold175.640/227.254. Its separate
319-turn case has passed all three streamed primers and is serving requests
after the second successful cache reset, with no logged errors at this check.
Thus the private preparation-order fix reaches actual model workload beyond
the previous failure point. The earlier310 ms pilot and failed first diagnostic
are retained; subsequent completion of this cohort is recorded below.

That first optimized cohort now finishes both cases with clean exit0 and
identical before/after eight-rank backend inventories. The independent reader
validates18012 measured requests,546 cold, cold p50/p95/p99181.046/309.215/323.156
ms, warm48.775/98.684 ms and all-request TPOT15.155/18.194 ms p50/p95. Thirty
cold requests exceed300 ms (5.49%). The same cohort's short case has two of37
above300 ms (5.41%), yet p95 is227.254: linear interpolation falls between
204.302 and319.062 ms there, versus307.638 and309.741 ms in the larger case.
Thus the short low percentile does not establish absence of that high group.
At that check the matched native cohort was starting, so no pair verdict existed.

Offline artifacts: `analyze_cold_repeat_v2.py` (reader SHA256
`ce77b90ae653c2e18ca1562b714f06477a078646aea3bc25e5e84553db7ea93e`),
`cold-repeat-v2-independent-first-long.{json,md}` and versioned partial reports.
The reader checks exact commands/environments, source-variant reconstruction,
actual timing-client lineage, worker/backend PID binding, sampler/cache rows,
timestamp-derived TTFT/latency and per-case/restart ratios. TPOT's exact last
timestamp is not saved, so its recorded values are bounded and re-aggregated,
not independently reconstructed. Final mode requires all six clean cohorts,
restored candidate bytes on disk and completed restoration status. The
diagnostic remains outside full acceptance and model-quality evaluation.

The matched native short case now validates as well, with cold p50/p95
189.636/263.572 ms, warm68.847/176.409 and all-request TPOT16.672/18.967.
The optimized/native short ratios are cold0.9262/0.8622, warm0.6579/0.4780 and
TPOT0.9218/0.9325. These pass the provisional point guards; the native long
case is still live, so neither a clean complete matched pair nor acceptance is
claimed. `cold-repeat-v2-independent-partial-04.json` retains this partial state.

The first matched restart pair now finishes both short and long cases with
clean exits and unchanged eight-rank backend inventories. Independent
`cold-repeat-v2-independent-partial-05.json` validates four completed cases and
passes cold/warm/TPOT point guards for this pair. The larger-case comparison is:

| Metric, long319 case | Native FP8 p50 / p95 ms | Optimized FP8 p50 / p95 ms | Optimized/native ratio |
| --- | ---: | ---: | ---: |
| Cold TTFT,546 each | 200.242 /327.358 | 181.046 /309.215 | 0.9041 /0.9446 |
| Warm TTFT,17466 each | 69.168 /196.653 | 48.775 /98.684 | 0.7052 /0.5018 |
| All-request TPOT | 16.494 /19.826 | 15.155 /18.194 | 0.9188 /0.9177 |

This is optimized versus unmodified FP8, not a new BF16 comparison. Two restart
pairs remain, and final source restoration is pending. The full
acceptance matrix, quality reruns and historical failed pilot are unchanged.

### Cold-forward GPU budget and rejected collective candidates

An offline read-only reanalysis of the existing matched M8248 cold forwards
joins exact worker NVTX-thread CUDA API correlations to all eight ranks. On the
latest-finishing ranks, physical forward spans are169.931 ms BF16 and147.115 ms
FP8. These old steps were eager under FULL1024; they are not a fresh measurement
of the current FULL32768 graph path.

| GPU category | BF16 rank6 ms | FP8 rank7 ms |
| --- | ---: | ---: |
| GEMM,240 calls | 77.278 | 45.506 |
| Collective,121 calls | 48.539 | 48.898 |
| Fused normalization/position,301 calls | 19.195 | 19.175 |
| FA4,60 calls | 16.632 | 16.203 |
| Activation quantization,240 FP8 calls | absent | 10.669 |
| KV writes,60 calls | 3.256 | 3.212 |
| GELU,60 calls | 2.495 | 2.503 |

All selected collective names are
`ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096>)`;
the name does not identify the executed algorithm or protocol. NCCL2.30.7 can
map other algorithms to this representative entry point and dispatch the actual
operation through its function table. The runtime function ID was not captured.
See the exact-version [kernel mapping](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/device/generate.py)
and [dispatch](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/device/common.h),
and the [NVIDIA maintainer explanation](https://github.com/NVIDIA/nccl/issues/502).
The earlier inference that the symbol proved LL and excluded NVLS is withdrawn.
Durations include waiting/participation, not
measured link bandwidth or guaranteed savings. Category intervals share one
stream with small overlaps; they must not be added as independent critical
paths or summed across the eight parallel ranks. FP8's GEMMs save about32 ms,
partly offset by10.7 ms of activation quantization and common communication.

At clean274, the Torch symmetric-memory communicator's H100/TP8 limit is64 MiB.
Inferred BF16 M8248×5376 activations are84.574 MiB, or85.313 MiB at padded8320;
these exceed the dispatch gate. Payload bytes were not captured directly.
This source/model inference predicts bypassing the Torch symmetric-memory gate;
the NCCL symbol alone does not establish the internal NCCL route. A larger
buffer is already supported by the communicator's constructor test override.
The isolated check now compares the full PyNCCL and symmetric-memory wrappers
at actual Gemma dimensions, including copies, correctness and distributed timing.
No size-limit or serving-backend change is selected from these measurements.

Corrected artifact: `cold-forward-gpu-breakdown-v2.{json,md}`. The original
`cold-forward-gpu-breakdown.{json,md}` and read-only analyzer are retained.
The standalone model-forward benchmark forces compilation off and would remove
Gemma's serving fusions; it is not used as a drop-in serving performance proof.

The single-stream TP8 microbenchmark uses hidden5376/BF16, three rotating graph
buffers,30 samples per shape and two reversed-order sweeps. A common device
rendezvous precedes each replay and is excluded from CUDA-event timing; the
reported latency is the median of per-sample maximum latency across eight ranks.
Correctness covers rank-diverse exact inputs, random inputs against an FP32
reference, graph replay, input preservation and redzones. The symmetric-memory
wrapper includes both copies. Its buffer override is384 MiB; serving is unchanged.

At8320 rows, default PyNCCL takes415–416 microseconds versus465–466 for the full
symmetric wrapper. At16512 rows the comparison is750–752 versus905; at32768,
1408–1409 versus1765. Near the existing64 MiB gate,6144 rows are approximately
351–353 versus349. Raising the gate is rejected. The original multi-stream run
emitted a signal-pad stream-identity warning; the single-stream repeat removes
that warning and confirms the performance result. Both outputs are retained.

A fresh process with the same single-stream source and `NCCL_PROTO=Simple`
takes414 microseconds at8320 rows and750 at16512: no material large-payload gain.
The64-row PyNCCL case worsens from about29 to43.5 microseconds. This controlled
counterfactual does not show that default execution used LL. No protocol flag
is selected. Sources, environment, correctness and raw rank timings are saved in
`gemma4-allreduce-microbench/results-v2.json` and `results-v2-simple.json` under
`/jobtmp/gemma4-staggered/`; the original `results-v1.json` remains unchanged.

The alternative Torch two-shot algorithm is also screened at8320 and16512 rows,
with the same correctness/timing protocol and reversed order. It takes674–676
and1317 microseconds, respectively, versus PyNCCL415 and749–752. Both sweeps
complete with passing correctness. Thus neither existing Torch algorithm
supports raising the gate at these representative long-prefill shapes. Source
`bench_v2_twoshot.py`, exact `two-shot-v2.patch`, the formatted prepared plan and
`results-v2-twoshot.json` retain this negative result; the earlier pre-format
plan is superseded, not executed.

NCCL's separate registered symmetric-memory path also completes its TP8 screen.
The actual production `all_reduce_symmetric_with_copy` operation stages input
once and returns its NCCL-pool output, with no added output copy. Two reversed
sweeps across64/800/8320/16512/18304/32768 rows pass1728 correctness checks and
864 allocation/registration proofs. Graph outputs are retained and distinct;
all rankMAX samples/medians reproduce exactly. Both arms share explicit
`NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1 VLLM_USE_NCCL_SYMM_MEM=1`; these flags
are local to the experiment and do not prove which NCCL device function ran.

At8320 rows, the candidate is409.376 versus415.472 microseconds PyNCCL, only
1.5% faster. At16512/18304/32768 it is5.0%/5.6%/9.1% slower. Its800-row gain
against PyNCCL is not a serving gain: production selects the existing Torch
symmetric-memory path at that payload. At64 rows the candidate26.208
microseconds also exceeds the prior Torch measurement23.296, although that is
a descriptive cross-process comparison. No global flag or size-policy change
is selected. The source, prepared plan, raw result and independent audit are
in `/jobtmp/gemma4-staggered/gemma4-nccl-symm-microbench/`, including
`results-v1.json` and `comparison-v1.{json,md}`. The small-shape comparison does
not justify a model-quality or serving claim.

### Paired Nsight GPU attribution for the remaining mixed tail

Both profiled arms validate, with matching worker/NVTX clock brackets of
3.443 microseconds BF16 and 3.287 microseconds FP8. Attribution follows worker
PID and NVTX step to enclosed CUDA API correlation IDs, then to kernels and
opaque graph activity. GPU work may end after the host NVTX range. Control-RPC
overlaps, capture-edge steps and pre-capture graph records are flagged. The
eligible capture contains 431/433 BF16 and 437/439 FP8 worker steps; timing
samples are forwards within one profiled launch, not independent restarts.

| Forward cohort | BF16 GPU span | FP8 GPU span | Scope |
| --- | ---: | ---: | --- |
| Cold q8192, distributed p50 | 172.37 ms | 147.53 ms | 6 BF16 / 8 FP8 forwards; differing surrounding traffic |
| q8192 + 56 decode, M8248, rank0 | 169.945 ms | 147.093 ms | One per arm; matching query counts, not identical full KV state |
| FULL q80 + 56 decode, M136, rank0 | 14.582 ms | 14.331 ms | 43 / 40 forwards; same query shape |
| FULL q800 + 55 decode, M855, rank0 | 28.361 ms | 27.347 ms | 5 / 3 forwards; same query shape |

FP8 is faster on the observed cold GPU work and near parity/slightly faster on
common captured warm shapes. For M8248, rank0 host execute is 46.24 ms BF16
versus 68.44 ms FP8; it overlaps device work and **must not be added** to the
GPU span. Forward GPU span also includes collective waits; it is not pure
arithmetic time or the entire token/sampling path.

The captured non-cold eager comparison lacks a common shape: one BF16 M1892
versus two FP8 M1093. It cannot establish a controlled eager precision ratio.
The two FP8 M1093 forwards each contain 240 input-quantization kernels, totaling
about 1.829 ms of rank0 device durations. This sum is not a critical-path
speedup prediction. Their rank0 host execute spends 54–58 ms outside recorded
CUDA APIs. Recorded device idle and summed collective duration redistribute
substantially even between these same-query-shape forwards. Long all-reduce
bars therefore remain insufficient evidence of a network-bandwidth problem.

These results support extending coverage and testing bounded prefill work;
they do not prove that the new threshold pilot preserves cold latency or
completes the serving goal. Report:
`nsys-v4-paired-gpu-correlation.{json,md}`, with raw
`nsys-v4-{bf16,fp8}-gpu-correlation.json`. Neither profiled TTFT nor profiled
decode summaries are acceptance evidence, especially around the long stop RPC.

Full quality evaluations complete with no errors/truncations:

| Precision | Native-main established / strict | Candidate established / strict |
| --- | ---: | ---: |
| FP8 | 1276 / 1257 | 1277 / 1258 |
| BF16 | 1278 / 1257 | 1274 / 1254 |

FP8 has two established gains and one loss. All five union established/strict
discordants were reviewed: interpretation, omitted host seating, percent
formatting and a sign-inconsistent reference explain the scored changes.
The newly wrong cannoli answer is byte-identical to an earlier native BF16
answer. There is no visible prompt leakage or detokenizer damage in the review,
but 420 texts change overall and only scored discordants were manually reviewed.
Raw generated token IDs/logits were not saved, so this is not a numerical
causal isolation or proof of per-question equivalence.

BF16 has four established losses and no gains, indices182/235/322/590. Manual
review finds a Jan/Jean name conflation, including John beyond the specified
20 pizza eaters, treating three boxes' combined capacity as one box, and
omitting the host's seat. These are coherent reasoning changes, not truncated
or visibly corrupted streams. Strict scoring additionally gains index473.
The observed BF16 score decrease remains a quality caveat; a McNemar p-value
does not erase it or certify equivalence.

Artifacts under `/jobtmp/gemma4-staggered/`:

- `serving-comparison-detok-window-pilot-v1.{json,md}` and both raw arm folders.
- `detok-window-full1024-pilot-v1-plan.json`, original controller failure log,
  `detok-window-full1024-pilot-v1-bf16-continuation.json` and continuation driver.
- `quality-full1319-{fp8,bf16}-vs-detok-window-pilot.json`, `quality-review.*`,
  `detok-pilot-mixed-tail.*` and full per-question raw evaluation files.
- `final-acceptance-campaign-plan.*`, draft-only collection/validation tooling;
  no three-restart campaign has run.
- `sparse-full-prefill-assessment.*` and `numerical-attribution-next.*` retain
  padding, scheduling and incomplete historical-attribution counterexamples.

### Full unmodified-main model-quality baselines

Both 1,319-question five-shot runs completed with the fixed local dataset,
temperature 0, seed 42, thinking disabled, concurrency 4, max output 1024, and
prefix-cache reset. Runtime source for both measured arms was exactly HEAD
`274a5e19a18460ed41d47117398494b721e3df16` for the two candidate files,
FULL_AND_PIECEWISE/512, TP8 H100, FA4, BF16 KV, no speculation. The launcher
archives full provenance and verifies exact restoration of candidate bytes.
Both arms exit zero; no requests failed and no outputs were truncated.

| Scorer | BF16 | FP8 | FP8 wins / losses |
| --- | ---: | ---: | ---: |
| Established | 1278 / 1319 (96.89%) | 1276 / 1319 (96.74%) | 3 / 5 |
| Strict | 1257 / 1319 (95.30%) | 1257 / 1319 (95.30%) | 4 / 4 |

Established paired delta is −0.152 percentage points, with a descriptive
95% paired-bootstrap interval of [−0.607, +0.227] points. Strict equality of the
aggregate score does not mean identical per-question outcomes. These baselines
do not establish candidate quality or historical numerical equivalence.

Artifacts: `quality-full1319-baselines-v1-{fp8,bf16}/`,
`quality-full1319-baselines-v1-source-transition/status.json`
(`complete=true`, `candidate_restored=true`), and
`quality-full1319-bf16-vs-fp8.json`. The launcher, frozen plan, source backups,
per-item outputs, protocol validation, and before/after Inductor inventories
are retained. Its process guard ignores verified dead/zombie processes while
rejecting live experiments; this node has many historical VLLM zombies.

### Cross-launch numerical differences: verified reduction sensitivity

Independent cache forensics mapped the historical default/dense AOT artifacts
from server logs. Their compile ranges match: (1,48) and (49,32768). All ten
rank-zero generated Inductor modules per precision are byte-identical after
normalizing only AOT directory keys. A different compile-range cutoff is not
supported as the explanation in these runs.

Their autotuning metadata differs. For example, the first 5376-feature
scale-and-RMSnorm reduction uses R0_BLOCK=8192 versus 2048 in default/dense
BF16, and the opposite choices in FP8. These blocks alter the reduction tree.
Saved AOT artifacts can retain multiple compiled candidates and autotune again
after loading; identical generated source and AOT keys therefore do not prove
identical selected arithmetic across launches.

The completed FP8 intervention holds one loaded model and all other selected
kernel configurations fixed while changing only the first scale-and-RMSnorm's
`R0_BLOCK`. CUDA graph dispatch is disabled. Original selections use 2048 on
ranks 0/1/3/4/6 and 8192 on ranks 2/5/7; the two interventions force every rank
to 2048 or 8192 while retaining the other launch parameters. Device-specific
kernel hashes differ, but all ranks have identical normalized arithmetic.

Each phase tests **29 distinct q2048 prompts, repeated twice**: 58 records,
not 58 independent prompts. Each repetition resets and rebuilds the same
8192-token prefix. Every rank observes exactly two 8208-token setup forwards
and 58 2048-token query forwards per phase. Thus both prefix construction and
the cached-prefix query are affected; suffix normalization alone is not
isolated. Non-target selected configuration sets and object multiplicities
remain unchanged. All eight ranks prepare, switch, and restore successfully.

| Comparison | Distinct first-token flips | Repeated records flipped | Top-20 records changed |
| --- | ---: | ---: | ---: |
| Original → all 2048 | 1/29 | 2/58 | 58/58 |
| Original → all 8192 | 2/29 | 4/58 | 58/58 |
| All 2048 → all 8192 | 1/29 | 2/58 | 58/58 |
| Original → restored original | 0/29 | 0/58 | 0/58 |

One original→8192 flip begins with a **6.41992-nat original top-one margin**
and ends with a 0.1875-nat margin for the opposite winner. This is not explained
by nearly tied original logits. Maximum shared top-20 logprob deltas are
8.256 and 9.454 nats against original for the 2048/8192 interventions.
Each phase's two repetitions match exactly. Restoring the original selections
recovers all original token IDs and full returned logprob records exactly,
and the final RPC verifies original attribute identities on all ranks.
Independent postanalysis passes all 184 gates. This establishes a reversible
causal effect of the early reduction configuration on these FP8 outputs.

The original and restored phases also match all 58 historical default-FP8
records exactly. Original→2048 reproduces one of the two historical distinct
token pairs and its direction (`after`, index 3, 715→37798); original→8192 does
too. Their probabilities do not match historical dense2048. The historical
`after`, index 4 flip with a 2.59375-nat original margin is not reproduced.
The analysis field `historical_token_pair_reproduced=[]` refers only to the
forced-2048-versus-forced-8192 comparison, not these original-versus-forced
comparisons. The full historical dense distribution and sole cause remain
unreconstructed.

The source's reduction grouping changes FP32 addition order, but this study
does not measure where the later amplification occurs. FP8 quantization,
attention, nonlinear layers, and changed prefix KV values are possible
mediators; no BF16 counterpart or layerwise quantization-boundary intervention
has been run. Source identity and compilation also do not establish kernel
correctness. The kernel writes both a scaled/residual buffer and normalized
output. Its variance uses unrounded FP32 scaled values while its numerator
reloads BF16 stored values, so the source-contract and eager BF16-scale-then-RMS
references must be checked separately.

The isolated H100 check now **passes all 48 source-contract cases** using
`check_reduction_kernel.py` (SHA256
`752bfc0a874a71a90e37ae8393d2e77041460ed4ac838c3cf28ce7d42be68dc8`).
It executes the exact audited generated kernel with only the reduction tile
changed, on disjoint input/output buffers matching the TP8 caller. Coverage is
four row counts (1, 33, 2048, 8208), six input patterns, and both tiles. Patterns
include random, zero, constant, mixed magnitudes, signed powers, and a fixture
of 52 actual checkpoint embedding rows with the first norm weight.

Every scaled/residual buffer is bit-exact against the reference. All guard
regions and read-only weights/scalars remain intact; all normalized values are
finite and within one BF16 ULP of the FP32 source-contract reference. Error
against the separately reported eager-rounded-variance reference and between
the two tiles is also at most one BF16 ULP. TensorFloat-32 is disabled. The
report is complete with zero failed cases at
`/jobtmp/gemma4-staggered/reduction-kernel-correctness/correctness.json`.
This validates this source under these tested inputs and alias contract; it
is not a model-quality pass or proof that amplified differences are harmless.
Full paired evaluation remains required.

An explicit deterministic-Inductor comparison remains available as a separate
diagnostic: `inductor_compile_config.deterministic=true` with
`TORCHINDUCTOR_DETERMINISTIC=1` changes compiler/config/AOT hashes in this
environment. It is not the measured performance candidate or a substitute for
the same-process intervention.

Completed causal artifacts:
`/jobtmp/gemma4-staggered/reduction-causal-embedding-v2-fp8/`, including raw phase
outputs, per-rank selections/counters/inventories, source snapshots,
`reduction-analysis.json`, and final restoration/status. Standalone findings:
`/jobtmp/gemma4-staggered/numerical_finding.md`. Analysis helper:
`analyze_reduction_control.py`; fixture/auditor CPU tests are retained alongside
the worker. Historical snapshots remain separate from these live controls.

Report and snapshots:
`/jobtmp/gemma4-staggered/numerics-cache-snapshot/REPORT.md`.
Read-only worker probe: `numerics_probe_worker.py` in the same study directory.
An external five-shot GSM8K evaluator is also ready at `quality_eval.py`, with
1,319 local test items, per-item outputs/hashes, established and strict scores,
and truncation/error accounting. The initial 256-question default and candidate
checks are recorded above; full model-quality acceptance remains pending.

### Historical FULL2048 outputs: same-model causal reproduction

The new `q2048-heterovec-fa4-attribution-v1-fp8` diagnostic completes with exit0.
It keeps one loaded model, filters dispatch toNONE and runs58 q2048 first-token
controls per phase,29 distinct prompts twice, with two8208-token cache primers.
The phases are original, the historical first-RMS rank vector
`[8192,2048,2048,8192,2048,2048,8192,8192]`, that vector plus local-attention
NoSplit, and exact original-code restoration. Input/prefix-cache identities match
throughout. The intervention changes code handles; it does not reload weights.

Actual native configuration observations confirm local q2048/B1/K10240 uses
SplitKV2 originally and SplitKV1 only under the scoped intervention. All eight
ranks record2900 such query-local calls per observed phase,580 global query
calls,100 local primer calls and20 global primer calls. Global query Split1 and
primer NoSplit configurations remain unchanged. First-RMS counters verify58
query and two primer calls per rank/phase and the requested rank vector.

| Comparison | Changed first-token rows /58 | Changed retained top20 payloads /58 |
| --- | ---: | ---: |
| Historical native default → current original | 0 | 0 |
| Current original → RMS vector only | 2 | 58 |
| RMS vector → vector plus local NoSplit | 4 | 58 |
| Historical FULL2048 → vector plus local NoSplit | **0** | **0** |
| Current original → restored original code | **0** | **0** |

RMS-only flips `captured/index3` in both repetitions. Adding local NoSplit
reverses that flip and changes `after/index3`, reproducing the historical FULL
control exactly, including saved logprobs. This demonstrates sufficiency of
the combined reduction/attention changes on this fixed model/control set;
isolating only one arithmetic change can give the wrong explanation. All
original RMS handles, singleton/native observer identities and selected
configurations are restored before the final58 controls.

Independent raw-artifact audit passes5262 CPU checks, recomputing phase/input/
output hashes, every saved comparison, all eight actual native censuses, source
snapshots and restoration. Preparation discovers one additional unselected
RMS tuner on ranks1–7; selected configuration sets remain unchanged and the
post-preparation inventory is restored exactly. This inventory detail is
retained rather than treated as a weight/model change.

Artifacts: `q2048-heterovec-fa4-attribution-v1-fp8/` and
`q2048-heterovec-fa4-independent-audit.{py,json,md}`. Executed plan v3 SHA256:
`a0f4ed2a0efc01772d09e60ed2e36d5341d25a352cbc90343839ea54e8731cc3`.
No serving timing or full-sequence quality claim comes from this diagnostic.
Only generated IDs and retained top20 payloads were captured, not all logits.
The older147-PIECEWISE/index4 difference and attribution of every changed GSM8K
answer remain separate; this intervention does not establish their sole cause.

## Portable serving reproduction

Use the isolated measured checkout with the focused runtime patches, and run
the reusable client from a checkout containing
`benchmarks/multi_turn/benchmark_prefix_cache_staggered.py`. The client imports
no model runtime. The study's complete token-ID fixture is retained at
`/jobtmp/gemma4-fp8-regression-study/corpus.json`: 120,000 IDs with raw-file
SHA256 `14d3dcdede5ae8ad126738c170ea18e02ea753037b143c1e31add13ccff16cc4`.
Use the same unchanged file for every precision/profile. Another natural-text
fixture can reproduce the workload protocol, but not the recorded prompt hashes
or numerical results. The default offsets require at least 98,192 IDs, and
marker IDs 10,000 through 189,999 must be valid ordinary tokens; this is a
large-vocabulary Gemma4 study, not a model-agnostic random-token benchmark.

Example **candidate FULL1024** launch, from the isolated runtime checkout:

```bash
GEMMA_MODEL_PATH=/models/RedHatAI/gemma-4-31B-it-FP8-block
OMP_NUM_THREADS=1 VLLM_SERVER_DEV_MODE=1 \
VLLM_V1_OUTPUT_PROC_CHUNK_SIZE=128 \
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$GEMMA_MODEL_PATH" --host 127.0.0.1 --port 30000 \
  --max-model-len 65536 --gpu-memory-utilization 0.85 \
  --dtype bfloat16 --kv-cache-dtype bfloat16 --tensor-parallel-size 8 \
  --max-num-batched-tokens 32768 --language-model-only \
  --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
  --attention-backend FLASH_ATTN --attention-config.flash_attn_version 4 \
  --hf-overrides '{"text_config":{"use_bidirectional_attention":null}}' \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --served-model-name /models/google/gemma-4-31b-it \
  --enable-prompt-tokens-details --enable-per-request-metrics \
  --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":1024}'
```

For native BF16, change only the model path to
`/models/google/gemma-4-31b-it`. Retain both BF16 references: unmodified main's
`FULL_AND_PIECEWISE`/512 profile and the same patched FULL1024 profile used for
FP8. The baseline FP8 reference also uses unmodified runtime source and
`FULL_AND_PIECEWISE`/512. Merely changing flags while retaining the patch is
not a pure source-baseline comparison. No speculation or MTP option is supplied.
The original server commands and source snapshots remain in each arm's
`provenance.json`; the diagnostic worker extension is omitted from this portable
serving replay. Explicit FA4 selection is the operative setting.

For the completed **FULL32768/maxseq64** pilot, retain those runtime patches
and common flags, generate the explicit graph ladder below, and replace the
example's final compilation option with the four options shown. This is the
candidate with the retained C57 cold-p95 failure, not an accepted deployment.

```bash
GEMMA_GRAPH_CONFIG=$(
  .venv/bin/python - <<'PY'
import json

sizes = [1, 2, 4] + list(range(8, 257, 8))
sizes += list(range(272, 1025, 16)) + list(range(1152, 32769, 128))
print(json.dumps({
    "cudagraph_mode": "FULL",
    "max_cudagraph_capture_size": 32768,
    "cudagraph_capture_sizes": sizes,
}))
PY
)
```

```bash
  --max-num-seqs 64 \
  --long-prefill-token-threshold 0 \
  --no-long-prefill-token-threshold-adaptive \
  --compilation-config "$GEMMA_GRAPH_CONFIG"
```

This keeps the32768-token batch budget and avoids the earlier forced2048-token
cap. It changes capacity, memory reservation, compilation and attention metadata;
it is not a padding-only comparison. Use this same profile for the matched BF16
arm. Preserve the separate unmodified-main references and all cold-tail results.

Run the client after server health is ready, with the same model alias:

```bash
for GEMMA_CLIENTS in 4 13 57; do
  .venv/bin/python benchmarks/multi_turn/benchmark_prefix_cache_staggered.py \
    --base-url http://127.0.0.1:30000 \
    --model /models/google/gemma-4-31b-it \
    --token-ids /jobtmp/gemma4-fp8-regression-study/corpus.json \
    --output "fp8-c${GEMMA_CLIENTS}.json" \
    --concurrency "$GEMMA_CLIENTS" --turns 24 --warmup-turns 3 \
    --repeats 2 --seed 20261027 --cases warm mixed
done
```

The client resets prefix caches between cases, retains setup/warmup/background
records, and measures 21 requests per client per case/repeat while background
requests sustain the other clients' load. Output lengths are seeded at
80–140 tokens. Cache counts, requested/observed output counts, corpus hashes,
and complete measured prompt plans are validated and saved. Warm and cold
summaries are separate. Client TTFT ends after parsing a token-bearing SSE
message; returned prompt token IDs make that first message relatively large.
Its formula differs from server SCHEDULED-to-first-output timing.

Run at least three independent server restarts and reverse FP8/BF16 order on
alternate restarts; within-server repeats alone do not meet this gate. Save
source HEAD/diff, flags, profiler attachment state, and every complete JSON.
The current fairness diagnostic changes the existing output chunk setting to
8 and adds external timing instrumentation. It is not the above unattached
acceptance replay. Isolated q800/long-prefill controls and full model evaluation
remain additional gates; this command does not replace them.

## Artifacts and reproduction

These paths are local to the investigation host; they are not repository files:

- Original serving study: `/jobtmp/gemma4-fp8-regression-study/report.json`,
  `results.csv`, `regressions.png`, and `run_study.py`.
- Nsight report: `/jobtmp/gemma4-fp8-nsys/report.json`.
- Raw timing tables: `/jobtmp/gemma4-fp8-nsys/timings.csv` and
  `/jobtmp/gemma4-fp8-nsys/graph-ablation.csv`.
- Eight-GPU timeline: `/jobtmp/gemma4-fp8-nsys/timeline-800.png` and `.svg`.
- Raw captures and SQLite exports:
  `/jobtmp/gemma4-fp8-nsys/{baseline,minimal,graph2048}-{bf16,fp8}/serving.nsys-rep`
  and `serving.sqlite`.
- Capture harness: `/jobtmp/gemma4-fp8-nsys/capture_serving.py`.
- Analysis: `/jobtmp/gemma4-fp8-nsys/analyze.py` and `build_report.py`.

Example capture, from the isolated worktree:

```bash
.venv/bin/python /jobtmp/gemma4-fp8-nsys/capture_serving.py \
  --tag example --mode minimal --graph-limit 2048 --arm fp8
```

Use a fresh tag to avoid overwriting prior artifacts. The harness saves the
complete Nsight and server command in each arm's `provenance.json`.

This file records AI-assisted investigation. No PR is being proposed and no
model-quality or all-cases-performance claim has been cleared.

### C57 tail trace staging

User prioritizes tracing the tail before further kernel tuning. A separate
instrumented diagnostic is being staged under
`/jobtmp/gemma4-staggered/gemma4-tail-trace-stage-v1/`; the running V7 performance
screen remains untouched. The staged client adds headers-received, first-line,
first-data and JSON-parse timings, with an AST check that only its request body
changes. This is source staging, not an executed trace. The trace must correlate
request IDs across frontend receipt/send, engine queued/scheduled/first output,
and GPU work; preserve the existing C57 workload and mark measured timings as
instrumented. Kernel-shape causality remains unproven in either direction.

The staged client passes a synthetic SSE replay: identical HTTP payloads and
output identities versus the existing client, ordered diagnostic timestamps,
and nonzero parse timing. This checks instrumentation mechanics only; no live
HTTP or model was used. Proof: `gemma4-tail-trace-stage-v1/client-smoke-v1.json`
under `/jobtmp/gemma4-staggered/`. Existing request-admission/Nsight tooling in
`mixed-tail-nsys-v1/` is being reviewed for reuse, including request-to-step and
GPU clock correlation, before another server is launched.

Client trace staging now includes per-request aiohttp connection-pool, connect,
headers-sent and response-header events plus a5ms event-loop heartbeat. A local
HTTP smoke with a one-connection pool verifies request-ID correlation and queue
start/end events. Updated synthetic SSE replay preserves payload/output identity.
Current proof is `gemma4-tail-trace-stage-v1/client-stage-proof-v2.json`, binding
`client-smoke-v2.json` and `transport-smoke-v1.json`; earlier v1 client proof
belongs to the pre-transport-context revision and is superseded. These are
instrumentation checks, not live model performance evidence.

Worker trace staging reuses the buffered `MixedTailProbe` / `StepProbe` hooks
and combines them with the current `FullChainModelProbe` eligibility checks.
A synthetic scheduler/dispatch smoke preserves the returned descriptor and
records856 scheduled tokens,864 padded tokens,57 request IDs and
per-request prompt contributions (800 new tokens plus56 decode tokens).
The synthetic request-context flags are not a live scheduling observation.
Proof: `gemma4-tail-trace-stage-v1/worker-smoke-v1.json`. Actual GPU and
scheduler compatibility remain untested until the separate diagnostic runs.

Independent raw/source review confirms the corrected per-request interpretation:
`/jobtmp/gemma4-staggered/g4-corrected-v2-c57-tail-review-v1.md`.
Server TTFT ends at the CPU timestamp attached to `EngineCoreOutputs`; it is
not GPU-only time. QUEUED starts at scheduler admission and omits earlier core
input backlog. Final metrics arrive in the usage chunk. Therefore the trace
must split pre-QUEUED and post-output delay as well as the SCHEDULED interval.
The existing admission probe plus worker-step logs provide the first causal
check; GPU profiling will resolve remaining worker-versus-device ambiguity.

The first trace draft targets the corrected V6 runtime with fusion off/on,
matching the pair that exhibited the tail, and the unchanged C57 mixed plan.
`gemma4-tail-trace-stage-v1/trace-plan-draft-v1.json` is explicitly plan-only;
it does not launch a server. The new `measure_tail.py` stages whole-case worker
tracing, client transport/loop tracing, eligibility checks and an explicit
post-drain flush. Server-hook CPU smoke, exact execution pins and all-process
startup verification remain required before interpreting any live trace.

Trace startup validation is staged in `startup_guard.py`: require10 distinct
process manifests, same host/boot/time namespace, bracketed perf/monotonic clock
samples, exact probe source, loaded runtime source hashes, and unique API/core
processes with all required timing/flush hooks installed. `measure_tail.py`
calls this guard before worker tracing or benchmark traffic. Syntax passes;
synthetic guard failure/success tests and live startup validation remain pending.

The startup guard passes synthetic valid-process manifests and rejects missing
processes/hooks, wrong source hashes, incompatible clock brackets and wrong
runtime paths (`guard-smoke-v1.json`). `execute_trace.py` now stages plan
freezing, source/runtime validation, the shared exclusive GPU-run lock, GPU-idle
checks, isolated per-arm caches/probe output and orderly server teardown for
the original V6 off/on C57 diagnostic. It is not frozen or executed yet;
server-hook smoke and driver review remain pending.

Server-hook CPU smoke now passes two concurrent request joins, unchanged ASGI
body forwarding, cancellation-context reset and explicit API flush completion
before reset returns. Evidence:
`/jobtmp/gemma4-staggered/gemma-tail-server-hooks-cpu-smoke-v2/proof.json`.
The hook source SHA256 is
`91e59386a3782a9b3d9974b95e4b9f158de16f65d8c8d42e9ebfe96077cae665`.
This uses local engine/ASGI stubs, imports no Torch, and leaves real
core/startup/clock/GPU validation pending. The execution driver is under
independent review before freezing; no second server has launched.

V7 is now serving. First raw-validated isolated q80 result (400 measured):
p50/p95 23.553/24.160ms; changes versus V6
+0.70%/-0.06%.
All-eight-rank eligibility and runtime source routing validate. Other cases and
post-serving eligibility remain pending. Artifact:
`/jobtmp/gemma4-staggered/g4-carry-v1-reader-first-serving-v1.json`.

Execution-driver review found missing eligibility enforcement, retained weight/
NCCL/protected-source guards, command/environment comparison and plugin METADATA
pins. The draft now includes those checks. The first diagnostic explicitly
collects CPU admission and worker submission/dispatch traces without Nsight;
`gpu_profile=false` prevents treating host intervals as GPU durations. Actual
batch composition and graph padding will guide a subsequent GPU capture if
needed. The user's batching observation remains central: request q800 can be
executed at a different GEMM M after co-batching/padding, so isolated q800
latency cannot exclude a mixed-shape kernel or dispatch regression.

V7 isolated progress (`g4-carry-v1-reader-progress-v3.json`,400 measured each):
q800 p50/p95 36.340/37.130ms; p50 versus V6 -0.89%, versus historical native-main -52.28%.
q8192 p50/p95 176.556/178.275ms; p50 versus V6 -0.28%, versus historical native-main -8.08%.
Long q18304, all serving cases and post-serving checks remain pending.

The trace harness now validates post-drain API/core flush identities/reasons,
zero buffer overflow and empty pending first-token maps, and stores a hash-bound
flush barrier. It also validates the original frozen C57 mixed request-plan
hash/count after traffic, and labels timings instrumented via a measured-window
callback recording both clocks. These are staged checks; real execution and
trace analysis remain pending, and no host interval is labeled GPU duration.

C57 trace execution is now frozen in
`/jobtmp/gemma4-staggered/gemma4-tail-trace-stage-v1/execution-plan-v1.json`,
SHA256 `0ad1fd21ee9ba44e4733c3adb21cebb188a421b3370b94072fed3bb00200a992`.
The reviewed post-drain serialization issue is fixed. Queue controller
`queue_g4_tail_trace_v1.py` waits for the exact live V7 controller PID1610111,
requires its clean completed status, then invokes the frozen trace driver,
which independently takes the shared run lock and verifies GPU idle. Queue
status is `/jobtmp/gemma4-staggered/g4-tail-v1-queue-status.json`.
The first trace is CPU admission/worker dispatch only; actual GPU/kernel
profiling remains a separate follow-up if needed. No diagnostic server has
started while the V7 benchmark is running.

V7 completes all isolated controls. q18304 median/p95 is
394.465/396.683ms, median versus V6
-0.04% and versus historical native-main
-5.68%. Artifact:
`g4-carry-v1-reader-progress-v5.json` under `/jobtmp/gemma4-staggered/`.
Serving cases and post-serving validation remain pending.

V7 full342-graph capture reports4.96GiB on every rank, the same rounded figure
as V6 (105sec V7 versus105–106sec V6). This does not demonstrate graph-pool
savings from final carry elision. C4 raw output is incremental: the reader's
first attempt encountered one completed case while expecting warm+mixed;
reader handling now records partial case sets as pending instead of failing.
The server/benchmark was not restarted or changed. No C4 comparison is accepted
until both cases validate.

Trace analyzer preparation identified a clock-evidence limitation: server/core/
worker manifests are bracketed and checked, but the client measured-window
sample lacks an upper monotonic bracket and persisted client clock metadata.
Preserve this qualification for cross-process client intervals; it does not
invalidate independently validated server/core/worker joins or local client
elapsed durations. Frozen trace sources remain unchanged.

V7 C4 warm+mixed cases now validate (`g4-carry-v1-reader-progress-v7.json`):
c4:warm:r0: warm TTFT p50/p95 versus V6 -0.37%/-0.48%; decode median -0.15%; cumulative warm median versus historical native-main -23.12%.
c4:mixed:r0: warm TTFT p50/p95 versus V6 -0.51%/-0.73%; decode median -0.03%; cumulative warm median versus historical native-main -22.84%.
C13/C57 serving and final post-serving routing remain pending.

Historical actual batch-shape check (not current V6/V7): the validated native
FP8 C57 Nsight diagnostic has148 rank0 steps containing an800-token prompt
contribution across the whole case, including warmup/background. Only2 use
M800. Padded M864 accounts for73 steps, M944 for33 and M1024 for21; larger
steps also occur. Common batches are855/856 scheduled tokens padded864, and
935 scheduled tokens (880 context+55 decode) padded944. This empirically
supports the user's batching point while proving no cause for the current
tail. Artifact `/jobtmp/gemma4-staggered/historical-q800-batch-shapes-v1.json`
binds the original233MB worker trace. The new current-candidate trace must
identify actual shape and dispatch rather than infer them from request q800.

V7 C13 warm+mixed validate (`g4-carry-v1-reader-progress-v9.json`):
c13:warm:r0: warm median/p95 versus V6 -0.45%/-1.51%, decode median -0.06%; cumulative warm median versus historical native-main -20.07%.
c13:mixed:r0: warm median/p95 versus V6 -0.26%/+3.63%, decode median -0.30%; cumulative warm median versus historical native-main -19.25%.
The mixed warm p95 regression of3.63% versus V6 remains a counterexample.
C57 mixed and post-serving checks are pending; the exact live controller and
queued waiter have both been polled and remain active.

V7 screen COMPLETE, controller exit0, server clean, unchanged runtime and no
pending reader checks (`g4-carry-v1-reader-complete-v1.json`). Quality remains
unassessed. At C57 mixed, versus V6 warm median/p95 changes+0.53%/-8.00%,
q800 median/p95 changes-4.88%/-58.34%, decode median-0.32%, but cold p95
regresses+37.85%. Against fusion-disabled control, q800 p95 is-26.83% and
cold p95+36.05%; against historical native-main cold p95 is+36.84%.
Thus the q800-tail improvement cannot establish a general tail fix; retain the
new cold-tail regression. V7 remains a mixed performance screen, not acceptance.
The queue observed V7 terminal clean completion and advanced to the frozen
original-V6 C57 trace driver. Actual diagnostic startup/routing and trace
completeness still require verification.

Trace v1 startup was deliberately interrupted before any measured trace request:
11 startup manifests were observed, including an absent short-lived helper with
empty installed hooks, alongside the10 live API/core/worker roles. The frozen
fixed-count startup guard would reject this normal extra helper. Driver1701290
received SIGTERM; queue session89707 is terminal exit1. Preserve this startup
attempt; it is not measured performance evidence. The next trace package must
bind the live API/core and eight eligibility worker PIDs while retaining the
auxiliary manifest, rather than counting all historical startup files. No frozen
v1 source has been changed. V7's completed serving results remain unaffected.

### V7 three-way recap against historical native baselines

Raw-validated completed V7 summaries versus the saved historical main BF16/FP8
report are collected in `/jobtmp/gemma4-staggered/g4-v7-bf16-native-threeway-v1.json`.
V7 warm median/p95 beats historical BF16 across all six C4/C13/C57 warm/mixed
cases; decode medians are also lower in each. This is not matched-restart
acceptance: historical C4/C13 windows differ and cumulative changes include
serving settings/code. The C57 mixed cold p95 counterexample is BF16258.344ms,
nativeFP8229.823ms, V7314.497ms: V7 is worse than both. V7's own quality screen
is unassessed, and fresh BF16/native three-restart validation remains outstanding.

Trace v2 is isolated at `/jobtmp/gemma4-staggered/gemma4-tail-trace-stage-v2/`.
The guard binds the eight actual eligibility worker PIDs plus unique live
API/core roles and verifies parentage. It retains inactive empty-hook auxiliary
manifests instead of rejecting all counts above10. Synthetic positive11-manifest
startup and8 negative cases pass (`guard-smoke-v2.json`). `measure_tail.py`
now also records bracketed client perf/monotonic clocks with host/boot/time
namespace metadata before traffic. No v1 frozen input changed. Independent
source review and source/runtime freeze validation precede relaunch.

Trace v2 execution launched with frozen plan SHA256 `797057221e62252f1339ce9eff9e44c187cf99854076a897ef033d765d38d782`.
Independent source review: `g4-tail-v2-independent-startup-source-review-v1.json`
under `/jobtmp/gemma4-staggered/`, no blocking issue found. Root execution
session72087 owns GPU work; actual startup/role/clock validation remains pending.

Request-tail analyzer is prepared outside frozen trace inputs:
`/jobtmp/gemma4-staggered/analyze_g4_request_tail_v1.py`, SHA256 `21675205b836f6dec97563df24815fa48fc9f8ab8fbb8a1c860c30e5993bf786`.
Synthetic smoke v5 covers exact opaque request-ID joins, two-chunk prompt
accounting, all-eight-rank shapes, invalid clocks/sources, missing client clock
fallback and a validated full-TTFT partition. Real trace analysis remains
pending, with independent source review in progress. The off arm is live in
DeepGEMM warmup. Analysis will begin only after its clean server completion;
it will not attach instrumentation to the running next arm.

Trace v2 off arm completes cleanly (status complete/exit0). Actual startup guard
verifies API1707986, core1708889,8worker PIDs and1inactive helper; measured
C57 mixed workload, worker steps and explicit zero-overflow flush records are
saved. Client prints an instrumented q800 p95 of189.969ms and cold p95 of
199.901ms. These are diagnostic observations, not acceptance timings or a
kernel attribution. The reviewed offline analyzer now joins the saved records;
first current-trace interval/batch-shape evidence remains pending. The driver
continues to the fusion-enabled arm under the same instrumentation.

### First qualified current C57 trace: two tail mechanisms

Off-arm posthoc analysis is `COMPLETE_QUALIFIED_SERVER_WORKER_DIAGNOSTIC`:
all1551 requests join, including1197 measured, with validated all-eight-rank
shapes, client/server clocks, sources and flushes. Artifact:
`/jobtmp/gemma4-staggered/g4-tail-v2-off-posthoc-v1.json`, SHA256
`fd9ea4bb4b4114465d88443719d095808d0b1eea17018ef38e5260c6e5940e82`.
Among8 measured q800 requests above150ms,6 wait136–145ms specifically between
core input enqueue and scheduler.add_request; their scheduled intervals are
37.9–56.0ms at FULL M864/M944/M1664. The two requests bracketing q800 p95
wait139.675/139.345ms there and then take38.339/37.928ms scheduled. Two other
slow q800 requests share their own first-prefill batch with an8192-token cold
request at FULL M9088/M9216 and take171.221/175.560ms scheduled. The318.201ms
cold outlier includes143.137ms input backlog plus158.751ms scheduled.
This off arm has custom fusion disabled and all these batches use FULL graphs.
The evidence prioritizes core admission blocking and long-prefill co-batching;
it does not establish the device-level reason for the blocked admission, nor
rule out a shape/kernel contribution to the large mixed batches. Predecessor
worker/core overlap and the on-arm trace are the next checks. Host submission
bounds remain distinct from actual GPU duration.

Predecessor association strengthens the off-arm localization: all6 q800 input-
backlog outliers were enqueued during the same recorded core active-step tuple
that emitted a preceding8192-cold first output; target scheduler admission follows
that native output by0.075–0.346ms. Cold IDs independently join to worker FULL
M8320/M8448; this does not equate core counters with worker counters. The
engine busy loop in the frozen `vllm/v1/engine/core.py` drains input before
`_process_engine_step`, whose selected step function waits for model results.
This identifies host-side admission blocking behind long work, not frontend
output delivery. It is not GPU kernel-duration accounting. Moving admission
earlier alone would shift reported queue intervals without necessarily improving
client TTFT; subsequent changes must shorten interfering work or reduce actual
contention while retaining cold-latency gates. The on-arm trace remains pending.

Compact pinned off-arm review:
`/jobtmp/gemma4-staggered/g4-tail-v2-off-localization-v1.md` (SHA256
`1c899c163b523a04d28a5ecac01046eedb265aabf2ff55f3e961e6fbc25c82c4`).
Across the whole case,156 q800 requests map to148 unique q800-containing
worker steps, all148 FULL (M86478steps, M94434steps). The measured subgroup is
117requests across111steps. Unlike the older1024-cap trace, no NONE dispatch
is assigned to these current requests. This rules out fallback as the cause
for the traced off-arm tails, but not device execution cost within FULL graphs.
The enabled arm and its separately gated offline analysis remain pending.

Next priority after the enabled host trace is a bounded Nsight device timeline
for the interfering long-prefill work. Source-only preparation is delegated
in a new `gemma4-tail-nsys-stage-v1/` package; nothing is launched yet.
Reuse the same C57 mixed workload, FULL32768 and current V6 candidate; a short
CUDA-profiler API capture must record its actual bounds and qualify profiling
overhead. Graph-level duration and node-level kernel attribution have different
costs and must be labeled. Keep all cold-tail gates open; do not accept a change
that only moves time from pre-admission into reported scheduler queueing.

Both current host trace arms now complete cleanly and qualify all joins:
off1551/on1556 raw requests,1197 measured each. Both retain S3 and differ in
fusion flag. Enabled q800 p95 is197.157ms versus off189.969ms; slow>150ms
counts10 versus8. Enabled cold p95 is311.739ms versus off199.901ms. Four enabled
cold requests exceed300ms: two wait before admission and two share one FULL
M16512 batch containing two8192-token cold prefills (scheduled284.080ms).
Five enabled slow q800 requests wait135.8–153.3ms before admission; three share
their own batch with a cold prefill at M9216; two have long scheduled intervals
at small own M864/M944. For those last two, a preceding cold output occurs
149.716/146.648ms after target scheduling; the target output follows26.484/
29.139ms later. Their own host submissions last only1.53/1.61ms. This supports
prior-pipeline interference; actual GPU duration/queueing remains pending.

The independently reviewed bounded Nsight graph run is now launched as root
session25350 using `execute_g4_tail_nsys_v1.py` and
`g4-ns1g-plan-review-v1.json`. It starts only after the clean host pair and GPU
idle. Its native GPU contents and process ancestry must be validated after
collection; report/export existence alone is not evidence of valid GPU timing.
A genuine untouched-main runtime audit/control plan is prepared separately;
none of these modified fusion-off controls is relabeled as untouched main.

Historical cold counterexample check: C57 mixed has37 measured cold requests
in all these runs. Original saved BF16 and native-main FP8 each have0 above
300ms; modified V6 off/on each have2, and V7 has3. V6 off's two slow colds share
reported scheduled interval290.4ms, while V7 includes both long scheduled and
large residual cases. Thus current host-trace evidence proves the mechanism
exists without custom fusion, but does not prove that untouched-main tails are
equivalent or that S3/config changes are irrelevant. Keep the genuine-main
comparison as an explicit outstanding requirement rather than attributing all
percentile movement to fusion or all movement to random phase.

Genuine pinned-main source runtime is now materialized separately at
`/jobtmp/gemma4-native-main-tail-runtime-v1`: all8542 tracked git blobs/type/modes
and1004 original untracked native/vendor artifacts
verify, with clean tracked diff and independent git metadata. Original native
SHA256 is `5bb7d1c488d7890e9dcfe42c975c19e1c78accabc007b94d2c655ae64abd09aa`; V6's extended native is not reused.
Setup proof: `/jobtmp/gemma4-staggered/gemma4-native-main-tail-runtime-setup-v1.json`
(SHA256 `c0a47f161f60ca0a90e4d5fa6e83ca3751c7a6f5e2db45c40a57c4825f61bab4`).
This verifies source/artifact bytes only, not live imports, model execution or
the precompiled binary's build commit. A new private native-main probe will
validate actual source/config/FA4/TP8/BF16-KV/dispatch before timing.
Original-serving and current-CLI-matched profiles remain distinct; neither
existing checkout or frozen trace source has been restored or modified.

Private native-main trace harness preparation now exists at
`/jobtmp/gemma4-staggered/gemma4-native-main-tail-control-v1/`. General client,
ASGI/core and worker-step modules are copied unchanged from host V2; the server
runner points at the independently materialized main runtime. It remains
incomplete and unlaunched: the V2 measure template must be adapted to the new
native-only probe/validator, and both serving profiles require a frozen reviewed
controller. No full-chain eligibility call may be bypassed or treated as a
native-main validation. Nsight run25350 remains the only GPU owner.

Native-main measure adaptation now calls `probe_native_main_identity` and the
new profile-specific `build_expectations`/`validate_native_identity` contract,
comparing stable identities before/after traffic. It contains no full-chain
eligibility import/call or bypass. Source syntax passes, but the private probe
and validator are still being completed and must be installed/reviewed before
execution; the native-main controller remains pending. The source runtime and
all frozen host/Nsight trace inputs remain unchanged.

Native-main controller source now exists at
`/jobtmp/gemma4-staggered/execute_g4_native_main_tail_v1.py`. It stages separate
original-serving/current-CLI profiles with short cache paths, private native-only
eligibility, source/native overlays, weight/NCCL checks, original interpreter,
shared GPU lock/idle checks and exact command/environment verification. Syntax
passes; no native-main plan is frozen or executed until the probe/validator
CPU checks, installed helper bytes and independent driver review are complete.
Nsight25350 remains the only live GPU run.

The native probe/validator are now installed together in the control package;
source binding and synthetic CPU checks for both profiles pass. The controller
removes the fusion opt-in environment variable entirely, matching the native
absence requirement rather than treating flag0 as a main identity. Review plan
`g4-native-main-tail-plan-review-v3.json` additionally pins corpus, provider
reference and benchmark sources. Separate reviews check actual main API/property
assumptions and root controller integration. Actual worker RPC remains unrun;
no native-main model server has launched, and no eligibility gate is bypassed.

Native probe review caught a real RPC context bug before launch: capability
checks can reach `get_current_vllm_config()` outside an active context. The
control copy now scopes both checks with `set_current_vllm_config(config)`.
`support-context-proof-v1.json` reproduces the unwrapped assertion using main's
actual getter/setter AST, then verifies the wrapped calls and restoration.
`probe-context-correction-v1.json` proves the only AST additions are that import
and scope; both metadata profiles pass again in `cpu-validator-proof-v3.json`.
A separate source reviewer clears the fix; live native RPC remains pending.

The genuine-main two-profile plan is frozen at
`/jobtmp/gemma4-staggered/g4-native-main-tail-plan-v1.json`, SHA256
`5ae6d6451c8c0cb167ce5361729cb757c903f0b0c1e1a347b6e1957fbf0137ef`.
Queue session85204 observes exact live Nsight controller1842488 and its creation
time, waits for terminal state, then invokes the independently source/lock/idle-
gated native controller. Prior Nsight status is retained even if unsupported;
no GPU overlap or successful-profile assumption is allowed. Queue status:
`g4-native-main-tail-queue-status-v1.json` under the same artifact directory.

Independent graph-reader review identified partial launch coverage as a gap:
every relevant launch inside a step must have native activity before its device
envelope is called complete. This is being tightened and tested with a
multi-launch missing-activity case before any current trace interpretation.

### User steering: establish repeatability before further optimization

The user notes the tail failure is intermittent and may also occur on native
main. This is plausible: existing screens show failed sample gates and host
mechanisms, not a repeatable fusion-caused distributional regression. Hold
further kernel/scheduler changes until matched repeated unprofiled comparisons
with genuine main establish whether it worsens the tail. The running native
control and completed profile remain evidence collection. Retain every failed
screen/counterexample; do not dismiss it as noise or waive any final gate.
Original-serving and matched-CLI controls must be labeled, order reversed
across independent restarts, and tail results kept per restart.

Current Nsight native timing qualifies an interior subset only: no q800 request
above150ms falls inside its8s window. Do not use the profile as direct GPU
proof of q800 tails. It captures a q80 small FULL graph with~13.9ms own GPU work
and~138ms queued behind a preceding~144.6ms8192-prefill graph, and a cold
pre-admission wait overlapping prior long GPU work. These establish observed
interference mechanisms, not that fusion introduced them or that native main
has a different tail distribution. The compact report retains that scope.

Qualified compact GPU report:
`/jobtmp/gemma4-staggered/g4-ns1g-nativegraph-localization-v2.md`, SHA256
`fff640699e6bfc21d913821aad752e2c983e27139531168d7fa6e38067e20b11`.
The reader matches3632 native graphs across454 all-eight-rank steps;452 are
eligible and2 excluded. Eight eligible8192-prefill steps have MAX-rank graph
durations144.56–146.95ms at FULL M8320/M8448. Clock-offset bracket is3.988µs.
Generic diagnostics/warnings are retained; there is no global losslessness
claim. No >150ms q800 tail lies in this capture, so the report's direct GPU
causality remains scoped to its captured q80/cold examples. This does not
establish a repeated main-versus-candidate performance regression.

### Genuine-main original-CLI control also exhibits the tail

The first genuine-main diagnostic, `g4-mn0-fp8`, completed with exit0,
before/after native identity validation, and1197 measured requests matching
the original C57 mixed24-turn workload hash
`791c09765a6d977b0d779cbbbb7454f7d216d8c5759a0266437ee4007ad0d830`.
Its cold TTFT p95 is325.875ms across37 requests; q800 TTFT p95 is303.714ms
across117 requests. Thus large tails occur on genuine main without the fused
kernel. This is an instrumented host diagnostic using original serving CLI,
not an unprofiled matched repeatability comparison or final acceptance result.
Do not infer that the candidate has no tail effect, or compare these numbers
as a causal speedup. The matched-current-CLI main control is starting next.
Further optimization changes remain held pending repeated unprofiled controls.

Artifacts: `/jobtmp/gemma4-staggered/g4-mn0-fp8/`, including `status.json`,
`eligibility-before.json`, `eligibility-after.json`, `trace-workload-check.json`,
`trace-measure-complete.json`, and raw `c57-requests.json`.

Independent CPU review now confirms all-eight-rank native identity before/after,
actual source/native file fingerprints, unchanged1197-request measured plan,
all1551 raw requests joined to host records, and3182 matching worker steps per
rank. No unmatched/overflow/pending records were found in these checked buffers.
Cold has6/37 requests above300ms (maximum336.247ms); q800 has7/117 above300ms
(maximum395.905ms). These descriptive thresholds are not new acceptance gates.
Review: `/jobtmp/gemma4-staggered/g4-mn0-independent-review-v1/report.json`,
SHA256 `e44300d48f2c055538912f378f13387d4d273429e81b632ad9d2561450d3d4e6`.
Original-CLI q800 dispatch is NONE, whereas current-CLI is FULL; this and
different batch/arrival histories prohibit direct fusion-only comparisons.

### Unprofiled repetition preparation preserves the original benchmark

The source-only readiness audit in
`/jobtmp/gemma4-staggered/gemma4-tail-repeatability-plan-v1/` retains all six
C4/C13/C57 warm/mixed plans,107/40/24 turns, three warmup turns, and410 isolated
requests per query (10 warmup,400 measured). The proposed focused comparison
uses genuine-main original CLI (N0), genuine-main current CLI (N1), and frozen
V7, each across three fresh servers with reversed ordering. It has no BF16 or
quality arms and cannot complete the original60/180-point acceptance campaign.
Those exact gate equations and every prior counterexample remain retained.

Private harness preparation addresses native graph-restore RPC support and
profile-specific identity checks, without production kernel/scheduler changes.
Native N1 has331 mixed captures; V7 has342 descriptors including11 narrow
decode descriptors. A342-only candidate validator must not be applied to native
main or bypassed. A metadata-only worker helper will avoid tracing inheritance;
no ASGI middleware, step hooks, ClientTrace or loop monitor will run during
measured traffic. Fresh source-bound cache paths remain per launch.

The descriptive results reader under
`/jobtmp/gemma4-staggered/gemma4-tail-repeatability-reader-v1/` reuses the pinned
original workload validator, records per-restart quantiles and p95 contributor
IDs, and computes paired ratios without pooled percentiles or acceptance claims.
CPU replay of all4188 existing V7 measured requests passes, with negative checks
for traced inputs, changed workload hashes, altered summaries and incomplete
status. This is reader validation on old data, not a new repeatability result.
New unprofiled runs are not yet launched; the second native diagnostic remains
the GPU owner while its fresh-cache startup completes.

The reader's independent CPU review passes at source SHA256
`d070895bde1c70c57300142969f9ac68414223d0f7e65a0c70e4ad1a8de3108a`.
It verifies every real-case p95 contributor bracket, same-profile ratios,
a known changed ratio, rejection of error/forced teardown/invalid exit-code
status, hook-enabled configuration, and preservation of existing output files.
Report: `/jobtmp/gemma4-staggered/g4-repeatability-reader-independent-cpu-v2.json`,
SHA256 `bb28d433b87bac658d0a85cefdade454bfe4105156624c4f4b83c37a971603a1`.
Source/runtime eligibility remains separately required. BF16 native/fallback
eligibility preparation is a separate CPU-only task for eventual full-goal
validation; it does not alter the focused repeatability design or waive quality.

The private native unprofiled helper is now implemented at
`/jobtmp/gemma4-staggered/gemma4-native-unprofiled-probe-v1/`.
Its object-only metadata extension avoids Step/Mixed inheritance, validates the
actual execution/dispatch wrapper chains, supports explicit eager controls,
and restores the original candidate keyspace/descriptor identities. Root
reran its CPU behavioral/source proofs successfully: both native profiles,
17 corruptions per profile, all five graph limits, exact restoration,
replacement-manager rejection and inverse-AST preservation of original guards.
Worker SHA256: `dc2a9ab55a5436030c9fd0c0e8a64e5898a281ddb347b116ef0b7d4392d7d970`;
validator SHA256: `fd7f919c218f8e48bdaa40d6f52a5f736f7523d1aef3f9ff6cc2fab8d83322ee`.
Actual runtime RPC validation is still required; CPU proofs are not serving
measurements. The live second native diagnostic remains unchanged.

BF16 validation design must follow actual V7 fallback behavior. With the FP8
opt-in enabled, its broad `_full_chain_config_supported` and layer fusion flags
can be true for unquantized BF16. The whole-chain guard still fails for its
unsupported linear keys/markers; the helper falls back to ordinary norm and
linear execution. Those broad flags alone do not prove private FP8 execution.
Future BF16 validation must positively check actual unquantized linears,
whole-chain fallback and zero private FX calls, with pinned source/environment,
instead of inventing a requirement that every broad eligibility flag be false.

Root review found a pre-launch cache-isolation bug in the new repetition
controller draft: its remapping literal named the V6 cache root while its
base environment contained V7 cache paths. Without correction, purportedly
fresh restarts would reuse old caches and TMPDIR. No unprofiled repetition
was launched with this draft. Require derivation from the actual frozen base
cache root, rejection of paths outside that root, disjoint per-slot cache/TMP
paths, and short socket-path verification before freezing. The reviewer also
requires full expected V7 source state beyond the11 critical metadata hashes;
recording an arbitrary git diff is not source-freeze validation. Both fixes
are private harness preparation, with production performance code held.

### Native current-CLI control completed; both main profiles show large tails

`g4-mn1-fp8` completed cleanly on genuine main274 with original native artifacts,
FULL32768/331 mixed captures/maxseq64/threshold0 and no private fusion. Warm
TTFT p50/p95 is65.531/137.703ms, q800 is78.617/206.773ms, and cold is
196.773/342.650ms. Cold has6/37 requests above300ms (maximum361.467ms),
q800 has2/117 (maximum362.564ms). These thresholds are descriptive, not gates.

Root mechanically adapted the independent main0 checker to the second profile;
`adaptation.json` records path/profile/index/scope replacements without changing
validation logic. Strict before/after native identity, original workload/cache
checks and source/native fingerprints pass. All1553 raw requests join to host
records; all8ranks have3190 matching worker steps, with no unmatched/overflow/
pending records in checked buffers. Root replay is not a new independent review.
The two-profile controller completed with final GPU idle and no error.

Report: `/jobtmp/gemma4-staggered/g4-mn1-root-review-v1/report.json`, SHA256
`c1df2bf4c3c1a9e10001ea30f674a130d970e511b60c6b95acdc9533cdd1ba6d`.
Its adjacent README preserves the two-profile table and p95 contributor IDs.
This remains instrumented diagnostic evidence. Same measured plans do not fix
arrivals/batches, and current native graph policy differs from V7. Neither
fusion-specific tail causality nor candidate harmlessness follows from these
two single runs. Proceed with the reviewed unprofiled restart comparison;
do not change performance code or waive old failed gates on this evidence.
