# Internal Gemma4 validation record

**All12 serving arms, native correctness and model-quality checks are complete.
The full speed goal remains open:160/162 speed points pass, with two retained
C57 mixed warm-TTFT misses against the same-policy BF16 control.**

The study uses pinned main `274a5e19a18460ed41d47117398494b721e3df16`,
RedHatAI Gemma4 31B FP8-block versus native Google BF16, TP8 H100, the
vllm-project FA4 fork, BF16 activations/KV and no speculation/MTP.
F0/B0 use the original native profile with effective max_num_seqs1024.
F1/B1 use optimized source, FULL32768 and max_num_seqs64; B1 proves ordinary
BF16 fallback with zero private FP8 calls. Main-to-optimized gains therefore
combine code and configuration changes, with an explicit capacity tradeoff.

## Final serving results

[The complete metrics](final_three_restart_metrics.json) retain all12 arm
summaries, source/provider/config identities, quality aggregates and all180
point outcomes. No raw requests, corpus tokens or model answers are included.
The [reproduction README](../README_gemma4_fp8.md) records exact restart/model,
concurrency and case orders, external inputs and launch commands.

| Restart block | Blocking speed checks | Retained failure |
| --- | ---: | --- |
| 1 | 53/54 pass | C57 mixed warm p95 F1/B1=92.947/87.979ms, ratio1.056472 above1.03 |
| 2 | 54/54 pass | None |
| 3 | 53/54 pass | C57 mixed warm p50 F1/B1=48.056/45.177ms, ratio1.063736 above1.03 |

These are separate per-restart outcomes. Neither failure is pooled away,
waived as cold-tail variance, or promoted to a kernel-causality conclusion.
The third-block failure is a median regression and remains under investigation.

Ranges below are min/max reductions over18 case/restart ratios, not pooled
percentiles. All measured warm p50/p95 points against original-native BF16 B0
pass; the two misses above concern the stricter same-policy B1 control.

| Comparison | Warm TTFT p50 reduction | Warm TTFT p95 reduction | Decode p50 reduction |
| --- | ---: | ---: | ---: |
| F1 versus main FP8 F0 | 21.2–36.7% | 24.2–55.9% | 1.6–10.4% |
| F1 versus original-native BF16 B0 | 20.6–31.3% | 19.8–59.2% | 6.45–13.8% |

All36 decode p50/p95 guards against main FP8 pass the1.03 bound; all36
BF16 decode-advantage points against B0/B1 pass. All12 isolated long-prefill
p50/p95 guards against main FP8 pass. Isolated controls have an8192-token
cached prefix,10warmup and400measured requests per query. Query size is the
new suffix, not the total context; q800 has8992 total prompt tokens.

| Restart | Main FP8 q800 p50 /p95 ms | F1 q800 p50 /p95 ms | Median reduction |
| --- | ---: | ---: | ---: |
| 1 | 78.088 /81.777 | 36.640 /37.411 | 53.08% |
| 2 | 74.656 /78.943 | 36.384 /37.172 | 51.26% |
| 3 | 78.467 /81.384 | 36.406 /37.058 | 53.60% |

All six q800 points meet the30% reduction target. The original FP8/BF16 q800
regression also repeats in all three baseline pairs: native FP8 p50 is34–41%
slower. Native warm p95 is slower in five of six cells in every pair; C57 mixed
p95 is faster in every pair. Mixed C13/C57 median comparisons reverse in the
third pair. Native C57 mixed decode p50 is0.83–1.66% slower in all three pairs,
so a universal native-FP8 decode advantage is not claimed.

## Retained descriptive counters

Cold tails remain separate/nonblocking by user direction:17/18 descriptive
cold calculations pass. The retained failure is block2 C57 mixed cold p95,
310.673ms F1 versus289.153ms F0 (+7.44%). It does not replace the two warm
failures or trigger a new cold-tail campaign.

Block2 C13 mixed q800-cohort p95 is57.537ms F1 versus48.745ms B1 (+18.04%,
49 requests each), although its full warm p95 point passes. Its warm p99 is
162.792ms versus152.787ms F0,122.527ms B0 and54.210ms B1. Block2 C57 warm
p99 is3.27% higher than B1. Other unfavorable cohort/p99 values remain in the
metrics and investigation log; they are not added acceptance gates or causal
claims about actual batched GEMM shapes.

## Model quality

The actual integrated native F1 completed all1319 five-shot GSM8K questions,
temperature0,seed42,max1024,concurrency4,thinking disabled. There are zero
errors or truncations. No baseline/BF16 model evaluation was repeated.

| Artifact | Established correct | Strict correct |
| --- | ---: | ---: |
| Required saved corrected-v2 flag-off FP8 reference | 1276 | 1256 |
| Actual integrated native F1 | 1277 | 1257 |

Both nondecrease gates pass. Paired wins831/951 and loss1016 were privately
reviewed as reasoning differences. The reference is saved flag-off FP8,
not a stronger genuine-main/BF16 provenance claim. An earlier full FP8 reference
scores1276/1257 under identical prompt/payload/scorer identities; against that
saved artifact, F1 gains one established answer and strict score is unchanged.
No statistical quality gain, equivalence or identical-text gate is asserted.
Candidate quality artifact SHA256:
`8b2c0494306b453898a6fc03ca38948e17b386f9b54796ee5912c160a62709dc`.

## Build, correctness and delivery-source checks

The normal CMake/TorchStable native integration is built, installed and loaded;
there is no required standalone ctypes library. Native SHA256:
`05868591290baf1097a6affae8604ae14c55f24e9318bcaa0bdd239987715db2`.
All8 ranks in every serving arm satisfy their source/native/provider,
attention/KV, census/FX and before/after control-restoration contracts.

| Validation | Result |
| --- | --- |
| Native chain / UE8 / existing RMSNorm tests | 41 /65 /1160 passed |
| Hybrid/TMA / graph-policy / selected offline detokenizer tests | 64 /32 /44 passed |
| Selected structured / selected final / original adapter proofs | 72 /9 /64 passed |
| Maximum M32768 compiled/CUDA-graph proof | Passed; final carry allocation elided |
| Delivery-only accelerator API adaptation | 64 two-device eager/compile/graph cases and2 native-error device restorations passed; CPU reject/fake metadata passed |
| Source packaging | Enumerated21-file proof still matches; CUDA lexical/math content unchanged; test type guards and generic accelerator APIs accounted for |

One pre-existing offline detokenizer case requiring an unavailable external
Gemma3 tokenizer was deselected. The selected72/9 structured/final checks do
not repeat all216/27 historical prototype configurations. These limits are
retained; no test tolerances were relaxed to obtain the results.

The delivery adapter check loaded actual delivery Python beside native058685,
compared the exact original wrapper bodies without double registration,
and tested noncurrent devices, nondefault streams, graph replay and exception
restoration. Its exact local-route comparison adds no model bitwise goal.
GPU validation proof SHA256:
`b93e790a964b7a5ff15a880c8e3e7cc965ee8179038fea208d1edeb697d5d89c`.

The portable quality helper uses project regex with an explicit C0 whitespace
class. CPU checks preserve strict predictions on3957 saved answers/golds,
4807 edgecases and the recorded Unicode-property scan; inference functions
retain their ASTs. See `protocol-equivalence.json` for versions, hashes and
scope. The measured client remains byte-identical to client9bf.

## Bounded cross-launch numerical explanation

Identical generated sources need not select the same FP32 reduction tree:
autotuning can choose different `R0_BLOCK` values across launches. In one loaded
FP8 model, changing only the first embedding scale/RMS reduction flipped1/29
or2/29 distinct q2048 first tokens (29 prompts repeated twice), including one
6.41992-nat original margin; restoring the original selections recovered all
58 token/top20 records. All48 isolated source-contract cases passed, with
normalized outputs within one BF16 ULP under the tested inputs/alias contract.
References: `reduction-causal-embedding-v2-fp8/reduction-analysis.json` and
`reduction-kernel-correctness/correctness.json`.

A later same-loaded-model control combined the historical eight-rank first-RMS
vector with the scoped local-attention q2048 SplitKV2→1 change. Together they
reproduced all58 historical FULL2048 first-token/top20 payloads; exact code
restoration recovered all58 originals. The independent
`q2048-heterovec-fa4-independent-audit.json` passes5262 saved-data checks. This
establishes sufficiency on that fixed control set, not the cause of every
GSM8K answer difference, the older PIECEWISE counterexample, or all logits.
It supplies no quality/performance pass. Bitwise output equality is **not** an
acceptance requirement; actual task-quality and matched performance evidence
remain the criteria.

## Provenance and review scope

Frozen plan SHA256:
`b6a94f066f4e322d12c87e72141cedc21b4aec3a17aa0862dd4cd608a6a8afcc`.
Final strict report SHA256:
`692ad62954f6e6ecb32bc9c3ebfb4768432c8da00a6dd27c251d656411d2702c`.
The portable final JSON is a checked export of those outcomes, not a new
runtime attestation or an automatic goal-completion decision. Exact historical
replay still requires the disclosed external corpus. Earlier unmatched V7
prototype results remain in `historical_metrics.json` and
[the root investigation log](../../../GEMMA4_FP8_SERVING_INVESTIGATION.md);
they are not substituted for this completed three-restart campaign.

AI assistance was used in implementation, review and validation. Known
producer/consumer prerequisites retain their source attribution. This is an
internal review branch; no upstream overlap review or PR submission is claimed.
The full goal stays open for the two warm misses and their remaining investigation.
