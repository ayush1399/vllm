# Gemma4 patched.4 CUDA crash investigation

Initial snapshot: 2026-11-03 12:41:59 UTC. Updated after the local fault at
12:43:24 UTC and CUDA debugger inspection. Prepared with AI assistance.

## Current result

**A GPU memory-address fault was reproduced on the exact patched.4 source with
Gemma4 PDL disabled. CUDA core dumps and cuda-gdb identify `combine_quant` as
the faulting kernel. The fix passed memory-safety, latency, real-model and
17-minute sustained-serving validation on local 8xH100 hardware.** See the
dated fix update at the end for the latest results. Earlier sections preserve
the investigation as it stood before the fix.

The local driver reports `CUDBG_EXCEPTION_WARP_OUT_OF_RANGE_ADDRESS`, whereas
the supplied production traces report `CUDA_ERROR_ILLEGAL_ADDRESS` at event
synchronization. The local run reproduces an engine-fatal GPU address fault in
the same source/fallback path; the supplied production logs alone cannot prove
that every production failure has this exact kernel attribution.

Two local LLM checks passed on the original checkout. A separate small diagnostic
demonstrated stale slot-mapping behavior when replacing a captured context tensor,
but it did not reproduce CUDA illegal memory access or establish the production
failure's cause. The exact-source server subsequently completed startup and
failed during the first three long-context requests, producing three CUDA core
dumps. The server and traffic driver have exited following the failure. See the
reproduction update below for the concrete evidence.

At the initial snapshot the primary checkout had no retained code/test/CI edits.
The pre-existing untracked `GEMMA4_BRANCH_COMPARISON.md` was left untouched.
This log is a new file. No commit, push, or PR was made.

## User instructions and scope

- Investigate the branch's illegal memory accesses and sampling RPC timeouts.
- Use `origin/perf/gemma4-fp8-mtp-fused-patched.4`, which the user identified as
  the deployed source.
- Focus on the path used when Gemma4 PDL does not link.
- The user clarified that patched.0/.2 source comparisons are unnecessary;
  continue directly with patched.4.
- Save everything tried so far in a Markdown file.

## Production evidence

The user supplied `/home/ayushman/p4-crash/`, containing:

- `fp8-p4-node6-warmup-c25.launch.txt`
- `fp8-p4-node6-warmup-c25-try1.vllm.log`
- `fp8-p4-node6-warmup-c25.pod.json`
- `fp8-p4-mtp-node2-warmup-c25.launch.txt`
- `fp8-p4-mtp-node2-warmup-c25-try1.vllm.log`
- `fp8-p4-mtp-node2-warmup-c25.pod.json`

No CUDA core dump was present among those six supplied files.

The user's Cloud Logging aggregate was:

| Observation | Value |
| --- | --- |
| Fatal engine errors, all patched.4 | 65 |
| Non-MTP / MTP | 33 / 32 |
| CUDA illegal-address errors | 53 |
| `sample_tokens` RPC timeouts | 12 |
| Errors observed on patched.0 / patched.2 | None reported |
| Example time after serving began | About 1.5 minutes |
| Example active requests / KV usage | 3 / about 1.5% |
| Some successful runs | A full 16-minute level |

This is an intermittent runtime failure, not merely a startup failure or a
high-load-only failure. The aggregate counts do not establish 65 independent
kernel faults. Repeated CUDA messages can follow a sticky CUDA-context error;
whether all 12 timeouts have that same origin remains unconfirmed.

### First failures in the supplied logs

| Run | First reported CUDA failure | Location |
| --- | --- | --- |
| Non-MTP node 6 | 2026-11-03 06:39:33 UTC | Log lines 618–632 |
| MTP node 2 | 2026-11-03 06:56:24 UTC | Log lines 782–796 |

Both first traces report `Error getting async model runner output`, followed by:

```text
enqueue_output -> output.get_output()
vllm/v1/worker/gpu/async_utils.py:175
self.copy_event.synchronize()
CUDA_ERROR_ILLEGAL_ADDRESS from cuEventSynchronize
```

The event synchronization reports an earlier asynchronous CUDA failure; it does
not identify the kernel that caused it. Later entries explicitly report a sticky
error. Neither supplied log contains the exact `RPC call ... timed out` text.

The scheduler dumps at failure show:

- Non-MTP: 3 scheduled decode tokens, 3 requests, computed-token counts
  `[21152, 21109, 26103]`, KV usage approximately `0.01504`.
- MTP: 15 scheduled verification tokens, 3 requests, k=4, computed-token counts
  `[21237, 21159, 33625]`, KV usage approximately `0.01840`.

Because async execution can queue work, these scheduler dumps are useful context,
not proof that the displayed step contains the faulting kernel.

### PDL evidence

Both production runs show Gemma4 PDL extension link failures on all ranks:

```text
/usr/bin/ld: cannot find -lcuda: No such file or directory
Gemma4 PDL unavailable; using existing GEMMs
```

This excludes execution of the unavailable Gemma4 PDL extension in those runs.
It does not disable the branch's other specialized attention, cache-write,
normalization, activation, or FP8 paths.

The current local reproduction explicitly uses `"gemma4_pdl":false` to select
the same GEMM fallback without requiring a deliberate link failure. The extension
was also absent from the inspected TP0 worker memory mappings during startup.
An unrelated log saying DeepGEMM PDL is enabled refers to DeepGEMM's own machinery,
not successful loading of the branch's Gemma4 PDL extension.

### Relevant production launch configuration

- Image: `vllm-openai-sesame:0.32.0.patched.4`.
- Reported vLLM version: `0.33.0rc1.dev127+ge6a6bf7cb`.
- 8 H100 80GB GPUs, TP8, BF16 activations, automatic KV dtype.
- Target `/models/pt14-FP8-block`.
- MTP arm additionally uses `/models/gemma-4-31B-it-assistant-FP8-block`, k=4.
- `--max-model-len 131072`, `--max-num-batched-tokens 32768`.
- `--gpu-memory-utilization 0.85`.
- FlashAttention 4, `decode_backend=latency_specialized`.
- Default `FULL_AND_PIECEWISE` graph policy and capture ladder through 512.
- `VLLM_USE_FLASHINFER_SAMPLER=1`.
- Chunked prefill, prefix caching, and async output are active.
- Replay workload: concurrency 25, duration 420s, warmup 60s, start stagger 300s,
  seed 17. The private replay workload itself was not run locally.

## Source and environment

### Original checkout

Path: `/home/ayushman/vllm-ayush`.

```text
Branch: perf/gemma4-fp8-mtp-fused
HEAD:   5ee91920dc530dd748fbb49df4b63a45200d55ea
```

The initial two LLM checks and isolated context diagnostics used this checkout.
They must not be presented as tests of the exact production source.

### Exact deployed source fetched later

```bash
git fetch origin perf/gemma4-fp8-mtp-fused-patched.4
git worktree add --detach /jobtmp/gemma4-patched4-crash \
  origin/perf/gemma4-fp8-mtp-fused-patched.4
```

The fetched SHA exactly matches the crash image identifier:

```text
e6a6bf7cba3adfaf45573a52ef1eb452de8972ad
```

Compared with the original checkout, the final tracked-tree diff is four files:

```text
docker/Dockerfile
vllm/v1/attention/backends/flash_attn.py
vllm/v1/worker/gpu/model_runner.py
vllm/v1/worker/gpu/sample/gumbel.py
```

Those changes add source packaging and attention/sampler warmup coverage. The
specialized Gemma4 runtime and CUDA producer source are unchanged between these
two checked trees. Source review alone does not exclude warmup/cache effects.

`git ls-remote --heads origin 'perf/gemma4-fp8*'` found the fused, patched.4,
serving, and serving-clean branches; it did not find patched.0/.2 branches.
Further comparison with those versions was dropped following user direction.

### Local runtime provenance and limitations

- 8 locally available H100 80GB GPUs; initially idle.
- `.venv` points to `/jobtmp/gemma-main-env/.venv`.
- Torch observed locally: `2.14.0+cu132`.
- Python commands used `.venv/bin/python`; no system Python or bare pip was used.
- The editable environment's generated package version reports an older build
  identifier (`0.1.dev422+g18e879113`). That label is not proof of imported source.
- For the exact-source worktree, `PYTHONPATH` points at that worktree. Imports of
  `vllm`, `gumbel.py`, and `flash_attn.py` were checked to resolve there.
- Existing compiled extensions were linked into the worktree. No CUDA rebuild
  was performed; the relevant CUDA source is unchanged between the checkouts.
- Generated FlashAttention CuTe files were copied from the existing build; the
  adaptive-SM90-split and output-block-scale patches were verified present.
- Other missing generated dependency files were linked without overwriting
  tracked files. The fetched worktree has no tracked source edits.
- The local model is `/models/RedHatAI/gemma-4-31B-it-FP8-block`, not the private
  `/models/pt14-FP8-block` used in production.
- The local reproduction is not a run of the production container image or its
  exact private request stream. It uses the exact fetched vLLM source with the
  available local dependencies and synthetic inputs.

## Work performed and results

### 1. Initial source and log inspection

Read the `debug-ima` skill and its referenced vLLM CUDA-core-dump guide:
<https://vllm.ai/blog/2025-08-11-cuda-debugging>.

Reviewed the branch diff, existing validation notes, and relevant runtime code:

- `vllm/models/gemma4/nvidia/specialized_decode/attention.py`
- `attention_prep.cu`, including cache-write indexing and `combine_quant`
- `model.py`, `config.py`, `linear.py`, `pdl.py`, and MTP selection
- `cmake/patches/flash_attn_sm90_adaptive.patch`
- `cmake/patches/flash_attn_zero_window.patch`
- `vllm/v1/attention/backends/flash_attn.py`
- Multiprocess executor RPC timeout handling and async output synchronization
- Graph partition defaults and the newer GPU runner's metadata/slot buffers
- Existing attention/kernel and Gemma4 MTP regression coverage

Searched local Gemma4 logs for earlier failures. An older unrelated native-smoke
run had an illegal access under `/jobtmp/gemma4-fp8/pass4/`; it was not identified
as the supplied production incident and was not used as reproduction evidence.
The existing `GEMMA4_BRANCH_COMPARISON.md` also concerns earlier experiments and
does not establish the cause of this incident.

### 2. Isolated piecewise-context diagnostic

Script: `/tmp/repro_gemma4_piecewise_context.py`.

The script invokes the real specialized cache-preparation operator through a
small compiled module. It captures with one slot-mapping tensor, replaces the
context with a different tensor pointing to another KV page, and replays.
The minimal context holder is synthetic; this is a diagnostic, not a committed
production regression test or an LLM reproduction.

```bash
VLLM_DISABLE_COMPILE_CACHE=1 .venv/bin/python \
  /tmp/repro_gemma4_piecewise_context.py

VLLM_DISABLE_COMPILE_CACHE=1 .venv/bin/python \
  /tmp/repro_gemma4_piecewise_context.py --split

VLLM_DISABLE_COMPILE_CACHE=1 .venv/bin/python \
  /tmp/repro_gemma4_piecewise_context.py --inductor
```

| Variant | Observed result | Log |
| --- | --- | --- |
| Eager compilation backend, no added split point | Old page written; new page untouched; assertion failed | `/tmp/repro_gemma4_piecewise_context.log` |
| Eager backend, explicit cache-prep split point | Old page untouched; new page written; passed | `/tmp/repro_gemma4_piecewise_context_split.log` |
| Inductor backend, no added split point | Old page written; new page untouched; assertion failed | `/tmp/repro_gemma4_piecewise_inductor2.log` |

Representative nonzero counts:

```text
No split, eager:    old page 4096; current page 0
Explicit split:    old page 0;    current page 4092
No split, Inductor: old page 4094; current page 0
```

An initial script attempt failed with `KeyError: 'probe'` because the per-layer
attention metadata entry was missing. Adding `{"probe": None}` corrected the
diagnostic setup. One initial invocation used `--inductor` before the script
recognized that option; `...inductor2.log` is the explicitly configured Inductor
run and is the relevant evidence.

**Limit:** The real GPU runner deliberately reuses persistent slot-mapping and
block-table storage. Replacing the slot tensor in this diagnostic does not
reproduce that production contract. Earlier commentary described this too
broadly as a confirmed graph bug; the established result is narrower. No CUDA
illegal access occurred in these diagnostics.

### 3. Short LLM check on the original branch

A temporary test was added to the existing Gemma4 MTP test file, using the real
`vllm_runner`/`LLM` entry point, local FP8 weights, TP8, FA4, PDL disabled, and
`PIECEWISE` graphs with capture sizes `[1, 2, 4, 8, 16, 32]`.

It generated greedy answers to one-request and two-request arithmetic batches,
checking the observable integer answers. Model length and batched-token limit
were 2048. This was non-MTP target execution.

```bash
GEMMA4_FP8_MODEL=/models/RedHatAI/gemma-4-31B-it-FP8-block \
VLLM_DISABLE_COMPILE_CACHE=1 .venv/bin/python -m pytest \
  tests/v1/e2e/spec_decode/mtp/gemma4/test_mtp.py::test_fp8_piecewise_graphs_use_current_attention_context \
  -v -s > /tmp/gemma4-piecewise-e2e-before.log 2>&1
```

Result: **1 passed**, 150.33 seconds, no illegal memory access.

### 4. Long-context LLM check on the original branch

The same temporary test was changed to use approximately 20K-token synthetic
background prompts, maximum model length 32768, chunked prefill enabled, and
8192 maximum batched tokens. It retained greedy decoding, non-MTP execution,
PDL disabled, and the same small PIECEWISE capture ladder.

The pytest command was the same, with output redirected to:

```text
/tmp/gemma4-piecewise-long-e2e-before.log
```

Result: **1 passed**, 110.20 seconds, no illegal memory access.

Neither LLM check reproduced the production request stream or its default
FULL_AND_PIECEWISE graph ladder. Their passing results do not establish
patched.4 stability.

The tentative test and its Buildkite wiring were saved to:

```text
/tmp/gemma4-context-investigation.patch
```

Both repository edits were then removed. No graph-boundary fix was applied,
because these LLM checks did not fail before a fix.

### 5. Exact-source worktree setup attempts

The first worktree import check failed because generated FlashAttention native
extensions were absent. An attempted directory symlink landed inside the existing
tracked package directory; it was removed. Individual extension links and copied
CuTe sources corrected that setup.

The first server launch then failed during imports with:

```text
ModuleNotFoundError: No module named
'vllm.third_party.flashmla.flash_mla_interface'
```

Linking the remaining ignored/generated dependency files corrected that setup.
These were environment/import failures, not reproductions of the CUDA incident.
The server was restarted; its current `server.log` is from that restart.

## Current exact-source sustained reproduction

Worktree: `/jobtmp/gemma4-patched4-crash`.
Artifacts: `/jobtmp/gemma4-patched4-crash-artifacts/`.

### Server command

Run from the worktree:

```bash
env \
  PYTHONPATH=/jobtmp/gemma4-patched4-crash \
  VLLM_CACHE_ROOT=/jobtmp/gemma4-patched4-crash-artifacts/cache \
  VLLM_USE_FLASHINFER_SAMPLER=1 \
  VLLM_USE_DEEP_GEMM_E8M0=1 \
  CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 \
  CUDA_COREDUMP_SHOW_PROGRESS=1 \
  CUDA_COREDUMP_GENERATION_FLAGS=skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory \
  CUDA_COREDUMP_FILE='/jobtmp/gemma4-patched4-crash-artifacts/cuda_coredump_%h.%p.%t' \
  OMP_NUM_THREADS=1 \
  .venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model /models/RedHatAI/gemma-4-31B-it-FP8-block \
    --served-model-name gemma4-crash-repro \
    --host 127.0.0.1 --port 30174 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 --tensor-parallel-size 8 \
    --max-num-batched-tokens 32768 \
    --language-model-only \
    --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
    --attention-backend FLASH_ATTN \
    --attention-config '{"flash_attn_version":4}' \
    --additional-config '{"decode_backend":"latency_specialized","gemma4_pdl":false}' \
    --disable-uvicorn-access-log \
    > /jobtmp/gemma4-patched4-crash-artifacts/server.log 2>&1
```

Unlike the earlier LLM checks, this leaves the graph policy and capture sizes at
their server defaults. It preserves the production maximum context/batch limits
and FlashInfer sampler selection. The target is public local weights and the
initial sustained run is non-MTP, since production also fails without MTP.

### Traffic driver

Script: `/jobtmp/gemma4-patched4-crash-artifacts/traffic.py`.

```bash
env PYTHONPATH=/jobtmp/gemma4-patched4-crash \
  .venv/bin/python /jobtmp/gemma4-patched4-crash-artifacts/traffic.py \
  > /jobtmp/gemma4-patched4-crash-artifacts/traffic.log 2>&1
```

It waits for `/health`, then sends synthetic token-ID prompts to the local
`/v1/completions` endpoint. It records request metadata/status/usage/latency in
`traffic.jsonl`; it does not copy production prompts or answers.

| Phase | Concurrent clients | Duration after readiness | Prompt lengths |
| --- | --- | --- | --- |
| `decode-c3` | 3 | 240s | 21109, 21152, 26103, 33625 |
| `mixed-c25` | 25 | 480s | 80–33625, including lengths around graph boundaries |
| `mixed-c3` | 3 | 300s | 80, 255, 513, 21109, 26103 |

Requested output lengths cycle through 121/78/206/160 tokens with EOS ignored.
Temperatures cycle through 0/0.6/0.8/1.0, top-p varies, some requests have seeds,
and some use top-k=50. This exercises sampling paths skipped by greedy-only
checks. The client stops issuing new requests after an HTTP/client failure.

### State at initial snapshot, before the reproduced fault

- Server API process PID: `4074147`.
- Traffic process PID: `4073756`.
- Current execution sessions: server `21244`, traffic `48781`.
- Model compilation and initial graph/kernel warmup completed or advanced;
  DeepGEMM warmup was in progress, with the latest displayed progress passing
  2012 of 4020 cases. No serving traffic had completed yet.
- No local CUDA dump had been observed at the last artifact check.
- Both processes were left running; they were not stopped to write this log.

These PIDs/session IDs and progress are snapshot data, not a promise of their
state when this file is read later.

### Reproduction update: 2026-11-03 12:43:24 UTC

The first `decode-c3` wave failed. No concurrency-25 phase was needed or reached.

| Request | Prompt tokens | Temperature | Result |
| --- | --- | --- | --- |
| 0 | 21109 | 0.0 | HTTP 500 after about 15.47s |
| 1 | 21152 | 0.6 | HTTP 500 after about 15.47s |
| 2 | 26103 | 0.8 | HTTP 500 after about 15.47s |

The CUDA driver recorded address faults on multiple ranks at approximately
12:43:24. The engine reported its fatal failure at 12:43:36. Its scheduler dump
shows three running decode requests, three scheduled tokens, computed-token
counts `[21111, 21153, 26103]`, output-token counts `[3, 2, 1]`, and approximately
2.81% KV-cache usage. This confirms reproduction under light decode load with
Gemma4 PDL disabled and without MTP.

The first driver stack is at `server.log` lines 557–563:

```text
Detected an exception of type CUDBG_EXCEPTION_WARP_OUT_OF_RANGE_ADDRESS (5)
Error PC ... (combine_quant)
#0 ... _Z13combine_quantPKfS0_PKiS2_P13__nv_fp8_e4m3Pfiiiiilll
```

Three approximately 248MiB CUDA dumps were produced under the artifact directory:

```text
cuda_coredump_gemma4-bench-sandbox.4074244.1793709226
cuda_coredump_gemma4-bench-sandbox.4074249.1793709226
cuda_coredump_gemma4-bench-sandbox.4074252.1793709226
```

Debugger command:

```bash
/usr/local/cuda/bin/cuda-gdb --batch \
  -ex 'target cudacore /jobtmp/gemma4-patched4-crash-artifacts/cuda_coredump_gemma4-bench-sandbox.4074244.1793709226' \
  -ex 'bt' -ex 'info cuda kernels' \
  > /jobtmp/gemma4-patched4-crash-artifacts/cuda-gdb-first.txt 2>&1
```

The debugger confirms:

```text
CUDA Exception: Warp Out of range Address
combine_quant(...) <<<(4,8,1),(128,1,1)>>>
block (3,2,0), thread (32,0,0), device 1
```

Block x=3 is the fourth physical query row in this four-row launch, while the
batch has only three real decode tokens. This strongly supports investigating
the padded-row candidate below. It does not, by itself, reveal the exact value
read from the split-count buffer or establish the final code-level fix.

A second debugger invocation saved `x/24i $pc-96` successfully, with output at
`cuda-gdb-instructions.txt` in the artifact directory. The first backtrace above
is already sufficient to attribute the local fault to `combine_quant`.

The API process and traffic process had both exited at the subsequent process
check. `traffic.jsonl` contains three records. No serving process was deliberately
left running after this failure.

## Open hypotheses, not established causes

### Context capture and missing graph boundaries

The specialized `gemma4_attention_prep_cached`, `gemma4_attention_fp8`, and
`gemma4_norm_qkv_prep` operators are not in the default attention split-point
list. They read dynamic forward context internally. The replacement-buffer
diagnostic shows why that can matter, but the real runner's persistent buffers
and the two passing LLM checks prevent attributing this incident to that alone.

### Padded rows in fused attention combine

The custom `combine_quant` kernel launches over physical rows and reads
`splits[batch]` without first excluding rows beyond `cu[batches]`.
The adaptive attention kernel initializes split counts only for valid query
tiles. A padded zero-query request may therefore lack a fresh split-count write.

FULL graph metadata uses padded request/token counts, so a three-request decode
batch can be relevant to a four-row graph. Graph-pool memory reuse may make
stale temporary contents important. This is a source-level candidate only:
the natural three-request reproduction now faults in this kernel on the fourth
physical row. A focused padded-row test and a verified fix are still outstanding;
no sanitizer run or post-fix validation has been performed.

## Tooling and UI notes

The default shell sandbox failed at startup with:

```text
bwrap: Failed to make / slave: Permission denied
```

Shell investigation commands therefore used reviewed escalation. There was no
automatic-review rejection preventing the investigation. Git fetch printed a
missing VS Code `node` helper warning but fetched successfully.

The user reported messages displaying “this content can't be shown” and asked
about a cybersecurity classifier. No classifier decision/reason is visible to
the assistant, and no causal explanation for that notice was established.
Official OpenAI documentation acknowledges false positives, including unrelated
activity, and recommends `/feedback` for suspected Codex false positives:
<https://learn.chatgpt.com/docs/cyber-safety#false-positives>.
This documentation lookup did not diagnose the CUDA crash or change its scope.

## Next technical steps

1. Inspect the captured `combine_quant` fault-site instructions and the handling
   of rows beyond the logical query count.
2. Build a focused padded-query-row regression from the now-observed failure.
3. Preserve the three-request, PDL-disabled server reproduction as the real
   entry-point validation, including its core-dump attribution.
4. Apply a targeted fix only with evidence, then show failure before and success
   after through an appropriate real entry point; retain intermittent-run limits.

No passing short run, source hypothesis, or aggregate error count should be
reported as a reproduced and fixed production illegal memory access.

## Fix update: 2026-11-03, validation in progress

### Root cause and repair

The faulting fourth row belongs to graph padding, not a live request. The
original combine kernel reads its uninitialized split count and iterates beyond
`weights[128]`. In the captured core, R9 contained 1068515200 (0x3fb03f80),
R5 was already 144, and the faulting instruction was `LDS.64 R16,[R10+0x28]`.
The adaptive producer does not initialize metadata for zero-query requests.

The repaired kernel tests `row < cu[batch + 1]` before reading `splits[batch]`.
Inactive rows use count zero, never read LSE/partial buffers, and produce zero
FP8 output and unit scales. Active rows retain the original exponential,
normalization, BF16 rounding and FP8 conversion arithmetic.

A simple early return was safe but raised latency, so it was rejected. Several
subsequent implementations were measured rather than accepted on intuition:

| Experiment | Observation / decision |
| --- | --- |
| Early-return and lookup-only guards | Roughly 1–35% slower in measured cases; rejected |
| Smaller loop unroll | Hurt high split counts; rejected |
| Warp-sized normalization and integer amax reduction | Removed synchronization/reduction overhead |
| Broad non-power-of-two split sweep | Exposed loop remainder cliffs at counts 5, 8, 24 |
| Constant small-count partial accumulation | Removed those cliffs without changing FMA order |
| Eight-split preload and launch bounds | Improved instruction scheduling, but single-row overhead remained |
| Single-row 4/8-split load overlap (final v19) | Overlaps LSE and partial loads; preserves the original balanced sum tree; passes final latency matrix |

Intermediate benchmark JSON, generated kernels, build logs and sanitizer logs
are retained under `/jobtmp/gemma4-combine-fix/`. Earlier v13/v14/etc. results
are exploratory, not the final validation evidence.

The final CUDA source SHA256 is
`c433c987e56b6131887e520f8bd7b0d1c9a1767f4c4f56de414b76ff1433f825`.
It is installed in the main checkout and the detached exact patched.4 worktree.
The baseline kernel source SHA256 is
`01b225f55e6ecb675e34d79c7f77cb02168e59afa8a70be619e7da0130d06017`;
the original CUDA source is identical in main HEAD and patched.4.
The exact patched.4 source SHA is `e6a6bf7cba3adfaf45573a52ef1eb452de8972ad`.
The local generated package version label is not the production container label.

### Completed correctness and memory checks

- Focused regression fails before the fix with CUDA illegal memory access:
  `/jobtmp/gemma4-combine-fix/kernel-before.log`.
- Final regression: **88 passed under Compute Sanitizer, ERROR SUMMARY: 0 errors**:
  `/jobtmp/gemma4-combine-fix/memcheck-final-v19.log`.
  Cases cover head dimensions 256/512, query counts 1/5, split counts
  4/5/8/9/16/24/32/33/64/96/128, and one/four request slots. Graph replay
  changes logical request count, poisons unused split counts with 1<<30,
  and asserts exact active FP8 bytes/scales plus zero/unit padding.
- Final benchmark also compares baseline and candidate active FP8 bytes/scales
  exactly before measuring each shape, including poisoned padding.
- The new real-model `test_fp8_decode_graph_padding[0]` fails with the original
  binary. The worker reports CUDA illegal memory access and the engine later
  times out: `/jobtmp/gemma4-combine-fix/e2e-before.log` (439.32 seconds).
  Seven orphaned baseline workers needed explicit termination before continuing.
- An earlier combined FA4 sanitizer invocation reported 14 host capability-probe
  errors (`cuGetProcAddress_v2`, CUDA version 13041 vs driver 13040). Do not
  describe that earlier combined invocation as clean. The final standalone
  88-case combine regression above has zero errors.

Reproduce the final standalone memory check from the exact worktree:

```bash
env CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/jobtmp/gemma4-patched4-crash   compute-sanitizer --tool memcheck --error-exitcode 1   .venv/bin/python -m pytest tests/kernels/attention/test_flash_attn.py   -k gemma4_combine_quant_ignores_padded -v
```

### Completed final latency checks

Paired CUPTI kernel durations, both arms compiled into one library with the same
flags, CUDA graph replay, alternating AB/BA order, seeded inputs, compilation
and allocation outside timing. H100 80GB SXM, four local query heads, BF16-rounded
FP32 attention combination to FP8. Cold L2 is the default; warm L2 is separate.
No case in the following matrices has higher measured candidate latency:

| Matrix | Cases | Median reduction | Worst candidate/baseline ratio |
| --- | ---: | ---: | ---: |
| Full cold L2 | 64 | 17.00% | 0.945755 |
| Full warm L2 | 64 | 18.54% | 0.931034 |
| Broad splits, cold L2, 1530 MHz lock | 66 | 15.74% | 0.991857 |

Full matrices: physical/logical rows 1/1, 4/4, 4/3, 5/5, 8/8, 16/15,
20/20, 31/31; dimensions 256/512; splits 4/9/32/128; four paired passes,
100 measured iterations per arm/pass. Broad matrix: rows 1/1, 4/3, 20/20;
dimensions 256/512; splits 4/5/8/9/16/24/32/33/64/96/128; six passes,
200 iterations. The broad run locked GPU 0 to 1530 MHz, recorded telemetry,
and reset the clock on exit. Other matrices ran on separate idle GPUs.

Raw reports: `/jobtmp/gemma4-combine-fix/validated-full-cold.json`,
`validated-full-warm.json`, `validated-broad-cold.json`. Source hashes in all
three reports match the final source above. The reusable harness is
`benchmarks/kernels/benchmark_gemma4_combine_quant.py`.

```bash
.venv/bin/python benchmarks/kernels/benchmark_gemma4_combine_quant.py   --output /tmp/combine-cold.json
.venv/bin/python benchmarks/kernels/benchmark_gemma4_combine_quant.py   --warm-l2 --output /tmp/combine-warm.json
.venv/bin/python benchmarks/kernels/benchmark_gemma4_combine_quant.py   --quick --split-counts 4 5 8 9 16 24 32 33 64 96 128   --repeats 6 --iterations 200 --output /tmp/combine-broad.json
```

These are measured kernel latency claims for the stated matrix and hardware,
not an end-to-end serving speedup claim or a guarantee for every conceivable
shape/device. Sustained serving and real MTP validation remain in progress.

### Real-model and final suite update (14:27 UTC)

The final binary passed both real `LLM.chat` cases:
`test_fp8_decode_graph_padding[0]` (non-MTP) and `[4]` (MTP, four speculative
tokens). Both use TP8, FA4 and `gemma4_pdl=False`; each generates 16 tokens
for three roughly 20K-token prompts and checks answers 56, 27 and 4.
Local assistant weights were `/models/google/gemma-4-31B-it-assistant-FP8-block`.
Result: **2 passed in 289.32 seconds**; log:
`/jobtmp/gemma4-combine-fix/e2e-after.log`. This pairs with the same `[0]`
entry point failing on the original binary with CUDA illegal memory access.
The model runner emitted a shared-memory cleanup warning after the MTP case;
all GPU workers exited and the GPU process list was empty afterward.

Final combined kernel suite: **96 passed in 67.39 seconds**, covering the
88 padding cases plus eight existing adaptive-split/FA4 combine cases:
`/jobtmp/gemma4-combine-fix/kernel-final.log`.

```bash
env GEMMA4_FP8_MODEL=/models/RedHatAI/gemma-4-31B-it-FP8-block \
  GEMMA4_MTP_MODEL=/models/google/gemma-4-31B-it-assistant-FP8-block \
  PYTHONPATH=/jobtmp/gemma4-patched4-crash \
  VLLM_CACHE_ROOT=/jobtmp/gemma4-patched4-crash-artifacts/cache \
  .venv/bin/python -m pytest \
  tests/v1/e2e/spec_decode/mtp/gemma4/test_mtp.py::test_fp8_decode_graph_padding \
  -x -v -s

env CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/jobtmp/gemma4-patched4-crash \
  .venv/bin/python -m pytest tests/kernels/attention/test_flash_attn.py \
  -k 'gemma4_combine_quant_ignores_padded or fa4_combine_quant_preserves or fa4_adaptive_splits_read_current' \
  -q
```

The final build, installed main binary and candidate snapshot all have SHA256
`4617f739687f893c6ec5f1bae2bf31c2721cce06bd3f9bccb36d0788d9773aea`.
The exact-source worktree loads that installed binary through its symlink.

All repository pre-commit checks pass for the changed code, CI, benchmark and
investigation record: `/jobtmp/gemma4-combine-fix/precommit-final.log`.
Lint initially rejected a new `torch.cuda.synchronize` call in the benchmark.
The harness now uses `torch.accelerator` and its stream `native_handle`.
A smoke test caught that the generic stream lacks `.cuda_stream`; that host
API issue was corrected and the six-case smoke run passed. This does not alter
kernel code or the source hashes in the final latency reports.
The tests are selected explicitly by the existing 8xH100 Gemma4 CI job.

The sustained server was launched at 14:26 UTC using
`/jobtmp/gemma4-combine-fix/serving/run_server.sh`. Its traffic driver is an
unmodified copy of the original reproducer, in a separate directory to preserve
pre-fix logs. It waits for server health before running the 240/480/300-second
concurrency 3/25/3 phases. Logs are in that same `serving/` directory. Results
will be recorded once the full run finishes.

## Final validation result: 2026-11-03 14:46 UTC

**The current fix passes the validation gates for the reproduced combine kernel
fault, without a measured kernel latency regression in the tested matrix.**

The exact patched.4 server with the fixed native kernel completed the full
17-minute traffic schedule (240/480/300 seconds, concurrency 3/25/3).
The client exited with status 0. All **3,655 requests returned HTTP 200**, with
**516,250 completion tokens**. Every response had the requested prompt length
and requested completion length. Prompt lengths ranged from 80 to 33,625 tokens.

| Phase | Concurrency | Scheduled duration | Successful requests |
| --- | ---: | ---: | ---: |
| Long-context decode | 3 | 240 s | 313 |
| Mixed short/long contexts | 25 | 480 s | 2,716 |
| Mixed contexts after load | 3 | 300 s | 626 |

There were **zero request failures, CUDA errors, RPC timeouts, engine-fatal
errors, or CUDA core dumps** during the run. The final `/health` returned 200.
Prometheus recorded `request_success_total{finished_reason="length"}=3655`,
with error/abort counts zero and running/waiting queues zero. The client wall
time was 1153.09 seconds including server startup; startup is not counted as
part of the scheduled 17 minutes of traffic.

Final artifacts:

- `/jobtmp/gemma4-combine-fix/serving/validation-summary.json`
- `/jobtmp/gemma4-combine-fix/serving/traffic.jsonl`
- `/jobtmp/gemma4-combine-fix/serving/traffic.log`
- `/jobtmp/gemma4-combine-fix/serving/server.log`
- `/jobtmp/gemma4-combine-fix/serving/final-metrics.txt`
- `/jobtmp/gemma4-combine-fix/serving/run_server.sh`

The replay driver is unchanged from the pre-fix reproducer. The original server
crashed on its first three long-context requests. The final source and binary
hashes match those tested by the sanitizer and real-model checks above.
Gemma4 PDL is explicitly disabled throughout this replay. DeepGEMM's separate
PDL log does not enable the Gemma4 PDL path.

Completion evidence:

- Original failure reproduced and located in `combine_quant` by CUDA core dump.
- Regression fails on the original binary with CUDA illegal memory access.
- Final padded-row regression: 88 sanitizer cases, zero reported errors.
- Final existing/new kernel correctness suite: 96 passed.
- Real-model non-MTP and MTP k=4 cases: both passed, expected answers verified.
- Final kernel timing: 194 paired comparisons, none slower in the measured
  matrices; median latency reductions about 16–19%. Active FP8 bytes/scales
  match the baseline exactly before timing.
- Full sustained fallback-path server replay: 3,655/3,655 successful requests.
- Source/binary fingerprints verified; relevant pre-commit checks pass.

This validates the local reproduction and common kernel repair using public
Gemma4 FP8 target/assistant checkpoints on H100 TP8. It does not establish that
every production timeout had this cause, or replace deployment validation with
the private production checkpoint/container. The 17-minute serving replay was
non-MTP; MTP was checked through the real-model k=4 regression and shared-kernel
memory/correctness coverage. No production deployment, commit, push or PR was
performed. The changes remain in the main working tree and the detached exact
patched.4 worktree for review. The validation server is shut down after the run.
