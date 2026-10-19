# PCP for multimodal MLA

## Current progress (2026-10-19)

This AI-assisted work is on branch `feat/pcp-deepseek-vl2`, based on upstream
main `e0adacc9a67a5f2b2d8ba508f7f522bc1ff83c6d`. The implementation enables full
DeepSeek-VL2 image inputs with MRV2 PCP. Runtime changes are limited to allowing
its architecture, skipping empty token embedding calls on idle PCP ranks, and
correcting the full checkpoint's default decoder layer count to 30. The existing
embedding gather, scheduler, attention, and collective paths are reused.

Completed validation:

- Worker/config suites: 137 passed, one gated-model test deselected.
- Distributed TP2/PCP2 regression: passed, comparing log probabilities against
  PCP1 with single-image, multi-image, mixed text/image, and idle-rank decode.
- Full-weight runs on eight H100s: TP1/PCP8 and TP2/PCP4 passed; prefix-cache
  repeats preserve exact output token IDs.
- Decode CUDA graph replay, TP2/PCP2/PP2, and filesystem encoder transfer passed
  independently, with output token IDs matching the eager baseline.
- ChartQA first 32 samples: normalized exact match 20/32 for baseline and both
  tested PCP topologies. This is a small development evaluation, not the official
  full benchmark. PCP did not improve latency on the short prompts tested.
- Affected-file pre-commit checks passed. Detailed commands, artifacts, and
  limitations are recorded below. Temporary `/tmp` artifacts are not committed.

Stock VL2 speculation is not validated: n-gram is rejected by generic PCP
validation, MTP config is unsupported, and the default DSpark draft constructor
is incompatible with VL2 even without PCP. Dense MLA+DCP and LoRA restrictions
remain. Tiny uses MHA and is outside this path. Graph testing covers decode;
encoder transfer covers the filesystem example connector. Combined mode
cross-products and RDMA transfer remain untested.

The agreed next model is **DeepSeek-V4-Flash-Vision-Exp**. We will add its
multimodal PCP support later, reusing compatible upstream V4 core PCP support.
Open [PR #43809](https://github.com/vllm-project/vllm/pull/43809) overlaps the
required V4 compressor and sparse-attention work; implementing that core here
would duplicate it. Vision Exp is not enabled or GPU-tested by this patch.
Other model integrations are outside the current scope. See the final section
for the latest main audit and the boundary for a future Vision Exp change.

## Initial investigation (historical)

Research snapshot: 2026-10-18. Local source commit: `ce49c47cbb241b59a32ef110079a5261dbd08e49`.
Upstream main rechecked: `e0adacc9a67a5f2b2d8ba508f7f522bc1ff83c6d`.
This initial investigation concerns prefill context parallelism, not DCP or
vision-encoder tensor parallelism. Its original hypotheses are retained below;
the current implementation and validation status above supersede them.

## Scope and verified status

The current [MRV2 PCP manager](vllm/v1/worker/gpu/pcp_manager.py)
rejects PCP when the model supports multimodal inputs:

```text
MRV2 PCP does not support MM inputs yet.
```

The runner passes a model capability flag into validation, so this check is not
limited to requests containing an image. The same validator separately requires
MLA and rejects encoder-decoder models and LoRA. Dense MLA with combined PCP+DCP
also has an explicit restriction. None of these checks should be removed without
establishing the corresponding execution contracts.

[Kimi-K2.5](vllm/model_executor/models/kimi_k25.py) is a concrete architecture to
inspect: its wrapper constructs a `DeepseekV2ForCausalLM` language model, runs a
vision tower/projector, and forwards embeddings into the decoder. Confirm the
selected checkpoint resolves to MLA and the intended runner/backend. This example
does not establish that every multimodal MLA model uses the same embedding path.

Do not confuse Kimi-K2.5's multimodal wrapper with the separate Kimi-K3/Kimi-Linear
MLA implementation. Similar names do not imply shared PCP behavior.

## Existing work and search limits

The GitHub API query `repo:vllm-project/vllm is:pr is:open PCP multimodal` returned
eight results on the snapshot date. Their titles did not establish a dedicated
implementation of MRV2 multimodal MLA PCP. Broad alternative PCP implementations
and adjacent work still require review; keyword searches do not prove absence.

Relevant search results include
[alternative PCP implementation #33403](https://github.com/vllm-project/vllm/pull/33403),
[shared PCP cache ownership #49741](https://github.com/vllm-project/vllm/pull/49741),
and [Qwen3-VL sequence parallelism #43580](https://github.com/vllm-project/vllm/pull/43580).
The last is not evidence of support for an MLA decoder or this MRV2 path.

No specific issue has been established here as the authoritative tracker for
this feature. Refresh searches for `PCP`, `multimodal`, `Kimi-K2.5`, `KimiK25`,
and `prefill context parallel` before proposing a PR. Read linked discussions and
inline reviews before deciding whether an implementation overlaps existing work.

## End-to-end contract

For each scheduled decoder token, every participating rank must agree on the
global token position and receive the correct text or media embedding for the
rows it owns. Reordering and gathering must preserve the mapping from media
placeholder spans to encoder output rows. The final sampled row must correspond
to the original request's final prompt token.

PCP of decoder tokens does not imply PCP of the vision encoder. The implementation
must define where media is encoded and how decoder ranks obtain encoder outputs.
Replicating encoder work, assigning encoder ownership, or distributing the encoder
are different designs with different memory and communication costs.

## Work to establish

### Embeddings and positions

- Trace processed media, placeholder spans, encoder outputs, projected embeddings,
  and decoder insertion through one actual request.
- Apply exactly the decoder's PCP row mapping to its input embeddings and position
  tensors. Preserve model-specific position metadata; do not assume all models use
  a single ordinary position vector.
- Handle a media span crossing PCP segment or prefill-chunk boundaries, multiple
  images, and ranks receiving no media rows. Preserve ordering across requests.
- Determine whether partial media-span scheduling is legal in existing scheduler
  and encoder contracts before designing a boundary-crossing test or execution path.

### Encoder execution and cache lifecycle

- Define encoder ownership and delivery of embeddings to all decoder ranks that
  need them. All ranks must execute compatible collectives, including text-only
  requests and batches where some ranks have no local media.
- Preserve encoder-cache identity, lookup, retention, and release. Avoid freeing an
  entry while another rank or a later prompt chunk still consumes it.
- Account for vision activations, cached outputs, decoder embedding buffers, and
  communication scratch in profiling. PCP may reduce decoder work while leaving
  encoder cost unchanged.

### Decoder, sampling, and execution modes

- Confirm the chosen MLA backend accepts the PCP layout, including chunked-context
  attention, KV writes, and cached history. Keep PCP+DCP a separately validated
  combination rather than assuming DCP compatibility.
- Restore output rows and select logits correctly after PCP reordering. Verify
  mixed text-only and image-bearing requests together.
- Exercise the same embedding path during profiling and dummy execution that real
  multimodal requests require. Graph support needs buffer and shape validation.
- Define initial coverage explicitly: image input, video input, encoder TP,
  prefix caching, speculative decoding, and PP are separate combinations. A narrow
  implementation can retain clear errors for combinations it does not support.

## Source entry points

- [PCP manager](vllm/v1/worker/gpu/pcp_manager.py)
- [MRV2 runner](vllm/v1/worker/gpu/model_runner.py)
- [Default model state](vllm/v1/worker/gpu/model_states/default.py)
- [Multimodal encoder runner](vllm/v1/worker/gpu/mm/encoder_runner.py)
- [Encoder cache](vllm/v1/worker/gpu/mm/encoder_cache.py)
- [Multimodal position handling](vllm/v1/worker/gpu/mm/rope.py)
- [Kimi-K2.5 wrapper](vllm/model_executor/models/kimi_k25.py) and
  [vision tower](vllm/model_executor/models/kimi_k25_vit.py)
- [MLA execution](vllm/model_executor/layers/attention/mla_attention.py)

These are tracing starting points; changing every listed file is not a requirement.

## Validation required before claiming support

| Behavior | Evidence |
| --- | --- |
| Embedding row assignment | Known synthetic encoder outputs land in the correct global decoder positions after PCP distribution |
| Boundary handling | Legal media spans at segment/chunk boundaries produce equivalent decoder inputs and outputs |
| Ragged batches | Uneven lengths, multiple media items, text-only requests, and empty local segments complete without collective divergence |
| Cache lifecycle | Encoder-cache reuse and release preserve outputs across repeated and chunked requests |
| Generation | PCP=1 and PCP>1 agree within justified numerical tolerances; text and media-conditioned evaluation remain stable |
| Serving performance | Equal-GPU TTFT, encoder time, decoder prefill time, peak memory, and decode latency are reported separately |

Extend existing [PCP manager tests](tests/v1/worker/test_gpu_pcp_manager.py),
[context-parallel tests](tests/distributed/test_context_parallel.py), and relevant
multimodal processing/embedding suites before creating a new suite. Start with
behavioral tests for row mapping and ownership; end-to-end media serving and model
evaluation are still required for a support claim.

Record checkpoint revision, backend, hardware, TP/PCP/DCP sizes, image resolution
or video frame count, prompt lengths, cache settings, and measurement variability.
A text-only accuracy run cannot validate multimodal embedding placement.

No GPU reproduction, implementation, benchmark, or model evaluation was run for
this note. A complete contribution needs a reproduced baseline failure, a scoped
change, refreshed duplicate checks, human review, and the repository PR checklist.

## Follow-up: independent PR feasibility

This is a better candidate than recurrent hybrids for work that does not depend
on open PRs. Main already implements MLA PCP. A restricted image-input adapter
does not inherently require GQA PCP #49564 or hybrid recurrence #59304. This is a
source-based feasibility assessment, not validated runtime support.

### Refreshed duplicate checks

Public GitHub REST searches (the CLI is unavailable) covered open PRs for
`PCP multimodal`, `PCP vision`, `PCP "MM"`, plus issue/PR searches for
`PCP KimiK25`, `PCP "Kimi-K2.5"`, and context parallelism with DeepSeek-VL2.
No dedicated multimodal MLA PCP implementation or authoritative feature issue
was identified. The eight broad `PCP multimodal` results are keyword matches,
not eight competing implementations.

- #33403 is an alternative PCP topology/runtime implementation. Its 32-file
  list includes the legacy `gpu_model_runner.py`, but no MRV2 MM/model-state
  implementation; its body does not claim multimodal support.
- #49741 implements GLM-5.2 owner-sharded KV history, not media embedding insertion.
- #43580 implements Qwen2.5/3-VL vision-tower sequence parallelism, not PCP of an
  MLA decoder. It is not a prerequisite for a replicated-encoder implementation.

Refresh these checks before opening a PR. The current evidence supports an
independent narrow scope, not a guarantee against unpublished work.

### Why the implementation may be smaller

The runner partitions the decoder batch before calling
`DefaultModelState.prepare_inputs_embeds`. `PCPManager.partition_batch` preserves
the original request IDs (including repeated IDs for separate segments) and
sets each segment's absolute start in `num_computed_tokens_np`.
`ModelState.gather_mm_embeddings` passes precisely those local IDs, lengths,
offsets, and local query boundaries to `EncoderRunner.gather_mm_embeddings`.
The latter already slices cached media outputs by overlap with a token window.
Thus local embedding assembly may already support ordinary PCP segments without
a new full-embedding all-gather. Prove that behavior before adding new machinery.

The encoder executes scheduled global media inputs separately from local decoder
embedding gathering. A first implementation can retain full encoder execution
and cache outputs on each PCP rank, using the existing TP behavior within each
rank's TP group. Verify identical encoder scheduling and compatible collectives,
including ranks with no local media. Encoder sharding/ownership can be deferred.

Scope initially to a selected image model with ordinary 1D decoder positions,
DCP=1, PP=1, eager execution, no LoRA/speculation, no pruning, no encoder
connector, and prefix caching off. Preserve clear guards for unsupported models
and combinations. In particular, `DefaultModelState.prepare_inputs` builds
M-RoPE from global request progress rather than PCP segment starts; generic
multimodal support must not be inferred from a 1D-position model passing.

### Model choice is a practical constraint

[Kimi-K2.5's config](https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/config.json)
confirms `kv_lora_rank=512`, 61 language layers, hidden size 7168, and 384 routed
experts. It is a concrete MLA target but full-weight validation needs substantial
hardware. A reduced dummy configuration can test routing and collectives, not
real-model accuracy or useful serving performance.

Do not choose DeepSeek-VL2-tiny solely because its wrapper uses DeepSeek-V2:
its [published config](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny/blob/main/config.json)
explicitly sets `use_mla=false` and `kv_lora_rank=null`. It would bring back the
GQA PCP dependency. Other VL2 variants require resolved-config/backend checks;
missing JSON fields alone do not establish their effective architecture.

### Proposed first work and review evidence

1. Extend `tests/v1/worker/test_encoder_runner.py` and
   `tests/v1/worker/test_gpu_pcp_manager.py`: known encoder rows must assemble
   identically to slicing the global embedding tensor into actual PCP segments.
   Cover repeated request IDs, split media spans, multiple images, mixed text/media
   requests, short/empty segments, and `is_embed` masks.
2. Reproduce the current MM rejection, then run a restricted diagnostic with the
   guard relaxed. Trace actual input embeddings, positions, KV writes, output
   restoration, dummy/profiling runs, and encoder-cache free/reuse events.
   Keep the diagnostic restriction until these contracts are demonstrated.
3. Add the smallest production change supported by that evidence. Potentially
   this is capability gating plus targeted integration fixes and tests; do not
   assume a major redistribution mechanism or promise a line-count estimate.
4. Validate real multi-GPU image serving and legal chunked-prefill boundaries,
   then compare logprobs/generation and an image-conditioned evaluation against
   PCP=1. Text-only evaluation is insufficient. Report equal-GPU TTFT and memory,
   separating encoder cost from decoder gains; replicated encoding may dominate
   image-heavy requests. Wire the distributed regression into Buildkite.

PR-checklist assessment: design fit is promising with a replicated encoder and
existing MLA path; correctness and performance remain unvalidated; avoid adding
MM-specific logic throughout the common runner; the eventual description needs
explicit model/config coverage, duplicate rationale, commands/results, model eval,
human review, and AI disclosure. No implementation or GPU tests were performed.

## Model selected for an 8xH100 development node

Use **`deepseek-ai/deepseek-vl2-small`** as the first real-weight target.
The [paper, Table 1](https://arxiv.org/html/2412.10302v1#S2) explicitly distinguishes
Small's MLA (rank 512) from Tiny's MHA. Its JSON omits several MLA fields, but
resolving `language_config` with the installed `DeepseekV2Config` confirms:

| Field | Resolved value |
| --- | --- |
| Language model type | `deepseek_v2` |
| KV latent rank | 512 |
| QK content / rotary dimensions | 128 / 64 |
| Value head dimension | 128 |
| Attention heads / layers | 16 / 27 |
| Maximum context | 4096 |

The native vLLM wrapper constructs this text config and delegates to the DeepSeek
decoder. It uses ordinary decoder positions, not `SupportsMRoPE`. The supported
models table explicitly lists Small. This avoids the GQA and recurrence
prerequisites discussed above; multimodal PCP itself remains blocked on main.

The [checkpoint index](https://huggingface.co/deepseek-ai/deepseek-vl2-small/blob/main/model.safetensors.index.json)
reports 32,296,699,008 bytes (30.1 GiB) for all stored tensors, including the
multimodal components. This supports a comfortable fit estimate on 80GB H100s
even at TP1, subject to actual runtime profiling. Do not infer per-GPU allocation
by dividing the checkpoint size by the total GPU count: attention and MoE use
different parallel groups. See the measured placement below.

Start baseline serving at TP1/PCP1. After implementing support, compare two-GPU
TP2/PCP1 with TP1/PCP2. On all eight GPUs, compare TP8/PCP1, TP4/PCP2,
TP2/PCP4, and TP1/PCP8, with DCP1 and PP1 throughout. Check backend selection,
MoE collectives and encoder behavior for each layout rather than assuming fit
implies working execution. Begin with one image, then multiple images and mixed
requests, with prompt plus output fitting the 4096-token limit.

`deepseek-ai/deepseek-vl2` is a useful second model: Table 1 confirms MLA rank
512, and its checkpoint index reports 54,960,268,496 bytes (51.2 GiB). Prefer
Small for faster iterations and more activation/cache headroom. Both have short
native context, so they are strong correctness vehicles but limited evidence for
very-long-context PCP performance. Do not silently extrapolate their context.

These are config/source and weight-size checks. No weights were downloaded and
neither model was launched on GPUs during this investigation.

## Implementation on `feat/pcp-deepseek-vl2`

The branch starts at upstream main
`e0adacc9a67a5f2b2d8ba508f7f522bc1ff83c6d`. The implementation enables
the native `DeepseekVLV2ForCausalLM` image path by narrowing the existing
multimodal rejection to allow VL2. Initial validation used eager execution,
DCP1, PP1, prefix caching off, and no speculative decoding or encoder connector.
Encoder outputs remain replicated; existing per-segment embedding assembly is
reused. No recurrent-PCP or GQA-PCP prerequisite was added.

Full-model validation also reproduced an existing config regression: the full
VL2 checkpoint omits its decoder layer count, but Transformers defaults to 32
instead of the original VL2 config's 30. Loading fails on missing layer 30/31
weights. The adapter now preserves 30 as the default while honoring Tiny/Small's
explicit layer counts. The original definition is in
[DeepSeek's config](https://github.com/deepseek-ai/DeepSeek-VL2/blob/main/deepseek_vl2/models/configuration_deepseek.py).

Validation uses real `deepseek-ai/deepseek-vl2` weights at revision
`f363772d1c47f4239dd844015b4bd53beb87951b`, BF16, H100 80GB, the V2 runner,
and the environment's existing precompiled vLLM binaries/NVIDIA Torch runtime.

Existing `FusedMoEParallelConfig` flattens TP across PCP for expert weights even
without expert parallelism. Thus this MoE model does not replicate all weights
per PCP rank. The runner reported 28.29 GiB for model loading at TP1/PCP2 and
8.71 GiB at TP2/PCP4; KV caches and runtime workspace are additional allocations.
The earlier checkpoint-only fit estimates were conservative, not an allocation
formula.

- The affected worker/config suites produced **146 passed, 1 failed**. The failure
  is the existing `test_get_llama3_eos_token`, which requires authentication for
  gated `meta-llama/Llama-3.2-1B-Instruct`; all added tests passed. The simplified
  embedding-boundary regression was subsequently rerun: **4 passed**.
- `tests/distributed/test_context_parallel.py::test_deepseek_vl2_multimodal_pcp`
  passed on two H100s. It compares PCP1/PCP2 on full weights with a 512-token
  scheduling budget, single-image/multi-image/text requests and decode. The
  existing logprob comparator tolerates a near-tie between “depicts” and “shows”
  in a multi-image description; generated text is not bitwise identical.
- Full-weight PCP2 also generated correct stop-sign, flower-color, multi-image,
  and text-only responses with native-resolution test assets. The multi-image
  prompt occupied 3445 tokens. GPU runs used `VLLM_GPU_SYNC_CHECK=error`.
- Repository pre-commit checks passed, including formatting, lint, mypy and test
  tethering. The distributed suite already runs in the H100 Distributed Features
  Buildkite job; the full-weight regression skips GPUs with less than 70GB.
- After adding the TP1 restriction, the final affected-suite run passed:
  **147 passed, 1 deselected**. Only the previously identified gated Llama test
  was deselected. Final repository checks also passed.
- **TP1/PCP8 passed on all eight H100s** with full VL2 weights, a 512-token
  scheduling budget, DCP1/PP1, eager mode and GPU sync checks enabled. Prompts
  occupied 2033, 1428, 3445 and 14 tokens. Outputs correctly read “STOP,” described
  pink flowers, described two images, and answered the text-only arithmetic
  question. All GPU workers were released afterward. Result:
  `/tmp/vllm-pcp-vl2-pcp8.json`; runner: `/tmp/vllm-pcp-vl2-validate.py`.

### Bounded image-conditioned evaluation

First 32 examples of `HuggingFaceM4/ChartQA` test, greedy, maximum 32 output
tokens, scheduling budget 2048. The score below is exact answer matching after
trimming whitespace, lowercasing, and removing trailing periods; it is **not**
the official ChartQA relaxed metric or a full benchmark result.

| Configuration | GPUs | Normalized matches | Four-request, one-output-token latency, median of 3 |
| --- | --- | --- | --- |
| TP2 / PCP1 | 2 | 20/32 | 224 ms |
| TP1 / PCP2 | 2 | 20/32 | 352 ms |
| TP2 / PCP4 | 8 | 20/32 | 392 ms |

Thirty generated answers match exactly. Example 6 differs only in punctuation
(“62.” vs “62”); example 28 differs substantively (“23.” vs “60.”), and both
are incorrect under this metric. This compares the two-GPU configurations.
TP2/PCP4 matches 29/32 baseline answers exactly: example 6 differs in punctuation,
example 28 produces “60.”, and example 29 produces “55.” instead of “56.”; both
numeric alternatives remain incorrect against their labels. Equal aggregate scores do not prove universal
output parity. The latency probe includes local preprocessing/encoder/decoder
work after warmup, uses four fixed requests and three repetitions, and is not
an API TTFT or long-context scaling benchmark. It does not demonstrate a speedup.
The PCP-disabled baseline includes the layer-default correction; unmodified main
cannot load the full checkpoint in this environment.

### TP + PCP investigation and scope

An eight-GPU TP2/PCP4 run exposed a CUDA invalid-argument error when fewer decode
requests remain than PCP ranks. It reproduces with a one-layer, text-only dummy
model at TP2/PCP2, so it is not specific to image placement. Synchronous launch
reporting points at MLA output padding cleanup on an idle rank. CUDA pointer,
device and stream probes did not establish a root cause; isolated zero-fill,
cache-update and scheduler probes did not reproduce the full failure.

Follow-up isolation found that NCCL-only TP communication completed while
switching attention to Triton MLA still failed. Disabling custom all-reduce alone
also completed. On idle PCP ranks, multimodal embedding preparation passes zero
input IDs to the TP vocabulary embedding; its custom all-reduce accepts empty
tensors and can launch a zero-block CUDA grid. The reported MLA zero-fill error
was downstream of that empty lookup.

`EncoderRunner.get_inputs_embeds` now returns its existing buffer directly when
there are no local tokens. Encoder execution still happens before embedding
preparation, so scheduled image work and its TP collectives remain intact. The
one-layer TP2/PCP2 reproducer completes with custom all-reduce enabled after this
change. The provisional TP1 restriction is removed; no attention/kernel workaround
or diagnostic logging is retained. The distributed regression now covers TP2 in
addition to TP1, and a CPU regression covers empty embedding preparation.

Full-weight TP2/PCP4 now completes on all eight H100s with the original custom
all-reduce enabled. The 512-token-budget mixed batch reads “STOP”, identifies
pink flowers, describes both images, and answers two plus two correctly. The
3445-token two-image prompt crosses multiple PCP and prefill-chunk boundaries.
The same 32-example ChartQA smoke evaluation retains 20/32 normalized matches;
this is not a performance win over the two-GPU baseline. Updated CPU/config
checks pass with **148 passed, 1 deselected** (the existing gated Llama test).
All affected-file pre-commit checks pass.
The four-GPU TP2 regression also passes (**1 passed in 141.67s**), comparing
PCP1/PCP2 logprobs for the mixed batch and a single-image request that guarantees
idle PCP ranks during decode. Log: `/tmp/vllm-pcp-vl2-tp2-regression.log`.

The optional FlashInfer all-reduce path stalled during initial TP2 profiling in
this environment. The equal-GPU comparison uses
`VLLM_ALLREDUCE_USE_FLASHINFER=0` in both configurations. This workaround is not
part of the production change. The optional DeepGEMM binary is also incompatible
with the installed Torch build; runs select the existing Triton MoE fallback.

Reproduction commands and artifacts are available locally:

```bash
uv run --no-sync .venv/bin/python -m pytest \
    tests/v1/worker/test_gpu_pcp_manager.py \
    tests/v1/worker/test_encoder_runner.py \
    tests/transformers_utils/test_config.py -q

CUDA_VISIBLE_DEVICES=4,5 HF_HUB_CACHE=/tmp/vllm-pcp-hf/hub HF_HUB_OFFLINE=1 \
    VLLM_GPU_SYNC_CHECK=error uv run --no-sync .venv/bin/python -m pytest \
    tests/distributed/test_context_parallel.py -k deepseek_vl2_multimodal_pcp -v -s

CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_CACHE=/tmp/vllm-pcp-hf/hub HF_HUB_OFFLINE=1 \
    VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_GPU_SYNC_CHECK=error \
    uv run --no-sync .venv/bin/python -m pytest \
    'tests/distributed/test_context_parallel.py::test_deepseek_vl2_multimodal_pcp[2]' \
    -v -s
```

The temporary Hugging Face cache points at `/tmp/vllm-pcp-vl2-model`; the exact
evaluation inputs are `/tmp/vllm-pcp-chartqa/samples.json`. The runner is
`/tmp/vllm-pcp-vl2-eval.py` and the comparison report is
`/tmp/vllm-pcp-vl2-eval-summary.json`. These are development artifacts, not new
repository test/benchmark infrastructure. No commit, push, or PR was made.

TP2/PCP4 generation artifacts are `/tmp/vllm-pcp-vl2-tp2-pcp4-fixed.json` and
`/tmp/vllm-pcp-vl2-tp2-pcp4-fixed.log`. Its evaluation report and comparison are
`/tmp/vllm-pcp-vl2-eval-tp2-pcp4.json` and
`/tmp/vllm-pcp-vl2-eval-tp2-pcp4-summary.json`.

### Simplification after independent review

Three independent reviews covered runtime correctness, tests, and upstream
patterns. The final runtime patch retains only the VL2 exception to the existing
multimodal guard, the empty embedding lookup fast path, and the 30-layer default.
The additional eager/PP/cache/speculation/encoder-connector restrictions were
precautions without demonstrated failures and are removed. Existing generic PCP
validation, including dense MLA+DCP and LoRA checks, remains unchanged. Other
multimodal architectures remain rejected because their position handling differs.

The CPU segment regression now calls the existing gather API directly with
repeated request segments and absolute offsets. It no longer constructs a PCP
manager, patches host-to-device copies, or reads private gather indexes. Redundant
validation cases were removed. Both distributed TP cases remain: TP1 runs on the
existing two-GPU CI jobs, while TP2 covers idle-rank decode on four GPUs.
The simplified worker/config suites pass: **137 passed, 1 deselected**.
Final affected-file pre-commit checks pass. Full VL2 TP2/PCP4 also passes with
prefix caching enabled: repeating the mixed batch preserves exact output token
IDs and reports cached prompt counts `[2032, 1424, 3440, 0]`. The text-only
14-token prompt is shorter than a cache block. Artifacts:
`/tmp/vllm-pcp-vl2-prefix-review.py`, `/tmp/vllm-pcp-vl2-prefix-review.log`,
and `/tmp/vllm-pcp-vl2-prefix-review.json`.

### Additional mode validation

Full VL2 BF16 mode checks use the same four single-image, two-image, and text
prompts, plus repeated single-image requests. No runtime changes were needed.

| Mode | Topology | Result |
| --- | --- | --- |
| Decode CUDA graphs | TP2/PCP4/PP1 | `FULL_DECODE_ONLY`, compilation mode 0, configured capture sizes 1/2/4/8; 31 actual graph replays on each of eight GPUs |
| Pipeline parallelism | TP2/PCP2/PP2 | Mixed batch and single-request decode complete |
| Encoder transfer | TP2/PCP4/PP1 | `ECExampleConnector` exports two real encoder tensors, then a fresh consumer loads them with runtime image encoding forbidden |

All four generated token sequences match the eager TP2/PCP4 baseline exactly in
each successful mode, including producer and consumer. Exported encoder tensors
have shapes `[2017, 2560]` and `[1415, 2560]`. Graph coverage is decode-only, not
compiled prefill or encoder graphs. Transfer coverage is the filesystem example
connector, not RDMA. Modes were checked independently, not as a full cross-product.

Speculation was checked through real configuration and draft-constructor paths:

- N-gram configuration is accepted with PCP1 and rejected by existing generic
  PCP validation with PCP4 (`MRV2 PCP only supports DSpark or single-module MTP`).
- MTP is rejected for VL2 during speculative config resolution, with or without
  PCP (`Unsupported speculative method: 'mtp'`).
- Default DSpark configuration resolves to the DeepSeek-V4 draft implementation.
  Its actual constructor fails on VL2's config (`DeepseekVLV2Config` has no
  `hidden_size`), both with and without PCP. No speculative tokens were generated;
  this stock setup lacks a usable compatible draft integration.

Harnesses: `/tmp/vllm-pcp-mode-check.py` and
`/tmp/vllm_pcp_mode_probe.py`. Reports/logs use prefixes
`/tmp/vllm-pcp-graph`, `/tmp/vllm-pcp-pp`, `/tmp/vllm-pcp-ec-producer`, and
`/tmp/vllm-pcp-ec-consumer`. Speculation evidence is in
`/tmp/vllm-pcp-spec-check.json`, `/tmp/vllm-pcp-spec-baseline.json`,
`/tmp/vllm-pcp-dspark-init-check.json`, and
`/tmp/vllm-pcp-dspark-init-baseline.json`. All GPU runs retain the previously
documented FlashInfer all-reduce workaround and GPU synchronization checks.

### Narrowed model scope and duplicate-work boundary (2026-10-19)

The requested targets are full DeepSeek-VL2 and
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`. Other model integrations are outside
this effort. VL2 remains the only enabled multimodal architecture in this patch.

Vision Exp is multimodal, but its decoder uses the same V4 compression and sparse
attention paths as the text checkpoint. Auditing upstream main at
`36e0ae2df2882de8027bb484da9179c10646ccee` found no PCP group handling in
`vllm/models/deepseek_v4/compressor.py`, `attention.py`, or
`nvidia/flashmla.py`. The generic MLA path exercised by VL2 does not supply that
missing V4 integration.

Open [PR #43809](https://github.com/vllm-project/vllm/pull/43809) implements the
overlapping prerequisites: PCP-aware hybrid KV mapping, C4/C128 compressor state
merging, sparse attention result merging, and MegaMoE token sharding. Its current
benchmarks use V4-Pro, but the implementation touches the shared V4 core. A
different checkpoint name therefore does not make Vision Exp PCP independent.
Searches found no separate open PR specifically titled for Vision Exp PCP;
that does not remove the shared-core overlap.

Keep Vision Exp as a follow-up validation target after compatible core PCP
support is available. Any subsequent patch must isolate a demonstrated remaining
multimodal gap (image boundaries, image visibility, embedding segmentation, or
encoder transfer) and reuse that core implementation. Do not copy or independently
reimplement #43809 here. No Vision Exp runtime support or GPU validation is claimed.
