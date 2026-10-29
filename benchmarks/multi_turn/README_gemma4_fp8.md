# Gemma4 FP8 internal reproduction

These clients reproduce the measured workload and canonical quality protocol.
They do not launch servers or decide speedup-goal acceptance. Use a fresh output
folder for each source/model/restart. Run each command from the repository root
with its configured `.venv`; dependencies are `aiohttp`, `numpy` and `regex`.

The measured target is RedHatAI/gemma-4-31B-it-FP8-block versus
Google/gemma-4-31b-it, eight H100s, TP8, BF16 activations and KV cache, FA4 from
the vllm-project fork (revision `b273a02621d3c94bc747df67204f591b67f38fae`),
no speculation/MTP. Historical runs used Torch2.14.0+cu132 and FlashInfer0.7.2.post1.
Record actual source/binary/provider/device versions for a new run.

## Inputs

Supply `--token-ids` as an **external required input**: the same120000-ID
JSON corpus in every arm, SHA256
`14d3dcdede5ae8ad126738c170ea18e02ea753037b143c1e31add13ccff16cc4`.
The package does not contain it and is not a self-contained public recreation
of the historical workload. Arbitrary new text or a different tokenizer is a
new workload and must have a different recorded identity.

Local provenance identifies the input as a **synthetic code/document control**,
not production traffic or a quality benchmark: four distinct local repository
excerpt prompts, deterministic token truncation and a question repeated at the
end. The exact original prompt generator was not located. Synthetic/local
provenance alone does not establish that every excerpt is public or suitable
for redistribution; neither prompt text nor token IDs are exported.

The downstream local recipe was byte-verified: join the exact cached JSONL
records' `prompt` strings with newlines, tokenize with
`add_special_tokens=False`, double the list until at least120000IDs, truncate
to120000, and serialize `json.dumps(ids, indent=2) + "\n"`. With the exact
cached four-record JSONL (SHA256
`72792399a274e536c451582f862f3013a4bb2c8689b838e2863a8261a861b35e`)
and tokenizer JSON (SHA256
`cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`),
a local CPU tokenizer reproduced the required corpus bytes exactly. This is
conditional reproduction from existing local inputs, not a public generator.
See `gemma4_fp8_evidence/corpus_provenance.json` for hashes and scope.

Quality uses local GSM8K `train.jsonl` (7473rows) and `test.jsonl` (1319rows),
with hashes in `gemma4_fp8_evidence/protocol-equivalence.json`. Do not commit datasets,
weights, raw model answers or request traces. Obtain standard data separately
using the repository's GSM8K setup, then pass its directory explicitly.

## Serving configuration

Build and activate the internal branch using the normal repository workflow.
Set `MODEL_PATH` to a locally available model directory and set `SERVED_MODEL`
explicitly. The historical alias `/models/google/gemma-4-31b-it` is just an API
identifier; it need not match a local directory. Use that alias for identical
quality payload hashes. Use a separate fresh server for each arm/restart.

```bash
export MODEL_PATH=/path/to/RedHatAI/gemma-4-31B-it-FP8-block
export SERVED_MODEL=/models/google/gemma-4-31b-it
export VLLM_FLASH_ATTN_VERSION=4
export VLLM_SERVER_DEV_MODE=1
export VLLM_USE_DEEP_GEMM_E8M0=1
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER_FP8_INPUT=1
export VLLM_V1_OUTPUT_PROC_CHUNK_SIZE=128
export VLLM_BATCH_INVARIANT=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION=0
export OMP_NUM_THREADS=1

.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" --served-model-name "$SERVED_MODEL" \
  --host 127.0.0.1 --port 30000 \
  --max-model-len 65536 --gpu-memory-utilization 0.85 \
  --dtype bfloat16 --kv-cache-dtype bfloat16 --tensor-parallel-size 8 \
  --max-num-batched-tokens 32768 --max-num-seqs 64 \
  --long-prefill-token-threshold 0 \
  --no-long-prefill-token-threshold-adaptive \
  --language-model-only --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
  --attention-backend FLASH_ATTN --attention-config.flash_attn_version 4 \
  --hf-overrides '{"text_config":{"use_bidirectional_attention":null}}' \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --enable-prompt-tokens-details --enable-per-request-metrics \
  --compilation-config "$(cat benchmarks/multi_turn/gemma4-full32768.json)"
```

No private worker extension or timing hook is needed by these clients. The
server must permit prefix-cache reset and return prompt-token usage details and
streamed token IDs; `VLLM_SERVER_DEV_MODE=1` was set for the local study's reset
endpoint. Keep any such server on a trusted local interface. These commands
omit machine-specific cache/NCCL/provider paths; record any local overrides.
Use fresh, distinct compilation/cache locations per restart when collecting
final process provenance. The64-sequence cap is a capacity tradeoff: the
original native profile's effective default was1024 in this pinned study.
Main-to-optimized comparisons combine code and configuration changes; B1
provides the ordinary-BF16 control under the same64/FULL32768 policy.

For an **original-main native baseline**, use that source checkout and unset
`VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER_FP8_INPUT`. Keep the common
model/TP8/FA4/BF16KV options, omit the
max-seqs/long-prefill overrides, and use `gemma4-native512.json`. Those changes
are part of the historical native baseline profile. For an equally configured
optimized BF16 reference, keep the current64/FULL32768 profile and change the
model; actual eligibility must prove ordinary BF16 fallback and zero private
FP8 calls. A plain launch command alone does not prove either path.

## Restart order

The frozen final campaign uses the following order. F0/B0 are original-main
FP8/BF16 with the native profile. F1 is optimized FP8; B1 uses optimized source
and the same FULL32768/maxseq64 configuration with ordinary BF16 fallback.

| Restart block | Server role order | Concurrency order | Case order at each concurrency |
| --- | --- | --- | --- |
| 1 | F0, B0, F1, B1 | 4,13,57 | warm,mixed |
| 2 | B1, F1, B0, F0 | 57,13,4 | mixed,warm |
| 3 | B0, F0, B1, F1 | 13,4,57 | warm,mixed |

Start a fresh server with distinct caches for every role in every block. Run
each block's concurrency and case loops in the listed order; the example below
shows the first block's workload order. Compare each restart separately and
retain failures. The table describes the frozen collection protocol; see
`gemma4_fp8_evidence/VALIDATION.md` for completion status and results.

## Sustained serving workload

Use an8192-token cached prefix. `warm` always has80 new tokens. `mixed` uses
80/800 new tokens on prefix hits and8192-token cold requests under the fixed
sampler. Output lengths are deterministically80–140 tokens. Three warmup turns
per client are excluded; background traffic maintains load until all measured
quotas finish. The client fails on cache or prompt-count mismatches and records
payload-plan hashes. It does not retry measured requests.

```bash
export TOKEN_IDS=/path/to/corpus.json
export RESULT_DIR=/path/to/new-result-directory
mkdir -p "$RESULT_DIR"
for spec in '4 107' '13 40' '57 24'; do
  read -r clients turns <<< "$spec"
  for case_name in warm mixed; do
    .venv/bin/python benchmarks/multi_turn/benchmark_prefix_cache_staggered.py \
      --base-url http://127.0.0.1:30000 --model "$SERVED_MODEL" \
      --token-ids "$TOKEN_IDS" --output "$RESULT_DIR/c${clients}-${case_name}.json" \
      --concurrency "$clients" --turns "$turns" --warmup-turns 3 \
      --prefix-tokens 8192 --suffix-offset 90000 --stagger-seconds 1.2 \
      --min-output 80 --max-output 140 --seed 20261027 \
      --repeats 1 --cases "$case_name"
  done
done
```

This recipe uses a fresh client process per case. Historical collection also
ran warm+mixed in one client session per concurrency; sampler/payload/cache
semantics are identical, but session lifecycle and timing can differ. No
plain-client multi-case failure is asserted here. The source is byte-identical
to measured client9bf; the documented recipe is not a new performance result.

| Clients / total turns | Measured per case | Mixed80 /800 /cold8192 | Mixed warm TTFT samples |
| --- | ---: | ---: | ---: |
| 4 /107 | 416 | 362 /40 /14 | 402 |
| 13 /40 | 481 | 415 /49 /17 | 464 |
| 57 /24 | 1197 | 1043 /117 /37 | 1160 |

For the **final isolated protocol**, request10 warmups and400 measured
requests for each uncached suffix800/8192/18304, cached prefix8192 and one
output token. This produces410records/query (1230total), plus one unrecorded
prefix setup/query. Only400records/query enter quantiles:

```bash
.venv/bin/python benchmarks/multi_turn/benchmark_cached_prefill.py \
  --model "$SERVED_MODEL" --token-ids "$TOKEN_IDS" \
  --warmup-requests 10 --measured-requests 400 \
  --output "$RESULT_DIR/isolated-10warm-400measured.json"
```

The helper retains historical defaults `--warmup-requests 3
--measured-requests 12`; omitting the flags intentionally reproduces the old
screen, **not final400-sample validation**. Default payload hashes/request IDs
and results match v1 in a CPU fake-transport check; the10/400check verifies
400measured+10warmup per query. No new model/cache-server/timing execution is
claimed. Prefix reset, prompt/marker construction and the measured client's
request implementation are unchanged.

## Canonical quality evaluation

```bash
.venv/bin/python benchmarks/multi_turn/evaluate_gemma4_quality.py \
  --host http://127.0.0.1 --port 30000 --model "$SERVED_MODEL" \
  --data-dir /path/to/gsm8k-jsonl --repo . \
  --count 1319 --output "$RESULT_DIR/gsm8k.json" --label arm-restart
```

Protocol:1319 questions, first five train examples, no generated prefix,
`temperature=0`, `seed=42`, `max_tokens=1024`, `top_p=1`, `top_k=-1`, concurrency4,
thinking disabled, prefix reset before traffic. The wrapper extracts the
repository's existing prompt builder and established integer scorer without
importing vLLM/CUDA; it also records strict final-`####` Decimal scoring,
truncations, errors and per-item hashes. Different scored answers require human
review; aggregate nondecrease is not statistical equivalence or a quality gain.

## Evidence and remaining validation

See `gemma4_fp8_evidence/VALIDATION.md`,
`gemma4_fp8_evidence/historical_metrics.json` and
`gemma4_fp8_evidence/protocol-equivalence.json`. The CPU proof checks exact measured client
bytes, six serving plans and1319 quality prompt/payload/scorer identities.
The portable quality wrapper changes paths, the full-count default and CLI help.
It also uses the project's regex package with an explicit whitespace class to
preserve the measured strict scorer's acceptance of C0 separators. Inference
functions retain identical ASTs; the scorer has a separate compatibility proof
recorded in `protocol-equivalence.json`.

Historical prototype performance and completed V7 quality are encouraging.
Normal integrated native execution, model quality and three-restart collection
are complete. The full speed goal remains unmet:160/162 speed points pass,
with two retained C57 mixed warm misses against the same-policy BF16 control.
See `gemma4_fp8_evidence/VALIDATION.md` for completed results and retained failures.
Keep C57 cold-tail data
visible and separate; the tail also occurred on main and is nonblocking by the
user's direction. Do not infer causality from these historical comparisons.
