# Gemma4 FP8 MTP validation

The MTP extension is compared with serving branch commit
`d30ed80158a493fdb193ae5cc1b368c4189520e4` on eight H100 80GB HBM3 GPUs,
TP8/PP1. Both arms use the same FP8 target, FP8 assistant, native extension,
FA4, BF16 activations/KV cache, environment and graph policy. The extension
uses the existing `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER_FP8_INPUT=1` opt-in.

The target now permits its existing full-chain producer under Gemma4 MTP.
The assistant adds fused RMSNorm/FP8-quantization/GEMV for Q and gate/up,
fused quantization/GEMV for O and down, and fused concatenation/quantization/
pre-projection. The small pre-projection weight is replicated once across
TP ranks, using about 11 MiB extra per rank and removing its activation gather.
GEMVs specialize rows 1–8; the pre-projection specializes rows 1–20. Larger
shapes retain the existing native GEMM and sharded pre-projection path.
Row dispatch occurs inside opaque custom operations so that tracing a large
warmup does not fix the path used by small decode graphs.

## Measurement boundary

`benchmark_gemma4_mtp.py` measures complete `LLM.generate` wall time, including
prefill, draft calls, verification, sampling and CPU work. Each arm/depth uses
one process restart, two warmups and five measured samples per shape. Prefix
caching and detokenization are disabled; output length is 64 with EOS ignored.
Inputs are 128, 2,048 and 20,000 tokens, at batch sizes 1 and 4. Both arms use
`FULL_AND_PIECEWISE` capture sizes 1–32, maximum four sequences, a 4,096-token
batch limit and 0.6 GPU memory utilization. These are generation comparisons,
not online serving TPOT or model-forward measurements.

Profiles are collected after all timed and quality cases. An earlier k=1 run
with profiling between timed cases had a noisy C4 regression and is excluded;
the unprofiled rerun improves all six shapes. This observation does not establish
the cause of the earlier noise. Raw excluded and retained results remain under
`/jobtmp/gemma4-mtp-validation/final/` on the measurement machine.

## Matched generation results

Median latency reduction versus the same speculative depth on the serving
branch; positive values mean lower latency. All 24 original comparisons are
retained, including the k=3 short-input regression.

| Input tokens | Batch | k=1 | k=2 | k=3 | k=4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 5.46% | 4.68% | -2.09% | 3.17% |
| 128 | 4 | 4.59% | 3.26% | 2.38% | 3.29% |
| 2,048 | 1 | 5.88% | 7.08% | 7.90% | 11.85% |
| 2,048 | 4 | 4.76% | 4.10% | 3.18% | 2.73% |
| 20,000 | 1 | 1.91% | 1.88% | 2.19% | 1.99% |
| 20,000 | 4 | 0.49% | 0.56% | 1.16% | 0.56% |

Every baseline and candidate arm scores 31/32 on the quality screen, with
identical per-question correctness and no truncations. This does not imply
identical generated token sequences.

A separate k=3 repeat uses the same source, checkpoints and model configuration,
with three warmups and nine unprofiled samples per shape. All six shapes improve
by 1.02–13.25%. The original 128-token C1 regression does not
reproduce: baseline 205.79 ms versus candidate 198.27 ms
(3.66% lower latency). The original samples remain in the main table;
the repeat is recorded separately and does not establish the cause of variation.

## Kernel evidence

All 117 real-weight shard/shape cases pass the native-reference relative-L1
threshold of 1e-3; the maximum observed error is 0.000216. Of these, 81 use a
specialized kernel and 36 exercise the existing fallback. Specialized local
operations measure 1.05–2.68x faster using CUPTI, CUDA graphs, cold L2 and two
alternating-order rounds of 50 samples. The local boundary includes RMSNorm or
concatenation and activation quantization, but excludes TP collectives.
The replicated pre-projection measures 1.50–1.75x locally; its gather removal
is checked separately through the real model.

A bounded rank-0 k=4 trace at 128 input tokens and 16 output tokens shows:

| Batch | Native activation gathers | Fused activation gathers | Fused GEMVs | Fused pre-projections |
| --- | ---: | ---: | ---: | ---: |
| 1 | 48 | 28 | 320 | 20 |
| 4 | 57 | 34 | 304 | 23 |

Counts establish actual kernel selection. Collective durations include rank
skew and are not used to estimate latency savings. Representative generated
code has zero spills: the normalized GEMV uses 31 registers and 1 KiB shared
memory; the split-K pre-projection uses 91 registers and 36 KiB shared memory,
with FP8 tensor-core MMA instructions. These are representative variants, not
an assertion about every specialization. An initial tensor-core normalized-GEMM
prototype measured 17–30 microseconds versus roughly 9 microseconds natively
and was discarded. The retained small-row SIMT kernel was tuned over output
tile size and warp count; larger shapes keep the native implementation.

## Correctness and limits

FP32 reduction order differs from the native tensor-core path, so full-model
token sequences need not be identical. The 32-question five-shot GSM8K screen
uses greedy sampling, thinking disabled and a 1,024-token limit. Per-question
correctness, token hashes, truncations and prompt/scorer/data hashes are retained
in `mtp-metrics.json`. This screen is not a full model-quality evaluation or
proof of accuracy equivalence.

The real-entrypoint k=1 test fails on the original branch because target
producer eligibility is false on all eight ranks. It passes on the extension,
asserting actual target-chain, assistant GEMV and pre-projection launches, plus
exact short greedy token/text parity with the opt-in disabled. All four depths
are exercised by the real-model performance/quality matrix; the pytest test is
parameterized for all four depths in CI. An initial test-harness token-budget
configuration error was corrected before this before/after run.

The existing fused quantization/core suite passes 77 cases. Assistant kernel
coverage adds 36 GEMV cases, nine concatenation/projection cases and six
large-row fallback cases, all passing. Real-weight benchmarks check output
correctness before timing. CI includes all four speculative depths in the
real-LLM regression test and the new kernel cases in the H100 DeepGEMM suite.

Reproduction commands are in `../README_gemma4_fp8.md`. Compact source,
checkpoint config/index/tokenizer, native-library and raw-result hashes, all generation samples,
quality item results and kernel medians are in `mtp-metrics.json`. Raw kernel
samples, generated PTX, traces and test logs remain in
`/jobtmp/gemma4-mtp-validation/` on the measurement machine.

The correctness selectors used are:

```bash
.venv/bin/python -m pytest tests/kernels/core/test_fused_quant_layernorm.py \
  -k gemma4 -v
.venv/bin/python -m pytest tests/kernels/quantization/test_block_fp8.py \
  -k gemma4_mtp -v
GEMMA4_FP8_MODEL=/path/to/gemma4-31B-it-FP8-block \
GEMMA4_MTP_MODEL=/path/to/gemma4-assistant-FP8-block \
.venv/bin/python -m pytest \
  'tests/v1/e2e/spec_decode/mtp/gemma4/test_mtp.py::test_fp8_mtp_producer_outputs[1]' \
  -v -s
```

The real-entrypoint test defaults to the public BF16 assistant for CI; set
`GEMMA4_MTP_MODEL` to a compatible FP8 assistant to additionally assert actual
fused GEMV and concatenation/projection launches. The local depth matrix uses
the FP8 assistant for all four depths. The test compares short greedy math
answers with the feature enabled and disabled; broader numerical differences
are assessed separately by the quality screen above.
