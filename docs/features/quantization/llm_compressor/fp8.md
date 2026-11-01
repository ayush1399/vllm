# FP8 W8A8

vLLM supports FP8 (8-bit floating point) weight and activation quantization using hardware acceleration on GPUs such as Nvidia H100 and AMD MI300x.
Ada Lovelace, Hopper, and Blackwell GPUs are supported for W8A8.
Turing/Ampere GPUs are supported for W8A16 (weight-only FP8) utilizing Marlin kernels.
Quantization of models with FP8 allows for a 2x reduction in model memory requirements and up to a 1.6x improvement in throughput with minimal impact on accuracy.

Please visit the HF collection of [quantized FP8 checkpoints of popular LLMs ready to use with vLLM](https://huggingface.co/collections/neuralmagic/fp8-llms-for-vllm-666742ed2b78b7ac8df13127).

The FP8 types typically supported in hardware have two distinct representations, each useful in different scenarios:

- **E4M3**: Consists of 1 sign bit, 4 exponent bits, and 3 bits of mantissa. It can store values up to +/-448 and `nan`.
- **E5M2**: Consists of 1 sign bit, 5 exponent bits, and 2 bits of mantissa. It can store values up to +/-57344, +/- `inf`, and `nan`. The tradeoff for the increased dynamic range is lower precision of the stored values.

!!! note
    FP8 computation is supported on NVIDIA GPUs with compute capability >= 8.9 (Ada Lovelace, Hopper, Blackwell).
    FP8 models will run on compute capability >= 7.5 (Turing) as weight-only W8A16, utilizing FP8 Marlin.

!!! note "GEMM kernel selection"
    vLLM picks an FP8 GEMM kernel automatically at load time and logs a
    `Selected <kernel> for <module>` line at startup. For block-quantized
    checkpoints on CUDA it tries, in order: a FlashInfer/DeepGEMM hybrid
    (Hopper only), DeepGEMM, CUTLASS, Marlin, Triton, Humming, then a
    PyTorch fallback. GPUs without native FP8 support (e.g. Turing/Ampere)
    land on weight-only (W8A16) Marlin.

    If inference hangs with no error, try `VLLM_USE_DEEP_GEMM=0` or
    `--linear-backend cutlass`.

    `--linear-backend` only affects quantized linear layers; MoE experts
    use the separate `--moe-backend`. An explicit backend that is not
    supported on your hardware raises an error rather than falling back.
    The full list is documented under
    [`KernelConfig`](../../../cli/serve.md#kernelconfig) in the CLI
    reference and shown by `vllm serve --help=KernelConfig`.

## Experimental Gemma4 producer fusion on Hopper

`--additional-config '{"decode_backend":"latency_specialized"}'` enables an opt-in FP8-input
producer/consumer path for the FlashInfer/DeepGEMM block-FP8 hybrid on SM90.
It defaults to disabled. For dense Gemma4 31B with BF16 activations, group128
FP8 weights and TP8, the full-chain path combines surrounding normalization,
residual and scalar operations with activation quantization. Per-layer shape
and quantization guards retain supported fallback paths.

The default `decode_backend` value is `auto`, which uses the stock model.
`latency_specialized` selects the NVIDIA implementation under
`vllm/models/gemma4/nvidia/specialized_decode/`. It requires an SM90 GPU,
TP8/PP1, the Gemma4 31B block-FP8 target, BF16 activations, FlashInfer blockscale
GEMMs and `VLLM_USE_DEEP_GEMM_E8M0=1`. Unsupported deployments and unknown backend
names raise a startup error. This setting replaces the experimental
`VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER_FP8_INPUT` environment variable.

The Gemma4 CUDA source lives alongside the Python backend in
`vllm/models/gemma4/nvidia/specialized_decode/` and builds
into the normal
`_C_stable_libtorch` extension. Build it using the
[incremental compilation workflow](../../../contributing/incremental_build.md).
No separate manually installed shared library is needed. The specialized backend
rejects a build without this implementation at startup. Small batches retain the
hybrid's raw-scale FlashInfer path; larger batches use group128 UE8-rounded scales in the TMA-aligned layout.

The target full-chain path also supports Gemma4 MTP with a separate assistant
checkpoint. On the TP8 Gemma4 31B FP8 assistant path, small-row kernels fuse
RMSNorm, raw FP8 activation quantization and Q/gate-up GEMVs, or quantization
and O/down GEMVs. Batches above eight rows retain the existing GEMM path.
The pre-projection fuses input concatenation, quantization and GEMM for up to
20 rows. Its processed FP8 weights are replicated once after loading (about
11 MiB extra per rank) to avoid gathering projected activations every draft step.
Larger pre-projections retain the original sharded execution.

BF16 assistant projections use their ordinary execution. The normalized target
output and both assistant outputs (draft logits input and backbone feedback)
retain their existing roles. Auxiliary hidden-state extraction retains the
ordinary target loop. Explicit selection rejects fast prefill, LoRA, MoE and
unsupported parallelism at startup. These fusions preserve native raw-scale
FP8 tie rounding; FP32 reduction order and split-K projection accumulation can
still affect numerical results.

For example, add the following to a TP8 Gemma4 31B FP8 server command with
`VLLM_USE_DEEP_GEMM_E8M0=1` and
`--additional-config '{"decode_backend":"latency_specialized"}'`:

```bash
--speculative-config '{"method":"mtp","model":"/path/to/gemma4-assistant","num_speculative_tokens":4}'
```

Use `num_speculative_tokens` values 1, 2, 3 or 4 for the validated depth matrix.
The assistant checkpoint may use BF16 or compatible group128 block-FP8 weights.
Verification batches below 32 rows fuse raw-scale FP8 quantization into the
full-chain normalization producer and into target GELU/gating. Larger target
GELU/gating batches also produce FP8 directly, using UE8-rounded scales.
These producers
preserve the FlashInfer CUTLASS quantizer's approximate reciprocal. Activation
rounding follows the selected execution path: native CUDA GELU retains its
intermediate BF16 rounding, while fused Inductor GELU/gating rounds after the
multiply. Inductor's `emulate_precision_casts` setting retains intermediate casts. The scale-layout boundary remains at 32 rows.

A CUDA C++ kernel also combines per-head Q/K/V
normalization and NeoX RoPE. It supports both the 256-dimensional sliding heads
and the 512-dimensional global heads, including Gemma4's identity-padded
proportional RoPE. Assistant layers use the Q-only variant. On supported target
FlashAttention layers, the same kernel writes K/V directly into the paged BF16
cache. It supports the backend's packed cache layout and skips padded slots.
Unsupported cache formats and distributed-context attention retain the existing
preparation and insertion path.

The inter-layer residual carry uses BF16 storage after its existing BF16
rounding point. Normalization statistics and the post-attention residual still
use FP32. This reduces carry traffic without changing its storage precision
relative to that rounding point.

The full-chain path also connects normalization to gate/up and local QKV GEMMs
using programmatic dependent launch (PDL). The GEMM stages independent weights
before waiting for normalization, then reads the local FP8 activation. This
applies below 32 rows to supported 5376-input projections; larger batches retain
the existing GEMMs. No additional TP communication or residual replication is
required. The optional C++ kernels compile once during weight loading using
the installed CUDA toolkit, Ninja, and FlashInfer's CUTLASS/DeepGEMM headers.
They are cached under `VLLM_CACHE_ROOT`; unavailable builds retain the existing
path. Disable only this optimization with
`--additional-config '{"decode_backend":"latency_specialized","gemma4_pdl":false}'`.

Local QKV projections with up to eight rows use six K partitions. QKV
preparation consumes the FP32 partials directly, so the path does not launch a
separate partition-reduction kernel or materialize the full BF16 QKV tensor.
Larger projections retain the existing geometry.

For supported short-query target FA4 calls, the SM90 dependency patch selects
active splits from each request's sequence length and sliding window. The
captured maximum grid remains fixed; inactive splits skip attention work.
A CUDA C++ combine kernel produces FP8 activations and block scales directly
for the output projection. This removes the intermediate BF16 attention output
and a separate quantization pass. Build the patched `vllm-flash-attn` dependency
alongside the native extension; the CMake build applies the patches.

This experimental option changes normalization/quantization reduction and
rounding boundaries. Kernel contract tests do not establish model quality or
end-to-end serving gains; evaluate the exact checkpoint and workload before
deployment.

`benchmarks/multi_turn/benchmark_gemma4_mtp.py` queues each complete batch with
scheduling paused, then records resume-to-completion wall time,
per-request prefill/decode intervals and speculative acceptance counts.
Performance runs use a fixed acceptance schedule (`--accepted-draft-tokens`,
defaulting to all draft tokens). Run GSM8K separately with `--quality-only`,
which uses actual drafting and standard rejection sampling. The
`--no-gemma4-pdl` flag provides the matched runtime control.
Its JSON output separates the batch interval before the first output,
the interval where prefill and decode can overlap, and the decode tail after
every request has produced a first token. These are engine elapsed times,
including CPU work. Compare both runs with the same graph policy. The
`flash_attn_max_num_splits_for_cuda_graph` setting controls the FA3 AOT path;
it does not force 32 splits in FA4.

## Installation

To produce performant FP8 quantized models with vLLM, you'll need to install the [llm-compressor](https://github.com/vllm-project/llm-compressor/) library:

```bash
(venv-llm-compressor) pip install llmcompressor
```

Additionally, install `vllm` and `lm-evaluation-harness` for evaluation:

```bash
(venv-vllm) pip install vllm "lm-eval[api]>=0.4.12"
```

Please use separate environments for vLLM and llm-compressor as they might not work together.

## Quantization Process

The quantization process involves three main steps:

1. Loading the model
2. Applying quantization
3. Evaluating accuracy in vLLM

### 1. Loading the Model

Load your model and tokenizer using the standard `transformers` AutoModel classes:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2. Applying Quantization

For FP8 quantization, we can recover accuracy with simple RTN quantization. We recommend targeting all `Linear` layers using the `FP8_DYNAMIC` scheme, which uses:

- Static, per-channel quantization on the weights
- Dynamic, per-token quantization on the activations

Since simple RTN does not require data for weight quantization and the activations are quantized dynamically, we do not need any calibration data for this quantization flow.

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Configure the simple PTQ quantization
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)

# Apply the quantization algorithm.
oneshot(model=model, recipe=recipe)

# Save the model: Meta-Llama-3-8B-Instruct-FP8-Dynamic
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

### 3. Evaluating Accuracy

Load and run the model in `vllm`:

```python
from vllm import LLM

llm = LLM("./Meta-Llama-3-8B-Instruct-FP8-Dynamic")
result = llm.generate("Hello my name is")
print(result[0].outputs[0].text)
```

Evaluate accuracy with `lm_eval` (for example on 250 samples of `gsm8k`):

!!! note
    Quantized models can be sensitive to the presence of the `bos` token. `lm_eval` does not add a `bos` token by default, so make sure to include the `add_bos_token=True` argument when running your evaluations.

```bash
MODEL=$PWD/Meta-Llama-3-8B-Instruct-FP8-Dynamic
lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,add_bos_token=True \
  --tasks gsm8k  --num_fewshot 5 --batch_size auto --limit 250
```

Here's an example of the resulting scores:

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
| --- |------:| -------------- |-----:| --------- | - |----:| - |-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.768|±  |0.0268|
|     |       |strict-match    |     5|exact_match|↑  |0.768|±  |0.0268|
```

## Troubleshooting and Support

If you encounter any issues or have feature requests, please open an issue on the [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor/issues) GitHub repository.

## Online Dynamic Quantization

Dynamic quantization of an original precision BF16/FP16 model to FP8 can be achieved with vLLM without any calibration data required. You can enable the feature by specifying `--quantization="fp8_per_tensor"` in the command line or setting `quantization="fp8_per_tensor"` in the LLM constructor.

In this mode, all Linear modules (except for the final `lm_head`) have their weights quantized down to FP8_E4M3 precision with a per-tensor scale. Activations have their minimum and maximum values calculated during each forward pass to provide a dynamic per-tensor scale for high accuracy. As a result, latency improvements are limited in this mode.

```python
from vllm import LLM

llm = LLM("facebook/opt-125m", quantization="fp8_per_tensor")
# INFO 06-10 17:55:42 model_runner.py:157] Loading model weights took 0.1550 GB
result = llm.generate("Hello, my name is")
print(result[0].outputs[0].text)
```
