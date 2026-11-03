// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "torch_utils.h"
#include "cub_helpers.h"
#include <torch/csrc/stable/library.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

struct alignas(16) Vec {
  __nv_bfloat16 x[8];
};
template <int D, bool QONLY, bool WriteCache = false, bool Partials = false>
__global__ void prep(const void* packed, const __nv_bfloat16* qw,
                     const __nv_bfloat16* kw, const __nv_bfloat16* cache,
                     const int64_t* positions, __nv_bfloat16* qo,
                     __nv_bfloat16* ko, __nv_bfloat16* vo, int qheads,
                     int kvheads, int rotary, float eps,
                     bool round_before_weight, const int64_t* slots = nullptr,
                     int slot_count = 0, int block_size = 0,
                     int64_t block_stride = 0, int64_t token_stride = 0,
                     int64_t head_stride = 0, int rows = 0) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const bool isq = head < qheads;
  const bool isk = !isq && head < qheads + kvheads;
  int64_t slot = -1;
  if constexpr (WriteCache) {
    if (!isq) {
      if (token >= slot_count) return;
      slot = slots[token];
      if (slot < 0) return;
    }
  }
  const int idx = isq ? head : isk ? head - qheads : head - qheads - kvheads;
  constexpr int T = D / 8;
  using Reduce = cub::BlockReduce<float, 1024>;
  __shared__ typename Reduce::TempStorage temp;
  __shared__ float inv;
  __shared__ __nv_bfloat16 norm[D];
  const int tid = threadIdx.x;
  const int stride = (qheads + (QONLY ? 0 : 2 * kvheads)) * D;
  Vec x;
  if constexpr (Partials) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float value = 0.0f;
#pragma unroll
      for (int part = 0; part < 6; ++part)
        value = __fadd_rn(value,
                          static_cast<const float*>(
                              packed)[(int64_t(part) * rows + token) * stride +
                                      head * D + tid * 8 + j]);
      x.x[j] = __float2bfloat16_rn(value);
    }
  } else {
    x = reinterpret_cast<const Vec*>(static_cast<const __nv_bfloat16*>(packed) +
                                     int64_t(token) * stride + head * D)[tid];
  }
  float sum = 0;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    float v = __bfloat162float(x.x[j]);
    sum = __fadd_rn(sum, __fmul_rn(v, v));
  }
  float total = Reduce(temp).Reduce(sum, cuda::std::plus<float>{}, T);
  if (tid == 0) inv = rsqrtf(total / D + eps);
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 8; j++) {
    int d = tid * 8 + j;
    float v = __fmul_rn(__bfloat162float(x.x[j]), inv);
    if (isq || isk) {
      if (round_before_weight) v = __bfloat162float(__float2bfloat16_rn(v));
      v = __fmul_rn(v, __bfloat162float((isq ? qw : kw)[d]));
    }
    norm[d] = __float2bfloat16_rn(v);
  }
  __syncthreads();
  Vec result;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    int d = tid * 8 + j;
    float value = __bfloat162float(norm[d]);
    if ((isq || isk) && d < rotary) {
      int pair = d % (rotary / 2);
      auto c = cache + positions[token] * rotary;
      float cosine = __bfloat162float(c[pair]);
      float sine = __bfloat162float(c[pair + rotary / 2]);
      float a = __bfloat162float(norm[pair]);
      float b = __bfloat162float(norm[pair + rotary / 2]);
      value = d < rotary / 2 ? fmaf(a, cosine, -__fmul_rn(b, sine))
                             : fmaf(b, cosine, __fmul_rn(a, sine));
    }
    result.x[j] = __float2bfloat16_rn(value);
  }
  __nv_bfloat16* out = isq ? qo : isk ? ko : vo;
  int heads = isq ? qheads : kvheads;
  int64_t offset = int64_t(token) * heads * D + idx * D;
  if constexpr (WriteCache) {
    if (!isq)
      offset = (slot / block_size) * block_stride +
               (slot % block_size) * token_stride + idx * head_stride;
  }
  reinterpret_cast<Vec*>(out + offset)[tid] = result;
}
extern "C" int launch_prep(const void* x, const void* qw, const void* kw,
                           const void* cache, const void* pos, void* qo,
                           void* ko, void* vo, int m, int qh, int kh, int d,
                           int rotary, float eps, void* stream,
                           bool round_before_weight) {
  dim3 grid(m, qh + 2 * kh);
#define RUN(D, Q)                                                  \
  prep<D, Q><<<grid, D / 8, 0, (cudaStream_t)stream>>>(            \
      (const __nv_bfloat16*)x, (const __nv_bfloat16*)qw,           \
      (const __nv_bfloat16*)kw, (const __nv_bfloat16*)cache,       \
      (const int64_t*)pos, (__nv_bfloat16*)qo, (__nv_bfloat16*)ko, \
      (__nv_bfloat16*)vo, qh, kh, rotary, eps, round_before_weight)
  if (d == 256) {
    if (kh == 0) {
      RUN(256, true);
    } else {
      RUN(256, false);
    }
  } else if (d == 512) {
    if (kh == 0) {
      RUN(512, true);
    } else {
      RUN(512, false);
    }
  } else
    return 1;
  return int(cudaGetLastError());
}

void gemma4_attention_prep_out(
    torch::stable::Tensor const& input, torch::stable::Tensor const& q_weight,
    torch::stable::Tensor const& k_weight, torch::stable::Tensor const& cache,
    torch::stable::Tensor const& positions, torch::stable::Tensor& q,
    torch::stable::Tensor& k, torch::stable::Tensor& v, int64_t q_heads,
    int64_t kv_heads, int64_t head_dim, double epsilon,
    bool round_before_weight) {
  using ScalarType = torch::headeronly::ScalarType;
  STD_TORCH_CHECK(input.device().is_cuda(), "Gemma4 preparation requires CUDA");
  auto device = input.get_device_index();
  for (auto const* tensor : {&input, &q_weight, &k_weight, &cache,
                             static_cast<torch::stable::Tensor const*>(&q),
                             static_cast<torch::stable::Tensor const*>(&k),
                             static_cast<torch::stable::Tensor const*>(&v)}) {
    STD_TORCH_CHECK(
        tensor->device().is_cuda() && tensor->get_device_index() == device &&
            tensor->scalar_type() == ScalarType::BFloat16 &&
            tensor->is_contiguous(),
        "Gemma4 preparation requires contiguous BF16 tensors on one GPU");
  }
  STD_TORCH_CHECK(positions.device().is_cuda() &&
                      positions.get_device_index() == device &&
                      positions.scalar_type() == ScalarType::Long &&
                      positions.dim() == 1 && positions.is_contiguous(),
                  "Gemma4 positions must be contiguous CUDA int64");
  STD_TORCH_CHECK(input.dim() == 2 && input.size(0) > 0 &&
                      input.size(0) <= std::numeric_limits<int>::max() &&
                      q_heads > 0 && kv_heads >= 0 &&
                      q_heads + 2 * kv_heads <= 65535 &&
                      (head_dim == 256 || head_dim == 512) &&
                      input.size(1) == (q_heads + 2 * kv_heads) * head_dim,
                  "Invalid Gemma4 packed attention shape");
  auto rows = input.size(0);
  STD_TORCH_CHECK(q_weight.dim() == 1 && q_weight.numel() == head_dim &&
                      k_weight.dim() == 1 && k_weight.numel() == head_dim &&
                      cache.dim() == 2 && cache.size(1) == head_dim &&
                      positions.numel() == rows && epsilon == 1e-6,
                  "Invalid Gemma4 norm weights, rotary cache or positions");
  STD_TORCH_CHECK(q.dim() == 2 && q.size(0) == rows &&
                      q.size(1) == q_heads * head_dim && k.dim() == 2 &&
                      k.size(0) == rows && k.size(1) == kv_heads * head_dim &&
                      v.dim() == 2 && v.size(0) == rows &&
                      v.size(1) == kv_heads * head_dim,
                  "Invalid Gemma4 preparation output shapes");
  for (auto const* tensor :
       {&input, static_cast<torch::stable::Tensor const*>(&q),
        static_cast<torch::stable::Tensor const*>(&k),
        static_cast<torch::stable::Tensor const*>(&v)}) {
    STD_TORCH_CHECK(
        tensor->numel() == 0 ||
            reinterpret_cast<uintptr_t>(tensor->const_data_ptr()) % 16 == 0,
        "Gemma4 attention vectors require 16-byte alignment");
  }
  const torch::stable::accelerator::DeviceGuard guard(device);
  auto const* properties = get_device_prop();
  STD_TORCH_CHECK(properties->major == 9 && properties->minor == 0,
                  "Gemma4 preparation requires SM90");
  int error = launch_prep(input.const_data_ptr(), q_weight.const_data_ptr(),
                          k_weight.const_data_ptr(), cache.const_data_ptr(),
                          positions.const_data_ptr(), q.mutable_data_ptr(),
                          k.mutable_data_ptr(), v.mutable_data_ptr(), rows,
                          q_heads, kv_heads, head_dim, cache.size(1), epsilon,
                          get_current_cuda_stream(device), round_before_weight);
  STD_TORCH_CHECK(error == cudaSuccess, "Gemma4 preparation failed: ",
                  cudaGetErrorString(static_cast<cudaError_t>(error)));
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "gemma4_attention_prep_out(Tensor input, Tensor q_weight, Tensor "
      "k_weight, "
      "Tensor cache, Tensor positions, Tensor(a!) q, Tensor(b!) k, Tensor(c!) "
      "v, "
      "int q_heads, int kv_heads, int head_dim, float epsilon, bool "
      "round_before_weight=False) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("gemma4_attention_prep_out", TORCH_BOX(&gemma4_attention_prep_out));
}

template <bool Quant, bool UE8 = false>
__global__ void gelu_quant(const __nv_bfloat16* x, void* output, float* scales,
                           int k, int pitch, bool round_gate) {
  int row = blockIdx.x;
  int group = blockIdx.y * 4 + threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  if (group >= k / 128) return;
  float values[4];
  float maximum = 0;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    int col = group * 128 + lane * 4 + j;
    float gate = __bfloat162float(x[int64_t(row) * 2 * k + col]);
    float up = __bfloat162float(x[int64_t(row) * 2 * k + k + col]);
    float cube = gate * gate * gate;
    constexpr float beta = M_SQRT2 * M_2_SQRTPI * 0.5f;
    float inner = beta * (gate + 0.044715f * cube);
    float activated = 0.5f * gate * (1.0f + tanhf(inner));
    if (round_gate)
      activated = __bfloat162float(__float2bfloat16_rn(activated));
    values[j] = __bfloat162float(__float2bfloat16_rn(activated * up));
    maximum = fmaxf(maximum, fabsf(values[j]));
  }
  if constexpr (!Quant) {
    auto* out = static_cast<__nv_bfloat16*>(output);
#pragma unroll
    for (int j = 0; j < 4; ++j)
      out[int64_t(row) * k + group * 128 + lane * 4 + j] =
          __float2bfloat16_rn(values[j]);
    return;
  }
  auto* out = static_cast<__nv_fp8_e4m3*>(output);
#pragma unroll
  for (int mask = 16; mask; mask /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, mask));
  float inverse, scale;
  if constexpr (UE8) {
    scale = fmaxf(fmaxf(maximum, 1e-10f) / 448.0f, 1e-10f);
    unsigned bits = __float_as_uint(scale);
    scale = __uint_as_float((((bits >> 23) & 255) + ((bits & 0x7fffff) != 0))
                            << 23);
    inverse = 1.0f / scale;
  } else {
    inverse = maximum != 0 ? __fdividef(448.0f, maximum) : 1.0f;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(scale) : "f"(inverse));
  }
  if (lane == 0) scales[group * pitch + row] = scale;
#pragma unroll
  for (int j = 0; j < 4; j++)
    out[int64_t(row) * k + group * 128 + lane * 4 + j] =
        __nv_fp8_e4m3(values[j] * inverse);
}
extern "C" int launch_gelu(const void* x, void* q, void* s, int m, int k,
                           void* stream, bool round_gate) {
  if (m < 32) {
    gelu_quant<true, false>
        <<<dim3(m, (k / 128 + 3) / 4), 128, 0, (cudaStream_t)stream>>>(
            (const __nv_bfloat16*)x, q, (float*)s, k, (m + 3) / 4 * 4,
            round_gate);
  } else {
    gelu_quant<true, true>
        <<<dim3(m, (k / 128 + 3) / 4), 128, 0, (cudaStream_t)stream>>>(
            (const __nv_bfloat16*)x, q, (float*)s, k, (m + 3) / 4 * 4,
            round_gate);
  }
  return int(cudaGetLastError());
}

void gemma4_gelu_quant_out(torch::stable::Tensor const& input,
                           torch::stable::Tensor& output,
                           torch::stable::Tensor& scales, bool round_gate) {
  using ScalarType = torch::headeronly::ScalarType;
  STD_TORCH_CHECK(input.device().is_cuda(),
                  "Gemma4 GELU quantization requires CUDA");
  auto device = input.get_device_index();
  STD_TORCH_CHECK(
      input.scalar_type() == ScalarType::BFloat16 && input.dim() == 2 &&
          input.is_contiguous() && input.size(0) > 0 &&
          input.size(0) <= std::numeric_limits<int>::max() / input.size(1) &&
          input.size(1) > 0 && input.size(1) % 256 == 0 &&
          input.size(1) <= std::numeric_limits<int>::max(),
      "Gemma4 GELU quantization requires BF16 [M, 2*K], K%128=0");
  int64_t rows = input.size(0), width = input.size(1) / 2;
  STD_TORCH_CHECK(output.device() == input.device() &&
                      output.scalar_type() == ScalarType::Float8_e4m3fn &&
                      output.is_contiguous() && output.dim() == 2 &&
                      output.size(0) == rows && output.size(1) == width &&
                      scales.device() == input.device() &&
                      scales.scalar_type() == ScalarType::Float &&
                      scales.is_contiguous() && scales.dim() == 2 &&
                      scales.size(0) == width / 128 &&
                      scales.size(1) == (rows + 3) / 4 * 4,
                  "Invalid Gemma4 GELU FP8 outputs or scale layout");
  const torch::stable::accelerator::DeviceGuard guard(device);
  auto const* properties = get_device_prop();
  STD_TORCH_CHECK(properties->major == 9 && properties->minor == 0,
                  "Gemma4 GELU quantization requires SM90");
  int error = launch_gelu(input.const_data_ptr(), output.mutable_data_ptr(),
                          scales.mutable_data_ptr(), rows, width,
                          get_current_cuda_stream(device), round_gate);
  STD_TORCH_CHECK(error == cudaSuccess, "Gemma4 GELU quantization failed: ",
                  cudaGetErrorString(static_cast<cudaError_t>(error)));
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "gemma4_gelu_quant_out(Tensor input, Tensor(a!) output, Tensor(b!) "
      "scales, bool round_gate=True) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("gemma4_gelu_quant_out", TORCH_BOX(&gemma4_gelu_quant_out));
}

void gemma4_gelu_mul_out(torch::stable::Tensor const& input,
                         torch::stable::Tensor& output, bool round_gate) {
  using ScalarType = torch::headeronly::ScalarType;
  STD_TORCH_CHECK(input.device().is_cuda() && input.is_contiguous() &&
                      input.scalar_type() == ScalarType::BFloat16 &&
                      input.dim() == 2 && input.size(0) > 0 &&
                      input.size(0) <= std::numeric_limits<int>::max() &&
                      input.size(1) > 0 && input.size(1) % 256 == 0 &&
                      input.size(1) / 256 <= 65535 * 2,
                  "Invalid Gemma4 GELU input shape, dtype or layout");
  STD_TORCH_CHECK(output.device() == input.device() && output.is_contiguous() &&
                      output.scalar_type() == ScalarType::BFloat16 &&
                      output.dim() == 2 && output.size(0) == input.size(0) &&
                      output.size(1) == input.size(1) / 2,
                  "Invalid Gemma4 GELU output");
  auto device = input.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  auto const* properties = get_device_prop();
  STD_TORCH_CHECK(properties->major == 9 && properties->minor == 0,
                  "Gemma4 GELU requires SM90");
  int rows = input.size(0), width = input.size(1) / 2;
  gelu_quant<false><<<dim3(rows, (width / 128 + 3) / 4), 128, 0,
                      get_current_cuda_stream(device)>>>(
      static_cast<const __nv_bfloat16*>(input.const_data_ptr()),
      output.mutable_data_ptr(), nullptr, width, 0, round_gate);
  STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                  "Gemma4 GELU launch failed");
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "gemma4_gelu_mul_out(Tensor input, Tensor(a!) output, bool round_gate) "
      "-> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("gemma4_gelu_mul_out", TORCH_BOX(&gemma4_gelu_mul_out));
}

void gemma4_attention_prep_cache_out(
    torch::stable::Tensor const& input, torch::stable::Tensor const& q_weight,
    torch::stable::Tensor const& k_weight, torch::stable::Tensor const& cache,
    torch::stable::Tensor const& positions, torch::stable::Tensor const& slots,
    torch::stable::Tensor& q, torch::stable::Tensor& key_cache,
    torch::stable::Tensor& value_cache, int64_t q_heads, int64_t kv_heads,
    int64_t head_dim, bool round_before_weight) {
  using ScalarType = torch::headeronly::ScalarType;
  STD_TORCH_CHECK(input.device().is_cuda(),
                  "Gemma4 cache preparation requires CUDA");
  auto device = input.get_device_index();
  bool partials = input.scalar_type() == ScalarType::Float;
  STD_TORCH_CHECK(
      (partials && input.dim() == 3 && input.size(0) == 6) ||
          (!partials && input.scalar_type() == ScalarType::BFloat16 &&
           input.dim() == 2),
      "Expected BF16 QKV or six FP32 QKV partitions");
  int64_t rows = input.size(partials ? 1 : 0);
  for (auto const* t :
       {&q_weight, &k_weight, &cache,
        static_cast<torch::stable::Tensor const*>(&q),
        static_cast<torch::stable::Tensor const*>(&key_cache),
        static_cast<torch::stable::Tensor const*>(&value_cache)}) {
    STD_TORCH_CHECK(
        t->device() == input.device() &&
            t->scalar_type() == ScalarType::BFloat16,
        "Gemma4 cache preparation requires BF16 on one CUDA device");
  }
  STD_TORCH_CHECK(
      input.is_contiguous() && rows > 0 &&
          rows <= std::numeric_limits<int>::max() && q_heads > 0 &&
          kv_heads > 0 && q_heads + 2 * kv_heads <= 65535 &&
          (head_dim == 256 || head_dim == 512) &&
          input.size(input.dim() - 1) == (q_heads + 2 * kv_heads) * head_dim &&
          q.dim() == 2 && q.size(0) == rows &&
          q.size(1) == q_heads * head_dim && q.is_contiguous(),
      "Invalid Gemma4 cache preparation input or Q shape");
  STD_TORCH_CHECK(q_weight.dim() == 1 && k_weight.dim() == 1 &&
                      q_weight.numel() == head_dim &&
                      k_weight.numel() == head_dim &&
                      q_weight.is_contiguous() && k_weight.is_contiguous() &&
                      cache.dim() == 2 && cache.size(1) == head_dim &&
                      cache.is_contiguous(),
                  "Invalid Gemma4 norm weights or RoPE cache");
  for (auto const* t : {&positions, &slots})
    STD_TORCH_CHECK(t->device() == input.device() &&
                        t->scalar_type() == ScalarType::Long && t->dim() == 1 &&
                        t->is_contiguous(),
                    "Positions and slots must be contiguous CUDA int64");
  STD_TORCH_CHECK(positions.numel() == rows && slots.numel() <= rows,
                  "Invalid positions or slot mapping length");
  STD_TORCH_CHECK(
      key_cache.dim() == 4 && value_cache.dim() == 4 &&
          key_cache.size(0) == value_cache.size(0) &&
          key_cache.size(1) == value_cache.size(1) && key_cache.size(1) > 0 &&
          key_cache.size(2) == kv_heads && value_cache.size(2) == kv_heads &&
          key_cache.size(3) == head_dim && value_cache.size(3) == head_dim &&
          key_cache.stride(3) == 1 && value_cache.stride(3) == 1,
      "Expected paged BF16 [blocks, block_size, KV heads, head_dim] cache");
  for (int d = 0; d < 3; ++d)
    STD_TORCH_CHECK(key_cache.stride(d) == value_cache.stride(d) &&
                        key_cache.stride(d) > 0 && key_cache.stride(d) % 8 == 0,
                    "Incompatible paged cache strides");
  for (auto const* t :
       {&input, static_cast<torch::stable::Tensor const*>(&q),
        static_cast<torch::stable::Tensor const*>(&key_cache),
        static_cast<torch::stable::Tensor const*>(&value_cache)})
    STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(t->const_data_ptr()) % 16 == 0,
                    "Gemma4 cache vectors require 16-byte alignment");
  const torch::stable::accelerator::DeviceGuard guard(device);
  auto const* properties = get_device_prop();
  STD_TORCH_CHECK(properties->major == 9 && properties->minor == 0,
                  "Gemma4 cache preparation requires SM90");
  dim3 grid(rows, q_heads + 2 * kv_heads);
#define CACHE_PREP(D, P)                                                      \
  prep<D, false, true, P>                                                     \
      <<<grid, D / 8, 0, get_current_cuda_stream(device)>>>(                  \
          static_cast<const __nv_bfloat16*>(input.const_data_ptr()),          \
          static_cast<const __nv_bfloat16*>(q_weight.const_data_ptr()),       \
          static_cast<const __nv_bfloat16*>(k_weight.const_data_ptr()),       \
          static_cast<const __nv_bfloat16*>(cache.const_data_ptr()),          \
          static_cast<const int64_t*>(positions.const_data_ptr()),            \
          static_cast<__nv_bfloat16*>(q.mutable_data_ptr()),                  \
          static_cast<__nv_bfloat16*>(key_cache.mutable_data_ptr()),          \
          static_cast<__nv_bfloat16*>(value_cache.mutable_data_ptr()),        \
          q_heads, kv_heads, head_dim, 1e-6f, round_before_weight,            \
          static_cast<const int64_t*>(slots.const_data_ptr()), slots.numel(), \
          key_cache.size(1), key_cache.stride(0), key_cache.stride(1),        \
          key_cache.stride(2), rows)
  if (partials) {
    if (head_dim == 256) {
      CACHE_PREP(256, true);
    } else {
      CACHE_PREP(512, true);
    }
  } else {
    if (head_dim == 256) {
      CACHE_PREP(256, false);
    } else {
      CACHE_PREP(512, false);
    }
  }
#undef CACHE_PREP
  STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                  "Gemma4 cache preparation launch failed");
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "gemma4_attention_prep_cache_out(Tensor input, Tensor q_weight, Tensor "
      "k_weight, "
      "Tensor cache, Tensor positions, Tensor slots, Tensor(a!) q, "
      "Tensor(b!) key_cache, Tensor(c!) value_cache, int q_heads, int "
      "kv_heads, "
      "int head_dim, bool round_before_weight=False) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("gemma4_attention_prep_cache_out",
         TORCH_BOX(&gemma4_attention_prep_cache_out));
}

template <int Threads>
__device__ __forceinline__ void combine_weights_warp(const float* lse,
                                                     float* weights, int tid,
                                                     int count, int head,
                                                     int row, int64_t ls0,
                                                     int64_t ls1, int64_t ls2) {
  using Reduce = cub::WarpReduce<float, Threads>;
  __shared__ typename Reduce::TempStorage max_temp, sum_temp;
  if (tid < Threads) {
    constexpr unsigned mask = 0xffffffffu >> (32 - Threads);
    float ls = tid < count ? lse[int64_t(tid) * ls0 + head * ls1 + row * ls2]
                           : -__int_as_float(0x7f800000);
#if __CUDA_ARCH__ >= 800
    int ordered = __float_as_int(ls);
    ordered ^= ordered < 0 ? 0x7fffffff : 0;
    int mx_bits = __reduce_max_sync(mask, ordered);
    mx_bits ^= mx_bits < 0 ? 0x7fffffff : 0;
    float mx = __int_as_float(mx_bits);
#else
    float mx = Reduce(max_temp).Reduce(ls, CubMaxOp{});
    mx = __shfl_sync(mask, mx, 0);
#endif
    if (mx == -__int_as_float(0x7f800000)) mx = 0.0f;
    constexpr float log2e = 1.4426950408889634f;
    float exponent = __fsub_rn(__fmul_rn(ls, log2e), __fmul_rn(mx, log2e));
    float weight;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(weight) : "f"(exponent));
    float total = Reduce(sum_temp).Sum(weight);
    total = __shfl_sync(mask, total, 0);
    float inv = total > 0.0f ? __frcp_rn(total) : 0.0f;
    weights[tid] = __fmul_rn(weight, inv);
  }
}

template <int Splits>
__device__ __forceinline__ float combine_partials(const float* partial,
                                                  const float* weights,
                                                  int rows, int width, int row,
                                                  int col) {
  if constexpr (Splits == 8) {
    float p[8], w[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      p[s] = partial[(int64_t(s) * rows + row) * width + col];
      w[s] = weights[s];
    }
    // Issue independent loads before the dependent FMA chain.
    asm volatile(""
                 :
                 : "f"(p[0]), "f"(p[1]), "f"(p[2]), "f"(p[3]), "f"(p[4]),
                   "f"(p[5]), "f"(p[6]), "f"(p[7]), "f"(w[0]), "f"(w[1]),
                   "f"(w[2]), "f"(w[3]), "f"(w[4]), "f"(w[5]), "f"(w[6]),
                   "f"(w[7])
                 : "memory");
    float value = 0.0f;
#pragma unroll
    for (int s = 0; s < 8; ++s) value = fmaf(w[s], p[s], value);
    return value;
  }
  float value = 0.0f;
#pragma unroll
  for (int s = 0; s < Splits; ++s)
    value = fmaf(weights[s], partial[(int64_t(s) * rows + row) * width + col],
                 value);
  return value;
}

template <int Splits>
__device__ __forceinline__ float combine_single_row(const float* partial,
                                                    const float* lse, int width,
                                                    int col, int head,
                                                    int64_t ls0, int64_t ls1) {
  float p[Splits], ls[Splits], w[Splits];
#pragma unroll
  for (int s = 0; s < Splits; ++s) {
    p[s] = partial[int64_t(s) * width + col];
    ls[s] = lse[int64_t(s) * ls0 + head * ls1];
  }
  // Overlap partial-result loads with the normalizer's input loads.
  if constexpr (Splits == 4) {
    asm volatile(""
                 : "+f"(ls[0]), "+f"(ls[1]), "+f"(ls[2]), "+f"(ls[3])
                 : "f"(p[0]), "f"(p[1]), "f"(p[2]), "f"(p[3])
                 : "memory");
  } else {
    asm volatile(""
                 : "+f"(ls[0]), "+f"(ls[1]), "+f"(ls[2]), "+f"(ls[3]),
                   "+f"(ls[4]), "+f"(ls[5]), "+f"(ls[6]), "+f"(ls[7])
                 : "f"(p[0]), "f"(p[1]), "f"(p[2]), "f"(p[3]), "f"(p[4]),
                   "f"(p[5]), "f"(p[6]), "f"(p[7])
                 : "memory");
  }
  float mx = -__int_as_float(0x7f800000);
#pragma unroll
  for (int s = 0; s < Splits; ++s) mx = fmaxf(mx, ls[s]);
  if (mx == -__int_as_float(0x7f800000)) mx = 0.0f;
  constexpr float log2e = 1.4426950408889634f;
#pragma unroll
  for (int s = 0; s < Splits; ++s) {
    float exponent = __fsub_rn(__fmul_rn(ls[s], log2e), __fmul_rn(mx, log2e));
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(w[s]) : "f"(exponent));
  }
  float total = __fadd_rn(__fadd_rn(w[0], w[1]), __fadd_rn(w[2], w[3]));
  if constexpr (Splits == 8)
    total = __fadd_rn(total,
                      __fadd_rn(__fadd_rn(w[4], w[5]), __fadd_rn(w[6], w[7])));
  float inv = total > 0.0f ? __frcp_rn(total) : 0.0f;
  float value = 0.0f;
#pragma unroll
  for (int s = 0; s < Splits; ++s)
    value = fmaf(__fmul_rn(w[s], inv), p[s], value);
  return value;
}

__global__ __launch_bounds__(128, 1) void combine_quant(
    const float* partial, const float* lse, const int* cu, const int* splits,
    __nv_fp8_e4m3* output, float* scales, int rows, int heads, int dim,
    int batches, int pitch, int64_t ls0, int64_t ls1, int64_t ls2) {
  int row = blockIdx.x, group = blockIdx.y, tid = threadIdx.x;
  int batch = 0;
  while (batch + 1 < batches && row >= cu[batch + 1]) ++batch;
  // Padded requests have no initialized split count or attention partials.
  int count = row < cu[batch + 1] ? splits[batch] : 0;
  int head = group * 128 / dim;
  using Reduce = cub::BlockReduce<float, 128>;
  using MaxReduce = cub::BlockReduce<int, 128>;
  __shared__ typename Reduce::TempStorage temp;
  __shared__ typename MaxReduce::TempStorage max_temp;
  __shared__ float weights[128];
  __shared__ float maximum, inverse;
  int width = heads * dim, col = group * 128 + tid;
  float value = 0.0f;
  if (rows == 1 && count == 4) {
    value = combine_single_row<4>(partial, lse, width, col, head, ls0, ls1);
  } else if (rows == 1 && count == 8) {
    value = combine_single_row<8>(partial, lse, width, col, head, ls0, ls1);
  } else {
    constexpr float log2e = 1.4426950408889634f;
    if (count <= 4) {
      combine_weights_warp<4>(lse, weights, tid, count, head, row, ls0, ls1,
                              ls2);
    } else if (count <= 8) {
      combine_weights_warp<8>(lse, weights, tid, count, head, row, ls0, ls1,
                              ls2);
    } else if (count <= 16) {
      combine_weights_warp<16>(lse, weights, tid, count, head, row, ls0, ls1,
                               ls2);
    } else if (count <= 32) {
      combine_weights_warp<32>(lse, weights, tid, count, head, row, ls0, ls1,
                               ls2);
    } else {
      float ls = tid < count ? lse[int64_t(tid) * ls0 + head * ls1 + row * ls2]
                             : -__int_as_float(0x7f800000);
      float mx = Reduce(temp).Reduce(ls, CubMaxOp{});
      if (tid == 0) maximum = mx == -__int_as_float(0x7f800000) ? 0.0f : mx;
      __syncthreads();
      float exponent =
          __fsub_rn(__fmul_rn(ls, log2e), __fmul_rn(maximum, log2e));
      float weight;
      asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(weight) : "f"(exponent));
      float total = Reduce(temp).Sum(weight);
      if (tid == 0) inverse = total > 0.0f ? 1.0f / total : 0.0f;
      __syncthreads();
      weights[tid] = __fmul_rn(weight, inverse);
    }
    __syncthreads();
    if (count == 4) {
      value = combine_partials<4>(partial, weights, rows, width, row, col);
    } else if (count == 8) {
      value = combine_partials<8>(partial, weights, rows, width, row, col);
    } else if (count == 9) {
      value = combine_partials<9>(partial, weights, rows, width, row, col);
    } else if (count <= 32) {
#define COMBINE_SPLITS(S)                                                 \
  case S:                                                                 \
    value = combine_partials<S>(partial, weights, rows, width, row, col); \
    break
      switch (count) {
        COMBINE_SPLITS(1);
        COMBINE_SPLITS(2);
        COMBINE_SPLITS(3);
        COMBINE_SPLITS(5);
        COMBINE_SPLITS(6);
        COMBINE_SPLITS(7);
        COMBINE_SPLITS(10);
        COMBINE_SPLITS(11);
        COMBINE_SPLITS(12);
        COMBINE_SPLITS(13);
        COMBINE_SPLITS(14);
        COMBINE_SPLITS(15);
        COMBINE_SPLITS(16);
        COMBINE_SPLITS(17);
        COMBINE_SPLITS(18);
        COMBINE_SPLITS(19);
        COMBINE_SPLITS(20);
        COMBINE_SPLITS(21);
        COMBINE_SPLITS(22);
        COMBINE_SPLITS(23);
        COMBINE_SPLITS(24);
        COMBINE_SPLITS(25);
        COMBINE_SPLITS(26);
        COMBINE_SPLITS(27);
        COMBINE_SPLITS(28);
        COMBINE_SPLITS(29);
        COMBINE_SPLITS(30);
        COMBINE_SPLITS(31);
        COMBINE_SPLITS(32);
        default:
          break;
      }
#undef COMBINE_SPLITS
    } else {
#pragma unroll 32
      for (int s = 0; s < count; ++s)
        value = fmaf(weights[s],
                     partial[(int64_t(s) * rows + row) * width + col], value);
    }
  }
  value = __bfloat162float(__float2bfloat16_rn(value));
  int magnitude = __float_as_int(fabsf(value));
  // Preserve fmax's NaN handling; -1 also decodes as NaN for an all-NaN group.
  if (magnitude > 0x7f800000) magnitude = -1;
  int amax_bits = MaxReduce(max_temp).Reduce(magnitude, CubMaxOp{});
  if (tid == 0) {
    float amax = __int_as_float(amax_bits);
    inverse = amax != 0.0f ? __fdividef(448.0f, amax) : 1.0f;
    float scale;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(scale) : "f"(inverse));
    scales[group * pitch + row] = scale;
  }
  __syncthreads();
  output[int64_t(row) * width + col] = __nv_fp8_e4m3(__fmul_rn(value, inverse));
}

void gemma4_combine_quant_out(torch::stable::Tensor const& partial,
                              torch::stable::Tensor const& lse,
                              torch::stable::Tensor const& cu,
                              torch::stable::Tensor const& splits,
                              torch::stable::Tensor& output,
                              torch::stable::Tensor& scales) {
  using ScalarType = torch::headeronly::ScalarType;
  STD_TORCH_CHECK(partial.device().is_cuda(), "Gemma4 combine requires CUDA");
  for (auto const* t :
       {&lse, &cu, &splits, static_cast<torch::stable::Tensor const*>(&output),
        static_cast<torch::stable::Tensor const*>(&scales)})
    STD_TORCH_CHECK(t->device() == partial.device(),
                    "Gemma4 combine tensors must share a device");
  STD_TORCH_CHECK(partial.scalar_type() == ScalarType::Float &&
                      partial.dim() == 4 && partial.is_contiguous() &&
                      partial.size(0) <= 128 && partial.size(0) > 0 &&
                      partial.size(1) > 0 && partial.size(1) < 32 &&
                      partial.size(2) > 0 &&
                      (partial.size(3) == 256 || partial.size(3) == 512),
                  "Invalid Gemma4 attention partials");
  int rows = partial.size(1), heads = partial.size(2), dim = partial.size(3);
  STD_TORCH_CHECK(lse.scalar_type() == ScalarType::Float && lse.dim() == 3 &&
                      lse.size(0) == partial.size(0) && lse.size(1) == heads &&
                      lse.size(2) == rows,
                  "Invalid Gemma4 partial LSE");
  STD_TORCH_CHECK(cu.scalar_type() == ScalarType::Int && cu.dim() == 1 &&
                      cu.is_contiguous() &&
                      splits.scalar_type() == ScalarType::Int &&
                      splits.dim() == 1 && splits.is_contiguous() &&
                      cu.numel() == splits.numel() + 1 && splits.numel() > 0,
                  "Invalid Gemma4 split metadata");
  STD_TORCH_CHECK(
      output.scalar_type() == ScalarType::Float8_e4m3fn && output.dim() == 3 &&
          output.size(0) == rows && output.size(1) == heads &&
          output.size(2) == dim && output.is_contiguous() &&
          scales.scalar_type() == ScalarType::Float && scales.is_contiguous() &&
          scales.dim() == 2 && scales.size(0) == heads * dim / 128 &&
          scales.size(1) >= rows && scales.size(1) % 4 == 0,
      "Invalid Gemma4 FP8 combine output");
  auto device = partial.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  auto const* properties = get_device_prop();
  STD_TORCH_CHECK(properties->major == 9 && properties->minor == 0,
                  "Gemma4 combine requires SM90");
  combine_quant<<<dim3(rows, heads * dim / 128), 128, 0,
                  get_current_cuda_stream(device)>>>(
      static_cast<const float*>(partial.const_data_ptr()),
      static_cast<const float*>(lse.const_data_ptr()),
      static_cast<const int*>(cu.const_data_ptr()),
      static_cast<const int*>(splits.const_data_ptr()),
      static_cast<__nv_fp8_e4m3*>(output.mutable_data_ptr()),
      static_cast<float*>(scales.mutable_data_ptr()), rows, heads, dim,
      splits.numel(), scales.size(1), lse.stride(0), lse.stride(1),
      lse.stride(2));
  STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                  "Gemma4 FP8 combine launch failed");
}
STABLE_TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "gemma4_combine_quant_out(Tensor partial, Tensor lse, Tensor cu, Tensor "
      "splits, "
      "Tensor(a!) output, Tensor(b!) scales) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("gemma4_combine_quant_out", TORCH_BOX(&gemma4_combine_quant_out));
}
