// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "../../ops.h"
#include "../../torch_utils.h"
#include <cstdint>
#include <limits>

// Standalone experimental H5376 postnorm/add/prenorm/FP8 producer.
// Interior Stage B rounds carry/numerator to BF16, retaining FP32 statistics.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

constexpr int H = 5376;
template <int E>
struct alignas(16) BVec {
  __nv_bfloat16 v[E];
};
template <int E>
struct alignas(16) FVec {
  float v[E];
};
template <int E>
struct alignas(8) QVec {
  __nv_fp8_e4m3 v[E];
};
template <int E, bool FloatResidual, bool Scalar, bool Quant, bool Interior,
          bool StoreCarry>
__global__ void chain(const __nv_bfloat16* x, const void* residual,
                      const __nv_bfloat16* post_weight,
                      const __nv_bfloat16* pre_weight, float* residual_out,
                      void* q, float* scales, int pitch, float eps,
                      const __nv_bfloat16* scalar) {
  constexpr int ACTIVE = H / E;
  constexpr int THREADS = (ACTIVE + 31) / 32 * 32;
  constexpr int GROUP_THREADS = 128 / E;
  using Reduce = cub::BlockReduce<float, THREADS>;
  __shared__ typename Reduce::TempStorage temp;
  __shared__ float inverse;
  const int tid = threadIdx.x;
  float values[E];
  BVec<E> xv, pw, nw;
  FVec<E> rw;
  FVec<E> residual_vec;
  const int vector_offset = blockIdx.x * ACTIVE + tid;
  float sum = 0;
  if (tid < ACTIVE) {
    xv = reinterpret_cast<const BVec<E>*>(x)[vector_offset];
    pw = reinterpret_cast<const BVec<E>*>(post_weight)[tid];
    if constexpr (FloatResidual) {
      rw = reinterpret_cast<const FVec<E>*>(residual)[vector_offset];
    } else {
      const BVec<E> br =
          reinterpret_cast<const BVec<E>*>(residual)[vector_offset];
#pragma unroll
      for (int j = 0; j < E; ++j) rw.v[j] = __bfloat162float(br.v[j]);
    }
    nw = reinterpret_cast<const BVec<E>*>(pre_weight)[tid];
#pragma unroll
    for (int j = 0; j < E; ++j) {
      values[j] = __bfloat162float(xv.v[j]);
      sum = __fadd_rn(sum, __fmul_rn(values[j], values[j]));
    }
  }
  float total = Reduce(temp).Sum(sum);
  if (tid == 0) inverse = rsqrtf(total / H + eps);
  __syncthreads();
  const float inv_post = inverse;
  sum = 0;
  if (tid < ACTIVE) {
#pragma unroll
    for (int j = 0; j < E; ++j) {
      values[j] = fmaf(__fmul_rn(values[j], inv_post),
                       __bfloat162float(pw.v[j]), rw.v[j]);
      if constexpr (Scalar)
        values[j] = __fmul_rn(values[j], __bfloat162float(*scalar));
      sum = __fadd_rn(sum, __fmul_rn(values[j], values[j]));
      if constexpr (Interior)
        values[j] = __bfloat162float(__float2bfloat16_rn(values[j]));
      residual_vec.v[j] = values[j];
    }
  }
  if constexpr (StoreCarry) {
    if (tid < ACTIVE)
      reinterpret_cast<FVec<E>*>(residual_out)[vector_offset] = residual_vec;
  }
  __syncthreads();
  total = Reduce(temp).Sum(sum);
  if (tid == 0) inverse = rsqrtf(total / H + eps);
  __syncthreads();
  const float inv_pre = inverse;
  float maximum = 0;
  BVec<E> norm_vec;
  if (tid < ACTIVE) {
#pragma unroll
    for (int j = 0; j < E; ++j) {
      const float weight = __bfloat162float(nw.v[j]);
      values[j] = __bfloat162float(__float2bfloat16_rn(
          __fmul_rn(__fmul_rn(values[j], inv_pre), weight)));
      norm_vec.v[j] = __float2bfloat16_rn(values[j]);
      maximum = fmaxf(maximum, fabsf(values[j]));
    }
  }
  if constexpr (!Quant) {
    if (tid < ACTIVE) reinterpret_cast<BVec<E>*>(q)[vector_offset] = norm_vec;
    return;
  }
#pragma unroll
  for (int mask = GROUP_THREADS / 2; mask; mask /= 2)
    maximum = fmaxf(maximum,
                    __shfl_xor_sync(0xffffffff, maximum, mask, GROUP_THREADS));
  if (tid >= ACTIVE) return;
  float scale = fmaxf(fmaxf(maximum, 1e-10f) / 448.0f, 1e-10f);
  const unsigned bits = __float_as_uint(scale);
  scale =
      __uint_as_float((((bits >> 23) & 255) + ((bits & 0x7fffff) != 0)) << 23);
  if (tid % GROUP_THREADS == 0)
    scales[(tid / GROUP_THREADS) * pitch + blockIdx.x] = scale;
  QVec<E> packed;
#pragma unroll
  for (int j = 0; j < E; ++j) packed.v[j] = __nv_fp8_e4m3(values[j] / scale);
  reinterpret_cast<QVec<E>*>(q)[vector_offset] = packed;
}

extern "C" int launch_chain(const void* x, const void* residual,
                            const void* post_weight, const void* pre_weight,
                            void* residual_out, void* q, void* scales, int rows,
                            int elements_per_thread, float eps, void* stream,
                            int mode, const void* scalar) {
  if (rows <= 0 || mode < 0 || mode > 6 ||
      ((mode % 3 == 2 || mode == 6) && !scalar))
    return int(cudaErrorInvalidValue);
  const int pitch = (rows + 3) / 4 * 4;
#define RUN(E, F, S, Q, I, C)                                            \
  chain<E, F, S, Q, I, C><<<rows, ((H / E + 31) / 32 * 32), 0,           \
                            static_cast<cudaStream_t>(stream)>>>(        \
      static_cast<const __nv_bfloat16*>(x), residual,                    \
      static_cast<const __nv_bfloat16*>(post_weight),                    \
      static_cast<const __nv_bfloat16*>(pre_weight),                     \
      static_cast<float*>(residual_out), q, static_cast<float*>(scales), \
      pitch, eps, static_cast<const __nv_bfloat16*>(scalar))
#define SELECT(E)                             \
  if (mode == 0) {                            \
    RUN(E, false, false, true, false, true);  \
  } else if (mode == 1) {                     \
    RUN(E, true, false, true, false, true);   \
  } else if (mode == 2) {                     \
    RUN(E, true, true, true, true, true);     \
  } else if (mode == 3) {                     \
    RUN(E, false, false, false, false, true); \
  } else if (mode == 4) {                     \
    RUN(E, true, false, false, false, true);  \
  } else if (mode == 5) {                     \
    RUN(E, true, true, false, false, false);  \
  } else {                                    \
    RUN(E, true, true, false, true, true);    \
  }
  if (elements_per_thread == 8) {
    SELECT(8);
  } else if (elements_per_thread == 16) {
    SELECT(16);
  } else if (elements_per_thread == 32) {
    SELECT(32);
  } else
    return int(cudaErrorInvalidValue);
#undef SELECT
#undef RUN
  return int(cudaGetLastError());
}

void gemma4_full_chain_out(torch::stable::Tensor const& input,
                           torch::stable::Tensor const& residual,
                           torch::stable::Tensor const& post_weight,
                           torch::stable::Tensor const& pre_weight,
                           torch::stable::Tensor const& scalar,
                           torch::stable::Tensor& carry,
                           torch::stable::Tensor& output,
                           torch::stable::Tensor& scales, int64_t mode,
                           double epsilon) {
  using ScalarType = torch::headeronly::ScalarType;
  STD_TORCH_CHECK(input.dim() == 2 && input.size(1) == H,
                  "Gemma4 input must have shape [M, 5376]");
  int64_t const rows = input.size(0);
  STD_TORCH_CHECK(rows > 0 && rows <= std::numeric_limits<int>::max() / H,
                  "Gemma4 rows must be positive and fit kernel indexing");
  STD_TORCH_CHECK(mode >= 0 && mode <= 6, "Invalid Gemma4 native mode");
  STD_TORCH_CHECK(epsilon == 1e-6, "Gemma4 epsilon must be 1e-6");
  STD_TORCH_CHECK(input.device().is_cuda(), "Gemma4 input must be CUDA");
  int32_t const device = input.get_device_index();
  torch::stable::Tensor const* tensors[] = {
      &input,  &residual, &post_weight, &pre_weight,
      &scalar, &carry,    &output,      &scales};
  for (auto const* tensor : tensors) {
    STD_TORCH_CHECK(
        tensor->device().is_cuda() && tensor->get_device_index() == device,
        "Gemma4 tensors must share one CUDA device");
    STD_TORCH_CHECK(tensor->is_contiguous(),
                    "Gemma4 tensors must be contiguous");
  }
  STD_TORCH_CHECK(input.scalar_type() == ScalarType::BFloat16 &&
                      post_weight.scalar_type() == ScalarType::BFloat16 &&
                      pre_weight.scalar_type() == ScalarType::BFloat16 &&
                      scalar.scalar_type() == ScalarType::BFloat16,
                  "Gemma4 input, weights and scalar must be BF16");
  STD_TORCH_CHECK(
      post_weight.dim() == 1 && post_weight.numel() == H &&
          pre_weight.dim() == 1 && pre_weight.numel() == H &&
          scalar.numel() == 1,
      "Gemma4 weights must have shape [5376] and scalar one element");
  bool const bf16_residual = mode == 0 || mode == 3;
  STD_TORCH_CHECK(
      residual.dim() == 2 && residual.size(0) == rows &&
          residual.size(1) == H &&
          residual.scalar_type() ==
              (bf16_residual ? ScalarType::BFloat16 : ScalarType::Float),
      "Gemma4 residual has wrong shape or mode-specific dtype");
  STD_TORCH_CHECK(carry.scalar_type() == ScalarType::Float &&
                      ((mode == 5 && carry.dim() == 1 && carry.numel() == 0) ||
                       (mode != 5 && carry.dim() == 2 &&
                        carry.size(0) == rows && carry.size(1) == H)),
                  "Gemma4 carry must be FP32 [M, 5376], or empty for mode 5");
  bool const quantized = mode <= 2;
  STD_TORCH_CHECK(
      output.dim() == 2 && output.size(0) == rows && output.size(1) == H &&
          output.scalar_type() ==
              (quantized ? ScalarType::Float8_e4m3fn : ScalarType::BFloat16),
      "Gemma4 output has wrong shape or mode-specific dtype");
  int64_t const pitch = (rows + 3) / 4 * 4;
  STD_TORCH_CHECK(scales.scalar_type() == ScalarType::Float &&
                      scales.dim() == 2 && scales.size(0) == H / 128 &&
                      scales.size(1) == pitch,
                  "Gemma4 scale storage must be FP32 [42, roundup(M, 4)]");
  for (auto const* tensor :
       {&input, &residual, &post_weight, &pre_weight,
        static_cast<torch::stable::Tensor const*>(&carry),
        static_cast<torch::stable::Tensor const*>(&output)}) {
    STD_TORCH_CHECK(
        tensor->numel() == 0 ||
            reinterpret_cast<uintptr_t>(tensor->const_data_ptr()) % 16 == 0,
        "Gemma4 vector buffers require 16-byte alignment");
  }
  STD_TORCH_CHECK(
      reinterpret_cast<uintptr_t>(scalar.const_data_ptr()) % 2 == 0 &&
          reinterpret_cast<uintptr_t>(scales.const_data_ptr()) % 4 == 0,
      "Gemma4 scalar or scale storage is unaligned");
  auto overlaps = [](torch::stable::Tensor const& a,
                     torch::stable::Tensor const& b) {
    if (a.numel() == 0 || b.numel() == 0) return false;
    auto const ab = reinterpret_cast<uintptr_t>(a.const_data_ptr());
    auto const bb = reinterpret_cast<uintptr_t>(b.const_data_ptr());
    return ab < bb + b.numel() * b.element_size() &&
           bb < ab + a.numel() * a.element_size();
  };
  torch::stable::Tensor const* inputs[] = {&input, &residual, &post_weight,
                                           &pre_weight, &scalar};
  torch::stable::Tensor const* outputs[] = {&carry, &output, &scales};
  for (int i = 0; i < 3; ++i) {
    for (auto const* read : inputs) {
      STD_TORCH_CHECK(!overlaps(*outputs[i], *read),
                      "Gemma4 outputs must not alias inputs");
    }
    for (int j = 0; j < i; ++j) {
      STD_TORCH_CHECK(!overlaps(*outputs[i], *outputs[j]),
                      "Gemma4 outputs must not alias each other");
    }
  }
  const torch::stable::accelerator::DeviceGuard device_guard(device);
  auto const* properties = get_device_prop();
  STD_TORCH_CHECK(properties->major == 9 && properties->minor == 0,
                  "Gemma4 full chain requires SM90");
  int const error = launch_chain(
      input.const_data_ptr(), residual.const_data_ptr(),
      post_weight.const_data_ptr(), pre_weight.const_data_ptr(),
      carry.mutable_data_ptr(), output.mutable_data_ptr(),
      scales.mutable_data_ptr(), static_cast<int>(rows), rows <= 128 ? 8 : 16,
      static_cast<float>(epsilon), get_current_cuda_stream(device),
      static_cast<int>(mode), scalar.const_data_ptr());
  STD_TORCH_CHECK(error == cudaSuccess, "Gemma4 launch failed: ",
                  cudaGetErrorString(static_cast<cudaError_t>(error)));
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("gemma4_full_chain_out", TORCH_BOX(&gemma4_full_chain_out));
}
