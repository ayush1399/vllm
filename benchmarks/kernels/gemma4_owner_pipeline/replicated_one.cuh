// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
          bool StoreCarry, bool RawQuant = false>
__global__ void chain(__nv_bfloat16* mailbox, int capacity, int parity,
                      int* error, const void* residual,
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
  if (tid == 0) cudaTriggerProgrammaticLaunchCompletion();
  __shared__ int failed;
  if (tid == 0) failed = 0;
  __syncthreads();

  float values[E];
  BVec<E> xv, pw, nw;
  FVec<E> rw;
  FVec<E> residual_vec;
  const int vector_offset = blockIdx.x * ACTIVE + tid;
  float sum = 0;
  if (tid < ACTIVE) {
    float reduced[E] = {};
    const uint64_t start = clock64();
#pragma unroll
    for (int rank = 0; rank < 8; ++rank) {
      BVec<E> partial;
#pragma unroll
      for (int p = 0; p < E / 8; ++p) {
        auto* addr =
            mailbox + (((parity * 8 + rank) * capacity + blockIdx.x) * H +
                       tid * E + p * 8);
        uint4 value;
        bool ready;
        do {
          asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(value.x), "=r"(value.y), "=r"(value.z),
                         "=r"(value.w)
                       : "l"(addr)
                       : "memory");
          ready = value.x != 0x80000000u && value.y != 0x80000000u &&
                  value.z != 0x80000000u && value.w != 0x80000000u;
          if (clock64() - start > 2000000000ULL) {
            atomicExch(&failed, 1);
            atomicExch(error, 1);
            break;
          }
        } while (!ready);
        reinterpret_cast<uint4*>(&partial)[p] = value;
        asm volatile(
            "st.volatile.global.v4.u32 [%0], {%1,%1,%1,%1};" ::"l"(addr),
            "r"(0x80000000u)
            : "memory");
      }
#pragma unroll
      for (int j = 0; j < E; ++j)
        reduced[j] = __fadd_rn(reduced[j], __bfloat162float(partial.v[j]));
    }
#pragma unroll
    for (int j = 0; j < E; ++j) xv.v[j] = __float2bfloat16_rn(reduced[j]);

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
  __syncthreads();
  if (failed) return;
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
  float scale;
  float inverse_scale;
  if constexpr (RawQuant) {
    // Match the FlashInfer CUTLASS quantizer, including FP8 rounding ties.
    inverse_scale = maximum != 0.0f ? __fdividef(448.0f, maximum) : 1.0f;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(scale) : "f"(inverse_scale));
  } else {
    scale = fmaxf(fmaxf(maximum, 1e-10f) / 448.0f, 1e-10f);
    const unsigned bits = __float_as_uint(scale);
    scale = __uint_as_float((((bits >> 23) & 255) + ((bits & 0x7fffff) != 0))
                            << 23);
  }
  if (tid % GROUP_THREADS == 0)
    scales[(tid / GROUP_THREADS) * pitch + blockIdx.x] = scale;
  QVec<E> packed;
#pragma unroll
  for (int j = 0; j < E; ++j) {
    if constexpr (RawQuant)
      packed.v[j] = __nv_fp8_e4m3(__fmul_rn(values[j], inverse_scale));
    else
      packed.v[j] = __nv_fp8_e4m3(values[j] / scale);
  }
  reinterpret_cast<QVec<E>*>(q)[vector_offset] = packed;
}
