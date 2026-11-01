// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cstdint>

constexpr int H = 5376;
constexpr int E = 8;
constexpr int Threads = H / E;

template <class T, int N>
struct alignas(sizeof(T) * N) Vec {
  T v[N];
};

template <bool Interior, int Early, int Publication>
__global__ void owner_norm(__nv_bfloat16* scatter, int capacity, int parity,
                           int rank, int* error, const float* residual,
                           const __nv_bfloat16* post, const __nv_bfloat16* pre,
                           float* carry, uint64_t q_mc, uint64_t scales_mc,
                           uint64_t ready_peers, uint32_t* generations,
                           int pitch, const __nv_bfloat16* scalar) {
  if constexpr (Early == 1) {
    if (threadIdx.x == 0) cudaTriggerProgrammaticLaunchCompletion();
  }
  const int tid = threadIdx.x;
  const int row = blockIdx.x * 8 + rank;
  if (row >= capacity) {
    if constexpr (Early == 2) {
      if (tid == 0) cudaTriggerProgrammaticLaunchCompletion();
    }
    return;
  }
  const int owner_cap = (capacity + 7) / 8;
  using Reduce = cub::BlockReduce<float, Threads>;
  __shared__ typename Reduce::TempStorage temp;
  __shared__ float inverse;
  __shared__ int failed;
  if (tid == 0) failed = 0;
  __syncthreads();
  Vec<float, E> rw;
  Vec<__nv_bfloat16, E> pw, nw;
  auto* r_addr = residual + blockIdx.x * H + tid * E;
  auto* p_addr = post + tid * E;
  auto* n_addr = pre + tid * E;
  auto* rr = reinterpret_cast<uint4*>(&rw);
#pragma unroll
  for (int v = 0; v < 2; ++v) {
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(rr[v].x), "=r"(rr[v].y), "=r"(rr[v].z), "=r"(rr[v].w)
                 : "l"(r_addr + v * 4)
                 : "memory");
  }
  auto* pp = reinterpret_cast<uint4*>(&pw);
  auto* nn = reinterpret_cast<uint4*>(&nw);
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(pp->x), "=r"(pp->y), "=r"(pp->z), "=r"(pp->w)
               : "l"(p_addr)
               : "memory");
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(nn->x), "=r"(nn->y), "=r"(nn->z), "=r"(nn->w)
               : "l"(n_addr)
               : "memory");
  float values[E] = {};
  const auto start = clock64();
  uint4 parts[8];
#pragma unroll
  for (int src = 0; src < 8; ++src) {
    auto* addr =
        scatter + (((parity * 8 + src) * owner_cap + blockIdx.x) * H + tid * E);
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(parts[src].x), "=r"(parts[src].y), "=r"(parts[src].z),
                   "=r"(parts[src].w)
                 : "l"(addr)
                 : "memory");
  }
  if constexpr (Early == 2) {
    __syncthreads();
    if (tid == 0) cudaTriggerProgrammaticLaunchCompletion();
  }
#pragma unroll
  for (int src = 0; src < 8; ++src) {
    auto* addr =
        scatter + (((parity * 8 + src) * owner_cap + blockIdx.x) * H + tid * E);
    uint4 v = parts[src];
    bool ready = v.x != 0x80000000u && v.y != 0x80000000u &&
                 v.z != 0x80000000u && v.w != 0x80000000u;
    while (!ready) {
      asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                   : "l"(addr)
                   : "memory");
      ready = v.x != 0x80000000u && v.y != 0x80000000u && v.z != 0x80000000u &&
              v.w != 0x80000000u;
      if (clock64() - start > 2000000000ULL) {
        atomicExch(&failed, 1);
        atomicExch(error, 1);
        break;
      }
    }
    auto part = *reinterpret_cast<Vec<__nv_bfloat16, E>*>(&v);
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%1,%1,%1};" ::"l"(addr),
                 "r"(0x80000000u)
                 : "memory");
#pragma unroll
    for (int j = 0; j < E; ++j)
      values[j] = __fadd_rn(values[j], __bfloat162float(part.v[j]));
  }
  __syncthreads();
  if (failed) return;
  float sum = 0;
#pragma unroll
  for (int j = 0; j < E; ++j) {
    values[j] = __bfloat162float(__float2bfloat16_rn(values[j]));
    sum = __fadd_rn(sum, __fmul_rn(values[j], values[j]));
  }
  float total = Reduce(temp).Sum(sum);
  if (tid == 0) inverse = rsqrtf(total / H + 1e-6f);
  __syncthreads();
  const float inv_post = inverse;
  Vec<float, E> out;
  sum = 0;
#pragma unroll
  for (int j = 0; j < E; ++j) {
    values[j] = fmaf(__fmul_rn(values[j], inv_post), __bfloat162float(pw.v[j]),
                     rw.v[j]);
    if constexpr (Interior)
      values[j] = __fmul_rn(values[j], __bfloat162float(*scalar));
    sum = __fadd_rn(sum, __fmul_rn(values[j], values[j]));
    if constexpr (Interior)
      values[j] = __bfloat162float(__float2bfloat16_rn(values[j]));
    out.v[j] = values[j];
  }
  reinterpret_cast<Vec<float, E>*>(carry)[blockIdx.x * Threads + tid] = out;
  __syncthreads();
  total = Reduce(temp).Sum(sum);
  if (tid == 0) inverse = rsqrtf(total / H + 1e-6f);
  __syncthreads();
  float maximum = 0;
#pragma unroll
  for (int j = 0; j < E; ++j) {
    values[j] = __bfloat162float(__float2bfloat16_rn(
        __fmul_rn(__fmul_rn(values[j], inverse), __bfloat162float(nw.v[j]))));
    maximum = fmaxf(maximum, fabsf(values[j]));
  }
#pragma unroll
  for (int mask = 8; mask; mask /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, mask, 16));
  float inverse_scale = maximum != 0 ? __fdividef(448.0f, maximum) : 1.0f;
  float scale;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(scale) : "f"(inverse_scale));
  Vec<__nv_fp8_e4m3, E> packed;
#pragma unroll
  for (int j = 0; j < E; ++j)
    packed.v[j] = __nv_fp8_e4m3(__fmul_rn(values[j], inverse_scale));
  uint2 bits = *reinterpret_cast<uint2*>(&packed);
  if constexpr (Publication == 1) {
    for (int dst = 0; dst < 8; ++dst) {
      const uint64_t q_addr = reinterpret_cast<const uint64_t*>(q_mc)[dst] +
                              uint64_t(parity * capacity + row) * H + tid * E;
      asm volatile("st.relaxed.sys.global.v2.u32 [%0], {%1,%2};" ::"l"(q_addr),
                   "r"(bits.x), "r"(bits.y)
                   : "memory");
      if (tid % 16 == 0) {
        const uint64_t s_addr =
            reinterpret_cast<const uint64_t*>(scales_mc)[dst] +
            (parity * (H / 128) * pitch + (tid / 16) * pitch + row) * 4;
        asm volatile("st.relaxed.sys.global.f32 [%0], %1;" ::"l"(s_addr),
                     "f"(scale)
                     : "memory");
      }
    }
  } else {
    const uint64_t q_addr = q_mc + uint64_t(row) * H + tid * E;
    asm volatile(
        "multimem.st.relaxed.sys.global.v2.f32 [%0], {%1,%2};" ::"l"(q_addr),
        "f"(__uint_as_float(bits.x)), "f"(__uint_as_float(bits.y))
        : "memory");
    if (tid % 16 == 0) {
      const uint64_t s_addr = scales_mc + ((tid / 16) * pitch + row) * 4;
      asm volatile("multimem.st.relaxed.sys.global.f32 [%0], %1;" ::"l"(s_addr),
                   "f"(scale)
                   : "memory");
    }
  }
  if constexpr (Publication == 3) return;
  if constexpr (Publication == 2)
    asm volatile("fence.proxy.alias;" ::: "memory");
  else
    asm volatile("fence.acq_rel.sys;" ::: "memory");
  __syncthreads();
  if (tid == 0) {
    uint32_t epoch = ++generations[blockIdx.x];
    asm volatile("fence.acq_rel.sys;" ::: "memory");
    for (int dst = 0; dst < 8; ++dst) {
      const uint64_t addr =
          reinterpret_cast<const uint64_t*>(ready_peers)[dst] + row * 4;
      asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(addr), "r"(epoch)
                   : "memory");
    }
  }
}

extern "C" int launch_owner(void* scatter, int rows, int parity, int rank,
                            int* error, const float* residual, const void* post,
                            const void* pre, float* carry, uint64_t q_mc,
                            uint64_t scales_mc, uint64_t ready_peers,
                            uint32_t* generations, int mode, const void* scalar,
                            void* stream, int early, int publication) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = (rows + 7) / 8;
  cfg.blockDim = Threads;
  cfg.stream = static_cast<cudaStream_t>(stream);
#define RUN(I, P, B)                                                           \
  cudaLaunchKernelEx(&cfg, owner_norm<I, P, B>,                                \
                     static_cast<__nv_bfloat16*>(scatter), rows, parity, rank, \
                     error, residual, static_cast<const __nv_bfloat16*>(post), \
                     static_cast<const __nv_bfloat16*>(pre), carry, q_mc,      \
                     scales_mc, ready_peers, generations, (rows + 3) / 4 * 4,  \
                     static_cast<const __nv_bfloat16*>(scalar))
#define SELECT(B)                                \
  if (mode == 2) {                               \
    if (early == 2) return int(RUN(true, 2, B)); \
    if (early) return int(RUN(true, 1, B));      \
    return int(RUN(true, false, B));             \
  }                                              \
  if (early == 2) return int(RUN(false, 2, B));  \
  if (early) return int(RUN(false, 1, B));       \
  return int(RUN(false, false, B))
  if (publication == 1) {
    SELECT(1);
  }
  if (publication == 2) {
    SELECT(2);
  }
  if (publication == 3) {
    SELECT(3);
  }
  SELECT(0);
#undef SELECT
#undef RUN
}
