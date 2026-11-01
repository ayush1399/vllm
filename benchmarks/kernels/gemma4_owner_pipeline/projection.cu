// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Prototype built against FlashInfer's vendored CUTLASS/DeepGEMM SM90 kernel.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <string>
#include <cstdio>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include "deep_gemm/fp8_gemm_impl.cuh"

struct alignas(64) Params {
  CUtensorMap w, x, xs, d;
  void* output;
  float* ws;
  int rows, k, bm, bn, stages;
};

static int encode(CUtensorMap* map, void* ptr, CUtensorMapDataType type,
                  uint64_t cols, uint64_t rows, uint64_t stride,
                  uint32_t tile_cols, uint32_t tile_rows,
                  CUtensorMapSwizzle swizzle) {
  uint64_t dims[2] = {cols, rows};
  uint32_t box[2] = {tile_cols, tile_rows}, element[2] = {1, 1};
  return int(cuTensorMapEncodeTiled(map, type, 2, ptr, dims, &stride, box,
                                    element, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                    swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

extern "C" int prepare(void* x, void* w, void* xs, void* ws, void* out,
                       int rows, int k, int bm, int bn, int stages,
                       void** result) {
  auto* p = new Params{};
  p->rows = rows;
  p->k = k;
  p->bm = bm;
  p->bn = bn;
  p->stages = stages;
  p->output = out;
  p->ws = (float*)ws;
  int e = encode(&p->w, w, CU_TENSOR_MAP_DATA_TYPE_UINT8, k, 5376, k, 128, bm,
                 CU_TENSOR_MAP_SWIZZLE_128B);
  if (!e)
    e = encode(&p->x, x, CU_TENSOR_MAP_DATA_TYPE_UINT8, k, rows, k, 128, bn,
               CU_TENSOR_MAP_SWIZZLE_128B);
  const int pitch = (rows + 3) / 4 * 4;
  if (!e)
    e = encode(&p->xs, xs, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, pitch, k / 128,
               pitch * 4, bn, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
  if (!e)
    e = encode(&p->d, out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5376, rows,
               5376 * 2, bm, bn < rows ? bn : rows, CU_TENSOR_MAP_SWIZZLE_NONE);
  if (e) {
    delete p;
    return e;
  }
  *result = p;
  return 0;
}

template <int K, int BM, int BN, int STAGES, bool PUB, bool SCATTER = false>
static int run(Params* p, uint64_t mc, int cap, int parity, int rank,
               cudaStream_t stream) {
  using namespace deep_gemm;
  using S = NormalSchedulerSwapAB<5376, BM, BN, 1, 1>;
  auto kernel =
      fp8_gemm_kernel_swapAB<5376, K, BM, BN, 128, 1, STAGES, 128, 128, 1, S,
                             NormalSchedulerInputSwapAB, PUB, SCATTER>;
  const int smem =
      BM * BN * 2 +
      STAGES * (BM * 128 + BN * 128 + ((BN * 4 + 127) / 128 * 128)) +
      ((K / 128 * 4 + 7) / 8 * 8) +
      2 * STAGES * sizeof(cutlass::arch::ClusterTransactionBarrier);
  cudaError_t e = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e != cudaSuccess) return int(e);
  cudaLaunchConfig_t config{};
  config.gridDim = 132;
  config.blockDim = (BM == 64 ? 256 : 384);
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim = {1, 1, 1};
  config.attrs = &attribute;
  config.numAttrs = 1;
  NormalSchedulerInputSwapAB input{uint32_t(p->rows), nullptr};
  e = cudaLaunchKernelEx(&config, kernel, (__nv_bfloat16*)p->output, p->ws,
                         input, p->w, p->x, p->xs, p->d, mc, cap, parity, rank);
  return int(e);
}

template <int K, int BM, int BN, int STAGES>
static int dispatch_publish(Params* p, uint64_t mc, int cap, int parity,
                            int rank, cudaStream_t stream, int pub) {
  if (pub == 2)
    return run<K, BM, BN, STAGES, true, true>(p, mc, cap, parity, rank, stream);
  return pub ? run<K, BM, BN, STAGES, true>(p, mc, cap, parity, rank, stream)
             : run<K, BM, BN, STAGES, false>(p, mc, cap, parity, rank, stream);
}
template <int K, int BM, int BN>
static int dispatch_stages(Params* p, uint64_t mc, int cap, int parity,
                           int rank, cudaStream_t stream, int pub) {
  if (p->stages == 4)
    return dispatch_publish<K, BM, BN, 4>(p, mc, cap, parity, rank, stream,
                                          pub);
  if (p->stages == 6)
    return dispatch_publish<K, BM, BN, 6>(p, mc, cap, parity, rank, stream,
                                          pub);
  return dispatch_publish<K, BM, BN, 8>(p, mc, cap, parity, rank, stream, pub);
}
template <int K>
static int dispatch_tile(Params* p, uint64_t mc, int cap, int parity, int rank,
                         cudaStream_t stream, int pub) {
  if (p->bm == 64) {
    if (p->bn == 16)
      return dispatch_stages<K, 64, 16>(p, mc, cap, parity, rank, stream, pub);
    return dispatch_stages<K, 64, 32>(p, mc, cap, parity, rank, stream, pub);
  }
  if (p->bn == 16)
    return dispatch_stages<K, 128, 16>(p, mc, cap, parity, rank, stream, pub);
  return dispatch_stages<K, 128, 32>(p, mc, cap, parity, rank, stream, pub);
}
extern "C" int launch_projection(void* params, uint64_t mc, int cap, int parity,
                                 int rank, void* stream, int pub) {
  auto* p = (Params*)params;
  if (p->k == 1024)
    return dispatch_tile<1024>(p, mc, cap, parity, rank, (cudaStream_t)stream,
                               pub);
  if (p->k == 2048)
    return dispatch_tile<2048>(p, mc, cap, parity, rank, (cudaStream_t)stream,
                               pub);
  if (p->k == 2688)
    return dispatch_tile<2688>(p, mc, cap, parity, rank, (cudaStream_t)stream,
                               pub);
  return 1;
}
extern "C" void release(void* params) { delete (Params*)params; }
