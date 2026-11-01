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
#include "next_deep_gemm/fp8_gemm_impl.cuh"

struct alignas(128) Params {
  CUtensorMap w, x, xs, d;
  void* output;
  float* ws;
  int rows, k, bm, bn, stages, n;
  uint32_t *ready, *expected;
  int* error;
  int alias;
  uint64_t* ready_peers;
  int rank;
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
                       int rows, int n, int bm, int bn, int stages,
                       uint32_t* ready, uint32_t* expected, int* error,
                       int alias, uint64_t* ready_peers, int rank,
                       void** result) {
  auto* p = new Params{};
  int k = 5376;
  p->rows = rows;
  p->k = k;
  p->n = n;
  p->ready = ready;
  p->expected = expected;
  p->error = error;
  p->alias = alias;
  p->ready_peers = ready_peers;
  p->rank = rank;
  p->bm = bm;
  p->bn = bn;
  p->stages = stages;
  p->output = out;
  p->ws = (float*)ws;
  int e = encode(&p->w, w, CU_TENSOR_MAP_DATA_TYPE_UINT8, k, n, k, 128, bm,
                 CU_TENSOR_MAP_SWIZZLE_128B);
  if (!e)
    e = encode(&p->x, x, CU_TENSOR_MAP_DATA_TYPE_UINT8, k, rows, k, 128, bn,
               CU_TENSOR_MAP_SWIZZLE_128B);
  const int pitch = (rows + 3) / 4 * 4;
  if (!e)
    e = encode(&p->xs, xs, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, pitch, k / 128,
               pitch * 4, bn, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
  if (!e)
    e = encode(&p->d, out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, n, rows, n * 2, bm,
               bn < rows ? bn : rows, CU_TENSOR_MAP_SWIZZLE_NONE);
  if (e) {
    delete p;
    return e;
  }
  *result = p;
  return 0;
}

template <int N, int BM, int BN, int STAGES, bool WAIT>
static int run(Params* p, cudaStream_t stream, int early) {
  using namespace deep_gemm;
  using S = NormalSchedulerSwapAB<N, BM, BN, 1, 1>;
  auto kernel =
      fp8_gemm_kernel_swapAB<N, 5376, BM, BN, 128, 1, STAGES, 128, 128, 1, S,
                             NormalSchedulerInputSwapAB, WAIT>;
  const int smem =
      BM * BN * 2 +
      STAGES * (BM * 128 + BN * 128 + ((BN * 4 + 127) / 128 * 128)) +
      ((5376 / 128 * 4 + 7) / 8 * 8) +
      2 * STAGES * sizeof(cutlass::arch::ClusterTransactionBarrier);
  auto e = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e != cudaSuccess) return int(e);
  cudaLaunchConfig_t config{};
  config.gridDim = 132;
  config.blockDim = (BM == 64 ? 256 : 384);
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {1, 1, 1};
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = early ? 2 : 1;
  NormalSchedulerInputSwapAB input{uint32_t(p->rows), nullptr};
  return int(cudaLaunchKernelEx(&config, kernel, (__nv_bfloat16*)p->output,
                                p->ws, input, p->w, p->x, p->xs, p->d, p->ready,
                                p->expected, p->error, p->alias, p->ready_peers,
                                p->rank));
}

template <int N, int BM, int BN, int ST>
static int dispatch_wait(Params* p, cudaStream_t s, int wait, int early) {
  return wait ? run<N, BM, BN, ST, true>(p, s, early)
              : run<N, BM, BN, ST, false>(p, s, early);
}
template <int N, int BM, int BN>
static int dispatch_stages(Params* p, cudaStream_t s, int wait, int early) {
  if (p->stages == 4) return dispatch_wait<N, BM, BN, 4>(p, s, wait, early);
  if constexpr (BM == 64) {
    if (p->stages == 16) return dispatch_wait<N, BM, BN, 16>(p, s, wait, early);
    if constexpr (BN == 16) {
      if (p->stages == 20)
        return dispatch_wait<N, BM, BN, 20>(p, s, wait, early);
    }
  } else {
    if (p->stages == 10) return dispatch_wait<N, BM, BN, 10>(p, s, wait, early);
  }
  return dispatch_wait<N, BM, BN, 8>(p, s, wait, early);
}
template <int N>
static int dispatch_tile(Params* p, cudaStream_t s, int wait, int early) {
  if (p->bm == 64) {
    if (p->bn == 16) return dispatch_stages<N, 64, 16>(p, s, wait, early);
    return dispatch_stages<N, 64, 32>(p, s, wait, early);
  }
  if (p->bn == 16) return dispatch_stages<N, 128, 16>(p, s, wait, early);
  return dispatch_stages<N, 128, 32>(p, s, wait, early);
}
extern "C" int launch_next(void* params, void* stream, int wait, int early) {
  auto* p = (Params*)params;
  if (p->n == 5376)
    return dispatch_tile<5376>(p, (cudaStream_t)stream, wait, early);
  if (p->n == 2048)
    return dispatch_tile<2048>(p, (cudaStream_t)stream, wait, early);
  return int(cudaErrorInvalidValue);
}
extern "C" void release(void* params) { delete (Params*)params; }
