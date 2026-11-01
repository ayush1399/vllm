// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda.h>
#include <cuda_runtime.h>

#include "full_chain.cuh"
#include "pdl_gemm.cuh"

namespace {
struct Params {
  CUtensorMap weight, input, scales, output;
  __nv_bfloat16* out;
  float* weight_scales;
  int rows, sms;
};

template <int N, int BN, int Stages, int SplitK = 1>
auto kernel() {
  using namespace deep_gemm;
  using Scheduler = NormalSchedulerSwapAB<N, 64, BN, 1, 1>;
  return gemma4_pdl_gemm_kernel<N, 5376 / SplitK, 64, BN, 128, 1, Stages, 128,
                                128, 1, Scheduler, NormalSchedulerInputSwapAB,
                                5376>;
}

template <int BN, int Stages>
constexpr int smem_size() {
  return 64 * BN * 2 + Stages * (64 * 128 + BN * 128 + 128) + 168 +
         2 * Stages * sizeof(cutlass::arch::ClusterTransactionBarrier);
}

template <int N, int BN, int Stages, int SplitK = 1>
int initialize_kernel() {
  return int(cudaFuncSetAttribute(kernel<N, BN, Stages, SplitK>(),
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  smem_size<BN, Stages>()));
}

int encode(CUtensorMap* map, void* data, CUtensorMapDataType dtype,
           uint64_t cols, uint64_t rows, uint64_t stride, uint32_t tile_cols,
           uint32_t tile_rows, CUtensorMapSwizzle swizzle) {
  uint64_t dims[] = {cols, rows};
  uint32_t tile[] = {tile_cols, tile_rows}, step[] = {1, 1};
  return int(cuTensorMapEncodeTiled(map, dtype, 2, data, dims, &stride, tile,
                                    step, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                    swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

template <int N, int BN, int Stages, int SplitK = 1>
int launch(Params& p, cudaStream_t stream) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(SplitK == 1 ? p.sms : (N / 64) * ((p.rows + BN - 1) / BN),
                     SplitK, 1);
  cfg.blockDim = 256;
  cfg.dynamicSmemBytes = smem_size<BN, Stages>();
  cfg.stream = stream;
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {1, 1, 1};
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;
  deep_gemm::NormalSchedulerInputSwapAB input{uint32_t(p.rows), nullptr};
  return int(cudaLaunchKernelEx(&cfg, kernel<N, BN, Stages, SplitK>(), p.out,
                                p.weight_scales, input, p.weight, p.input,
                                p.scales, p.output));
}
}  // namespace

extern "C" int initialize_pdl() {
  int e = initialize_kernel<5376, 16, 16>();
  if (!e) e = initialize_kernel<5376, 32, 8>();
  if (!e) e = initialize_kernel<2048, 16, 16>();
  if (!e) e = initialize_kernel<2048, 16, 7, 6>();
  return e;
}

static int launch_norm_gemm_impl(
    const __nv_bfloat16* x, const void* residual, const __nv_bfloat16* post,
    const __nv_bfloat16* pre, const __nv_bfloat16* scalar, void* carry, void* q,
    float* scales, void* weight, float* weight_scales, __nv_bfloat16* output,
    int rows, int n, int mode, int sms, cudaStream_t stream, bool split_k) {
  if (rows < 1 || rows >= 32 || (n != 5376 && n != 2048) || mode < 0 ||
      (mode > 2 && mode != 11 && mode != 12) || sms < 1)
    return int(cudaErrorInvalidValue);
  const int pitch = (rows + 3) / 4 * 4;
  const int bn = n == 5376 && rows > 16 ? 32 : 16;
  Params p{};
  p.rows = rows;
  p.sms = sms;
  p.out = output;
  p.weight_scales = weight_scales;
  int e = encode(&p.weight, weight, CU_TENSOR_MAP_DATA_TYPE_UINT8, 5376, n,
                 5376, 128, 64, CU_TENSOR_MAP_SWIZZLE_128B);
  if (!e)
    e = encode(&p.input, q, CU_TENSOR_MAP_DATA_TYPE_UINT8, 5376, rows, 5376,
               128, bn, CU_TENSOR_MAP_SWIZZLE_128B);
  if (!e)
    e = encode(&p.scales, scales, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, pitch, 42,
               pitch * 4, bn, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
  if (!e)
    e = encode(&p.output, output, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, n, rows,
               n * 2, 64, bn < rows ? bn : rows, CU_TENSOR_MAP_SWIZZLE_NONE);
  if (e) return e;

#define NORM(F, S, I, B)                                                  \
  chain<8, F, S, true, I, true, true, true, B><<<rows, 672, 0, stream>>>( \
      x, residual, post, pre, carry, q, scales, pitch, 1e-6f, scalar)
  if (mode == 0) {
    NORM(false, false, false, false);
  } else if (mode == 1) {
    NORM(true, false, false, false);
  } else if (mode == 11) {
    NORM(false, false, false, true);
  } else if (mode == 12) {
    NORM(true, true, true, true);
  } else {
    NORM(true, true, true, false);
  }
#undef NORM
  e = int(cudaGetLastError());
  if (e) return e;
  if (split_k) {
    if (n != 2048 || rows > 8) return int(cudaErrorInvalidValue);
    return launch<2048, 16, 7, 6>(p, stream);
  }
  if (n == 2048) return launch<2048, 16, 16>(p, stream);
  if (rows > 16) return launch<5376, 32, 8>(p, stream);
  return launch<5376, 16, 16>(p, stream);
}

extern "C" int launch_norm_gemm(
    const __nv_bfloat16* x, const void* residual, const __nv_bfloat16* post,
    const __nv_bfloat16* pre, const __nv_bfloat16* scalar, void* carry, void* q,
    float* scales, void* weight, float* weight_scales, __nv_bfloat16* output,
    int rows, int n, int mode, int sms, cudaStream_t stream) {
  return launch_norm_gemm_impl(x, residual, post, pre, scalar, carry, q, scales,
                               weight, weight_scales, output, rows, n, mode,
                               sms, stream, false);
}

extern "C" int launch_norm_gemm_split(
    const __nv_bfloat16* x, const void* residual, const __nv_bfloat16* post,
    const __nv_bfloat16* pre, const __nv_bfloat16* scalar, void* carry, void* q,
    float* scales, void* weight, float* weight_scales, __nv_bfloat16* output,
    int rows, int n, int mode, int sms, cudaStream_t stream) {
  return launch_norm_gemm_impl(x, residual, post, pre, scalar, carry, q, scales,
                               weight, weight_scales, output, rows, n, mode,
                               sms, stream, true);
}
