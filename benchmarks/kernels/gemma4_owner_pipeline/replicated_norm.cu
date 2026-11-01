// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cstdint>
#include <limits>
namespace local_impl {
#include "local_norm.cuh"
}
namespace one_impl {
#include "replicated_one.cuh"
}
namespace two_impl {
#include "replicated_two.cuh"
}
extern "C" int launch_local(const __nv_bfloat16* x, const float* res,
                            const __nv_bfloat16* post, const __nv_bfloat16* pre,
                            float* carry, void* q, float* scales, int rows,
                            int mode, const __nv_bfloat16* scalar,
                            cudaStream_t s) {
  const int pitch = (rows + 3) / 4 * 4;
  if (mode == 2)
    local_impl::chain<8, true, true, true, true, true, true>
        <<<rows, 672, 0, s>>>(x, res, post, pre, carry, q, scales, pitch, 1e-6f,
                              scalar);
  else
    local_impl::chain<8, true, false, true, false, true, true>
        <<<rows, 672, 0, s>>>(x, res, post, pre, carry, q, scales, pitch, 1e-6f,
                              scalar);
  return int(cudaGetLastError());
}
extern "C" int launch_replicated(__nv_bfloat16* scatter, __nv_bfloat16* box,
                                 __nv_bfloat16* sums, uint64_t sum_mc, int rows,
                                 int parity, int rank, int* error,
                                 const float* res, const __nv_bfloat16* post,
                                 const __nv_bfloat16* pre, float* carry,
                                 void* q, float* scales, int mode,
                                 const __nv_bfloat16* scalar, cudaStream_t s) {
  const int pitch = (rows + 3) / 4 * 4;
  if (rows <= 16) {
    if (mode == 2)
      one_impl::chain<8, true, true, true, true, true, true>
          <<<rows, 672, 0, s>>>(box, rows, parity, error, res, post, pre, carry,
                                q, scales, pitch, 1e-6f, scalar);
    else
      one_impl::chain<8, true, false, true, false, true, true>
          <<<rows, 672, 0, s>>>(box, rows, parity, error, res, post, pre, carry,
                                q, scales, pitch, 1e-6f, scalar);
  } else {
    if (mode == 2)
      two_impl::chain<8, true, true, true, true, true, true>
          <<<rows, 672, 0, s>>>(scatter, sums, sum_mc, rows, parity, rank,
                                error, res, post, pre, carry, q, scales, pitch,
                                1e-6f, scalar);
    else
      two_impl::chain<8, true, false, true, false, true, true>
          <<<rows, 672, 0, s>>>(scatter, sums, sum_mc, rows, parity, rank,
                                error, res, post, pre, carry, q, scales, pitch,
                                1e-6f, scalar);
  }
  return int(cudaGetLastError());
}
