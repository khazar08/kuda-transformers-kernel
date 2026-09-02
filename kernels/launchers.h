#pragma once

#ifndef CUDA_KERNEL_CPU_EMULATION
#include <cuda_runtime.h>
#endif

#define SGEMM_LAUNCHER_SIGNATURE(name)                                         \
  void name(const float *A, const float *B, float *C, int M, int N, int K,     \
            float alpha, float beta, cudaStream_t stream)

SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_cublas);        // ceiling / reference
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_naive);         // stage 1
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_coalesced);     // stage 2
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_smem);          // stage 3
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_blocktile_1d);  // stage 4
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_blocktile_2d);  // stage 5
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_vectorized);    // stage 6

void launch_softmax_naive(const float *X, float *Y, float *row_max,
                          float *row_sum, int rows, int cols,
                          cudaStream_t stream);

void launch_softmax_fused(const float *X, float *Y, int rows, int cols,
                          cudaStream_t stream);

void launch_layernorm_fused(const float *X, const float *gamma,
                            const float *beta, float *Y, int rows, int cols,
                            float eps, cudaStream_t stream);
