#include "common.cuh"
#include "launchers.h"

__global__ void sgemm_naive_kernel(const float *__restrict__ A,
                                   const float *__restrict__ B,
                                   float *__restrict__ C, int M, int N, int K,
                                   float alpha, float beta) {

  const uint row = blockIdx.x * blockDim.x + threadIdx.x;
  const uint col = blockIdx.y * blockDim.y + threadIdx.y;

  // Bounds check: the grid is sized with CEIL_DIV, so for matrix dimensions
  // that aren't multiples of 32 the trailing block has threads with no work.
  if (row < (uint)M && col < (uint)N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      
      acc = fmaf(A[row * K + k], B[k * N + col], acc);
    }

   
    if (beta == 0.0f) {
      C[row * N + col] = alpha * acc;
    } else {
      C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
  }
}

#ifndef CUDA_KERNEL_CPU_EMULATION

SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_naive) {
  dim3 block(32, 32);
  dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  sgemm_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}
#endif  // CUDA_KERNEL_CPU_EMULATION
