#include "common.cuh"
#include "launchers.h"

__global__ void sgemm_coalesced_kernel(const float *__restrict__ A,
                                       const float *__restrict__ B,
                                       float *__restrict__ C, int M, int N,
                                       int K, float alpha, float beta) {
  // THE FIX: threadIdx.x -> col, so consecutive lanes read consecutive floats.
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;

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
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_coalesced) {
  dim3 block(32, 32);
  // Grid axes swap too, so that blockIdx.x tracks the N (column) dimension and
  // the block-to-tile mapping stays consistent with the thread mapping above.
  dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
  sgemm_coalesced_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha,
                                                     beta);
}
#endif  // CUDA_KERNEL_CPU_EMULATION
