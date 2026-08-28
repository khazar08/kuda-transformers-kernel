#include "common.cuh"
#include "launchers.h"

template <const int BLOCKSIZE>
__global__ void sgemm_smem_kernel(const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  float *__restrict__ C, int M, int N, int K,
                                  float alpha, float beta) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // The block is launched as a 1D array of BLOCKSIZE*BLOCKSIZE threads and we
  // derive 2D coordinates by hand. Doing it manually (rather than a dim3 block)
  // keeps the warp->lane mapping explicit and obvious: threadIdx.x is the fast
  // axis, so tCol is the fast axis, so global loads stay coalesced.
  const uint tRow = threadIdx.x / BLOCKSIZE;
  const uint tCol = threadIdx.x % BLOCKSIZE;

  // The staging buffers. 2 * 32 * 32 * 4B = 8 KB of the T4's 64 KB per SM.
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  const uint globalRow = cRow * BLOCKSIZE + tRow;
  const uint globalCol = cCol * BLOCKSIZE + tCol;

  float acc = 0.0f;

  // March along K one tile at a time.
  for (int kt = 0; kt < K; kt += BLOCKSIZE) {
    const uint aCol = kt + tCol;
    const uint bRow = kt + tRow;
    As[tRow * BLOCKSIZE + tCol] =
        (globalRow < (uint)M && aCol < (uint)K) ? A[globalRow * K + aCol] : 0.0f;
    Bs[tRow * BLOCKSIZE + tCol] =
        (bRow < (uint)K && globalCol < (uint)N) ? B[bRow * N + globalCol] : 0.0f;

    __syncthreads();

    for (int d = 0; d < BLOCKSIZE; ++d) {
      acc = fmaf(As[tRow * BLOCKSIZE + d], Bs[d * BLOCKSIZE + tCol], acc);
    }

    // Barrier #2: nobody may overwrite the tile on the next iteration until
    // every thread has finished reading it. Forgetting this second barrier is
    // the classic tiled-GEMM bug -- it produces results that are *mostly* right,
    // which is far worse than results that are obviously wrong.
    __syncthreads();
  }

  if (globalRow < (uint)M && globalCol < (uint)N) {
    const uint idx = globalRow * N + globalCol;
    C[idx] = (beta == 0.0f) ? alpha * acc : alpha * acc + beta * C[idx];
  }
}

#ifndef CUDA_KERNEL_CPU_EMULATION
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_smem) {
  const int BLOCKSIZE = 32;
  dim3 block(BLOCKSIZE * BLOCKSIZE);
  dim3 grid(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE));
  sgemm_smem_kernel<BLOCKSIZE>
      <<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}
#endif  // CUDA_KERNEL_CPU_EMULATION
