#include "common.cuh"
#include "launchers.h"

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_blocktile_1d_kernel(const float *__restrict__ A,
                                          const float *__restrict__ B,
                                          float *__restrict__ C, int M, int N,
                                          int K, float alpha, float beta) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

 
  const uint threadCol = threadIdx.x % BN;  // 0..63
  const uint threadRow = threadIdx.x / BN;  // 0..7, each owns TM=8 rows

  const uint innerColA = threadIdx.x % BK;  // 0..7
  const uint innerRowA = threadIdx.x / BK;  // 0..63
  const uint innerColB = threadIdx.x % BN;  // 0..63
  const uint innerRowB = threadIdx.x / BN;  // 0..7

 
  // falls off a cliff -- check the ptxas output for "lmem" to catch this.
  float threadResults[TM] = {0.0f};

  for (int kt = 0; kt < K; kt += BK) {
    const uint aRow = cRow * BM + innerRowA;
    const uint aCol = kt + innerColA;
    As[innerRowA * BK + innerColA] =
        (aRow < (uint)M && aCol < (uint)K) ? A[aRow * K + aCol] : 0.0f;

    const uint bRow = kt + innerRowB;
    const uint bCol = cCol * BN + innerColB;
    Bs[innerRowB * BN + innerColB] =
        (bRow < (uint)K && bCol < (uint)N) ? B[bRow * N + bCol] : 0.0f;

    __syncthreads();

    for (uint d = 0; d < BK; ++d) {
      // Hoist the Bs value into a register ONCE, then reuse it TM times. This
      // single line is the entire optimization: without it the compiler would
      // re-issue an LDS for every one of the TM iterations below.
      const float Btmp = Bs[d * BN + threadCol];
      for (uint i = 0; i < TM; ++i) {
        threadResults[i] =
            fmaf(As[(threadRow * TM + i) * BK + d], Btmp, threadResults[i]);
      }
    }
    __syncthreads();
  }

  for (uint i = 0; i < TM; ++i) {
    const uint outRow = cRow * BM + threadRow * TM + i;
    const uint outCol = cCol * BN + threadCol;
    if (outRow < (uint)M && outCol < (uint)N) {
      const uint idx = outRow * N + outCol;
      C[idx] = (beta == 0.0f) ? alpha * threadResults[i]
                              : alpha * threadResults[i] + beta * C[idx];
    }
  }
}

#ifndef CUDA_KERNEL_CPU_EMULATION
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_blocktile_1d) {
  const int BM = 64, BN = 64, BK = 8, TM = 8;
  dim3 block((BM * BN) / TM);  // 512 threads
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  sgemm_blocktile_1d_kernel<BM, BN, BK, TM>
      <<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}
#endif  // CUDA_KERNEL_CPU_EMULATION
