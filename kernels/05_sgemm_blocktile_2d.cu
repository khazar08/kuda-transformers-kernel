#include "common.cuh"
#include "launchers.h"

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_blocktile_2d_kernel(const float *__restrict__ A,
                                          const float *__restrict__ B,
                                          float *__restrict__ C, int M, int N,
                                          int K, float alpha, float beta) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint numThreads = (BM * BN) / (TM * TN);

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const uint threadCol = threadIdx.x % (BN / TN);  // 0..15
  const uint threadRow = threadIdx.x / (BN / TN);  // 0..15

  const uint innerRowA = threadIdx.x / BK;   // 0..31
  const uint innerColA = threadIdx.x % BK;   // 0..7
  const uint strideA = numThreads / BK;      // 32 rows per pass
  const uint innerRowB = threadIdx.x / BN;   // 0..1
  const uint innerColB = threadIdx.x % BN;   // 0..127
  const uint strideB = numThreads / BN;      // 2 rows per pass

  float threadResults[TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (int kt = 0; kt < K; kt += BK) {
    for (uint off = 0; off < BM; off += strideA) {
      const uint aRow = cRow * BM + innerRowA + off;
      const uint aCol = kt + innerColA;
      As[(innerRowA + off) * BK + innerColA] =
          (aRow < (uint)M && aCol < (uint)K) ? A[aRow * K + aCol] : 0.0f;
    }
    for (uint off = 0; off < BK; off += strideB) {
      const uint bRow = kt + innerRowB + off;
      const uint bCol = cCol * BN + innerColB;
      Bs[(innerRowB + off) * BN + innerColB] =
          (bRow < (uint)K && bCol < (uint)N) ? B[bRow * N + bCol] : 0.0f;
    }
    __syncthreads();

    for (uint d = 0; d < BK; ++d) {
      // Pull the two strips into registers ONCE...
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + d];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[d * BN + threadCol * TN + i];
      }
      // ...then spend them on TM*TN = 64 FMAs with no further memory traffic.
      // 16 loads : 64 FMAs is the whole reason this stage is fast.
      for (uint m = 0; m < TM; ++m) {
        for (uint n = 0; n < TN; ++n) {
          threadResults[m * TN + n] =
              fmaf(regM[m], regN[n], threadResults[m * TN + n]);
        }
      }
    }
    __syncthreads();
  }

  // write back the 8x8 square
  for (uint m = 0; m < TM; ++m) {
    for (uint n = 0; n < TN; ++n) {
      const uint outRow = cRow * BM + threadRow * TM + m;
      const uint outCol = cCol * BN + threadCol * TN + n;
      if (outRow < (uint)M && outCol < (uint)N) {
        const uint idx = outRow * N + outCol;
        const float v = threadResults[m * TN + n];
        C[idx] = (beta == 0.0f) ? alpha * v : alpha * v + beta * C[idx];
      }
    }
  }
}

#ifndef CUDA_KERNEL_CPU_EMULATION
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_blocktile_2d) {
  const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
  dim3 block((BM * BN) / (TM * TN));  // 256 threads
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  sgemm_blocktile_2d_kernel<BM, BN, BK, TM, TN>
      <<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}
#endif  // CUDA_KERNEL_CPU_EMULATION
