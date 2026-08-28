#include "common.cuh"
#include "launchers.h"

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_vectorized_kernel(const float *__restrict__ A,
                                        const float *__restrict__ B,
                                        float *__restrict__ C, int M, int N,
                                        int K, float alpha, float beta) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  constexpr uint AS_STRIDE = BM + 4;
  __shared__ __align__(16) float As[BK * AS_STRIDE];  // transposed, [BK][BM+4]
  __shared__ __align__(16) float Bs[BK * BN];  // normal layout, [BK][BN]

  const uint threadCol = threadIdx.x % (BN / TN);  // 0..15
  const uint threadRow = threadIdx.x / (BN / TN);  // 0..15

  const uint colChunk0 = threadCol * 4;             // 0, 4, ... 60
  const uint colChunk1 = threadCol * 4 + BN / 2;    // 64, 68, ... 124


  const uint innerRowA = threadIdx.x / (BK / 4);  // 0..127
  const uint innerColA = threadIdx.x % (BK / 4);  // 0..1  (float4 index)
  const uint innerRowB = threadIdx.x / (BN / 4);  // 0..7
  const uint innerColB = threadIdx.x % (BN / 4);  // 0..31 (float4 index)

  float threadResults[TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (int kt = 0; kt < K; kt += BK) {
    {
      const float4 tmp = AS_CONST_FLOAT4(
          A[(cRow * BM + innerRowA) * K + kt + innerColA * 4]);
      // Four scalar stores, striding by BM, because we are writing a column of
      // the transposed tile. This is the scatter the transpose costs us.
      As[(innerColA * 4 + 0) * AS_STRIDE + innerRowA] = tmp.x;
      As[(innerColA * 4 + 1) * AS_STRIDE + innerRowA] = tmp.y;
      As[(innerColA * 4 + 2) * AS_STRIDE + innerRowA] = tmp.z;
      As[(innerColA * 4 + 3) * AS_STRIDE + innerRowA] = tmp.w;
    }


    AS_FLOAT4(Bs[innerRowB * BN + innerColB * 4]) = AS_CONST_FLOAT4(
        B[(kt + innerRowB) * N + cCol * BN + innerColB * 4]);

    __syncthreads();

    #pragma unroll
    for (uint d = 0; d < BK; ++d) {
      #pragma unroll
      for (uint i = 0; i < TM; i += 4) {
        AS_FLOAT4(regM[i]) =
            AS_CONST_FLOAT4(As[d * AS_STRIDE + threadRow * TM + i]);
      }
      // Two vector loads matching the split-column mapping above.
      AS_FLOAT4(regN[0]) = AS_CONST_FLOAT4(Bs[d * BN + colChunk0]);
      AS_FLOAT4(regN[4]) = AS_CONST_FLOAT4(Bs[d * BN + colChunk1]);
      #pragma unroll
      for (uint m = 0; m < TM; ++m) {
        #pragma unroll
        for (uint n = 0; n < TN; ++n) {
          threadResults[m * TN + n] =
              fmaf(regM[m], regN[n], threadResults[m * TN + n]);
        }
      }
    }
    __syncthreads();
  }

  // vectorized write-back, fully sector-utilizing 
  #pragma unroll
  for (uint m = 0; m < TM; ++m) {
    const uint outRow = cRow * BM + threadRow * TM + m;
    const uint rowBase = outRow * N + cCol * BN;

    #pragma unroll
    for (uint chunk = 0; chunk < 2; ++chunk) {
      const uint idx = rowBase + (chunk == 0 ? colChunk0 : colChunk1);
      const uint r = m * TN + chunk * 4;  // 4 accumulators feeding this store
      float4 out;
      if (beta == 0.0f) {
        out.x = alpha * threadResults[r + 0];
        out.y = alpha * threadResults[r + 1];
        out.z = alpha * threadResults[r + 2];
        out.w = alpha * threadResults[r + 3];
      } else {
        out = AS_CONST_FLOAT4(C[idx]);
        out.x = alpha * threadResults[r + 0] + beta * out.x;
        out.y = alpha * threadResults[r + 1] + beta * out.y;
        out.z = alpha * threadResults[r + 2] + beta * out.z;
        out.w = alpha * threadResults[r + 3] + beta * out.w;
      }
      AS_FLOAT4(C[idx]) = out;
    }
  }
}

#ifndef CUDA_KERNEL_CPU_EMULATION
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_vectorized) {
  const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

  if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
    launch_sgemm_blocktile_2d(A, B, C, M, N, K, alpha, beta, stream);
    return;
  }

  dim3 block((BM * BN) / (TM * TN));  // 256 threads
  dim3 grid(N / BN, M / BM);
  sgemm_vectorized_kernel<BM, BN, BK, TM, TN>
      <<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}
#endif  // CUDA_KERNEL_CPU_EMULATION
