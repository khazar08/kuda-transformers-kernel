// =============================================================================
// STAGE 5 -- 2D THREAD TILING.
//
// THE BOTTLENECK BEING REMOVED
//   Stage 4 reused a Bs value across 8 accumulators but still re-read As on
//   every one of those 8 FMAs -- ~1.125 smem loads per FMA. Better than 2, still
//   dominated by data movement.
//
// THE FIX
//   Give each thread a TM x TN = 8x8 SQUARE of outputs instead of an 8x1 strip.
//   Per k-step a thread loads 8 values of A and 8 of B into registers, then
//   forms the full OUTER PRODUCT: 16 loads feed 64 FMAs. The smem-load-per-FMA
//   ratio drops to 0.25, a further ~4.5x cut over stage 4.
//
//   This is the central insight of fast GEMM. Reuse in a 2D tile grows as
//   O(T^2) FMAs from O(T) loads, so squares beat strips and bigger squares beat
//   smaller ones -- until you run out of registers. That register budget is the
//   real reason GEMM kernels look the way they do.
//
// THE REGISTER BUDGET (the constraint that picks TM=TN=8)
//   64 accumulators + 8 regM + 8 regN = 80 floats minimum, ~100-110 registers
//   once addressing is included. Turing has 65536 registers per SM and caps a
//   thread at 255. At 256 threads/block: 256 * ~110 = ~28K registers, so 2
//   blocks fit per SM = 512 of the 1024 thread slots = 50% occupancy.
//
//   50% sounds bad and is not. Occupancy exists to hide latency with thread
//   parallelism; this kernel instead hides it with INSTRUCTION parallelism --
//   64 independent FMAs per k-step, no dependencies between them, so the
//   scheduler always has work. Chasing higher occupancy here would mean fewer
//   registers, hence a smaller tile, hence worse reuse. That is the classic
//   occupancy-vs-ILP tradeoff and it is worth stating in the README: low
//   occupancy is a symptom of a well-blocked GEMM, not a bug.
//
// A PREDICTION WORTH CHECKING IN THE BENCHMARKS
//   A 128x128 tile means M=N=256 produces a 2x2 = 4-block grid. The T4 has 40
//   SMs, so 36 of them sit COMPLETELY IDLE. Expect stage 5 to lose to stage 4
//   at the small end of the sweep and only pull ahead once the grid is large
//   enough to fill the machine. If the numbers show that, it is not a bug --
//   it is the tile-size-vs-parallelism tradeoff, and the sweep is what exposes
//   it. This is exactly the kind of finding the README should call out.
//
// EXPECTED RESULT
//   The largest jump in the ladder at 1024 and above, and the first stage that
//   should land in the same order of magnitude as cuBLAS.
// =============================================================================

#include "common.cuh"
#include "launchers.h"

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_blocktile_2d_kernel(const float *__restrict__ A,
                                          const float *__restrict__ B,
                                          float *__restrict__ C, int M, int N,
                                          int K, float alpha, float beta) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Threads per block = outputs per block / outputs per thread.
  // 128*128 / (8*8) = 256.
  const uint numThreads = (BM * BN) / (TM * TN);

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // --- OUTPUT mapping: this thread owns an 8x8 square ----------------------
  // BN/TN = 16 thread-columns, BM/TM = 16 thread-rows => 16*16 = 256 threads.
  const uint threadCol = threadIdx.x % (BN / TN);  // 0..15
  const uint threadRow = threadIdx.x / (BN / TN);  // 0..15

  // --- LOAD mapping ---------------------------------------------------------
  // The tiles now hold more elements than we have threads (1024 vs 256), so
  // each thread performs 4 strided loads per tile. The stride is chosen so that
  // consecutive threadIdx.x always land on consecutive addresses -- that is what
  // keeps these loads coalesced.
  const uint innerRowA = threadIdx.x / BK;   // 0..31
  const uint innerColA = threadIdx.x % BK;   // 0..7
  const uint strideA = numThreads / BK;      // 32 rows per pass
  const uint innerRowB = threadIdx.x / BN;   // 0..1
  const uint innerColB = threadIdx.x % BN;   // 0..127
  const uint strideB = numThreads / BN;      // 2 rows per pass

  // 64 accumulators, held in registers. Indexed only by unrolled counters.
  float threadResults[TM * TN] = {0.0f};
  // The two operand strips for the outer product.
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (int kt = 0; kt < K; kt += BK) {
    // --- cooperative strided load ------------------------------------------
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

    // --- compute: outer product per k-slice --------------------------------
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

  // --- write back the 8x8 square -------------------------------------------
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
// The launcher uses the <<<>>> launch syntax, which only nvcc can parse, so it
// is compiled out for the CPU-emulation correctness build (tools/cpu_emu_test).
// That build invokes the kernel above directly through cuda_cpu::launch().
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_blocktile_2d) {
  // These five constants are the entire tuning surface of the kernel. They are
  // the first thing to sweep once real T4 numbers exist -- 128/128/8/8/8 is the
  // standard starting point, not a measured optimum.
  const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
  dim3 block((BM * BN) / (TM * TN));  // 256 threads
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  sgemm_blocktile_2d_kernel<BM, BN, BK, TM, TN>
      <<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}
#endif  // CUDA_KERNEL_CPU_EMULATION
