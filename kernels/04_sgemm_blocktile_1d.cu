// =============================================================================
// STAGE 4 -- 1D THREAD TILING (REGISTER BLOCKING).
//
// THE BOTTLENECK BEING REMOVED
//   Stage 3 executed two shared-memory loads for every single FMA. Shared memory
//   is fast but it is not free: it has finite bandwidth and every access is a
//   real instruction competing for issue slots. At a 2:1 load:FMA ratio the SM
//   spends most of its issue bandwidth on data movement, not arithmetic, so the
//   FP32 pipes idle no matter how fast the scratchpad is.
//
// THE FIX
//   Give each thread a COLUMN STRIP of TM=8 outputs instead of one, and keep the
//   8 accumulators in registers. Now look at the inner loop: one value pulled
//   from Bs is reused across all 8 accumulators. The ratio collapses from 2
//   smem loads per FMA to roughly 1.125 (8 As loads + 1 Bs load feeding 8 FMAs).
//   That is a ~9x reduction in shared-memory traffic per unit of arithmetic.
//
// WHY REGISTERS ARE THE ONLY PLACE THIS CAN LIVE
//   Registers are the sole storage the FP32 pipes read operands from directly:
//   zero latency, no bank arbitration, ~256 KB per SM on Turing. The whole
//   optimization ladder from here on is one idea -- push the working set down
//   the hierarchy (DRAM -> smem -> registers) so each level's bandwidth serves
//   proportionally more flops.
//
// THE TILE ARITHMETIC (worth internalizing, every later stage reuses it)
//   BM=64, BN=64, BK=8, TM=8. Each block computes a 64x64 tile of C.
//   Threads needed = (BM*BN)/TM = 4096/8 = 512.
//   Shared memory = (BM*BK + BK*BN)*4B = (512+512)*4 = 4 KB. Comfortable.
//   512 threads/block on sm_75 (1024 threads/SM cap) => 2 blocks/SM, which
//   finally gives the scheduler a second block to run during __syncthreads().
//
// A DELIBERATE TRADEOFF IN THE A-TILE LOAD
//   With BK=8, consecutive threads cover only 8 contiguous floats (32 bytes)
//   before jumping a full row of A. That is a 32-byte sector rather than a full
//   128-byte line -- imperfect coalescing. We accept it: the tile is loaded once
//   and reused 64 times, so load efficiency matters far less here than the
//   reuse it buys. Stage 6 recovers it with float4 loads.
//
// EXPECTED RESULT
//   Roughly 2-3x over stage 3. Arithmetic intensity rises sharply, the kernel
//   should move meaningfully rightward on the roofline plot, and the limiter
//   should start reading as compute/issue rather than memory.
// =============================================================================

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

  // --- OUTPUT mapping: which strip of C does this thread own? ---------------
  // threadCol is the fast axis so that the final stores are coalesced: 32
  // consecutive lanes write 32 consecutive columns of one row of C.
  const uint threadCol = threadIdx.x % BN;  // 0..63
  const uint threadRow = threadIdx.x / BN;  // 0..7, each owns TM=8 rows

  // --- LOAD mapping: which element of each smem tile does this thread fetch? -
  // This is a SEPARATE mapping from the output mapping and that is the single
  // most confusing thing about this kernel. A thread loads one element of As
  // and one of Bs, then computes on a strip that has nothing to do with what it
  // loaded. The __syncthreads() is what makes that legal.
  // With 512 threads and BM*BK = BK*BN = 512, it is exactly one element each.
  const uint innerColA = threadIdx.x % BK;  // 0..7
  const uint innerRowA = threadIdx.x / BK;  // 0..63
  const uint innerColB = threadIdx.x % BN;  // 0..63
  const uint innerRowB = threadIdx.x / BN;  // 0..7

  // Per-thread accumulators. This array is indexed only by compile-time-unrolled
  // loop counters, which is what lets nvcc keep it in registers rather than
  // spilling it to local memory. If you ever index a per-thread array with a
  // runtime value, it silently becomes a local-memory array and performance
  // falls off a cliff -- check the ptxas output for "lmem" to catch this.
  float threadResults[TM] = {0.0f};

  for (int kt = 0; kt < K; kt += BK) {
    // --- cooperative load, zero-padded out of range -----------------------
    const uint aRow = cRow * BM + innerRowA;
    const uint aCol = kt + innerColA;
    As[innerRowA * BK + innerColA] =
        (aRow < (uint)M && aCol < (uint)K) ? A[aRow * K + aCol] : 0.0f;

    const uint bRow = kt + innerRowB;
    const uint bCol = cCol * BN + innerColB;
    Bs[innerRowB * BN + innerColB] =
        (bRow < (uint)K && bCol < (uint)N) ? B[bRow * N + bCol] : 0.0f;

    __syncthreads();

    // --- compute: the reuse happens here ----------------------------------
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

  // --- write back the strip ------------------------------------------------
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
// The launcher uses the <<<>>> launch syntax, which only nvcc can parse, so it
// is compiled out for the CPU-emulation correctness build (tools/cpu_emu_test).
// That build invokes the kernel above directly through cuda_cpu::launch().
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_blocktile_1d) {
  const int BM = 64, BN = 64, BK = 8, TM = 8;
  dim3 block((BM * BN) / TM);  // 512 threads
  dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  sgemm_blocktile_1d_kernel<BM, BN, BK, TM>
      <<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}
#endif  // CUDA_KERNEL_CPU_EMULATION
