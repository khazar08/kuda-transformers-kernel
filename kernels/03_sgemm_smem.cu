// =============================================================================
// STAGE 3 -- SHARED MEMORY BLOCK TILING.
//
// THE BOTTLENECK BEING REMOVED
//   Stage 2 made each memory transaction efficient but did nothing about the
//   VOLUME. Every thread still streamed a whole row of A and column of B from
//   DRAM, so we moved O(M*N*K) floats to perform O(M*N*K) flops -- arithmetic
//   intensity ~0.25 flops/byte against a T4 ridge point near 25. Hopeless.
//
// THE FIX
//   Cooperative staging through shared memory. The block walks the K dimension
//   in chunks of 32. For each chunk, the 1024 threads collectively load one
//   32x32 tile of A and one of B into shared memory (one element each), sync,
//   then every thread reads what it needs from the on-die scratchpad. A value
//   loaded from DRAM once is now consumed 32 times instead of once, cutting
//   global traffic by ~32x.
//
// WHY SHARED MEMORY IS THE RIGHT TOOL
//   It is SRAM physically on the SM: ~20-30 cycle latency versus ~400-600 for
//   DRAM, and it is explicitly programmer-managed, so unlike L1 we can GUARANTEE
//   the reuse rather than hope the cache holds the line.
//
// THE BANK-CONFLICT CHECK (always do this when you write to smem)
//   Shared memory is 32 banks of 4 bytes. A warp here is 32 consecutive
//   threadIdx.x, which means one tRow and all 32 tCol values.
//     As[tRow*32 + d]  -- identical address for all 32 lanes => BROADCAST, free.
//     Bs[d*32 + tCol]  -- tCol spans 0..31 => 32 distinct banks => conflict-free.
//   Both access patterns are clean; no padding needed at this tile size.
//
// WHERE THIS STAGE STILL LOSES (motivates stage 4)
//   Each thread still computes exactly ONE output, so the inner loop is two
//   shared-memory loads per single FMA. The FP32 pipes are now starved by the
//   smem bandwidth instead of DRAM. Worse, on sm_75 a 32x32 block is 1024
//   threads = the ENTIRE per-SM thread budget, so only one block fits per SM
//   and there is nothing to hide the __syncthreads() stalls with.
//
// EXPECTED RESULT
//   A solid multiple over stage 2, DRAM traffic down roughly 32x, and the
//   bottleneck visibly migrating from DRAM to shared-memory throughput.
// =============================================================================

#include "common.cuh"
#include "launchers.h"

template <const int BLOCKSIZE>
__global__ void sgemm_smem_kernel(const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  float *__restrict__ C, int M, int N, int K,
                                  float alpha, float beta) {
  // Which BLOCKSIZE x BLOCKSIZE tile of C this block is responsible for.
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
    // --- cooperative load -------------------------------------------------
    // Out-of-range elements are zero-filled rather than skipped. Zero is the
    // identity for the accumulation that follows, so padding with it makes the
    // kernel correct for non-multiple-of-32 dimensions without needing a
    // separate cleanup path or a shortened inner loop.
    const uint aCol = kt + tCol;
    const uint bRow = kt + tRow;
    As[tRow * BLOCKSIZE + tCol] =
        (globalRow < (uint)M && aCol < (uint)K) ? A[globalRow * K + aCol] : 0.0f;
    Bs[tRow * BLOCKSIZE + tCol] =
        (bRow < (uint)K && globalCol < (uint)N) ? B[bRow * N + globalCol] : 0.0f;

    // Barrier #1: nobody may read the tile until every thread has finished
    // writing its element. Without this you get a race on partially-filled smem.
    __syncthreads();

    // --- compute on the staged tile ---------------------------------------
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
// The launcher uses the <<<>>> launch syntax, which only nvcc can parse, so it
// is compiled out for the CPU-emulation correctness build (tools/cpu_emu_test).
// That build invokes the kernel above directly through cuda_cpu::launch().
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_smem) {
  const int BLOCKSIZE = 32;
  dim3 block(BLOCKSIZE * BLOCKSIZE);
  dim3 grid(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE));
  sgemm_smem_kernel<BLOCKSIZE>
      <<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}
#endif  // CUDA_KERNEL_CPU_EMULATION
