// =============================================================================
// STAGE 2 -- GLOBAL MEMORY COALESCING.
//
// WHAT CHANGED FROM STAGE 1
//   Two lines. threadIdx.x now maps to `col` instead of `row`. That is the
//   entire diff. Same FLOPs, same instruction count, same occupancy, same
//   everything -- only the mapping of lanes to addresses differs.
//
// WHY IT WINS
//   With threadIdx.x -> col, a warp's 32 lanes now have consecutive `col` and
//   a single shared `row`:
//     A[row * K + k]  -- uniform across the warp => broadcast, 1 transaction
//                        (was 32 strided sectors in stage 1).
//     B[k * N + col]  -- 32 consecutive floats = 128 contiguous bytes => the
//                        coalescer merges them into 4 sectors, one trip.
//     C[row * N + col] -- likewise 4 sectors instead of 32.
//
//   Per k-iteration a warp goes from roughly 33 transactions to roughly 5. The
//   DRAM is being asked for the same useful bytes but is no longer forced to
//   discard 7/8 of every sector it fetches.
//
// WHAT THIS STAGE DOES *NOT* FIX
//   Data reuse is still zero -- every thread still streams the whole of its row
//   of A and column of B from global memory, so we are still moving O(M*N*K)
//   data for O(M*N*K) flops. Arithmetic intensity is ~0.25 flops/byte, which on
//   the roofline plot sits far left of the T4 ridge point (~25 flops/byte).
//   Coalescing makes the traffic efficient; stage 3 makes it SMALLER.
//
// EXPECTED RESULT
//   A large multiple over stage 1 -- this is typically the single biggest
//   speedup in the whole ladder for the least code. Achieved DRAM bandwidth
//   should jump toward a meaningful fraction of the T4's 320 GB/s, while
//   "sectors per request" in the profiler drops from ~32 toward ~4.
// =============================================================================

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
// The launcher uses the <<<>>> launch syntax, which only nvcc can parse, so it
// is compiled out for the CPU-emulation correctness build (tools/cpu_emu_test).
// That build invokes the kernel above directly through cuda_cpu::launch().
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_coalesced) {
  dim3 block(32, 32);
  // Grid axes swap too, so that blockIdx.x tracks the N (column) dimension and
  // the block-to-tile mapping stays consistent with the thread mapping above.
  dim3 grid(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
  sgemm_coalesced_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha,
                                                     beta);
}
#endif  // CUDA_KERNEL_CPU_EMULATION
