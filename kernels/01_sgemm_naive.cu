// =============================================================================
// STAGE 1 -- NAIVE SGEMM. One thread per output element.
//
// WHAT IT DOES
//   Each thread owns exactly one element C[row][col] and serially walks the
//   full K-length dot product for it. This is the direct transcription of the
//   textbook triple loop, and it is the baseline every later stage is measured
//   against.
//
// THE BOTTLENECK (this is the whole point of the stage)
//   Look at the index mapping below: threadIdx.x is mapped to `row`. Threads in
//   a warp have consecutive threadIdx.x, so within one warp `row` varies 0..31
//   while `col` is IDENTICAL for all 32 lanes. Now look at the two loads:
//
//     A[row * K + k]  -- row varies per lane, so the 32 addresses are K floats
//                        apart. The coalescer cannot merge them: 32 separate
//                        32-byte sectors are fetched, and 28 bytes of each 32
//                        are thrown away. This is the killer.
//     B[k * N + col]  -- col is uniform across the warp, so all 32 lanes want
//                        the SAME address. That is a broadcast, which the
//                        hardware handles in a single transaction. Fine.
//     C[row * N + col] -- strided again, same problem as A.
//
//   So a warp burns ~33 memory transactions per k-iteration where a well-mapped
//   kernel needs ~5. There is also zero data reuse: every thread re-reads the
//   entire row of A and column of B from global memory, so total traffic is
//   O(M*N*K) floats instead of the O(M*K + K*N) the problem actually requires.
//
// EXPECTED RESULT
//   Low hundreds of GFLOP/s on a T4 -- a few percent of the 8.1 TFLOP/s FP32
//   peak. It should be overwhelmingly memory-bound, and the arithmetic units
//   should sit almost completely idle waiting on loads.
// =============================================================================

#include "common.cuh"
#include "launchers.h"

__global__ void sgemm_naive_kernel(const float *__restrict__ A,
                                   const float *__restrict__ B,
                                   float *__restrict__ C, int M, int N, int K,
                                   float alpha, float beta) {
  // THE DELIBERATE MISTAKE: threadIdx.x -> row.
  // Stage 2 swaps these two lines and nothing else. Keeping the rest of the
  // kernel byte-identical is what makes the A/B comparison honest -- any
  // measured difference is attributable purely to the access pattern.
  const uint row = blockIdx.x * blockDim.x + threadIdx.x;
  const uint col = blockIdx.y * blockDim.y + threadIdx.y;

  // Bounds check: the grid is sized with CEIL_DIV, so for matrix dimensions
  // that aren't multiples of 32 the trailing block has threads with no work.
  if (row < (uint)M && col < (uint)N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      // fmaf() is a fused multiply-add: one instruction, one rounding step.
      // nvcc would contract a * b + c into an FMA anyway at -O3, but writing
      // it explicitly documents the intent and pins the numerics so the error
      // we report against torch is stable across compiler versions.
      acc = fmaf(A[row * K + k], B[k * N + col], acc);
    }

    // beta == 0 is the overwhelmingly common case (a fresh output buffer). We
    // special-case it so we never READ C at all: it saves a full M*N-element
    // read, and more importantly it makes the kernel safe when C is allocated
    // with torch.empty() and contains garbage -- 0.0f * NaN is NaN, not 0.
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
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_naive) {
  // 32x32 = 1024 threads per block, the hardware maximum on Turing. Note this
  // means ONE block fully occupies an SM's 1024-thread budget on sm_75.
  dim3 block(32, 32);
  dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  sgemm_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}
#endif  // CUDA_KERNEL_CPU_EMULATION
