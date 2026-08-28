// =============================================================================
// STAGE 6 (STRETCH) -- VECTORIZED float4 LOADS + TRANSPOSED As.
//
// THE BOTTLENECK BEING REMOVED
//   By stage 5 the FMAs are efficient but the kernel issues a LOT of separate
//   load instructions, and every instruction consumes an issue slot the SM
//   could have spent on arithmetic. On a memory system that transacts in
//   128-byte lines, four consecutive 32-bit loads are strictly worse than one
//   128-bit load that fetches the same bytes: 4 instructions, 4 address
//   computations, 4 entries in the load-store queue, for identical data.
//
// TWO CHANGES
//   (1) float4 (128-bit) loads. One LDG.E.128 replaces four LDG.E.32, quartering
//       memory instruction count on the global->shared path and on the
//       shared->register path.
//   (2) As is stored TRANSPOSED in shared memory, as [BK][BM] instead of
//       [BM][BK]. In stage 5 the regM strip read As[(threadRow*TM+i)*BK + d] --
//       stride BK between consecutive i, so the 8 values a thread wants are
//       scattered and cannot be vectorized. Transposing makes them contiguous
//       in m, so the whole strip becomes two float4 reads. The transpose costs
//       a scatter at load time, but that is paid once per tile and repaid on
//       every one of the BK iterations that follow.
//
// THE COST WE ACCEPT (be honest about this in the README)
//   The transposed store into As has a 2-way shared-memory bank conflict: bank
//   is (row*BM + col) % 32 and BM=128 is a multiple of 32, so the bank depends
//   only on innerRowA, and within a warp each innerRowA value appears twice.
//   Padding As to BM+4 would remove it at the cost of shared memory and messier
//   float4 alignment. Whether that trade pays is an empirical question -- flag
//   it as future work rather than guessing.
//
// ALIGNMENT REQUIREMENT (why this kernel has a fallback)
//   128-bit accesses must be 16-byte aligned, which requires the row strides to
//   be multiples of 4 floats. Rather than bolt on a slow scalar cleanup path,
//   the launcher checks divisibility and cleanly falls back to stage 5 when the
//   shapes do not cooperate. A kernel with an honest precondition and a correct
//   fallback is better engineering than one that is subtly wrong on odd shapes.
//
// EXPECTED RESULT
//   A more modest gain than stages 3-5 -- maybe 10-30%. By this point the
//   kernel is close to the issue-rate limit, so this is a refinement, not a
//   breakthrough. Watch for reduced instruction count and higher achieved
//   memory throughput at the same FLOP rate.
// =============================================================================

#include "common.cuh"
#include "launchers.h"

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_vectorized_kernel(const float *__restrict__ A,
                                        const float *__restrict__ B,
                                        float *__restrict__ C, int M, int N,
                                        int K, float alpha, float beta) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // __align__(16) is REQUIRED. A plain __shared__ float array is only
  // guaranteed 4-byte aligned, and a 128-bit access to a misaligned address is
  // undefined behaviour -- in practice a misaligned-address fault at runtime.
  // PADDED STRIDE (added after Nsight Compute measured the conflict).
  //
  // The transposed As tile is indexed As[d * stride + m]. With stride = BM = 128,
  // a multiple of 32, the bank reduces to m % 32 -- and within a warp only 16
  // distinct innerRowA values occur, each twice, so every shared store collides
  // 2-way. ncu measured 2.4-way across 2.6M shared-store requests, 33% of all
  // store wavefronts, est. 25.5% speedup.
  //
  // Padding the stride to BM + 4 = 132 makes the bank (d*4 + m) % 32, which
  // spreads the two halves of the warp 16 banks apart and makes the access
  // conflict-free. 132 floats = 528 bytes is still a multiple of 16, so the
  // float4 alignment the vectorized reads depend on is preserved. Cost: 128
  // extra bytes of shared memory per block, out of 64 KB.
  constexpr uint AS_STRIDE = BM + 4;
  __shared__ __align__(16) float As[BK * AS_STRIDE];  // transposed, [BK][BM+4]
  __shared__ __align__(16) float Bs[BK * BN];  // normal layout, [BK][BN]

  const uint threadCol = threadIdx.x % (BN / TN);  // 0..15
  const uint threadRow = threadIdx.x / (BN / TN);  // 0..15

  // SPLIT-COLUMN OUTPUT MAPPING (added after profiling with Nsight Compute).
  //
  // The obvious mapping gives each thread TN=8 CONSECUTIVE columns, written as
  // two float4 stores. ncu flagged it: "only 16.0 of the 32 bytes transmitted
  // per sector are utilized", est. speedup 44.9%. The reason is that within one
  // store instruction thread t writes bytes [t*32, t*32+16) -- the other half of
  // every 32-byte sector belongs to the OTHER store instruction, so each store
  // wastes half of every sector it touches.
  //
  // Fix: hand each thread two 4-wide chunks BN/2 apart instead of one 8-wide
  // run. Now a single store instruction has consecutive threads writing
  // consecutive float4s -- 16 threads x 16 B = 256 contiguous bytes, fully
  // utilizing every sector. Same 64 outputs per thread, same registers, same
  // FLOPs; only the mapping changed.
  const uint colChunk0 = threadCol * 4;             // 0, 4, ... 60
  const uint colChunk1 = threadCol * 4 + BN / 2;    // 64, 68, ... 124

  // --- LOAD mapping, in float4 units ---------------------------------------
  // A tile: BM*BK = 1024 floats = 256 float4, and we have 256 threads, so each
  // thread issues exactly one 128-bit load per tile. Same for B.
  const uint innerRowA = threadIdx.x / (BK / 4);  // 0..127
  const uint innerColA = threadIdx.x % (BK / 4);  // 0..1  (float4 index)
  const uint innerRowB = threadIdx.x / (BN / 4);  // 0..7
  const uint innerColB = threadIdx.x % (BN / 4);  // 0..31 (float4 index)

  float threadResults[TM * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[TN] = {0.0f};

  for (int kt = 0; kt < K; kt += BK) {
    // --- A: one float4 load, scattered into As transposed -------------------
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

    // --- B: one float4 load, one float4 store, no transpose needed ----------
    // This path is perfectly coalesced: a warp's 32 lanes share innerRowB and
    // span innerColB 0..31, i.e. 32 * 16B = 512 contiguous bytes.
    AS_FLOAT4(Bs[innerRowB * BN + innerColB * 4]) = AS_CONST_FLOAT4(
        B[(kt + innerRowB) * N + cCol * BN + innerColB * 4]);

    __syncthreads();

    // --- compute -----------------------------------------------------------
    #pragma unroll
    for (uint d = 0; d < BK; ++d) {
      // Because As is transposed, the TM values this thread needs are now
      // CONTIGUOUS, so TM/4 = 2 vector reads replace TM = 8 scalar ones.
      #pragma unroll
      for (uint i = 0; i < TM; i += 4) {
        AS_FLOAT4(regM[i]) =
            AS_CONST_FLOAT4(As[d * AS_STRIDE + threadRow * TM + i]);
      }
      // Two vector loads matching the split-column mapping above.
      AS_FLOAT4(regN[0]) = AS_CONST_FLOAT4(Bs[d * BN + colChunk0]);
      AS_FLOAT4(regN[4]) = AS_CONST_FLOAT4(Bs[d * BN + colChunk1]);
      // The #pragma unroll above matters for more than speed: taking the
      // address of regM/regN for the float4 cast is only register-safe when the
      // indices are compile-time constants. If these loops were not fully
      // unrolled, nvcc would spill regM/regN to LOCAL memory and performance
      // would collapse. Verify with --ptxas-options=-v: any nonzero "lmem"
      // figure for this kernel means the unrolling did not happen.
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

  // --- vectorized write-back, fully sector-utilizing ------------------------
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
// The launcher uses the <<<>>> launch syntax, which only nvcc can parse, so it
// is compiled out for the CPU-emulation correctness build (tools/cpu_emu_test).
// That build invokes the kernel above directly through cuda_cpu::launch().
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_vectorized) {
  const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

  // PRECONDITION CHECK. The vectorized kernel requires exact tile divisibility
  // so that every 128-bit access is in bounds and 16-byte aligned. Rather than
  // produce wrong answers on shapes it cannot handle, degrade to stage 5, which
  // is fully general. The benchmark harness reports which path ran.
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
