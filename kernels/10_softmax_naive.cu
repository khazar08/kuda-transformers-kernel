// =============================================================================
// PART 2a BASELINE -- 3-PASS SOFTMAX.
//
// This is the version fusion is meant to beat, and it exists purely so the
// comparison in the README is against something real rather than a strawman.
// Each pass is a correct, coalesced, perfectly reasonable kernel. The problem
// is not any individual pass -- it is that there are three of them.
//
// THE STRUCTURAL COST
//   A kernel launch is a global barrier. Registers and shared memory do not
//   survive it. So the row max computed in pass 1 has to be written to DRAM and
//   read back in pass 2, and likewise the row sum for pass 3. The intermediate
//   Y tensor is written in full and read back in full. None of that data
//   movement does any arithmetic work; it exists solely to carry state across
//   launch boundaries.
//
//   Traffic: ~3N reads + 2N writes. Launches: 3.
//   Fused equivalent: ~2N reads + 1N write. Launches: 1.
// =============================================================================

#include "common.cuh"
#include "launchers.h"
#include "reduce.cuh"

// PASS 1: row maxima -> DRAM.
template <int BLOCK>
__global__ void softmax_pass1_max(const float *__restrict__ X,
                                  float *__restrict__ row_max, int rows,
                                  int cols) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const float *xr = X + (size_t)row * cols;
  float m = -FLT_MAX;
  for (int i = threadIdx.x; i < cols; i += BLOCK) m = fmaxf(m, xr[i]);
  m = blockReduceMax<BLOCK>(m);
  if (threadIdx.x == 0) row_max[row] = m;
}

// PASS 2: read the max back, exponentiate into Y, reduce the sum -> DRAM.
template <int BLOCK>
__global__ void softmax_pass2_exp_sum(const float *__restrict__ X,
                                      float *__restrict__ Y,
                                      const float *__restrict__ row_max,
                                      float *__restrict__ row_sum, int rows,
                                      int cols) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const float *xr = X + (size_t)row * cols;
  float *yr = Y + (size_t)row * cols;
  const float m = row_max[row];  // round-tripped through DRAM from pass 1
  float s = 0.0f;
  for (int i = threadIdx.x; i < cols; i += BLOCK) {
    const float e = expf(xr[i] - m);
    yr[i] = e;  // full-size intermediate write that fusion removes entirely
    s += e;
  }
  s = blockReduceSum<BLOCK>(s);
  if (threadIdx.x == 0) row_sum[row] = s;
}

// PASS 3: read Y back and scale it. Pure data movement, zero useful arithmetic.
template <int BLOCK>
__global__ void softmax_pass3_normalize(float *__restrict__ Y,
                                        const float *__restrict__ row_sum,
                                        int rows, int cols) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  float *yr = Y + (size_t)row * cols;
  const float inv = 1.0f / row_sum[row];
  for (int i = threadIdx.x; i < cols; i += BLOCK) yr[i] *= inv;
}

#ifndef CUDA_KERNEL_CPU_EMULATION
void launch_softmax_naive(const float *X, float *Y, float *row_max,
                          float *row_sum, int rows, int cols,
                          cudaStream_t stream) {
  constexpr int BLOCK = 256;
  softmax_pass1_max<BLOCK><<<rows, BLOCK, 0, stream>>>(X, row_max, rows, cols);
  softmax_pass2_exp_sum<BLOCK>
      <<<rows, BLOCK, 0, stream>>>(X, Y, row_max, row_sum, rows, cols);
  softmax_pass3_normalize<BLOCK>
      <<<rows, BLOCK, 0, stream>>>(Y, row_sum, rows, cols);
}
#endif  // CUDA_KERNEL_CPU_EMULATION
