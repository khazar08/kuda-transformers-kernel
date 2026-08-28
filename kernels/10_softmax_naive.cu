#include "common.cuh"
#include "launchers.h"
#include "reduce.cuh"

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
