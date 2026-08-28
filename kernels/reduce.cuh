#pragma once
#include "common.cuh"
#include <cfloat>
#include <cmath>

template <int BLOCK>
__device__ void blockReduceSoftmaxState(float &m, float &s) {
  __shared__ float s_max[BLOCK];
  __shared__ float s_sum[BLOCK];
  const int tid = threadIdx.x;

  s_max[tid] = m;
  s_sum[tid] = s;
  __syncthreads();

  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float m1 = s_max[tid], s1 = s_sum[tid];
      const float m2 = s_max[tid + stride], s2 = s_sum[tid + stride];
      const float mn = fmaxf(m1, m2);
      s_max[tid] = mn;
      s_sum[tid] = s1 * expf(m1 - mn) + s2 * expf(m2 - mn);
    }
    __syncthreads();
  }

  m = s_max[0];
  s = s_sum[0];
  // Trailing barrier: every thread has now read the result, so the buffers are
  // safe for the caller to reuse on a later reduction.
  __syncthreads();
}

template <int BLOCK>
__device__ void blockReduceWelford(float &count, float &mean, float &m2) {
  __shared__ float s_cnt[BLOCK];
  __shared__ float s_mean[BLOCK];
  __shared__ float s_m2[BLOCK];
  const int tid = threadIdx.x;

  s_cnt[tid] = count;
  s_mean[tid] = mean;
  s_m2[tid] = m2;
  __syncthreads();

  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float na = s_cnt[tid], nb = s_cnt[tid + stride];
      const float n = na + nb;
      if (n > 0.0f) {
        const float delta = s_mean[tid + stride] - s_mean[tid];
        // Chan et al.'s parallel merge. The nb/n weighting is what keeps this
        // exact for unequal partition sizes -- which happens whenever the row
        // length is not a multiple of the block size.
        s_mean[tid] = s_mean[tid] + delta * (nb / n);
        s_m2[tid] = s_m2[tid] + s_m2[tid + stride] + delta * delta * (na * nb / n);
        s_cnt[tid] = n;
      }
    }
    __syncthreads();
  }

  count = s_cnt[0];
  mean = s_mean[0];
  m2 = s_m2[0];
  __syncthreads();
}

template <int BLOCK>
__device__ float blockReduceMax(float v) {
  __shared__ float buf[BLOCK];
  const int tid = threadIdx.x;
  buf[tid] = v;
  __syncthreads();
  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (tid < stride) buf[tid] = fmaxf(buf[tid], buf[tid + stride]);
    __syncthreads();
  }
  const float out = buf[0];
  __syncthreads();
  return out;
}

template <int BLOCK>
__device__ float blockReduceSum(float v) {
  __shared__ float buf[BLOCK];
  const int tid = threadIdx.x;
  buf[tid] = v;
  __syncthreads();
  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (tid < stride) buf[tid] += buf[tid + stride];
    __syncthreads();
  }
  const float out = buf[0];
  __syncthreads();
  return out;
}
