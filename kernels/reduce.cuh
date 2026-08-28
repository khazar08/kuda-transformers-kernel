#pragma once
// =============================================================================
// reduce.cuh -- block-level reductions used by the fused kernels.
//
// A DELIBERATE DESIGN CHOICE: SHARED MEMORY, NOT WARP SHUFFLES
//   The fastest block reduction uses __shfl_down_sync within a warp and only
//   touches shared memory to combine across warps. We use a plain shared-memory
//   tree instead, for one reason: warp shuffles cannot be emulated on the CPU,
//   and this project's correctness harness (tools/) runs the real kernel source
//   on the host. For softmax and LayerNorm the reduction is amortized over a
//   full row of DRAM traffic and is nowhere near the critical path -- these are
//   memory-bound kernels, and the bytes decide the runtime, not the tree. Test
//   coverage is worth more here than an optimization on a non-bottleneck.
//   Switching to shuffles is listed as future work in the README.
// =============================================================================

#include "common.cuh"
#include <cfloat>
#include <cmath>

// --- softmax state reduction ------------------------------------------------
// Combines per-thread (max, sum-of-exp) pairs into the block-wide pair.
//
// The merge rule is the heart of streaming/online softmax:
//   m = max(m1, m2)
//   s = s1*exp(m1 - m) + s2*exp(m2 - m)
// Each partial sum was computed relative to its own local max, so it must be
// rescaled to the new common max before the two can be added. This is the same
// identity FlashAttention uses to tile softmax across blocks.
//
// NUMERICAL NOTE: the identity element is (-FLT_MAX, 0), NOT (-inf, 0). With
// -inf, merging two empty states gives (-inf) - (-inf) = NaN, which silently
// poisons the whole row. -FLT_MAX makes the difference a finite 0, exp(0) = 1,
// and a zero sum still contributes nothing.
template <int BLOCK>
__device__ void blockReduceSoftmaxState(float &m, float &s) {
  __shared__ float s_max[BLOCK];
  __shared__ float s_sum[BLOCK];
  const int tid = threadIdx.x;

  s_max[tid] = m;
  s_sum[tid] = s;
  __syncthreads();

  // Tree reduction. The __syncthreads() sits OUTSIDE the `if` so that every
  // thread in the block reaches it -- a barrier inside divergent control flow
  // is undefined behaviour and a classic source of hangs.
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

// --- Welford reduction ------------------------------------------------------
// Combines per-thread (count, mean, M2) triples, where M2 is the sum of squared
// deviations from the mean, so variance = M2 / count.
//
// WHY WELFORD AND NOT sum / sum-of-squares
//   The textbook one-pass trick computes var = E[x^2] - E[x]^2. It is faster and
//   it is numerically dangerous: when the mean is large relative to the spread,
//   it subtracts two nearly equal large numbers and catastrophic cancellation
//   destroys the result -- in fp32 this can produce a NEGATIVE variance, and
//   then rsqrt of a negative is NaN. Welford never forms E[x^2], so it stays
//   accurate regardless of the input's offset. LayerNorm runs on activations
//   that can drift far from zero mid-network, so this matters in practice.
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

// --- plain max / sum reductions --------------------------------------------
// Used only by the 3-pass naive softmax baseline, which needs them as separate
// standalone passes -- that separation is exactly the thing being measured.
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
