// =============================================================================
// PART 2a -- FUSED ROW-WISE SOFTMAX with streaming (online) max.
//
// WHY SOFTMAX IS A FUSION PROBLEM, NOT A COMPUTE PROBLEM
//   Per element softmax does one exp and a couple of flops against 4 bytes read
//   and 4 written -- arithmetic intensity around 0.1 flops/byte, versus a T4
//   ridge point near 25. It sits pinned against the left wall of the roofline.
//   Nothing you do to the arithmetic matters; the ONLY lever is bytes moved.
//
// WHAT THE NAIVE VERSION COSTS (see 10_softmax_naive.cu)
//   Pass 1: read X, write row_max          -> N reads
//   Pass 2: read X, write Y, write row_sum -> N reads + N writes
//   Pass 3: read Y, write Y                -> N reads + N writes
//   Total ~3N reads + 2N writes = 5N element-touches, across 3 kernel launches.
//   The passes CANNOT share state any other way: a kernel launch is a global
//   barrier and registers do not survive it, so the row statistics have to be
//   round-tripped through DRAM.
//
// WHAT THIS KERNEL COSTS
//   Pass A: read X, keep (max, sum) in registers
//   Pass B: read X, write Y
//   Total ~2N reads + N writes = 3N, in ONE launch. That is a 5/3 = 1.67x cut
//   in traffic and 3 launches -> 1. In practice the win is larger than 1.67x
//   because the second read of X frequently hits L2 (4 MB on a T4) while it is
//   still warm from pass A, whereas the naive version's passes are far enough
//   apart that the line has usually been evicted.
//
// THE STREAMING MAX (this is the FlashAttention identity)
//   Subtracting the row max before exp is mandatory for stability: exp(89.f)
//   already overflows fp32. But the naive way to get the max is a separate pass.
//   Instead we maintain a running (m, s) and, whenever a larger element appears,
//   rescale the sum we have accumulated so far:
//       m_new = max(m_old, x)
//       s_new = s_old * exp(m_old - m_new) + exp(x - m_new)
//   Every partial sum stays expressed relative to the current max, so one pass
//   suffices and no intermediate ever exceeds exp(0) = 1. This is exactly the
//   trick that lets FlashAttention tile softmax without materializing the row.
//
// EXPECTED RESULT
//   Kernel launches 3 -> 1, global traffic 5N -> 3N, and achieved bandwidth
//   should climb toward the T4's 320 GB/s ceiling. Speedup versus the 3-pass
//   version should land somewhere between 1.7x and 3x depending on cache reuse.
//   It should be close to torch.softmax, which is itself a fused kernel -- if
//   we match it, that is the correct outcome, not a disappointment.
// =============================================================================

#include "common.cuh"
#include "launchers.h"
#include "reduce.cuh"

template <int BLOCK>
__global__ void softmax_fused_kernel(const float *__restrict__ X,
                                     float *__restrict__ Y, int rows,
                                     int cols) {
  // One block per row. The block cooperatively owns the entire reduction, so
  // no cross-block communication is ever needed -- which is what makes the
  // single-launch formulation possible at all.
  const int row = blockIdx.x;
  if (row >= rows) return;  // uniform across the block, so no barrier hazard

  const float *__restrict__ xr = X + (size_t)row * cols;
  float *__restrict__ yr = Y + (size_t)row * cols;

  // --- PASS A: one streaming pass for both max and sum ---------------------
  // Grid-stride within the row, so consecutive threads touch consecutive
  // floats and every load is fully coalesced.
  float m = -FLT_MAX;  // identity: see the note in reduce.cuh on -FLT_MAX
  float s = 0.0f;
  for (int i = threadIdx.x; i < cols; i += BLOCK) {
    const float v = xr[i];
    const float mn = fmaxf(m, v);
    // Rescale the running sum to the new max, then add this element.
    s = s * expf(m - mn) + expf(v - mn);
    m = mn;
  }

  // Combine the per-thread (m, s) partials into the row-wide pair.
  blockReduceSoftmaxState<BLOCK>(m, s);

  // --- PASS B: normalize ---------------------------------------------------
  // One reciprocal for the whole row instead of a divide per element: division
  // is a multi-instruction sequence on the SM, multiplication is one.
  const float inv_sum = 1.0f / s;
  for (int i = threadIdx.x; i < cols; i += BLOCK) {
    yr[i] = expf(xr[i] - m) * inv_sum;
  }
}

// =============================================================================
// MEASURED OPTIMIZATION ROUND (added after the first T4 benchmark)
//
// The scalar kernel above beat the 3-pass baseline by ~1.9-2.6x, confirming the
// fusion thesis, but LOST to torch.softmax at both ends of the shape sweep:
// 3.0x slower at 4096x256 and 1.4x slower at 4096x4096. Two distinct causes,
// each with a distinct fix:
//
//   WIDE ROWS (4096x4096): we issued one 32-bit load per element. torch uses
//   128-bit accesses. Fix: float4 loads, quartering memory instruction count on
//   a kernel whose runtime is decided by the memory pipeline.
//
//   NARROW ROWS (4096x256): with cols=256 and a fixed 256-thread block, every
//   thread handled exactly ONE element and then paid a full 8-level shared-
//   memory tree reduction. The reduction completely dominated the useful work.
//   Fix: size the block to the row, so the tree stays shallow and threads are
//   not idle.
//
// This is what benchmarking is for. Both fixes were invisible from inspection.
// =============================================================================

template <int BLOCK>
__global__ void softmax_fused_vec4_kernel(const float *__restrict__ X,
                                          float *__restrict__ Y, int rows,
                                          int cols) {
  const int row = blockIdx.x;
  if (row >= rows) return;

  // Legal only when cols % 4 == 0: torch allocations are 512-byte aligned, so
  // each row start is 16-byte aligned exactly when the row length is a multiple
  // of 4 floats. The launcher enforces this.
  const float4 *__restrict__ xr =
      reinterpret_cast<const float4 *>(X + (size_t)row * cols);
  float4 *__restrict__ yr = reinterpret_cast<float4 *>(Y + (size_t)row * cols);
  const int cols4 = cols / 4;

  float m = -FLT_MAX, s = 0.0f;
  for (int i = threadIdx.x; i < cols4; i += BLOCK) {
    const float4 v = xr[i];
    // One streaming-max update for all four lanes at once. The rescale of the
    // running sum is paid once per float4 instead of once per element.
    const float mn = fmaxf(m, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    s = s * expf(m - mn) + expf(v.x - mn) + expf(v.y - mn) + expf(v.z - mn) +
        expf(v.w - mn);
    m = mn;
  }

  blockReduceSoftmaxState<BLOCK>(m, s);

  const float inv_sum = 1.0f / s;
  for (int i = threadIdx.x; i < cols4; i += BLOCK) {
    const float4 v = xr[i];
    float4 o;
    o.x = expf(v.x - m) * inv_sum;
    o.y = expf(v.y - m) * inv_sum;
    o.z = expf(v.z - m) * inv_sum;
    o.w = expf(v.w - m) * inv_sum;
    yr[i] = o;
  }
}

#ifndef CUDA_KERNEL_CPU_EMULATION
// Pick the smallest block that still gives every thread work. A 256-thread
// block on a 256-wide row wastes 7 of 8 reduction levels on a tree whose inputs
// are mostly a single element each.
static int softmax_block_for(int work_items) {
  if (work_items <= 32) return 32;
  if (work_items <= 64) return 64;
  if (work_items <= 128) return 128;
  return 256;
}

void launch_softmax_fused(const float *X, float *Y, int rows, int cols,
                          cudaStream_t stream) {
  // Vectorize when the row length allows it; fall back to the fully general
  // scalar kernel otherwise. Correctness never depends on the fast path.
  const bool vec4 = (cols % 4 == 0);
  const int block = softmax_block_for(vec4 ? cols / 4 : cols);

#define LAUNCH_SOFTMAX(B)                                                      \
  do {                                                                         \
    if (vec4)                                                                  \
      softmax_fused_vec4_kernel<B><<<rows, B, 0, stream>>>(X, Y, rows, cols);   \
    else                                                                       \
      softmax_fused_kernel<B><<<rows, B, 0, stream>>>(X, Y, rows, cols);        \
  } while (0)

  switch (block) {
    case 32:  LAUNCH_SOFTMAX(32);  break;
    case 64:  LAUNCH_SOFTMAX(64);  break;
    case 128: LAUNCH_SOFTMAX(128); break;
    default:  LAUNCH_SOFTMAX(256); break;
  }
#undef LAUNCH_SOFTMAX
}
#endif  // CUDA_KERNEL_CPU_EMULATION
