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


  const float inv_sum = 1.0f / s;
  for (int i = threadIdx.x; i < cols; i += BLOCK) {
    yr[i] = expf(xr[i] - m) * inv_sum;
  }
}


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
