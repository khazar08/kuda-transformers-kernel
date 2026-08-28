// =============================================================================
// PART 2b -- FUSED LAYERNORM. Mean and variance in ONE pass, then normalize.
//
// THE OP
//   y = (x - mean) / sqrt(var + eps) * gamma + beta,  per row (per token).
//   Like softmax it is memory-bound: a handful of flops per 4-byte element.
//
// THE NAIVE STRUCTURE THIS REPLACES
//   Pass 1: read X, reduce mean          -> write mean
//   Pass 2: read X, read mean, reduce var -> write var
//   Pass 3: read X, read mean/var, write Y
//   ~3N reads + N writes across 3 launches. The fused kernel does ~2N reads +
//   1N write in a single launch, and the mean/var never leave registers.
//
// ONE PASS FOR BOTH STATISTICS -- AND WHY NOT THE OBVIOUS WAY
//   The textbook one-pass trick accumulates sum and sum-of-squares, then uses
//   var = E[x^2] - E[x]^2. It is one line and it is a numerical trap: when the
//   mean is large relative to the standard deviation, those two terms are large
//   and nearly equal, catastrophic cancellation eats the significand, and in
//   fp32 you can get a NEGATIVE variance -> rsqrt -> NaN. Transformer
//   activations genuinely drift away from zero mean deeper in the network, so
//   this is a real failure mode, not a hypothetical.
//
//   Welford's algorithm updates (count, mean, M2) incrementally and never forms
//   E[x^2], so it stays accurate whatever the offset. The parallel merge rule
//   in reduce.cuh lets each thread run its own Welford over a strided slice and
//   then combine them exactly -- including when the row length is not a
//   multiple of the block size, so the partitions have unequal counts.
//
// A DETAIL WORTH NOTICING: rsqrtf
//   rsqrtf is a single hardware instruction on the SFU. Computing 1.0f/sqrtf(x)
//   instead costs a square root AND a division, both multi-instruction. On a
//   memory-bound kernel this is not where the time goes, but it is free to get
//   right and it is the kind of thing that shows up in a code review.
//
// EXPECTED RESULT
//   3 launches -> 1, traffic 4N -> 3N, and parity with torch.nn.LayerNorm
//   (which is itself fused, so matching it is the target, not beating it).
// =============================================================================

#include "common.cuh"
#include "launchers.h"
#include "reduce.cuh"

// Identity affine values, used when gamma/beta are absent. Written as helpers
// so the vectorized path has no branch inside its inner loop.
__device__ __forceinline__ float4 make_float4_ones() {
  float4 v; v.x = v.y = v.z = v.w = 1.0f; return v;
}
__device__ __forceinline__ float4 make_float4_zeros() {
  float4 v; v.x = v.y = v.z = v.w = 0.0f; return v;
}

template <int BLOCK>
__global__ void layernorm_fused_kernel(const float *__restrict__ X,
                                       const float *__restrict__ gamma,
                                       const float *__restrict__ beta,
                                       float *__restrict__ Y, int rows,
                                       int cols, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;

  const float *__restrict__ xr = X + (size_t)row * cols;
  float *__restrict__ yr = Y + (size_t)row * cols;

  // --- PASS A: per-thread Welford over a coalesced strided slice ------------
  float count = 0.0f, mean = 0.0f, m2 = 0.0f;
  for (int i = threadIdx.x; i < cols; i += BLOCK) {
    const float v = xr[i];
    count += 1.0f;
    const float delta = v - mean;
    mean += delta / count;
    // Note the second delta is recomputed AGAINST THE UPDATED MEAN. Using the
    // same delta twice here is the single most common way to get Welford wrong,
    // and it produces a variance that is subtly too small rather than obviously
    // broken.
    m2 += delta * (v - mean);
  }

  // Merge the per-thread triples into the row-wide statistic.
  blockReduceWelford<BLOCK>(count, mean, m2);

  // Population variance (divide by N, not N-1) -- this is what LayerNorm and
  // torch.nn.LayerNorm both use.
  const float var = m2 / count;
  const float rstd = rsqrtf(var + eps);

  // --- PASS B: normalize and apply the affine transform --------------------
  for (int i = threadIdx.x; i < cols; i += BLOCK) {
    const float g = (gamma != nullptr) ? gamma[i] : 1.0f;
    const float b = (beta != nullptr) ? beta[i] : 0.0f;
    // Written as an FMA chain: ((x - mean) * rstd) * g + b.
    yr[i] = fmaf((xr[i] - mean) * rstd, g, b);
  }
}

// =============================================================================
// MEASURED OPTIMIZATION ROUND (added after the first T4 benchmark)
//
// The scalar kernel lost to torch.layer_norm at every shape except the widest
// (1.1x-1.45x slower), for the same two reasons as softmax: 32-bit loads on a
// memory-bound kernel, and a fixed 256-thread block that leaves most threads
// idle on narrow rows while still paying the full reduction depth.
//
// Welford vectorizes cleanly because the parallel merge rule is associative:
// each thread runs Welford over its four lanes sequentially, exactly as it
// would over four separate iterations. No accuracy is given up for the speed.
// =============================================================================

template <int BLOCK>
__global__ void layernorm_fused_vec4_kernel(const float *__restrict__ X,
                                            const float *__restrict__ gamma,
                                            const float *__restrict__ beta,
                                            float *__restrict__ Y, int rows,
                                            int cols, float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;

  const float4 *__restrict__ xr =
      reinterpret_cast<const float4 *>(X + (size_t)row * cols);
  float4 *__restrict__ yr = reinterpret_cast<float4 *>(Y + (size_t)row * cols);
  const float4 *__restrict__ g4 = reinterpret_cast<const float4 *>(gamma);
  const float4 *__restrict__ b4 = reinterpret_cast<const float4 *>(beta);
  const int cols4 = cols / 4;

  float count = 0.0f, mean = 0.0f, m2 = 0.0f;
  for (int i = threadIdx.x; i < cols4; i += BLOCK) {
    const float4 v = xr[i];
    // Four sequential Welford updates. Unrolled by hand so the compiler keeps
    // the running triple in registers rather than reloading it.
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float x = (k == 0) ? v.x : (k == 1) ? v.y : (k == 2) ? v.z : v.w;
      count += 1.0f;
      const float delta = x - mean;
      mean += delta / count;
      m2 += delta * (x - mean);  // second delta uses the UPDATED mean
    }
  }

  blockReduceWelford<BLOCK>(count, mean, m2);

  const float rstd = rsqrtf(m2 / count + eps);

  for (int i = threadIdx.x; i < cols4; i += BLOCK) {
    const float4 v = xr[i];
    const float4 g = (gamma != nullptr) ? g4[i] : make_float4_ones();
    const float4 b = (beta != nullptr) ? b4[i] : make_float4_zeros();
    float4 o;
    o.x = fmaf((v.x - mean) * rstd, g.x, b.x);
    o.y = fmaf((v.y - mean) * rstd, g.y, b.y);
    o.z = fmaf((v.z - mean) * rstd, g.z, b.z);
    o.w = fmaf((v.w - mean) * rstd, g.w, b.w);
    yr[i] = o;
  }
}

#ifndef CUDA_KERNEL_CPU_EMULATION
static int layernorm_block_for(int work_items) {
  if (work_items <= 32) return 32;
  if (work_items <= 64) return 64;
  if (work_items <= 128) return 128;
  return 256;
}

void launch_layernorm_fused(const float *X, const float *gamma,
                            const float *beta, float *Y, int rows, int cols,
                            float eps, cudaStream_t stream) {
  const bool vec4 = (cols % 4 == 0);
  const int block = layernorm_block_for(vec4 ? cols / 4 : cols);

#define LAUNCH_LN(B)                                                           \
  do {                                                                         \
    if (vec4)                                                                  \
      layernorm_fused_vec4_kernel<B>                                           \
          <<<rows, B, 0, stream>>>(X, gamma, beta, Y, rows, cols, eps);         \
    else                                                                       \
      layernorm_fused_kernel<B>                                                \
          <<<rows, B, 0, stream>>>(X, gamma, beta, Y, rows, cols, eps);         \
  } while (0)

  switch (block) {
    case 32:  LAUNCH_LN(32);  break;
    case 64:  LAUNCH_LN(64);  break;
    case 128: LAUNCH_LN(128); break;
    default:  LAUNCH_LN(256); break;
  }
#undef LAUNCH_LN
}
#endif  // CUDA_KERNEL_CPU_EMULATION
