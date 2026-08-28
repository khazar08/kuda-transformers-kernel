//   y = (x - mean) / sqrt(var + eps) * gamma + beta,  per row (per token).
#include "common.cuh"
#include "launchers.h"
#include "reduce.cuh"

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

  float count = 0.0f, mean = 0.0f, m2 = 0.0f;
  for (int i = threadIdx.x; i < cols; i += BLOCK) {
    const float v = xr[i];
    count += 1.0f;
    const float delta = v - mean;
    mean += delta / count;
    m2 += delta * (v - mean);
  }

  blockReduceWelford<BLOCK>(count, mean, m2);

  const float var = m2 / count;
  const float rstd = rsqrtf(var + eps);

  for (int i = threadIdx.x; i < cols; i += BLOCK) {
    const float g = (gamma != nullptr) ? gamma[i] : 1.0f;
    const float b = (beta != nullptr) ? beta[i] : 0.0f;
    // Written as an FMA chain: ((x - mean) * rstd) * g + b.
    yr[i] = fmaf((xr[i] - mean) * rstd, g, b);
  }
}


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
