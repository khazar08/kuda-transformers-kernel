#pragma once
// =============================================================================
// launchers.h -- the ABI between the CUDA kernels and the PyTorch binding.
//
// Every stage of the optimization ladder exposes the identical signature, which
// is what makes the benchmark harness able to treat them as interchangeable:
// the Python side just picks a name and the numbers are directly comparable
// because nothing else about the call differs.
//
// Semantics are standard SGEMM: C = alpha * (A @ B) + beta * C
//   A is M x K, B is K x N, C is M x N, ALL ROW-MAJOR.
//
// Row-major matters. cuBLAS is column-major, so the cuBLAS wrapper has to do an
// argument swap to stay compatible with everyone else -- see 00_sgemm_cublas.cu.
//
// The stream parameter is threaded through so the kernels run on PyTorch's
// current stream rather than the legacy default stream. If we ignored it, our
// kernels and torch's kernels would land on different streams and the CUDA
// event timings would be measuring the wrong thing.
// =============================================================================

#ifndef CUDA_KERNEL_CPU_EMULATION
#include <cuda_runtime.h>
#endif

#define SGEMM_LAUNCHER_SIGNATURE(name)                                         \
  void name(const float *A, const float *B, float *C, int M, int N, int K,     \
            float alpha, float beta, cudaStream_t stream)

SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_cublas);        // ceiling / reference
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_naive);         // stage 1
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_coalesced);     // stage 2
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_smem);          // stage 3
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_blocktile_1d);  // stage 4
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_blocktile_2d);  // stage 5
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_vectorized);    // stage 6

// --- Part 2: fused kernels ---------------------------------------------------
// Row-wise ops over a (rows x cols) row-major matrix. In transformer terms
// `rows` is tokens (or batch*heads*queries) and `cols` is the reduction axis.

// The 3-pass baseline. Needs two scratch buffers of length `rows` because the
// per-row statistics cannot survive a kernel launch boundary any other way --
// which is precisely the cost fusion eliminates.
void launch_softmax_naive(const float *X, float *Y, float *row_max,
                          float *row_sum, int rows, int cols,
                          cudaStream_t stream);

void launch_softmax_fused(const float *X, float *Y, int rows, int cols,
                          cudaStream_t stream);

void launch_layernorm_fused(const float *X, const float *gamma,
                            const float *beta, float *Y, int rows, int cols,
                            float eps, cudaStream_t stream);
