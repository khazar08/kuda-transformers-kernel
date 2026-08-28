// =============================================================================
// REFERENCE / CEILING -- cublasSgemm.
//
// This is not a stage of the ladder; it is the target. Every stage is reported
// as a percentage of this number.
//
// WHY cuBLAS IS A FAIR COMPARISON ON A T4 SPECIFICALLY
//   On Ampere and later, cublasSgemm may silently dispatch to TF32 tensor cores,
//   which would put a hand-written FP32 SIMT kernel in a fight against hardware
//   it is not allowed to use -- an apples-to-oranges comparison that makes the
//   percentages meaningless. Turing (sm_75) has no TF32 path, so cublasSgemm
//   here is genuine FP32 SIMT running the same instructions our kernels run.
//   The comparison is honest, which is exactly why the T4 is a good teaching
//   target for this project.
//
// THE ROW-MAJOR / COLUMN-MAJOR TRICK (the thing everyone gets wrong)
//   cuBLAS is column-major; our whole codebase is row-major. Copying or
//   explicitly transposing would add a memory pass and destroy the comparison.
//   Instead we exploit an identity that costs nothing:
//
//     A row-major (r,c) matrix with leading dimension c IS, bit for bit, the
//     column-major representation of its transpose (c,r) with the same ld.
//
//   So reinterpreting our row-major inputs as column-major gives cuBLAS A^T,
//   B^T and C^T for free. We want C = A @ B, which transposes to
//   C^T = B^T @ A^T. That is a plain no-transpose GEMM in cuBLAS's world --
//   we simply pass B and A in SWAPPED ORDER with the dimensions permuted:
//
//     m=N, n=M, k=K,  first operand B (lda=N),  second operand A (ldb=K)
//
//   Zero data movement; just a different interpretation of the same bytes.
// =============================================================================

#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

#include "common.cuh"
#include "launchers.h"

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status_ = (call);                                           \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                    \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)status_,         \
                   __FILE__, __LINE__);                                        \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

// One handle for the process lifetime. Creating a cuBLAS handle allocates
// workspace and costs milliseconds -- doing it per call would dominate the
// measurement and make cuBLAS look artificially slow, which is exactly the kind
// of benchmarking error that invalidates a comparison. Function-local statics
// are initialized exactly once and thread-safely under C++11 and later.
static cublasHandle_t get_handle() {
  static cublasHandle_t handle = [] {
    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    // Pin the math mode so a future CUDA version cannot silently opt us into a
    // reduced-precision path and invalidate the FP32-vs-FP32 comparison.
    CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
    return h;
  }();
  return handle;
}

#ifndef CUDA_KERNEL_CPU_EMULATION
// The launcher uses the <<<>>> launch syntax, which only nvcc can parse, so it
// is compiled out for the CPU-emulation correctness build (tools/cpu_emu_test).
// That build invokes the kernel above directly through cuda_cpu::launch().
SGEMM_LAUNCHER_SIGNATURE(launch_sgemm_cublas) {
  cublasHandle_t handle = get_handle();

  // Bind cuBLAS to PyTorch's current stream. If we skipped this, cuBLAS would
  // run on the legacy default stream while our CUDA events were recorded on
  // torch's stream, and the timings would be measuring stream synchronization
  // rather than the GEMM.
  CUBLAS_CHECK(cublasSetStream(handle, stream));

  // See the header comment: operands swapped, dimensions permuted.
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           /*m=*/N, /*n=*/M, /*k=*/K, &alpha,
                           /*A=*/B, /*lda=*/N,
                           /*B=*/A, /*ldb=*/K, &beta,
                           /*C=*/C, /*ldc=*/N));
}
#endif  // CUDA_KERNEL_CPU_EMULATION
