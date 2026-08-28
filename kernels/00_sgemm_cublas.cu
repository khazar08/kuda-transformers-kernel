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


  CUBLAS_CHECK(cublasSetStream(handle, stream));

  // See the header comment: operands swapped, dimensions permuted.
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           /*m=*/N, /*n=*/M, /*k=*/K, &alpha,
                           /*A=*/B, /*lda=*/N,
                           /*B=*/A, /*ldb=*/K, &beta,
                           /*C=*/C, /*ldc=*/N));
}
#endif  // CUDA_KERNEL_CPU_EMULATION
