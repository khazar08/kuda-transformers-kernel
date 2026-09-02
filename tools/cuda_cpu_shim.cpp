#define CUDA_KERNEL_CPU_EMULATION
#include "cuda_cpu_shim.hpp"

namespace cuda_cpu {
thread_local dim3 t_threadIdx;
thread_local dim3 t_blockIdx;
dim3 g_blockDim;
dim3 g_gridDim;
Barrier *g_barrier = nullptr;
}
