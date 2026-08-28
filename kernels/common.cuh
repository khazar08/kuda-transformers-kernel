#pragma once

#ifndef CUDA_KERNEL_CPU_EMULATION
#include <cuda_runtime.h>
#endif

typedef unsigned int uint;

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#define AS_FLOAT4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])
#define AS_CONST_FLOAT4(ptr) (reinterpret_cast<const float4 *>(&(ptr))[0])
