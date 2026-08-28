#pragma once
// =============================================================================
// common.cuh -- shared helpers for every kernel stage.
//
// Deliberately tiny and dependency-free: nothing in kernels/ includes torch or
// ATen headers. The kernels are plain CUDA C++ and can be compiled standalone
// with nvcc, or run on the CPU by tools/cpu_emu (see tools/cuda_cpu_shim.hpp).
// The torch glue lives entirely in extension/bindings.cpp.
// =============================================================================

#ifndef CUDA_KERNEL_CPU_EMULATION
#include <cuda_runtime.h>
#endif

// Integer ceiling division. Used everywhere to size grids so that a matrix
// dimension that isn't an exact multiple of the tile size still gets covered
// (the extra threads are masked off by bounds checks inside the kernel).
// `uint` is widely used in CUDA sample code but is NOT provided by the CUDA
// headers -- on Linux it leaks in from sys/types.h, which is fragile. Declaring
// it explicitly makes the kernels portable. Redeclaring an identical typedef is
// legal C++, so this is safe even where the platform already defines it.
typedef unsigned int uint;

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// Reinterpret a float* as a float4* so we can issue 128-bit (16-byte) loads.
// A single float4 load compiles to one LDG.E.128 instruction instead of four
// LDG.E.32s, which quarters the number of memory instructions in flight and
// lets one warp request a full 128-byte cache line in one go. Only legal when
// the pointer is 16-byte aligned -- every use site in this repo checks that.
#define AS_FLOAT4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])
#define AS_CONST_FLOAT4(ptr) (reinterpret_cast<const float4 *>(&(ptr))[0])
