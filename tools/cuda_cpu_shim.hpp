#pragma once

#ifndef CUDA_KERNEL_CPU_EMULATION
#error "cuda_cpu_shim.hpp requires -DCUDA_KERNEL_CPU_EMULATION"
#endif

#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

typedef unsigned int uint;

struct dim3 {
  unsigned x = 1, y = 1, z = 1;
  dim3() = default;
  dim3(unsigned x_) : x(x_) {}
  dim3(unsigned x_, unsigned y_) : x(x_), y(y_) {}
  dim3(unsigned x_, unsigned y_, unsigned z_) : x(x_), y(y_), z(z_) {}
};

struct alignas(16) float4 {
  float x, y, z, w;
};

using cudaStream_t = void *;

namespace cuda_cpu {
class Barrier {
 public:
  explicit Barrier(unsigned n) : n_(n), count_(n), gen_(0) {}
  void wait() {
    std::unique_lock<std::mutex> lk(m_);
    const unsigned g = gen_;
    if (--count_ == 0) {
      gen_++;
      count_ = n_;
      cv_.notify_all();
    } else {
      cv_.wait(lk, [&] { return gen_ != g; });
    }
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  unsigned n_, count_, gen_;
};

extern thread_local dim3 t_threadIdx;
extern thread_local dim3 t_blockIdx;
extern dim3 g_blockDim;
extern dim3 g_gridDim;
extern Barrier *g_barrier;

inline void sync() {
  if (g_barrier) g_barrier->wait();
}

template <typename F>
void launch(dim3 grid, dim3 block, F body) {
  g_blockDim = block;
  g_gridDim = grid;
  const unsigned nthreads = block.x * block.y * block.z;

  for (unsigned bz = 0; bz < grid.z; ++bz) {
    for (unsigned by = 0; by < grid.y; ++by) {
      for (unsigned bx = 0; bx < grid.x; ++bx) {
        Barrier bar(nthreads);
        g_barrier = &bar;
        std::vector<std::thread> threads;
        threads.reserve(nthreads);
        for (unsigned tz = 0; tz < block.z; ++tz) {
          for (unsigned ty = 0; ty < block.y; ++ty) {
            for (unsigned tx = 0; tx < block.x; ++tx) {
              threads.emplace_back([=] {
                t_threadIdx = dim3(tx, ty, tz);
                t_blockIdx = dim3(bx, by, bz);
                body();
              });
            }
          }
        }
        for (auto &t : threads) t.join();
        g_barrier = nullptr;
      }
    }
  }
}
}

#define threadIdx cuda_cpu::t_threadIdx
#define blockIdx cuda_cpu::t_blockIdx
#define blockDim cuda_cpu::g_blockDim
#define gridDim cuda_cpu::g_gridDim
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline

#define __shared__ static

#define __align__(x) __attribute__((aligned(x)))
#define __syncthreads() cuda_cpu::sync()

inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
