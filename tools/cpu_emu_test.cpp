#define CUDA_KERNEL_CPU_EMULATION
#include "cuda_cpu_shim.hpp"

#include "../kernels/01_sgemm_naive.cu"
#include "../kernels/02_sgemm_coalesced.cu"
#include "../kernels/03_sgemm_smem.cu"
#include "../kernels/04_sgemm_blocktile_1d.cu"
#include "../kernels/05_sgemm_blocktile_2d.cu"
#include "../kernels/06_sgemm_vectorized.cu"
#include "../kernels/10_softmax_naive.cu"
#include "../kernels/11_softmax_fused.cu"
#include "../kernels/12_layernorm_fused.cu"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static int g_failures = 0;

static void reference_sgemm(const std::vector<float> &A,
                            const std::vector<float> &B,
                            const std::vector<float> &C_in,
                            std::vector<float> &C_out, int M, int N, int K,
                            float alpha, float beta) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k)
        acc += (double)A[m * K + k] * (double)B[k * N + n];
      double v = (double)alpha * acc;
      if (beta != 0.0f) v += (double)beta * (double)C_in[m * N + n];
      C_out[m * N + n] = (float)v;
    }
  }
}

static void check(const std::string &name, const std::vector<float> &got,
                  const std::vector<float> &want, int M, int N, float tol) {
  double max_abs = 0.0;
  int bad_m = -1, bad_n = -1;
  for (int i = 0; i < M * N; ++i) {
    const double d = std::fabs((double)got[i] - (double)want[i]);
    if (d > max_abs) { max_abs = d; bad_m = i / N; bad_n = i % N; }
  }
  const bool ok = max_abs <= tol;
  if (!ok) ++g_failures;
  std::printf("  %-22s max_abs_err = %-12.3e  %s", name.c_str(), max_abs,
              ok ? "PASS\n" : "FAIL");
  if (!ok)
    std::printf("  (worst at [%d,%d]: got %.6f want %.6f)\n", bad_m, bad_n,
                got[bad_m * N + bad_n], want[bad_m * N + bad_n]);
}

static void test_fused() {
  std::mt19937 rng(99);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  constexpr int BLOCK = 256;

  const int rows = 6, cols = 257;

  struct Case { float offset; const char *label; };

  const std::vector<Case> cases = {
      {0.0f, "centered  (offset    0)"},
      {200.0f, "hot       (offset  200)  <- naive exp() overflows here"},
      {1000.0f, "offset    (offset 1000)  <- E[x^2]-E[x]^2 cancels here"},
  };

  for (const auto &c : cases) {
    std::printf("\n=== fused kernels, %s ===\n", c.label);
    std::vector<float> X(rows * cols), gamma(cols), beta_v(cols);
    for (auto &v : X) v = dist(rng) + c.offset;
    for (auto &v : gamma) v = dist(rng);
    for (auto &v : beta_v) v = dist(rng);

    std::vector<float> want(rows * cols);
    for (int r = 0; r < rows; ++r) {
      double mx = -1e300;
      for (int i = 0; i < cols; ++i) mx = std::fmax(mx, (double)X[r * cols + i]);
      double sum = 0.0;
      for (int i = 0; i < cols; ++i) sum += std::exp((double)X[r * cols + i] - mx);
      for (int i = 0; i < cols; ++i)
        want[r * cols + i] = (float)(std::exp((double)X[r * cols + i] - mx) / sum);
    }

    const float *pX = X.data();
    std::vector<float> got(rows * cols);

    {
      const int c4 = 256, r4 = 4;
      std::vector<float> Xv(r4 * c4), wantv(r4 * c4), gotv(r4 * c4);
      for (auto &v : Xv) v = dist(rng) + c.offset;
      for (int r = 0; r < r4; ++r) {
        double mx = -1e300;
        for (int i = 0; i < c4; ++i) mx = std::fmax(mx, (double)Xv[r * c4 + i]);
        double sum = 0.0;
        for (int i = 0; i < c4; ++i) sum += std::exp((double)Xv[r * c4 + i] - mx);
        for (int i = 0; i < c4; ++i)
          wantv[r * c4 + i] = (float)(std::exp((double)Xv[r * c4 + i] - mx) / sum);
      }
      const float *pXv = Xv.data();
      float *pYv = gotv.data();
      cuda_cpu::launch(dim3(r4), dim3(64), [=] {
        softmax_fused_vec4_kernel<64>(pXv, pYv, r4, c4);
      });
      check("softmax vec4", gotv, wantv, r4, c4, 1e-6f);

      const float eps4 = 1e-5f;
      for (int r = 0; r < r4; ++r) {
        double mean = 0.0;
        for (int i = 0; i < c4; ++i) mean += (double)Xv[r * c4 + i];
        mean /= c4;
        double var = 0.0;
        for (int i = 0; i < c4; ++i) {
          const double d = (double)Xv[r * c4 + i] - mean;
          var += d * d;
        }
        var /= c4;
        const double rstd = 1.0 / std::sqrt(var + (double)eps4);
        for (int i = 0; i < c4; ++i)
          wantv[r * c4 + i] = (float)(((double)Xv[r * c4 + i] - mean) * rstd);
      }
      cuda_cpu::launch(dim3(r4), dim3(64), [=] {
        layernorm_fused_vec4_kernel<64>(pXv, nullptr, nullptr, pYv, r4, c4, eps4);
      });
      check("layernorm vec4", gotv, wantv, r4, c4, 2e-4f);
    }

    {
      float *pY = got.data();
      cuda_cpu::launch(dim3(rows), dim3(BLOCK),
                       [=] { softmax_fused_kernel<BLOCK>(pX, pY, rows, cols); });
      check("softmax fused", got, want, rows, cols, 1e-6f);
    }

    {
      std::vector<float> rmax(rows), rsum(rows);
      float *pY = got.data();
      float *pM = rmax.data(), *pS = rsum.data();
      cuda_cpu::launch(dim3(rows), dim3(BLOCK), [=] {
        softmax_pass1_max<BLOCK>(pX, pM, rows, cols);
      });
      cuda_cpu::launch(dim3(rows), dim3(BLOCK), [=] {
        softmax_pass2_exp_sum<BLOCK>(pX, pY, pM, pS, rows, cols);
      });
      cuda_cpu::launch(dim3(rows), dim3(BLOCK), [=] {
        softmax_pass3_normalize<BLOCK>(pY, pS, rows, cols);
      });
      check("softmax 3-pass", got, want, rows, cols, 1e-6f);
    }

    const float eps = 1e-5f;
    for (int r = 0; r < rows; ++r) {
      double mean = 0.0;
      for (int i = 0; i < cols; ++i) mean += (double)X[r * cols + i];
      mean /= cols;
      double var = 0.0;
      for (int i = 0; i < cols; ++i) {
        const double d = (double)X[r * cols + i] - mean;
        var += d * d;
      }
      var /= cols;
      const double rstd = 1.0 / std::sqrt(var + (double)eps);
      for (int i = 0; i < cols; ++i)
        want[r * cols + i] =
            (float)(((double)X[r * cols + i] - mean) * rstd * (double)gamma[i] +
                    (double)beta_v[i]);
    }
    {
      float *pY = got.data();
      const float *pG = gamma.data(), *pB = beta_v.data();
      cuda_cpu::launch(dim3(rows), dim3(BLOCK), [=] {
        layernorm_fused_kernel<BLOCK>(pX, pG, pB, pY, rows, cols, eps);
      });

      check("layernorm fused", got, want, rows, cols, 2e-4f);
    }
  }
}

int main() {
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  struct Shape { int M, N, K; float alpha, beta; const char *label; };
  const std::vector<Shape> shapes = {
      {128, 128, 32, 1.0f, 0.0f, "aligned  128x128x32  alpha=1 beta=0"},
      {128, 128, 32, 0.5f, 2.0f, "aligned  128x128x32  alpha=.5 beta=2"},

      {67, 53, 41, 1.0f, 0.0f, "ragged    67x53x41   alpha=1 beta=0"},
      {67, 53, 41, 0.5f, 2.0f, "ragged    67x53x41   alpha=.5 beta=2"},
  };

  for (const auto &s : shapes) {
    const int M = s.M, N = s.N, K = s.K;
    std::printf("\n=== %s ===\n", s.label);

    std::vector<float> A(M * K), B(K * N), C_init(M * N), want(M * N), got(M * N);
    for (auto &v : A) v = dist(rng);
    for (auto &v : B) v = dist(rng);
    for (auto &v : C_init) v = dist(rng);
    reference_sgemm(A, B, C_init, want, M, N, K, s.alpha, s.beta);

    const float tol = 2e-5f * K;
    const float *pA = A.data(), *pB = B.data();
    const float al = s.alpha, be = s.beta;

    auto run = [&](const char *name, dim3 grid, dim3 block, auto fn) {
      got = C_init;
      float *pC = got.data();
      cuda_cpu::launch(grid, block, [=] { fn(pA, pB, pC, M, N, K, al, be); });
      check(name, got, want, M, N, tol);
    };

    run("01 naive", dim3((M + 31) / 32, (N + 31) / 32), dim3(32, 32),
        sgemm_naive_kernel);
    run("02 coalesced", dim3((N + 31) / 32, (M + 31) / 32), dim3(32, 32),
        sgemm_coalesced_kernel);
    run("03 smem tiled", dim3((N + 31) / 32, (M + 31) / 32), dim3(32 * 32),
        sgemm_smem_kernel<32>);
    run("04 blocktile 1d", dim3((N + 63) / 64, (M + 63) / 64), dim3(64 * 64 / 8),
        sgemm_blocktile_1d_kernel<64, 64, 8, 8>);
    run("05 blocktile 2d", dim3((N + 127) / 128, (M + 127) / 128),
        dim3(128 * 128 / 64), sgemm_blocktile_2d_kernel<128, 128, 8, 8, 8>);

    if (M % 128 == 0 && N % 128 == 0 && K % 8 == 0) {
      run("06 vectorized", dim3(N / 128, M / 128), dim3(128 * 128 / 64),
          sgemm_vectorized_kernel<128, 128, 8, 8, 8>);
    } else {
      std::printf("  %-22s skipped (falls back to stage 5 on ragged shapes)\n",
                  "06 vectorized");
    }
  }

  test_fused();

  std::printf("\n%s  (%d failure%s)\n", g_failures ? "TESTS FAILED" : "ALL TESTS PASSED",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures ? 1 : 0;
}
