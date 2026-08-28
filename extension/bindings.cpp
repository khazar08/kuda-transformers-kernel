#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../kernels/launchers.h"

#define CHECK_INPUT(x)                                                         \
  TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");                     \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");                  \
  TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")

using LauncherFn = void (*)(const float *, const float *, float *, int, int,
                            int, float, float, cudaStream_t);

static const std::unordered_map<std::string, LauncherFn> &registry() {
  static const std::unordered_map<std::string, LauncherFn> m = {
      {"cublas", &launch_sgemm_cublas},
      {"naive", &launch_sgemm_naive},
      {"coalesced", &launch_sgemm_coalesced},
      {"smem", &launch_sgemm_smem},
      {"blocktile_1d", &launch_sgemm_blocktile_1d},
      {"blocktile_2d", &launch_sgemm_blocktile_2d},
      {"vectorized", &launch_sgemm_vectorized},
  };
  return m;
}

std::vector<std::string> sgemm_stages() {
  return {"naive",        "coalesced",    "smem",  "blocktile_1d",
          "blocktile_2d", "vectorized",   "cublas"};
}

torch::Tensor sgemm(torch::Tensor A, torch::Tensor B, const std::string &stage,
                    double alpha, double beta,
                    std::optional<torch::Tensor> out) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  TORCH_CHECK(B.dim() == 2, "B must be 2D, got ", B.dim(), "D");
  TORCH_CHECK(A.dim() >= 2, "A must be at least 2D, got ", A.dim(), "D");

  const int64_t K = B.size(0);
  const int64_t N = B.size(1);
  TORCH_CHECK(A.size(-1) == K, "shape mismatch: A last dim ", A.size(-1),
              " != B first dim ", K);

  const int64_t M = A.numel() / K;

  TORCH_CHECK(M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX,
              "dimensions exceed the int32 indexing used by these kernels");

  auto it = registry().find(stage);
  TORCH_CHECK(it != registry().end(), "unknown sgemm stage '", stage, "'");

  std::vector<int64_t> out_shape(A.sizes().begin(), A.sizes().end() - 1);
  out_shape.push_back(N);

  torch::Tensor C;
  if (out.has_value()) {
    C = out.value();
    CHECK_INPUT(C);
    TORCH_CHECK(C.numel() == M * N, "out has wrong number of elements");
  } else {
    TORCH_CHECK(beta == 0.0,
                "beta != 0 requires an explicit `out` tensor to accumulate into");
    C = torch::empty(out_shape, A.options());
  }

  it->second(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
             (int)M, (int)N, (int)K, (float)alpha, (float)beta,
  // Torch's current stream, not the legacy default: otherwise CUDA-event
  // timings would measure stream synchronization rather than the kernel.
             at::cuda::getCurrentCUDAStream());

  return C.view(out_shape);
}

torch::Tensor softmax(torch::Tensor X, bool fused) {
  CHECK_INPUT(X);
  const int64_t cols = X.size(-1);
  const int64_t rows = X.numel() / cols;
  TORCH_CHECK(rows <= INT32_MAX && cols <= INT32_MAX, "dimensions too large");

  auto Y = torch::empty_like(X);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (fused) {
    launch_softmax_fused(X.data_ptr<float>(), Y.data_ptr<float>(), (int)rows,
                         (int)cols, stream);
  } else {
    auto row_max = torch::empty({rows}, X.options());
    auto row_sum = torch::empty({rows}, X.options());
    launch_softmax_naive(X.data_ptr<float>(), Y.data_ptr<float>(),
                         row_max.data_ptr<float>(), row_sum.data_ptr<float>(),
                         (int)rows, (int)cols, stream);
  }
  return Y;
}

torch::Tensor layernorm(torch::Tensor X, std::optional<torch::Tensor> gamma,
                        std::optional<torch::Tensor> beta, double eps) {
  CHECK_INPUT(X);
  const int64_t cols = X.size(-1);
  const int64_t rows = X.numel() / cols;

  const float *g = nullptr;
  const float *b = nullptr;
  if (gamma.has_value()) {
    CHECK_INPUT(gamma.value());
    TORCH_CHECK(gamma.value().numel() == cols, "gamma must have ", cols,
                " elements");
    g = gamma.value().data_ptr<float>();
  }
  if (beta.has_value()) {
    CHECK_INPUT(beta.value());
    TORCH_CHECK(beta.value().numel() == cols, "beta must have ", cols,
                " elements");
    b = beta.value().data_ptr<float>();
  }

  auto Y = torch::empty_like(X);
  launch_layernorm_fused(X.data_ptr<float>(), g, b, Y.data_ptr<float>(),
                         (int)rows, (int)cols, (float)eps,
                         at::cuda::getCurrentCUDAStream());
  return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sgemm", &sgemm, "SGEMM, selectable optimization stage", py::arg("A"),
        py::arg("B"), py::arg("stage") = "vectorized", py::arg("alpha") = 1.0,
        py::arg("beta") = 0.0, py::arg("out") = std::nullopt);
  m.def("sgemm_stages", &sgemm_stages, "ladder stage names, in order");
  m.def("softmax", &softmax, "row-wise softmax", py::arg("X"),
        py::arg("fused") = true);
  m.def("layernorm", &layernorm, "fused LayerNorm", py::arg("X"),
        py::arg("gamma") = std::nullopt, py::arg("beta") = std::nullopt,
        py::arg("eps") = 1e-5);
}
