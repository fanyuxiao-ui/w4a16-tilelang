#include <torch/extension.h>

torch::Tensor dequant_gemm_cuda(
    torch::Tensor a, torch::Tensor q, torch::Tensor scales, int64_t group_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dequant_gemm", &dequant_gemm_cuda, "W4A16 dequant GEMM (CUDA)");
}
