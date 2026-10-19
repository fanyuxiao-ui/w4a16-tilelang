#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int BM = 16;
constexpr int BN = 16;
constexpr int BK = 32;

__global__ void dequant_gemm_kernel(
    const half* __restrict__ a,
    const uint8_t* __restrict__ q,
    const half* __restrict__ scales,
    half* __restrict__ c,
    int m, int n, int k, int groups, int group_size) {
  __shared__ half a_tile[BM][BK];
  __shared__ half w_tile[BN][BK];

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * BM + threadIdx.y;
  const int col = blockIdx.x * BN + threadIdx.x;
  float acc = 0.0f;

  for (int k0 = 0; k0 < k; k0 += BK) {
    for (int idx = tid; idx < BM * BK; idx += BM * BN) {
      const int r = idx / BK;
      const int kk = idx % BK;
      const int gr = blockIdx.y * BM + r;
      a_tile[r][kk] = (gr < m && k0 + kk < k)
          ? a[gr * k + k0 + kk]
          : __float2half(0.0f);
    }
    // Each byte contains two signed int4 values.  The scale is shared by
    // all values in a group, and is applied before storing W in shared.
    for (int idx = tid; idx < BN * BK; idx += BM * BN) {
      const int r = idx / BK;
      const int kk = idx % BK;
      const int gr = blockIdx.x * BN + r;
      const int gk = k0 + kk;
      half out = __float2half(0.0f);
      if (gr < n && gk < k) {
        const uint8_t packed = q[gr * (k / 2) + gk / 2];
        int v = (packed >> ((gk & 1) * 4)) & 0xF;
        v = (v ^ 8) - 8;
        const float s = __half2float(scales[gr * groups + gk / group_size]);
        out = __float2half(static_cast<float>(v) * s);
      }
      w_tile[r][kk] = out;
    }
    __syncthreads();

    if (row < m && col < n) {
      #pragma unroll
      for (int kk = 0; kk < BK; ++kk) {
        acc += __half2float(a_tile[threadIdx.y][kk]) *
               __half2float(w_tile[threadIdx.x][kk]);
      }
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    c[row * n + col] = __float2half(acc);
  }
}
}  // namespace

torch::Tensor dequant_gemm_cuda(
    torch::Tensor a, torch::Tensor q, torch::Tensor scales, int64_t group_size) {
  TORCH_CHECK(a.is_cuda() && q.is_cuda() && scales.is_cuda(), "all tensors must be CUDA");
  TORCH_CHECK(a.scalar_type() == torch::kFloat16, "A must be float16");
  TORCH_CHECK(q.scalar_type() == torch::kUInt8, "Q must be uint8");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16, "scales must be float16");
  TORCH_CHECK(a.dim() == 2 && q.dim() == 2 && scales.dim() == 2, "expected rank-2 tensors");
  TORCH_CHECK(a.is_contiguous() && q.is_contiguous() && scales.is_contiguous(), "tensors must be contiguous");
  const int64_t m = a.size(0), k = a.size(1), n = q.size(0);
  TORCH_CHECK(q.size(1) == k / 2, "Q shape does not match A");
  TORCH_CHECK(k % 2 == 0 && k % group_size == 0, "K must be divisible by 2 and group_size");
  TORCH_CHECK(scales.size(1) == k / group_size, "scale shape does not match K/group_size");

  auto c = torch::empty({m, n}, a.options());
  const dim3 block(BN, BM);
  const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
  dequant_gemm_kernel<<<grid, block>>>(
      reinterpret_cast<const half*>(a.data_ptr<at::Half>()), q.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<half*>(c.data_ptr<at::Half>()), static_cast<int>(m), static_cast<int>(n),
      static_cast<int>(k), static_cast<int>(k / group_size), static_cast<int>(group_size));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}
