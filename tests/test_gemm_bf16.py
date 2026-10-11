import torch

from kernels.gemm_bf16 import gemm_bf16


def _check(M, N, K, **kwargs):
    kernel = gemm_bf16(M, N, K, **kwargs)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)
    kernel(a, b, c)
    torch.testing.assert_close(c, a @ b.t(), rtol=1e-2, atol=1e-2)


def test_gemm_bf16_basic():
    _check(1024, 1024, 1024)
    print("gemm_bf16 passed")


def test_gemm_bf16_nonsquare():
    """非方阵：B 布局写反这类错误在方阵下会静默通过，这里会露馅。"""
    for M, N, K in [(256, 512, 1024), (512, 256, 1024), (1024, 2048, 512)]:
        _check(M, N, K)
    print("gemm_bf16 nonsquare passed")


if __name__ == "__main__":
    test_gemm_bf16_basic()
    test_gemm_bf16_nonsquare()
