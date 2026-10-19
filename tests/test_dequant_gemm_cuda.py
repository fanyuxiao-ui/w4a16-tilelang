import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.cuda.load_dequant_gemm import dequant_gemm_cuda
from baseline.torch_ref import dequantize_int4_symmetric, quantize_int4_symmetric


def check(M, N, K, group_size=128):
    torch.manual_seed(0)
    a = (torch.randn(M, K, device="cuda") * 0.1).half()
    w = (torch.randn(N, K, device="cuda") * 0.02).half()
    q, s = quantize_int4_symmetric(w, group_size)
    ref = (a.float() @ dequantize_int4_symmetric(q, s, group_size).t()).half()
    out = dequant_gemm_cuda(a, q, s, group_size)
    rel = ((out.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)).item()
    print(f"{M}x{N}x{K}: rel_mean={rel:.7f}")
    assert rel < 1e-2


def test_cuda_small():
    check(256, 512, 1024)


def test_cuda_medium():
    check(512, 1024, 1024)


def test_cuda_large():
    check(1024, 4096, 4096)


if __name__ == "__main__":
    for case in [(256, 512, 1024), (512, 1024, 1024), (1024, 4096, 4096)]:
        check(*case)
    print("cuda dequant gemm passed")
