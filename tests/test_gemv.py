import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.torch_ref import dequantize_int4_symmetric, quantize_int4_symmetric
from kernels.gemv_w4a16 import gemv_w4a16


def check(N, K, group_size=128):
    torch.manual_seed(0)
    a = (torch.randn(K, device="cuda") * 0.1).half()
    w = (torch.randn(N, K, device="cuda") * 0.02).half()
    q, s = quantize_int4_symmetric(w, group_size)
    ref = (dequantize_int4_symmetric(q, s, group_size) @ a.float()).half()
    out = torch.empty(N, device="cuda", dtype=torch.float16)
    gemv_w4a16(N, K, group_size)(a, q, s, out)
    rel = ((out.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)).item()
    print(f"{N}x{K}: rel_mean={rel:.7f}")
    assert rel < 1e-3


def test_gemv_4096():
    check(4096, 4096)


def test_gemv_ffn_up():
    check(11008, 4096)


def test_gemv_ffn_down():
    check(4096, 11008)


if __name__ == "__main__":
    for case in [(4096, 4096), (11008, 4096), (4096, 11008)]:
        check(*case)
    print("gemv passed")
