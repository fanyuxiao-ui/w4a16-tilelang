"""Day 4 验收：TileLang INT4 反量化与 torch 黄金参考逐位对齐。"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.torch_ref import (  # noqa: E402
    dequantize_int4_symmetric,
    quantize_int4_symmetric,
)
from kernels.dequant_int4 import dequant_int4_kernel  # noqa: E402


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Day 4 kernel test requires CUDA")


@pytest.mark.parametrize(
    "N,K,group_size",
    [
        (256, 1024, 128),
        (4096, 4096, 128),
        (11008, 4096, 128),
        (4096, 11008, 64),
    ],
)
def test_dequant_matches_golden(N, K, group_size):
    torch.manual_seed(0)
    w = (torch.randn(N, K, device="cuda") * 0.02).half()
    packed, scales = quantize_int4_symmetric(w, group_size=group_size)
    w_ref = dequantize_int4_symmetric(packed, scales, group_size=group_size).half()

    w_tl = torch.empty(N, K, device="cuda", dtype=torch.float16)
    kernel = dequant_int4_kernel(N, K, group_size=group_size)
    kernel(packed, scales, w_tl)
    torch.cuda.synchronize()

    max_err = (w_tl.float() - w_ref.float()).abs().max().item()
    print(f"[{N},{K},g={group_size}] max_err={max_err:.6f}")
    assert max_err < 1e-3
