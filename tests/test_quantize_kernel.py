"""Optional Day 3 device quantizer check."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.torch_ref import (  # noqa: E402
    quantize_int4_symmetric,
    unpack_nibbles,
)
from kernels.quantize_int4 import quantize_int4_kernel  # noqa: E402


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="optional kernel requires CUDA")


def test_quantize_kernel_matches_torch():
    torch.manual_seed(0)
    N, K, group_size = 64, 256, 128
    w = (torch.randn(N, K, device="cuda") * 0.02).half()
    q_ref, s_ref = quantize_int4_symmetric(w, group_size)
    q = torch.empty_like(q_ref)
    s = torch.empty_like(s_ref)
    quantize_int4_kernel(N, K, group_size=group_size, blk_n=64)(w, q, s)
    torch.cuda.synchronize()
    # CUDA ``round`` and torch's CPU/GPU round can choose opposite ties-to-even
    # directions.  Compare decoded signed values, where a tie differs by at most
    # one quantization step; the packed bytes themselves may differ by 0x10.
    q_delta = (unpack_nibbles(q).to(torch.int16) - unpack_nibbles(q_ref).to(torch.int16)).abs()
    assert q_delta.max().item() <= 1
    torch.testing.assert_close(s, s_ref, rtol=0, atol=0)
