import os

import torch
from torch.utils.cpp_extension import load


_DIR = os.path.dirname(os.path.abspath(__file__))
_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load(
            name="dequant_gemm_cuda_day07",
            sources=[
                os.path.join(_DIR, "dequant_gemm.cu"),
                os.path.join(_DIR, "dequant_gemm_binding.cpp"),
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_80", "-lineinfo"],
            extra_cflags=["-O3"],
            verbose=True,
        )
    return _ext


def dequant_gemm_cuda(a: torch.Tensor, q: torch.Tensor, scales: torch.Tensor, group_size: int = 128):
    return _get_ext().dequant_gemm(a, q, scales, group_size)
