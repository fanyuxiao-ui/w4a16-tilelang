"""Day 5: compare the fused W4A16 GEMM with two independent references."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.torch_ref import (
    dequantize_int4_symmetric,
    quantize_int4_symmetric,
)
from kernels.dequant_gemm_w4a16 import dequant_gemm_v1
from kernels.dequant_int4 import dequant_int4_kernel
from kernels.gemm_bf16 import gemm_bf16


def _compare(name, actual, reference):
    diff = (actual.float() - reference.float()).abs()
    ref_mean = reference.float().abs().mean().clamp_min(1e-6)
    rel_mean = (diff.mean() / ref_mean).item()
    max_abs = diff.max().item()
    print(f"  vs {name}: rel_mean={rel_mean:.6f}, max_abs={max_abs:.6f}")
    assert rel_mean < 0.01, f"{name} reference mismatch: {rel_mean:.6f}"


def run_case(M, N, K, group_size, block_M=64, block_N=64, block_K=128):
    torch.manual_seed(0)
    A = (torch.randn(M, K, device="cuda") * 0.1).half()
    W = (torch.randn(N, K, device="cuda") * 0.02).half()
    Q, S = quantize_int4_symmetric(W, group_size=group_size)

    W_dequant = dequantize_int4_symmetric(Q, S, group_size=group_size)
    ref_torch = (A.float() @ W_dequant.t()).half()

    W_dequant_h = torch.empty(N, K, device="cuda", dtype=torch.float16)
    dequant_int4_kernel(N, K, group_size=group_size)(Q, S, W_dequant_h)
    ref_two_kernel = torch.empty(M, N, device="cuda", dtype=torch.float16)
    gemm_bf16(
        M,
        N,
        K,
        block_M=block_M,
        block_N=block_N,
        block_K=64,
    )(A, W_dequant_h, ref_two_kernel)

    actual = torch.empty(M, N, device="cuda", dtype=torch.float16)
    dequant_gemm_v1(
        M,
        N,
        K,
        group_size=group_size,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
    )(A, Q, S, actual)
    torch.cuda.synchronize()

    print(f"\n=== M={M}, N={N}, K={K}, group_size={group_size} ===")
    _compare("torch", actual, ref_torch)
    _compare("two-kernel", actual, ref_two_kernel)


def test_dequant_gemm_small():
    run_case(256, 512, 1024, 128)


def test_dequant_gemm_square():
    run_case(1024, 4096, 4096, 128)


def test_dequant_gemm_group64():
    run_case(256, 4096, 4096, 64)


def test_dequant_gemm_ffn():
    run_case(512, 11008, 4096, 128)


if __name__ == "__main__":
    for case in [
        (256, 512, 1024, 128),
        (1024, 4096, 4096, 128),
        (256, 4096, 4096, 64),
        (512, 11008, 4096, 128),
    ]:
        run_case(*case)
    print("\nall dequant GEMM correctness cases passed")
