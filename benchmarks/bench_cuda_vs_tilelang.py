"""Day 7: hand-written CUDA versus TileLang v2 on identical inputs."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.cuda.load_dequant_gemm import dequant_gemm_cuda
from baseline.torch_ref import dequantize_int4_symmetric, quantize_int4_symmetric
from benchmarks.bench_common import bench_fn, save_results, tflops_gemm
from kernels.dequant_gemm_w4a16 import dequant_gemm_v2


CASES = [(1, 4096, 4096), (16, 4096, 4096), (512, 4096, 4096), (1024, 4096, 4096), (4096, 4096, 4096)]


def run(cases=CASES):
    rows = []
    for M, N, K in cases:
        torch.manual_seed(0)
        a = (torch.randn(M, K, device="cuda") * 0.1).half()
        w = (torch.randn(N, K, device="cuda") * 0.02).half()
        q, s = quantize_int4_symmetric(w, 128)
        ref = (a.float() @ dequantize_int4_symmetric(q, s, 128).t()).half()

        out_cuda = dequant_gemm_cuda(a, q, s, 128)
        cuda_rel = ((out_cuda.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)).item()
        kt = dequant_gemm_v2(M, N, K)
        out_tl = torch.empty_like(ref)
        kt(a, q, s, out_tl)
        tl_rel = ((out_tl.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)).item()

        cuda_ms = float(bench_fn(lambda: dequant_gemm_cuda(a, q, s, 128)))
        tl_ms = float(bench_fn(lambda: kt(a, q, s, out_tl)))
        row = {
            "day": 7, "M": M, "N": N, "K": K, "cuda_ms": round(cuda_ms, 5),
            "tilelang_v2_ms": round(tl_ms, 5),
            "cuda_tflops": round(tflops_gemm(M, N, K, cuda_ms), 4),
            "tilelang_v2_tflops": round(tflops_gemm(M, N, K, tl_ms), 4),
            "tilelang_speedup": round(cuda_ms / tl_ms, 3),
            "cuda_rel_err": cuda_rel, "tilelang_rel_err": tl_rel,
        }
        rows.append(row)
        print(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default="day07_cuda_vs_tilelang.json")
    args = parser.parse_args()
    rows = run(CASES[:3] if args.quick else CASES)
    save_results(args.output, rows)


if __name__ == "__main__":
    main()
