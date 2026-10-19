"""Day 6 one-variable-at-a-time and block/stage sweep."""

import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.torch_ref import dequantize_int4_symmetric, quantize_int4_symmetric
from benchmarks.bench_common import bench_fn, save_results, tflops_gemm
from kernels.dequant_gemm_w4a16 import dequant_gemm_v2


def run(M=128, N=1024, K=1024):
    torch.manual_seed(0)
    a = (torch.randn(M, K, device="cuda") * 0.1).half()
    w = (torch.randn(N, K, device="cuda") * 0.02).half()
    q, s = quantize_int4_symmetric(w, 128)
    ref = (a.float() @ dequantize_int4_symmetric(q, s, 128).t()).half()
    configs = [
        {"block_M": 64, "block_N": 64, "block_K": 128, "num_stages": 1, "threads": 128},
        {"block_M": 64, "block_N": 64, "block_K": 128, "num_stages": 2, "threads": 128},
        {"block_M": 64, "block_N": 128, "block_K": 128, "num_stages": 2, "threads": 128},
        {"block_M": 128, "block_N": 128, "block_K": 128, "num_stages": 2, "threads": 128},
        {"block_M": 64, "block_N": 128, "block_K": 256, "num_stages": 1, "threads": 128},
        {"block_M": 64, "block_N": 128, "block_K": 256, "num_stages": 2, "threads": 128},
        {"block_M": 128, "block_N": 128, "block_K": 128, "num_stages": 2, "threads": 256},
    ]
    rows = []
    for config in configs:
        row = dict(config, M=M, N=N, K=K)
        try:
            kernel = dequant_gemm_v2(M, N, K, **config)
            out = torch.empty_like(ref)
            kernel(a, q, s, out)
            rel = ((out.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)).item()
            row["rel_err"] = rel
            if rel > 1e-2:
                row["status"] = "incorrect"
            else:
                ms = float(bench_fn(lambda: kernel(a, q, s, out)))
                row["status"] = "passed"
                row["ms"] = round(ms, 5)
                row["tflops"] = round(tflops_gemm(M, N, K, ms), 4)
        except Exception as exc:
            row["status"] = "compile_error"
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        print(row)
        rows.append(row)
    save_results("day06_v2_sweep.json", rows)
    return rows


if __name__ == "__main__":
    run()
