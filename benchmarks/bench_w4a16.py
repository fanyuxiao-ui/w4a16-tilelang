"""Unified W4A16 benchmark for Day 6 and later comparisons.

All implementations use the same packed Q, fp16 scales, input tensor, and
CUDA event timing.  Compile failures are retained in the JSON instead of
being silently turned into fabricated latency values.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.torch_ref import quantize_int4_symmetric
from benchmarks.bench_common import bench_fn, print_table, save_results, tflops_gemm
from kernels.dequant_gemm_w4a16 import dequant_gemm_v1, dequant_gemm_v2
from kernels.dequant_int4 import dequant_int4_kernel


SHAPES = [
    (1, 4096, 4096, "decode"),
    (16, 4096, 4096, "decode"),
    (128, 4096, 4096, "small-batch"),
    (512, 4096, 4096, "prefill"),
    (1024, 4096, 4096, "prefill"),
    (4096, 4096, 4096, "prefill-big"),
    (1024, 11008, 4096, "ffn-up"),
    (1024, 4096, 11008, "ffn-down"),
]


def _measure(fn):
    return round(float(bench_fn(fn)), 5)


def run(shapes=SHAPES, group_size=128, include_swizzle=False):
    rows = []
    for M, N, K, tag in shapes:
        torch.manual_seed(0)
        a = (torch.randn(M, K, device="cuda") * 0.1).half()
        w = (torch.randn(N, K, device="cuda") * 0.02).half()
        q, s = quantize_int4_symmetric(w, group_size=group_size)
        w_deq = torch.empty(N, K, device="cuda", dtype=torch.float16)
        dequant_int4_kernel(N, K, group_size=group_size)(q, s, w_deq)
        c = torch.empty(M, N, device="cuda", dtype=torch.float16)
        row = {"day": 6, "M": M, "N": N, "K": K, "tag": tag, "group_size": group_size}

        try:
            k1 = dequant_gemm_v1(M, N, K, group_size=group_size)
            k1(a, q, s, c)
            row["v1_ms"] = _measure(lambda: k1(a, q, s, c))
            row["v1_tflops"] = round(tflops_gemm(M, N, K, row["v1_ms"]), 3)
        except Exception as exc:
            row["v1_error"] = f"{type(exc).__name__}: {exc}"[:300]

        try:
            k2 = dequant_gemm_v2(M, N, K, group_size=group_size)
            k2(a, q, s, c)
            row["v2_ms"] = _measure(lambda: k2(a, q, s, c))
            row["v2_tflops"] = round(tflops_gemm(M, N, K, row["v2_ms"]), 3)
        except Exception as exc:
            row["v2_error"] = f"{type(exc).__name__}: {exc}"[:300]

        if include_swizzle:
            try:
                ks = dequant_gemm_v2(M, N, K, group_size=group_size, use_swizzle=True)
                ks(a, q, s, c)
                row["v2_swizzle_ms"] = _measure(lambda: ks(a, q, s, c))
            except Exception as exc:
                row["v2_swizzle_error"] = f"{type(exc).__name__}: {exc}"[:300]

        try:
            two_step_kernel = dequant_int4_kernel(N, K, group_size=group_size)
            row["two_step_ms"] = _measure(lambda: (two_step_kernel(q, s, w_deq), torch.mm(a, w_deq.t())))
        except Exception as exc:
            row["two_step_error"] = f"{type(exc).__name__}: {exc}"[:300]

        try:
            row["cublas_fp16_ms"] = _measure(lambda: torch.mm(a, w_deq.t()))
            row["cublas_fp16_tflops"] = round(tflops_gemm(M, N, K, row["cublas_fp16_ms"]), 3)
        except Exception as exc:
            row["cublas_error"] = f"{type(exc).__name__}: {exc}"[:300]

        if row.get("v1_ms") and row.get("v2_ms"):
            row["v1_to_v2_speedup"] = round(row["v1_ms"] / row["v2_ms"], 3)
        if row.get("two_step_ms") and row.get("v2_ms"):
            row["v2_vs_two_step"] = round(row["two_step_ms"] / row["v2_ms"], 3)
        rows.append(row)
        print(f"done {tag} {M}x{N}x{K}")

    print_table(
        [[r["tag"], f'{r["M"]}x{r["N"]}x{r["K"]}', r.get("v1_ms"), r.get("v2_ms"),
          r.get("two_step_ms"), r.get("cublas_fp16_ms"), r.get("v2_vs_two_step")] for r in rows],
        ["scene", "shape", "v1(ms)", "v2(ms)", "two-step(ms)", "cuBLAS(ms)", "two/v2"],
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="run four representative shapes")
    parser.add_argument("--swizzle", action="store_true", help="also compile the block-swizzle variant")
    parser.add_argument("--output", default="day06_w4a16_all.json")
    args = parser.parse_args()
    shapes = [SHAPES[i] for i in (0, 1, 2, 3)] if args.quick else SHAPES
    rows = run(shapes, include_swizzle=args.swizzle)
    save_results(args.output, rows)


if __name__ == "__main__":
    main()
