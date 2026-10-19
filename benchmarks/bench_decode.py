"""Decode (M=1) comparison with effective bandwidth accounting."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.torch_ref import quantize_int4_symmetric
from benchmarks.bench_common import bench_fn, gbps, print_table, save_results
from kernels.dequant_gemm_w4a16 import dequant_gemm_v2
from kernels.dequant_int4 import dequant_int4_kernel
from kernels.gemv_w4a16 import gemv_w4a16


CASES = [(4096, 4096), (11008, 4096), (4096, 11008), (8192, 8192)]


def run(cases=CASES):
    rows = []
    for N, K in cases:
        torch.manual_seed(0)
        a = (torch.randn(K, device="cuda") * 0.1).half()
        w = (torch.randn(N, K, device="cuda") * 0.02).half()
        q, s = quantize_int4_symmetric(w, group_size=128)
        w_deq = torch.empty(N, K, device="cuda", dtype=torch.float16)
        dequant_kernel = dequant_int4_kernel(N, K, group_size=128)
        dequant_kernel(q, s, w_deq)
        packed_bytes = N * (K // 2) + N * (K // 128) * 2
        row = {"day": 9, "N": N, "K": K, "group_size": 128, "weight_bytes": packed_bytes}

        c = torch.empty(N, device="cuda", dtype=torch.float16)
        kg = gemv_w4a16(N, K)
        kg(a, q, s, c)
        ref = (w_deq.float() @ a.float()).half()
        rel = ((c.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)).item()
        row["gemv_rel_err"] = rel
        row["gemv_ms"] = round(float(bench_fn(lambda: kg(a, q, s, c))), 5)
        row["gemv_gbps"] = round(gbps(packed_bytes + K * 2 + N * 2, row["gemv_ms"]), 3)
        row["gemv_util_pct"] = round(row["gemv_gbps"] / 2039 * 100, 2)

        a2 = a.unsqueeze(0).contiguous()
        c2 = torch.empty(1, N, device="cuda", dtype=torch.float16)
        k2 = dequant_gemm_v2(1, N, K)
        k2(a2, q, s, c2)
        row["v2M1_ms"] = round(float(bench_fn(lambda: k2(a2, q, s, c2))), 5)
        row["v2M1_gbps"] = round(gbps(packed_bytes, row["v2M1_ms"]), 3)

        row["two_step_ms"] = round(float(bench_fn(lambda: (dequant_kernel(q, s, w_deq), torch.mv(w_deq, a)))), 5)
        row["cublas_ms"] = round(float(bench_fn(lambda: torch.mv(w_deq, a))), 5)
        row["cublas_gbps"] = round(gbps(N * K * 2, row["cublas_ms"]), 3)
        rows.append(row)
        print(f"done {N}x{K}")

    print_table(
        [[r["N"], r["K"], r["gemv_ms"] * 1000, r["gemv_gbps"], r["gemv_util_pct"],
          r["v2M1_ms"] * 1000, r["two_step_ms"] * 1000, r["cublas_ms"] * 1000, r["cublas_gbps"]]
         for r in rows],
        ["N", "K", "GEMV us", "GEMV GB/s", "util %", "v2 us", "two-step us", "cuBLAS us", "cuBLAS GB/s"],
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default="day09_decode.json")
    args = parser.parse_args()
    rows = run(CASES[:2] if args.quick else CASES)
    save_results(args.output, rows)


if __name__ == "__main__":
    main()
