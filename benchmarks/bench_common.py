"""统一的计时 / 指标计算 / 结果保存工具。

从 Day 2 起，仓库里所有性能数字都必须由脚本产出、落成 JSON：
手敲的、凭记忆的数字一律不算数。
"""
import json
import os

import torch
from tilelang.profiler import do_bench

# Bench 口径：warmup/rep 都是毫秒级总时长（triton 风格）。
WARMUP_MS = 25
REP_MS = 100


def bench_fn(fn, warmup=WARMUP_MS, rep=REP_MS):
    """返回单次调用延迟（毫秒）。"""
    return do_bench(fn, warmup=warmup, rep=rep)


def tflops_gemm(m, n, k, ms):
    """GEMM 的 TFLOPS = 2*M*N*K / 时间(s) / 1e12"""
    return 2.0 * m * n * k / (ms * 1e-3) / 1e12


def gbps(bytes_moved, ms):
    """等效带宽 GB/s = 字节数 / 时间(s) / 1e9"""
    return bytes_moved / (ms * 1e-3) / 1e9


def save_results(name, rows, outdir="benchmarks/results"):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n-> saved: {path}")
    return path


def print_table(rows, headers):
    """rows 里缺数据的位置用 None，打印成 '-' 而不是 'None'（失败列看起来才像空的）。"""
    def cell(v):
        return "-" if v is None else v

    widths = [max(len(str(h)), max(len(str(cell(r[i]))) for r in rows)) for i, h in enumerate(headers)]
    line = " | ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(str(cell(c)).ljust(w) for c, w in zip(r, widths)))
