"""Day 4 步骤 2：INT4 反量化 kernel 带宽基准。"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from benchmarks.bench_common import bench_fn, gbps, save_results  # noqa: E402
from kernels.dequant_int4 import dequant_int4_kernel  # noqa: E402


A800_BW_GBPS = 2039.0
SHAPES = [(4096, 4096), (11008, 4096), (4096, 11008)]


def run(N, K, group_size=128):
    if not torch.cuda.is_available():
        raise RuntimeError("bench_dequant.py requires CUDA")
    torch.manual_seed(0)
    packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    scales = (torch.rand(N, K // group_size, device="cuda") * 0.01 + 1e-5).half()
    output = torch.empty(N, K, dtype=torch.float16, device="cuda")
    kernel = dequant_int4_kernel(N, K, group_size=group_size)
    kernel(packed, scales, output)
    torch.cuda.synchronize()

    ms = bench_fn(lambda: kernel(packed, scales, output))
    bytes_moved = packed.numel() + scales.numel() * scales.element_size() + output.numel() * output.element_size()
    bw = gbps(bytes_moved, ms)
    row = {
        "N": N,
        "K": K,
        "group_size": group_size,
        "ms": round(ms, 4),
        "bytes": bytes_moved,
        "gbps": round(bw, 1),
        "util": round(bw / A800_BW_GBPS, 4),
        "util_percent": round(bw / A800_BW_GBPS * 100, 1),
    }
    print(f"[{N}x{K} g={group_size}] {ms:.4f} ms, {bw:.1f} GB/s, "
          f"利用率 {row['util_percent']:.1f}%")
    return row


def main():
    rows = [run(N, K) for N, K in SHAPES]
    save_results("day04_dequant_bw.json", rows, outdir=str(ROOT / "benchmarks/results"))


if __name__ == "__main__":
    main()
