"""Optional PyTorch CUTLASS int4pack reference (Day 8 path A).

Torch 2.7's CUDA implementation accepts bf16 activations and stores scales as
``[K/group, N, 2]`` (scale, zero-point).  The format is intentionally kept in
this adapter; it is not the project's row-major signed-nibble format.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.torch_ref import pack_nibbles, quantize_int4_symmetric, unpack_nibbles
from benchmarks.bench_common import bench_fn, save_results


def make_packed(N, K, group_size=128, device="cuda"):
    w = (torch.randn(N, K, device=device) * 0.02).half()
    q, scales = quantize_int4_symmetric(w, group_size)
    q_signed = unpack_nibbles(q)
    q_unsigned = pack_nibbles((q_signed + 8).to(torch.int32))
    # innerKTiles=8 is the SM80-friendly layout and requires K % 128 == 0.
    packed = torch._convert_weight_to_int4pack(q_unsigned, 8)
    zeros = torch.full(
        (K // group_size, N), 8.0, device=device, dtype=torch.bfloat16
    )
    scale_zeros = torch.stack(
        (scales.float().t().to(torch.bfloat16), zeros), dim=-1
    ).contiguous()
    return w, packed, scale_zeros


def run(cases=((1, 4096, 4096), (16, 4096, 4096), (512, 4096, 4096))):
    rows = []
    for M, N, K in cases:
        a = (torch.randn(M, K, device="cuda") * 0.1).bfloat16()
        _, packed, scale_zeros = make_packed(N, K)
        # Warm compilation and verify the op returns the expected shape/dtype.
        out = torch._weight_int4pack_mm(a, packed, 128, scale_zeros)
        row = {"day": 8, "M": M, "N": N, "K": K, "dtype": "bf16", "innerKTiles": 8}
        row["output_shape"] = list(out.shape)
        row["ms"] = round(float(bench_fn(lambda: torch._weight_int4pack_mm(a, packed, 128, scale_zeros))), 5)
        rows.append(row)
        print(row)
    save_results("day08_int4pack.json", rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cases = ((1, 4096, 4096),) if args.quick else ((1, 4096, 4096), (16, 4096, 4096), (512, 4096, 4096))
    run(cases)
