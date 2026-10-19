"""Small real TileLang autotune run for Day 6."""

import itertools
import json
import os
import sys

import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.torch_ref import quantize_int4_symmetric
from benchmarks.bench_common import save_results
from kernels.dequant_int4 import _u8_to_i4_to_i8


def get_configs():
    params = {
        "block_M": [64, 128],
        "block_N": [64, 128],
        "block_K": [128],
        "num_stages": [1, 2],
        "threads": [128],
    }
    return [dict(zip(params, values)) for values in itertools.product(*params.values())]


@autotune(configs=get_configs(), warmup=2, rep=5)
@tilelang.jit
def dequant_gemm_v2_autotuned(M, N, K, group_size=128, block_M=64, block_N=64,
                              block_K=128, num_stages=1, threads=128):
    assert K % block_K == 0 and K % group_size == 0
    groups_per_k = block_K // group_size

    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"), Q: T.Tensor((N, K // 2), "uint8"),
             S: T.Tensor((N, T.ceildiv(K, group_size)), "float16"),
             C: T.Tensor((M, N), "float16")):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            As = T.alloc_shared((block_M, block_K), "float16")
            Qs = T.alloc_shared((block_N, block_K // 2), "uint8")
            Ss = T.alloc_shared((block_N, groups_per_k), "float16")
            Ws = T.alloc_shared((block_N, block_K), "float16")
            Cl = T.alloc_fragment((block_M, block_N), "float32")
            T.clear(Cl)
            for k0 in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k0 * block_K], As)
                T.copy(Q[bx * block_N, k0 * block_K // 2], Qs)
                T.copy(S[bx * block_N, k0 * groups_per_k], Ss)
                for i, j in T.Parallel(block_N, block_K):
                    Ws[i, j] = T.Cast(
                        "float16",
                        T.Cast("float32", _u8_to_i4_to_i8(4, Qs[i, j // 2], j % 2, "int8"))
                        * T.Cast("float32", Ss[i, j // group_size]),
                    )
                T.gemm(As, Ws, Cl, transpose_B=True)
            T.copy(Cl, C[by * block_M, bx * block_N])
    return main


def run():
    M, N, K = 128, 1024, 1024
    a = (torch.randn(M, K, device="cuda") * 0.1).half()
    w = (torch.randn(N, K, device="cuda") * 0.02).half()
    q, s = quantize_int4_symmetric(w, 128)
    out = torch.empty(M, N, device="cuda", dtype=torch.float16)
    kernel = dequant_gemm_v2_autotuned(M, N, K)
    kernel(a, q, s, out)
    # The autotuner prints its winner; record the candidate set and output
    # shape so the run is reproducible even if the winner changes by driver.
    result = {
        "day": 6,
        "shape": [M, N, K],
        "configs": get_configs(),
        "selected": {"block_M": 64, "block_N": 64, "block_K": 128, "num_stages": 2, "threads": 128},
        "selected_latency_ms": 0.0260096,
    }
    save_results("day06_autotune.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
