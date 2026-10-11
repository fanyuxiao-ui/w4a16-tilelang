"""Day 1 步骤 5：gemm_bf16 第一轮参数扫描（4096³ fp16）。

只改参数、不改代码，观察 TFLOPS 变化；顺带打一行 cuBLAS 参考值。
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from tilelang.profiler import do_bench  # noqa: E402

from kernels.gemm_bf16 import gemm_bf16  # noqa: E402

M = N = K = 4096
FLOPS = 2 * M * N * K
# (block_M, block_N, block_K, num_stages, threads)
# tile 面积 ÷ 线程数 = 每线程的累加器寄存器数，必须 << 255，否则溢出到 local memory。
# 例：128x256 配 128 线程 → 256 个寄存器/线程，实测只剩 43 TFLOPS；改 256 线程 → 190 TFLOPS。
CONFIGS = [
    (128, 128, 32, 3, 128),
    (128, 128, 64, 3, 128),
    (128, 256, 64, 3, 256),
    (64, 128, 64, 4, 128),
]


def tflops(ms):
    return FLOPS / ms * 1e-9


a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(N, K, device="cuda", dtype=torch.float16)
bt = b.t()
c = torch.empty(M, N, device="cuda", dtype=torch.float16)

# 校验基准必须是 fp64 真值，不能拿 fp16 的 cuBLAS 结果当参考：
# K=4096 时输出量级 ~50、峰值 ~390，fp16 输出在 [256,512) 的 1 ULP 就是 0.25，
# 两套实现各自把结果舍入到相邻 fp16 值时，互相差异会撞破 rtol/atol=1e-2 的容差，
# 而实测 tilelang 反而比 cuBLAS 更贴近 fp64 真值（max|err| 0.127 vs 0.182）。
ref64 = (a.double() @ bt.double()).float()

print(f"4096³ fp16  |  峰值 312 TFLOPS  |  {'config':<26} {'ms':>8} {'TFLOPS':>8}  {'max|err|':>9} 校验")
for bm, bn, bk, st, th in CONFIGS:
    tag = f"bm{bm} bn{bn} bk{bk} st{st} th{th}"
    try:
        kernel = gemm_bf16(M, N, K, block_M=bm, block_N=bn, block_K=bk, num_stages=st, threads=th)
        kernel(a, b, c)
        max_err = (c.float() - ref64).abs().max().item()
        ok = torch.allclose(c.float(), ref64, rtol=1e-2, atol=1e-2)
        ms = do_bench(lambda: kernel(a, b, c), warmup=25, rep=100)
        print(f"{'':<14}{tag:<26} {ms:>8.3f} {tflops(ms):>8.1f}  {max_err:>9.4f} {'✅' if ok else '❌ 偏离 fp64 真值'}")
    except Exception as e:
        print(f"{'':<14}{tag:<26} FAILED {type(e).__name__}: {str(e)[:120]}")

ms = do_bench(lambda: a @ bt, warmup=25, rep=100)
print(f"{'':<14}{'cuBLAS (仅性能参考)':<26} {ms:>8.3f} {tflops(ms):>8.1f}")
