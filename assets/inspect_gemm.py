"""打印 gemm_bf16 生成的 CUDA 源码并存档。

- Day 1 步骤 4：`block_M=128, block_N=128, block_K=64, stages=3, threads=128` → gemm_bf16_sm80.cu
- Day 2 步骤 5：最优配置（Day 1 扫描的 188.2 TFLOPS 那组）→ gemm_bf16_best.cu

无 GPU 也能跑：显式指定 Target 绕过 auto_detect_target()（后者必须有真卡）。
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tilelang  # noqa: F401,E402  —— 先 import tilelang 才会注册内置 tvm
from tvm.target import Target  # noqa: E402

from kernels.gemm_bf16 import gemm_bf16  # noqa: E402

M = N = K = 4096

CASES = [
    # (输出文件, 说明, 配置)
    ("gemm_bf16_sm80.cu", "Day 1 步骤 4 初始配置",
     dict(block_M=128, block_N=128, block_K=64, num_stages=3)),
    ("gemm_bf16_best.cu", "Day 2 步骤 5 最优配置（188.2 TFLOPS）",
     dict(block_M=128, block_N=256, block_K=64, num_stages=3, threads=256)),
]


def main():
    for name, note, cfg in CASES:
        with Target({"kind": "cuda", "arch": "sm_80"}):
            kernel = gemm_bf16(M, N, K, **cfg)
        src = kernel.get_kernel_source()
        out = ROOT / "assets" / name
        # 原样存档，不加任何头注释：results.md 里引用了这些文件的具体行号
        # （Day 1 记的 L33/L49/L60/L76/L25/L68），插行会让已有记录全部失准。
        # 配置与 shape 的对应关系记在 docs/results.md 里。
        out.write_text(src)
        print(f"{name:<24} {len(src):>6} 字符 {len(src.splitlines()):>4} 行  <- {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
