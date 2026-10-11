"""打印 gemm_bf16 生成的 CUDA 源码并存档（Day 1 步骤 4）。"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernels.gemm_bf16 import gemm_bf16  # noqa: E402

M = N = K = 4096
kernel = gemm_bf16(M, N, K, block_M=128, block_N=128, block_K=64, num_stages=3)
src = kernel.get_kernel_source()

out = ROOT / "assets" / "gemm_bf16_sm80.cu"
out.write_text(src)
print("生成代码长度:", len(src), "字符")
print("已存档:", out)
print("行数:", len(src.splitlines()))
