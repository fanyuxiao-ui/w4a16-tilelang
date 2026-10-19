"""Day 3 步骤 4：把 DeepSeek-V3.2-Exp 的 act_quant / fp8_gemm 离线编译成 CUDA 来读。

为什么生成代码而不是只读 tilelang 源码：`T.reduce_absmax` 的真实开销（跨线程归约走了
几条 shuffle）、`T.Pipelined(1, num_stages=2)` 到底生成了什么、epilogue 的 2xAcc 结构长什么样，
都只在生成代码里看得见。这是 Day 1 建立的方法，这里复用它。

无卡可跑（同 Day 2 的 --check-build）：显式 Target 绕过 auto_detect_target()。
输出到 /tmp/dsv32_cu/，**不入库**——生成的 CUDA 是 DeepSeek 上游 kernel 的编译产物，
notes 里只引用关键片段 + 上游行号。复现（refs-deepseek/v32 已就位）：
    python assets/inspect_dsv32.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DSV32 = ROOT.parent / "refs-deepseek" / "v32" / "inference"
OUT = Path("/tmp/dsv32_cu")

sys.path.insert(0, str(DSV32))
os.environ.setdefault("TILELANG_PRINT_ON_COMPILATION", "0")

import tilelang  # noqa: F401,E402  先 import tilelang 才会注册内置 tvm
from tvm.target import Target  # noqa: E402


def load_upstream_kernel():
    """加载上游 kernel.py，并做**一处语义等价**的版本适配。

    上游 requirements.txt pin 的是 `tilelang==0.1.6`，本项目 pin 的是 0.1.15（差 9 个版本），
    期间 `PassConfigKey.TL_DISABLE_FAST_MATH` 被改名/反转为 `TL_ENABLE_FAST_MATH`：
        - 0.1.6：`TL_DISABLE_FAST_MATH: True` → 显式关掉 fast math
        - 0.1.15：`libgen.py:70` `.get(PassConfigKey.TL_ENABLE_FAST_MATH, False)`，
          只有显式为 True 才追加 `--use_fast_math`
    即"fast math 关闭"在 0.1.15 里**已是默认状态**，所以删掉那一行是**等价替换，不是近似**。
    按行删除（不依赖缩进/行序），只在内存里改，clone 保持原样；
    compile 时传原始文件名，traceback 里的行号仍指向真实文件。
    """
    import types
    src = (DSV32 / "kernel.py").read_text()
    kept = [ln for ln in src.splitlines(keepends=True)
            if "TL_DISABLE_FAST_MATH" not in ln
            and not (ln.lstrip().startswith("#") and "fast-math" in ln)]
    patched = "".join(kept)
    assert len(patched) < len(src), "上游 kernel.py 里找不到该配置了，适配失效——请人工检查"
    mod = types.ModuleType("dsv32_kernel")
    mod.__file__ = str(DSV32 / "kernel.py")
    exec(compile(patched, str(DSV32 / "kernel.py"), "exec"), mod.__dict__)  # noqa: S102
    return mod


_k = load_upstream_kernel()
act_quant_kernel, fp8_gemm_kernel = _k.act_quant_kernel, _k.fp8_gemm_kernel

ARCH = "sm_80"  # 目标机 A800；fp8 mma 是 sm89+，但 act_quant 里没有 mma，应该能编译


def dump(name, fn, *args, **kwargs):
    path = OUT / f"{name}.cu"
    try:
        with Target({"kind": "cuda", "arch": ARCH}):
            kernel = fn(*args, **kwargs)
        src = kernel.get_kernel_source()
    except Exception as e:
        print(f"{name:<28} FAIL  {type(e).__name__}: {str(e)[:110]}")
        return None
    path.write_text(src)
    lines = src.splitlines()
    marks = [s for s in ("cp_async", "__shfl", "shfl_xor", "shfl_down", "mma", "pred", "cp.async")
             if s in src]
    print(f"{name:<28} OK    {len(lines):>4} 行  {len(src):>7} 字符  命中: {','.join(marks) or '-'}")
    return src


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"target = cuda/{ARCH}，输出目录 {OUT}\n")
    dump("act_quant_ns2", act_quant_kernel, 4096, round_scale=False)   # num_stages=2
    dump("act_quant_roundscale", act_quant_kernel, 4096, round_scale=True)  # num_stages=0
    dump("fp8_gemm", fp8_gemm_kernel, 4096, 4096)
    return 0


if __name__ == "__main__":
    sys.exit(main())
