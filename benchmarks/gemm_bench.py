"""Day 2 步骤 2：GEMM 三列对标（cuBLAS vs TileLang vs 手写 CUDA）。

用法：
    python benchmarks/gemm_bench.py                # 需要 GPU，跑完整对标并落 JSON
    python benchmarks/gemm_bench.py --check-build  # 无 GPU 时只校验 kernel 能否编译

第三列（手写 CUDA）见 Day 2 步骤 3，今天是空的，允许缺席。
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from benchmarks.bench_common import bench_fn, print_table, save_results, tflops_gemm  # noqa: E402
from kernels.gemm_bf16 import gemm_bf16  # noqa: E402

SHAPES = [
    (1024, 1024, 1024),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (16, 4096, 4096),      # decode 形态（小 M）
    (1, 4096, 4096),       # GEMV 形态
]

# 起点配置取 Day 1 扫描出的最佳：4096³ → 188.2 TFLOPS（cuBLAS 220.8 的 85%）。
# 五个 shape 刻意共用同一份配置——小 M 上的浪费必须能归因到"tilelang 规则 tiling"，
# 而不是归因到我们给大 shape 特调过。这正是要观察的现象（Day 9 会专门治它）。
TL_CONFIG = dict(block_M=128, block_N=256, block_K=64, num_stages=3, threads=256)

# 正确性判据：Frobenius 相对误差 ||c-ref||_F / ||ref||_F。
# 两个坑在 Day 1 都踩过（见 docs/results.md 踩坑二）：
#   ① 不能拿 fp16 的 cuBLAS 当真值——它比 tilelang 离真值更远（max|err| 0.182 vs 0.127）；
#   ② 不能用手册模板里的 rtol/atol=1e-2——K=4096 时输出量级 ~390，fp16 半个 ULP 就是
#      0.1268，|ref|≈1 附近 rtol 几乎不贡献容差，两套实现的舍入差直接撞破 atol。
# 正确做法是比 fp32 真值 + 尺度无关的相对误差。正确实现的预期值 ~3e-4（fp16 输出舍入），
# 布局写反这类真 bug 是 O(1)，1e-2 的门限两头都能分开。
REL_ERR_TOL = 1e-2


def make_inputs(m, n, k, seed=0):
    """固定种子，保证重跑可比。"""
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(m, k, device="cuda", dtype=torch.float16, generator=g)
    b = torch.randn(n, k, device="cuda", dtype=torch.float16, generator=g)
    return a, b


def bench_cublas(a, b):
    """cuBLAS 列。计时口径：a @ b.T 每次分配新输出（与 tilelang 的 out 版本略有差异）。"""
    return bench_fn(lambda: a @ b.t())


def frob_rel_err(a, b, c, chunk=512):
    """Frobenius 相对误差 ||c-ref||_F / ||ref||_F，参考值按 N 维分块算。

    分块不为提速，为省内存：8192³ 若一次性物化（a.float()/b.float()/ref 各 268MB），
    叠上 torch+tilelang 约 560MB 的基线就会顶到本机 2GB 的 cgroup 上限。
    """
    n = b.shape[0]
    af = a.float()  # 只转一次，循环里每次 a.float() 会重复分配 268MB
    num = den = 0.0
    for i in range(0, n, chunk):
        ref = af @ b[i:i + chunk].t().float()
        d = c[:, i:i + chunk].float() - ref
        num += d.pow(2).sum().item()
        den += ref.pow(2).sum().item()
    return (num / den) ** 0.5


def bench_tilelang(a, b, cfg=TL_CONFIG):
    """返回 (ms, rel_err)。先跑一次做正确性快检，再计时。"""
    m, k = a.shape
    n = b.shape[0]
    c = torch.empty(m, n, device="cuda", dtype=torch.float16)
    kernel = gemm_bf16(m, n, k, **cfg)
    kernel(a, b, c)

    # fp32 真值分块算，大 shape 上要几十秒，打行提示避免看起来像卡死。
    print(f"  [ref] 计算 {m}x{n}x{k} 的 fp32 参考值 ...", flush=True)
    rel_err = frob_rel_err(a, b, c)

    ms = bench_fn(lambda: kernel(a, b, c))
    return ms, rel_err


def run_shape(m, n, k):
    """单个 shape 的两列对标。任一列失败都记进 JSON，不中断整轮。"""
    r = {"M": m, "N": n, "K": k}
    a, b = make_inputs(m, n, k)

    try:
        ms, rel_err = bench_tilelang(a, b)
        r["tilelang_ms"] = round(ms, 4)
        r["tilelang_tflops"] = round(tflops_gemm(m, n, k, ms), 1)
        r["tilelang_rel_err"] = round(rel_err, 6)
        r["tilelang_ok"] = rel_err < REL_ERR_TOL
    except Exception as e:
        r["tilelang_ms"] = None
        r["tilelang_err"] = f"{type(e).__name__}: {e}"

    try:
        ms = bench_cublas(a, b)
        r["cublas_ms"] = round(ms, 4)
        r["cublas_tflops"] = round(tflops_gemm(m, n, k, ms), 1)
    except Exception as e:
        r["cublas_ms"] = None
        r["cublas_err"] = f"{type(e).__name__}: {e}"

    if r.get("tilelang_ms") and r.get("cublas_ms"):
        r["tl/cublas"] = round(r["tilelang_ms"] / r["cublas_ms"], 3)
    return r


def check_build(arch):
    """无 GPU 时的离线校验：只编译 kernel，不执行、不产出任何性能数字。

    auto_detect_target() 需要真实设备，无卡机器上会直接抛
    "No registered target detector found an available target"，
    所以这里用显式 target 把编译链路单独走通（nvcc 12.8 在无卡机器上是完好的）。
    """
    import tilelang  # noqa: F401  —— 先 import tilelang 才会注册内置 tvm
    from tvm.target import Target

    print(f"离线编译校验（target=cuda/arch={arch}，不产出性能数字）\n")
    rows = []
    for (m, n, k) in SHAPES:
        try:
            with Target({"kind": "cuda", "arch": arch}):
                kernel = gemm_bf16(m, n, k, **TL_CONFIG)
            src = kernel.get_kernel_source()
            # 用 cp_async（下划线）：裸 PTX 串 "cp.async" 只在部分配置里出现
            # （128/256/64 走的是 tl::cp_async_gs 封装），这个能两种都覆盖。
            marks = [s for s in ("cp_async", "ldmatrix", "mma_sync", "rasterization") if s in src]
            rows.append([m, n, k, "OK", len(src), ",".join(marks)])
        except Exception as e:
            rows.append([m, n, k, "FAIL", "-", f"{type(e).__name__}: {str(e)[:60]}"])
    print_table(rows, ["M", "N", "K", "编译", "源码字符数", "命中特征"])
    bad = [r for r in rows if r[3] != "OK"]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} 个 shape 编译通过。"
          "注意：编译通过只说明 kernel 能构建，不代表结果正确、更不代表快——"
          "延迟必须等有卡时跑完整对标得到。")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description="Day 2 GEMM 三列对标")
    ap.add_argument("--check-build", action="store_true",
                    help="无 GPU 时只校验所有 shape 能否编译（不产出性能数字）")
    ap.add_argument("--arch", default="sm_80", help="离线编译的架构，默认 sm_80（A800）")
    ap.add_argument("--out", default="day02_gemm_bf16_fp16.json")
    args = ap.parse_args()

    if args.check_build:
        return check_build(args.arch)

    # 无卡时必须硬失败：手册的铁律是"手敲的、凭记忆的数字一律不算数"，
    # 这里宁可什么都不产出，也不能落一份看起来像真数据的 JSON。
    if not torch.cuda.is_available():
        print("错误：没有可用的 CUDA 设备，对标无法执行。\n"
              "  - 想确认脚本和 kernel 构建链路是否完好，跑："
              "python benchmarks/gemm_bench.py --check-build\n"
              "  - 想要性能数字，必须在有卡的机器上跑（结果会写到 "
              f"{ROOT / 'benchmarks/results'}/）", file=sys.stderr)
        return 1

    print(f"设备：{torch.cuda.get_device_name(0)}  |  "
          f"配置：{TL_CONFIG}  |  warmup/rep=25/100 ms\n")

    rows = []
    for (m, n, k) in SHAPES:
        r = run_shape(m, n, k)
        rows.append(r)
        print(f"done {m}x{n}x{k}")

    print()
    print_table(
        [[r["M"], r["N"], r["K"], r.get("tilelang_ms"), r.get("tilelang_tflops"),
          r.get("tilelang_rel_err"), r.get("cublas_ms"),
          r.get("cublas_tflops"), r.get("tl/cublas")] for r in rows],
        ["M", "N", "K", "TL(ms)", "TL(TF)", "TL_rel_err", "cuBLAS(ms)", "cuBLAS(TF)", "TL/cuBLAS"],
    )
    save_results(args.out, rows, outdir=str(ROOT / "benchmarks/results"))

    bad = [r for r in rows if r.get("tilelang_ok") is False]
    if bad:
        print(f"\n警告：{len(bad)} 个 shape 的 TileLang 结果超出 rel_err < {REL_ERR_TOL} 门限："
              f"{[(r['M'], r['N'], r['K']) for r in bad]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
