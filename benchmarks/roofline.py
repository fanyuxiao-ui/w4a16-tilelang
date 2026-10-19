"""Day 2 步骤 4：Roofline 粗算（纯算术，不需要 GPU）。

判断一个 shape 是 compute-bound 还是 memory-bound：拿计算强度 AI = FLOPs/Bytes
和机器平衡点比。平衡点 = 峰值算力 / 带宽，A800 fp16 上是 312e12/2039e9 ≈ 153 FLOP/Byte。

同理给出 W4A16 的理论下界——这是整个项目的动机：decode 阶段（M 小）时间全花在读权重上，
权重压到 4-bit，理论上限直接 ×4。

口径说明：bytes 只算"读 A、读 B、写 C 各一次"，不考虑 L2 复用，所以是**理想下界**；
真实 kernel 不可能比它更快，除非少读字节。
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

A800_PEAK_TFLOPS = 312.0
A800_BW_GBPS = 2039.0

SHAPES = [(1, 4096, 4096), (16, 4096, 4096), (256, 4096, 4096), (4096, 4096, 4096)]

GROUP_SIZE = 128  # W4A16 常见的 per-group 量化粒度


def analyze(m, n, k, peak_tflops=A800_PEAK_TFLOPS, bw_gbps=A800_BW_GBPS):
    """fp16 口径的 Roofline。返回一行结果字典。"""
    flops = 2 * m * n * k
    bytes_moved = 2 * (m * k + n * k + m * n)  # fp16: 每元素 2 字节
    ai = flops / bytes_moved
    balance = peak_tflops * 1e12 / (bw_gbps * 1e9)
    return {
        "M": m, "N": n, "K": k,
        "flops": flops,
        "bytes_fp16": bytes_moved,
        "ai_fp16": round(ai, 1),
        "balance": round(balance, 1),
        "bound": "compute-bound" if ai > balance else "memory-bound",
        "t_ideal_fp16_us": round(bytes_moved / (bw_gbps * 1e9) * 1e6, 2),
    }


def analyze_w4a16(row, bw_gbps=A800_BW_GBPS):
    """W4A16 口径：权重 int4（0.5 B/元素）+ per-group scale（2 B / 128 元素），
    激活仍 fp16，输出 fp16。FLOPs 不变——所以 AI 直接涨上去。"""
    m, n, k = row["M"], row["N"], row["K"]
    weight_bytes = n * k * 0.5 + n * (k / GROUP_SIZE) * 2
    bytes_moved = 2 * m * k + weight_bytes + 2 * m * n
    row["bytes_w4a16"] = round(bytes_moved)
    row["ai_w4a16"] = round(row["flops"] / bytes_moved, 1)
    row["t_ideal_w4a16_us"] = round(bytes_moved / (bw_gbps * 1e9) * 1e6, 2)
    row["speedup_ceiling"] = round(row["t_ideal_fp16_us"] / row["t_ideal_w4a16_us"], 2)
    return row


def main():
    rows = []
    print(f"A800：峰值 {A800_PEAK_TFLOPS} TFLOPS，带宽 {A800_BW_GBPS} GB/s，"
          f"机器平衡点 = {A800_PEAK_TFLOPS * 1e12 / (A800_BW_GBPS * 1e9):.1f} FLOP/Byte")
    print(f"W4A16 口径：权重 int4 + per-group({GROUP_SIZE}) scale，激活/输出 fp16\n")
    print(f"{'M':>5} {'N':>5} {'K':>5} | {'AI(fp16)':>9} {'AI(w4a16)':>10} | {'判定':>14} | "
          f"{'下界fp16':>9} {'下界w4a16':>10} {'加速上限':>8}")
    print("-" * 92)
    for s in SHAPES:
        r = analyze_w4a16(analyze(*s))
        rows.append(r)
        print(f"{r['M']:>5} {r['N']:>5} {r['K']:>5} | {r['ai_fp16']:>9.1f} {r['ai_w4a16']:>10.1f} | "
              f"{r['bound']:>14} | {r['t_ideal_fp16_us']:>8.1f}us {r['t_ideal_w4a16_us']:>9.1f}us "
              f"{r['speedup_ceiling']:>7.2f}x")

    outdir = ROOT / "benchmarks" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "day02_roofline.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\n-> saved: {path}\n")

    # 把结论直接打成可背的句子（Day 13 博客 / 面试用）
    for r in rows:
        flip = "已经翻过平衡点" if r["ai_w4a16"] > r["balance"] else "仍在平衡点左侧"
        print(f"M={r['M']:<5} fp16 AI={r['ai_fp16']:<7.1f} → W4A16 后 AI={r['ai_w4a16']:<7.1f}"
              f"（{flip}），延迟下界 {r['t_ideal_fp16_us']}us → {r['t_ideal_w4a16_us']}us "
              f"= {r['speedup_ceiling']}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
