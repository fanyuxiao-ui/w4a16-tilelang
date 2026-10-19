"""Create the Day 13 decode latency comparison figure."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "benchmarks/results/day09_decode.json") as handle:
    rows = json.load(handle)

labels = [f'{row["N"]}×{row["K"]}' for row in rows]
x = list(range(len(rows)))
ours = [row["gemv_ms"] * 1000 for row in rows]
cublas = [row["cublas_ms"] * 1000 for row in rows]
width = 0.38

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar([i - width / 2 for i in x], ours, width, label="ours W4A16 GEMV")
ax.bar([i + width / 2 for i in x], cublas, width, label="cuBLAS fp16")
for i, (a, b) in enumerate(zip(ours, cublas)):
    ax.text(i, max(a, b) * 1.02, f"{a / b:.2f}×", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20)
ax.set_ylabel("latency (μs, M=1)")
ax.set_title("Decode latency: W4A16 GEMV vs cuBLAS fp16 on A800")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
out = ROOT / "assets/figs/decode_speedup.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")
