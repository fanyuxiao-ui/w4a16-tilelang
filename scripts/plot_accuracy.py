"""把 Day 4 精度实验 JSON 画成 group_size-误差曲线。"""
import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "benchmarks/results/day04_accuracy.json"))
    parser.add_argument("--output", default=str(ROOT / "assets/figs/accuracy_vs_group.png"))
    args = parser.parse_args()
    rows = json.loads(pathlib.Path(args.input).read_text())

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tensor in sorted({r["tensor"] for r in rows}):
        points = sorted(
            (r["group_size"], r["rel_l2"] * 100)
            for r in rows
            if r["tensor"] == tensor and r["gran"].startswith("group")
        )
        if points:
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", label=tensor)
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64, 128])
    ax.set_xticklabels(["32", "64", "128"])
    ax.set_xlabel("group_size")
    ax.set_ylabel("relative L2 error (%)")
    ax.set_title("W4A16 quantization error vs group size (DeepSeek-Coder-6.7B)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
