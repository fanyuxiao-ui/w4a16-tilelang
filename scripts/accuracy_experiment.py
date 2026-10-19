"""Day 4 步骤 3：真实 DeepSeek 权重上的粒度-误差实验。"""
import argparse
import gc
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseline.torch_ref import (  # noqa: E402
    dequantize_int4_symmetric,
    quantize_int4_symmetric,
)
from scripts.load_weights import DEFAULT_MODEL_DIR, load_tensor  # noqa: E402


TENSORS = [
    "model.layers.10.self_attn.q_proj.weight",  # [4096, 4096]
    "model.layers.10.mlp.gate_proj.weight",  # [11008, 4096]
    "model.layers.10.mlp.down_proj.weight",  # [4096, 11008]
]
GRANULARITIES = [32, 64, 128]


def rel_l2(w_ref, w_deq):
    diff = (w_deq.float() - w_ref.float()).flatten()
    denom = w_ref.float().flatten().norm().clamp_min(1e-12)
    return (diff.norm() / denom).item()


def per_tensor_quant(w):
    scale = (w.float().abs().max() / 7.0).clamp_min(1e-7)
    q = torch.round(w.float() / scale).clamp(-8, 7)
    return q * scale


def run(model_dir=DEFAULT_MODEL_DIR, device=None, output=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if output is None:
        output = ROOT / "benchmarks/results/day04_accuracy.json"
    output = pathlib.Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in TENSORS:
        w = load_tensor(name, model_dir=model_dir, device=device)
        # The model stores BF16.  W4A16's output contract is FP16, so compare
        # against the same FP16 reconstruction used by the dequant kernel.
        w = w.to(torch.float16)
        short = name.split(".")[-2]
        N, K = w.shape
        print(f"\n=== {short} {tuple(w.shape)} ({w.dtype}, {w.device}) ===", flush=True)

        deq = per_tensor_quant(w).half()
        err = rel_l2(w, deq)
        print(f"per-tensor       : rel_l2 = {err:.6f}")
        rows.append({"tensor": short, "shape": [N, K], "gran": "tensor", "rel_l2": err})
        del deq

        q, scales = quantize_int4_symmetric(w, group_size=K)
        deq = dequantize_int4_symmetric(q, scales, group_size=K).half()
        err = rel_l2(w, deq)
        cos = torch.nn.functional.cosine_similarity(w.float().flatten(), deq.float().flatten(), dim=0).item()
        print(f"per-channel      : rel_l2 = {err:.6f}, cos = {cos:.6f}")
        rows.append({"tensor": short, "shape": [N, K], "gran": "channel", "rel_l2": err, "cos": cos})
        del q, scales, deq

        for group_size in GRANULARITIES:
            q, scales = quantize_int4_symmetric(w, group_size=group_size)
            deq = dequantize_int4_symmetric(q, scales, group_size=group_size).half()
            err = rel_l2(w, deq)
            cos = torch.nn.functional.cosine_similarity(
                w.float().flatten(), deq.float().flatten(), dim=0
            ).item()
            meta_ratio = 4.0 / group_size
            print(f"per-group {group_size:<4}   : rel_l2 = {err:.6f}, cos = {cos:.6f}, "
                  f"元数据开销 = {meta_ratio * 100:.1f}%")
            rows.append({
                "tensor": short,
                "shape": [N, K],
                "gran": f"group{group_size}",
                "group_size": group_size,
                "rel_l2": err,
                "cos": cos,
                "meta_ratio": meta_ratio,
            })
            del q, scales, deq

        del w
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nsaved to {output}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--device", default=None, help="默认 cuda（有卡）或 cpu")
    parser.add_argument("--output", default=str(ROOT / "benchmarks/results/day04_accuracy.json"))
    args = parser.parse_args()
    run(args.model_dir, args.device, args.output)


if __name__ == "__main__":
    main()
