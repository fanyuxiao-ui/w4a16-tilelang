"""Day 10 MoE validation.

When a full Qwen1.5-MoE checkpoint is available, this module can be extended
to its named expert modules.  The default run is a deterministic, small MoE
that exercises the exact expert Linear shapes and the same W4A16 path; it is
used here because the workspace is below the 35 GB safety threshold for the
roughly 29 GB bf16 checkpoint.
checkpoint.
"""

import copy
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.torch_ref import dequantize_int4_symmetric, quantize_int4_symmetric
from benchmarks.bench_common import bench_fn, gbps, save_results
from kernels.gemv_w4a16 import gemv_w4a16


class ToyExpert(nn.Module):
    def __init__(self, hidden=512, intermediate=1024):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ToyMoE(nn.Module):
    def __init__(self, hidden=512, intermediate=1024, experts=8, vocab=256, top_k=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.router = nn.Linear(hidden, experts, bias=False)
        self.experts = nn.ModuleList([ToyExpert(hidden, intermediate) for _ in range(experts)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.top_k = top_k

    def forward(self, ids):
        x = self.embed(ids)
        logits = self.router(x)
        topv, topi = logits.topk(self.top_k, dim=-1)
        weights = topv.softmax(dim=-1)
        out = torch.zeros_like(x)
        for slot in range(self.top_k):
            for expert_id, expert in enumerate(self.experts):
                mask = topi[..., slot] == expert_id
                if mask.any():
                    vals = expert(x[mask])
                    out[mask] += vals * weights[..., slot][mask].unsqueeze(-1)
        return self.lm_head(out)


def quantize_expert_weights(model, group_size):
    stats, packed = [], {}
    for ei, expert in enumerate(model.experts):
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(expert, proj_name)
            w = proj.weight.detach().cuda().half().contiguous()
            q, s = quantize_int4_symmetric(w, group_size)
            w_deq = dequantize_int4_symmetric(q, s, group_size).half()
            rel = (w_deq.float() - w.float()).norm() / w.float().norm().clamp_min(1e-12)
            key = f"experts.{ei}.{proj_name}"
            stats.append({"module": key, "shape": list(w.shape), "rel_err": float(rel)})
            packed[key] = (q, s)
    return stats, packed


def model_ppl(model, ids, labels):
    with torch.no_grad():
        logits = model(ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    return float(torch.exp(loss))


def make_quantized_model(model, group_size):
    out = copy.deepcopy(model).cuda().eval()
    stats, packed = quantize_expert_weights(out, group_size)
    for ei, expert in enumerate(out.experts):
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            key = f"experts.{ei}.{proj_name}"
            q, s = packed[key]
            getattr(expert, proj_name).weight.data.copy_(dequantize_int4_symmetric(q, s, group_size).half())
    return out, stats, packed


def single_expert_bench(expert, group_size=128):
    rows = []
    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        w = getattr(expert, proj_name).weight.detach().cuda().half().contiguous()
        n, k = w.shape
        q, s = quantize_int4_symmetric(w, group_size)
        x = (torch.randn(k, device="cuda") * 0.1).half()
        out = torch.empty(n, device="cuda", dtype=torch.float16)
        kg = gemv_w4a16(n, k, group_size=group_size)
        kg(x, q, s, out)
        int4_bytes = n * (k // 2) + n * (k // group_size) * 2
        bf16_ms = float(bench_fn(lambda: torch.mv(w, x)))
        int4_ms = float(bench_fn(lambda: kg(x, q, s, out)))
        rows.append({"projection": proj_name, "N": n, "K": k, "bf16_ms": bf16_ms,
                     "w4a16_ms": int4_ms, "int4_bytes": int4_bytes,
                     "w4a16_gbps": gbps(int4_bytes + k * 2 + n * 2, int4_ms)})
    return rows


def run():
    torch.manual_seed(0)
    model = ToyMoE().cuda().eval()
    ids = torch.randint(0, 256, (2, 32), device="cuda")
    labels = torch.randint(0, 256, (2, 32), device="cuda")
    base_ppl = model_ppl(model, ids, labels)
    all_stats = {}
    quant_models = {}
    for g in (128, 64):
        qm, stats, packed = make_quantized_model(model, g)
        ppl = model_ppl(qm, ids, labels)
        errs = [x["rel_err"] for x in stats]
        all_stats[str(g)] = {"mean_rel_err": sum(errs) / len(errs), "max_rel_err": max(errs),
                             "ppl": ppl, "ppl_increase_pct": (ppl / base_ppl - 1) * 100,
                             "layers": stats}
        quant_models[g] = qm

    expert_bench = single_expert_bench(model.experts[0], 128)
    active_experts = 2
    expert_params = sum(p.numel() for p in model.experts[0].parameters())
    bf16_bytes = active_experts * expert_params * 2
    w4_bytes = active_experts * sum(x["int4_bytes"] for x in expert_bench)
    # Full toy-model timing is a control: its quantized model stores dequantized
    # fp16 weights, so it measures model-level overhead, not packed-kernel speed.
    base_ms = float(bench_fn(lambda: model(ids)))
    deq_ms = float(bench_fn(lambda: quant_models[128](ids)))
    result = {
        "day": 10, "model": "deterministic ToyMoE fallback",
        "reason": "Qwen1.5-MoE-A2.7B checkpoint needs ~29 GB bf16; workspace is below the 35 GB safety threshold",
        "experts": len(model.experts), "top_k": active_experts, "hidden": 512, "intermediate": 1024,
        "baseline_ppl": base_ppl, "quantization": all_stats,
        "single_expert": expert_bench,
        "theoretical_weight_bytes": {"bf16": bf16_bytes, "w4a16": w4_bytes, "ratio": bf16_bytes / w4_bytes},
        "toy_e2e_ms": {"bf16": base_ms, "w4a16_dequantized_fp16_control": deq_ms},
    }
    save_results("day10_moe.json", result)
    with open("docs/moe_quant.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: result[k] for k in ("baseline_ppl", "quantization", "theoretical_weight_bytes", "toy_e2e_ms")}, indent=2))
    return result


if __name__ == "__main__":
    run()
