"""DeepSeek-Coder model loader and W4A16 Linear wrapper for Day 11."""

import argparse
import gc
import os
import pathlib
import sys
from typing import Iterable

import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.torch_ref import quantize_int4_symmetric
from kernels.dequant_gemm_w4a16 import dequant_gemm_v2
from kernels.gemv_w4a16 import gemv_w4a16
from scripts.paths import MODEL_DIR


GROUP_SIZE = 128
BUCKETS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def bucket_of(m: int) -> int:
    for bucket in BUCKETS:
        if m <= bucket:
            return bucket
    return ((m + 4095) // 4096) * 4096


class W4A16Linear(nn.Module):
    """Drop-in ``nn.Linear`` using packed W4A16 weights.

    M=1 uses the decode GEMV kernel. Other M values are padded to a cached
    bucket and use the fused dequant GEMM kernel.
    """

    def __init__(self, base: nn.Linear, group_size: int = GROUP_SIZE):
        super().__init__()
        weight = base.weight.detach().to(dtype=torch.float16).contiguous()
        out_features, in_features = weight.shape
        if in_features % group_size or in_features % 2:
            raise ValueError(f"unsupported Linear shape {tuple(weight.shape)} for group_size={group_size}")
        q, scales = quantize_int4_symmetric(weight, group_size=group_size)
        self.register_buffer("q", q, persistent=False)
        self.register_buffer("scales", scales, persistent=False)
        if base.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", base.bias.detach().clone(), persistent=False)
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self._kernels = {}

    def _get_kernel(self, m_bucket: int):
        if m_bucket not in self._kernels:
            if m_bucket == 1:
                kernel = gemv_w4a16(self.out_features, self.in_features, group_size=self.group_size)
                self._kernels[m_bucket] = ("gemv", kernel)
            else:
                kernel = dequant_gemm_v2(
                    m_bucket,
                    self.out_features,
                    self.in_features,
                    group_size=self.group_size,
                )
                self._kernels[m_bucket] = ("gemm", kernel)
        return self._kernels[m_bucket]

    def forward(self, x):
        original_shape = x.shape
        x2 = x.reshape(-1, original_shape[-1]).contiguous().to(torch.float16)
        m = x2.shape[0]
        if m == 1:
            _, kernel = self._get_kernel(1)
            out = torch.empty(self.out_features, device=x.device, dtype=torch.float16)
            kernel(x2.reshape(-1), self.q, self.scales, out)
            out = out.view(1, self.out_features)
        else:
            m_bucket = bucket_of(m)
            if m_bucket != m:
                x2 = torch.cat((x2, torch.zeros(m_bucket - m, x2.shape[1], device=x.device, dtype=x2.dtype)), dim=0)
            _, kernel = self._get_kernel(m_bucket)
            out = torch.empty(m_bucket, self.out_features, device=x.device, dtype=torch.float16)
            kernel(x2, self.q, self.scales, out)
            out = out[:m]
        if self.bias is not None:
            out = out + self.bias
        return out.view(*original_shape[:-1], self.out_features).to(dtype=x.dtype)


def load_model(model_dir: str | os.PathLike = MODEL_DIR):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = str(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).cuda().eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def wrap_model_w4a16(model: nn.Module, skip_names: Iterable[str] = ()):
    skip_names = set(skip_names)
    replaced, skipped = [], []

    def walk(parent: nn.Module, prefix: str = ""):
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}{name}"
            if full_name in skip_names:
                continue
            if isinstance(child, W4A16Linear):
                continue
            if isinstance(child, nn.Linear):
                try:
                    setattr(parent, name, W4A16Linear(child))
                    replaced.append(full_name)
                except ValueError:
                    skipped.append(full_name)
            else:
                walk(child, full_name + ".")

    walk(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return replaced, skipped


@torch.no_grad()
def prewarm(model: nn.Module, tokenizer, lengths=(1, 8, 32)):
    vocab = int(getattr(model.config, "vocab_size", len(tokenizer)))
    for length in lengths:
        ids = torch.randint(0, vocab, (1, length), device="cuda")
        model(input_ids=ids, use_cache=False)
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--skip", nargs="*", default=[])
    args = parser.parse_args()
    model, tokenizer = load_model(args.model_dir)
    before = torch.cuda.memory_allocated() / 2**30
    replaced, skipped = wrap_model_w4a16(model, args.skip)
    print({"memory_before_gib": before, "memory_after_gib": torch.cuda.memory_allocated() / 2**30,
           "replaced": len(replaced), "skipped": skipped[:20]})
    prompt = tokenizer("def fib(n):\n", return_tensors="pt").to("cuda")
    out = model.generate(**prompt, max_new_tokens=2, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
