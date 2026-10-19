"""Small, fixed-text perplexity comparison for Day 11."""

import argparse
import gc
import json
import math
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e_w4a16 import load_model, wrap_model_w4a16
from scripts.paths import RESULT_DIR


TEXT = """import torch
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)
""".strip()


@torch.no_grad()
def perplexity(model, tokenizer, text, max_tokens=128):
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids.cuda()
    return math.exp(float(model(input_ids=ids, labels=ids, use_cache=False).loss))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    kwargs = {} if args.model_dir is None else {"model_dir": args.model_dir}
    model, tokenizer = load_model(**kwargs)
    p0 = perplexity(model, tokenizer, TEXT, args.max_tokens)
    del model, tokenizer; gc.collect(); torch.cuda.empty_cache()
    model, tokenizer = load_model(**kwargs)
    replaced, skipped = wrap_model_w4a16(model)
    p1 = perplexity(model, tokenizer, TEXT, args.max_tokens)
    result = {"text_tokens": args.max_tokens, "fp16_ppl": p0, "w4a16_ppl": p1,
              "relative_change_pct": (p1 / p0 - 1) * 100, "replaced": len(replaced), "skipped": skipped}
    output = args.output or str(RESULT_DIR / "day11_ppl.json")
    with open(output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(result)


if __name__ == "__main__":
    main()
