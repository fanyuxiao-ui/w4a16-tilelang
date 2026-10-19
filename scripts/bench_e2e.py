"""Day 11 end-to-end fp16 versus W4A16 measurement."""

import argparse
import gc
import json
import pathlib
import sys
import time
from statistics import mean, stdev

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e_w4a16 import load_model, prewarm, wrap_model_w4a16
from scripts.paths import RESULT_DIR


PROMPT = "# Write a short Python function that computes Fibonacci numbers.\ndef fib(n):\n"


@torch.no_grad()
def bench_generate(model, tokenizer, prompt=PROMPT, new_tokens=8, warmup=1, repeats=2):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    for _ in range(warmup):
        model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    samples, text = [], ""
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model.generate(
            **inputs, max_new_tokens=new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
        torch.cuda.synchronize()
        samples.append(new_tokens / (time.perf_counter() - start))
        text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {
        "tokens_per_s": mean(samples),
        "tokens_per_s_stdev": stdev(samples) if len(samples) > 1 else 0.0,
        "sample": text[:500],
    }


def measure(tag, model, tokenizer, new_tokens, warmup, repeats):
    torch.cuda.reset_peak_memory_stats()
    result = bench_generate(model, tokenizer, new_tokens=new_tokens, warmup=warmup, repeats=repeats)
    result.update({
        "tag": tag,
        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    })
    print(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    kwargs = {} if args.model_dir is None else {"model_dir": args.model_dir}
    results = []
    model, tokenizer = load_model(**kwargs)
    results.append(measure("fp16-baseline", model, tokenizer, args.new_tokens, args.warmup, args.repeats))
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()

    model, tokenizer = load_model(**kwargs)
    replaced, skipped = wrap_model_w4a16(model)
    prewarm(model, tokenizer, lengths=(1, 8, 32))
    result = measure("w4a16", model, tokenizer, args.new_tokens, args.warmup, args.repeats)
    result["replaced_linear_count"] = len(replaced)
    result["skipped_linear_names"] = skipped
    results.append(result)
    output = args.output or str(RESULT_DIR / "day11_e2e.json")
    with open(output, "w") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(f"-> {output}")


if __name__ == "__main__":
    main()
