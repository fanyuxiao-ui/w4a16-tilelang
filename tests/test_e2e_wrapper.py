import torch
from torch import nn

from baseline.torch_ref import dequantize_int4_symmetric
from scripts.e2e_w4a16 import W4A16Linear


def _check(batch_tokens):
    torch.manual_seed(0)
    base = nn.Linear(128, 64, bias=True).cuda().half()
    wrapped = W4A16Linear(base)
    x = torch.randn(1, batch_tokens, 128, device="cuda", dtype=torch.float16)
    out = wrapped(x)
    ref_w = dequantize_int4_symmetric(wrapped.q, wrapped.scales, 128).half()
    ref = torch.nn.functional.linear(x, ref_w, base.bias)
    rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)
    assert rel.item() < 1e-3


def test_w4a16_wrapper_decode():
    _check(1)


def test_w4a16_wrapper_prefill_bucket():
    _check(3)
