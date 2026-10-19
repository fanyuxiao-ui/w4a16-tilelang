# W4A16 Quantized Inference GEMM in TileLang

INT4 weight-only quantized (W4A16) GEMM/GEMV kernels implemented in [TileLang](https://github.com/tile-ai/tilelang),
with hand-written CUDA baselines, validated on A800 (SM80).

## Status
- [x] Day 1–4: FP16 GEMM baseline, INT4 reference/packing, dequant kernel and real-weight study
- [x] W4A16 dequant GEMM v1 (Day 5 correctness/performance baseline)
- [ ] Hand-written CUDA baseline
- [ ] Benchmarks vs cuBLAS / Marlin
- [ ] End-to-end evaluation on DeepSeek-Coder-6.7B

Day 1–4 reproduction details, JSON results and plots are in `docs/results.md` and
`benchmarks/results/`. The Day 2 hand-written CUDA column is intentionally deferred to
the dedicated Day 7 dequant GEMM baseline so FP16 and FP32 kernels are not mixed in one table.

## Environment
- GPU: NVIDIA A800 (SM80, 80GB, 2039 GB/s HBM)
- CUDA: 12.8 | PyTorch: 2.7.0+cu128 | TileLang: 0.1.15
