# W4A16 Quantized Inference GEMM in TileLang

INT4 weight-only quantized (W4A16) GEMM/GEMV kernels implemented in [TileLang](https://github.com/tile-ai/tilelang),
with hand-written CUDA baselines, validated on A800 (SM80).

## Status
- [ ] W4A16 dequant GEMM (TileLang)
- [ ] Hand-written CUDA baseline
- [ ] Benchmarks vs cuBLAS / Marlin
- [ ] End-to-end evaluation on DeepSeek-Coder-6.7B

## Environment
- GPU: NVIDIA A800 (SM80, 80GB, 2039 GB/s HBM)
- CUDA: 12.8 | PyTorch: 2.7.0+cu128 | TileLang: 0.1.15
