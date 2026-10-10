# 实验结果记录

## 环境基线
- GPU：A800 80GB（SM80），带宽 2039 GB/s，BF16 峰值 312 TFLOPS
- CUDA 12.8 / PyTorch 2.7.0+cu128 / TileLang 0.1.15（Day 0 实测）
- tilelang 官方 checkout：v0.1.15（commit: a35f8dd，2026-09-30 release）

## Day 0：冒烟测试

| 脚本 | 配置 | 校验 | 延迟 | 备注 |
|---|---|---|---|---|
| quickstart.py | 1024³ fp16，ReLU | matches PyTorch reference | 0.0299 ms | — |
| example_gemm.py | 1024³ fp16 | All check passed | 0.0258 ms | ≈ 83 TFLOPS（峰值 312 的 ~27%，Day 2 对标基线） |

## Day 2：BF16 GEMM 对标
（待填）

## Day 4：量化精度实验
（待填）

...
