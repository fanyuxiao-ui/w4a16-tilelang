# 简历条目：W4A16 量化推理算子库

**项目：W4A16 量化推理算子库（TileLang + CUDA，A800/SM80）**

- 使用 TileLang 实现 per-group INT4 权重反量化 GEMM/GEMV，将解包与 scale 融合进主循环，覆盖 prefill 与 decode；4096³ 达 78.3 TFLOPS，保留 cuBLAS、两步走和工业 int4pack 对照。
- 针对 M=1 decode 单独设计 GEMV，记录 packed 权重有效带宽；当前 4096² 为 89.5 GB/s（A800 理论带宽 4.39%），并基于数据定位 warp 级归约与寄存器解包的后续方向。
- 手写 CUDA 实现同款 dequant GEMM 作为 baseline，4096³ 与 TileLang v2 的实测为 67.1 ms vs 1.75 ms，完成流水线、MMA、布局和调度差距归因。
- 在 HF Transformers 单请求、batch=1、贪心解码口径下，将 DeepSeek-Coder-6.7B 的 225 个 Linear 替换为 W4A16：已分配显存从 12.56 GiB 降至 3.43 GiB；当前 GEMV baseline 使吞吐从 46.6 降至 31.3 token/s，PPL 从 2.826 增至 2.955，并明确记录限制和下一步优化。
- 全部代码、pytest 测试、benchmark JSON、生成 CUDA 和实验文档开源：<https://github.com/fanyuxiao-ui/w4a16-tilelang>。

## 电梯版

我在 A800 上用 TileLang 从零实现了 W4A16 融合 dequant GEMM 和 decode GEMV，配套手写 CUDA、cuBLAS、工业 int4pack 对标以及 DeepSeek-Coder-6.7B 端到端验证；代码、测试和原始性能数据全部开源，并把显存收益、吞吐负结果和 GEMV 瓶颈都按可复现口径记录下来。
