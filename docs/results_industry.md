# 工业对标：PyTorch int4pack 与 Marlin 路线

## 计时条件

- 本机：NVIDIA A800 80GB PCIe（SM80），CUDA 12.8，PyTorch 2.7.0+cu128。
- 本项目 kernel：fp16 激活、W4A16、group size 128、signed nibble 行主序，计时不含编译。
- PyTorch int4pack：bf16 激活、`torch._convert_weight_to_int4pack(innerKTiles=8)`，scale/zero 输入是 `[K/group,N,2]`，计时不含 packing。它的布局和本项目不同，不能作为数值等价实现。

## 同 shape 对比

| shape | 我们 v2 (ms) | 两步走 (ms) | cuBLAS fp16 (ms) | PyTorch int4pack (ms) | 备注 |
|---|---:|---:|---:|---:|---|
| 1×4096×4096 | 0.07877 | 0.06733 | 0.04287 | 0.01804* | `*` int4pack 为 bf16 |
| 16×4096×4096 | 0.07958 | 0.06762 | 0.04277 | 0.04504* | `*` int4pack 为 bf16 |
| 512×4096×4096 | 0.28086 | 0.12504 | 0.09114 | 0.93610* | `*` int4pack 为 bf16 |
| 4096×4096×4096 | 1.75480 | 0.66569 | 0.63746 | — | 未在 int4pack 脚本中重复大 shape |

PyTorch 的路径 A 已跑通，因此没有把外部公开数字冒充本机结果。vLLM/独立 Marlin 未安装；本表没有填入未实测的 Marlin 延迟。

## 差距归因

Marlin 类 kernel 与本项目 v2 的关键差别是数据流，而不是单个魔法指令：

1. Marlin 离线重排权重，使一个 warp 为 MMA fragment 需要的 nibble 在 128-bit 读中连续出现；本项目仍是 `[N,K/2]` 行主序。
2. Marlin 在寄存器中解包并乘 scale，避免 v2 的 Q shared→W shared 往返；本项目的 shared W 是为了复用已有 `T.gemm` 接口。
3. Marlin 把 cp.async、多级缓冲、dequant 和 MMA 交错起来，并针对 decode/prefill 选择不同调度；本项目 v2 只做了 scale 预取、block/stage sweep 和可选 rasterization swizzle。

如果只用一天搬运一条路线，优先做离线 magic layout + register dequant；它会同时减少 shared 流量和线程间搬运，收益比继续微调 block 更容易解释。

## 可复现实验

```bash
python benchmarks/bench_int4pack_mm.py
python benchmarks/bench_w4a16.py --output day06_w4a16_all.json
```

结果源文件是 `benchmarks/results/day08_int4pack.json` 和 `benchmarks/results/day06_w4a16_all.json`。
