# 手写 CUDA vs TileLang：Day 7 实测笔记

## 实测数据

同一张 A800、同一份 `[N,K/2] uint8` packed Q、fp16 scale、fp16 激活和 fp16 输出，计时只包含 kernel 调用，warmup/rep 使用 `bench_common.bench_fn`。两种实现都通过同一份 torch 反量化参考，平均相对误差约 `1.7e-4`。

| shape | 手写 CUDA (ms) | TileLang v2 (ms) | TileLang 加速 | CUDA TFLOPS | TileLang TFLOPS |
|---|---:|---:|---:|---:|---:|
| 1×4096×4096 | 0.21142 | 0.07867 | 2.69× | 0.159 | 0.427 |
| 16×4096×4096 | 0.48259 | 0.07982 | 6.05× | 1.113 | 6.726 |
| 512×4096×4096 | 8.47174 | 0.28060 | 30.19× | 2.028 | 61.226 |
| 1024×4096×4096 | 16.84828 | 0.46641 | 36.12× | 2.039 | 73.669 |
| 4096×4096×4096 | 67.08633 | 1.75474 | 38.23× | 2.049 | 78.324 |

CUDA 实现是可读的 BM=BN=16、BK=32 shared-memory baseline：Q 在 load 阶段解包、乘 scale 后写入 shared，再做逐线程 fp32 累加。它没有 cp.async、多级流水、寄存器分块或 Tensor Core，因此这些数字是 baseline，不是工业 CUDA kernel 的上限。

## 为什么会慢

1. 每个 32 元素 K tile 都由同步的 global→register→shared 路径装载；TileLang 的 `T.Pipelined` 生成了多级缓冲和 cp.async。
2. CUDA baseline 每线程只有一个输出元素，shared tile 之后仍然是标量 FMA；TileLang 的 `T.gemm` 走 SM80 MMA。
3. baseline 没有 swizzle 和 layout specialization；TileLang 由布局推断处理 ldmatrix 所需的线程到元素映射。

## GEMM 优化阶梯（本次完成到 v0）

| 版本 | 增量 | 4096³ TFLOPS | 相对上一层 | 解释 |
|---|---|---:|---:|---|
| v0 朴素 shared tile | 解包到 shared，每线程一个输出 | 2.068 | — | 正确性基线；没有 Tensor Core |
| v1+寄存器分块 | 未在本次运行实现 | — | — | 后续需要改变线程输出映射 |
| v2+向量化 | 未在本次运行实现 | — | — | 需要对 A/Q 做 half8/uint4 装载 |
| v3+双缓冲 | 未在本次运行实现 | — | — | cp.async 与 barrier 配对 |
| v4+swizzle | 未在本次运行实现 | — | — | shared bank conflict 实验 |
| v5+mma | TileLang v2 对照 | 78.318 | 37.86× | `T.gemm` 生成 `mma.sync` |

这里没有把未实现的层伪造为数字；代码和 JSON 只记录了确实运行过的 baseline 及 TileLang 对照。

## 七个维度的对比

| 维度 | 手写 CUDA | TileLang |
|---|---|---|
| 实现时间 | 新增 `.cu`、binding、loader，需处理 ABI 和边界 | 新增一个 Python kernel 函数 |
| 流水线 | 手动管理 shared 和同步 | `T.Pipelined` 生成流水 |
| swizzle | 需要手推线程映射 | `T.use_swizzle` / layout annotation |
| MMA | 需要手写 fragment 与 PTX/intrinsic | `T.gemm` |
| 调优 | 修改代码后重新编译 | 参数 sweep / autotune |
| 自定义 dequant | 完全自由 | `T.Parallel` 中可以直接表达位解包和 scale |
| 可诊断性 | kernel 级 printf、Nsight 直接可看 | 可保存生成 CUDA，再回到底层检查 |

结论是当前实现级别下 TileLang 更快，因为它自动提供了结构性优化。这个结果不能代表所有手写 CUDA；它准确表示本仓库的朴素 CUDA baseline 和当前 v2 在同一输入上的差距。
