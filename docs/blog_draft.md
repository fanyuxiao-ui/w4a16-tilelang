# 在 A800 上用 TileLang 写 W4A16 量化推理算子

> 两周内实现融合 dequant GEMM、decode GEMV 和手写 CUDA 对照，在 A800 上完成从位打包、正确性、性能到真实模型的验证。完整代码和原始 JSON 结果见 [w4a16-tilelang](https://github.com/fanyuxiao-ui/w4a16-tilelang)。

## 1. 为什么 decode 是带宽墙

推理的 prefill 和 decode 看起来都在做矩阵乘法，瓶颈却不同。prefill 的 M 较大，同一份权重会被多个 token 复用，Tensor Core 的计算吞吐开始重要。decode 通常是 M=1，每生成一个 token 都要重新读取权重，计算量反而很小。

以 N=K=4096 为例，fp16 权重需要约 33.6 MB；W4A16 的 packed int4 加上 group scale 约 8.6 MB。A800 的理论带宽是 2039 GB/s，因此仅按读写下界估算，fp16 约 16.5 μs，W4A16 约 4.2 μs，字节数上限接近 3.9×。同一层的计算量只有约 33.6 MFLOP，远小于 A800 在一个微秒级窗口内的计算能力。M 越小，带宽越主导，这也是量化在 decode 阶段最有意义的原因。

## 2. 为什么选择 W4A16

本项目使用的是 A800，架构为 SM80。它适合 FP16、INT8 和 INT4 路径，但没有 H100 那种 FP8 Tensor Core 路径。因此选用权重 INT4、激活 FP16 的 W4A16：权重离线量化，运行时只解包、乘分组 scale，再参加 FP32 累加。

格式固定为 `[N,K/2] uint8`。每个 byte 的低四位保存第一个元素，高四位保存第二个元素；INT4 使用 `[-8,7]` 补码。scale 是 `[N,K/group_size]` 的 fp16 张量。对称量化省掉 zero point，kernel 侧只需一次乘法。

如果迁移到 H100，整体工程方法仍然适用，但 dtype、MMA 指令和 scale 的应用位置都要重新验证。DeepSeek 的 FP8 kernel 可以作为这个迁移方向的参考，但不能把 A800 的数字直接搬到 H100。

## 3. TileLang 和手写 CUDA 的分工

我先用 TileLang 写了 FP16 GEMM，随后把同样的主循环扩展成 W4A16。TileLang 负责 tile、流水线、ldmatrix 和 MMA 的结构代码；我需要明确的是 Q 的布局、解包顺序以及 scale 应该放在数据流的哪一段。

Day 7 又写了一个简单的 CUDA baseline：BM=BN=16、BK=32，每个线程负责一个输出元素，Q 在 shared memory 装载阶段解包并乘 scale，然后用标量 FP32 累加。4096³ 上，手写 baseline 是 67.086 ms，TileLang v2 是 1.755 ms。这个差距不是“CUDA 天生慢”，而是当前 CUDA baseline 没有 cp.async、多级缓冲、寄存器分块和 MMA。它的价值是把每个优化层次的成本固定下来。

TileLang 的优势在于消灭结构代码，算子工程师仍然要设计数据流。下一版如果要接近 Marlin，关键不是继续增加 shared buffer，而是离线重排权重，让一个 warp 需要的 nibble 连续出现在 128-bit 读中，并把解包和 scale 保留在寄存器路径。

## 4. 融合 dequant GEMM

两步走的流程是先把 Q 反量化成完整 fp16 权重，再调用 cuBLAS。它会产生一次完整的 fp16 写入和一次完整的 fp16 读取。融合 kernel 则把 packed Q 搬到 shared，直接解包成当前 tile，再参加 GEMM，避免完整权重矩阵落地。

核心循环的逻辑很短：

```python
T.copy(Q[bx * block_N, k * block_K // 2], Q_shared)
T.copy(S[bx * block_N, (k * block_K) // group_size], S_shared)
for i, j in T.Parallel(block_N, block_K):
    W_shared[i, j] = T.Cast(
        "float16",
        T.Cast("float32", decode(Q_shared[i, j // 2], j % 2))
        * T.Cast("float32", S_shared[i, j // group_size]),
    )
T.gemm(A_shared, W_shared, C_local, transpose_B=True)
```

正确性没有只依赖一个参考。每个 shape 同时对比 torch 黄金实现、独立反量化 kernel 加 GEMM，以及融合 kernel。4 组 Day 5 shape 和后续 CUDA/GEMV 测试全部通过，融合 GEMM 相对 torch 的平均误差约 `1.7e-4`。

## 5. 优化归因

Day 6 做了 scale 预取、block/stage sweep、rasterization swizzle 和 TileLang autotune。autotune 在 `128×1024×1024` 上测试 8 个配置，选择 `block_M=64, block_N=64, block_K=128, num_stages=2, threads=128`，延迟约 0.0260 ms。

在 4096³ 上，v1 是 1.77607 ms，v2 是 1.75480 ms，打开 block swizzle 后是 1.75071 ms。收益不到 2×，但这个结果很有解释力：scale 张量很小，已经能被 cache 较好地命中，新增 shared copy 和同步开销抵消了大部分收益。下一步应该做 register dequant 和 Marlin 式布局，而不是把更多元数据搬进 shared。

完整 prefill 表如下：

| shape | fused v2 | 两步走 | cuBLAS fp16 |
|---|---:|---:|---:|
| 512×4096×4096 | 0.28086 ms | 0.12504 ms | 0.09114 ms |
| 4096×4096×4096 | 1.75480 ms | 0.66569 ms | 0.63746 ms |

当前实现低于 cuBLAS 是预期结果：融合 kernel 还要做解包和 scale，且 prefill 计算复用充分，fp16 cuBLAS 的 Tensor Core 峰值更高。W4A16 的主要优势仍然在权重流量，而不是所有 M 都能得到更低的 kernel 延迟。

## 6. 工业 int4 kernel 的差距

PyTorch 2.7 的 `torch._convert_weight_to_int4pack` 和 `_weight_int4pack_mm` 路径在本机跑通。它要求 bf16 激活、`innerKTiles=8`，scale/zero 的布局是 `[K/group,N,2]`，所以只能作为工业参考，不能与本项目的 fp16、row-major signed-nibble 格式直接当成同一实现。

M=1、N=K=4096 时，PyTorch int4pack 是 0.01804 ms；本项目 v2 是 0.07877 ms。Marlin 类实现的主要差距有三点：魔法布局减少跨线程搬运，寄存器解包避免 shared 往返，以及针对 decode/pre-fill 使用不同调度。`docs/results_industry.md` 保留了 layout、dtype、计时对象和来源说明。

## 7. decode GEMV 的实测

我为 M=1 单独写了 GEMV，每个线程负责一个输出行，沿 K 流式读取 packed Q。正确性通过 4096²、11008×4096 和 4096×11008 三组 shape，但带宽利用率仍然偏低：

| N×K | GEMV 延迟 | 有效带宽 | A800 利用率 | cuBLAS fp16 |
|---|---:|---:|---:|---:|
| 4096×4096 | 96.82 μs | 89.518 GB/s | 4.39% | 42.62 μs |
| 8192×8192 | 228.37 μs | 151.665 GB/s | 7.44% | 111.56 μs |

官方 fp16xint4 GEMV 示例在 8192²、无 scale 的参考口径下是 0.036570 ms，约 45% 的 packed-weight 带宽利用率。这个对比说明自研版本的主要瓶颈是每线程一列导致的访问形态，而不是 INT4 解包数学本身。warp 级 K 归约、16B 向量装载和寄存器解包是下一步。

## 8. 真实模型端到端

Day 11 把 DeepSeek-Coder-6.7B 的 225 个 Linear 全部替换成 `W4A16Linear`。M=1 使用 GEMV，其他 M 按桶使用 GEMM，第一次命中桶时编译，随后复用。

| 指标 | fp16 | W4A16 | 变化 |
|---|---:|---:|---:|
| 已分配显存 | 12.563 GiB | 3.429 GiB | -72.7% |
| 生成吞吐 | 48.147 token/s | 31.793 token/s | -34.0% |
| 小样本 PPL | 2.8262 | 2.9551 | +4.56% |

这是一个有用的负结果。权重显存明显下降，但吞吐没有同步提升，因为当前 GEMV 只达到理论带宽的 4%–11%，而端到端还包含 attention、KV cache、norm、采样和 Python/module 调度。测试是 HF Transformers、单请求、batch=1、贪心解码，不能写成 vLLM 服务吞吐提升。

Day 10 的 MoE 实验使用确定性 ToyMoE fallback，因为 Qwen1.5-MoE-A2.7B 权重超过当前工作区安全容量。它仍然验证了专家矩阵的 W4A16 误差和 top-2 权重流量预算，但 PPL proxy 不能冒充真实 Qwen 评测。

## 9. 三个收获

第一，先诊断再手术比堆优化技巧更重要。scale 预取的收益很小，只有在同一计时口径下对照才看得出来。第二，TileLang 自动化的是结构，不是数据流；布局、解包和融合位置仍然决定 kernel 的上限。第三，负结果同样是工程证据。真实的显存下降、吞吐下降和 GEMV 带宽数字，比一个没有口径的漂亮加速倍数更能支持下一步决策。

## 10. 后续工作

下一步是 Marlin 式离线布局和寄存器 dequant，接着是 warp 级 GEMV 归约，最后再评估 grouped MoE W4A16 与 H100 FP8 迁移。当前所有已完成范围、原始结果和已知限制都保存在仓库中，未完成项列在 `docs/todo.md`。
