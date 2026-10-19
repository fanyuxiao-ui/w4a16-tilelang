# Day 3 步骤 4：DeepSeek `act_quant_kernel` 精读笔记

> 精读对象：`refs-deepseek-v32/inference/kernel.py`（commit `87e509a`，2025-11-18）
> `act_quant_kernel` = L36-111、`fp8_gemm_kernel` = L114-196、`fp8_index_kernel` = L199-275
> tilelang 源码对照：本项目 pin 的 `tilelang/` @ v0.1.15（`reduce_op.py` / `tl_templates/cuda/reduce.h` / `pipeline_planning.cc`）
> TileKernels：`refs-tilekernels` @ `c1cf914`（v2.1.0）
>
> **证据分级**：本文标 `【实测】` 的是跑出来/读到的原文；标 `【推断】` 的是我的解释，附推理链；
> 末尾单列"未验证项"，那是本次没能确认的东西，不要当成结论引用。

---

## 0. 先纸笔：一个 block 在干什么

`T.Kernel(ceildiv(M, 32), ceildiv(N, 128), threads=128)` —— 每个 block 负责**一个 32 token × 128 元素**的 tile，
而 128 = `group_size`，所以**一个 block 恰好是一个 token-block 的一个量化组**（grid 的 by 维就是 group 下标）。

```
读 X[M, N] bf16  ──copy──> x_shared[32,128] ──copy──> x_local[32,128] fragment
                                                              │
                                        T.reduce_absmax(dim=1) ──> amax_local[32]
                                                              │  max(·, 1e-4) → s = amax/448
                                                              ▼
                                        y_local[i,j] = clamp(x_local[i,j] / s_local[i], -448, 448)
                                                              │
                              S[.., pid_n] = s_local[i]  <────┤────>  y_shared ──copy──> Y[M,N] fp8
```

【实测】`blk_m=32`、`group_size=128`、`threads=128`、`fp8_min/max = ∓448.0`、
防除零用的是 `T.max(amax, 1e-4)`（不是 1e-10/1e-8——**因为他们的 scale 是 fp32**，
而我们的 scale 是 fp16，floor 必须按存储 dtype 选，见 `docs/int4_pack_spec.md` 不变量 5）。

## 1. Q1：数据流 —— 和 Day 4 的 INT4 反量化 kernel 有什么异同

**字节账**（按"每元素"计，N 取 V3.2 的 7168）：

| | act_quant（BF16→FP8） | Day 4 的反量化 kernel（INT4→FP16） |
|---|---|---|
| 读 | `X` bf16 = **2.0 B**/elem | `q_packed` uint8/2elem = **0.5 B**/elem |
| 读 | — | `scales` fp16/128elem = 0.016 B/elem |
| 写 | `Y` fp8 = **1.0 B**/elem | `W` fp16 = **2.0 B**/elem |
| 写 | `S` fp32/128elem = 0.031 B/elem | — |
| **读写比** | 2.0 : 1.03 ≈ **1.9 : 1（宽读窄写）** | 0.52 : 2.0 ≈ **1 : 3.9（窄读宽写）** |

**方向正好相反**。这不是巧合，是"量化 vs 反量化"的定义决定的：量化把宽类型压成窄类型，
反量化把窄类型摊回宽类型。对性能的含义也相反——act_quant 的成本受**读**带宽支配，
反量化的成本受**写**带宽支配。

**共同点（结构同源，可以直接迁移）**：

1. 都是 **memory-bound 的 elementwise + 行内归约**：每元素只做 1~2 次算术，算术强度远低于 1 flop/byte。
2. **group_size 都是 128，都沿最后一维分组**。DeepSeek V3 的量化食谱：训练用 128×128 block FP8、
   推理激活用 1×128、我们 W4A16 权重用 1×128——**都是 128，是同一条"越细越准、元数据越贵"的曲线的不同取点**。
3. 都走 `shared → fragment → 算 → shared → global` 的路子（fragment 是寄存器里的 tile，
   归约和逐元素计算都必须在这一层做）。

**根本区别（这一条决定了后面能不能融合）**：

| | act_quant | 我们的 dequant |
|---|---|---|
| 数据来源 | **激活**：每 token 每层现算，数据相关 | **权重**：离线量化一次，静态不变 |
| scale | 必须**在线算**（`reduce_absmax` → 跨线程通信） | 离线已知，直接读 |
| 能否融合进 GEMM | 不能（夹在上一算子与 GEMM 之间，是算子边界） | **能**（Day 5：在 GEMM 内部 shared→register 那一步顺手做掉） |

→ **"权重是静态的、激活是动态的"是 weight-only 量化能在 GEMM 内部融合的根本原因**，
而激活量化要融合就得跨算子边界（把 act_quant 塞进上一个算子的 epilogue），代价和收益完全不同。

## 2. Q2：为什么用 `T.reduce_absmax` 而不是在 `T.Parallel` 里手写归约

**核心答案：语言这一层没有"线程"这个概念，所以手写归约写不出正确的代码。**

证据链：

1. 【实测】`tilelang/language/reduce_op.py:159-169`：当输入输出都是 fragment 时，`reduce_absmax`
   退化成**一条不透明的 tile 级 intrinsic**（`tl.tileop.reduce`），不展开成任何循环：
   ```python
   elif is_fragment(buffer) and is_fragment(out):
       tirx.call_intrin("handle", tirx.op.Op.get(_REDUCE_OP_KEY), to_tile_region(buffer, "r"), ...)
   ```
   注意参数是 **`to_tile_region(...)`（逻辑 tile 区域）**，不是线程、不是索引。
   （另外 `reduce_op.py:103-140` 还处理了 shared/fragment 混用的四种 scope 组合，会自动插入 `T.copy` 中转。）
2. 【实测】`src/op/reduce.cc:185`：lowering 阶段才把它变成
   *"a call to a templated `tl::AllReduce<...>::run`"*。
3. 【实测】`src/tl_templates/cuda/reduce.h:211-253`：`AllReduce` 的模板参数是
   `threads`（参与归约的线程数）和 `scale`（参与线程在 threadIdx 空间的**步长**）——
   也就是说，"哪些线程持有这一行的数据"这个信息，**只有 lowering 才拿得到**。
   实现是递归 XOR-butterfly：**`offset >= 32` 走 shared memory + barrier，`offset < 32` 走 `shfl_xor_sync`**
   （注释原文），递归在 `threads == scale` 时停止（每个归约组塌缩到一个线程）。
   还有两条 static_assert：归约宽度与 step 都必须是 2 的幂。

所以**在 `T.Parallel(blk_m, group_size)` 里写 `amax[i] = max(amax[i], |x[i,j]|)` 会怎样**：

- `T.Parallel` 是一个**逻辑**并行循环，它的线程绑定是 layout inference 之后才决定的。
  一行 128 个元素平摊给 128 个线程（本 kernel 共 4096 元素/128 线程 = 每线程 32 个），
  一行必然跨多个线程 → 该写法是**多线程对同一地址的无保护读改写（race）**；
- tilelang **不会**替你在这种写法上插 atomic（`atomic_reduce.cc` 是另一套显式 API）；
- 就算你手动加锁/atomic，性能上也远不如 butterfly（atomic 走 L2，butterfly 走寄存器+shuffle）。

**还有一层更隐蔽的**：归约结果 `amax_local[32]` 本身是 fragment，它的**布局由编译器推断**——
必须让"持有第 i 行某段数据的那些线程"都拿到 `amax[i]`，因为紧接着的
`y_local[i,j] = x_local[i,j] / s_local[i]` 要在**每个线程各自的寄存器里**用到它。
这个"结果该落在哪些线程上"的映射是 layout inference 的输出，手写不可能对上。
**这就是为什么 tilelang 把 `reduce_absmax` 做成原语而不是让你写循环。**

一句话总结：**手写归约错在"这一层还没有线程"，而不是错在"写得不够优化"。**
类比 CUDA：这是 `cub::BlockReduce` vs 手写 `__syncthreads` 树形归约的区别，
但比那更彻底——在 tilelang 语言层，你连 `threadIdx` 都没有。

## 3. Q3：为什么 `T.Pipelined` 只有一次迭代，`num_stages` 还置 0

【实测】`kernel.py:44,65`：`num_stages = 0 if round_scale else 2`，循环是 `for _ in T.Pipelined(1, num_stages=num_stages)`。

**(a) 为什么只有 1 次迭代**：每个 block 只搬一块数据（32×128），循环体只执行一次。
软件流水的收益**全部**来自迭代之间的重叠（第 k+1 块的 load 与第 k 块的 compute 重叠），
1 次迭代时**没有"下一块"可以预取**，多级缓冲也就无从谈起。
手册的提示是对的：*"整个 kernel 每个 block 只搬一块数据，pipelining 在这里意义不大"*。

**(b) `num_stages=0` 到底是什么意思**（手册没写清，这里补上）：
【实测】`tilelang/language/loop.py:130-133` 的文档字符串原文——
*"For compiler-inferred pipelines, if `num_stages` is 0, pipeline will not be enabled."*
配合 `src/transform/pipeline_planning.cc:1146-1174`：没有 `num_stages` 标注就直接跳过流水规划；
有标注则 `ICHECK(num_stages >= 1)`。**所以 `0` 是"明确关闭流水线"的开关，不是"深度为 0 的流水线"。**
它让这个循环退化成**普通串行循环 + 同步拷贝**，不再有 cp.async / commit / wait / 多版本缓冲这一套。

**(c) 那 `round_scale=False` 时为什么还留着 `num_stages=2`**：`T.Pipelined(1, num_stages=2)` 在
1 次迭代下买不到任何重叠，它剩下的唯一作用是**改变那次 global→shared 拷贝的发射方式**
（异步 cp.async 而非同步 ld/st）。【推断】这更像是一个"留着不碍事、还能让拷贝走上异步路径"的
保守选择，而不是有实测收益的调优——因为 loop 结束就要等它，重叠对象不存在。
**`round_scale=True` 时改成 0 的动机我无法从代码推出**，可能是：`fast_round_scale` 走的是
reinterpret + 位运算的标量路径（`kernel.py:20-33`），作者希望这条路径上编译器别再做任何
循环变换/多版本缓冲以免干扰。【未验证】——Day 8 对标工业实现或找到上游 PR 时确认。

## 4. Q4：激活侧也做量化，对"为什么推理框架喜欢融合算子"说明了什么

**证据**：

1. 【实测】`model.py:162-163`：每个 fp8 线性层的 forward 都是
   `x, scale = act_quant(x, block_size, scale_fmt); return fp8_gemm(x, scale, weight, weight.scale)`。
   → **激活量化在关键路径上**：每 token、每层、每个 linear 都要跑一次。
   而权重量化是 `convert.py` 里离线做完的，**零运行时成本**。
2. 【实测】`kernel.py:199-251` `fp8_index_kernel` 把整条 index-score 链条压进一个 kernel：
   `QK^T`（fp8 mma）→ `relu(logits) * q_scale` → `reduce_sum` → `* k_scale` → 写回 fp32。
   中间张量 `logits[blk_n2, h]` fp32 只在 fragment 里活，**一次都不落显存**。
   如果拆成 5 个 kernel，每个中间结果都要写回再读出，流量翻几倍。
3. 【实测】`model.py:479-480`：`q_scale` 被**提前折叠进 weights**
   （`weights = weights_proj(x) * n_heads**-0.5; weights = weights.unsqueeze(-1) * q_scale * softmax_scale`），
   于是 kernel 内部只需要处理 k 的 scale（`logits_sum[i3_n] *= k_s_frag[i3_n]`）。
   **scale 的应用点被挪到"能挪的最外层"**——和 `fp8_gemm` 里把 `scales_a * Scale_B` 预先乘成
   `Scale_C_shared` 是同一个思想。

**结论（三条，可直接用于面试）**：

1. **推理框架的性能预算里，elementwise 与访存流量和 GEMM 同等重要。** 量化把 GEMM 的
   TFLOPS 提上去之后，没被融合掉的 elementwise 和访存就成了新瓶颈（Amdahl）。
   这就是为什么"算子融合"是推理框架的核心竞争力，而不是锦上添花。
2. **融合的收益 = 省下的中间张量流量 × 出现频次。** act_quant 出现频次 = 层数 × token 数，
   所以哪怕每次只省 1 B/elem，总量也可观；这也是为什么它值得写成 kernel 而不是
   `torch` 逐元素操作（后者每个 op 都要单独读写一遍）。
3. **反过来说：没有融合进真实读路径的量化，收益可能是零甚至是负的。**
   这直接印证了项目三选题里那句话——"没有 memory manager 集成的量化没有吞吐收益"：
   KV cache 存成 4-bit 省了存储，但如果反量化是个独立的 kernel，
   它照样要把数据读一遍写一遍，省下的带宽又被吃回去。

## 5. 顺着读到的两件对项目有用的东西

1. **V3.2 的 KV cache 本身就是 FP8 的，而且是"量化存储 + 在读路径上直接消费"的样板**：
   【实测】`model.py:453-454` `k_cache` 是 `torch.float8_e4m3fn`、
   `k_scale_cache` 是 fp32 `[bsz, seq, head_dim//128]`（per-token × per-128-channel）；
   `model.py:474-480` 里 k 量化后直接存进 cache，`fp8_index` 直接吃量化后的 cache 和它的 scale。
   **项目三（KV cache 4-bit）可以先拿这个结构当基线**：它证明了"per-token × per-128-channel +
   读路径消费"这条路是工业界在用的，我们要回答的是"再压到 4-bit 会掉多少、以及怎么在
   paged 布局下寻址 scale"。
2. **量化前先做 Hadamard 旋转**：【实测】`model.py:428-432` `rotate_activation` 就是对 q/k 做
   `hadamard_transform(x, scale=hid^-0.5)`，在 `act_quant` **之前**调用。
   这是 QuaRot/SpinQuant 一脉的离群值抑制手法——旋转把能量摊平，amax 变小，
   量化步长 s 随之变小。**这给 Day 4 的精度实验留了一个对照臂**：per-group 是"空间维度的局部化"，
   rotation 是"把离群值摊到所有维度上"，两者是解决同一个问题的不同路线。

## 6. `fp8_gemm` 的 scale 应用方式（对照我们的 W4A16）

【实测】`kernel.py:146-166`：两个 accumulator 的结构

```python
T.clear(C_local); T.clear(C_local_accum)
for k in T.Pipelined(K_iters, num_stages=4):
    ...copy A_shared / B_shared...
    Scale_B = scales_b[bx * block_N // group_size, k]          # 标量：本 k-block 的 B scale
    for i in T.Parallel(block_M):
        Scale_C_shared[i] = scales_a[by*block_M+i, k] * Scale_B # 向量：A 的 per-token scale × B 的
    T.gemm(A_shared, B_shared, C_local, transpose_B=True)       # 裸 mma，不带任何 scale
    for i, j in T.Parallel(block_M, block_N):
        C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i] # 缩放到**另一个**累加器
    T.clear(C_local)                                             # 清掉，下一个 k-block 重新裸算
```

**为什么必须两个累加器**（结构上被数学逼出来的）：scale 沿 K 分组，`group_size=128 < K`，
所以**每个 k-block 的 scale 都不同**，`Σ_k s_k · (A_k B_k)` 里的 `s_k` 不能提到 Σ 外面——
每个 k-block 的裸乘积必须先乘自己的 scale 再加进总和。于是必须有一个"裸乘积"寄存器
（`C_local`，每轮清掉）和另一个"已缩放的运行总和"（`C_local_accum`）。
代价是**累加器寄存器翻倍** + 每个 k-block 多一遍 [block_M, block_N] 的 fp32 乘加。

【部分验证】上游注释写的是 `# Promote to enable 2xAcc`。`2xAcc` 这个字面词在 tilelang 全仓
搜不到（不是 tilelang 术语），最接近的公开概念是 DeepGEMM 的
**CUDA-core two-level accumulation（两级累加/提升）**——把 FP8 tensor core 的部分和用 CUDA core
提升到更高精度累加，用来治 FP8 mma 累加器的精度问题。也就是说这里的两个累加器
**可能同时承担两件事**：应用 per-k-block scale + 周期性"冲账"以限制 fp32 累加误差增长。
【未验证】这层精度动机是我从 DeepGEMM 的公开描述外推的，V3.2 这份代码里没有文档说明。

**为什么我们 W4A16 能反过来在 mma 之前乘**（手册让我想明白的那件事）：

| | 我们的 W4A16 | DeepSeek 的 FP8 |
|---|---|---|
| mma 的操作数 | fp16 × fp16（int4 必须**先**变成 fp16） | fp8 × fp8（硬件直接吃） |
| scale 能塞在哪 | **塞进反量化那次乘法**——反正 `w = q*s` 本来就要乘一次，空间是免费的 | fp8 张量**没有地方**吸收 fp32 scale：塞进去要么重新量化回 fp8（有损），要么把 mma 变成 fp32 仿真（丢掉 fp8 的全部意义） |
| 结果 | dequant 在 mma 前（shared→register 那一步） | scale 只能挪到 mma 之后（epilogue） |

**一句话**：scale 的应用点不是设计偏好，而是被"**操作数 dtype 有没有空间吸收 scale**" 和
"**scale 沿哪个维度分组、能否提出累加循环**"这两个约束共同决定的。
我们的 per-group scale 也沿 K 分组、同样提不出 K 循环，但我们能在 mma 前乘，
仅仅因为**反量化本来就要做一次乘法**——这是 weight-only 量化相对 activation 量化的一个结构性红利。

## 7. 手册与实际代码的差异（4 处）

| # | 手册 | 实际代码 | 影响 |
|---|---|---|---|
| 1 | 骨架只给了 `s = amax * fp8_max_inv` | 还有 `round_scale` 分支：`fast_log2_ceil` / `fast_pow2` / `fast_round_scale`（L20-33）把 scale 取成 **2 的幂**（配合 ue8m0 格式的 scale） | 手册漏了一条完整路径；而且 `num_stages` 是**跟着它**切的 |
| 2 | "查 tilelang 源码 `language/reduce.py`" | 实际是 **`language/reduce_op.py`** | 按手册路径找不到文件 |
| 3 | "TileKernels 的 `_kernel.py` / `_cuda.py` 拆分方式" | 是**三件套**：`_kernel.py`（后端无关入口）/ `_cuda.py`（`target='cuda'`）/ **`_asc.py`**（`tilelang.ascend.language`，昇腾后端） | Day 12 仓库工程化时若要对齐，多一个后端抽象层要学 |
| 4 | "`num_stages` 甚至置 0" | 语义是"**明确关闭流水线**"（loop.py:130-133） | 已补进 Q3 |

## 8. 未验证项（别当结论用）

1. **没能生成 CUDA 看真实指令**。`assets/inspect_dsv32.py` 已经写好（用 Day 2 的显式 `Target` 手法
   离线编译 act_quant / fp8_gemm），但执行它需要 import 上游 `kernel.py`，在自动模式下被权限拦下了。
   因此"blk_m=32 × 128 列必然跨多个线程"、"num_stages=2 时那次拷贝走 cp.async"这两条是
   **从布局规则和文档推断的**，没有机器码证据。有卡或有授权后跑一次该脚本即可钉死。
2. **`2xAcc` 的确切含义**（见第 6 节），以及"两级累加"的精度动机在 V3.2 这份代码里是否成立。
3. **`round_scale=True` 时置 `num_stages=0` 的真实动机**（Q3(c)）。
4. **本笔记的所有数字都没有性能含义**：全是访存/结构层面的静态分析，没有任何延迟或带宽实测。

## 参考

- `refs-deepseek-v32/inference/kernel.py` L36-196（act_quant + fp8_gemm）、L199-275（fp8_index）
- `refs-deepseek-v32/inference/model.py` L155-163（linear 融合点）、L428-432（Hadamard）、L453-480（KV cache）
- `tilelang/tilelang/language/reduce_op.py`、`tilelang/src/tl_templates/cuda/reduce.h`、`tilelang/src/transform/pipeline_planning.cc`
- `refs-tilekernels/tile_kernels/quant/`（文件组织：`cast_kernel.py` / `cast_cuda.py` / `cast_asc.py` 三件套）
