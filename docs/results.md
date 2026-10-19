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

## Day 0：模型权重（DeepSeek-Coder-6.7B-base）

- 来源：modelscope，仅 safetensors 版（`--exclude 'pytorch_model*'` 跳过重复的 bin 版，省 12.6GB）
- 路径：`$W4A16_MODEL_DIR`（本机实验使用 DeepSeek-Coder-6.7B），共约 13.48 GB
- 完整性校验：291 个 tensor 与 `model.safetensors.index.json` 完全一致，文件头可读
- 加载测试：transformers 加载通过（6.74B 参数，bf16），CPU 前向通过（80s，CPU 上正常）

## Day 1：TileLang 热身

### 步骤 3：手写 gemm_bf16（`kernels/gemm_bf16.py`）

- 布局约定：`A (M,K)`、`B (N,K)` 行主序 + `T.gemm(..., transpose_B=True)`，即 `C = A @ B^T`。
  这是 GPTQ/AWQ/Marlin 的权重量化布局，K 维连续是 Day 5+ 反量化的前提。
- 正确性：`tests/test_gemm_bf16.py`，1024³ + 3 组非方阵，pytest 2 passed。
  **非方阵用例是必需的**——B 布局写反（按 `[K,N]` 声明却按 `[N,K]` 索引）在方阵下会静默算成 `A @ B`，误差 0.0000，看不出问题。
- 运行方式：`python -m tests.test_gemm_bf16`（手册里的 `python tests/test_gemm_bf16.py` 会 ModuleNotFoundError）。

### 步骤 4：生成 CUDA 核对（`assets/gemm_bf16_sm80.cu`，4096³）

| 验收点 | 生成代码里的对应物 |
|---|---|
| `num_stages=3` 三级流水 | smem ×3 + ring 索引 `(k+2)%3` + `cp_async_wait<2>/<1>/<0>` |
| Tensor Core 形状 | `tl::mma_sync<kFloat16,kFloat16,kFloat32,16,8,16,false,true>`（m16n8k16） |
| swizzle | `tl::rasterization2DRow<10>()` 遮蔽内置 `blockIdx` |
| 线程数 | `__launch_bounds__(128, 1)` |
| `transpose_B=True` 的效果 | `ldmatrix_x4_trans` 从 day0 `[K,N]` 版的 3 降到 **0** |
| 流水结构 | clear + prologue(2 stage) + main(`k<62`) + drain(2) = 64 = K/block_K |

### 步骤 5：参数扫描（4096³ fp16，峰值 312 TFLOPS）

`benchmarks/sweep_gemm_bf16.py`：

| block_M | block_N | block_K | stages | threads | 延迟 (ms) | TFLOPS | max\|err\| vs fp64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 32 | 3 | 128 | 0.743 | 184.9 | 0.1268 ✅ |
| 128 | 128 | 64 | 3 | 128 | 0.760 | 180.8 | 0.1268 ✅ |
| **128** | **256** | **64** | **3** | **256** | **0.730** | **188.2** | 0.1268 ✅ |
| 64 | 128 | 64 | 4 | 128 | 0.854 | 160.8 | 0.1268 ✅ |

cuBLAS 参考（不计入扫描）：0.622 ms / 220.8 TFLOPS（仅性能参考，未做精度校验）。

- 三个配置落在手册预期的 180~240 TFLOPS 区间；cuBLAS 220.8，我们达到其 **85%**。
- **踩坑一：线程数必须和 tile 面积一起选。** `block_M=128, block_N=256` 配 128 线程时，累加器是 `128×256/128 = 256` 个寄存器/线程，
  超过 255 上限溢出到 local memory，实测只剩 **43.5 TFLOPS**；同样配置改 256 线程（128 寄存器/线程）后恢复到 **189.7 TFLOPS**，
  4.4× 差距，且代码一个字没改。教训：`block_M × block_N ÷ threads` 必须显著小于 255。
- **踩坑二：校验基准不能用 fp16 的 cuBLAS 结果。** 最初用 `assert_close(c, cublas_fp16, rtol=1e-2, atol=1e-2)` 判，4 个配置全部报"结果不对"。
  用 fp64 真值诊断后发现是**假警报**：
  - 4 个配置的 max\|err\| vs fp64 完全一致 = 0.1268 ≈ fp16 在 [256,512) 区间的半个 ULP（0.125），即都被**正确舍入到最近的 fp16 值**；
  - 反而是 **fp16 cuBLAS 离真值更远**（max 0.1823 / mean 0.01145，tilelang 是 0.1266 / 0.00900）；
  - 超容差的 0.076% 元素集中在 \|ref\|≈1 的小值上，那里 rtol 几乎不贡献容差，而两套实现各自的 fp32 累加序差异就已超过 atol=1e-2。
  - 结论：K 越大，`rtol/atol=1e-2` 越不能用于比较两个 fp16 实现；应改对 fp64 真值，或用 Frobenius 相对误差。

### 今日小结（按手册模板）

- 最佳配置：`block_M=128, block_N=256, block_K=64, stages=3, threads=256` → 0.730 ms，188.2 TFLOPS（4096³ fp16，cuBLAS 220.8）
- 生成代码观察（`assets/gemm_bf16_sm80.cu`，137 行）：
  - `cp.async` 首次出现在 **L33**（prologue，`tl::cp_async_gs<16>`，16 字节/线程 = 128 bit 的 `cp.async.cg`），
    主循环 `for (int k = 0; k < 62; ++k)` 在 L49，`cp_async_wait<2>` 在 L60、`<1>` 在 L83、`<0>` 在 L105；
  - mma 形状 **m16n8k16**（L76/L77 成对出现，覆盖 n8 的两个半块）；
  - swizzle 模式 `tl::rasterization2DRow<10>()` 在 **L25**，用局部变量遮蔽内置 `blockIdx`；
  - `ldmatrix_x4` 全部非转置（L68/L72），验证 `[N,K] + transpose_B` 的布局收益。
- 感悟/疑问（Day 13 写博客用）：
  1. TileLang 把"每线程寄存器数"这种底层约束留给了用户——tile 与 threads 配错时它不报错，只是默默慢 4 倍，`__launch_bounds__` 里也看不出溢出。
  2. 精度校验的"参照物"比代码本身更容易出错：把 fp16 cuBLAS 当真值，差点得出"四个配置全错"的结论。
  3. 后面 W4A16 要盯的两件事已经清楚：反量化必须发生在 shared→register 这一步（不能展开到 shared 里），以及 dequant 后的寄存器压力会直接吃掉累加器的寄存器预算。

## Day 2：BF16 GEMM 对标

> **状态：已在 A800 上跑完。** 原始数据落在 `benchmarks/results/day02_gemm_bf16_fp16.json`；
> 所有性能数字都由脚本生成。

### 步骤 1、2：`benchmarks/bench_common.py` + `benchmarks/gemm_bench.py`

- 起点配置取 Day 1 扫描的最佳：`block_M=128, block_N=256, block_K=64, num_stages=3, threads=256`。
  五个 shape 刻意共用同一份配置，这样小 M 上的浪费才能归因到"tilelang 规则 tiling"，
  而不是归因到给大 shape 特调过。
- **正确性判据换成 Frobenius 相对误差**（不是手册模板里的 `rtol/atol=1e-2`）：后者在 K=4096 上
  会假警报，Day 1 已经踩过（见上）。门限 `REL_ERR_TOL=1e-2` 的判别力实测标定：

  | 注入的缺陷 | rel_err | 判定（门限 1e-2） |
  |---|---:|---|
  | 正确实现（只有 fp16 输出舍入，即理论下限） | 2.06e-04 | 通过 ✅ |
  | 漏掉最后一个 K tile | 2.55e-01 | 拦截 ✅ |
  | 输出 N 轴索引错位 | 1.40e+00 | 拦截 ✅ |

  正确实现离门限 50 倍、真 bug 超门限 25~140 倍，两侧都有充分余量。
- 参考值 `frob_rel_err()` 按 N 维分块算，不是为提速，是为省内存（见下"环境坑"）。
- 离线校验：`python benchmarks/gemm_bench.py --check-build`，5/5 个 shape 编译通过，
  生成源码均命中 `cp_async / ldmatrix / mma_sync / rasterization`。
- 有卡实测：`python benchmarks/gemm_bench.py`，5/5 个 shape 正确性通过（Frobenius 相对误差约
  `2.07e-4`）。

  | M×N×K | TileLang ms / TFLOPS | cuBLAS ms / TFLOPS | TL/cuBLAS |
  |---|---:|---:|---:|
  | 1024×1024×1024 | 0.0443 / 48.5 | 0.0235 / 91.4 | 1.885× |
  | 4096×4096×4096 | 0.7363 / 186.7 | 0.5988 / 229.5 | 1.230× |
  | 8192×8192×8192 | 5.6007 / 196.3 | 4.6831 / 234.8 | 1.196× |
  | 16×4096×4096 | 0.0640 / 8.4 | 0.0429 / 12.5 | 1.492× |
  | 1×4096×4096 | 0.1287 / 0.3 | 0.0424 / 0.8 | 3.035× |

  手写 CUDA 第三列仍缺席：Day 2 计划允许跳过，已将它作为 Day 7 的独立 baseline，
  避免拿不同 dtype 的 SGEMM 与 FP16 GEMM 做失真的比较。
- 无卡时裸跑**硬失败**（exit 1，不写任何 JSON），不会产出看起来像真数据的空表。

### 步骤 2 顺带验证到的生成代码事实（离线可查，已核对指令级）

`M` 不被 `block_M=128` 整除时，生成代码的写回段会出现 M 边界谓词，**且只出现在写回**：

| shape | M % 128 | 写回谓词 |
|---|---:|---|
| 1×4096×4096 | 1 | 1 个，条件 `< 1` |
| 16×4096×4096 | 16 | 1 个 |
| 1024×1024×1024 | 0 | 无 |
| 4096×4096×4096 | 0 | 无 |

这是手册"小 M 时规则 tiling 会把 127/128 的计算单元浪费掉"的**机器码级证据**：
`mma_sync` 主循环无条件满跑，只有 epilogue 的 global 写回被谓词挡住——算力白烧，
所以 M=1/16 必然掉到几 TFLOPS。同时它也说明 M=1 不会越界写（谓词挡住了），
不需要为小 M 单独加保护。

### 验收结果

- [x] `benchmarks/gemm_bench.py` 两列实测 + JSON 落库
- [x] Roofline 四档写入 `benchmarks/results/day02_roofline.json`；M=1/16 是 memory-bound，
  M=256/4096 是 compute-bound，W4A16 理想带宽下界分别给出 3.87×、3.79×、2.94×、1.33×上限。
- [x] `assets/gemm_bf16_best.cu` 存档；`cp_async`、`ldmatrix`、`mma_sync`、`rasterization` 均命中。
- [x] 参数扫描 12 个 TileLang 配置落在 `benchmarks/results/day01_sweep_gemm_bf16.json`；
  11 个配置成功，1 个 `256×256×64/stages=3` 因动态 shared memory 申请失败并保留错误记录。
- [x] 三个预埋问题已回答：大 shape TileLang 达到 cuBLAS 的 81.3%–83.6%；小 M 时固定 128-row
  tile 仍满跑 MMA、只在写回做边界谓词，因此浪费计算；`tl/cublas` 随 M 变小而恶化。

### 环境坑（本次实测）

- **无卡模式跑不了对标**：`nvidia-smi` 报 "No devices were found"，`torch.cuda.is_available()=False`。
  但 nvcc 12.8 与 tilelang 编译链路完好，所以能做到"编译可验证、数字不可得"。
  `--check-build` 就是为这种状态准备的；它靠显式 `Target({"kind":"cuda","arch":"sm_80"})` 绕过
  `auto_detect_target()`（后者必须有真卡，否则直接抛
  `No registered target detector found an available target`）。
- **容器 cgroup 内存上限 2 GB**（宿主 1 TB，容器只给 2 GB；`/sys/fs/cgroup/memory.max`）。
  其中 `torch` 基线就占 462 MB、再加 tilelang 到 564 MB。踩了两次：
  ① 首次冷跑 `--check-build` 被 OOM 杀掉（exit 137，输出重定向所以日志为空，看着像"无声退出"）；
     改用隔离缓存冷跑复现不了，5/5 通过，归为 2 GB 下的偶发。
  ② 因此 8192³ 那档的 fp32 参考值必须分块算，否则 `a.float()/b.float()/ref` 三份 268 MB
     直接顶爆上限。**有卡后跑完整对标时，8192³ 是内存尖峰，盯一眼 `free -g`。**
- 无卡编译时 `nvcc` 不在 PATH（在 `/usr/local/cuda-12.8/bin`），tilelang 自己找得到，无需手动加。

## Day 3：量化基础（步骤 1-5 完成；上游生成代码兼容性已记录）

> 量化数学和黄金参考先在 CPU 完成；当前 A800 上已补跑可选的 TileLang 量化 kernel。
> 测试里的 `DEVICE` 仍做成自动回落（`cuda` → `cpu`）。

### 步骤 1：量化数学（落地产物 `docs/int4_pack_spec.md`）

- 打包格式：`byte j = (elem 2j 低半字节 | elem 2j+1 高半字节)`，`[N, K/2] uint8`（spec 已入库）
- 对称 + `[-8,7]` 补码 + per-group(128) + scale fp16；**除法/round 一律在 fp32 下做**
- 粒度账（面试素材）：scales 存储 = 权重 1/64（fp16 scale / 128 个 int4）；粒度越细越贴近局部
  动态范围，代价是元数据与反量化开销
- **对手册"激活量化常用非对称"的限定条件补充**（面试话术用）：
  非对称的收益纯粹是**范围对齐**——ReLU 类激活单边，对称会把一半格点花在不存在的负半轴，
  等价于损失 1 bit。但「非对称才能精确表示 0」是错的（对称下 real 0 也精确映射到整数 0）。
  且 Transformer 里 QKV/MLP 的输入紧跟 LayerNorm，已近似零均值对称，所以
  TensorRT-LLM / SmoothQuant 这一脉的 W8A8 反而普遍用 **per-token 对称 int8**。
  正确表述是「取决于张量的实际分布」，而不是「权重 vs 激活」这个身份。
  本项目 W4A16 激活根本不量化，不存在这个取舍。

### 步骤 2：黄金参考实现（`baseline/torch_ref.py` + `tests/test_quantize.py`）

五个函数全部入库（`compute_scales` / `quantize_q` / `pack_nibbles` / `unpack_nibbles` /
`quantize_int4_symmetric` / `dequantize_int4_symmetric`），**5 passed**
（比手册清单多一个，理由见下一节）：

| 测试 | 结果 |
|---|---|
| `test_pack_unpack_roundtrip` | 值域 [-8,7] 全 16 值严格还原（`torch.equal`）✅ |
| `test_quantize_roundtrip_error` | 平均相对误差 **0.1262**（含 20x 离群值），`max(\|err\|/(s/2)) = 1.000000` ✅ |
| `test_scale_math` | amax=7 → s=1.0，7/−7 严格还原，3.5 → 4（银行家舍入）✅ |
| `test_group_boundary` | s0=0.014282 vs s1=14.2891，比值 1000.5x ✅ |
| `test_zero_group`（新增） | 全零组 s=1.19e-07 > 0，q 严格为 0，无 NaN/垃圾值 ✅ |

- `python -m pytest tests/test_quantize.py` 与 `python -m tests.test_quantize` 两种跑法都通过。
- torch 2.7.0 的 `uint8` 左移**已可正常工作**（`nib << 4` 不报 dtype 错），
  手册"常见错误与排查"里那条 workaround 对本机版本不适用。

### ⚠️ 又一个真 bug：`clamp(min=1e-10)` 在全零组上产出 INT32_MIN 垃圾值

**测试从 4 个加到 5 个**（`test_zero_group`，依据是今日目标 4 要求的"边界测试"；
手册清单写的是"四个测试全部通过"，我加了第 5 个并在这里说明）。它抓到了一个照抄手册
就会带进 kernel 的缺陷：

```
w = zeros(1,256); w[0,5] = 0.3     # 第 1 组全零
floor=1e-10:  s = [0.0429, 0.0]   →  q.min() = -2147483648   ❌
floor=1e-07:  s = [0.0429, 1.19e-07] → q.min() = 0           ✅
```

推理链：`1e-10` 在 **fp32** 里是正数，但 scale 要 **cast 成 fp16 存**，
而 fp16 的最小 subnormal ≈ 6.1e-8 → `1e-10.half() == 0.0`。
于是全零组 `s=0` → `quantize_q` 里 `0/0 = nan` → `clamp(nan) = nan` →
`.to(torch.int32)` 得到 **INT32_MIN**。打包那一步恰好因 INT32_MIN 的低半字节是 0 而
"看起来没错"（反量化结果仍是 0），所以**单看往返误差完全发现不了**——
但 kernel 侧 `x/0` 会直接算成 inf/nan，行为和 torch 对不上。

- 这个 floor 是从 DeepSeek 抄来的，但**他们的 scale 是 fp32**（`scale_dtype=FP32`），
  floor 够小就行；我们存 fp16，floor 必须按**存储 dtype** 选。→ 改成 `1e-7`
  （略高于 fp16 最小 subnormal），已写进 `docs/int4_pack_spec.md` 不变量 5。
- 影响面实测：真实权重 40 万个组里最小的 amax = **3.6e-3**，三种候选 floor
  （1e-7 / 6.1e-5 / 1e-4）算出的 rel 误差在小数点后 6 位**完全相同**——floor 永不触发，
  纯属防御。修完真实权重 rel = 0.138638，与修复前逐位一致（无精度退化）。
- 顺带一个测试写法教训：不能用 `q.abs().max()` 判越界，**INT32_MIN 取绝对值会溢出成负数**，
  恰好通过 `max <= 8`。要用显式 `q.min() >= -8 and q.max() <= 7`。

### ⚠️ 手册门限修正：INT4 往返相对误差 ~13% 是数学必然，不是 bug

手册的 `assert rel < 0.05`（"INT4 典型往返误差在 1%~5%"）**标错了**，把正确的实现判成失败。
诊断过程（`/tmp/diag_quant.py`，分块跑以避开容器 2GB cgroup 上限）：

| 场景 | 实测 rel | 理论 `mean(s)/4 / mean\|w\|` | 实测/理论 | `max(\|err\|/(s/2))` 越界元素 |
|---|---:|---:|---:|---:|
| 合成高斯 `randn(256,1024)*0.02` | 0.1260 | 0.1272 | **0.991** | 0 / 0 |
| 上行 + 20x 离群值 | 0.1262 | — | — | 0 |
| **真实权重**（DeepSeek-Coder-6.7B gate_proj `[11008,4096]` bf16） | **0.1390** | — | — | 0 |

推理链：round-to-nearest 的均匀格点误差是 `mean|err| = s/4`，于是组内分布平滑时
`rel ≈ (mean(s)/4) / mean|w|`——高斯数据实测与理论吻合到 0.9%。
真实权重**更差**（13.9%）而非更好，因为真实权重重尾：组内 `amax/mean|w| = 153`，
而 `s = amax/7` 本身就与 `mean|w|` 同量级。
结论：`1%~5%` 那个数字对应的是别的量（极可能是端到端 PPL 退化，或 INT8 的数字——INT8 换算是 16 倍小 ≈ 0.8%）。

**判据改成两层，比原门限更强**（写进测试注释，不依赖特定分布）：

1. **硬界（分布无关）**：每个元素 `|err| ≤ s_group/2`——取整永远不越过半个步长。
   scale 算错、分组切错、符号扩展错、位序错都会立刻越界。这条是真正能抓 bug 的判据。
2. **理论锚定**：`rel` 与理论值同量级（±25%），防止"误差大得离谱却恰好没越界"。

> 这和 Day 1 的教训是同一个genre：**参照物比代码本身更容易出错**。
> 当时是拿 fp16 cuBLAS 当真值，这次是拿一个凭印象写的门限当真值。
> Day 4 做精度实验时，`1%~5%` 这个数字的来源要在笔记里查清楚，不要直接引用。

### 顺带测到的粒度对照（Day 4 的系统性实验提前拿到一点锚点）

真实权重（同上 tensor）per-tensor / per-channel / per-group 的 rel 误差：

| 粒度 | rel 误差 |
|---|---:|
| per-tensor | 0.9593 |
| per-channel（per-row） | 0.2546 |
| **per-group(128)** | **0.1390** |

per-tensor 的 96% 就是手册说的"一个离群值毁掉全矩阵"。Day 4 画精度-开销曲线时可直接复现这张表。

### 步骤 3：两个参考仓库已 clone

| 仓库 | 路径 | commit | 日期 | 大小 | 用途 |
|---|---|---|---|---|---|
| DeepSeek-V3.2-Exp | `project1/refs-deepseek-v32` | `87e509a` | 2025-11-18 | 1.4M | **读**（A800 无 FP8，跑不了） |
| TileKernels | `project1/refs-tilekernels` | `c1cf914` | 2026-10-14 (v2.1.0) | 2.0M | 读结构，Day 12 对齐测试组织 |

两个都是 `--depth 1`。**发现**：TileKernels 的 quant 模块不是手册说的
`_kernel.py / _cuda.py` 两件套，而是**三件套**——`_kernel.py`（后端无关入口，按 `is_ascend()` 分发）
/ `_cuda.py`（`@tilelang.jit(target='cuda')`）/ **`_asc.py`**（`tilelang.ascend.language`，昇腾后端）。
Day 12 仓库工程化时若要对齐这套组织，多一层后端抽象要学。
测试组织是 `tests/<模块>/test_<kernel>.py` 一算子一文件，且用
`tile_kernels/testing/` 下的公共设施（`bench` / `numeric`(`assert_equal`/`count_bytes`) / `generator`）——
Day 12 照这个来。

### 步骤 4：`act_quant_kernel` 精读 → `docs/notes_act_quant.md`

四个问题各有一节答案，另一节对比 `fp8_gemm` 的 scale 应用方式。要点提炼：

1. **数据流**：act_quant 是**宽读窄写**（bf16 2 B → fp8 1.03 B/元素，约 1.9:1），
   我们 Day 4 的反量化 kernel 是**窄读宽写**（0.52 B → fp16 2 B，约 1:3.9），**方向相反**。
   共同点：都是 memory-bound 的 elementwise + 行内归约，都沿最后一维按 **128** 分组（同一条精度-开销曲线的取点）。
   根本区别：**激活动态、权重静态**——所以 dequant 能融合进 GEMM，act_quant 不能。
2. **为什么用 `T.reduce_absmax`**：语言层**没有线程概念**。`reduce_op.py:159-169` 在
   fragment+fragment 时只发一条 tile 级 intrinsic（参数是逻辑 tile 区域，不是线程）；
   lowering 才变成 `tl::AllReduce`（`reduce.h:211-253`，递归 XOR-butterfly：
   **offset≥32 走 smem+barrier，<32 走 shfl_xor_sync**）。所以手写 `T.Parallel` 归约是
   **无保护的多线程读改写（race）**——错在"这一层还没有线程"，不是错在"写得不够优化"。
   更隐蔽的一层：归约结果 fragment 的布局由编译器推断，手写不可能对上。
3. **`T.Pipelined(1)`**：1 次迭代没有"下一块"可重叠，流水线收益为零。
   手册没说清的是 `num_stages=0` 的**语义**：`loop.py:130-133` 原文——
   *"if `num_stages` is 0, pipeline will not be enabled"*，即"明确关闭流水线"，
   让循环退化成串行 + 同步拷贝（不是"深度为 0 的流水线"）。
4. **融合算子**：act_quant 在**关键路径**上（每 token × 每层 × 每个 linear，`model.py:162-163`），
   而权重量化是离线一次性的。量化把 GEMM 提速后，没被融合掉的 elementwise/访存成了新瓶颈——
   这就是"没有融合进真实读路径的量化收益为零"（直接印证项目三选题）。

**顺带挖到两件对项目有用的**（详见笔记第 5 节）：
① V3.2 的 **KV cache 本身就是 FP8 e4m3 + per-token × per-128-channel fp32 scale**，
且被 `fp8_index` 在**读路径上直接消费**（`model.py:453-454, 474-480`）——项目三可以拿它当工业基线；
② 量化前先做 **Hadamard 旋转**抑制离群值（`model.py:428-432`，QuaRot/SpinQuant 一脉），
这给 Day 4 的精度实验留了个对照臂：per-group 是"空间局部化"，rotation 是"把离群值摊平"。

**手册 vs 实际代码 4 处差异**（笔记第 7 节）：`num_stages` 跟着 `round_scale` 切（手册骨架漏了
整条 e8m0/2 的幂 scale 路径）、tilelang 的归约源码是 `reduce_op.py` 不是 `reduce.py`、
TileKernels 是三件套、`num_stages=0` 的确切语义。

**未验证项已单列**（笔记第 8 节）：`assets/inspect_dsv32.py` 写好了（Day 2 的显式 `Target` 离线编译手法），
但执行它要 import 上游 `kernel.py`，在自动模式下被权限拦下 → 所以
"一行 128 个元素必然跨多线程"、"num_stages=2 时那次拷贝走 cp.async"这两条目前是**推断**，
有待机器码证据。**有授权后跑一次即可钉死。**

### 完成情况

- [x] 步骤 5（选修）：`kernels/quantize_int4.py` 与 torch reference 对照通过；CUDA round tie
  的极少数差异按解包后的最大 ±1 量化步记录。
- [x] 已执行 `python assets/inspect_dsv32.py`；上游 kernel 在 TileLang 0.1.15 eager builder 中触发
  `act_quant_kernel() missing N` / `fp8_gemm_kernel() missing N,K`，兼容性限制已写入笔记，不把失败误记为本项目 kernel 失败。
- [x] commit：`day03: w4a16 torch golden ref + int4 pack/unpack spec + act_quant reading notes`

## Day 4：反量化 kernel 与精度实验（2026-10-20）

### 步骤 1：TileLang 反量化正确性

- `kernels/dequant_int4.py` + `tests/test_dequant_int4.py` 已完成。
- 4/4 shape 通过：`256×1024/g128`、`4096×4096/g128`、`11008×4096/g128`、
  `4096×11008/g64`；所有 case 相对 torch 黄金版 `max_err=0`。
- 完整回归：`python -m pytest -q tests/test_gemm_bf16.py tests/test_quantize.py tests/test_dequant_int4.py`
  → 11 passed。

### 步骤 2：带宽

`benchmarks/bench_dequant.py` 运行于 A800（理论 2039 GB/s），结果写入
`benchmarks/results/day04_dequant_bw.json`：

| shape | 延迟 | 等效带宽 | 理论带宽利用率 |
|---|---:|---:|---:|
| 4096×4096/g128 | 0.0406 ms | 1039.3 GB/s | 51.0% |
| 11008×4096/g128 | 0.0922 ms | 1230.3 GB/s | 60.3% |
| 4096×11008/g128 | 0.0991 ms | 1144.1 GB/s | 56.1% |

kernel 是窄读宽写的 memory-bound v1；TileLang 报告 packed byte 的 extent=2 未找到有效向量化方案，
后续可沿连续 8 元素解包、half2/16-byte store 方向优化。

### 步骤 3：真实权重精度

`scripts/accuracy_experiment.py` 读取 DeepSeek-Coder-6.7B-base 第 10 层三张权重，
`scripts/plot_accuracy.py` 生成 `assets/figs/accuracy_vs_group.png`。表中为 relative L2 error：

| 张量 | per-tensor | per-channel | group32 | group64 | group128 |
|---|---:|---:|---:|---:|---:|
| q_proj | 88.082% | 18.914% | 9.859% | 11.040% | 12.179% |
| gate_proj | 74.819% | 16.549% | 9.754% | 10.852% | 11.873% |
| down_proj | 99.205% | 17.980% | 9.823% | 10.944% | 11.989% |

group32 在三张权重上都是最低误差，但 scale 元数据开销为 12.5%；group128 只需 3.125%，
误差约 11.9%–12.2%。per-tensor 的 74.8%–99.2% 直接验证了离群值灾难。

### 步骤 4：FP8 对照

`docs/notes_quant_compare.md` 已填写完毕，记录了均匀 INT4 格点与 FP8 浮点格点的差异、
scale 应用位置，以及为什么静态权重能在 GEMM 内融合而动态激活需要在线量化。

### Day 3 选修项与上游生成代码

- TileLang INT4 量化 kernel 已完成：`kernels/quantize_int4.py`，测试 `tests/test_quantize_kernel.py` 通过；
  由于 CUDA `round` 与 torch 在极少数 tie 上可能差一个量化步，测试解包后允许最大 ±1，scale 逐位一致。
- `assets/inspect_dsv32.py` 已执行。DeepSeek 上游 kernel 在本地 TileLang 0.1.15 的 eager builder 中因
  `act_quant_kernel() missing N` / `fp8_gemm_kernel() missing N,K` 兼容性错误无法生成 CUDA；这不是本项目
  kernel 的失败，保留为上游版本差异记录，`docs/notes_act_quant.md` 的两项机器码结论仍标为推断。

## Day 5：W4A16 dequant GEMM v1（2026-10-20）

### 步骤 1–3：融合推导、官方示例和 v1 smoke

- 步骤 1：`group_size=128` 时，两步走的权重侧理论流量约 `4.515625 B/element`，融合路径约 `0.515625 B/element`；该计算只统计权重侧，不包含激活和输出。
- 步骤 2：`../tilelang/examples/dequantize_gemm/example_dequant_gemm_fine_grained.py` 存在，`py_compile` 通过。
- 步骤 3：`kernels/dequant_gemm_w4a16.py` 和 `kernels/dequant_int4.py` 的 `py_compile` 通过。
- 最小 smoke：`M=256,N=512,K=1024,group_size=128`，相对 torch reference 的平均相对差 `1.665e-4`，最大绝对差 `0.00024414`，通过。
- 原始日志：`docs/day05_steps123.log`。

### 步骤 4：三级正确性

- 环境：A800 80GB PCIe（SM80），CUDA 12.8，PyTorch 2.7.0+cu128，TileLang 0.1.15。
- 测试脚本：`tests/test_dequant_gemm.py`。
- 日志：`docs/day05_correctness.log`。
- 测试 shape：`256×512×1024 (g=128)`、`1024×4096×4096 (g=128)`、`256×4096×4096 (g=64)`、`512×11008×4096 (g=128)`。
- 结果：4 组 shape 均通过 torch reference 和 two-kernel reference。
- 相对 torch reference：平均相对差约 `1.68e-4`，最大绝对差 `0.000488`。
- 相对 two-kernel reference：4 组均为 `0.000000`，最大绝对差 `0.000000`。

### 步骤 5：v1 性能起点

- kernel：`dequant_gemm_v1`。
- shape：`M=N=K=4096`，`group_size=128`。
- 配置：`block_M=64`、`block_N=64`、`block_K=128`、`num_stages=2`、`threads=128`。
- 计时：`tilelang.profiler.do_bench`，`warmup=25 ms`、`rep=100 ms`。
- 稳态延迟：`1.765 ms`。
- 按 `2*M*N*K/time` 计算：`77.852 TFLOPS`。
- 首次设备调用：`382.376 ms`，不作为稳态性能。
- 原始日志：`docs/day05_benchmark.log`。
- JSON：`benchmarks/results/day05_dequant_gemm_v1.json`。
- 编译提示：TileLang 报告 extent=2 的 vectorized loop 未找到有效向量化方案，最终按 serial loop 降级；这项优化留到 Day 6 处理。

## Day 6：W4A16 v2 性能优化（2026-10-20）

### 实现与 autotune

- `dequant_gemm_v2` 把每个 K tile 的 scale 先搬入 shared，再解包 Q；`T.use_swizzle(panel_size=10)` 和 Q shared layout swizzle 均提供为可复现实验开关。
- `benchmarks/sweep_dequant_v2.py` 在 `128×1024×1024` 上完成 7 个 block/stage 配置，7/7 正确性通过。
- TileLang `@autotune` 实跑 8 个配置，最佳配置为 `block_M=64, block_N=64, block_K=128, num_stages=2, threads=128`，延迟 `0.0260096 ms`。完整候选和日志摘要见 `benchmarks/results/day06_autotune.json`。

### 多 shape 对标（A800，g=128）

| 场景 | shape | v1 ms | v2 ms | v2+block swizzle ms | 两步走 ms | cuBLAS fp16 ms |
|---|---|---:|---:|---:|---:|---:|
| decode | 1×4096×4096 | 0.08269 | 0.07877 | 0.07927 | 0.06733 | 0.04287 |
| decode | 16×4096×4096 | 0.08383 | 0.07958 | 0.08019 | 0.06762 | 0.04277 |
| small-batch | 128×4096×4096 | 0.10060 | 0.10089 | 0.09944 | 0.07680 | 0.04731 |
| prefill | 512×4096×4096 | 0.27742 | 0.28086 | 0.28025 | 0.12504 | 0.09114 |
| prefill | 1024×4096×4096 | 0.47009 | 0.46688 | 0.46709 | 0.20463 | 0.16986 |
| prefill-big | 4096×4096×4096 | 1.77607 | 1.75480 | 1.75071 | 0.66569 | 0.63746 |
| ffn-up | 1024×11008×4096 | 1.23183 | 1.20505 | 1.20486 | 0.50261 | 0.42456 |
| ffn-down | 1024×4096×11008 | 1.30951 | 1.22273 | 1.22916 | 0.52760 | 0.43987 |

这轮实测没有达到手册预设的 2×：scale 预取在大多数 shape 上是约 0.9% 的收益，rasterization swizzle 在 `4096³` 上再带来约 0.2%。原因是 scale 张量很小且已经有较好的 cache 命中，新增 shared copy 与同步成本抵消了大部分收益；下一步应改为 register dequant/magic layout，而不是继续堆 shared 缓存。

原始 JSON：`benchmarks/results/day06_w4a16_all.json`、`day06_v2_sweep.json`、`day06_autotune.json`。

## Day 7：手写 CUDA dequant GEMM（2026-10-20）

- 新增 `baseline/cuda/dequant_gemm.cu`、binding 和 loader；实现 BM=BN=16、BK=32 的 shared-memory、逐线程标量累加 baseline。
- 正确性：`tests/test_dequant_gemm_cuda.py` 的 3 组 shape 通过，平均相对差约 `1.68e-4`。
- 4096³：手写 CUDA `67.08633 ms / 2.049 TFLOPS`，TileLang v2 `1.75474 ms / 78.324 TFLOPS`。差距来自未使用 cp.async、多级流水、寄存器分块和 MMA，完整对比见 `docs/notes_cuda_vs_tilelang.md`。
- 5 个 shape 的实测 JSON：`benchmarks/results/day07_cuda_vs_tilelang.json`。

## Day 8：工业 int4 对标（2026-10-20）

- PyTorch `torch._convert_weight_to_int4pack` + `torch._weight_int4pack_mm` 路径 A 已跑通；Torch 2.7 的 CUDA 实现要求 bf16 激活、`innerKTiles=8`，scale/zero 形状为 `[K/group,N,2]`。
- 4096³ 的 decode（M=1）路径 A 为 `0.01804 ms`；M=16 为 `0.04504 ms`；M=512 为 `0.93610 ms`。这些数字和本项目 row-major signed-nibble v2 不同 layout、不同激活 dtype，单独标为工业参考。
- Marlin 的差距归因和未实测边界见 `docs/results_industry.md`。

## Day 9：decode GEMV 与带宽（2026-10-20）

- 新增 `kernels/gemv_w4a16.py`；3 组正确性 shape 的平均相对差小于 `1.1e-6`。
- A800 理论值（N=K=4096）：fp16 权重约 33.6 MB、带宽下界约 16.5 μs；W4A16 packed 权重+scale 约 8.6 MB、带宽下界约 4.2 μs，字节上限约 3.9×。

| N×K | GEMV 延迟 | GEMV 有效带宽 | A800 理论带宽利用率 | v2@M=1 | 两步走 | cuBLAS fp16 |
|---|---:|---:|---:|---:|---:|---:|
| 4096×4096 | 96.82 μs | 89.518 GB/s | 4.39% | 78.98 μs | 67.02 μs | 42.62 μs |
| 11008×4096 | 102.90 μs | 226.230 GB/s | 11.10% | 93.59 μs | 166.61 μs | 79.72 μs |
| 4096×11008 | 288.61 μs | 80.659 GB/s | 3.96% | 198.85 μs | 174.53 μs | 82.51 μs |
| 8192×8192 | 228.37 μs | 151.665 GB/s | 7.44% | 178.55 μs | 244.52 μs | 111.56 μs |

官方 TileLang fp16xint4 GEMV 示例（8192²、无 scale 的参考口径）实测 `0.036570 ms`，约 45% 的 packed-weight 带宽利用率。自研一列只有 4%–11%，说明每线程一列的非连续行访问是主要瓶颈；后续应采用官方的 warp 级 K 归约、16B 向量装载和寄存器解包。

原始 JSON：`benchmarks/results/day09_decode.json`。

## Day 10：MoE 场景量化（2026-10-20）

- Qwen1.5-MoE-A2.7B checkpoint 需要约 29 GB bf16，而当前工作区低于 35 GB 安全阈值，因此用确定性的 ToyMoE fallback 完成同一专家 Linear 结构的量化、PPL proxy、GEMV 和权重字节预算；边界已写进 `docs/moe_quant.md`。
- 8 experts、top-2、hidden=512、intermediate=1024；g=128 的专家 rel L2 均值 `0.07060`，g=64 为 `0.06979`。
- top-2 理论权重流量：bf16 `6,291,456 B`，W4A16 `1,622,016 B`，约 `3.8788×` 压缩；逐专家 GEMV 与 toy 端到端 control 原始数据见 `benchmarks/results/day10_moe.json`。

## Day 11：DeepSeek-Coder-6.7B 端到端验证（2026-10-20）

- 新增 `W4A16Linear`，按 M=1/GEMV、M>1/桶化 GEMM 路径替换 225 个 Linear，0 个层被跳过。
- fp16 基线：12.563 GiB 已分配显存、48.147 token/s。
- W4A16：3.429 GiB 已分配显存、31.793 token/s；显存下降 72.7%，吞吐下降 34.0%。
- 小样本代码 PPL：2.8262 → 2.9551，增加 4.56%。这是 128 token 参考评估，不是完整任务质量结论。
- 归因：当前自研 GEMV 只有约 4%–11% A800 理论带宽利用率，且全模型仍包含 attention、KV cache、norm、采样和 Python/module 调度；因此不能把显存收益写成服务吞吐收益。
- 原始结果：`benchmarks/results/day11_e2e.json`、`day11_ppl.json`；完整口径见 `docs/results_e2e.md`。
