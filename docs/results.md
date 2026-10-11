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
- 路径：`/root/autodl-tmp/models/deepseek-coder-6.7b-base`，共 13.48 GB
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

| config | 延迟 ms | TFLOPS | max\|err\| vs fp64 真值 |
|---|---|---|---|
| bm128 bn128 bk32 st3 th128 | 0.743 | 184.9 | 0.1268 ✅ |
| bm128 bn128 bk64 st3 th128 | 0.760 | 180.8 | 0.1268 ✅ |
| **bm128 bn256 bk64 st3 th256** | **0.730** | **188.2** | 0.1268 ✅ |
| bm64 bn128 bk64 st4 th128 | 0.854 | 160.8 | 0.1268 ✅ |
| cuBLAS（仅性能参考） | 0.622 | 220.8 | — |

- 三个配置落在手册预期的 180~240 TFLOPS 区间；cuBLAS 220.8，我们达到其 **85%**。
- **踩坑一：线程数必须和 tile 面积一起选。** `bm128 bn256` 配 128 线程时，累加器是 `128×256/128 = 256` 个寄存器/线程，
  超过 255 上限溢出到 local memory，实测只剩 **43.5 TFLOPS**；同样配置改 256 线程（128 寄存器/线程）后恢复到 **189.7 TFLOPS**，
  4.4× 差距，且代码一个字没改。教训：`block_M × block_N ÷ threads` 必须显著小于 255。
- **踩坑二：校验基准不能用 fp16 的 cuBLAS 结果。** 最初用 `assert_close(c, cublas_fp16, rtol=1e-2, atol=1e-2)` 判，4 个配置全部报"结果不对"。
  用 fp64 真值诊断后发现是**假警报**：
  - 4 个配置的 max\|err\| vs fp64 完全一致 = 0.1268 ≈ fp16 在 [256,512) 区间的半个 ULP（0.125），即都被**正确舍入到最近的 fp16 值**；
  - 反而是 **fp16 cuBLAS 离真值更远**（max 0.1823 / mean 0.01145，tilelang 是 0.1266 / 0.00900）；
  - 超容差的 0.076% 元素集中在 \|ref\|≈1 的小值上，那里 rtol 几乎不贡献容差，而两套实现各自的 fp32 累加序差异就已超过 atol=1e-2。
  - 结论：K 越大，`rtol/atol=1e-2` 越不能用于比较两个 fp16 实现；应改对 fp64 真值，或用 Frobenius 相对误差。

### 今日小结（按手册模板）

- 最佳配置：bm128 bn256 bk64 stages3 threads256 → 0.730 ms，188.2 TFLOPS（4096³ fp16，cuBLAS 220.8）
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
（待填）

## Day 4：量化精度实验
（待填）

...
