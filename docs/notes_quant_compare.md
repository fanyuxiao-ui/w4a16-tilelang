# W4A16（per-group INT4）与 DeepSeek FP8（block-wise）对照

## 格式与数据流

| 维度 | 本项目 W4A16 | DeepSeek V3.2 FP8 |
|---|---|---|
| 目标 dtype | 对称 INT4 补码，值域 `[-8, 7]` | e4m3 FP8，浮点格点 |
| scale 粒度 | 权重每行每 128 个 K 元素（1×128） | 激活每 token 每 128 个元素（1×128）；训练权重使用 128×128 block |
| scale 类型 | FP16 | 推理 kernel 使用 FP32，部分部署使用 e8m0 |
| scale 应用点 | `q * scale` 后进入 FP16 MMA；Day 5 会把它融合进 GEMM | FP8 MMA 后按每个 K block 缩放累加贡献 |
| 硬件路径 | A800/SM80 上走 FP16 MMA，INT4 作为存储格式 | FP8 Tensor Core 需要更新的 GPU 架构；A800 只做代码阅读 |
| 核心代价 | 均匀 4-bit 格点遇到重尾权重时需要更细分组 | FP8 格点随指数变化，对离群值更友好，但每元素占用更多位 |

INT4 的量化误差来自固定步长 `s = amax / 7`。在组内分布较平滑时，round-to-nearest 的平均绝对误差约为
`s/4`；因此减小 `group_size` 会降低误差，同时增加 scale 元数据和反量化读取。FP8 的指数让格点密度随数值范围变化，
对离群值更宽容，但它不能直接替代本项目的 4-bit 存储目标。

## 真实权重实验

数据来自 DeepSeek-Coder-6.7B-base 的第 10 层 `q_proj`、`gate_proj`、`down_proj`。运行
`python scripts/accuracy_experiment.py` 后，结果写入 `benchmarks/results/day04_accuracy.json`，
曲线由 `python scripts/plot_accuracy.py` 写入 `assets/figs/accuracy_vs_group.png`。

| 张量 | per-tensor | per-channel | group32 | group64 | group128 |
|---|---:|---:|---:|---:|---:|
| q_proj | 88.082% | 18.914% | 9.859% | 11.040% | 12.179% |
| gate_proj | 74.819% | 16.549% | 9.754% | 10.852% | 11.873% |
| down_proj | 99.205% | 17.980% | 9.823% | 10.944% | 11.989% |

元数据开销为 `4 / group_size`：group128 为 3.125%，group64 为 6.25%，group32 为 12.5%。这次实测中，三张权重都呈现
group32 误差最低（约 9.75%–9.86%），group64/128 依次升高；per-tensor 则被离群值推到 74.8%–99.2%。实验结论应同时看
误差、余弦相似度和这项开销，而不能只挑误差最低的粒度。

## 从 act_quant 迁移来的设计结论

1. DeepSeek 的 `act_quant_kernel` 和本项目反量化都沿最后一维按 128 分组，但数据流方向相反：前者宽读窄写，
   后者窄读宽写。两者都属于低算术强度的访存型算子。
2. 激活是运行时动态数据，scale 需要在线归约；权重是静态数据，scale 可以离线生成并在 GEMM 的 shared→register
   阶段融合。这是 W4A16 能把额外乘法藏进主计算路径的结构性原因。
3. FP8 的浮点格点对重尾分布更友好；INT4 依靠 per-group 把离群值局部化。精度曲线就是“用 scale 元数据换动态范围”的
   直接测量。
