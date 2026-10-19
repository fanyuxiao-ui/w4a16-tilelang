# Day 10：MoE 专家权重 W4A16 验证

## 实验边界

计划首选 Qwen1.5-MoE-A2.7B，但该 checkpoint 的 bf16 权重约 29 GB，而当前工作区低于 35 GB 安全阈值。为保证当天所有量化、PPL、单层 decode 和字节预算步骤都真实运行，本次使用确定性的 ToyMoE：8 个专家、top-2、hidden=512、intermediate=1024、vocab=256。代码沿用真实 MoE 的 `gate_proj/up_proj/down_proj` 专家结构；结果标记为 fallback，不能解读为 Qwen 的模型级 PPL。

## 量化精度

| 配置 | 专家矩阵数量 | rel L2 均值 | rel L2 最大 | PPL proxy | 相对 bf16 |
|---|---:|---:|---:|---:|---:|
| bf16 | 24 | — | — | 255.2253 | — |
| W4A16, g=128 | 24 | 0.07060 | 0.07075 | 255.1216 | -0.0406% |
| W4A16, g=64 | 24 | 0.06979 | 0.06993 | 255.0381 | -0.0734% |

PPL proxy 在随机 token/label 上只用于确认同一批输入下的前向行为；它不是 WikiText PPL。权重量化的逐元素误差和模型 PPL 必须分开报告。

## decode 权重侧预算

ToyMoE 的一个专家有 3 个矩阵，top-2 激活时：

| 口径 | bf16 | W4A16 |
|---|---:|---:|
| 理论权重字节 | 6,291,456 B | 1,622,016 B |
| 压缩比例 | — | 3.8788× |
| 单专家 GEMV | 脚本中的 `bf16_ms` | 脚本中的 `w4a16_ms` |
| 整个 ToyMoE | 6.2256 ms | 6.2244 ms（专家暂存为 dequantized fp16 的 control） |

单层的 W4A16 GEMV 直接消费 packed Q/scale；整模型 control 为了隔离模型路由开销，使用反量化后的 fp16 专家权重，因此不能当成 packed kernel 的端到端加速数字。

## 运行

```bash
python scripts/moe_quant_validation.py
```

原始逐层数据写入 `benchmarks/results/day10_moe.json`。若以后提供 Qwen checkpoint，只需在同一脚本中把 `ToyMoE.experts` 替换为模型的专家模块，再保持 group size、文本和长度不变即可复用记录格式。
