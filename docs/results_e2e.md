# Day 11：DeepSeek-Coder-6.7B 端到端验证

## 测试口径

- 框架：HF Transformers `generate`。
- 并发：单请求，batch=1，无 continuous batching。
- 采样：贪心解码，`do_sample=False`。
- 生成：固定英文代码 prompt，8 个新 token，1 次 warmup、2 次正式测量。
- 硬件：NVIDIA A800 80GB，CUDA 12.8，PyTorch 2.7.0+cu128，TileLang 0.1.15。
- 模型：DeepSeek-Coder-6.7B，225 个 Linear 全部替换，0 个跳过。

## 结果

| 指标 | fp16 | W4A16 | 变化 |
|---|---:|---:|---:|
| 已分配显存 | 12.563 GiB | 3.429 GiB | -72.7% |
| 峰值显存 | 12.578 GiB | 3.444 GiB | -72.6% |
| 生成吞吐 | 46.595 token/s | 31.346 token/s | -32.7% |
| 吞吐标准差 | 0.050 | 0.036 | 方差低于 0.2% |
| 小样本 PPL（128 token） | 2.8262 | 2.9551 | +4.56% |

样例输出均保持代码补全结构；W4A16 输出只截取为日志中的前 500 字符。原始数据位于 `benchmarks/results/day11_e2e.json` 和 `day11_ppl.json`。

## 归因与边界

这次端到端结果的主要收益是权重显存下降，吞吐反而下降。原因是当前自研 GEMV 的带宽利用率只有约 4%–11%，低于 cuBLAS fp16 的有效带宽；225 个 Linear 的每次调用还要承担 Python/module 调度，而 attention、KV cache、norm 和采样都没有变化。这个数字是 HF 单请求实验，不能外推到 vLLM 服务吞吐。

PPL 上升 4.56% 来自只使用一段短代码文本的参考性评估，不能替代 WikiText 或任务级评测。它说明本次全模型替换可以完成，但质量仍需要更大样本和更高带宽的 GEMV v2 再评估。

## 复现

```bash
export W4A16_MODEL_DIR=/path/to/deepseek-coder-6.7b-base
python scripts/bench_e2e.py --new-tokens 8 --warmup 1 --repeats 2
python scripts/ppl_eval.py --max-tokens 128
```
