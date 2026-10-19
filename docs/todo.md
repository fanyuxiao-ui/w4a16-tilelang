# TODO / Future Work

## 明确知道路线

- [ ] Marlin 式 magic layout：离线置换 Q，使 MMA fragment 所需 nibble 连续，进一步把解包和 scale 留在寄存器路径。
- [ ] Warp 级 GEMV：参考官方 fp16xint4 示例的 K 归约、16B 向量装载和 LOP3 解包，把当前 4%–11% 带宽利用率提高到可用水平。
- [ ] Grouped MoE W4A16：在真实 Qwen MoE checkpoint 可用时替换 ToyMoE fallback，补全专家路由和端到端质量评估。
- [ ] H100 FP8 迁移：复用测试框架，重新选择 dtype、MMA 和 scale 应用位置。

## 已知限制

- Day 11 的 E2E 结果是 HF Transformers、batch=1、贪心解码，不代表 vLLM/SGLang 服务吞吐。
- Day 10 的 ToyMoE PPL proxy 不是 Qwen 模型 PPL。
- 工业 int4pack 使用 bf16 和不同 packed layout，只能作为参考，不能当作同格式正确性对照。
