# 面试问答演练记录

以下答案都能回到仓库中的代码或 JSON；数字使用 A800、CUDA 12.8、PyTorch 2.7.0+cu128、TileLang 0.1.15 的本机结果。

1. **为什么做 W4A16？** A800 的 SM80 没有 H100 的 FP8 Tensor Core，因此使用 INT4 权重、FP16 激活；权重读流量下降，decode 更容易受益。
2. **scale 粒度怎么选？** 以误差、scale 元数据和 kernel 延迟共同决定；本项目主链路用 group size 128，另测了 64。
3. **dequant 放在哪里？** 当前 v2 在 Q shared 后解包并乘 scale，再交给 `T.gemm`；Marlin 式下一步是寄存器直通。
4. **为什么要流水线？** 让下一块 K tile 的 global→shared 搬运与当前块 MMA 重叠；级数受 shared memory 和寄存器预算限制。
5. **CUDA 与 TileLang 如何公平比较？** 相同 Q、scale、shape、dtype、warmup、迭代和同步；编译时间单独排除。
6. **bank conflict 在哪里？** 重点检查 Q/W tile 的线程映射和 stride；相邻 nibble 共用 byte 本身不等于 bank conflict。
7. **Marlin magic layout 解决什么？** 预先按 MMA fragment 需要的顺序重排权重，减少跨线程交换和 shared 往返。
8. **decode 的实测瓶颈？** 当前 4096² GEMV 为 89.5 GB/s，仅占 A800 理论带宽 4.39%；访问映射和归约是主要问题。
9. **换 H100 要改什么？** 更换 FP8 dtype/MMA，并重新验证 scale 在 MMA 前还是 epilogue 使用。
10. **为什么端到端收益小？** attention、KV cache、norm、采样和 Python 调度不在 Linear kernel 的加速范围内。
11. **zero point 怎么处理？** 本项目采用对称 INT4，范围 `[-8,7]`，没有 zero point，反量化只有乘 scale。
12. **累加精度？** 解包和 scale 在 fp32 表达，GEMM 累加器 fp32，最后转换 fp16。
13. **M 变大后优势为什么变小？** 权重复用上升，瓶颈从带宽向 Tensor Core 计算移动；因此 prefill 不能只看字节数。
14. **JIT/autotune 是否值得？** 固定 shape 可以一次编译换稳定延迟；E2E 用 M 桶缓存避免每个 token 编译。
15. **INT4 与 FP8 的差别？** INT4 是均匀整数格点，FP8 是浮点格点；scale 粒度和应用位置都要按硬件重新验证。
16. **能和 vLLM 吞吐直接比吗？** 不能。本项目是 HF、batch=1、无 continuous batching；服务吞吐需要相同并发和调度条件。
17. **能用于 MoE 吗？** Day 10 的 ToyMoE 验证了专家矩阵误差和约 3.88× 权重压缩；真实 Qwen MoE 仍是后续工作。
18. **公开仓库风险？** 扫描绝对路径、token、模型权重和大文件；模型不进 Git，路径由 `W4A16_MODEL_DIR` 提供。
19. **尚未完成的优化怎么回答？** 说明已测范围、证据、停止原因和下一步路线，不把计划数字写成实测数字。
20. **如何判断答案可靠？** 数字能回到 JSON/日志，机制解释与代码一致，同时说明硬件、dtype、计时对象和边界。
