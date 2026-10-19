# W4A16 规范化打包格式（spec）

> 建立于 Day 3。**全项目统一，之后不许改。** 所有 kernel 的正确性以本文件 + `baseline/torch_ref.py` 为准。
> 来源：手册 Day 3 第 1.3 节。

## 1. 量化方案

| 项 | 取值 | 理由 |
|---|---|---|
| 对称性 | **对称**（`z = 0`） | 权重近似零均值；反量化只剩一次乘法 `w = q * s`，没有加法 |
| 值域 | **`[-8, 7]`**（INT4 补码） | 补码，负数直接喂 dp4a/mma |
| 粒度 | **per-group，沿 K 维切，`group_size = 128`** | 离群值只污染它所在的 128 个元素 |
| scale dtype | `fp16`，形状 `[N, K/group_size]` | 存储开销 = 权重的 1/64 |
| 中间计算 | 除法/round 在 **fp32** 下做 | fp16 除法会引入额外舍入 |

```
q = clamp(round(w / s), -8, 7)          # w, s 都先转 fp32
s = amax / 7                            # amax = max(|w_group|)
```

## 2. 打包布局（`pack_nibbles` / `unpack_nibbles` 的定义）

**沿 K 维**打包，byte `j` 装元素 `2j`（低 4 位）和元素 `2j+1`（高 4 位）：

```
q_packed[n, j] = (q[n, 2j] & 0xF) | ((q[n, 2j+1] & 0xF) << 4)
q_packed 形状 [N, K/2]，dtype uint8
```

负数以 4-bit 补码存入 nibble（如 −3 → `0b1101`）。解包时先取 nibble 再符号扩展：

```
lo = byte & 0x0F
hi = (byte >> 4) & 0x0F
v  = (nibble ^ 8) - 8        # 4-bit 补码 -> int32 有符号
```

即：**偶 k 在低半字节，奇 k 在高半字节，K 维连续**。这一条是整个格式里最容易被写反、
也最难从误差数字上发现的地方（写反后 K 相邻元素互换，误差统计上完全看不出），
所以 `test_pack_unpack_roundtrip` 用全部 16 个值做严格 `torch.equal` 比对。

## 3. 与官方 tilelang 示例格式的关系

官方 `fp16xint4` GEMV 示例用的是**另一套** interleave 布局（为配合 LOP3 快速解码指令），
**不要混用**：

- 用官方示例时，按其要求调用 `interleave_weight`；
- 自己写的 kernel 流水线，全部使用上面第 2 节的格式。

两套格式的边界就是 `interleave_weight` 这个函数，跨边界时必须显式调用。

## 4. 接口（`baseline/torch_ref.py`，黄金参考）

```
compute_scales(w[N,K] fp16, group_size=128)      -> scales[N, K//g] fp16
quantize_q(w, scales, group_size)                -> q[N, K] int32, 值域 [-8,7]
pack_nibbles(q[N,K] int32)                       -> q_packed[N, K/2] uint8
unpack_nibbles(q_packed[N, K/2] uint8)           -> q[N, K] int32（符号已恢复）
quantize_int4_symmetric(w, group_size)           -> (q_packed, scales)
dequantize_int4_symmetric(q_packed, scales, g)   -> w[N, K] float32（只供对照）
```

## 5. 不变量（kernel 侧必须逐条成立，可用作 kernel 测试的判据）

1. `unpack_nibbles(pack_nibbles(q))` 与 `q` **严格相等**（非近似）。
2. 反量化误差的**分布无关硬界**：每个元素 `|w_deq − w| ≤ s_group / 2`。
   任何 scale 算错、分组切错、符号扩展错、位序错都会立刻破坏它。
3. 不同 group 的 scale 互不影响（第 0 组埋 0.1、第 1 组埋 100，scale 比值应 ≈ 1000）。
4. `K % group_size == 0`，否则直接 assert 失败（不做尾块处理，Day 3 明确不引入这个复杂度）。
5. **`s > 0` 恒成立（在 fp16 里也严格大于 0）**，否则 `quantize_q` 走 0/0 → nan 路径。
   因此 scale 的 floor 必须按**存储 dtype** 选，而不是按中间计算 dtype：`1e-10` 在 fp32 里是正数，
   cast 成 fp16 会下溢为精确的 0。本项目取 `floor = 1e-7`（fp16 最小 subnormal ≈ 6.1e-8）。
   实测真实权重 40 万个组的最小 amax = 3.6e-3，此 floor 永不触发，纯防御。
