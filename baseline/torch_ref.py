# baseline/torch_ref.py
"""W4A16 量化的 torch 黄金参考实现。全项目的正确性基准。

规范打包格式（全项目统一，之后不许改，见 docs/int4_pack_spec.md）：
    沿 K 维：byte j 包含元素 2j（低 4 位）和元素 2j+1（高 4 位）
    q_packed[n, j] = (q[n, 2j] & 0xF) | ((q[n, 2j+1] & 0xF) << 4)
    q_packed 形状 [N, K/2]，dtype uint8

量化方案：对称（z = 0）、per-group（group_size=128）、INT4 补码值域 [-8, 7]。
反量化只剩一次乘法 w = q * s，没有加法——这是零点的收益，kernel 侧直接受益。

约定：所有量化相关的除法/round 都在 fp32 下做，scale 存 fp16（与 kernel 对齐）。
"""
import torch


def compute_scales(w: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """[N, K] fp16 -> [N, K//group_size] fp16。对称量化：s = amax / 7。"""
    N, K = w.shape
    assert K % group_size == 0
    amax = w.float().view(N, K // group_size, group_size).abs().amax(dim=-1)  # [N, K//g]
    # floor 必须能在**存储 dtype（fp16）**里存活，不是 fp32 里够小就行：
    # 1e-10 在 fp32 里是正数，cast 成 fp16 却恰好下溢为 0 → 全零组 s=0 →
    # quantize_q 里 0/0=nan → clamp(nan)=nan → .to(int32) 得到 INT32_MIN 垃圾值
    # （打包恰好因低位为 0 而侥幸没暴露，kernel 侧会直接炸）。
    # fp16 最小 subnormal ≈ 6.1e-8，故取 1e-7。
    # 实测真实权重的 40 万个组里最小 amax = 3.6e-3，这个 floor 永远不触发，纯属防御。
    scales = (amax / 7.0).clamp(min=1e-7)
    return scales.to(torch.float16).contiguous()


def quantize_q(w: torch.Tensor, scales: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """[N, K] fp16 + scales -> [N, K] int32，值域 [-8, 7]。"""
    N, K = w.shape
    wg = w.float().view(N, K // group_size, group_size)
    q = torch.round(wg / scales.float().unsqueeze(-1)).clamp(-8, 7).to(torch.int32)
    return q.view(N, K).contiguous()


def pack_nibbles(q: torch.Tensor) -> torch.Tensor:
    """[N, K] int32（值域 [-8,7]）-> [N, K/2] uint8。规范格式：偶 k 在低半字节。"""
    nib = (q & 0xF).to(torch.uint8)
    packed = nib[:, 0::2] | (nib[:, 1::2] << 4)
    return packed.contiguous()


def unpack_nibbles(q_packed: torch.Tensor) -> torch.Tensor:
    """[N, K/2] uint8 -> [N, K] int32（符号已恢复，值域 [-8,7]）。"""
    N, K2 = q_packed.shape
    qi = q_packed.to(torch.int32)
    lo = qi & 0x0F
    hi = (qi >> 4) & 0x0F
    q = torch.empty(N, K2 * 2, dtype=torch.int32, device=q_packed.device)
    q[:, 0::2] = lo
    q[:, 1::2] = hi
    return ((q ^ 8) - 8).contiguous()          # 4-bit 补码 -> 有符号


def quantize_int4_symmetric(w: torch.Tensor, group_size: int = 128):
    """一步到底：w -> (q_packed [N,K/2] uint8, scales [N,K//g] fp16)。"""
    scales = compute_scales(w, group_size)
    q = quantize_q(w, scales, group_size)
    return pack_nibbles(q), scales


def dequantize_int4_symmetric(q_packed: torch.Tensor, scales: torch.Tensor,
                              group_size: int = 128) -> torch.Tensor:
    """黄金反量化：返回 [N, K] float32（供对照，别直接用于推理）。"""
    q = unpack_nibbles(q_packed)
    N, K = q.shape
    w = q.float().view(N, K // group_size, group_size) * scales.float().unsqueeze(-1)
    return w.view(N, K)
