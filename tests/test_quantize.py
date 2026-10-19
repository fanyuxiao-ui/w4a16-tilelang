"""Day 3 验收：量化/打包/解包/反量化的黄金参考测试。

量化数学与设备无关，DEVICE 在无卡模式下自动回落 CPU（kernel 测试则必须等卡）。
"""
import torch
import pytest

from baseline.torch_ref import (
    compute_scales, quantize_q, pack_nibbles, unpack_nibbles,
    quantize_int4_symmetric, dequantize_int4_symmetric,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_pack_unpack_roundtrip():
    """打包->解包必须严格还原（覆盖全部 16 个值）。"""
    q = torch.arange(-8, 8, device=DEVICE, dtype=torch.int32).repeat(64)  # [1024]
    q = q.unsqueeze(0)  # [1, 1024]
    packed = pack_nibbles(q)
    q2 = unpack_nibbles(packed)
    assert torch.equal(q, q2)
    print(f"pack roundtrip: 值域 [-8,7] 全 16 值严格还原 ({DEVICE})")


def test_quantize_roundtrip_error():
    """往返误差：判据是「分布无关的硬界 + 理论锚定」，不是手册里的固定 5% 门限。

    手册原文写 `rel < 0.05`（"INT4 典型往返误差 1%~5%"），实测 0.126 被判失败。
    经诊断（2026-10-15，含真实权重）确认是**门限标错，不是实现有 bug**：
      - round-to-nearest 的均匀格点误差 mean|err| = s/4，于是对一个组内平滑的分布
        rel ≈ (mean(s)/4) / mean|w|。高斯数据实测 0.1260 / 理论 0.1272 = 0.991（吻合）；
        真实权重（DeepSeek-Coder-6.7B gate_proj）0.1390，比高斯更差而非更好——
        因为真实权重重尾（组内 amax/mean|w| = 153），INT4 的步长 s = amax/7
        本身就与 mean|w| 同量级，13% 是这个粒度下的数学必然。
      - 结论：`1%~5%` 那个数字对应的是别的量（极可能是端到端 PPL 退化或 INT8），
        不是「mean|err|/mean|w|」这个统计量。
    因此这里改用两层判据，任何真实的实现错误都会破坏第一层：
      1. 硬界（分布无关）：每个元素 |err| ≤ s_group/2 —— 取整永远不会越过半个步长。
         搞错 scale、分组切错、符号扩展错、打包位序错，都会立刻越界。
      2. 相对误差与理论值 mean(s)/4 / mean|w| 同量级（±25%）——防止"误差大得离谱
         却恰好没越界"的情形（例如 scale 整体系统性偏小）。
    """
    torch.manual_seed(0)
    w = (torch.randn(256, 1024, device=DEVICE) * 0.02).half()
    # 埋一个离群值，验证 group 化的保护作用
    w[0, :128] *= 20
    packed, scales = quantize_int4_symmetric(w, group_size=128)
    w_deq = dequantize_int4_symmetric(packed, scales, group_size=128)
    N, K = w.shape
    s_exp = scales.float().unsqueeze(-1).expand(N, K // 128, 128).reshape(N, K)

    # 判据 1：硬界
    ratio = ((w_deq - w.float()).abs() / (s_exp / 2)).max().item()
    assert ratio <= 1.0 + 1e-3, f"反量化误差越过半个步长: {ratio}"

    # 判据 2：与理论值同量级
    rel = ((w_deq - w.float()).abs().mean() / w.float().abs().mean()).item()
    pred = (scales.float().mean() / 4 / w.float().abs().mean()).item()
    print(f"平均相对误差: {rel:.4f}（含 20x 离群值, {DEVICE}）| 理论 {pred:.4f} | "
          f"max(|err|/(s/2)) = {ratio:.6f}")
    assert 0.75 < rel / pred < 1.25, f"相对误差 {rel:.4f} 与理论 {pred:.4f} 不符，查实现"


def test_scale_math():
    """手工验证一组：amax=7 时 scale=1，值应严格还原。"""
    w = torch.zeros(1, 128, device=DEVICE)
    w[0, 0] = 7.0
    w[0, 1] = -7.0
    w[0, 2] = 3.5
    scales = compute_scales(w.half(), 128)
    q = quantize_q(w.half(), scales, 128)
    assert q[0, 0].item() == 7 and q[0, 1].item() == -7
    assert abs(q[0, 2].item()) in (3, 4)  # round(3.5)：允许银行家舍入
    assert scales[0, 0].item() == pytest.approx(1.0)
    print(f"scale math: amax=7 -> s=1.0, 7/-7 严格还原, 3.5 -> {q[0, 2].item()}")


def test_group_boundary():
    """不同 group 的 scale 互不影响。"""
    w = torch.zeros(1, 256, device=DEVICE)
    w[0, 0] = 0.1       # 第 0 组 amax=0.1
    w[0, 200] = 100.0   # 第 1 组 amax=100
    packed, scales = quantize_int4_symmetric(w.half(), 128)
    # 第 0 组的 scale 应远小于第 1 组
    assert scales[0, 1].item() > scales[0, 0].item() * 100
    print(f"group boundary: s0={scales[0, 0].item():.6f}, s1={scales[0, 1].item():.4f}, "
          f"比值={scales[0, 1].item() / scales[0, 0].item():.1f}x")


def test_zero_group():
    """全零组（真实权重里存在，如 padding/pruned 行）不得产生 nan 或垃圾整数。

    这个测试是手册四个测试之外新增的（Day 3 目标 4 明确要求"边界测试"）。
    它抓到的真实缺陷：floor=1e-10 在 fp32 里有效，cast 成 fp16 后下溢为 0，
    于是 0/0=nan → .to(int32) 得到 INT32_MIN。打包那一步恰好因低位为 0 而
    "看起来没错"，但 q 本身已经是垃圾——kernel 侧会直接算出 nan。
    """
    w = torch.zeros(1, 256, dtype=torch.float16, device=DEVICE)
    w[0, 5] = 0.3            # 第 0 组非零，第 1 组全零
    packed, scales = quantize_int4_symmetric(w, group_size=128)
    assert (scales > 0).all(), f"scale 必须严格大于 0，实际 {scales}"
    q = quantize_q(w, scales, 128)
    # 用显式上下界而非 q.abs().max()：INT32_MIN 取绝对值会溢出成负数，恰好通过 max 检查
    assert q.min().item() >= -8 and q.max().item() <= 7, \
        f"q 出现越界垃圾值: {q.min().item()}..{q.max().item()}"
    assert (q[:, 128:] == 0).all()          # 全零组量化后必须是精确的 0
    d = dequantize_int4_symmetric(packed, scales, 128)
    assert (d[:, 128:] == 0).all() and not torch.isnan(d).any()
    print(f"zero group: s_min={scales.min().item():.3e}, 全零组 q 严格为 0，无 NaN")


if __name__ == "__main__":
    test_pack_unpack_roundtrip()
    test_quantize_roundtrip_error()
    test_scale_math()
    test_group_boundary()
    test_zero_group()
    print("all quantize tests passed")
