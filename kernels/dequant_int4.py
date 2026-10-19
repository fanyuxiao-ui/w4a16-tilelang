"""INT4 打包权重的反量化 helper 与 TileLang kernel。"""

import tilelang
import tilelang.language as T

try:
    from tvm import tirx
except ImportError:  # pragma: no cover - compatibility with TileLang's bundled TVM
    from tilelang import tvm as _tvm

    tirx = _tvm.tirx


def _u8_to_i4_to_i8(nbit: int, val, pos, dtype: str = "int8"):
    """Extract one signed ``nbit`` value from a packed uint8 expression."""

    assert nbit == 4
    assert dtype == T.int8
    assert val.dtype == T.uint8

    mask = tirx.const((1 << nbit) - 1, T.uint8)
    i4 = (val >> (pos.astype(T.uint8) * tirx.const(nbit, T.uint8))) & mask
    i8_shifted = tirx.reinterpret(T.int8, i4 << tirx.const(4, T.uint8))
    return i8_shifted >> tirx.const(4, T.int8)


@tilelang.jit
def dequant_int4_kernel(
    N: int,
    K: int,
    group_size: int = 128,
    blk_n: int = 16,
    threads: int = 256,
):
    """Dequantize packed signed INT4 weights into an fp16 matrix."""

    assert K % 2 == 0 and K % group_size == 0

    @T.prim_func
    def main(
        Q: T.Tensor((N, K // 2), "uint8"),
        S: T.Tensor((N, K // group_size), "float16"),
        W: T.Tensor((N, K), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, blk_n), threads=threads) as (bx):
            for i, k in T.Parallel(blk_n, K):
                n = bx * blk_n + i
                v = _u8_to_i4_to_i8(4, Q[n, k // 2], k % 2, "int8")
                W[n, k] = T.Cast(
                    "float16",
                    T.Cast("float32", v)
                    * T.Cast("float32", S[n, k // group_size]),
                )

    return main
