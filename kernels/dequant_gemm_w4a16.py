"""融合的 W4A16 dequant GEMM：C[M,N] = A[M,K] @ dequant(Q, S)[N,K]^T。

v1 优先保证正确性：打包权重先搬入 shared memory，解包并乘 scale
后再交给标准的 TileLang ``T.gemm``。
"""

import tilelang
import tilelang.language as T

from kernels.dequant_int4 import _u8_to_i4_to_i8


@tilelang.jit
def dequant_gemm_v1(
    M: int,
    N: int,
    K: int,
    group_size: int = 128,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
):
    """Compute ``C[M,N] = A[M,K] @ dequant(Q, S)[N,K].T``."""

    assert K % block_K == 0 and K % group_size == 0
    assert block_K % group_size == 0 or group_size % block_K == 0

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        Q: T.Tensor((N, K // 2), "uint8"),
        S: T.Tensor((N, T.ceildiv(K, group_size)), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            Q_shared = T.alloc_shared((block_N, block_K // 2), "uint8")
            W_shared = T.alloc_shared((block_N, block_K), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(Q[bx * block_N, k * block_K // 2], Q_shared)

                for i, j in T.Parallel(block_N, block_K):
                    W_shared[i, j] = T.Cast(
                        "float16",
                        T.Cast(
                            "float32",
                            _u8_to_i4_to_i8(4, Q_shared[i, j // 2], j % 2, "int8"),
                        )
                        * T.Cast(
                            "float32",
                            S[bx * block_N + i, (k * block_K + j) // group_size],
                        ),
                    )

                T.gemm(A_shared, W_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


@tilelang.jit
def dequant_gemm_v2(
    M: int,
    N: int,
    K: int,
    group_size: int = 128,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
    use_swizzle: bool = False,
    swizzle_q_layout: bool = False,
):
    """W4A16 GEMM v2.

    The v1 kernel reads the scale tensor once for every dequantized element.
    v2 stages the (much smaller) scale tile in shared memory, so each scale is
    loaded once per output row and K group.  ``use_swizzle`` controls the
    block-rasterisation swizzle; ``swizzle_q_layout`` is exposed separately
    because uint8 shared layouts are not supported by every TileLang/CUDA
    combination.
    """
    assert K % block_K == 0 and K % group_size == 0
    assert block_K % group_size == 0 or group_size % block_K == 0
    groups_per_k = (block_K + group_size - 1) // group_size

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        Q: T.Tensor((N, K // 2), "uint8"),
        S: T.Tensor((N, T.ceildiv(K, group_size)), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            Q_shared = T.alloc_shared((block_N, block_K // 2), "uint8")
            S_shared = T.alloc_shared((block_N, groups_per_k), "float16")
            W_shared = T.alloc_shared((block_N, block_K), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            if swizzle_q_layout:
                # Kept opt-in: some TileLang releases cannot infer a bank
                # swizzle for byte buffers.  The variant is still useful for
                # an explicit compile/performance experiment.
                T.annotate_layout({Q_shared: tilelang.layout.make_swizzled_layout(Q_shared)})
            if use_swizzle:
                T.use_swizzle(panel_size=10)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(Q[bx * block_N, k * block_K // 2], Q_shared)
                # Use the absolute K-group index: when block_K is smaller
                # than group_size, several K tiles share the same scale row.
                T.copy(S[bx * block_N, (k * block_K) // group_size], S_shared)

                for i, j in T.Parallel(block_N, block_K):
                    W_shared[i, j] = T.Cast(
                        "float16",
                        T.Cast(
                            "float32",
                            _u8_to_i4_to_i8(4, Q_shared[i, j // 2], j % 2, "int8"),
                        )
                        * T.Cast("float32", S_shared[i, j // group_size]),
                    )

                T.gemm(A_shared, W_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main
