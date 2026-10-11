"""Baseline BF16/FP16 GEMM in TileLang，作为后续 W4A16 的对照基线。"""
import tilelang
import tilelang.language as T


@tilelang.jit
def gemm_bf16(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 64,
    num_stages: int = 3,
    threads: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """C[M,N] = A[M,K] @ B[N,K]^T （B 为行主序 [N,K]，转置乘法）"""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.use_swizzle(panel_size=10)  # 提升 L2 命中率（Day 2 会对比开/关）

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main
