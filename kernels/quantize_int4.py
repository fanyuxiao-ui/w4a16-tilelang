"""Optional Day 3 TileLang INT4 quantizer.

The torch implementation remains the golden reference.  This kernel exists to
exercise the same group/packing convention on device before Day 5 fusion.
"""
import tilelang
import tilelang.language as T


@tilelang.jit
def quantize_int4_kernel(
    N: int,
    K: int,
    group_size: int = 128,
    blk_n: int = 64,
    threads: int = 128,
):
    assert K % group_size == 0 and group_size % 2 == 0
    groups = K // group_size
    packed_group = group_size // 2

    @T.prim_func
    def main(
        W: T.Tensor((N, K), "float16"),
        Q: T.Tensor((N, K // 2), "uint8"),
        S: T.Tensor((N, groups), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, blk_n), groups, threads=threads) as (bx, by):
            w_shared = T.alloc_shared((blk_n, group_size), "float16")
            w_local = T.alloc_fragment((blk_n, group_size), "float16")
            amax = T.alloc_fragment((blk_n,), "float32")
            scales = T.alloc_fragment((blk_n,), "float32")
            # Keep the integer tile in shared memory: LayoutInference cannot
            # infer a fragment layout when the packing loop indexes 2*j/2*j+1.
            q_local = T.alloc_shared((blk_n, group_size), "int32")

            T.copy(W[bx * blk_n, by * group_size], w_shared)
            T.copy(w_shared, w_local)
            T.reduce_absmax(w_local, amax, dim=1)
            for i in T.Parallel(blk_n):
                amax[i] = T.max(amax[i], 1e-7)
                scales[i] = amax[i] / 7.0
                S[bx * blk_n + i, by] = T.Cast("float16", scales[i])
            for i, j in T.Parallel(blk_n, group_size):
                q_local[i, j] = T.Cast(
                    "int32",
                    T.round(T.Cast("float32", w_local[i, j]) / scales[i]),
                )
                q_local[i, j] = T.max(T.min(q_local[i, j], 7), -8)
            for i, j in T.Parallel(blk_n, packed_group):
                lo = q_local[i, 2 * j] & 0xF
                hi = q_local[i, 2 * j + 1] & 0xF
                Q[bx * blk_n + i, by * packed_group + j] = T.Cast(
                    "uint8", lo | (hi << 4)
                )

    return main
