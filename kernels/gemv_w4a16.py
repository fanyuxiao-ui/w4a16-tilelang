"""M=1-specialized W4A16 GEMV.

Each CUDA thread owns one output row and streams that row of packed weights
along K.  The reference format is the project's normal ``[N, K/2] uint8``
packed-int4 tensor with fp16 per-group scales.
"""

import tilelang
import tilelang.language as T

from kernels.dequant_int4 import _u8_to_i4_to_i8


@tilelang.jit
def gemv_w4a16(
    N: int,
    K: int,
    group_size: int = 128,
    block_N: int = 256,
    threads: int = 256,
    vec_bytes: int = 16,
):
    assert threads == block_N
    assert K % (vec_bytes * 2) == 0
    assert K % group_size == 0
    elems_per_vec = vec_bytes * 2
    n_vec = K // elems_per_vec

    @T.prim_func
    def main(
        A: T.Tensor((K,), "float16"),
        Q: T.Tensor((N, K // 2), "uint8"),
        S: T.Tensor((N, T.ceildiv(K, group_size)), "float16"),
        C: T.Tensor((N,), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
            tx = T.get_thread_binding()
            a_shared = T.alloc_shared((K,), "float16")
            qbuf = T.alloc_local((vec_bytes,), "uint8")
            acc = T.alloc_local((1,), "float32")

            T.copy(A, a_shared)
            acc[0] = 0.0
            if bx * block_N + tx < N:
                for v in T.serial(n_vec):
                    for b in T.vectorized(vec_bytes):
                        qbuf[b] = Q[bx * block_N + tx, v * vec_bytes + b]
                    for e in T.serial(elems_per_vec):
                        k = v * elems_per_vec + e
                        acc[0] += (
                            T.Cast("float32", _u8_to_i4_to_i8(4, qbuf[e // 2], e % 2, "int8"))
                            * T.Cast("float32", a_shared[k])
                            * T.Cast("float32", S[bx * block_N + tx, k // group_size])
                        )
                C[bx * block_N + tx] = T.Cast("float16", acc[0])

    return main
