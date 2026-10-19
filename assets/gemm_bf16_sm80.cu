#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 49152));
  float C_local[128];
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[(((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[((((((int)blockIdx.y) * 524288) + (i_1 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)B_shared)[(((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((((int)blockIdx.x) * 524288) + (i_2 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[((((((i_3 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(A[(((((((int)blockIdx.y) * 524288) + (i_3 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)threadIdx.x) & 7) * 8)) + 64)])));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 8; ++i_4) {
    tl::cp_async_gs<16>((&(((half_t*)B_shared)[((((((i_4 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B[(((((((int)blockIdx.x) * 524288) + (i_4 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + ((((int)threadIdx.x) & 7) * 8)) + 64)])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 62; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      tl::cp_async_gs<16>((&(((half_t*)A_shared)[((((((((k + 2) % 3) * 8192) + (i_5 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A[((((((((int)blockIdx.y) * 524288) + (i_5 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (k * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 128)])));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      tl::cp_async_gs<16>((&(((half_t*)B_shared)[((((((((k + 2) % 3) * 8192) + (i_6 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B[((((((((int)blockIdx.x) * 524288) + (i_6 * 65536)) + ((((int)threadIdx.x) >> 3) * 4096)) + (k * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 128)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<2>();
    __syncthreads();
    {
      half_t A_local[32];
      half_t B_local[32];
      for (int ki = 0; ki < 4; ++ki) {
        #pragma unroll
        for (int i_7 = 0; i_7 < 4; ++i_7) {
          tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[((((((((k % 3) * 8192) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (i_7 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[(i_7 * 8)])));
        }
        #pragma unroll
        for (int i_8 = 0; i_8 < 4; ++i_8) {
          tl::ptx_ldmatrix_x4((&(((half_t*)B_shared)[(((((((((k % 3) * 8192) + ((((int)threadIdx.x) >> 6) * 4096)) + (i_8 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_8 * 8)])));
        }
        for (int i_9 = 0; i_9 < 4; ++i_9) {
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_9 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_9 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_9 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_9 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    half_t A_local_1[32];
    half_t B_local_1[32];
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      #pragma unroll
      for (int i_10 = 0; i_10 < 4; ++i_10) {
        tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[(((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + (i_10 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(A_local_1[(i_10 * 8)])));
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 4; ++i_11) {
        tl::ptx_ldmatrix_x4((&(((half_t*)B_shared)[(((((((((((int)threadIdx.x) >> 6) * 4096) + (i_11 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(B_local_1[(i_11 * 8)])));
      }
      for (int i_12 = 0; i_12 < 4; ++i_12) {
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_12 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_12 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_12 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_12 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t A_local_2[32];
    half_t B_local_2[32];
    for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
      #pragma unroll
      for (int i_13 = 0; i_13 < 4; ++i_13) {
        tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + (i_13 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_2[(i_13 * 8)])));
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 4; ++i_14) {
        tl::ptx_ldmatrix_x4((&(((half_t*)B_shared)[((((((((((int)threadIdx.x) >> 6) * 4096) + (i_14 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_2[(i_14 * 8)])));
      }
      for (int i_15 = 0; i_15 < 4; ++i_15) {
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_15 * 32) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_15 * 32) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 64; ++i_16) {
    half_t C_local_cast[2];
    uint1 __1;
    float2 v_ = *(float2*)(C_local + (i_16 * 2));
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    *(uint1*)(C_local_cast + 0) = __1;
    *(uint1*)(C + (((((((((((int)blockIdx.y) * 524288) + (((((int)threadIdx.x) & 63) >> 5) * 262144)) + ((i_16 >> 4) * 65536)) + ((i_16 & 1) * 32768)) + (((((int)threadIdx.x) & 31) >> 2) * 4096)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_16 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_local_cast + 0);
  }
}

