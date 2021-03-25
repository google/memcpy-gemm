#include "src/cuda_compute_copy.cu.h"

#include <cuda_runtime.h>
#include <stdint.h>

#define UNROLL 8
#define WARP_SIZE 32
namespace platforms_gpus {
namespace memcpy_gemm {

__global__ static void simple_copy(void *dst_v, const void *src_v,
                                   size_t size) {
  ulong2 *dst = (ulong2 *)dst_v;
  const ulong2 *src = (ulong2 *)src_v;

  const int nwarps = gridDim.x * blockDim.x / WARP_SIZE;
  const int warp = (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;
  const int wid = threadIdx.x % WARP_SIZE;

  size /= sizeof(*src);

  const ulong2 *s = src + warp * WARP_SIZE * UNROLL;
  ulong2 *d = dst + warp * WARP_SIZE * UNROLL;

  while (s < src + size) {
    // It's faster to do a bunch of reads, followed by a bunch of writes,
    // instead of going one by one.
    ulong2 data[UNROLL];

#pragma unroll
    for (int u = 0; u < UNROLL; u++) {
      data[u] = s[u * WARP_SIZE + wid];
    }

#pragma unroll
    for (int u = 0; u < UNROLL; u++) {
      d[u * WARP_SIZE + wid] = data[u];
    }

    s += nwarps * WARP_SIZE * UNROLL;
    d += nwarps * WARP_SIZE * UNROLL;
  }
}

cudaError_t cudaComputeCopy(cudaStream_t stream, void *dst_v, const void *src_v,
                            size_t size) {
  int minGridSize = 0, blockSize = 1024;
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, simple_copy);
  cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) return error;
  simple_copy<<<14, blockSize, 0, stream>>>(dst_v, src_v, size);
  return cudaPeekAtLastError();
}

cudaError_t cudaComputeCopyAdaptive(cudaStream_t stream, void *dst_v,
                                    const void *src_v, size_t size,
                                    int flow_cnt) {
  int min_grid_size = 0, block_size = 1024;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, simple_copy);
  cudaError_t error = cudaPeekAtLastError();
  if (error != cudaSuccess) {
    return error;
  }
  int sm_count = 0;
  error = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
  if (error != cudaSuccess) {
    return error;
  }
  int grid_size = sm_count / flow_cnt;
  if (grid_size < 1) {
    grid_size = 1;
  }
  // If we use maxinum block size,
  // we do see some of the performance not good compare to half.
  block_size /= 2;
  if (block_size < 1) {
    block_size = 1;
  }
  simple_copy<<<grid_size, block_size, 0, stream>>>(dst_v, src_v, size);
  return cudaPeekAtLastError();
}

}  // namespace memcpy_gemm

}  // namespace platforms_gpus
