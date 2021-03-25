#ifndef THIRD_PARTY_GPU_TEST_UTILS_CUDA_COMPUTE_COPY_CU_H_
#define THIRD_PARTY_GPU_TEST_UTILS_CUDA_COMPUTE_COPY_CU_H_

#include <driver_types.h>

namespace platforms_gpus {
namespace memcpy_gemm {

// Explicit use cuda kernel for copying data.
// For efficiency, currently the size should be aligned to 32 bytes.
// The code inside don't do out of bound check.
cudaError_t cudaComputeCopy(cudaStream_t stream, void *dst_v, const void *src_v,
                            size_t size);

// Adaptive adjust grid number for each copy flow
cudaError_t cudaComputeCopyAdaptive(cudaStream_t stream, void *dst_v,
                                    const void *src_v, size_t size,
                                    int flow_cnt);
}  // namespace memcpy_gemm
}  // namespace platforms_gpus

#endif  // THIRD_PARTY_GPU_TEST_UTILS_CUDA_COMPUTE_COPY_CU_H_
