// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef PLATFORMS_GPUS_TESTING_NVIDIA_GEMM_TEST_LIB_INTERNAL_H_
#define PLATFORMS_GPUS_TESTING_NVIDIA_GEMM_TEST_LIB_INTERNAL_H_

#include <stddef.h>

#include <memory>
#include <string>

#include "src/gemm_test_lib.h"
#include "src/matrix_lib.h"
#include "src/memory_allocator_interface.h"
#include "src/multi_gemm_lib.h"

namespace platforms_gpus {
namespace gemm_test {
namespace internal {

// This abstract class serves as an interface for cuda cublasGemmEx API that
// utilized by gemm_test. This interface is introduced mainly for
// mocking/testing purpose.
class GpuComputationInterface {
 public:
  virtual ~GpuComputationInterface() {}

  // This function wraps cublasGemmEx()
  virtual cublasStatus_t MatrixMultiComputation(
      size_t dim_size_m, size_t dim_size_n, size_t dim_size_k, bool transa,
      bool transb, const std::string &compute_type,
      const std::string &algorithm, const std::string &algorithm_tc,
      cublasHandle_t handle, const void *alpha, const void *A, const void *B,
      const void *beta, void *C, int compute_capability_major) = 0;
};

template <typename P_in, typename P_out>
class CudaCublasInterface final : public GpuComputationInterface {
 public:
  CudaCublasInterface() {}

  ~CudaCublasInterface() override {}

  cublasStatus_t MatrixMultiComputation(
      size_t dim_size_m, size_t dim_size_n, size_t dim_size_k, bool transa,
      bool transb, const std::string &compute_type,
      const std::string &algorithm, const std::string &algorithm_tc,
      cublasHandle_t handle, const void *alpha, const void *A, const void *B,
      const void *beta, void *C, int compute_capability_major) override;
};

// Derived and templated classes of HostContext class. Used to instantiate
// HostContext with different data input/output combinations.
template <typename P_in, typename P_out>
class MixedPrecisionHostContext : public HostContext {
 public:
  explicit MixedPrecisionHostContext(const ContextOption &options);

  MixedPrecisionHostContext(
      const ContextOption &options,
      std::unique_ptr<MemoryAllocatorInterface> memory_allocator,
      std::unique_ptr<CudaCublasInterface<P_in, P_out>> computation_interface);

  ~MixedPrecisionHostContext() override {}

 protected:
  std::unique_ptr<GpuContext> CreateGpuContext(int gpu_num) override;
  std::unique_ptr<MemoryAllocatorInterface> memory_allocator_;
  std::unique_ptr<CudaCublasInterface<P_in, P_out>> computation_interface_;
  CudaRandomMatrix<P_in> a_;
  CudaRandomMatrix<P_in> b_;
};

// Derived and templated classes of GpuContext class. Used to instantiate
// GpuContext with different data input/output combinations.
template <typename P_in, typename P_out>
class MixedPrecisionGpuContext : public GpuContext {
 public:
  MixedPrecisionGpuContext(HostContext *h,
                           RandomMatrix<P_in> const *const matrix_a_p,
                           RandomMatrix<P_in> const *const matrix_b_p,
                           int gpu_num,
                           GpuComputationInterface *compute_interface);

  ~MixedPrecisionGpuContext() override;

  void StreamSynchronize() override;

  cudaError_t StreamQuery() override;

  void LaunchKernel() override;

 protected:
  // cublasGemmEx() requires that the scaling factors alpha and
  // beta should be the same data type as ComputeType. Also, addresses for alpha
  // and beta, which are saved in dev_alpha_ and dev_beta_, should be aligned
  // with ComputeType. Note that ComputeType could be different than
  // data_type_in or data_type_out. One example is that both data_type_in and
  // data_type_out are half type, while ComputeType could be float type.
  void *dev_alpha_;
  void *dev_beta_;
  P_in *dev_a_;
  P_in *dev_b_;
  P_out *dev_c_;

  // major revision numbers defining the device's compute capability.
  int compute_capability_major_rev_;
  cudaStream_t stream_;
  cublasHandle_t cublas_handle_;
  GpuComputationInterface *compute_interface_;
};

}  // namespace internal
}  // namespace gemm_test
}  // namespace platforms_gpus

#endif  // PLATFORMS_GPUS_TESTING_NVIDIA_GEMM_TEST_LIB_INTERNAL_H_
