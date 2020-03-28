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

#include "absl/strings/string_view.h"
#include "src/gemm_test_lib.h"
#include "src/matrix_lib.h"
#include "src/memory_allocator_interface.h"
#include "src/multi_gemm_lib.h"
#include "cuda/include/cublasLt.h"
#include "cuda/include/driver_types.h"

namespace platforms_gpus {
namespace gemm_test {
namespace internal {

// This abstract class serves as an interface for cuda cublasGemmEx API that
// utilized by gemm_test. The actual backend cuBLAS call depends on the data
// types and GPU compute capability.
class GpuComputationInterface {
 public:
  virtual ~GpuComputationInterface() {}
  // Executes a computation cycle. The backend cuBLAS function call depends on
  // the GPU architecture and potentially the CUDA version.
  virtual cublasStatus_t MatrixMultiComputation(
      const ContextOption &context_options,
      const void *alpha, const void *A, const void *B, const void *beta,
      void *C) = 0;

  // Sets the stream to use for any cuBLAS library calls.
  virtual cublasStatus_t SetStream(cudaStream_t stream_id) = 0;

  // Account for any context-specific initialization. Most backends do not need any setup, but
  // the cublasLT API requires some parameter settings that are dependent on matrix options, which
  // we don't want to put in the critical path.
  virtual cublasStatus_t InitializeCublasSettings(const ContextOption &context_options) = 0;
};

// Modern interface for compute capability >= 5.0. Allows half and mixed
// precision computation, and limited integer based math.
class CudaCublasInterface final : public GpuComputationInterface {
 public:
  CudaCublasInterface();

  ~CudaCublasInterface() override;

  cublasStatus_t MatrixMultiComputation(const ContextOption &context_options,
                                        const void *alpha, const void *A,
                                        const void *B, const void *beta,
                                        void *C) override;

  cublasStatus_t SetStream(const cudaStream_t stream_id) override;

   cublasStatus_t InitializeCublasSettings(const ContextOption &context_options) override {

       LOG(INFO) <<"Called";
       return CUBLAS_STATUS_SUCCESS;
   }

private:
  cublasHandle_t cublas_handle_;

};

// Legacy interface for compute capability <= 5 (k80 and earlier). Supports
// only single and double precision.
template <typename T>
class LegacyCudaCublasInterface final : public GpuComputationInterface {
 public:
    LegacyCudaCublasInterface();

    ~LegacyCudaCublasInterface() override;

      cublasStatus_t MatrixMultiComputation(const ContextOption &context_options,
                                            const void *alpha, const void *A,
                                            const void *B, const void *beta,
                                            void *C) override;

   cublasStatus_t SetStream(const cudaStream_t stream_id) override;
    cublasStatus_t InitializeCublasSettings(const ContextOption &context_options) override {
        LOG(INFO) <<"Called";
        return CUBLAS_STATUS_SUCCESS;
    }
        private:
        cublasHandle_t cublas_handle_;
};

class CudaInt8TensorInterface final: public GpuComputationInterface {
  public:

    CudaInt8TensorInterface();

    ~CudaInt8TensorInterface() override;

    cublasStatus_t MatrixMultiComputation(const ContextOption &context_options,
                                          const void *alpha, const void *A,
                                          const void *B, const void *beta,
                                          void *C) override;

    cublasStatus_t SetStream(const cudaStream_t stream_id) override;
    cublasStatus_t InitializeCublasSettings(const ContextOption &context_options) override;

  private:
    // Layout description constants.
    static constexpr cublasLtOrder_t kMatrixACLayout = CUBLASLT_ORDER_COL32;
    static constexpr cublasLtOrder_t kMatrixBLayout = CUBLASLT_ORDER_COL4_4R2_8C;
    // Transpose operation constants.
    static constexpr cublasOperation_t kTransOpA = CUBLAS_OP_N;
    static constexpr cublasOperation_t kTransOpB = CUBLAS_OP_T;
    // Pointer mode constant.
    static constexpr  cublasPointerMode_t kPointerMode = CUBLAS_POINTER_MODE_DEVICE;

    cublasLtHandle_t cublas_handle_;
    cudaStream_t stream_;
    // CUDA math operation descriptor.
    cublasLtMatmulDesc_t matmul_desc_;
    // CUDA matrix descriptors.
    cublasLtMatrixLayout_t layout_a_;
    cublasLtMatrixLayout_t layout_b_;
    cublasLtMatrixLayout_t layout_c_;

};

// Selects and creates a GEMM interface based on the precision of computation to
// be done and the capabilities of the GPU.
std::unique_ptr<GpuComputationInterface> SelectGemmInterface(
    absl::string_view compute_type, float compute_capability);

// GpuDataHandler stores pointers to GPU data, and handles allocation and
// freeing of those pointers. A GpuDataHandler is assigned to a single GPU.
// Allocation failures lead to immediate program exit.
// TODO: Return error status rather than crashing program on
// CUDA allocation failures.
// TODO: Template on compute type.
template <typename InputPrecision, typename OutputPrecision>
class GpuDataHandler {
 public:
  GpuDataHandler() = default;
  ~GpuDataHandler();

  // Allocates input and output arrays. Allocates and sets scaling factors.
  // Copies input array data from host to device.
  void Initialize(const RandomMatrix<InputPrecision> *data_in_a,
                  const RandomMatrix<InputPrecision> *data_in_b,
                  const cudaStream_t stream);

  void SetComputeType(absl::string_view compute_type) {
    compute_type_ = compute_type;
  }
  void SetGpuId(const int id) { gpu_id_ = id; }

  // Accessors.
  InputPrecision *InputA() const { return input_a_; }
  InputPrecision *InputB() const { return input_b_; }
  OutputPrecision *Output() const { return output_; }
  void *Alpha() const { return alpha_; }
  void *Beta() const { return beta_; }

 private:
  // Data will be allocated on this GPU
  int gpu_id_ = 0;

  // Precision in which GEMM operations on this data will be performed. This
  // determines the data type of the scaling factors. Must be one of "single"
  // "double", "half", or "int32".
  std::string compute_type_;

  // Input (alpha_) and output (beta_) scaling factors. On the device these
  // factors should be the same type as ComputeType The addresses for
  // alpha_ and beta_should be aligned with ComputeType.
  // Note that ComputeType could be different than the input or output types.
  // One example is that both input and output matrices are half type, while
  // ComputeType could be float type.
  void *alpha_ = nullptr;
  void *beta_ = nullptr;

  // Input and output matrices.
  InputPrecision *input_a_ = nullptr;
  InputPrecision *input_b_ = nullptr;
  OutputPrecision *output_ = nullptr;
};

// Derived and templated classes of HostContext class. Used to instantiate
// HostContext with different data input/output combinations.
// TODO: Template on compute type.
template <typename P_in, typename P_out>
class MixedPrecisionHostContext : public HostContext {
 public:
  explicit MixedPrecisionHostContext(const ContextOption &options);

  MixedPrecisionHostContext(
      const ContextOption &options,
      std::unique_ptr<MemoryAllocatorInterface> memory_allocator);

  ~MixedPrecisionHostContext() override {}

 protected:
  std::unique_ptr<GpuContext> CreateGpuContext(int gpu_num) override;
  std::unique_ptr<MemoryAllocatorInterface> memory_allocator_;
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
                           std::unique_ptr<GpuComputationInterface> compute_interface);

  ~MixedPrecisionGpuContext() override;

  void StreamSynchronize() override;

  cudaError_t StreamQuery() override;

  void LaunchKernel() override;

 protected:
  GpuDataHandler<P_in, P_out> data_handler_;
  cudaStream_t stream_;
  std::unique_ptr<GpuComputationInterface> compute_interface_;
};

extern template class GpuDataHandler<half_float::half, half_float::half>;
extern template class GpuDataHandler<half_float::half, float>;
extern template class GpuDataHandler<float, float>;
extern template class GpuDataHandler<double, double>;
extern template class GpuDataHandler<int8_t, int8_t>;
extern template class GpuDataHandler<int8_t, int32_t>;
extern template class GpuDataHandler<int8_t, float>;

extern template class MixedPrecisionHostContext<half_float::half,
                                                half_float::half>;
extern template class MixedPrecisionHostContext<half_float::half, float>;
extern template class MixedPrecisionHostContext<float, float>;
extern template class MixedPrecisionHostContext<double, double>;
extern template class MixedPrecisionHostContext<int8_t, int8_t>;
extern template class MixedPrecisionHostContext<int8_t, int32_t>;
extern template class MixedPrecisionHostContext<int8_t, float>;

}  // namespace internal
}  // namespace gemm_test
}  // namespace platforms_gpus

#endif  // PLATFORMS_GPUS_TESTING_NVIDIA_GEMM_TEST_LIB_INTERNAL_H_
