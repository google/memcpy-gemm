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
#include "src/memory_allocator_interface.h"
#include "src/multi_gemm_lib.h"
#include "cuda/include/cublas_api.h"
#include "cuda/include/driver_types.h"

#if CUDA_VERSION >= 10010
#include "cuda/include/cublasLt.h"
#endif
#if CUDA_VERSION >= BF16_CUDA_VERSION
#include "cuda/include/cuda_bf16.h"
#endif

namespace platforms_gpus {
namespace gemm_test {
namespace internal {

enum class kPtrType { kDevicePtr = 0, kHostPtr = 1 };
struct GEMMData {
  const void *alpha = nullptr;
  const void *beta = nullptr;
  const void *matA = nullptr;
  const void *matB = nullptr;
  void *matC = nullptr;
  kPtrType scalePtrType = kPtrType::kDevicePtr;
};

struct Algo {
  cublasGemmAlgo_t cublas_algo_ = CUBLAS_GEMM_DFALT;
#if CUDA_VERSION >= 10010
  absl::optional<cublasLtMatmulAlgo_t> cublasLt_algo_;
#endif  // CUDA_VERSION >= 10010
};

struct MatricesInfo {
  static constexpr uint64_t kDefaultMatrixSize = 128;
  GEMMData gemmData;
  uint64_t m = kDefaultMatrixSize;
  uint64_t n = kDefaultMatrixSize;
  uint64_t k = kDefaultMatrixSize;
  cudaDataType_t input_data_type = CUDA_R_64F;
  cudaDataType_t output_data_type = CUDA_R_64F;
  cudaDataType_t compute_data_type = CUDA_R_64F;
  int64_t lda = kDefaultMatrixSize;
  int64_t ldb = kDefaultMatrixSize;
  int64_t ldc = kDefaultMatrixSize;
  cublasOperation_t opA = CUBLAS_OP_N;
  cublasOperation_t opB = CUBLAS_OP_N;
};

// This abstract class serves as an interface for cuda cublasGemmEx API that
// utilized by gemm_test. The actual backend cuBLAS call depends on the data
// types and GPU compute capability. This class is also responsible for
// backend-dependent setup. Each compute thread should have a single compute
// interface.
class GpuComputationInterface {
 public:
  virtual ~GpuComputationInterface() {}
  // Initializes the cublas environment and sets the stream to use for any
  // cuBLAS API calls. Note that this will prepare whichever device is currently
  // selected with cudaSetDevice(), or GPU 0 by default. Setup is separated from
  // constructor code so that the correct implementation of this interface can
  // be determined separately from GPU selection.
  virtual void Initialize(cudaStream_t stream) = 0;

  // Initialize the Matrix multiply A,B,C data params
  virtual void BindGemmMatrices(const ContextOption &context_options,
                                GEMMData &gemmData) = 0;

  // Executes a computation cycle. The backend cuBLAS function call depends on
  // the GPU architecture and potentially the CUDA version.
  virtual cublasStatus_t MatrixMultiComputation(const Algo &algo) = 0;

 protected:
};

// Modern interface for compute capability >= 5.0. Allows half and mixed
// precision computation, and limited integer based math.
class CudaCublasInterface final : public GpuComputationInterface {
 public:
  CudaCublasInterface() {}

  ~CudaCublasInterface() override;

  cublasStatus_t MatrixMultiComputation(const Algo &algo) override;

  void Initialize(cudaStream_t stream) override;

  void BindGemmMatrices(const ContextOption &context_options,
                        GEMMData &gemmData) override;

 private:
  cublasHandle_t cublas_handle_ = nullptr;
  MatricesInfo matrices_info_;
};

// Legacy interface for compute capability <= 5 (k80 and earlier). Supports
// only single and double precision.
template <typename T>
class LegacyCudaCublasInterface final : public GpuComputationInterface {
 public:
  LegacyCudaCublasInterface() {}

  ~LegacyCudaCublasInterface() override;

  cublasStatus_t MatrixMultiComputation(const Algo &algo) override;

  void Initialize(cudaStream_t stream) override;

  void BindGemmMatrices(const ContextOption &context_options,
                        GEMMData &gemmData) override;

 private:
  cublasHandle_t cublas_handle_ = nullptr;
  MatricesInfo matrices_info_;
};

#if CUDA_VERSION >= 10010
class CudaCublasLtInterface final : public GpuComputationInterface {
 public:
  CudaCublasLtInterface() {}
  ~CudaCublasLtInterface() override;

  cublasStatus_t MatrixMultiComputation(const Algo &algo) override;

  void Initialize(cudaStream_t stream) override;

  void BindGemmMatrices(const ContextOption &context_options,
                        GEMMData &gemmData) override;

  std::vector<cublasLtMatmulHeuristicResult_t> GetHeuristicResults(int maxNum);

 private:
  void FreeReuseResource();

 private:
  // Transpose operation constants.
  static constexpr cublasOperation_t kTransOpA = CUBLAS_OP_N;
  static constexpr cublasOperation_t kTransOpB = CUBLAS_OP_T;
  // Pointer mode constant.
  static constexpr cublasPointerMode_t kPointerMode =
      CUBLAS_POINTER_MODE_DEVICE;
  static constexpr int kAlpha = 1;
  static constexpr int kBeta = 0;

  size_t workspacesize_ = 0;
  void *workspace_ = nullptr;

  cublasLtHandle_t cublas_handle_ = nullptr;
  // There is no cublasSetStream equivalent for cublasLT. Instead, we store
  // the stream to use as input to the API call.
  cudaStream_t stream_;
  // CUDA math operation descriptor.
  cublasLtMatmulDesc_t matmul_desc_ = nullptr;
  // CUDA matrix descriptors.
  cublasLtMatrixLayout_t layout_a_ = nullptr;
  cublasLtMatrixLayout_t layout_b_ = nullptr;
  cublasLtMatrixLayout_t layout_c_ = nullptr;

  // allocated memory for transformed data
  void *mat_a_ = nullptr;
  void *mat_b_ = nullptr;
  void *mat_c_ = nullptr;

  // non transformed layout descriptor
  cublasLtMatrixLayout_t orig_a_ = nullptr;
  cublasLtMatrixLayout_t orig_b_ = nullptr;
  cublasLtMatrixLayout_t orig_c_ = nullptr;

  // transform descriptor
  cublasLtMatrixTransformDesc_t transform_desc_ = nullptr;

 private:
  MatricesInfo matrices_info_;
};
#endif  // CUDA_VERSION >= 10010

// Selects and creates a GEMM interface based on the precision of computation to
// be done and the capabilities of the GPU.
std::unique_ptr<GpuComputationInterface> SelectGemmInterface(
    const ContextOption &options, const ComputeCapability &compute_capability);

// GpuDataHandler stores pointers to GPU data, and handles allocation and
// freeing of those pointers. A GpuDataHandler is assigned to a single GPU.
// Allocation failures lead to immediate program exit.
// TODO: Return error status rather than crashing program on
// CUDA allocation failures.
template <typename InputPrecision, typename OutputPrecision,
          typename ComputePrecision>
class GpuDataHandler {
 public:
  GpuDataHandler() = default;
  ~GpuDataHandler();

  // Allocates input and output arrays. Allocates and sets scaling factors.
  // Copies input array data from host to device.
  void Initialize(const RandomMatrix<InputPrecision> *data_in_a,
                  const RandomMatrix<InputPrecision> *data_in_b,
                  const cudaStream_t stream);

  // Changes the GPU to which data will be copied.
  void SetGpuId(const int id) { gpu_id_ = id; }

  // Accessors.
  InputPrecision *InputA() const { return input_a_; }
  InputPrecision *InputB() const { return input_b_; }
  OutputPrecision *Output() const { return output_; }
  void *Alpha() const { return static_cast<void *>(alpha_); }
  void *Beta() const { return static_cast<void *>(beta_); }

 private:
  // Data will be allocated on this GPU
  int gpu_id_ = 0;

  // Input (alpha_) and output (beta_) scaling factors.
  ComputePrecision *alpha_ = nullptr;
  ComputePrecision *beta_ = nullptr;

  // Input and output matrices.
  InputPrecision *input_a_ = nullptr;
  InputPrecision *input_b_ = nullptr;
  OutputPrecision *output_ = nullptr;
};

// Derived and templated classes of HostContext class. Used to instantiate
// HostContext with different data input/output combinations.
template <typename P_in, typename P_out, typename P_compute>
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
template <typename P_in, typename P_out, typename P_compute>
class MixedPrecisionGpuContext : public GpuContext {
 public:
  MixedPrecisionGpuContext(
      HostContext *h, RandomMatrix<P_in> const *const matrix_a_p,
      RandomMatrix<P_in> const *const matrix_b_p, int gpu_num,
      std::unique_ptr<GpuComputationInterface> compute_interface);

  ~MixedPrecisionGpuContext() override;

  void StreamSynchronize() override;

  cudaError_t StreamQuery() override;

  void LaunchKernel() override;

  void AutoTuning() override;

 protected:
  GpuDataHandler<P_in, P_out, P_compute> data_handler_;
  cudaStream_t stream_;
  std::unique_ptr<GpuComputationInterface> compute_interface_;
};

extern template class GpuDataHandler<half_float::half, half_float::half,
                                     half_float::half>;
extern template class GpuDataHandler<half_float::half, float, float>;
extern template class GpuDataHandler<float, float, float>;
extern template class GpuDataHandler<double, double, double>;
extern template class GpuDataHandler<int8_t, int32_t, int32_t>;
extern template class GpuDataHandler<int8_t, float, float>;

extern template class MixedPrecisionHostContext<
    half_float::half, half_float::half, half_float::half>;
extern template class MixedPrecisionHostContext<half_float::half, float, float>;
extern template class MixedPrecisionHostContext<float, float, float>;
extern template class MixedPrecisionHostContext<double, double, double>;
extern template class MixedPrecisionHostContext<int8_t, int32_t, int32_t>;
extern template class MixedPrecisionHostContext<int8_t, float, float>;

#if CUDA_VERSION >= BF16_CUDA_VERSION
extern template class GpuDataHandler<nv_bfloat16, nv_bfloat16, float>;
extern template class MixedPrecisionHostContext<nv_bfloat16, nv_bfloat16,
                                                float>;
extern template class GpuDataHandler<nv_bfloat16, float, float>;
extern template class MixedPrecisionHostContext<nv_bfloat16, float, float>;
#endif

}  // namespace internal
}  // namespace gemm_test
}  // namespace platforms_gpus

#endif  // PLATFORMS_GPUS_TESTING_NVIDIA_GEMM_TEST_LIB_INTERNAL_H_
