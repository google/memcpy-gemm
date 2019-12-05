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

#include "gemm_test_lib_internal.h"

#include <cstdint>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"

namespace {

// This class is instantialized when registering a device to be used for GPU
// executions. Before register current gpu_num, it will store previous one.
// And restore previous gpu_num on deconstuction of the instance.
class WithCUDADevice {
 public:
  explicit WithCUDADevice(int gpu_num) {
    CUDA_CHECK(cudaGetDevice(&old_gpu_num_));
    CUDA_CHECK(cudaSetDevice(gpu_num));
  }
  ~WithCUDADevice() { CUDA_CHECK(cudaSetDevice(old_gpu_num_)) }

 private:
  int old_gpu_num_;
};

#if CUDA_VERSION >= 8000
cudaDataType_t GetCudaComputeType(const std::string &compute_type) {
  if (compute_type == "int32") {
    return CUDA_R_32I;
  } else if (compute_type == "half") {
    return CUDA_R_16F;
  } else if (compute_type == "single") {
    return CUDA_R_32F;
  } else if (compute_type == "double") {
    return CUDA_R_64F;
  }

  LOG(ERROR) << "Unsupported compute type, using double type.";
  return CUDA_R_64F;
}

cublasGemmAlgo_t GetGemmAlgorithm(const std::string &algorithm) {
  if (algorithm == "gemm_algo_default") {
    return CUBLAS_GEMM_DFALT;
  } else if (algorithm == "gemm_algo_0") {
    return CUBLAS_GEMM_ALGO0;
  } else if (algorithm == "gemm_algo_1") {
    return CUBLAS_GEMM_ALGO1;
  } else if (algorithm == "gemm_algo_2") {
    return CUBLAS_GEMM_ALGO2;
  } else if (algorithm == "gemm_algo_3") {
    return CUBLAS_GEMM_ALGO3;
  } else if (algorithm == "gemm_algo_4") {
    return CUBLAS_GEMM_ALGO4;
  } else if (algorithm == "gemm_algo_5") {
    return CUBLAS_GEMM_ALGO5;
  } else if (algorithm == "gemm_algo_6") {
    return CUBLAS_GEMM_ALGO6;
  } else if (algorithm == "gemm_algo_7") {
    return CUBLAS_GEMM_ALGO7;
#if CUDA_VERSION >= 9000
  } else if (algorithm == "gemm_algo_8") {
    return CUBLAS_GEMM_ALGO8;
  } else if (algorithm == "gemm_algo_9") {
    return CUBLAS_GEMM_ALGO9;
  } else if (algorithm == "gemm_algo_10") {
    return CUBLAS_GEMM_ALGO10;
  } else if (algorithm == "gemm_algo_11") {
    return CUBLAS_GEMM_ALGO11;
  } else if (algorithm == "gemm_algo_12") {
    return CUBLAS_GEMM_ALGO12;
  } else if (algorithm == "gemm_algo_13") {
    return CUBLAS_GEMM_ALGO13;
  } else if (algorithm == "gemm_algo_14") {
    return CUBLAS_GEMM_ALGO14;
  } else if (algorithm == "gemm_algo_15") {
    return CUBLAS_GEMM_ALGO15;
  } else if (algorithm == "gemm_algo_16") {
    return CUBLAS_GEMM_ALGO16;
  } else if (algorithm == "gemm_algo_17") {
    return CUBLAS_GEMM_ALGO17;
  } else if (algorithm == "gemm_tensor_algo_default") {
    return CUBLAS_GEMM_DFALT_TENSOR_OP;
  } else if (algorithm == "gemm_tensor_algo_0") {
    return CUBLAS_GEMM_ALGO0_TENSOR_OP;
  } else if (algorithm == "gemm_tensor_algo_1") {
    return CUBLAS_GEMM_ALGO1_TENSOR_OP;
  } else if (algorithm == "gemm_tensor_algo_2") {
    return CUBLAS_GEMM_ALGO2_TENSOR_OP;
  } else if (algorithm == "gemm_tensor_algo_3") {
    return CUBLAS_GEMM_ALGO3_TENSOR_OP;
  } else if (algorithm == "gemm_tensor_algo_4") {
    return CUBLAS_GEMM_ALGO4_TENSOR_OP;
#endif  // #if CUDA_VERSION >= 9000
#if CUDA_VERSION >= 10000
  } else if (algorithm == "gemm_algo_18") {
    return CUBLAS_GEMM_ALGO18;
  } else if (algorithm == "gemm_algo_19") {
    return CUBLAS_GEMM_ALGO19;
  } else if (algorithm == "gemm_algo_20") {
    return CUBLAS_GEMM_ALGO20;
  } else if (algorithm == "gemm_algo_21") {
    return CUBLAS_GEMM_ALGO21;
  } else if (algorithm == "gemm_algo_22") {
    return CUBLAS_GEMM_ALGO22;
  } else if (algorithm == "gemm_algo_23") {
    return CUBLAS_GEMM_ALGO23;
  } else if (algorithm == "gemm_tensor_algo_5") {
    return CUBLAS_GEMM_ALGO5_TENSOR_OP;
  } else if (algorithm == "gemm_tensor_algo_6") {
    return CUBLAS_GEMM_ALGO6_TENSOR_OP;
  } else if (algorithm == "gemm_tensor_algo_7") {
    return CUBLAS_GEMM_ALGO7_TENSOR_OP;
  } else if (algorithm == "gemm_tensor_algo_8") {
    return CUBLAS_GEMM_ALGO8_TENSOR_OP;
  } else if (algorithm == "gemm_tensor_algo_9") {
    return CUBLAS_GEMM_ALGO9_TENSOR_OP;
  } else if (algorithm == "gemm_tensor_algo_10") {
    return CUBLAS_GEMM_ALGO10_TENSOR_OP;
  } else if (algorithm == "gemm_tensor_algo_11") {
    return CUBLAS_GEMM_ALGO11_TENSOR_OP;
  } else if (algorithm == "gemm_tensor_algo_12") {
    return CUBLAS_GEMM_ALGO12_TENSOR_OP;
  } else if (algorithm == "gemm_tensor_algo_13") {
    return CUBLAS_GEMM_ALGO13_TENSOR_OP;
  } else if (algorithm == "gemm_tensor_algo_14") {
    return CUBLAS_GEMM_ALGO14_TENSOR_OP;
  } else if (algorithm == "gemm_tensor_algo_15") {
    return CUBLAS_GEMM_ALGO15_TENSOR_OP;
#endif  // #if CUDA_VERSION >= 10000
  }
  LOG(ERROR) << "Unsupported algorithm, using default.";
  return CUBLAS_GEMM_DFALT;
}
#endif  // #if CUDA_VERSION >= 8000
}  //  namespace

namespace platforms_gpus {
namespace gemm_test {
namespace internal {

template <>
cublasStatus_t
CudaCublasInterface<half_float::half, half_float::half>::MatrixMultiComputation(
    size_t dim_size_m, size_t dim_size_n, size_t dim_size_k, bool transa,
    bool transb, const std::string &compute_type, const std::string &algorithm,
    const std::string &algorithm_tc, cublasHandle_t handle, const void *alpha,
    const void *A, const void *B, const void *beta, void *C,
    int compute_capability_major) {
#if CUDA_VERSION >= 8000
    cublasStatus_t cublas_err;
    cublas_err = cublasGemmEx(handle, transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                      transb ? CUBLAS_OP_T : CUBLAS_OP_N, dim_size_m,
                      dim_size_n, dim_size_k, alpha, A, CUDA_R_16F, dim_size_m,
                      B, CUDA_R_16F, dim_size_k, beta, C, CUDA_R_16F,
                      dim_size_m, GetCudaComputeType(compute_type),
                      GetGemmAlgorithm(algorithm));
    if (cublas_err != CUBLAS_STATUS_SUCCESS)
        return cublas_err;
    if (!algorithm_tc.empty()) {
        cublas_err =  cublasGemmEx(handle, transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                      transb ? CUBLAS_OP_T : CUBLAS_OP_N, dim_size_m,
                      dim_size_n, dim_size_k, alpha, A, CUDA_R_16F, dim_size_m,
                      B, CUDA_R_16F, dim_size_k, beta, C, CUDA_R_16F,
                      dim_size_m, GetCudaComputeType(compute_type),
                      GetGemmAlgorithm(algorithm_tc));
    }
    return cublas_err;
#endif
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

template <>
cublasStatus_t
CudaCublasInterface<half_float::half, float>::MatrixMultiComputation(
    size_t dim_size_m, size_t dim_size_n, size_t dim_size_k, bool transa,
    bool transb, const std::string &compute_type, const std::string &algorithm,
    const std::string &algorithm_tc, cublasHandle_t handle, const void *alpha,
    const void *A, const void *B, const void *beta, void *C,
    int compute_capability_major) {
#if CUDA_VERSION >= 8000
  if (compute_capability_major >= 5) {
    cublasStatus_t cublas_err;
    cublas_err = cublasGemmEx(handle, transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                      transb ? CUBLAS_OP_T : CUBLAS_OP_N, dim_size_m,
                      dim_size_n, dim_size_k, alpha, A, CUDA_R_16F, dim_size_m,
                      B, CUDA_R_16F, dim_size_k, beta, C, CUDA_R_32F,
                      dim_size_m, GetCudaComputeType(compute_type),
                      GetGemmAlgorithm(algorithm));
    if (cublas_err != CUBLAS_STATUS_SUCCESS)
        return cublas_err;
    if (!algorithm_tc.empty()) {
        cublas_err =  cublasGemmEx(handle, transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                      transb ? CUBLAS_OP_T : CUBLAS_OP_N, dim_size_m,
                      dim_size_n, dim_size_k, alpha, A, CUDA_R_16F, dim_size_m,
                      B, CUDA_R_16F, dim_size_k, beta, C, CUDA_R_32F,
                      dim_size_m, GetCudaComputeType(compute_type),
                      GetGemmAlgorithm(algorithm_tc));
    }
    return cublas_err;
  } else {
    LOG(ERROR) << "This GPU doesn't support half data type";
  }
#endif
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

template <>
cublasStatus_t CudaCublasInterface<float, float>::MatrixMultiComputation(
    size_t dim_size_m, size_t dim_size_n, size_t dim_size_k, bool transa,
    bool transb, const std::string &compute_type, const std::string &algorithm,
    const std::string &algorithm_tc, cublasHandle_t handle, const void *alpha,
    const void *A, const void *B, const void *beta, void *C,
    int compute_capability_major) {
#if CUDA_VERSION >= 8000
  if (compute_capability_major >= 5) {
    cublasStatus_t cublas_err;
    cublas_err = cublasGemmEx(handle, transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                      transb ? CUBLAS_OP_T : CUBLAS_OP_N, dim_size_m,
                      dim_size_n, dim_size_k, alpha, A, CUDA_R_32F, dim_size_m,
                      B, CUDA_R_32F, dim_size_k, beta, C, CUDA_R_32F,
                      dim_size_m, GetCudaComputeType(compute_type),
                      GetGemmAlgorithm(algorithm));
    if (cublas_err != CUBLAS_STATUS_SUCCESS)
        return cublas_err;
    if (!algorithm_tc.empty()) {
        cublas_err = cublasGemmEx(handle, transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                      transb ? CUBLAS_OP_T : CUBLAS_OP_N, dim_size_m,
                      dim_size_n, dim_size_k, alpha, A, CUDA_R_32F, dim_size_m,
                      B, CUDA_R_32F, dim_size_k, beta, C, CUDA_R_32F,
                      dim_size_m, GetCudaComputeType(compute_type),
                      GetGemmAlgorithm(algorithm_tc));
    }
    return cublas_err;
  }
#endif
  return cublasSgemm(handle, transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                   transb ? CUBLAS_OP_T : CUBLAS_OP_N, dim_size_m,
                   dim_size_n, dim_size_k,
                   reinterpret_cast<const float *>(alpha),
                   reinterpret_cast<const float *>(A), dim_size_m,
                   reinterpret_cast<const float *>(B), dim_size_k,
                   reinterpret_cast<const float *>(beta),
                   reinterpret_cast<float *>(C), dim_size_m);
}

template <>
cublasStatus_t CudaCublasInterface<double, double>::MatrixMultiComputation(
    size_t dim_size_m, size_t dim_size_n, size_t dim_size_k, bool transa,
    bool transb, const std::string &compute_type, const std::string &algorithm,
    const std::string &algorithm_tc, cublasHandle_t handle, const void *alpha,
    const void *A, const void *B, const void *beta, void *C,
    int compute_capability_major) {
#if CUDA_VERSION >= 8000
  if (compute_capability_major >= 5) {
      cublasStatus_t cublas_err;
      cublas_err =  cublasGemmEx(handle, transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                      transb ? CUBLAS_OP_T : CUBLAS_OP_N, dim_size_m,
                      dim_size_n, dim_size_k, alpha, A, CUDA_R_64F, dim_size_m,
                      B, CUDA_R_64F, dim_size_k, beta, C, CUDA_R_64F,
                      dim_size_m, GetCudaComputeType(compute_type),
                      GetGemmAlgorithm(algorithm));
      if (cublas_err != CUBLAS_STATUS_SUCCESS)
          return cublas_err;
      if (!algorithm_tc.empty()) {
        cublas_err = cublasGemmEx(handle, transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                      transb ? CUBLAS_OP_T : CUBLAS_OP_N, dim_size_m,
                      dim_size_n, dim_size_k, alpha, A, CUDA_R_64F, dim_size_m,
                      B, CUDA_R_64F, dim_size_k, beta, C, CUDA_R_64F,
                      dim_size_m, GetCudaComputeType(compute_type),
                      GetGemmAlgorithm(algorithm_tc));
    }
    return cublas_err;
  }
#endif
  return cublasDgemm(handle, transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                     transb ? CUBLAS_OP_T : CUBLAS_OP_N, dim_size_m,
                     dim_size_n, dim_size_k,
                   reinterpret_cast<const double *>(alpha),
                   reinterpret_cast<const double *>(A), dim_size_m,
                   reinterpret_cast<const double *>(B), dim_size_k,
                   reinterpret_cast<const double *>(beta),
                   reinterpret_cast<double *>(C), dim_size_m);
}

// TODO: Use builder pattern to avoid the long parameter list
template <>
cublasStatus_t CudaCublasInterface<int8_t, int32_t>::MatrixMultiComputation(
    size_t dim_size_m, size_t dim_size_n, size_t dim_size_k, bool transa,
    bool transb, const std::string &compute_type, const std::string &algorithm,
    const std::string &algorithm_tc, cublasHandle_t handle, const void *alpha,
    const void *A, const void *B, const void *beta, void *C,
    int compute_capability_major) {
#if CUDA_VERSION >= 8000
  if (compute_capability_major >= 5) {
      cublasStatus_t cublas_err;
      cublas_err =  cublasGemmEx(handle, transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                      transb ? CUBLAS_OP_T : CUBLAS_OP_N, dim_size_m,
                      dim_size_n, dim_size_k, alpha, A, CUDA_R_8I, dim_size_m,
                      B, CUDA_R_8I, dim_size_k, beta, C, CUDA_R_32I,
                      dim_size_m, GetCudaComputeType(compute_type),
                      GetGemmAlgorithm(algorithm));
      if (cublas_err != CUBLAS_STATUS_SUCCESS)
          return cublas_err;
      if (!algorithm_tc.empty()) {
        cublas_err = cublasGemmEx(handle, transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                      transb ? CUBLAS_OP_T : CUBLAS_OP_N, dim_size_m,
                      dim_size_n, dim_size_k, alpha, A, CUDA_R_8I, dim_size_m,
                      B, CUDA_R_8I, dim_size_k, beta, C, CUDA_R_32I,
                      dim_size_m, GetCudaComputeType(compute_type),
                      GetGemmAlgorithm(algorithm_tc));
    }
    return cublas_err;
  }
#endif
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

template <typename P_in, typename P_out>
MixedPrecisionHostContext<P_in, P_out>::MixedPrecisionHostContext(
    const ContextOption &options)
    : MixedPrecisionHostContext<P_in, P_out>(
          options, absl::make_unique<CudaMemoryAllocator>(),
          absl::make_unique<internal::CudaCublasInterface<P_in, P_out>>()) {}

template <typename P_in, typename P_out>
MixedPrecisionHostContext<P_in, P_out>::MixedPrecisionHostContext(
    const ContextOption &options,
    std::unique_ptr<MemoryAllocatorInterface> memory_allocator,
    std::unique_ptr<CudaCublasInterface<P_in, P_out>> computation_interface)
    : HostContext(options),
      memory_allocator_(std::move(memory_allocator)),
      computation_interface_(std::move(computation_interface)),
      a_(options.dim_size_m, options.dim_size_k, memory_allocator_.get()),
      b_(options.dim_size_k, options.dim_size_n, memory_allocator_.get()) {
  a_.Initialize(options.rng, /*scale=*/1e30, options.gaussian);
  b_.Initialize(options.rng, /*scale=*/1e-30, options.gaussian);
}

template <typename P_in, typename P_out>
std::unique_ptr<GpuContext>
MixedPrecisionHostContext<P_in, P_out>::CreateGpuContext(int gpu_num) {
  return absl::make_unique<MixedPrecisionGpuContext<P_in, P_out>>(
      this, &a_, &b_, gpu_num, computation_interface_.get());
}

template <typename Compute_Type>
void CopyScalingFactorsToDevice(void **alpha, void **beta,
                                const cudaStream_t &stream) {
  static Compute_Type constants[] = {1., 0.};
  CUDA_CHECK(cudaMalloc(alpha, sizeof(constants)));
  *beta = reinterpret_cast<Compute_Type *>(*alpha) + 1;
  CUDA_CHECK(cudaMemcpyAsync(*alpha, constants, sizeof(constants),
                             cudaMemcpyHostToDevice, stream));
}

template <>
void CopyScalingFactorsToDevice<int32_t>(void **alpha, void **beta,
                                const cudaStream_t &stream) {
  static int32_t constants[] = {1, 0};
  CUDA_CHECK(cudaMalloc(alpha, sizeof(constants)));
  *beta = reinterpret_cast<int32_t *>(*alpha) + 1;
  CUDA_CHECK(cudaMemcpyAsync(*alpha, constants, sizeof(constants),
                             cudaMemcpyHostToDevice, stream));
}

template <>
void CopyScalingFactorsToDevice<half_float::half>(void **alpha, void **beta,
                                                  const cudaStream_t &stream) {
  // place alpha and beta on the device so they don't have
  // to be refetched in each iteration.
  static half_float::half constants[2];
  constants[0] = half_float::detail::float2half<std::round_to_nearest>(1.);
  constants[1] = half_float::detail::float2half<std::round_to_nearest>(0.);

  CUDA_CHECK(cudaMalloc(alpha, sizeof(constants)));
  *beta = reinterpret_cast<half_float::half *>(*alpha) + 1;
  CUDA_CHECK(cudaMemcpyAsync(*alpha, constants, sizeof(constants),
                             cudaMemcpyHostToDevice, stream));
}

void GenerateScalingFactor(
    const ContextOption &options, void **alpha, void **beta,
    const cudaStream_t &stream) {
  if (options.compute_type == "int32") {
    CopyScalingFactorsToDevice<int32_t>(alpha, beta, stream);
  } else if (options.compute_type == "half") {
    CopyScalingFactorsToDevice<half_float::half>(alpha, beta, stream);
  } else if (options.compute_type == "single") {
    CopyScalingFactorsToDevice<float>(alpha, beta, stream);
  } else if (options.compute_type == "double") {
    CopyScalingFactorsToDevice<double>(alpha, beta, stream);
  } else {
    LOG(ERROR) << "Unsupported computeType for cublasGemmEx";
  }
}

// Caller of this function has the ownership for pointers matrix_a_p, and
// matrix_a_p. Caller should make sure these two pointers are valid until
// cudaMemcpyAsync() complete (received callback).
template <typename P_in, typename P_out>
MixedPrecisionGpuContext<P_in, P_out>::MixedPrecisionGpuContext(
    HostContext *h, RandomMatrix<P_in> const *const matrix_a_p,
    RandomMatrix<P_in> const *const matrix_b_p, int gpu_num,
    GpuComputationInterface *compute_interface)
    : GpuContext(h->GetOption(), gpu_num),
      compute_interface_(compute_interface) {
  WithCUDADevice device(gpu_num_);
  cudaDeviceProp dev_prop;
  CUDA_CHECK(cudaGetDeviceProperties(&dev_prop, gpu_num_));
  LOG(INFO) << absl::StrFormat("using GPU %d at %04x:%02x:%02x model %s\n",
                               gpu_num_, dev_prop.pciDomainID,
                               dev_prop.pciBusID, dev_prop.pciDeviceID,
                               dev_prop.name);
  compute_capability_major_rev_ = dev_prop.major;
  CUDA_CHECK(cudaStreamCreate(&stream_));
  CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream_));
  CUBLAS_CHECK(
      cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE));
  // place alpha and beta on the device so they don't have
  // to be refetched in each iteration.
  GenerateScalingFactor(options_, &dev_alpha_, &dev_beta_, stream_);
  size_t nr_bytes_a = options_.dim_size_m * options_.dim_size_k * sizeof(P_in);
  size_t nr_bytes_b = options_.dim_size_k * options_.dim_size_n * sizeof(P_in);
  size_t nr_bytes_c = options_.dim_size_m * options_.dim_size_n * sizeof(P_out);
  CUDA_CHECK(cudaMalloc(&dev_a_, nr_bytes_a));
  CUDA_CHECK(cudaMalloc(&dev_b_, nr_bytes_b));
  CUDA_CHECK(cudaMalloc(&dev_c_, nr_bytes_c));
  CUDA_CHECK(cudaMemcpyAsync(dev_a_, matrix_a_p->Get(), nr_bytes_a,
                             cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaMemcpyAsync(dev_b_, matrix_b_p->Get(), nr_bytes_b,
                             cudaMemcpyHostToDevice, stream_));
}

template <typename P_in, typename P_out>
MixedPrecisionGpuContext<P_in, P_out>::~MixedPrecisionGpuContext() {
  CUDA_CHECK(cudaFree(dev_a_));
  CUDA_CHECK(cudaFree(dev_b_));
  CUDA_CHECK(cudaFree(dev_c_));
  // when allocating memory for alpha and beta togeter, with the address stored
  // in dev_alpha_, so we only free dev_alpha_, but not dev_beta_.
  CUDA_CHECK(cudaFree(dev_alpha_));
  CUDA_CHECK(cudaStreamDestroy(stream_));
}

template <typename P_in, typename P_out>
void MixedPrecisionGpuContext<P_in, P_out>::StreamSynchronize() {
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

template <typename P_in, typename P_out>
cudaError_t MixedPrecisionGpuContext<P_in, P_out>::StreamQuery() {
  return cudaStreamQuery(stream_);
}


template <typename P_in, typename P_out>
void MixedPrecisionGpuContext<P_in, P_out>::LaunchKernel() {
  WithCUDADevice device(gpu_num_);

  CUBLAS_CHECK(compute_interface_->MatrixMultiComputation(
      options_.dim_size_m, options_.dim_size_n, options_.dim_size_k,
      options_.transa, options_.transb, options_.compute_type,
      options_.algorithm, options_.algorithm_tc, cublas_handle_, dev_alpha_,
      dev_a_, dev_b_, dev_beta_, dev_c_, compute_capability_major_rev_));
}

template class MixedPrecisionHostContext<half_float::half, half_float::half>;
template class MixedPrecisionHostContext<half_float::half, float>;
template class MixedPrecisionHostContext<float, float>;
template class MixedPrecisionHostContext<double, double>;
template class MixedPrecisionHostContext<int8_t, int32_t>;

}  // namespace internal
}  // namespace gemm_test
}  // namespace platforms_gpus
