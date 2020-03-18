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

#include "src/gemm_test_lib_internal.h"

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "src/matrix_lib.h"

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

cudaDataType_t GetCudaComputeType(absl::string_view compute_type) {
  static const absl::flat_hash_map<absl::string_view, cudaDataType_t>
      compute_mapping{{"half", CUDA_R_16F},
                      {"single", CUDA_R_32F},
                      {"double", CUDA_R_64F},
                      {"int8", CUDA_R_8I},
                      {"int32", CUDA_R_32I}};

  if (const auto it = compute_mapping.find(compute_type);
      it != compute_mapping.end()) {
    return it->second;
  }
  LOG(ERROR) << absl::Substitute(
      "Unsupported compute type '$0', using double type.", compute_type);
  return CUDA_R_64F;
}

cublasGemmAlgo_t GetGemmAlgorithm(absl::string_view algorithm) {
  static const absl::flat_hash_map<absl::string_view, cublasGemmAlgo_t>
      algorithm_mapping {
    {"gemm_algo_default", CUBLAS_GEMM_DFALT},
        {"gemm_algo_0", CUBLAS_GEMM_ALGO0}, {"gemm_algo_1", CUBLAS_GEMM_ALGO1},
        {"gemm_algo_2", CUBLAS_GEMM_ALGO2}, {"gemm_algo_3", CUBLAS_GEMM_ALGO3},
        {"gemm_algo_4", CUBLAS_GEMM_ALGO4}, {"gemm_algo_5", CUBLAS_GEMM_ALGO5},
        {"gemm_algo_6", CUBLAS_GEMM_ALGO6}, {"gemm_algo_7", CUBLAS_GEMM_ALGO7},
        {"gemm_algo_8", CUBLAS_GEMM_ALGO8}, {"gemm_algo_9", CUBLAS_GEMM_ALGO9},
        {"gemm_algo_10", CUBLAS_GEMM_ALGO10},
        {"gemm_algo_11", CUBLAS_GEMM_ALGO11},
        {"gemm_algo_12", CUBLAS_GEMM_ALGO12},
        {"gemm_algo_13", CUBLAS_GEMM_ALGO13},
        {"gemm_algo_14", CUBLAS_GEMM_ALGO14},
        {"gemm_algo_15", CUBLAS_GEMM_ALGO15},
        {"gemm_algo_16", CUBLAS_GEMM_ALGO16},
        {"gemm_algo_17", CUBLAS_GEMM_ALGO17},
#if CUDA_VERSION >= 10000
        {"gemm_algo_18", CUBLAS_GEMM_ALGO18},
        {"gemm_algo_19", CUBLAS_GEMM_ALGO19},
        {"gemm_algo_20", CUBLAS_GEMM_ALGO20},
        {"gemm_algo_21", CUBLAS_GEMM_ALGO21},
        {"gemm_algo_22", CUBLAS_GEMM_ALGO22},
        {"gemm_algo_23", CUBLAS_GEMM_ALGO23},
#endif  // CUDA_VERSION >= 10000
        {"gemm_tensor_algo_default", CUBLAS_GEMM_DFALT_TENSOR_OP},
        {"gemm_tensor_algo_0", CUBLAS_GEMM_ALGO0_TENSOR_OP},
        {"gemm_tensor_algo_1", CUBLAS_GEMM_ALGO1_TENSOR_OP},
        {"gemm_tensor_algo_2", CUBLAS_GEMM_ALGO2_TENSOR_OP},
        {"gemm_tensor_algo_3", CUBLAS_GEMM_ALGO3_TENSOR_OP},
        {"gemm_tensor_algo_4", CUBLAS_GEMM_ALGO4_TENSOR_OP},
#if CUDA_VERSION >= 10000
        {"gemm_tensor_algo_5", CUBLAS_GEMM_ALGO5_TENSOR_OP},
        {"gemm_tensor_algo_6", CUBLAS_GEMM_ALGO6_TENSOR_OP},
        {"gemm_tensor_algo_7", CUBLAS_GEMM_ALGO7_TENSOR_OP},
        {"gemm_tensor_algo_8", CUBLAS_GEMM_ALGO8_TENSOR_OP},
        {"gemm_tensor_algo_9", CUBLAS_GEMM_ALGO9_TENSOR_OP},
        {"gemm_tensor_algo_10", CUBLAS_GEMM_ALGO10_TENSOR_OP},
        {"gemm_tensor_algo_11", CUBLAS_GEMM_ALGO11_TENSOR_OP},
        {"gemm_tensor_algo_12", CUBLAS_GEMM_ALGO12_TENSOR_OP},
        {"gemm_tensor_algo_13", CUBLAS_GEMM_ALGO13_TENSOR_OP},
        {"gemm_tensor_algo_14", CUBLAS_GEMM_ALGO14_TENSOR_OP},
        {"gemm_tensor_algo_15", CUBLAS_GEMM_ALGO15_TENSOR_OP},
#endif  // CUDA_VERSION >= 10000
  };

  if (const auto it = algorithm_mapping.find(algorithm);
      it != algorithm_mapping.end()) {
    return it->second;
  }
  LOG(ERROR) << absl::Substitute(
      "Unsupported algorithm type '$0', using default.", algorithm);
  return CUBLAS_GEMM_DFALT;
}

// Detects the compute capability of GPU 0.
float GetComputeCapability() {
  WithCUDADevice device(0);
  cudaDeviceProp dev_prop;
  CUDA_CHECK(cudaGetDeviceProperties(&dev_prop, 0));
  return dev_prop.major;
}

}  //  namespace

namespace platforms_gpus {
namespace gemm_test {
namespace internal {

std::unique_ptr<GpuComputationInterface> SelectGemmInterface(
    absl::string_view compute_type, const float compute_capability) {
  // If on a capable machine, use the modern cublasGemmEx() wrapper which can
  // handle any combination of data types.
  if (compute_capability >= 5.0) {
    LOG(INFO) << "Using cublasGemmEx for GEMM computation";
    return absl::make_unique<CudaCublasInterface>();
  }
  // If on an older machine, the data type determines the function to be called.
  // Note that we don't have to worry about mixed precision, since older devices
  // do not support it.
  if (compute_type == "single") {
    LOG(INFO) << "Using cublasSGemm for GEMM computation";
    return absl::make_unique<LegacyCudaCublasInterface<float>>();
  } else if (compute_type == "double") {
    LOG(INFO) << "Using cublasDGemm for GEMM computation";
    return absl::make_unique<LegacyCudaCublasInterface<double>>();
  }
  LOG(ERROR) << absl::Substitute(
      "Unsupported data type $0 for legacy cublas interface", compute_type);
  return nullptr;
}

cublasStatus_t CudaCublasInterface::MatrixMultiComputation(
    const ContextOption &context_options, cublasHandle_t handle,
    const void *alpha, const void *A, const void *B, const void *beta,
    void *C) {
  return cublasGemmEx(handle,
                      // Transpose before multiplication.
                      context_options.transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                      context_options.transb ? CUBLAS_OP_T : CUBLAS_OP_N,
                      // Matrix dimensions,
                      context_options.dim_size_m, context_options.dim_size_n,
                      context_options.dim_size_k,
                      // Input arrays and scaling factors.
                      alpha, A,
                      GetCudaComputeType(context_options.data_type_in),
                      context_options.dim_size_m, B,
                      GetCudaComputeType(context_options.data_type_in),
                      // Output array options.
                      context_options.dim_size_k, beta, C,
                      GetCudaComputeType(context_options.data_type_out),
                      context_options.dim_size_m,
                      // Compute options.
                      GetCudaComputeType(context_options.compute_type),
                      GetGemmAlgorithm(context_options.algorithm));
}

template <>
cublasStatus_t LegacyCudaCublasInterface<float>::MatrixMultiComputation(
    const ContextOption &context_options, cublasHandle_t handle,
    const void *alpha, const void *A, const void *B, const void *beta,
    void *C) {
  return cublasSgemm(
      handle, context_options.transa ? CUBLAS_OP_T : CUBLAS_OP_N,
      context_options.transb ? CUBLAS_OP_T : CUBLAS_OP_N,
      context_options.dim_size_m, context_options.dim_size_n,
      context_options.dim_size_k, reinterpret_cast<const float *>(alpha),
      reinterpret_cast<const float *>(A), context_options.dim_size_m,
      reinterpret_cast<const float *>(B), context_options.dim_size_k,
      reinterpret_cast<const float *>(beta), reinterpret_cast<float *>(C),
      context_options.dim_size_m);
}

template <>
cublasStatus_t LegacyCudaCublasInterface<double>::MatrixMultiComputation(
    const ContextOption &context_options, cublasHandle_t handle,
    const void *alpha, const void *A, const void *B, const void *beta,
    void *C) {
  return cublasDgemm(
      handle, context_options.transa ? CUBLAS_OP_T : CUBLAS_OP_N,
      context_options.transb ? CUBLAS_OP_T : CUBLAS_OP_N,
      context_options.dim_size_m, context_options.dim_size_n,
      context_options.dim_size_k, reinterpret_cast<const double *>(alpha),
      reinterpret_cast<const double *>(A), context_options.dim_size_m,
      reinterpret_cast<const double *>(B), context_options.dim_size_k,
      reinterpret_cast<const double *>(beta), reinterpret_cast<double *>(C),
      context_options.dim_size_m);
}

template <typename P_in, typename P_out>
MixedPrecisionHostContext<P_in, P_out>::MixedPrecisionHostContext(
    const ContextOption &options)
    : MixedPrecisionHostContext<P_in, P_out>(
          options, absl::make_unique<CudaMemoryAllocator>(),
          SelectGemmInterface(options.compute_type, GetComputeCapability())) {}

template <typename P_in, typename P_out>
MixedPrecisionHostContext<P_in, P_out>::MixedPrecisionHostContext(
    const ContextOption &options,
    std::unique_ptr<MemoryAllocatorInterface> memory_allocator,
    std::unique_ptr<GpuComputationInterface> computation_interface)
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

void GenerateScalingFactor(absl::string_view compute_type, void **alpha,
                           void **beta, const cudaStream_t &stream) {
  if (compute_type == "int32") {
    CopyScalingFactorsToDevice<int32_t>(alpha, beta, stream);
  } else if (compute_type == "half") {
    CopyScalingFactorsToDevice<half_float::half>(alpha, beta, stream);
  } else if (compute_type == "single") {
    CopyScalingFactorsToDevice<float>(alpha, beta, stream);
  } else if (compute_type == "double") {
    CopyScalingFactorsToDevice<double>(alpha, beta, stream);
  } else {
    LOG(ERROR) << absl::Substitute(
        "Unsupported computeType '$0' for cublasGemmEx", compute_type);
  }
}

template <typename InputPrecision, typename OutputPrecision>
GpuDataHandler<InputPrecision, OutputPrecision>::~GpuDataHandler() {
  // cudaFree is safe to call on already freed or unallocated memory, so no
  // allocation tracking logic is needed.
  cudaFree(input_a_);
  cudaFree(input_b_);
  cudaFree(output_);
  // Alpha and beta were allocated as a size 2 array from alpha.
  cudaFree(alpha_);
}

template <typename InputPrecision, typename OutputPrecision>
void GpuDataHandler<InputPrecision, OutputPrecision>::Initialize(
    const RandomMatrix<InputPrecision> *data_in_a,
    const RandomMatrix<InputPrecision> *data_in_b, const cudaStream_t stream) {
  GenerateScalingFactor(compute_type_, &alpha_, &beta_, stream);
  const int num_bytes_a = data_in_a->SizeInBytes();
  const int num_bytes_b = data_in_b->SizeInBytes();
  CUDA_CHECK(cudaMalloc(&input_a_, num_bytes_a));
  CUDA_CHECK(cudaMalloc(&input_b_, num_bytes_b));
  const int n_bytes_out = data_in_a->GetDimSizeM() * data_in_b->GetDimSizeK() *
                          sizeof(OutputPrecision);
  CUDA_CHECK(cudaMalloc(&output_, n_bytes_out));

  CUDA_CHECK(cudaMemcpyAsync(input_a_, data_in_a->Get(), num_bytes_a,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(input_b_, data_in_b->Get(), num_bytes_b,
                             cudaMemcpyHostToDevice, stream));
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
  CUDA_CHECK(cudaStreamCreate(&stream_));
  CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream_));
  CUBLAS_CHECK(
      cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE));

  data_handler_.SetComputeType(options_.compute_type);
  data_handler_.SetGpuId(gpu_num_);
  data_handler_.Initialize(matrix_a_p, matrix_b_p, stream_);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

template <typename P_in, typename P_out>
MixedPrecisionGpuContext<P_in, P_out>::~MixedPrecisionGpuContext() {
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
      options_, cublas_handle_, data_handler_.Alpha(), data_handler_.InputA(),
      data_handler_.InputB(), data_handler_.Beta(), data_handler_.Output()));
}

template class LegacyCudaCublasInterface<float>;
template class LegacyCudaCublasInterface<double>;

template class GpuDataHandler<half_float::half, half_float::half>;
template class GpuDataHandler<half_float::half, float>;
template class GpuDataHandler<float, float>;
template class GpuDataHandler<double, double>;
template class GpuDataHandler<int8_t, int32_t>;
template class GpuDataHandler<int8_t, float>;

template class MixedPrecisionHostContext<half_float::half, half_float::half>;
template class MixedPrecisionHostContext<half_float::half, float>;
template class MixedPrecisionHostContext<float, float>;
template class MixedPrecisionHostContext<double, double>;
template class MixedPrecisionHostContext<int8_t, int32_t>;
template class MixedPrecisionHostContext<int8_t, float>;

}  // namespace internal
}  // namespace gemm_test
}  // namespace platforms_gpus
