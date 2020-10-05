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

#include <cstddef>
#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "src/matrix_lib.h"
#include "cuda/include/cuda.h"

#if CUDA_VERSION >= BF16_CUDA_VERSION
#include "cuda/include/cuda_bf16.h"
#endif

namespace platforms_gpus {
namespace gemm_test {
namespace internal {

namespace {
// This class is instantiated when registering a device to be used for GPU
// executions. Before register current gpu_num, it will store previous one.
// And restore previous gpu_num on deconstruction of the instance.
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

cudaDataType_t GetCudaDataType(absl::string_view data_type) {
  static const absl::flat_hash_map<absl::string_view, cudaDataType_t>
      data_mapping {
    {"half", CUDA_R_16F}, {"single", CUDA_R_32F}, {"double", CUDA_R_64F},
        {"int8", CUDA_R_8I}, {"int32", CUDA_R_32I},
#if CUDA_VERSION >= 11000
        {"bf16", CUDA_R_16BF},
#endif  // CUDA_VERSION >= 11000
  };

  if (const auto it = data_mapping.find(data_type); it != data_mapping.end()) {
    return it->second;
  }
  LOG(ERROR) << absl::Substitute(
      "Unsupported compute type '$0', using double type.", data_type);
  return CUDA_R_64F;
}

// TODO CUDA 11 cublasGemmEx uses cublasComputeType_t instead
// of cublasDataType_t. This setup works at the moment because of enum
// equivalence, but is very sketchy. Separate out cuda 10 vs 11 implementations.
cudaDataType_t GetCudaComputeType(absl::string_view compute_type) {
  static const absl::flat_hash_map<absl::string_view, cudaDataType_t>
      compute_mapping{
          {"half", CUDA_R_16F},
          {"single", CUDA_R_32F},
          {"double", CUDA_R_64F},
          {"int32", CUDA_R_32I},
      };

  if (const auto it = compute_mapping.find(compute_type);
      it != compute_mapping.end()) {
    return it->second;
  }
  LOG(ERROR) << absl::Substitute(
      "Unsupported compute type '$0', using double type.", compute_type);
  return CUDA_R_64F;
}
#if CUDA_VERSION >= 11000
// In Cuda 11,  compute type is sepearated from data type
// The newly introduced "cublasComputeType_t" changes function prototypes on the
// API: cublasGemmEx, cublasGemmBatchedEx, and cublasGemmStridedBatchedEx have a
// new signature that uses cublasComputeType_t for the computeType parameter.
// Backward compatibility is ensured with internal mapping for C users and with
// added overload for C++ users.
// TODO will support more compute type,
cublasComputeType_t GetNewComputeDataType(absl::string_view compute_type) {
  static const absl::flat_hash_map<absl::string_view, cublasComputeType_t>
      compute_mapping{
          {"half", CUBLAS_COMPUTE_16F},
          {"single", CUBLAS_COMPUTE_32F},
          {"double", CUBLAS_COMPUTE_64F},
          {"int32", CUBLAS_COMPUTE_32I},
      };
  if (const auto it = compute_mapping.find(compute_type);
      it != compute_mapping.end()) {
    return it->second;
  }
  LOG(ERROR) << absl::Substitute(
      "Unsupported compute type '$0', using double type.", compute_type);
  return CUBLAS_COMPUTE_64F;
}
#endif  // CUDA_VERSION >= 11000

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

bool dataTypeIMMAKernel(const ContextOption &options) {
  return options.data_type_in == "int8" && options.compute_type == "int32" &&
         (options.data_type_out == "int32" || options.data_type_out == "int8");
}

// Returns true if the data layout of matrices A and B support IMMA kernel
// accelerated int8 computation. In our current implementation, this is the case
// if matrics A and B have identical dimensions and are square, with
// input type int8 and output/compute type int32.
bool OptionsSupportIMMA(const ContextOption &options) {
  return dataTypeIMMAKernel(options) &&
         options.dim_size_m == options.dim_size_n &&
         options.dim_size_n == options.dim_size_k;
}

}  //  namespace

std::unique_ptr<GpuComputationInterface> SelectGemmInterface(
    const ContextOption &options, const ComputeCapability &compute_capability) {
#if CUDA_VERSION >= 10010
  // TODO Currently Only Enabled when Specified,
  // And also disable IMMA kernel, IMMA need more transformation work
  if (options.use_cublasLt_ && !dataTypeIMMAKernel(options)) {
    LOG(INFO) << "Using cublasLT for GEMM computation";
    return absl::make_unique<CudaCublasLtInterface>();
  }
  // Turing and later architectures support tensor int8 operations. Note that
  // this does not include compute capability == 7 (Volta), which has CUDA core
  // only int8 support with int32 output. Currently, this implementation is
  // restricted to square matrix multiplication.
  if (OptionsSupportIMMA(options) &&
      (compute_capability.major > 7 ||
       (compute_capability.major == 7 && compute_capability.minor > 0))) {
    LOG(INFO) << "Using cublasLT IMMA for GEMM computation";
    return absl::make_unique<CudaCublasLtInterface>();
  }
#endif  // CUDA_VERSION >= 10010

  // If on a capable machine, use the modern cublasGemmEx() wrapper which can
  // handle any combination of data types except int8 out.
  if (compute_capability.major >= 5 && options.data_type_out != "int8") {
    LOG(INFO) << "Using cublasGemmEx for GEMM computation";
    return absl::make_unique<CudaCublasInterface>();
  }
  // If on an older machine, the data type determines the function to be called.
  // Note that we don't have to worry about mixed precision, since older devices
  // do not support it.
  if (options.data_type_in == "single" && options.data_type_out == "single" &&
      options.compute_type == "single") {
    LOG(INFO) << "Using cublasSGemm for GEMM computation";
    return absl::make_unique<LegacyCudaCublasInterface<float>>();
  }
  if (options.data_type_in == "double" && options.data_type_out == "double" &&
      options.compute_type == "double") {
    LOG(INFO) << "Using cublasDGemm for GEMM computation";
    return absl::make_unique<LegacyCudaCublasInterface<double>>();
  }
  LOG(ERROR) << absl::Substitute(
      "Memcpy-gemm does not support the combination of:\n"
      "input_type=$0\noutput_type=$1\ncompute_type=$2\n"
      "with GPU compute capability $3.$4",
      options.data_type_in, options.data_type_out, options.compute_type,
      compute_capability.major, compute_capability.minor);
  return nullptr;
}

CudaCublasInterface::~CudaCublasInterface() {
  if (cublas_handle_ != nullptr) {
    cublasDestroy(cublas_handle_);
  }
}

void CudaCublasInterface::Initialize(cudaStream_t stream,
                                     const ContextOption &context_options) {
  CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  CUBLAS_CHECK(
      cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE));
  CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream));
}

cublasStatus_t CudaCublasInterface::MatrixMultiComputation(
    const ContextOption &context_options, const void *alpha, const void *A,
    const void *B, const void *beta, void *C) {
  return cublasGemmEx(cublas_handle_,
                      // Transpose before multiplication.
                      context_options.transa ? CUBLAS_OP_T : CUBLAS_OP_N,
                      context_options.transb ? CUBLAS_OP_T : CUBLAS_OP_N,
                      // Matrix dimensions,
                      context_options.dim_size_m, context_options.dim_size_n,
                      context_options.dim_size_k,
                      // Input arrays and scaling factors.
                      alpha, A, GetCudaDataType(context_options.data_type_in),
                      context_options.dim_size_m, B,
                      GetCudaDataType(context_options.data_type_in),
                      // Output array options.
                      context_options.dim_size_k, beta, C,
                      GetCudaDataType(context_options.data_type_out),
                      context_options.dim_size_m,
                      // Compute options.
                      GetCudaComputeType(context_options.compute_type),
                      GetGemmAlgorithm(context_options.algorithm));
}

template <typename T>
LegacyCudaCublasInterface<T>::~LegacyCudaCublasInterface() {
  if (cublas_handle_ != nullptr) {
    cublasDestroy(cublas_handle_);
  }
}

template <typename T>
void LegacyCudaCublasInterface<T>::Initialize(
    cudaStream_t stream, const ContextOption &context_options) {
  CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  CUBLAS_CHECK(
      cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE));
  CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream));
}

template <>
cublasStatus_t LegacyCudaCublasInterface<float>::MatrixMultiComputation(
    const ContextOption &context_options, const void *alpha, const void *A,
    const void *B, const void *beta, void *C) {
  return cublasSgemm(
      cublas_handle_, context_options.transa ? CUBLAS_OP_T : CUBLAS_OP_N,
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
    const ContextOption &context_options, const void *alpha, const void *A,
    const void *B, const void *beta, void *C) {
  return cublasDgemm(
      cublas_handle_, context_options.transa ? CUBLAS_OP_T : CUBLAS_OP_N,
      context_options.transb ? CUBLAS_OP_T : CUBLAS_OP_N,
      context_options.dim_size_m, context_options.dim_size_n,
      context_options.dim_size_k, reinterpret_cast<const double *>(alpha),
      reinterpret_cast<const double *>(A), context_options.dim_size_m,
      reinterpret_cast<const double *>(B), context_options.dim_size_k,
      reinterpret_cast<const double *>(beta), reinterpret_cast<double *>(C),
      context_options.dim_size_m);
}

#if CUDA_VERSION >= 10010
CudaCublasLtInterface::~CudaCublasLtInterface() {
  if (workspace_ != nullptr) {
    CUDA_CHECK(cudaFree(workspace_));
  }
  if (matmul_desc_ != nullptr) {
    cublasLtMatmulDescDestroy(matmul_desc_);
  }
  if (layout_a_ != nullptr) {
    cublasLtMatrixLayoutDestroy(layout_a_);
  }
  if (layout_b_ != nullptr) {
    cublasLtMatrixLayoutDestroy(layout_b_);
  }
  if (layout_c_ != nullptr) {
    cublasLtMatrixLayoutDestroy(layout_c_);
  }
  if (cublas_handle_ != nullptr) {
    cublasLtDestroy(cublas_handle_);
  }
}

void CudaCublasLtInterface::Initialize(cudaStream_t stream,
                                       const ContextOption &options) {
  stream_ = stream;
  // workspace size should be 256 bytes aligned.
  // Too small  workspace size may cause some routines to fail with
  // CUBLAS_STATUS_ALLOC_FAILED error returned or cause large regressions in
  // performance. Workspace size equal to or larger than 16KiB is enough to
  // prevent CUBLAS_STATUS_ALLOC_FAILED error, while a larger workspace can
  // provide performance benefits for some routines. Recommended size of
  // user-provided workspace is at least 4MiB (to match cuBLAS’ default
  // workspace pool). Currently We use 16MB.
  static constexpr size_t kWorkspaceSize = 16 * 1024 * 1024;
  workspacesize_ = kWorkspaceSize;

  const cudaDataType_t input_type = GetCudaDataType(options.data_type_in);
  const cudaDataType_t output_type = GetCudaDataType(options.data_type_out);
  const cudaDataType_t compute_type = GetCudaComputeType(options.compute_type);
  cublasOperation_t transA = input_type != CUDA_R_8I
                                 ? (options.transa ? CUBLAS_OP_T : CUBLAS_OP_N)
                                 : kTransOpA;
  cublasOperation_t transB = input_type != CUDA_R_8I
                                 ? (options.transb ? CUBLAS_OP_T : CUBLAS_OP_N)
                                 : kTransOpB;

  CUBLAS_CHECK(cublasLtCreate(&cublas_handle_));
#if CUDA_VERSION >= 11000
  CUBLAS_CHECK(cublasLtMatmulDescCreate(
      &matmul_desc_, GetNewComputeDataType(options.compute_type),
      compute_type));
#else
  CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmul_desc_, compute_type));
#endif  // CUDA_VERSION >= 11000

  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_desc_, CUBLASLT_MATMUL_DESC_POINTER_MODE, &kPointerMode,
      sizeof(kPointerMode)));

  // Computation descriptors.
#if CUDA_VERSION < 11000
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_desc_, CUBLASLT_MATMUL_DESC_COMPUTE_TYPE, &compute_type,
      sizeof(compute_type)));
#endif
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      matmul_desc_, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &compute_type,
      sizeof(compute_type)));

  // Matrix descriptors
  CUBLAS_CHECK(
      cublasLtMatrixLayoutCreate(&layout_a_, input_type, options.dim_size_m,
                                 options.dim_size_k, options.dim_size_m))

  const cublasLtOrder_t matACLayout =
      input_type != CUDA_R_8I ? CUBLASLT_ORDER_COL : CUBLASLT_ORDER_COL32;
  CUBLAS_CHECK(
      cublasLtMatrixLayoutSetAttribute(layout_a_, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                       &matACLayout, sizeof(matACLayout)));
  CUBLAS_CHECK(
      cublasLtMatrixLayoutCreate(&layout_b_, input_type, options.dim_size_k,
                                 options.dim_size_n, options.dim_size_k));

  cublasLtOrder_t matBLayout = CUBLASLT_ORDER_COL;
  if (input_type == CUDA_R_8I) {
    const ComputeCapability compute_capability = GetComputeCapability();
    if (compute_capability.major == 7 && compute_capability.minor > 0) {
      LOG(INFO) << "Optimizing matrix layout for Turing architecture";
      matBLayout = CUBLASLT_ORDER_COL4_4R2_8C;
    }
#if CUDA_VERSION >= 11000
    // int8 compute with CUDA 11 on Ampere machines can be further optimized
    // using a new CUDA 11 matrix layout.
    else if (compute_capability.major >= 8) {
      LOG(INFO) << "Optimizing matrix layout for Ampere architecture";
      matBLayout = CUBLASLT_ORDER_COL32_2R_4R4;
    }
#endif  // CUDA_VERSION >= 11000
  }

  CUBLAS_CHECK(
      cublasLtMatrixLayoutSetAttribute(layout_b_, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                       &matBLayout, sizeof(matBLayout)));
  CUBLAS_CHECK(
      cublasLtMatrixLayoutCreate(&layout_c_, output_type, options.dim_size_m,
                                 options.dim_size_n, options.dim_size_m));
  CUBLAS_CHECK(
      cublasLtMatrixLayoutSetAttribute(layout_c_, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                       &matACLayout, sizeof(matACLayout)));

  if (workspacesize_ > 0) {
    CUDA_CHECK(cudaMalloc(&workspace_, workspacesize_));
  }
}

cublasStatus_t CudaCublasLtInterface::MatrixMultiComputation(
    const ContextOption &context_options, const void *alpha, const void *A,
    const void *B, const void *beta, void *C) {
  const cublasLtMatmulAlgo_t *algo = nullptr;
  if (context_options.algo) {
    algo = &context_options.algo.value();
  }
  return cublasLtMatmul(cublas_handle_, matmul_desc_,
                        // input and scaling factors.
                        alpha, A, layout_a_, B, layout_b_,
                        // output and scaling factors.
                        beta, C, layout_c_,
                        // This spot is for 'D', the output variable, but in our
                        // case the math is in-place and C=D.
                        C, layout_c_, algo, workspace_, workspacesize_,
                        stream_);
}

std::vector<cublasLtMatmulHeuristicResult_t>
CudaCublasLtInterface::GetHeuristicResults(int maxNum) {
  std::vector<cublasLtMatmulHeuristicResult_t> results(maxNum);

  cublasLtMatmulPreference_t preference = nullptr;
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspacesize_,
      sizeof(workspacesize_)));

  int heuristicNum = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      cublas_handle_, matmul_desc_, layout_a_, layout_b_, layout_c_, layout_c_,
      preference, maxNum, results.data(), &heuristicNum));

  if (preference) CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  LOG(INFO) << absl::StreamFormat("GetMatmulAlgoHeuristic %d returned ",
                                  heuristicNum);

  results.resize(heuristicNum);
  return results;
}
#endif  // CUDA_VERSION >= 10010

template <typename P_in, typename P_out, typename P_compute>
MixedPrecisionHostContext<P_in, P_out, P_compute>::MixedPrecisionHostContext(
    const ContextOption &options)
    : MixedPrecisionHostContext<P_in, P_out, P_compute>(
          options, absl::make_unique<CudaMemoryAllocator>()) {}

template <typename P_in, typename P_out, typename P_compute>
MixedPrecisionHostContext<P_in, P_out, P_compute>::MixedPrecisionHostContext(
    const ContextOption &options,
    std::unique_ptr<MemoryAllocatorInterface> memory_allocator)
    : HostContext(options),
      memory_allocator_(std::move(memory_allocator)),
      a_(options.dim_size_m, options.dim_size_k, memory_allocator_.get()),
      b_(options.dim_size_k, options.dim_size_n, memory_allocator_.get()) {
  a_.Initialize(options.rng, /*scale=*/1e30, options.gaussian);
  b_.Initialize(options.rng, /*scale=*/1e-30, options.gaussian);
}

template <typename P_in, typename P_out, typename P_compute>
std::unique_ptr<GpuContext>
MixedPrecisionHostContext<P_in, P_out, P_compute>::CreateGpuContext(
    int gpu_num) {
  std::unique_ptr<GpuComputationInterface> compute_interface =
      SelectGemmInterface(options_, GetComputeCapability());
  if (compute_interface == nullptr) return nullptr;
  return absl::make_unique<MixedPrecisionGpuContext<P_in, P_out, P_compute>>(
      this, &a_, &b_, gpu_num, std::move(compute_interface));
}

template <typename Compute_Type>
constexpr std::array<Compute_Type, 2> CreateScalingFactors() {
  return {{1.0, 0.0}};
}

template <>
constexpr std::array<int32_t, 2> CreateScalingFactors<int32_t>() {
  return {{1, 0}};
}

template <>
std::array<half_float::half, 2> CreateScalingFactors<half_float::half>() {
  std::array<half_float::half, 2> scaling_factors;
  scaling_factors[0] =
      half_float::detail::float2half<std::round_to_nearest>(1.);
  scaling_factors[1] =
      half_float::detail::float2half<std::round_to_nearest>(0.);
  return scaling_factors;
}

#if CUDA_VERSION >= BF16_CUDA_VERSION
template <>
std::array<nv_bfloat16, 2> CreateScalingFactors<nv_bfloat16>() {
  std::array<nv_bfloat16, 2> scaling_factors;
  scaling_factors[0] = __float2bfloat16(1.0);
  scaling_factors[1] = __float2bfloat16(0.0);
  return scaling_factors;
}
#endif  // CUDA_VERSION >= BF16_CUDA_VERSION

template <typename Compute_Type>
void CopyScalingFactorsToDevice(Compute_Type **alpha, Compute_Type **beta,
                                const cudaStream_t &stream) {
  static std::array<Compute_Type, 2> constants =
      CreateScalingFactors<Compute_Type>();
  CUDA_CHECK(cudaMalloc(alpha, sizeof(constants)));
  *beta = reinterpret_cast<Compute_Type *>(*alpha) + 1;
  CUDA_CHECK(cudaMemcpyAsync(static_cast<void *>(*alpha), constants.data(),
                             sizeof(constants), cudaMemcpyHostToDevice,
                             stream));
}

template <typename InputPrecision, typename OutputPrecision,
          typename ComputePrecision>
GpuDataHandler<InputPrecision, OutputPrecision,
               ComputePrecision>::~GpuDataHandler() {
  // cudaFree is safe to call on already freed or unallocated memory, so no
  // allocation tracking logic is needed.
  cudaFree(input_a_);
  cudaFree(input_b_);
  cudaFree(output_);
  // Alpha and beta were allocated as a size 2 array from alpha.
  cudaFree(alpha_);
}

template <typename InputPrecision, typename OutputPrecision,
          typename ComputePrecision>
void GpuDataHandler<InputPrecision, OutputPrecision, ComputePrecision>::
    Initialize(const RandomMatrix<InputPrecision> *data_in_a,
               const RandomMatrix<InputPrecision> *data_in_b,
               const cudaStream_t stream) {
  CopyScalingFactorsToDevice<ComputePrecision>(&alpha_, &beta_, stream);
  const size_t num_bytes_a = data_in_a->SizeInBytes();
  const size_t num_bytes_b = data_in_b->SizeInBytes();
  CUDA_CHECK(cudaMalloc(&input_a_, num_bytes_a));
  CUDA_CHECK(cudaMalloc(&input_b_, num_bytes_b));
  const size_t n_bytes_out = data_in_a->GetDimSizeM() *
                             data_in_b->GetDimSizeK() * sizeof(OutputPrecision);
  CUDA_CHECK(cudaMalloc(&output_, n_bytes_out));

  CUDA_CHECK(cudaMemcpyAsync(input_a_, data_in_a->Get(), num_bytes_a,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(input_b_, data_in_b->Get(), num_bytes_b,
                             cudaMemcpyHostToDevice, stream));
}

// Caller of this function has the ownership for pointers matrix_a_p, and
// matrix_a_p. Caller should make sure these two pointers are valid until
// cudaMemcpyAsync() complete (received callback).
template <typename P_in, typename P_out, typename P_compute>
MixedPrecisionGpuContext<P_in, P_out, P_compute>::MixedPrecisionGpuContext(
    HostContext *h, RandomMatrix<P_in> const *const matrix_a_p,
    RandomMatrix<P_in> const *const matrix_b_p, int gpu_num,
    std::unique_ptr<GpuComputationInterface> compute_interface)
    : GpuContext(h->GetOption(), gpu_num),
      compute_interface_(std::move(compute_interface)) {
  WithCUDADevice device(gpu_num_);
  cudaDeviceProp dev_prop;
  CUDA_CHECK(cudaGetDeviceProperties(&dev_prop, gpu_num_));
  LOG(INFO) << absl::StrFormat(
      "Running GEMM on a Cuda device with the following properties:\n"
      "Device ID = %d\t PCI Bus-Id = %04x:%02x:%02x\t UUID = %s\t Model = %s",
      gpu_num_, dev_prop.pciDomainID, dev_prop.pciBusID, dev_prop.pciDeviceID,
      absl::BytesToHexString(dev_prop.uuid.bytes), dev_prop.name);
  CUDA_CHECK(cudaStreamCreate(&stream_));
  compute_interface_->Initialize(stream_, options_);

  data_handler_.SetGpuId(gpu_num_);
  data_handler_.Initialize(matrix_a_p, matrix_b_p, stream_);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

template <typename P_in, typename P_out, typename P_compute>
MixedPrecisionGpuContext<P_in, P_out, P_compute>::~MixedPrecisionGpuContext() {
  CUDA_CHECK(cudaStreamDestroy(stream_));
}

template <typename P_in, typename P_out, typename P_compute>
void MixedPrecisionGpuContext<P_in, P_out, P_compute>::StreamSynchronize() {
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

template <typename P_in, typename P_out, typename P_compute>
cudaError_t MixedPrecisionGpuContext<P_in, P_out, P_compute>::StreamQuery() {
  return cudaStreamQuery(stream_);
}

template <typename P_in, typename P_out, typename P_compute>
void MixedPrecisionGpuContext<P_in, P_out, P_compute>::LaunchKernel() {
  WithCUDADevice device(gpu_num_);

  CUBLAS_CHECK(compute_interface_->MatrixMultiComputation(
      options_, data_handler_.Alpha(), data_handler_.InputA(),
      data_handler_.InputB(), data_handler_.Beta(), data_handler_.Output()));
}

float median(std::vector<float> times) {
  const size_t size = times.size();
  if (size == 0) {
    return 0;
  }
  std::sort(times.begin(), times.end());

  const size_t mid = size / 2;
  if (size % 2 == 0) {
    return (times[mid] + times[mid - 1]) / 2;
  } else {
    return times[mid];
  }
}

#if CUDA_VERSION >= 10010
void PrintAlgoInfo(const cublasLtMatmulAlgo_t &algo) {
  int algoId, tile, swizzle, customOption, numSplitsK, reductionScheme;

  CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), NULL));
  CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL));
  CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK),
      NULL));
  CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme,
      sizeof(reductionScheme), NULL));
  CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle),
      NULL));
  CUBLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption,
      sizeof(customOption), NULL));

  LOG(INFO) << absl::StreamFormat(
      "algo={ Id=%d, tileIdx=%d splitK=%d reduction=%d swizzle=%d custom=%d }",
      algoId, tile, numSplitsK, reductionScheme, swizzle, customOption);
}
#endif  // CUDA_VERSION >= 10010

template <typename P_in, typename P_out, typename P_compute>
void MixedPrecisionGpuContext<P_in, P_out, P_compute>::AutoTuning() {
#if CUDA_VERSION >= 10010
  static constexpr int kRepeatAlgoCount = 5;
  static constexpr int kMaxResultNum = 16;
  CudaCublasLtInterface *pLasLt =
      dynamic_cast<CudaCublasLtInterface *>(compute_interface_.get());
  if (pLasLt == nullptr) return;

  WithCUDADevice device(gpu_num_);

  std::vector<cublasLtMatmulHeuristicResult_t> candidates =
      pLasLt->GetHeuristicResults(kMaxResultNum);
  int bestIndex;
  float bestArgoTime, time;
  std::vector<float> algoTimes(kRepeatAlgoCount);

  LOG(INFO) << absl::StreamFormat("Gpu %d Auto Tune Start ...", gpu_num_);

  cudaEvent_t startEvent, stopEvent;
  CUDA_CHECK(cudaEventCreate(&startEvent));
  CUDA_CHECK(cudaEventCreate(&stopEvent));

  ContextOption opt = options_;
  for (size_t candidate_index = 0; candidate_index < candidates.size();
       candidate_index++) {
    opt.algo = candidates[candidate_index].algo;
    for (int repeat_index = 0; repeat_index < kRepeatAlgoCount;
         repeat_index++) {
      CUDA_CHECK(cudaEventRecord(startEvent, stream_));
      CUBLAS_CHECK(compute_interface_->MatrixMultiComputation(
          opt, data_handler_.Alpha(), data_handler_.InputA(),
          data_handler_.InputB(), data_handler_.Beta(),
          data_handler_.Output()));
      CUDA_CHECK(cudaEventRecord(stopEvent, stream_));
      CUDA_CHECK(cudaEventSynchronize(stopEvent));
      CUDA_CHECK(cudaEventElapsedTime(&time, startEvent, stopEvent));
      algoTimes[repeat_index] = time;
    }

    time = median(algoTimes);
    if (candidate_index == 0 || time < bestArgoTime) {
      bestArgoTime = time;
      bestIndex = candidate_index;
    }
  }
  options_.algo = candidates[bestIndex].algo;
  LOG(INFO) << absl::StreamFormat("Gpu %d Auto Tune Finished", gpu_num_);
  PrintAlgoInfo(candidates[bestIndex].algo);

#endif  // CUDA_VERSION >= 10010
}

template class LegacyCudaCublasInterface<float>;
template class LegacyCudaCublasInterface<double>;

template class GpuDataHandler<half_float::half, half_float::half,
                              half_float::half>;
template class GpuDataHandler<half_float::half, float, float>;
template class GpuDataHandler<float, float, float>;
template class GpuDataHandler<double, double, double>;
template class GpuDataHandler<int8_t, int32_t, int32_t>;
template class GpuDataHandler<int8_t, float, float>;

template class MixedPrecisionHostContext<half_float::half, half_float::half,
                                         half_float::half>;
template class MixedPrecisionHostContext<half_float::half, float, float>;
template class MixedPrecisionHostContext<float, float, float>;
template class MixedPrecisionHostContext<double, double, double>;
template class MixedPrecisionHostContext<int8_t, int32_t, int32_t>;
template class MixedPrecisionHostContext<int8_t, float, float>;

#if CUDA_VERSION >= BF16_CUDA_VERSION
template class GpuDataHandler<nv_bfloat16, nv_bfloat16, nv_bfloat16>;
template class MixedPrecisionHostContext<nv_bfloat16, nv_bfloat16, nv_bfloat16>;
#endif

}  // namespace internal
}  // namespace gemm_test
}  // namespace platforms_gpus
