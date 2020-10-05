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

#ifndef PLATFORMS_GPUS_TESTING_NVIDIA_GEMM_TEST_LIB_H_
#define PLATFORMS_GPUS_TESTING_NVIDIA_GEMM_TEST_LIB_H_

#include <stddef.h>

#include <memory>
#include <string>

#define CUDA_NO_HALF

#include "absl/random/random.h"
#include "cuda/include/cublas_v2.h"
#include "cuda/include/cuda.h"
#include "cuda/include/cuda_runtime.h"
#if CUDA_VERSION >= 10010
#include "cuda/include/cublasLt.h"
#endif  // CUDA_VERSION >= 10010

namespace platforms_gpus {
namespace gemm_test {

// Describes the compute capability of a GPU. NVIDIA typically
// reports compute capability as a float, with value {major}.{minor}.
// For example, a Tesla T4 has compute capability 7.5, so major=7, minor=5.
struct ComputeCapability {
  int major = 0;
  int minor = 0;
};

// Detects the compute capability of GPU 0.
ComputeCapability GetComputeCapability();

// Checks that the combination of GEMM input/output/compute precisions is
// supported on the hardware.
bool GemmPrecisionIsSupported(const ComputeCapability &compute_capability,
                              absl::string_view input_precision,
                              absl::string_view output_precision,
                              absl::string_view compute_precision);

// Converts a vector of strings of GPU IDs to a vector of ints. Program
// exits on parse failure.
std::vector<int64_t> ParseGpuIDsOrDie(absl::Span<const std::string> gpus);

class HostContext;

// Configuration options for Hostcontext object and GpuContext objects.
struct ContextOption {
  // Specify the input and output matrices data precision types
  // CUDA supports "half", "single", and "double", which are 16-bit,
  // 32-bit, 64-bit floating point precision respectively. Limited support also
  // exists for integer operations, with data_type_in = "int8" and data_type
  // out = "int32" or float.
  std::string data_type_in;
  std::string data_type_out;

  // The cublasGemmEx computation type. Currently cublasGemmEx supports
  // following types: CUDA_R_16F, CUDA_R_32F, CUDA_R_32I, CUDA_R_64F
  // CUDA_C_32F, CUDA_C_64F
  std::string compute_type;

  // Dimensions of two input matrices, which should be
  // "dim_size_m by dim_size_n", and "dim_size_n by dim_size_k" respectively.
  size_t dim_size_m;
  size_t dim_size_n;
  size_t dim_size_k;

  // Whether matrices A and B are transposed before being multiplied.
  bool transa = false;
  bool transb = false;

  // Whether generate data with Gaussian distribution for input matrices. If
  // it's set to false, will generate uniformly distributed data instead.
  bool gaussian;

  // Pointer to a randomized data generator
  absl::BitGen *rng;

  // The cublasGemmEx computation algorithm, use the default algorithm if use
  // don't specify one.
  // Supported algorithms for CUDA8 are: "gemm_algo_default",
  // "gemm_algo_0", "gemm_algo_1", "gemm_algo_2", "gemm_algo_3", "gemm_algo_4"
  // "gemm_algo_5", "gemm_algo_6", "gemm_algo_7".
  // CUDA9 additionally supports more algorithms:"gemm_algo_8", "gemm_algo_9",
  // "gemm_algo_10", "gemm_algo_11", "gemm_algo_12", "gemm_algo_13",
  // "gemm_algo_14", "gemm_algo_15", "gemm_algo_16", "gemm_algo_17",
  // "gemm_tensor_algo_default", "gemm_tensor_algo_0", "gemm_tensor_algo_1",
  // "gemm_tensor_algo_2", "gemm_tensor_algo_3", "gemm_tensor_algo_4".
  std::string algorithm = "gemm_algo_default";

#if CUDA_VERSION >= 10010
  // algo only applies for cublasLT heuristic algorithm determination
  // AutoTuning will generate this algo value.
  absl::optional<cublasLtMatmulAlgo_t> algo;
#endif  //  CUDA_VERSION >= 10010

  // TODO Currently Just specified, will set cublasLt as default
  bool use_cublasLt_ = false;
};

// The base class for GpuContext objects. It Contains common functions and data
// membersfor all derived classes. Thread-safety was not tested.
class GpuContext {
 public:
  // factory method that creates GpuContext object. It will be created with the
  // same parameters that were passed to the HostContext object.
  static std::unique_ptr<GpuContext> Create(HostContext *h, int gpu_num);

  GpuContext(const ContextOption &options, int gpu_num)
      : options_(options), gpu_num_(gpu_num) {}

  virtual ~GpuContext() {}

  const size_t GetDimSizeK() { return options_.dim_size_k; }
  const size_t GetDimSizeM() { return options_.dim_size_m; }
  const size_t GetDimSizeN() { return options_.dim_size_n; }
  const bool GetTransa() { return options_.transa; }
  const bool GetTransb() { return options_.transb; }
  void ResetLoopCount() { loop_count_ = 0; }
  void IncLoopCount() { ++loop_count_; }
  int GetLoopCount() const { return loop_count_; }
  int GetGpuIndex() const { return gpu_num_; }
  // Block host until the stream has completed all operations.
  virtual void StreamSynchronize() = 0;
  virtual cudaError_t StreamQuery() = 0;

  // Run kernel on GPU.
  virtual void LaunchKernel() = 0;

  virtual void AutoTuning() = 0;

 protected:
  GpuContext() {}

  // TODO: options_ contains redundant info that also saved in
  // HostContext, will consolidate it.
  ContextOption options_;

  // The gpu index number that the GpuContext is created for.
  int gpu_num_;
  // The loop number that a GpuContext runs. We are now running all GPUs
  // independently so we have to store the loop number in each GpuContext
  int loop_count_;
};

// The base class for HostContext objects. It Contains common functions and data
// members for all derived classes.
class HostContext {
 public:
  // factory method to create HostContext object, options is used to specify
  // config parameters for the object.
  static std::unique_ptr<HostContext> Create(ContextOption *options);

  virtual ~HostContext() {}

  const ContextOption GetOption() { return options_; }
  const size_t GetDimSizeK() { return options_.dim_size_k; }
  const size_t GetDimSizeM() { return options_.dim_size_m; }
  const size_t GetDimSizeN() { return options_.dim_size_n; }
  const bool GetTransa() { return options_.transa; }
  const bool GetTransb() { return options_.transb; }
  const std::string GetComputeType() { return options_.compute_type; }
  const std::string GetCublasAlgorithm() { return options_.algorithm; }

 protected:
  explicit HostContext(const ContextOption &options) : options_(options) {}

  virtual std::unique_ptr<GpuContext> CreateGpuContext(int gpu_num) = 0;

  ContextOption options_;

 private:
  // GpuContext::Create invokes HostContext::CreateGpuContext to do the
  // real job to create GpuContext. This way we only need to parse
  // ContextOption once (in HostContext).
  friend std::unique_ptr<GpuContext> GpuContext::Create(HostContext *h,
                                                        int gpu_num);
};

}  // namespace gemm_test
}  // namespace platforms_gpus

#endif  // PLATFORMS_GPUS_TESTING_NVIDIA_GEMM_TEST_LIB_H_
