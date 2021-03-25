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

#include "src/gemm_test_lib.h"

#include "glog/logging.h"
#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/substitute.h"
#include "src/gemm_test_lib_internal.h"
#include "src/matrix_lib.h"
#include "include/half.hpp"

namespace {
// If the precision for "data_type_in", "data_type_out" and "compute_type" are
// the same, We allow user to only specify "data_type_in" in ContextOption.
void ProcessContextOptionPrecision(
    platforms_gpus::gemm_test::ContextOption *options) {
  if (options->data_type_out.empty() && options->compute_type.empty()) {
    options->data_type_out = options->data_type_in;
    options->compute_type = options->data_type_in;
  }
}
}  // namespace

namespace platforms_gpus {
namespace gemm_test {

ComputeCapability GetComputeCapability() {
  int major = 0, minor = 0;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
  CUDA_CHECK(
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0));
  return {major, minor};
}

bool GemmPrecisionIsSupported(const ComputeCapability &compute_capability,
                              absl::string_view input_precision,
                              absl::string_view output_precision,
                              absl::string_view compute_precision) {
  static constexpr int kPascalComputeCap = 6;
  static constexpr int kVoltaComputeCap = 7;
  static constexpr int kAmpereComputeCap = 8;

  // First handle simple options that are supported on all hardware.
  if (input_precision == "single" && output_precision == "single" &&
      compute_precision == "single") {
    return true;
  }
  if (input_precision == "double" && output_precision == "double" &&
      compute_precision == "double") {
    return true;
  }

  // K80s don't support half precision or integer options.
  if (compute_capability.major < kPascalComputeCap) {
    return false;
  }
  // Half precision input supports half and single output/compute.
  if (input_precision == "half" && output_precision == "half" &&
      compute_precision == "half") {
    return true;
  }
  if (input_precision == "half" && output_precision == "single" &&
      compute_precision == "single") {
    return true;
  }

  // Pascal does not support integer ops or bf16.
  if (compute_capability.major < kVoltaComputeCap) {
    return false;
  }
  if (input_precision == "int8" && output_precision == "int32" &&
      compute_precision == "int32") {
    return true;
  }
  if (input_precision == "int8" && output_precision == "single" &&
      compute_precision == "single") {
    return true;
  }

  // Ampere and greater support bf16.
  if (compute_capability.major < kAmpereComputeCap) {
    return false;
  }
  if (input_precision == "bf16" &&
      (output_precision == "bf16" || output_precision == "single") &&
      compute_precision == "single") {
    return true;
  }
  if (input_precision == "single" &&
      (output_precision == "single") &&
      compute_precision == "f32_tf32") {
    return true;
  }
  return false;
}

std::vector<int64_t> ParseGpuIDsOrDie(absl::Span<const std::string> gpus) {
  std::vector<int64_t> gpu_ids;
  gpu_ids.reserve(gpus.size());
  for (absl::string_view gpu : gpus) {
    CHECK(absl::SimpleAtoi(gpu, &gpu_ids.emplace_back()))
        << "Failed to parse GPU ID '" << gpu << "' to int.";
  }
  return gpu_ids;
}

// Currently cublasGemmEx() supports: "int8:int32", "half:half", "half:float",
// "float:float", "double:double", "int8:int32", and "int8:float" precision
// combinations. The caller of this function has the ownership of options,
// it should be kept valid during the construction of HostContext.
std::unique_ptr<HostContext> HostContext::Create(ContextOption *options) {
  ProcessContextOptionPrecision(options);

  if (options->data_type_in == "int8" && options->data_type_out == "int32") {
    return absl::make_unique<
        internal::MixedPrecisionHostContext<int8_t, int32_t, int32_t>>(
        *options);
  } else if (options->data_type_in == "int8" &&
             options->data_type_out == "single") {
    return absl::make_unique<
        internal::MixedPrecisionHostContext<int8_t, float, float>>(*options);
  } else if (options->data_type_in == "half" &&
             options->data_type_out == "half") {
    return absl::make_unique<internal::MixedPrecisionHostContext<
        half_float::half, half_float::half, half_float::half>>(*options);
  } else if (options->data_type_in == "half" &&
             options->data_type_out == "single") {
    return absl::make_unique<
        internal::MixedPrecisionHostContext<half_float::half, float, float>>(
        *options);
  } else if (options->data_type_in == "single" &&
             options->data_type_out == "single") {
    return absl::make_unique<
        internal::MixedPrecisionHostContext<float, float, float>>(*options);
  } else if (options->data_type_in == "double" &&
             options->data_type_out == "double") {
    return absl::make_unique<
        internal::MixedPrecisionHostContext<double, double, double>>(*options);
#if CUDA_VERSION >= BF16_CUDA_VERSION
  } else if (options->data_type_in == "bf16" &&
             options->data_type_out == "bf16" &&
             options->compute_type == "single") {
    return absl::make_unique<
        internal::MixedPrecisionHostContext<nv_bfloat16, nv_bfloat16, float>>(
        *options);
  } else if (options->data_type_in == "bf16" &&
             options->data_type_out == "single" &&
             options->compute_type == "single") {
    return absl::make_unique<
        internal::MixedPrecisionHostContext<nv_bfloat16, float, float>>(
        *options);
#endif  // CUDA_VERSION >= BF16_CUDA_VERSION
  }

  LOG(ERROR) << absl::Substitute(
      "Unsupported input/output precision combination of $0:$1",
      options->data_type_in, options->data_type_out);
  return nullptr;
}

std::unique_ptr<GpuContext> GpuContext::Create(HostContext *h, int gpu_num) {
  CHECK(h != nullptr);
  return h->CreateGpuContext(gpu_num);
}

}  // namespace gemm_test
}  // namespace platforms_gpus
