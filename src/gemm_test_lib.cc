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

#include "absl/memory/memory.h"
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

// Currently cublasGemmEx() supports: "int8:int32", "half:half", "half:float",
// "float:float", "double:double", "int8:int32", and "int8:float" precision
// combinations. The caller of this function has the ownership of options,
// it should be kept valid during the construction of HostContext.
std::unique_ptr<HostContext> HostContext::Create(ContextOption *options) {
  ProcessContextOptionPrecision(options);

  if (options->data_type_in == "int8" && options->data_type_out == "int32") {
    return absl::make_unique<internal::MixedPrecisionHostContext<
        int8_t, int32_t>>(*options);
  } else if (options->data_type_in == "int8" &&
             options->data_type_out == "single") {
    return absl::make_unique<
        internal::MixedPrecisionHostContext<int8_t, float>>(*options);
  } else if (options->data_type_in == "half" &&
             options->data_type_out == "half") {
    return absl::make_unique<internal::MixedPrecisionHostContext<
        half_float::half, half_float::half>>(*options);
  } else if (options->data_type_in == "half" &&
             options->data_type_out == "single") {
    return absl::make_unique<
        internal::MixedPrecisionHostContext<half_float::half, float>>(*options);
  } else if (options->data_type_in == "single" &&
             options->data_type_out == "single") {
    return absl::make_unique<internal::MixedPrecisionHostContext<float, float>>(
        *options);
  } else if (options->data_type_in == "double" &&
             options->data_type_out == "double") {
    return absl::make_unique<
        internal::MixedPrecisionHostContext<double, double>>(*options);
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
