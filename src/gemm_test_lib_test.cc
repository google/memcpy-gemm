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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/random.h"
#include "src/gemm_test_lib_internal.h"
#include "src/gemm_test_lib_mock.h"
#include "include/half.hpp"

#ifndef BF16_CUDA_VERSION
#DEFINE BF16_CUDA_VERSION 11000
#endif

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::NotNull;

namespace platforms_gpus {
namespace gemm_test {
namespace {

static constexpr ComputeCapability kKeplerComputeCap = {3, 5};
static constexpr ComputeCapability kPascalComputeCap = {6, 0};
static constexpr ComputeCapability kVoltaComputeCap = {7, 0};
static constexpr ComputeCapability kAmpereComputeCap = {8, 0};

TEST(SupportedGemmPrecisionTest, SinglePrecision) {
  EXPECT_TRUE(GemmPrecisionIsSupported(kKeplerComputeCap, "single", "single",
                                       "single"));
}

TEST(SupportedGemmPrecisionTest, DoublePrecision) {
  EXPECT_TRUE(GemmPrecisionIsSupported(kPascalComputeCap, "double", "double",
                                       "double"));
}

TEST(SupportedGemmPrecisionTest, HalfPrecision) {
  EXPECT_TRUE(
      GemmPrecisionIsSupported(kPascalComputeCap, "half", "half", "half"));
}

TEST(SupportedGemmPrecisionTest, MixedPrecision) {
  EXPECT_TRUE(
      GemmPrecisionIsSupported(kPascalComputeCap, "half", "single", "single"));
}

TEST(SupportedGemmPrecisionTest, MixedFailsOnKepler) {
  EXPECT_FALSE(
      GemmPrecisionIsSupported(kKeplerComputeCap, "half", "single", "single"));
}

TEST(SupportedGemmPrecisionTest, Int8SinglePrecision) {
  EXPECT_TRUE(
      GemmPrecisionIsSupported(kVoltaComputeCap, "int8", "single", "single"));
}

TEST(SupportedGemmPrecisionTest, Int8Int32Precision) {
  EXPECT_TRUE(
      GemmPrecisionIsSupported(kVoltaComputeCap, "int8", "int32", "int32"));
}

TEST(SupportedGemmPrecisionTest, Bf16Bf16Precision) {
  EXPECT_TRUE(
      GemmPrecisionIsSupported(kAmpereComputeCap, "bf16", "bf16", "single"));
}
TEST(SupportedGemmPrecisionTest, Bf16SinglePrecision) {
  EXPECT_TRUE(
      GemmPrecisionIsSupported(kAmpereComputeCap, "bf16", "single", "single"));
}
TEST(SupportedGemmPrecisionTest, tf32ComputeSinglePrecision) {
  EXPECT_TRUE(GemmPrecisionIsSupported(kAmpereComputeCap, "single", "single",
                                       "f32_tf32"));
}

TEST(SupportedGemmPrecisionTest, IntFailsOnPascal) {
  EXPECT_FALSE(
      GemmPrecisionIsSupported(kPascalComputeCap, "int8", "int32", "int32"));
}

TEST(SupportedGemmPrecisionTest, Bf16FailsOnVolta) {
  EXPECT_FALSE(
      GemmPrecisionIsSupported(kVoltaComputeCap, "bf16", "bf16", "bf16"));
}

TEST(SupportedGemmPrecisionTest, FailsOnUnsupportedInput) {
  EXPECT_FALSE(
      GemmPrecisionIsSupported(kPascalComputeCap, "half", "single", "int8"));
}

TEST(ParseGpuIDsTest, EmptyInput) {
  std::vector<std::string> input;
  EXPECT_THAT(ParseGpuIDsOrDie(input), IsEmpty());
}

TEST(ParseGpuIDsTest, GoodValues) {
  std::vector<std::string> input = {{"0", "1", "5"}};
  EXPECT_THAT(ParseGpuIDsOrDie(input), ElementsAre(0, 1, 5));
}

TEST(ParseGpuIDsTest, FailureCase) {
  std::vector<std::string> input = {{"0", "1", "abc"}};
  EXPECT_DEATH(ParseGpuIDsOrDie(input), "Failed to parse GPU ID 'abc' to int.");
}

struct HalfInHalfOut {
  using Input = half_float::half;
  using Output = half_float::half;
  using Compute = half_float::half;
};

struct HalfInFloatOut {
  using Input = half_float::half;
  using Output = float;
  using Compute = float;
};

struct FloatInFloatOut {
  using Input = float;
  using Output = float;
  using Compute = float;
};

struct DoubleInDoubleOut {
  using Input = double;
  using Output = double;
  using Compute = double;
};

struct IntInIntOut {
  using Input = int8_t;
  using Output = int32_t;
  using Compute = int32_t;
};

struct IntInFloatOut {
  using Input = int8_t;
  using Output = float;
  using Compute = float;
};

#if CUDA_VERSION >= BF16_CUDA_VERSION
struct Bf16InBf16Out {
  using Input = nv_bfloat16;
  using Output = nv_bfloat16;
  using Compute = nv_bfloat16;
};
#endif  // CUDA_VERSION >= BF16_CUDA_VERSION

// This test verifies factory method of HostContext would fail to creates
// MixedPrecisionHostContext with unsupported in/out precision combination.
TEST(GemmTestLibTest, CreateHostContextFailed) {
  absl::BitGen rng;
  platforms_gpus::gemm_test::ContextOption options;

  options.dim_size_m = 10;
  options.dim_size_n = 20;
  options.dim_size_k = 10;
  options.transa = true;
  options.transb = false;
  options.gaussian = true;
  options.rng = &rng;

  // this is an unsupported in/out precision combination.
  options.data_type_in = "half";
  options.data_type_out = "double";

  auto hostcontext = platforms_gpus::gemm_test::HostContext::Create(&options);

  // With unsupported in/out precision combination, we should return nullptr.
  EXPECT_EQ(hostcontext, nullptr);
}

template <typename T>
class HostContextTest : public ::testing::Test {
 protected:
  HostContextTest() {
    options_.dim_size_m = 10;
    options_.dim_size_n = 20;
    options_.dim_size_k = 10;
    options_.transa = true;
    options_.transb = false;
    options_.gaussian = true;
    options_.algorithm = "gemm_algo_0";
  }

  ~HostContextTest() override {}

  platforms_gpus::gemm_test::ContextOption options_;
};

TYPED_TEST_SUITE_P(HostContextTest);

TYPED_TEST_P(HostContextTest, CreateHostContextSuccess) {
  using InputType = typename TypeParam::Input;
  using OutputType = typename TypeParam::Output;
  using ComputeType = typename TypeParam::Compute;

  auto memory_allocator = absl::make_unique<MockMemoryAllocator>();
  auto cublas = absl::make_unique<internal::CudaCublasInterface>();

  size_t a_size = this->options_.dim_size_m * this->options_.dim_size_k;
  size_t b_size = this->options_.dim_size_k * this->options_.dim_size_n;

  auto matrix_a = absl::make_unique<InputType[]>(a_size);
  auto matrix_b = absl::make_unique<InputType[]>(b_size);

  auto host_context = absl::make_unique<
      internal::MixedPrecisionHostContext<InputType, OutputType, ComputeType>>(
      this->options_, std::move(memory_allocator));

  ASSERT_THAT(host_context, NotNull());
  EXPECT_EQ(this->options_.dim_size_m, host_context->GetDimSizeM());
  EXPECT_EQ(this->options_.dim_size_n, host_context->GetDimSizeN());
  EXPECT_EQ(this->options_.dim_size_k, host_context->GetDimSizeK());
  EXPECT_EQ(this->options_.transa, host_context->GetTransa());
  EXPECT_EQ(this->options_.transb, host_context->GetTransb());
  EXPECT_EQ(this->options_.compute_type, host_context->GetComputeType());
  EXPECT_EQ(this->options_.algorithm, host_context->GetCublasAlgorithm());
};

REGISTER_TYPED_TEST_SUITE_P(HostContextTest, CreateHostContextSuccess);

#if CUDA_VERSION >= BF16_CUDA_VERSION
using MyTypes = ::testing::Types<HalfInHalfOut, HalfInFloatOut, FloatInFloatOut,
                                 DoubleInDoubleOut, IntInIntOut, IntInFloatOut,
                                 Bf16InBf16Out>;
#else
using MyTypes = ::testing::Types<HalfInHalfOut, HalfInFloatOut, FloatInFloatOut,
                                 DoubleInDoubleOut, IntInIntOut, IntInFloatOut>;
#endif  // CUDA_VERSION >= BF16_CUDA_VERSION

INSTANTIATE_TYPED_TEST_SUITE_P(MixPrecisionHostContextTest, HostContextTest,
                               MyTypes);

}  // namespace
}  // namespace gemm_test
}  // namespace platforms_gpus
