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

#include "gemm_test_lib.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/random.h"
#include "gemm_test_lib_internal.h"
#include "gemm_test_lib_mock.h"
#include "include/half.hpp"

using testing::NotNull;

namespace platforms_gpus {
namespace gemm_test {
namespace {

struct HalfInHalfOut {
  using Input = half_float::half;
  using Output = half_float::half;
};

struct HalfInFloatOut {
  using Input = half_float::half;
  using Output = float;
};

struct FloatInFloatOut {
  using Input = float;
  using Output = float;
};

struct DoubleInDoubleOut {
  using Input = double;
  using Output = double;
};

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

  auto memory_allocator = absl::make_unique<MockMemoryAllocator>();
  auto cublas =
      absl::make_unique<internal::CudaCublasInterface<InputType, OutputType>>();

  size_t a_size = this->options_.dim_size_m * this->options_.dim_size_k;
  size_t b_size = this->options_.dim_size_k * this->options_.dim_size_n;

  auto matrix_a = absl::make_unique<InputType[]>(a_size);
  auto matrix_b = absl::make_unique<InputType[]>(b_size);

  auto host_context = absl::make_unique<
      internal::MixedPrecisionHostContext<InputType, OutputType>>(
      this->options_, std::move(memory_allocator), std::move(cublas));

  ASSERT_THAT(host_context, NotNull());
  EXPECT_EQ(this->options_.dim_size_m, host_context->GetDimSizeM());
  EXPECT_EQ(this->options_.dim_size_n, host_context->GetDimSizeN());
  EXPECT_EQ(this->options_.dim_size_k, host_context->GetDimSizeK());
  EXPECT_EQ(this->options_.transa, host_context->GetTransa());
  EXPECT_EQ(this->options_.transb, host_context->GetTransb());
  EXPECT_EQ(this->options_.compute_type, host_context->GetComputeType());
  EXPECT_EQ(this->options_.algorithm, host_context->GetCublasAlgorithm());
  EXPECT_EQ(this->options_.algorithm_tc,
      host_context->GetCublasAlgorithmTensorCore());
};

REGISTER_TYPED_TEST_SUITE_P(HostContextTest, CreateHostContextSuccess);

using MyTypes = ::testing::Types<HalfInHalfOut, HalfInFloatOut, FloatInFloatOut,
                                 DoubleInDoubleOut>;

INSTANTIATE_TYPED_TEST_SUITE_P(MixPrecisionHostContextTest, HostContextTest,
                               MyTypes);

}  // namespace
}  // namespace gemm_test
}  // namespace platforms_gpus
