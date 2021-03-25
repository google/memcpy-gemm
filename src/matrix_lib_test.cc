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

#include "src/matrix_lib.h"

#ifdef RUN_CUDA_TESTS
#include "src/matrix_lib_cuda.h"
#endif  // RUN_CUDA_TESTS

#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/random.h"
#include "src/distribution_tests.h"
#include "include/half.hpp"

#ifdef RUN_CUDA_TESTS

#include "cuda/include/cuda.h"

#if CUDA_VERSION >= BF16_CUDA_VERSION
#include "cuda/include/cuda_bf16.h"
#define HAS_BF16_CUDA
#endif

#endif  // RUN_CUDA_TESTS

namespace {

using half_float::half;

// Converts types to double. For the case of integers,
// converts to a double within the range of -1, 1, scaled
// to the limits of the type.
template <typename T>
double ConvertToDouble(T data) {
  return static_cast<double>(data);
}

#ifdef HAS_BF16_CUDA
// Bfloat16 does not have an explicit cast to double, so we must convert
// through an intermediate.
template <>
double ConvertToDouble(nv_bfloat16 data) {
  return static_cast<double>(__bfloat162float(data));
}
#endif  // HAS_BF16_CUDA

// Converts an integer type to double, and scales it down from its original
// numeric limits to the range of [-1, 1]
template <typename T, typename = std::enable_if_t<std::is_integral_v<T> &&
                                                  std::is_signed_v<T>>>
double IntTypeToBoundDouble(const T input) {
  // This should bound correctly for 2's complement integers
  static constexpr double scale_factor =
      -1 * static_cast<double>(std::numeric_limits<T>::min());
  return static_cast<double>(input) / scale_factor;
}

// Calculate A^2 value defined in ANDERSON_DARLING method, the value is saved
// to result.
template <typename T>
bool TestNormalDistribution(const RandomMatrix<T> &matrix, double *result) {
  std::vector<double> samples;

  const int count = matrix.GetDimSizeK() * matrix.GetDimSizeM();
  const T *data_p = matrix.Get();
  // convert matric into a vector with "double" data type
  for (int j = 0; j < count; ++j, ++data_p) {
    samples.push_back(ConvertToDouble<T>(*data_p));
  }
  // sort data before pass it into DistributionTests::TestStatistic
  std::sort(samples.begin(), samples.end());
  return platforms_gpus::DistributionTests::TestStatistic(
      samples, platforms_gpus::DistributionTests::TestType::ANDERSON_DARLING,
      result);
}

// Determine whether the values in matrix follow Uniform distribution.
// Elements in matrix are assumed to be in (-1, 1) range, which is true in our
// case if setting the scale parameter to be 1.0 when initializing the matrix.
template <typename T>
bool TestUniformDistribution(const RandomMatrix<T> &matrix,
                             double chi_squared_threshold) {
  auto bucket_size = static_cast<int64_t>((1.0 - (-1.0)) / 0.1);
  std::vector<int> bucket(bucket_size, 0);
  const int count = matrix.GetDimSizeK() * matrix.GetDimSizeM();
  const T *data_p = matrix.Get();

  // convert matric into a vector with "double" data type
  for (int j = 0; j < count; ++j, ++data_p) {
    double cur_data;
    if constexpr (std::is_integral_v<T>) {
      cur_data = IntTypeToBoundDouble(*data_p);
    } else {
      cur_data = ConvertToDouble<T>(*data_p);
    }
    EXPECT_GE(1.0, cur_data);
    EXPECT_LE(-1.0, cur_data);

    auto cur_bucket = static_cast<int>((cur_data - (-1.0)) / 0.1);
    // The bounds on random floats should preclude having exactly -1.0, but
    // conversion imprecision involving bf16 can lead to such a value, so we'll
    // put it in the final bucket. Note that if something goes catastrophically
    // wrong and a number of < -1.0 values are generated, it will fail the
    // distribution test, so won't go unnoticed.
    if (cur_bucket == bucket_size) {
      cur_bucket--;
    }
    bucket[cur_bucket]++;
  }

  const double expected = count / bucket_size;
  double chi_squared = 0.0;
  for (int value : bucket) {
    chi_squared += (value - expected) * (value - expected) / expected;
  }

  return chi_squared <= chi_squared_threshold;
}

// Allocations always fail, allowing testing of failure path.
template <typename T>
class FakeAllocationMatrix : public RandomMatrix<T> {
 public:
  FakeAllocationMatrix() : RandomMatrix<T>(1024, 1024) {}

 protected:
  double *Allocate(size_t nr_bytes) override { return nullptr; }

 private:
  std::unique_ptr<double[]> internal_allocation_;
};

TEST(MatrixTests, MatrixAllocationFailureHandled) {
  FakeAllocationMatrix<double> matrix;
  absl::BitGen rng;
  EXPECT_FALSE(matrix.Initialize(&rng, 1, false));
}

template <class T>
class MatrixTest : public testing::Test {
 protected:
  MatrixTest() {}

  ~MatrixTest() override {}
};

#ifdef HAS_BF16_CUDA
using MyTypes = ::testing::Types<half, float, double, nv_bfloat16>;
#else
using MyTypes = ::testing::Types<half, float, double>;
#endif  // CUDA_VERSION >= BF16_CUDA_VERSION

TYPED_TEST_SUITE(MatrixTest, MyTypes);

// This test verifies that MatrixLib constructs matrix properly.
TYPED_TEST(MatrixTest, MatrixConstruction) {
  int m = 100;
  int k = 80;
  // Construct a 100 by 80 matrix
  RandomMatrix<TypeParam> test_matrix(m, k);
  EXPECT_EQ(nullptr, test_matrix.Get());
  EXPECT_EQ(m, test_matrix.GetDimSizeM());
  EXPECT_EQ(k, test_matrix.GetDimSizeK());
  EXPECT_EQ(m * k * sizeof(TypeParam), test_matrix.SizeInBytes());
}

// This test verifies that MatrixLib generates identical data with the same
// seed.
TYPED_TEST(MatrixTest, MatrixPseudoRandomDataGeneration) {
  int m = 10;
  int k = 8;
  absl::BitGen rng0(std::seed_seq{0});
  absl::BitGen rng1(std::seed_seq{0});

  // Construct two m by k matrices with double data type, and the same seed.
  RandomMatrix<TypeParam> test_matrix_0(m, k);
  RandomMatrix<TypeParam> test_matrix_1(m, k);
  ASSERT_TRUE(test_matrix_0.Initialize(&rng0, 3.0, true));
  ASSERT_TRUE(test_matrix_1.Initialize(&rng1, 3.0, true));
  const TypeParam *data_0_p = test_matrix_0.Get();
  const TypeParam *data_1_p = test_matrix_1.Get();
  for (int j = 0; j < m * k; ++j, ++data_0_p, ++data_1_p) {
    // Expect the random values are the same with the same seed.
    EXPECT_EQ(ConvertToDouble<TypeParam>(*data_0_p),
              ConvertToDouble<TypeParam>(*data_1_p));
  }
}

// This test verifies that MatrixLib generates Gaussian random data properly.
TYPED_TEST(MatrixTest, MatrixGaussianDataGeneration) {
  double result;
  int m = 120;
  int k = 120;
  absl::BitGen rng;
  // Construct a m by k matrix.
  RandomMatrix<TypeParam> test_matrix(m, k);
  // generate random date with Gaussian distribution.
  ASSERT_TRUE(test_matrix.Initialize(&rng, 3.0, true));
  // Returning "true" only means the test had been performed successfully,
  // it doesn't mean the data is with Gaussian distribution.

  EXPECT_TRUE(TestNormalDistribution(test_matrix, &result));
  // Normality is rejected if result exceeds 1.159 at 0.5% significance levels.
  // Reference: https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test
  EXPECT_LT(result, 1.159);
}

// This test verifies that MatrixLib generates uniform random data properly.
TYPED_TEST(MatrixTest, MatrixUniformDataGeneration) {
  int m = 120;
  int k = 120;
  absl::BitGen rng;

  // Construct a m by k matrix.
  RandomMatrix<TypeParam> test_matrix(m, k);
  // Generate random date with uniform distribution.
  ASSERT_TRUE(test_matrix.Initialize(&rng, 1.0, false));
  // Check whether data in the matrix "almost certainly" follow Uniform
  // distribution. The threshold is for p > 0.0001, or one in 10,000, for
  // 4799 degrees of freedom (m * k - 1).
  const double threshold = 5172;
  EXPECT_TRUE(TestUniformDistribution(test_matrix, threshold));
}

// int8 does not support Gaussian filling. Rather than have 2
// typed test suites to differentiate, we'll run those tests manually.
TEST(MatrixInt8Test, UniformFill) {
  int m = 60;
  int k = 80;
  absl::BitGen rng;

  RandomMatrix<int8_t> test_matrix(m, k);
  ASSERT_TRUE(test_matrix.Initialize(&rng, 1.0, false));
  // Check whether data in the matrix "almost certainly" follow Uniform
  // distribution. The threshold is for p > 0.0001, or one in 10,000, for
  // 4799 degrees of freedom (m * k - 1).
  const double threshold = 5172;
  EXPECT_TRUE(TestUniformDistribution(test_matrix, threshold));
}

TEST(MatrixInt8Test, FailGaussianFill) {
  int m = 60;
  int k = 80;
  absl::BitGen rng;

  RandomMatrix<int8_t> test_matrix(m, k);
  EXPECT_FALSE(test_matrix.Initialize(&rng, 1.0, true));
}

}  // namespace
