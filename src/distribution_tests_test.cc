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

// Tests for the statistical distribution tests.

#include "src/distribution_tests.h"

#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "absl/memory/memory.h"

namespace platforms_gpus {
namespace {

class DistributionTestsTest : public testing::Test {
 protected:
  // Data from D'Agostino & Stephens, Goodness-of-Fit Techniques, p. 546
  // Birth weight of 20 chicks, reckoned to be roughly normal.
  void LoadChickData() {
    data_ = {156, 162, 168, 182, 186, 190, 190, 196, 202, 210,
             214, 220, 226, 230, 230, 236, 236, 242, 246, 270};
  }
  std::vector<double> data_;
};

TEST_F(DistributionTestsTest, HotChicksAndersonDarling) {
  LoadChickData();
  double result;
  ASSERT_TRUE(DistributionTests::TestStatistic(
      data_, DistributionTests::TestType::ANDERSON_DARLING, &result));
  // Correct value (to 3SF) from Goodness-of-Fit Techniques, p. 125
  EXPECT_NEAR(0.223, result, 0.0005);
}

TEST_F(DistributionTestsTest, HotChicksCramerVonMises) {
  LoadChickData();
  double result;
  ASSERT_TRUE(DistributionTests::TestStatistic(
      data_, DistributionTests::TestType::CRAMER_VON_MISES, &result));
  // Correct value (to 3SF) from Goodness-of-Fit Techniques, p. 125
  EXPECT_NEAR(0.035, result, 0.0005);
}

TEST_F(DistributionTestsTest, NoSamples) {
  double result;
  ASSERT_FALSE(DistributionTests::TestStatistic(
      {}, DistributionTests::TestType::CRAMER_VON_MISES, &result));
}

TEST_F(DistributionTestsTest, NoVariance) {
  double result;
  data_ = {5, 5, 5, 5};
  ASSERT_FALSE(DistributionTests::TestStatistic(
      data_, DistributionTests::TestType::CRAMER_VON_MISES, &result));
}

TEST_F(DistributionTestsTest, BadMetric) {
  LoadChickData();
  double result;
  ASSERT_FALSE(DistributionTests::TestStatistic(
      data_, DistributionTests::TestType(3), &result));
}

}  // namespace
}  // namespace platforms_gpus
