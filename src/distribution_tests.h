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

// Class collecting statistical tests which attempt to determine whether data
// are drawn from a given distribution.

#ifndef PLATFORMS_GPUS_TESTING_NVIDIA_DISTRIBUTION_TESTS_H_
#define PLATFORMS_GPUS_TESTING_NVIDIA_DISTRIBUTION_TESTS_H_

#include <string>
#include <vector>

namespace platforms_gpus {

class DistributionTests {
 public:
  enum class TestType {
    // The Anderson-Darling statistic is a robust test of whether a set of
    // values are drawn from a Normal distribution. No particular values of the
    // distribution are assumed.
    // Function returns true if the test is successfully executed, in which
    // case *result will be set the Anderson-Darling test statistic for the
    // provided samples, which will be a non-negative real number. Smaller
    // values indicate greater confidence that the samples came from a Normal
    // distribution.
    // For more information, see
    // http://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test
    ANDERSON_DARLING,

    // The Cramér-von Mises statistic is a more forgiving test of normality
    // than Anderson-Darling, and may be more suitable for real data.
    // For more information, see
    // http://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93von_Mises_criterion
    CRAMER_VON_MISES,
  };

  // Computes the test statistic for the specified normality test.
  // REQUIRES: samples sorted into ascending order; samples does not contain
  //           any NaNs or infinite values.
  static bool TestStatistic(const std::vector<double>& samples, TestType type,
                            double* result);

 private:
  DistributionTests() {}
  ~DistributionTests() {}
};

}  // namespace platforms_gpus

#endif  // PLATFORMS_GPUS_TESTING_NVIDIA_DISTRIBUTION_TESTS_H_
