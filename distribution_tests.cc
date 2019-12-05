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

#include "distribution_tests.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <vector>

#include "glog/logging.h"
#include "absl/types/span.h"

namespace platforms_gpus {

namespace {

double StandardDeviation(absl::Span<const double> samples) {
  const size_t size = samples.size();
  const double mean =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  const double variance = std::accumulate(
      samples.begin(), samples.end(), 0.0,
      [mean, size](const double current, const double next) {
        return current + (next - mean) * (next - mean) / (size - 1);
      });
  return std::sqrt(variance);
}

const double kRecipSqrt2 = ::pow(2, -0.5);
}  // namespace

bool DistributionTests::TestStatistic(const std::vector<double>& samples,
                                      TestType type, double* result) {
  CHECK(result != nullptr);

  if (samples.empty()) {
    VLOG(3) << "Distribution test failed: no samples.";
    return false;
  }

  const int n = samples.size();
  const double mu = std::accumulate(samples.begin(), samples.end(), 0.0) / n;
  const double sigma = StandardDeviation(samples);

  if (sigma == 0) {
    VLOG(3) << "Distribution test failed: zero variance.";
    return false;
  }

  long double sum = 0;
  double prev = -std::numeric_limits<double>::infinity();
  // Index from 1 in accordance with textbook convention, since it makes the
  // formulae below easier to check.
  for (int i = 1; i <= samples.size(); ++i) {
    const double x = samples[i - 1];

    CHECK(!std::isnan(x));
    CHECK(!std::isinf(x));

    // Check samples are in ascending order.
    CHECK_GE(x, prev);
    prev = x;

    const long double y = kRecipSqrt2 * (x - mu) / sigma;
    const long double phi = 0.5 + 0.5 * std::erf(y);

    long double term;
    switch (type) {
      case ANDERSON_DARLING: {
        long double psi = 0.5 * std::erfc(y);
        int a = 2 * i - 1;
        term = a * std::log(phi) + (2 * n - a) * std::log(psi);
        // We could remove the factors of 0.5 and do
        // term = a lg (phi) + b lg(psi) - 2n lg 2
        // and then eliminate the final term from the loop, but this is
        // considerably less numerically stable.
        break;
      }
      case CRAMER_VON_MISES: {
        long double a = (2 * i - 1) / static_cast<long double>(2 * n);
        term = (a - phi) * (a - phi);
        break;
      }
      default:
        LOG(FATAL) << "Unknown test type " << type;
        return false;
    }

    sum += term;
  }

  // Apply modifications.
  switch (type) {
    case ANDERSON_DARLING: {
      double a_sq = -n - sum / n;
      *result = a_sq * (1 + 0.75 / n + 2.25 / (n * n));
      break;
    }
    case CRAMER_VON_MISES: {
      double w_sq = sum + 1.0 / (12 * n);
      *result = w_sq * (1 + 0.5 / n);
      break;
    }
  }
  return true;
}

}  // namespace platforms_gpus
