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

#include "matrix_lib.h"

#include <ieee754.h>
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>

#include "glog/logging.h"
#include "absl/base/casts.h"
#include "absl/flags/flag.h"
#include "absl/random/random.h"
#include "include/half.hpp"

using half_float::half;

ABSL_FLAG(int32_t, mantissa_random_bits, -1,
          "Number of random bits in mantissa, or -1 for all bits random.");
ABSL_FLAG(double, probability_of_negative, 0.5,
    "Fill the input matrices with negative values according to this "
    "probability.  Power consumption is highly sensitive to this value, "
    "and it can be tuned to avoid hitting GPU throttle limits.  Note: "
    "if probability_of_negative == p then inversions occur in hardware with "
    "frequency 2*p*(1-p) (i.e. when two signs don't match).");

namespace {

constexpr double kTwoPi = 2 * M_PI;
constexpr int baseMatrixDimm = 8192 * 8192;  // 8k * 8k matrix

// Generate two independent Gaussian floats based on Box-Muller algorithm,
// save them to result_0 and result_1.
void GenerateGaussianRandomFloats(absl::BitGen *rng, float *result_0,
                                  float *result_1) {
  float r1 = absl::Uniform<float>(*rng, 0, 1);  // r1 in range [0.0, 1.0)
  float r2 = absl::Uniform<float>(*rng, 0, 1);  // r2 in range [0.0, 1.0)
  float r3 = sqrt(-10.0 * std::log(r1));
  float r4 = kTwoPi * r2;

  // Box-Muller algorithm to convert data from Uniform distribution to
  // standard Gaussian distribution
  *result_0 = static_cast<float>(r3 * std::cos(r4));
  *result_1 = static_cast<float>(r3 * std::sin(r4));
}

}  // namespace

// We fill the matrices with random numbers.
//
// For highest power consumption we ultimately want to flip all bits in the
// datapath (mantissa and exponent) as frequently as possible.  We generate a
// mantissa via random values in the range [0., 1.) but for simplicity we
// select a fixed exponent k (written as "scale" below).  We have to avoid
// overflow/underflow so what we do for matrix A is generate k*[0., 1.) and for
// matrix B we compute (1/k)*[0., 1.).  We select k so that the sum of N such
// random variables does not exceed the floating point representation.
//
// We could probably do better than this if we knew exactly which order the
// matrix elements went through the MAC units, but that's not something we
// want to encode in a test at this level.
//
// The sign bits have a serious effect on power.  We let this be set from
// the command-line so that we can fine-tune the power consumed.
//
// Note: Over time, we've noticed that the time it takes to fill in arrays with
// random data has taken longer and longer. So, this code will now generate
// random data in a temporary array of size baseMatrixDimm and then fill the
// target array with multiple copies of that temp data.
template <class T>
void FillArray(T *A, int n, absl::BitGen *rng, float scale, bool nv_gauss);

template <>
void FillArray<double>(double *A, int n, absl::BitGen *rng, float scale,
                       bool nv_gauss) {
  std::unique_ptr<double[]> baseMatrixA(new double[baseMatrixDimm]);
  int fillInCount = std::min(baseMatrixDimm, n);
  if (nv_gauss) {
    for (int j = 0; j < fillInCount; j++) {
      double r1 = absl::Uniform<double>(*rng, 0, 1);
      double r2 = absl::Uniform<double>(*rng, 0, 1);
      baseMatrixA[j] = sqrt(-10.0 * log(r1)) * cos(kTwoPi * r2);
    }
  } else {
    // doubles have 52 bits in the mantissa (plus implicit 1 bit)
    int mantissa_random_bits = absl::GetFlag(FLAGS_mantissa_random_bits);
    if (mantissa_random_bits < 0 || mantissa_random_bits > 52) {
      mantissa_random_bits = 52;
    }
    uint64_t mask = ((static_cast<uint64_t>(1) << mantissa_random_bits) - 1)
        << (52 - mantissa_random_bits);
    float probability_of_negative =
        absl::GetFlag(FLAGS_probability_of_negative);
    for (int j = 0; j < fillInCount; j++) {
      // stolen from RandomBase::RandDouble()
      uint64_t mantissa = absl::Uniform<uint64_t>(*rng) & mask;
      union ieee754_double v;
      v.ieee.mantissa1 = mantissa;        // lower 32 bits
      v.ieee.mantissa0 = mantissa >> 32;  // 20 bits
      v.ieee.negative = 0;
      // Exponent is 11 bits wide, using an excess 1023 representation.
      v.ieee.exponent = 1023;
      double temp = v.d - static_cast<double>(1.0);
      // sign could be set in v.ieee.negative, but we need to conditionally
      // add/subtract for the normalization
      if (absl::Uniform<float>(*rng, 0, 1) < probability_of_negative) {
        temp = -temp;
      }
      baseMatrixA[j] = temp * scale;
    }
  }
  for (int i = 0; i < n; i++) {
    A[i] = baseMatrixA[i % baseMatrixDimm];
  }
}

template <>
void FillArray<float>(float *A, int n, absl::BitGen *rng, float scale,
                      bool nv_gauss) {
  std::unique_ptr<float[]> baseMatrixA(new float[baseMatrixDimm]);
  int fillInCount = std::min(baseMatrixDimm, n);
  if (nv_gauss) {
    for (int j = 0; j < fillInCount; j++) {
      float r1 = absl::Uniform<float>(*rng, 0, 1);
      float r2 = absl::Uniform<float>(*rng, 0, 1);
      baseMatrixA[j] =
          static_cast<float>(sqrt(-10.0 * std::log(r1)) * cos(kTwoPi * r2));
    }
  } else {
    // floats have 23 bits in the mantissa (plus implicit 1 bit)
    int mantissa_random_bits = absl::GetFlag(FLAGS_mantissa_random_bits);
    if (mantissa_random_bits < 0 || mantissa_random_bits > 23) {
      mantissa_random_bits = 23;
    }
    uint32_t mask = ((static_cast<uint64_t>(1) << mantissa_random_bits) - 1)
        << (23 - mantissa_random_bits);
    float probability_of_negative =
        absl::GetFlag(FLAGS_probability_of_negative);
    for (int j = 0; j < fillInCount; j++) {
      // stolen from RandomBase::RandFloat()
      uint32_t mantissa = absl::Uniform<uint32_t>(*rng) & mask;
      union ieee754_float v;
      v.ieee.mantissa = mantissa;
      v.ieee.negative = 0;
      // Exponent is 8 bits wide, using an excess 127 exponent representation.
      v.ieee.exponent = 127;
      float temp = v.f - static_cast<float>(1.0);
      // sign could be set in v.ieee.negative, but we need to conditionally
      // add/subtract for the normalization
      if (absl::Uniform<float>(*rng, 0, 1) < probability_of_negative) {
        temp = -temp;
      }
      baseMatrixA[j] = temp * scale;
    }
  }
  for (int i = 0; i < n; i++) {
    A[i] = baseMatrixA[i % baseMatrixDimm];
  }
}

// half precision data has 1 bit sign, 5 bits exponent, and 10 bits mantissa
// (plus implicit 1 bit, which is always with the value of 1).
// A half type value is generated in 3 steps:
// 1. Generating a random float type value in the range of [0, 1).
// 2. Scale it by multiplying "scale". Note that "scale" is a passed in
// parameter, caller should make sure it doesn't cause overflow.
// 3. Convert it to half type value.
template <>
void FillArray<half>(half *A, int n, absl::BitGen *rng, float scale,
                     bool nv_gauss) {
  std::unique_ptr<half[]> baseMatrixA(new half[baseMatrixDimm]);
  int fillInCount = std::min(baseMatrixDimm, n);
  if (nv_gauss) {
    float result_0;
    float result_1;

    for (int j = 0; j < fillInCount; j += 2) {
      GenerateGaussianRandomFloats(rng, &result_0, &result_1);

      // Rounding from float to half makes the distribution not perfectly
      // Gaussian.
      baseMatrixA[j] =
          half_float::detail::float2half<std::round_to_nearest>(result_0);
      baseMatrixA[j + 1] =
          half_float::detail::float2half<std::round_to_nearest>(result_1);
    }

    // There might be a single element left.
    if ((n % 2) == 1) {
      GenerateGaussianRandomFloats(rng, &result_0, &result_1);
      baseMatrixA[n - 1] =
          half_float::detail::float2half<std::round_to_nearest>(result_0);
    }
  } else {
    // floats have 23 bits in the mantissa (plus implicit 1 bit)
    int mantissa_random_bits = absl::GetFlag(FLAGS_mantissa_random_bits);
    if (mantissa_random_bits < 0 || mantissa_random_bits > 23) {
      mantissa_random_bits = 23;
    }
    uint32_t mask = ((static_cast<uint64_t>(1) << mantissa_random_bits) - 1)
                  << (23 - mantissa_random_bits);
    auto exp = static_cast<uint32_t>(127);
    float probability_of_negative =
        absl::GetFlag(FLAGS_probability_of_negative);
    for (int j = 0; j < fillInCount; j++) {
      // stolen from RandomBase::RandFloat()
      uint32_t mantissa = absl::Uniform<uint32_t>(*rng) & mask;
      uint32_t val = (exp << 23) | mantissa;
      // v is in the range of [1.0, 2.0), subtract 1.0 to make it in the range
      // of [0.0, 1.0)
      float temp = absl::bit_cast<float>(val) - static_cast<float>(1.0);
      // sign could be set in v.ieee.negative, but we need to conditionally
      // add/subtract for the normalization
      if (absl::Uniform<float>(*rng, 0, 1) < probability_of_negative) {
        temp = -temp;
      }
      baseMatrixA[j] =
          half_float::detail::float2half<std::round_to_nearest>(temp * scale);
    }
  }
  for (int i = 0; i < n; i++) {
    A[i] = baseMatrixA[i % baseMatrixDimm];
  }
}

// TODO: check with Nvidia about their Gaussian generator for INT8.
// This may not be very useful and old implementation is questionable.
template <>
void FillArray<int8_t>(int8_t *A, int n, absl::BitGen *rng, float scale,
                     bool nv_gauss) {
  std::unique_ptr<int8_t[]> baseMatrixA(new int8_t[baseMatrixDimm]);
  int fillInCount = std::min(baseMatrixDimm, n);
  if (nv_gauss) {
    LOG(ERROR) << "Gaussian distribution is not supported for INT8";
  } else {
    absl::BitGen gen;  // Thread-safety not guaranteed
    for (int j = 0; j < fillInCount; j++) {
      int8_t val = absl::Uniform<uint8_t>(gen) >> 1;
      if (absl::Uniform<float>(gen, 0, 1) <
          absl::GetFlag(FLAGS_probability_of_negative)) {
        val = -val;
      }
      baseMatrixA[j] = val;
    }
  }
  for (int i = 0; i < n; i++) {
    A[i] = baseMatrixA[i % baseMatrixDimm];
  }
}

template <class T>
bool RandomMatrix<T>::Initialize(absl::BitGen *rng, float scale,
                                 bool nv_gauss) {
  size_t nr_bytes = dim_size_m_ * dim_size_k_ * sizeof(T);
  host_memory_ = Allocate(nr_bytes);
  if (!host_memory_) {
    LOG(ERROR) << "Allocation failed for " << nr_bytes << " B";
    return false;
  }

  FillArray<T>(host_memory_, dim_size_m_ * dim_size_k_, rng, scale, nv_gauss);

  return true;
}

template <class T>
T *RandomMatrix<T>::Allocate(size_t nr_bytes) {
  internal_allocation_.reset(new T[nr_bytes / sizeof(T)]);
  return internal_allocation_.get();
}

template class RandomMatrix<int8_t>;
template class RandomMatrix<float>;
template class RandomMatrix<double>;
template class RandomMatrix<half>;
