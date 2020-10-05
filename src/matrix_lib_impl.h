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

#ifndef THIRD_PARTY_GPU_TEST_UTILS_MATRIX_LIB_IMPL_H_
#define THIRD_PARTY_GPU_TEST_UTILS_MATRIX_LIB_IMPL_H_

#include "src/matrix_lib.h"

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
#include "absl/random/distributions.h"
#include "absl/random/random.h"

namespace matrix_lib {
namespace internal {

// This many random numbers will be initialized and used to fill matrices.
// Matrices with counts larger than this will have duplicated numbers. This
// number is just a convenient heuristic - it takes on the order of 1 second to
// fill 1 million random numbers and is sufficiently large to give a nice
// distribution.
constexpr int kBaseMatrixSize = 1000000;

// Fill in an array with Gaussian random numbers.
template <class DataType>
void FillGaussian(absl::Span<DataType> data, absl::BitGen *rng) {
  static_assert(!std::is_integral_v<DataType>);
  for (DataType &element : data) {
    element = absl::Gaussian<DataType>(*rng);
  }
}

// Fill an array uniformly.
template <class DataType>
void FillUniform(absl::Span<DataType> data, absl::BitGen *rng,
                 const float scale) {
  static const DataType min = std::is_integral_v<DataType>
                                  ? std::numeric_limits<DataType>::min()
                                  : -1 * scale;
  static const DataType max = std::is_integral_v<DataType>
                                  ? std::numeric_limits<DataType>::max()
                                  : scale;

  for (DataType &element : data) {
    element = absl::Uniform<DataType>(*rng, min, max);
  }
}

// We fill the matrices with random numbers.
//
// If nv_gauss is set to true, a normal distribution will be generated. If
// nv_gauss is false, a uniform distribution will be generated. The scale
// parameter is only used if nv_gauss=false, and sets the upper and lower
// range of values to [-scale, scale] .
//
// This function can be used to generate uniform integer distributions. Integers
// will be evenly distributed along the entire numeric range, and the scale
// factor is ignored. Gaussian integer generation is not currently supported.
//
// Note: Over time, we've noticed that the time it takes to fill in arrays with
// random data has taken longer and longer. So, this code will now generate
// random data in a temporary array of size kBaseMatrixSize and then fill the
// target array with multiple copies of that temp data.
template <class T>
bool FillArray(T *A, const int n, absl::BitGen *rng, const float scale,
               const bool nv_gauss) {
  auto baseMatrix = std::make_unique<T[]>(kBaseMatrixSize);
  if (nv_gauss) {
    if constexpr (!std::is_integral_v<T>) {
      FillGaussian(absl::Span<T>(baseMatrix.get(), kBaseMatrixSize), rng);
    } else {
      LOG(ERROR)
          << "Gaussian distributions are unsupported for integral types.";
      return false;
    }
  } else {
    FillUniform(absl::Span<T>(baseMatrix.get(), kBaseMatrixSize), rng, scale);
  }
  for (int i = 0; i < n; i++) {
    A[i] = baseMatrix[i % kBaseMatrixSize];
  }
  return true;
}

}  // namespace internal
}  // namespace matrix_lib

template <class T>
bool RandomMatrix<T>::Initialize(absl::BitGen *rng, float scale,
                                 bool nv_gauss) {
  size_t nr_bytes = dim_size_m_ * dim_size_k_ * sizeof(T);
  host_memory_ = Allocate(nr_bytes);
  if (!host_memory_) {
    LOG(ERROR) << "Allocation failed for " << nr_bytes << " B";
    return false;
  }

  return ::matrix_lib::internal::FillArray<T>(
      host_memory_, dim_size_m_ * dim_size_k_, rng, scale, nv_gauss);
}

template <class T>
T *RandomMatrix<T>::Allocate(size_t nr_bytes) {
  internal_allocation_.reset(new T[nr_bytes / sizeof(T)]);
  return internal_allocation_.get();
}

#endif  // THIRD_PARTY_GPU_TEST_UTILS_MATRIX_LIB_IMPL_H_
