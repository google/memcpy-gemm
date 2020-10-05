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

#ifndef THIRD_PARTY_GPU_TEST_UTILS_MATRIX_LIB_H_
#define THIRD_PARTY_GPU_TEST_UTILS_MATRIX_LIB_H_

#include <stddef.h>

#include <memory>

#include "absl/random/random.h"
#include "include/half.hpp"

// Represents a MxK matrix of random values.
template <typename T>
class RandomMatrix {
 public:
  RandomMatrix(size_t M, size_t K)
      : dim_size_m_(M), dim_size_k_(K), host_memory_(nullptr) {}

  virtual ~RandomMatrix() {}

  bool Initialize(absl::BitGen *rng, float scale, bool nv_gauss);

  const T *Get() const { return host_memory_; }
  T *Get() { return host_memory_; }
  const size_t GetDimSizeK() const { return dim_size_k_; }
  const size_t GetDimSizeM() const { return dim_size_m_; }

  const size_t SizeInBytes() const {
    return dim_size_k_ * dim_size_m_ * sizeof(T);
  }

 protected:
  // Returns a pointer to at least nr_bytes storage that is addressable by the
  // host.
  virtual T *Allocate(size_t nr_bytes);

  const size_t dim_size_m_;
  const size_t dim_size_k_;

  T *host_memory_;

 private:
  // Used if Allocate is not overloaded.
  std::unique_ptr<T[]> internal_allocation_;
};

extern template class RandomMatrix<int8_t>;
extern template class RandomMatrix<float>;
extern template class RandomMatrix<double>;
extern template class RandomMatrix<half_float::half>;

#endif  // THIRD_PARTY_GPU_TEST_UTILS_MATRIX_LIB_H_
