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

#include "src/matrix_lib_cuda.h"

#include "src/matrix_lib_impl.h"
#include "cuda/include/cuda.h"

#if CUDA_VERSION >= BF16_CUDA_VERSION
#include "cuda/include/cuda_bf16.h"
#endif

#if CUDA_VERSION >= BF16_CUDA_VERSION
template <>
bool matrix_lib::internal::FillArray<nv_bfloat16>(nv_bfloat16 *A, int n,
                                                  absl::BitGen *rng,
                                                  float scale, bool nv_gauss) {
  auto baseMatrix = std::make_unique<float[]>(kBaseMatrixSize);
  if (nv_gauss) {
    FillGaussian<float>(absl::Span<float>(baseMatrix.get(), kBaseMatrixSize),
                        rng);
  } else {
    FillUniform<float>(absl::Span<float>(baseMatrix.get(), kBaseMatrixSize),
                       rng, scale);
  }
  for (int i = 0; i < n; i++) {
    A[i] = __float2bfloat16(baseMatrix[i % kBaseMatrixSize]);
  }
  return true;
}

template class RandomMatrix<nv_bfloat16>;

#endif  // CUDA_VERSION >= BF16_CUDA_VERSION
