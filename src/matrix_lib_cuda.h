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

#ifndef THIRD_PARTY_GPU_TEST_UTILS_MATRIX_LIB_CUDA_H_
#define THIRD_PARTY_GPU_TEST_UTILS_MATRIX_LIB_CUDA_H_

#include "src/matrix_lib.h"
#include "cuda/include/cuda.h"

// BF16 requires CUDA 11.0
#define BF16_CUDA_VERSION 11000

#if CUDA_VERSION >= BF16_CUDA_VERSION
#include "cuda/include/cuda_bf16.h"

extern template class RandomMatrix<nv_bfloat16>;
#endif

#endif  // THIRD_PARTY_GPU_TEST_UTILS_MATRIX_LIB_CUDA_H_
