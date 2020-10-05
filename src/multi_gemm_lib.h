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

#ifndef PLATFORMS_GPUS_TESTING_NVIDIA_MULTI_GEMM_LIB_H_
#define PLATFORMS_GPUS_TESTING_NVIDIA_MULTI_GEMM_LIB_H_

#include <stddef.h>

#include <cstdint>

#include "glog/logging.h"
#include "absl/random/random.h"
#include "src/cuda_check.h"
#include "src/matrix_lib.h"
#include "src/matrix_lib_cuda.h"
#include "src/memory_allocator_interface.h"
#include "cuda/include/cublas_v2.h"
#include "cuda/include/cuda_runtime.h"
#include "include/half.hpp"

#if CUDA_VERSION >= BF16_CUDA_VERSION
#include "cuda/include/cuda_bf16.h"
#endif

// The concrete class wraps real memory allocation APIs
class CudaMemoryAllocator final
    : public platforms_gpus::gemm_test::MemoryAllocatorInterface {
 public:
  ~CudaMemoryAllocator() override {}

  void *AllocateHostMemory(size_t nr_bytes) override {
    void *addr = nullptr;
    CUDA_CHECK(cudaMallocHost(&addr, nr_bytes));
    return addr;
  }

  void FreeHostMemory(void *addr) override { CUDA_CHECK(cudaFreeHost(addr)); }
};

// TODO: We can get rid of this class since the only thing it does is
// to invoke MemoryAllocatorInterface functions. This requires modification of
// AMD relate gemm code.
template <class T>
class CudaRandomMatrix : public RandomMatrix<T> {
 protected:
  using Base = RandomMatrix<T>;

 public:
  using Base::Base;

  // Caller of CudaRandomMatrix constructor should keep memory_allocator valid
  // in the lifetime of CudaRandomMatrix instance. It's also caller's
  // responsibility to release the object pointed by memory_allocator.
  CudaRandomMatrix(
      size_t M, size_t K,
      platforms_gpus::gemm_test::MemoryAllocatorInterface *memory_allocator)
      : RandomMatrix<T>(M, K), memory_allocator_(memory_allocator) {}

  ~CudaRandomMatrix() override;

 protected:
  T *Allocate(size_t nr_bytes) override;

  platforms_gpus::gemm_test::MemoryAllocatorInterface *memory_allocator_;
};

extern template class CudaRandomMatrix<int8_t>;
extern template class CudaRandomMatrix<half_float::half>;
extern template class CudaRandomMatrix<float>;
extern template class CudaRandomMatrix<double>;

#if CUDA_VERSION >= BF16_CUDA_VERSION
extern template class CudaRandomMatrix<nv_bfloat16>;
#endif


#endif  // PLATFORMS_GPUS_TESTING_NVIDIA_MULTI_GEMM_LIB_H_
