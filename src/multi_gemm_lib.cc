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

#include "src/multi_gemm_lib.h"

template <class T>
T *CudaRandomMatrix<T>::Allocate(size_t nr_bytes) {
  return reinterpret_cast<T *>(memory_allocator_->AllocateHostMemory(nr_bytes));
}

template <class T>
CudaRandomMatrix<T>::~CudaRandomMatrix() {
  if (Base::host_memory_) {
    memory_allocator_->FreeHostMemory(Base::host_memory_);
  }
}

template class CudaRandomMatrix<int8_t>;
template class CudaRandomMatrix<half_float::half>;
template class CudaRandomMatrix<float>;
template class CudaRandomMatrix<double>;

#if CUDA_VERSION >= BF16_CUDA_VERSION
template class CudaRandomMatrix<nv_bfloat16>;
template class CudaRandomMatrix<__nv_fp8_e4m3>;
#endif
