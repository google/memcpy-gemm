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


#include "multi_gemm_lib.h"

#include <stddef.h>
#include <string.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>

#include "glog/logging.h"
#include "absl/random/random.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "cuda/include/cublas_v2.h"
#include "cuda/include/cuda_runtime.h"

namespace {
void LogVersionsOnce(cublasHandle_t cublas) {
  static std::atomic<bool> done{false};

  // It's OK for other threads to proceed even if we haven't completed the
  // logging yet.
  if (done.exchange(true)) {
    return;
  }

  int rt_version = -1;
  int drv_version = -1;
  int cublas_version = -1;

  CUDA_CHECK(cudaRuntimeGetVersion(&rt_version));
  CUDA_CHECK(cudaDriverGetVersion(&drv_version));
  CUBLAS_CHECK(cublasGetVersion(cublas, &cublas_version));

  LOG(INFO) << "CUDA run-time version: " << rt_version;
  LOG(INFO) << "CUDA driver version: " << drv_version;
  LOG(INFO) << "CUBLAS version: " << cublas_version;
}
}  // namespace

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

template <class T>
HostContext<T>::HostContext(
    size_t M, size_t N, size_t K, absl::BitGen *rng, bool nv_gauss,
    platforms_gpus::gemm_test::MemoryAllocatorInterface *memory_allocator,
    bool transa, bool transb)
    : dim_size_m_(M),
      dim_size_n_(N),
      dim_size_k_(K),
      transa_(transa),
      transb_(transb),
      a_(M, K, memory_allocator),
      b_(K, N, memory_allocator) {
  a_.Initialize(rng, /*scale=*/1e30, nv_gauss);
  b_.Initialize(rng, /*scale=*/1e-30, nv_gauss);
}

template class HostContext<float>;
template class HostContext<double>;

// transa: whether to transpose matrix A before multiplication
// transb: whether to transpose matrix B before multiplication
//   CUBLAS_OP_T: transpose
//   CUBLAS_OP_N: no transpose
// lda: leading dimension of matrix A
// ldb: leading dimension of matrix B
// ldc: leading dimension of matrix C
template <class T>
inline cublasStatus_t DevGEMM(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const T *alpha, const T *A, int lda, const T *B,
                              int ldb, const T *beta, T *C, int ldc);

template <>
inline cublasStatus_t DevGEMM<double>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double *alpha, const double *A, int lda,
    const double *B, int ldb, const double *beta, double *C, int ldc) {
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}

template <>
inline cublasStatus_t DevGEMM<float>(cublasHandle_t handle,
                                     cublasOperation_t transa,
                                     cublasOperation_t transb, int m, int n,
                                     int k, const float *alpha, const float *A,
                                     int lda, const float *B, int ldb,
                                     const float *beta, float *C, int ldc) {
  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}

template <class T>
GpuContext<T>::GpuContext(HostContext<T> *h, int gpu_num) : gpu_num_(gpu_num) {
  dim_size_m_ = h->GetDimSizeM();
  dim_size_n_ = h->GetDimSizeN();
  dim_size_k_ = h->GetDimSizeK();
  transa_ = h->GetTransa();
  transb_ = h->GetTransb();
  CUDA_CHECK(cudaSetDevice(gpu_num_));
  cudaDeviceProp dev_prop;
  CUDA_CHECK(cudaGetDeviceProperties(&dev_prop, gpu_num_));
  LOG(INFO) << absl::StrFormat("using GPU %d at %04x:%02x:%02x model %s\n",
                               gpu_num_, dev_prop.pciDomainID,
                               dev_prop.pciBusID, dev_prop.pciDeviceID,
                               dev_prop.name);
  CUDA_CHECK(cudaStreamCreate(&stream_));
  CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  LogVersionsOnce(cublas_handle_);
  CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream_));
  CUBLAS_CHECK(
      cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE));
  // place alpha and beta on the device so they don't have
  // to be refetched in each iteration.
  static const T constants[] = {0., 1.};
  CUDA_CHECK(cudaMalloc(&dev_constants_, sizeof(constants)));
  CUDA_CHECK(cudaMemcpyAsync(dev_constants_, constants, sizeof(constants),
                             cudaMemcpyHostToDevice, stream_));
  size_t nr_bytes_a = dim_size_m_ * dim_size_k_ * sizeof(T);
  size_t nr_bytes_b = dim_size_k_ * dim_size_n_ * sizeof(T);
  size_t nr_bytes_c = dim_size_m_ * dim_size_n_ * sizeof(T);
  CUDA_CHECK(cudaMalloc(&dev_a_, nr_bytes_a));
  CUDA_CHECK(cudaMalloc(&dev_b_, nr_bytes_b));
  CUDA_CHECK(cudaMalloc(&dev_c_, nr_bytes_c));
  CUDA_CHECK(cudaMemcpyAsync(dev_a_, h->GetA(), nr_bytes_a,
                             cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaMemcpyAsync(dev_b_, h->GetB(), nr_bytes_b,
                             cudaMemcpyHostToDevice, stream_));
  // we don't bother synchronizing the copies
}

template <class T>
GpuContext<T>::~GpuContext() {
  CUDA_CHECK(cudaFree(dev_a_));
  CUDA_CHECK(cudaFree(dev_b_));
  CUDA_CHECK(cudaFree(dev_c_));
  CUDA_CHECK(cudaFree(dev_constants_));
  CUDA_CHECK(cudaStreamDestroy(stream_));
}

template <class T>
void GpuContext<T>::LaunchKernel() {
  CUDA_CHECK(cudaSetDevice(gpu_num_));
  cublasOperation_t transa = transa_ ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = transb_ ? CUBLAS_OP_T : CUBLAS_OP_N;
  // A: no transa => m x k; transa => k x m
  // B: no transa => k x n; transa => n x k
  // op(A): m x k; op(B): k x n; C: m x n
  int lda = transa_ ? dim_size_k_ : dim_size_m_;
  int ldb = transb_ ? dim_size_n_ : dim_size_k_;
  CUBLAS_CHECK(DevGEMM<T>(
      cublas_handle_, transa, transb, dim_size_m_, dim_size_n_,
      dim_size_k_, &dev_constants_[1], dev_a_, lda, dev_b_,
      ldb, &dev_constants_[0], dev_c_, dim_size_m_));
}


template class GpuContext<float>;
template class GpuContext<double>;
