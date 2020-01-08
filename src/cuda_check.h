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

#ifndef PLATFORMS_GPUS_TESTING_NVIDIA_CUDA_CHECK_H_
#define PLATFORMS_GPUS_TESTING_NVIDIA_CUDA_CHECK_H_

#define CUDA_CHECK(condition)                                \
  {                                                          \
    cudaError_t cu_err = (condition);                        \
    LOG_IF(FATAL, ABSL_PREDICT_FALSE(cu_err != cudaSuccess)) \
        << "CUDA_CHECK failed: " #condition " : "            \
        << cudaGetErrorString(cu_err);                       \
  }

#define CUBLAS_CHECK(condition)                                            \
  {                                                                        \
    cublasStatus_t cublas_err = (condition);                               \
    LOG_IF(FATAL, ABSL_PREDICT_FALSE(cublas_err != CUBLAS_STATUS_SUCCESS)) \
        << "CUBLAS_CHECK failed: " #condition " : " << cublas_err;         \
  }
#endif  // PLATFORMS_GPUS_TESTING_NVIDIA_CUDA_CHECK_H_
