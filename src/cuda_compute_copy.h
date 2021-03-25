#ifndef THIRD_PARTY_GPU_TEST_UTILS_CUDA_COMPUTE_COPY_H_
#define THIRD_PARTY_GPU_TEST_UTILS_CUDA_COMPUTE_COPY_H_

#include "cuda/include/cuda.h"
#include "cuda/include/cuda_runtime.h"

// We include this here because it requires declarations from cuda*.h when
// included in a standard C++ file, but the CUDA compiler will complain if
// added to a CUDA C source file.
#include "src/cuda_compute_copy.cu.h"

#endif  // THIRD_PARTY_GPU_TEST_UTILS_CUDA_COMPUTE_COPY_H_
