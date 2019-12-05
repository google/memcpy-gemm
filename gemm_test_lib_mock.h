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

#ifndef PLATFORMS_GPUS_TESTING_NVIDIA_GEMM_TEST_LIB_MOCK_H_
#define PLATFORMS_GPUS_TESTING_NVIDIA_GEMM_TEST_LIB_MOCK_H_

#include "gmock/gmock.h"
#include "gemm_test_lib.h"
#include "memory_allocator_interface.h"

namespace platforms_gpus {
namespace gemm_test {

class Mock_HostContext : public HostContext {
 private:
  MOCK_METHOD1(CreateGpuContext, std::unique_ptr<GpuContext>(int gpu_num));
};

class Mock_GpuContext : public GpuContext {
 public:
  MOCK_METHOD0(StreamSynchronize, void());
  MOCK_METHOD0(LaunchKernel, void());
};

// Mocked CudaInterface for unit tests.
class MockMemoryAllocator : public MemoryAllocatorInterface {
 public:
  MOCK_METHOD1(AllocateHostMemory, void *(size_t nr_bytes));
  MOCK_METHOD1(FreeHostMemory, void(void *addr));
};

}  // namespace gemm_test
}  // namespace platforms_gpus

#endif  // PLATFORMS_GPUS_TESTING_NVIDIA_GEMM_TEST_LIB_MOCK_H_
