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

#ifndef PLATFORMS_GPUS_TESTING_NVIDIA_MEMORY_ALLOCATOR_INTERFACE_H_
#define PLATFORMS_GPUS_TESTING_NVIDIA_MEMORY_ALLOCATOR_INTERFACE_H_



#include <stddef.h>

namespace platforms_gpus {
namespace gemm_test {

// This abstract class serves as an interface for cuda memory
// allocation/deallocation APIs that used by gemm_test. We introduce this class
// mainly for mocking/testing purpose.
class MemoryAllocatorInterface {
 public:
  virtual ~MemoryAllocatorInterface() {}

  // Caller of AllocateHostMemory() will obtain the ownership of the returned
  // pointer. The caller should call FreeHostMemory(void *addr) to free the
  // allocated memory later.
  virtual void *AllocateHostMemory(size_t nr_bytes) = 0;

  // Pointer "addr" should point to a piece of memory previously allocated by
  // AllocateHostMemory(). When calling FreeHostMemory(), the caller is to pass
  // ownership of the pointer "addr" to FreeHostMemory().
  virtual void FreeHostMemory(void *addr) = 0;
};

}  // namespace gemm_test
}  // namespace platforms_gpus

#endif  // PLATFORMS_GPUS_TESTING_NVIDIA_MEMORY_ALLOCATOR_INTERFACE_H_
