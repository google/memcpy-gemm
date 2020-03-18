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

#include "src/memcpy_gemm_lib.h"

#include <string.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "src/cuda_check.h"
#include "src/gemm_test_lib.h"
#include "cuda/include/cuda_runtime.h"
#include "numa.h"

namespace platforms_gpus {
namespace memcpy_gemm {

// Converts numbered CPU or GPU into a printable string.
std::string DeviceSpecToString(DeviceSpec dev) {
  return absl::StrFormat("%c%d", dev.first == GPU ? 'g' : 'c', dev.second);
}

char *BufferPool::GetBuffer(DeviceSpec dev, int buffer_index) {
  //numa_init();
  // Get readout from libnuma here, but do not fail unless the CPU path is used.
  const int numa_works = numa_available();
  CHECK_GE(buffer_index, -1);
  CHECK_GE(dev.second, 0);
  if (buffer_index >= 0) {
    if (buffers_.count(dev) == 0) {
      buffers_[dev] = std::vector<char *>(buffer_index + 1, nullptr);
    } else if (buffers_[dev].size() <= buffer_index) {
      buffers_[dev].resize(buffer_index + 1, nullptr);
    } else if (buffers_[dev][buffer_index]) {
      return buffers_[dev][buffer_index];
    }
  }
  char *mem;
  switch (dev.first) {
    case CPU:
      CHECK_GE(numa_works, 0) << "NUMA operations unsupported";
      mem = static_cast<char *>(numa_alloc_onnode(size_, dev.second));
      memset(mem, 0, size_);
      CUDA_CHECK(cudaHostRegister(mem, size_, cudaHostRegisterPortable));
      break;
    case GPU:
      CUDA_CHECK(cudaSetDevice(dev.second));
      CUDA_CHECK(cudaMalloc(&mem, size_));
      CUDA_CHECK(cudaMemset(mem, 0, size_));
      break;
  }
  LOG(INFO) << absl::StreamFormat("buffers_[%s,%d] = %p",
                                  DeviceSpecToString(dev).c_str(), buffer_index,
                                  mem);
  if (buffer_index >= 0) {
    buffers_[dev][buffer_index] = mem;
  }
  return mem;
}

static void MemcpyConfusino(Flow *f, bool use_cudaMemcpyPeerAsync,
                            bool use_cudaMemcpyDefault) {
  if (use_cudaMemcpyPeerAsync && f->to_dev_.first == GPU &&
      f->from_dev_.first == GPU) {
    CUDA_CHECK(cudaMemcpyPeerAsync(f->to_mem_, f->to_dev_.second, f->from_mem_,
                                   f->from_dev_.second, f->buf_size_,
                                   f->stream_));
  } else if (use_cudaMemcpyDefault) {
    CUDA_CHECK(cudaMemcpyAsync(f->to_mem_, f->from_mem_, f->buf_size_,
                               cudaMemcpyDefault, f->stream_));
  } else {
    cudaMemcpyKind kind;
    if (f->from_dev_.first == GPU && f->to_dev_.first == GPU) {
      kind = cudaMemcpyDeviceToDevice;
    } else if (f->from_dev_.first == GPU && f->to_dev_.first == CPU) {
      kind = cudaMemcpyDeviceToHost;
    } else if (f->from_dev_.first == CPU && f->to_dev_.first == GPU) {
      kind = cudaMemcpyHostToDevice;
    } else if (f->from_dev_.first == CPU && f->to_dev_.first == CPU) {
      kind = cudaMemcpyHostToHost;
    } else {
      kind = cudaMemcpyDefault;
    }
    CUDA_CHECK(cudaMemcpyAsync(f->to_mem_, f->from_mem_, f->buf_size_, kind,
                               f->stream_));
  }
}

void CopyThread::Start() {
  thread_handler_ = std::thread(&CopyThread::Run, this);
}

void CopyThread::Join() {
  CHECK(thread_handler_.has_value()) << "Join called before start";
  thread_handler_->join();
}

void CopySingleFlow::Run() {
  const int want_device = [this] {
    if (flow_->from_dev_.first == GPU) {
      return flow_->from_dev_.second;
    } else if (flow_->to_dev_.first == GPU) {
      return flow_->to_dev_.second;
    }

    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
  }();

  CUDA_CHECK(cudaSetDevice(want_device));
  CUDA_CHECK(cudaStreamCreate(&flow_->stream_));

  std::vector<cudaEvent_t> events;
  events.resize(batch_size_);
  // Send off initial batches.
  for (int i = 0; i < batch_size_; i++) {
    cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
    MemcpyConfusino(flow_, use_cudaMemcpyPeerAsync_, use_cudaMemcpyDefault_);
    CUDA_CHECK(cudaEventRecord(events[i]));
  }

  while (!stop_copying_.HasBeenNotified()) {
    absl::Time deadline = pulse_barrier_->HighTimeDeadline();
    do {
      // The GPU should execute the memcopies in order, so we can block on the
      // next expected completed batch. As soon as a batch is finished, it is
      // relaunched, so when waiting for one batch to finish, all others are
      // either in the stream queue or being executed.
      for (int i = 0; i < batch_size_; i++) {
        CUDA_CHECK(cudaEventSynchronize(events[i]));
        (*flow_->counter_)++;
        MemcpyConfusino(flow_, use_cudaMemcpyPeerAsync_,
                        use_cudaMemcpyDefault_);
        CUDA_CHECK(cudaEventRecord(events[i], flow_->stream_));
      }
    } while (absl::Now() < deadline && !stop_copying_.HasBeenNotified());
  }
}

void EventPollThread::Run() {
  int nr_flows = flows_.size();
  std::vector<cudaEvent_t> events(nr_flows);
  for (int i = 0; i < nr_flows; ++i) {
    Flow *f = flows_[i];
    if (f->from_dev_.first == GPU) {
      CUDA_CHECK(cudaSetDevice(f->from_dev_.second));
    } else if (f->to_dev_.first == GPU) {
      CUDA_CHECK(cudaSetDevice(f->to_dev_.second));
    }
    CUDA_CHECK(cudaStreamCreate(&f->stream_));
    MemcpyConfusino(f, use_cudaMemcpyPeerAsync_, use_cudaMemcpyDefault_);
    CUDA_CHECK(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(events[i], f->stream_));
  }
  int cur_event = 0;
  while (!stop_copying_.HasBeenNotified()) {
    if (cudaEventQuery(events[cur_event]) == cudaSuccess) {
      Flow *f = flows_[cur_event];
      MemcpyConfusino(f, use_cudaMemcpyPeerAsync_, use_cudaMemcpyDefault_);
      CUDA_CHECK(cudaEventRecord(events[cur_event], f->stream_));
      (*f->counter_)++;
    }
    ++cur_event;
    if (cur_event == nr_flows) cur_event = 0;
  }
  for (int i = 0; i < nr_flows; ++i) {
    CUDA_CHECK(cudaEventDestroy(events[i]));
    CUDA_CHECK(cudaStreamDestroy(flows_[i]->stream_));
  }
}

void PerGpuThread::Run() {
  CHECK_GT(flows_.size(), 0);
  Flow *f_zero = flows_[0];
  if (!group_by_dest_ && f_zero->from_dev_.first == GPU) {
    CUDA_CHECK(cudaSetDevice(f_zero->from_dev_.second));
  } else if (group_by_dest_ && f_zero->to_dev_.first == GPU) {
    CUDA_CHECK(cudaSetDevice(f_zero->to_dev_.second));
  }
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  for (auto f : flows_) {
    f->stream_ = stream;
    LOG(INFO) << NamePrefix() << " copying " << DeviceSpecToString(f->from_dev_)
              << " to " << DeviceSpecToString(f->to_dev_);
    MemcpyConfusino(f, use_cudaMemcpyPeerAsync_, use_cudaMemcpyDefault_);
  }
  while (!stop_copying_.HasBeenNotified()) {
    CUDA_CHECK(cudaStreamSynchronize(stream))
    for (auto f : flows_) {
      (*f->counter_)++;
      MemcpyConfusino(f, use_cudaMemcpyPeerAsync_, use_cudaMemcpyDefault_);
    }
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
}

std::unique_ptr<CopyThread> CreateGemmThread(
    platforms_gpus::gemm_test::HostContext *host_context,
    platforms_gpus::memcpy_gemm::PulseBarrier *pulse_barrier, int64_t gpu_num) {
  CHECK(host_context != nullptr);

  return absl::make_unique<GemmExComputeStream>(host_context, gpu_num,
                                                pulse_barrier);
}

}  // namespace memcpy_gemm
}  // namespace platforms_gpus
