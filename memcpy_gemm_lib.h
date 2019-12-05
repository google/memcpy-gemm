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


#ifndef PLATFORMS_GPUS_TESTING_NVIDIA_MEMCPY_GEMM_LIB_H_
#define PLATFORMS_GPUS_TESTING_NVIDIA_MEMCPY_GEMM_LIB_H_

#include <stddef.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "absl/memory/memory.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gemm_test_lib.h"
#include "multi_gemm_lib.h"
#include "cuda/include/cuda_runtime.h"

namespace platforms_gpus {
namespace memcpy_gemm {

enum DeviceType { CPU, GPU };
typedef std::pair<DeviceType, int> DeviceSpec;

std::string DeviceSpecToString(DeviceSpec dev);

// Allocates the buffer on CPU or GPU.
class BufferPool {
 public:
  explicit BufferPool(size_t size) : size_(size) {}

  // buffer_index == -1 creates an anonymous buffer
  char *GetBuffer(DeviceSpec dev, int buffer_index);

 protected:
  size_t size_;
  std::map<DeviceSpec, std::vector<char *>> buffers_;
};

// Stores description of the flow.
class Flow {
 public:
  Flow(DeviceSpec from_dev, char *from_mem, DeviceSpec to_dev, char *to_mem,
       size_t buf_size, std::atomic<int> *counter)
      : from_dev_(from_dev),
        from_mem_(from_mem),
        to_dev_(to_dev),
        to_mem_(to_mem),
        buf_size_(buf_size),
        counter_(counter) {}

  DeviceSpec from_dev_;
  char *from_mem_;
  DeviceSpec to_dev_;
  char *to_mem_;
  size_t buf_size_;
  std::atomic<int> *counter_;
  cudaStream_t stream_;
};

class PulseBarrier {
 public:
  PulseBarrier(absl::Duration high_time, absl::Duration low_time,
               bool sync_flows)
      : high_time_(high_time), low_time_(low_time), sync_flows_(sync_flows) {}

  // Returns the next high-time end time.
  // Doesn't return until low_time has expired in this cycle.
  absl::Time HighTimeDeadline() const {
    absl::Time now = absl::Now();
    absl::Time sync_now = now;

    if (sync_flows_) {
      // If requested to sync, align everything to the same window.
      sync_now = absl::UnixEpoch() +
                 absl::Floor(now - absl::UnixEpoch(), low_time_ + high_time_);
    }

    //  To ensure sleep precision, back off a bit and busy-poll.
    absl::SleepFor(sync_now - now + low_time_ - absl::Milliseconds(20));

    auto nanos = absl::ToUnixNanos(sync_now + low_time_);
    while (absl::GetCurrentTimeNanos() < nanos) {
    }

    return sync_now + low_time_ + high_time_;
  }

 private:
  absl::Duration high_time_;
  absl::Duration low_time_;
  bool sync_flows_;
};

// The CopyThread will exit by setting stop_copying_ (set by the signal
// handler thread).
class CopyThread {
 public:
  CopyThread() = default;
  virtual ~CopyThread() = default;

  void StopCopying() { stop_copying_.Notify(); }
  // Upon calling start, a new thread will be spawned executing Run().
  virtual void Run() = 0;
  // Execute the the contents of Run() in a new thread.
  void Start();
  // Wait until the thread has completed.
  void Join();

 protected:
  std::optional<std::thread> thread_handler_;
  absl::Notification stop_copying_;
};

// Copies one flow in one thread.
class CopySingleFlow : public CopyThread {
 public:
  CopySingleFlow(Flow *flow, const PulseBarrier *pulse_barrier, int wait_ns,
                 int batch_size, bool use_cudaMemcpyPeerAsync,
                 bool use_cudaMemcpyDefault)
      : flow_(flow),
        wait_ns_(wait_ns),
        batch_size_(batch_size),
        use_cudaMemcpyPeerAsync_(use_cudaMemcpyPeerAsync),
        use_cudaMemcpyDefault_(use_cudaMemcpyDefault),
        pulse_barrier_(pulse_barrier) {}

 protected:
  // With sync_flows_ set, repeatedly copies the specified flow for high_time,
  // then waits for the next synchronization of the blocking counter.
  void Run() override;

 private:
  Flow *flow_;
  int wait_ns_;
  int batch_size_;
  bool use_cudaMemcpyPeerAsync_;
  bool use_cudaMemcpyDefault_;
  const PulseBarrier *pulse_barrier_;
};

// Copy multiple flows in one thread, where each flow waits for its prior
// invocation to complete, but otherwise does not wait for other flows.
class EventPollThread : public CopyThread {
 public:
  EventPollThread(std::vector<Flow *> flows, bool use_cudaMemcpyPeerAsync,
                  bool use_cudaMemcpyDefault)
      : flows_(std::move(flows)),
        use_cudaMemcpyPeerAsync_(use_cudaMemcpyPeerAsync),
        use_cudaMemcpyDefault_(use_cudaMemcpyDefault) {}

 protected:
  void Run() override;

 private:
  const std::vector<Flow *> flows_;
  const bool use_cudaMemcpyPeerAsync_;
  const bool use_cudaMemcpyDefault_;
};

// Each GPU and its associated in/out flows is managed by one thread.
// All flows on the given GPU are synchronized between each copy.
class PerGpuThread : public CopyThread {
 public:
  PerGpuThread(absl::string_view name_prefix, std::vector<Flow *> flows,
               bool group_by_dest, bool use_cudaMemcpyPeerAsync,
               bool use_cudaMemcpyDefault)
      : flows_(std::move(flows)),
        group_by_dest_(group_by_dest),
        use_cudaMemcpyPeerAsync_(use_cudaMemcpyPeerAsync),
        use_cudaMemcpyDefault_(use_cudaMemcpyDefault),
        name_prefix_(name_prefix) {}

  std::string NamePrefix() { return name_prefix_; }

 protected:
  void Run() override;

 private:
  const std::vector<Flow *> flows_;
  const bool group_by_dest_;
  const bool use_cudaMemcpyPeerAsync_;
  const bool use_cudaMemcpyDefault_;
  const std::string name_prefix_;
};

class BaseComputeStream : public CopyThread {
 public:
  using CopyThread::CopyThread;
  double AccumulatedTime() { return accum_time_; }
  int Loops() { return loops_; }

 protected:
  void AddAccumulatedTime(double time) { accum_time_ += time; }
  void AddLoop() { loops_++; }

 private:
  double accum_time_ = 0;
  int loops_ = 0;
};

// Creates SGEMM or DGEMM compute stream using synchronization with CopyThread.
template <class T>
class ComputeStream : public BaseComputeStream {
 public:
  ComputeStream(std::shared_ptr<HostContext<T>> host_context, int gpu_number,
                const PulseBarrier *pulse_barrier)
      : host_context_(std::move(host_context)),
        gpu_context_(
            absl::make_unique<GpuContext<T>>(host_context_.get(), gpu_number)),
        pulse_barrier_(pulse_barrier) {}

 protected:
  // Performs the GEMM for high_time followed by low_time non-compute, then
  // waits for the blocking counter.
  void Run() override {
    while (!stop_copying_.HasBeenNotified()) {
      const absl::Time deadline = pulse_barrier_->HighTimeDeadline();
      absl::Time start_time = absl::Now();
      absl::Time end_time;
      do {
        gpu_context_->LaunchKernel();
        gpu_context_->StreamSynchronize();
        end_time = absl::Now();
        AddLoop();
        // The end condition ends up getting slightly smeared as the kernel
        // isn't scheduled perfectly. Cancellation isn't sent though to the GPU,
        // which would require special GPU kernel support, since it's not
        // provided by the NVidia runtime.
      } while (!stop_copying_.HasBeenNotified() && end_time < deadline);

      AddAccumulatedTime(absl::ToDoubleSeconds(end_time - start_time));
      VLOG(2) << "pulse high for " << (end_time - start_time);
    }
  }

 private:
  std::shared_ptr<HostContext<T>> host_context_;
  std::unique_ptr<GpuContext<T>> gpu_context_;
  const PulseBarrier *pulse_barrier_;
};

// Creates cublasGemmEx compute stream using synchronization with CopyThread.
// This class is pretty much copied from class ComputeStream. Difference is that
// GemmExComputeStream class uses mix-precision HostContext and GpuContext.
class GemmExComputeStream : public BaseComputeStream {
 public:
  // The HostContext should outlive the GemmExComputeStream
  GemmExComputeStream(platforms_gpus::gemm_test::HostContext *host_context,
                      int gpu_number, const PulseBarrier *pulse_barrier)
      : host_context_(host_context), pulse_barrier_(pulse_barrier) {
    gpu_context_ = platforms_gpus::gemm_test::GpuContext::Create(
        host_context_, gpu_number);
  }

 protected:
  void Run() override {
    while (!stop_copying_.HasBeenNotified()) {
      const absl::Time deadline = pulse_barrier_->HighTimeDeadline();
      absl::Time start_time = absl::Now();
      absl::Time end_time;
      do {
        gpu_context_->LaunchKernel();
        gpu_context_->StreamSynchronize();
        end_time = absl::Now();
        AddLoop();
        // The end condition ends up getting slightly smeared as the kernel
        // isn't scheduled perfectly. Cancellation isn't sent though to the GPU,
        // which would require special GPU kernel support, since it's not
        // provided by the NVidia runtime.
      } while (!stop_copying_.HasBeenNotified() && end_time < deadline);

      AddAccumulatedTime(absl::ToDoubleSeconds(end_time - start_time));
      VLOG(2) << "pulse high for " << (end_time - start_time);
    }
  }

 private:
  platforms_gpus::gemm_test::HostContext *host_context_;
  std::unique_ptr<platforms_gpus::gemm_test::GpuContext> gpu_context_;
  const PulseBarrier *pulse_barrier_;
};

// factory method to create a thread that wraps a multi_gemm task
// The HostContext should outlive the created CopyThread.
std::unique_ptr<CopyThread> CreateGemmThread(
    platforms_gpus::gemm_test::HostContext *host_context,
    memcpy_gemm::PulseBarrier *pulse_barrier, int64_t gpu_num);

}  // namespace memcpy_gemm
}  // namespace platforms_gpus

#endif  // PLATFORMS_GPUS_TESTING_NVIDIA_MEMCPY_GEMM_LIB_H_
