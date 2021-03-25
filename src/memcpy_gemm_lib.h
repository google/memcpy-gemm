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
#include "src/gemm_test_lib.h"
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

  ~BufferPool();

  BufferPool(const BufferPool &) = delete;
  BufferPool &operator=(const BufferPool &) = delete;

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
  int from_dev_flow_cnt_ = 0;
  int to_dev_flow_cnt_ = 0;
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

// Tuning parameters for threads handling memcpy flows.
struct FlowThreadParameters {
  // Timing parameters for the copy threads.
  int64_t wait_ns;
  std::string flow_model;
  // Control which CUDA API is used to perform the memcpy.
  bool use_cudaMemcpyPeerAsync;  // NOLINT - suffix matches a CUDA convention.
  bool use_cudaMemcpyDefault;    // NOLINT - suffix matches a CUDA convention.
  bool use_cudaComputeCopy;
  bool use_group_by_dest;
  // Copy batch_size buffers at a time.
  int batch_size;
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
  CopySingleFlow(Flow *flow, const PulseBarrier *pulse_barrier,
                 const FlowThreadParameters &params)
      : flow_(flow),
        wait_ns_(params.wait_ns),
        batch_size_(params.batch_size),
        use_cudaMemcpyPeerAsync_(params.use_cudaMemcpyPeerAsync),
        use_cudaMemcpyDefault_(params.use_cudaMemcpyDefault),
        use_cudaComputeCopy_(params.use_cudaComputeCopy),
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
  bool use_cudaComputeCopy_;
  const PulseBarrier *pulse_barrier_;
};

// Copy multiple flows in one thread, where each flow waits for its prior
// invocation to complete, but otherwise does not wait for other flows.
class EventPollThread : public CopyThread {
 public:
  EventPollThread(std::vector<Flow *> flows, const FlowThreadParameters &params)
      : flows_(std::move(flows)),
        use_cudaMemcpyPeerAsync_(params.use_cudaMemcpyPeerAsync),
        use_cudaMemcpyDefault_(params.use_cudaMemcpyDefault),
        use_cudaComputeCopy_(params.use_cudaComputeCopy) {}

 protected:
  void Run() override;

 private:
  const std::vector<Flow *> flows_;
  const bool use_cudaMemcpyPeerAsync_;
  const bool use_cudaMemcpyDefault_;
  const bool use_cudaComputeCopy_;
};

// Each GPU and its associated in/out flows is managed by one thread.
// All flows on the given GPU are synchronized between each copy.
class PerGpuThread : public CopyThread {
 public:
  PerGpuThread(absl::string_view name_prefix, const PulseBarrier *pulse_barrier,
               std::vector<Flow *> flows, const FlowThreadParameters &params)
      : flows_(std::move(flows)),
        group_by_dest_(params.use_group_by_dest),
        batch_size_(params.batch_size),
        use_cudaMemcpyPeerAsync_(params.use_cudaMemcpyPeerAsync),
        use_cudaMemcpyDefault_(params.use_cudaMemcpyDefault),
        use_cudaComputeCopy_(params.use_cudaComputeCopy),
        name_prefix_(name_prefix),
        pulse_barrier_(pulse_barrier) {}

  std::string NamePrefix() { return name_prefix_; }

 protected:
  void Run() override;

 private:
  const std::vector<Flow *> flows_;
  const bool group_by_dest_;
  const int batch_size_;
  const bool use_cudaMemcpyPeerAsync_;
  const bool use_cudaMemcpyDefault_;
  const bool use_cudaComputeCopy_;
  const std::string name_prefix_;
  const PulseBarrier *pulse_barrier_;
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

// Creates cublasGemmEx compute stream using synchronization with CopyThread.
// This class uses mix-precision HostContext and GpuContext.
class GemmExComputeStream : public BaseComputeStream {
 public:
  // The HostContext should outlive the GemmExComputeStream
  GemmExComputeStream(platforms_gpus::gemm_test::GpuContext *gpu_context,
                      const PulseBarrier *pulse_barrier,
                      int outstanding_operations_in_flight)
      : gpu_context_(gpu_context),
        pulse_barrier_(pulse_barrier),
        outstanding_operations_in_flight_(outstanding_operations_in_flight) {}

 protected:
  void Run() override {
    bool printWarning = false;
    while (!stop_copying_.HasBeenNotified()) {
      const absl::Time deadline = pulse_barrier_->HighTimeDeadline();
      absl::Time start_time = absl::Now();
      absl::Time last_time = start_time;
      absl::Time end_time;
      do {
        for (int i = 0; i < outstanding_operations_in_flight_; i++) {
          gpu_context_->LaunchKernel();
          AddLoop();
        }
        gpu_context_->StreamSynchronize();
        end_time = absl::Now();
        if (!printWarning && outstanding_operations_in_flight_ == 1 &&
            (end_time - last_time) < absl::Microseconds(20)) {
          printWarning = true;
          LOG(WARNING) << "Kernel execute time too short, may not accurate !!!";
        }
        last_time = end_time;
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
  platforms_gpus::gemm_test::GpuContext *gpu_context_;
  const PulseBarrier *pulse_barrier_;
  const int outstanding_operations_in_flight_;
};

// Creates GemmExComputeAutoTuneStream compute stream using synchronization with
// CopyThread. This class is used to find the best algorithm of GEMM gpu
// context.
class GemmExComputeAutoTuneStream : public BaseComputeStream {
 public:
  GemmExComputeAutoTuneStream(
      platforms_gpus::gemm_test::GpuContext *gpu_context)
      : gpu_context_(gpu_context) {}
  void Run() override { gpu_context_->AutoTuning(); }

 private:
  platforms_gpus::gemm_test::GpuContext *gpu_context_;
};

std::vector<std::unique_ptr<platforms_gpus::gemm_test::GpuContext>>
CreateGpuContexts(platforms_gpus::gemm_test::HostContext *host_ctx,
                  absl::Span<const int64_t> gpu_list);

void GemmAutoTune(
    std::vector<std::unique_ptr<platforms_gpus::gemm_test::GpuContext>>
        &gpu_ctxs);

// factory method to create a thread that wraps a multi_gemm task
// The HostContext should outlive the created CopyThread.
std::unique_ptr<CopyThread> CreateGemmThread(
    platforms_gpus::gemm_test::GpuContext *gpu_context,
    memcpy_gemm::PulseBarrier *pulse_barrier,
    int outstanding_operations_in_flight);

// Makes compute threads for the provided gpu_list initialized according to
// Host Context, with one thread per GPU managing the GEMM operations on that
// GPU.
std::vector<std::unique_ptr<CopyThread>> MakeComputeThreads(
    std::vector<std::unique_ptr<platforms_gpus::gemm_test::GpuContext>>
        &gpu_ctxs,
    PulseBarrier *pulse_barrier, int outstanding_operations_in_flight);

// Makes memcpy threads for the provided flows and params
std::vector<std::unique_ptr<CopyThread>> MakeMemcpyThreads(
    const FlowThreadParameters &params,
    std::vector<std::unique_ptr<Flow>> &flows, PulseBarrier *pulse_barrier);

}  // namespace memcpy_gemm
}  // namespace platforms_gpus

#endif  // PLATFORMS_GPUS_TESTING_NVIDIA_MEMCPY_GEMM_LIB_H_
