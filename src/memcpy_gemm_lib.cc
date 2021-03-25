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
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "src/cuda_check.h"
#include "src/cuda_compute_copy.h"
#include "src/gemm_test_lib.h"
#include "cuda/include/cuda_runtime.h"
#include "numa.h"

namespace platforms_gpus {
namespace memcpy_gemm {

// Converts numbered CPU or GPU into a printable string.
std::string DeviceSpecToString(DeviceSpec dev) {
  return absl::StrFormat("%c%d", dev.first == GPU ? 'g' : 'c', dev.second);
}

BufferPool::~BufferPool() {
  for (const auto &dev : buffers_) {
    for (auto ptr : dev.second) {
      switch (dev.first.first) {
        case CPU:
          CUDA_CHECK(cudaHostUnregister(ptr));
          numa_free(ptr, size_);
          break;
        case GPU:
          CUDA_CHECK(cudaFree(ptr));
          break;
      }
    }
  }
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
                            bool use_cudaMemcpyDefault,
                            bool use_cudaComputeCopy) {
  if (use_cudaComputeCopy && f->from_dev_.first == GPU &&
      f->to_dev_.first == GPU) {
    CUDA_CHECK(cudaComputeCopyAdaptive(f->stream_, f->to_mem_, f->from_mem_,
                                       f->buf_size_, f->to_dev_flow_cnt_));
  } else if (use_cudaMemcpyPeerAsync && f->to_dev_.first == GPU &&
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
    if (use_cudaComputeCopy_ && flow_->to_dev_.first == GPU &&
        flow_->from_dev_.first == GPU) {
      // Prefer pulling for SM copies as they are faster at least on HGX 8 A100
      return flow_->to_dev_.second;
    } else if (flow_->from_dev_.first == GPU) {
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
    MemcpyConfusino(flow_, use_cudaMemcpyPeerAsync_, use_cudaMemcpyDefault_,
                    use_cudaComputeCopy_);
    CUDA_CHECK(cudaEventRecord(events[i], flow_->stream_));
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
        MemcpyConfusino(flow_, use_cudaMemcpyPeerAsync_, use_cudaMemcpyDefault_,
                        use_cudaComputeCopy_);
        CUDA_CHECK(cudaEventRecord(events[i], flow_->stream_));
      }
    } while (absl::Now() < deadline && !stop_copying_.HasBeenNotified());
  }
  CUDA_CHECK(cudaStreamSynchronize(0));
  for (auto e : events) {
    CUDA_CHECK(cudaEventDestroy(e));
  }
  CUDA_CHECK(cudaStreamDestroy(flow_->stream_));
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
    MemcpyConfusino(f, use_cudaMemcpyPeerAsync_, use_cudaMemcpyDefault_,
                    use_cudaComputeCopy_);
    CUDA_CHECK(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(events[i], f->stream_));
  }
  int cur_event = 0;
  while (!stop_copying_.HasBeenNotified()) {
    if (cudaEventQuery(events[cur_event]) == cudaSuccess) {
      Flow *f = flows_[cur_event];
      MemcpyConfusino(f, use_cudaMemcpyPeerAsync_, use_cudaMemcpyDefault_,
                      use_cudaComputeCopy_);
      CUDA_CHECK(cudaEventRecord(events[cur_event], f->stream_));
      (*f->counter_)++;
    }
    ++cur_event;
    if (cur_event == nr_flows) cur_event = 0;
  }
  CUDA_CHECK(cudaStreamSynchronize(0));
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
  absl::flat_hash_map<cudaStream_t, std::vector<cudaEvent_t>> streamEvents;
  for (auto f : flows_) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    f->stream_ = stream;
    LOG(INFO) << NamePrefix() << " copying " << DeviceSpecToString(f->from_dev_)
              << " to " << DeviceSpecToString(f->to_dev_);
    for (int i = 0; i < batch_size_; i++) {
      cudaEvent_t event;
      CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      streamEvents[stream].push_back(event);
      MemcpyConfusino(f, use_cudaMemcpyPeerAsync_, use_cudaMemcpyDefault_,
                      use_cudaComputeCopy_);
      CUDA_CHECK(cudaEventRecord(event, stream));
    }
  }

  std::vector<int> eventCheckIndex(flows_.size(), 0);
  while (!stop_copying_.HasBeenNotified()) {
    const absl::Time deadline = pulse_barrier_->HighTimeDeadline();
    do {
      for (size_t i = 0; i < flows_.size(); i++) {
        const std::vector<cudaEvent_t> &events =
            streamEvents[flows_[i]->stream_];
        if (cudaEventQuery(events[eventCheckIndex[i]]) == cudaSuccess) {
          (*flows_[i]->counter_)++;
          MemcpyConfusino(flows_[i], use_cudaMemcpyPeerAsync_,
                          use_cudaMemcpyDefault_, use_cudaComputeCopy_);
          CUDA_CHECK(
              cudaEventRecord(events[eventCheckIndex[i]], flows_[i]->stream_));
          eventCheckIndex[i]++;
          eventCheckIndex[i] %= batch_size_;
        }
      }
    } while (absl::Now() < deadline && !stop_copying_.HasBeenNotified());
  }
  CUDA_CHECK(cudaStreamSynchronize(0));
  for (auto f : flows_) {
    std::vector<cudaEvent_t> &events = streamEvents[f->stream_];
    for (auto e : events) {
      CUDA_CHECK(cudaEventDestroy(e));
    }
    CUDA_CHECK(cudaStreamDestroy(f->stream_));
  }
}

std::vector<std::unique_ptr<gemm_test::GpuContext>> CreateGpuContexts(
    platforms_gpus::gemm_test::HostContext *host_ctx,
    absl::Span<const int64_t> gpu_list) {
  std::vector<std::unique_ptr<platforms_gpus::gemm_test::GpuContext>> gpu_ctxs;
  for (const int64_t gpu : gpu_list) {
    std::unique_ptr<platforms_gpus::gemm_test::GpuContext> gpuctx =
        platforms_gpus::gemm_test::GpuContext::Create(host_ctx, gpu);
    if (gpuctx == nullptr) {
      LOG(ERROR) << "Failed to create GPU context " << gpu;
      continue;
    }
    gpu_ctxs.push_back(std::move(gpuctx));
  }
  return gpu_ctxs;
}

void GemmAutoTune(
    std::vector<std::unique_ptr<platforms_gpus::gemm_test::GpuContext>>
        &gpu_ctxs) {
  std::vector<std::unique_ptr<GemmExComputeAutoTuneStream>> autotune_threads;
  for (size_t i = 0; i < gpu_ctxs.size(); i++) {
    auto th = std::make_unique<GemmExComputeAutoTuneStream>(gpu_ctxs[i].get());
    if (th == nullptr) {
      LOG(ERROR) << "Failed to create Auto Tune  thread failed "
                 << gpu_ctxs[i]->GetGpuIndex();
      continue;
    }
    autotune_threads.push_back(std::move(th));
  }

  LOG(INFO) << "Starting Auto Tune threads";
  for (const auto &t : autotune_threads) {
    t->Start();
  }
  for (const auto &t : autotune_threads) {
    t->Join();
  }
  LOG(INFO) << "Join Auto Tune threads";
}

std::unique_ptr<CopyThread> CreateGemmThread(
    platforms_gpus::gemm_test::GpuContext *gpu_context,
    platforms_gpus::memcpy_gemm::PulseBarrier *pulse_barrier,
    int outstanding_operations_in_flight) {
  CHECK(gpu_context != nullptr);
  return absl::make_unique<GemmExComputeStream>(
      gpu_context, pulse_barrier, outstanding_operations_in_flight);
}

std::vector<std::unique_ptr<CopyThread>> MakeComputeThreads(
    std::vector<std::unique_ptr<platforms_gpus::gemm_test::GpuContext>>
        &gpu_ctxs,
    PulseBarrier *pulse_barrier, int outstanding_operations_in_flight) {
  std::vector<std::unique_ptr<CopyThread>> threads;
  for (auto &ctx : gpu_ctxs) {
    std::unique_ptr<CopyThread> thread = CreateGemmThread(
        ctx.get(), pulse_barrier, outstanding_operations_in_flight);
    if (!thread) {
      LOG(ERROR) << "Failed to create GEMM Thread for GPU "
                 << ctx->GetGpuIndex();
      continue;
    }
    threads.push_back(std::move(thread));
  }
  return threads;
}

std::vector<std::unique_ptr<CopyThread>> MakeMemcpyThreads(
    const FlowThreadParameters &params,
    std::vector<std::unique_ptr<Flow>> &flows, PulseBarrier *pulse_barrier) {
  std::vector<std::unique_ptr<CopyThread>> threads;
  if (params.flow_model == "thread-per-flow") {
    for (int i = 0; i < flows.size(); i++) {
      threads.push_back(std::make_unique<CopySingleFlow>(
          flows[i].get(), pulse_barrier, params));
    }
  } else if (params.flow_model == "event-poll") {
    std::vector<Flow *> tmpf(flows.size(), nullptr);
    for (size_t i = 0; i < flows.size(); i++) {
      tmpf[i] = flows[i].get();
    }
    threads.emplace_back(new EventPollThread(tmpf, params));
  } else if (params.flow_model == "thread-per-gpu") {
    absl::flat_hash_map<DeviceSpec, std::vector<Flow *>> gpu_flows;
    for (auto &f : flows) {
      DeviceSpec d = params.use_group_by_dest ? f->to_dev_ : f->from_dev_;
      gpu_flows[d].push_back(f.get());
    }
    for (auto &d : gpu_flows) {
      threads.emplace_back(
          new PerGpuThread("per_gpu_thread_" + DeviceSpecToString(d.first),
                           pulse_barrier, d.second, params));
    }
  }
  return threads;
}

}  // namespace memcpy_gemm
}  // namespace platforms_gpus
