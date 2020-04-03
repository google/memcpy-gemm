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

#include <stddef.h>
#include <string.h>
#include <unistd.h>

#include <atomic>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "src/cuda_check.h"
#include "src/gemm_test_lib.h"
#include "src/memcpy_gemm_lib.h"
#include "cuda/include/cuda_runtime.h"
#include "re2/re2.h"

ABSL_FLAG(
    std::string, flows, "2:c0-g0-a0 2:g0-c0-a1",
    "Space-separated flow programs.  Each flow program begins with an "
    "optional \"\\d+:\" replica count.  Followed by a source and destination "
    "specification separated by a dash.  Source and dest spec both match regex "
    "\"[cg]\\d+(/\\d+)\" where c = cpu, g = gpu, the first number is the CPU "
    "socket number or GPU socket number.  The optional \"/\\d+\" allows for "
    "named buffer selection, otherwise anonymous buffers are used for each "
    "src/dst.  Each src-dest pair is followed by optional \"-a\\d+\" which "
    "indicates which accumulator to increment for the transfer.");
ABSL_FLAG(int32_t, buffer_size_KiB, 16 * 1024, "GPU buffer size in KiB.");
ABSL_FLAG(int32_t, sample_time, 1, "Sample time in seconds.");
ABSL_FLAG(absl::Duration, duration, absl::InfiniteDuration(),
          "Test duration in sec.  inf is infinite.");
ABSL_FLAG(int32_t, batch_size, 1,
          "How many cudaMemcpyAsync to issue in each batch.");
ABSL_FLAG(bool, use_cudaDeviceEnablePeerAccess, true,
          "call cudaDeviceEnablePeerAccess for each flow.");
ABSL_FLAG(bool, use_cudaMemcpyPeerAsync, false,
          "use cudaMemcpyPeerAsync instead of cudaMemcpyAsync.");
ABSL_FLAG(bool, use_cudaMemcpyDefault, true,
          "Use cudaMemcpyDefault and no other cudaMemcpyKind.");
ABSL_FLAG(std::string, flow_model, "thread-per-flow",
          "Choices:  thread-per-flow, event-poll, thread-per-gpu.");
ABSL_FLAG(int32_t, wait_ns, 0,
          "How many ns to wait before re-launching cudaMemcpy.");
ABSL_FLAG(bool, group_by_dest, false,
          "In thread-per-gpu group by destination gpu.");
ABSL_FLAG(bool, sync_flows, false, "Synchronize each flow to maximize power.");
ABSL_FLAG(std::vector<std::string>, gpus, {"0"},
          "Comma-separated list of GPUs to execute on.");
ABSL_FLAG(bool, gemm, false, "Perform GEMM operations.");

ABSL_FLAG(
    std::string, fp_precision, "",
    "Sets the input, output, and compute precision to the selected "
    "value. Overrides the 'input_precision', 'output_precision', and "
    "'compute_precision' fields. To used mixed precision, use specific "
    "combinations of those flags instead of 'fp_precision'. "
    "Accepted inputs for 'fp_precision' are 'half', 'single', and 'double'.");
ABSL_FLAG(
    std::string, input_precision, "single",
    "int8, half, single, or double precision. Only a subset of combinations of "
    "'input_precision', 'output_precision', and 'compute_precision' types are "
    "allowed, dependent on the CUDA version and GPU architecture. See the "
    "documentation for a full explanation of supported computation types.");
ABSL_FLAG(std::string, output_precision, "single",
          "int32, half, single, or double precision.");
ABSL_FLAG(std::string, compute_precision, "single",
          "int32, half, single, or double precision.");
ABSL_FLAG(std::string, algorithm, "", "cublasGemmEx compute algorithm.");

// The matrices for AxB=C have the following dimensions:
// A is MxK, B is KxN, and C is MxN. The default value for N is 128.
// If either K or M are not specified, they are set to N.
ABSL_FLAG(int32_t, iterations, 10,
          "Number of iterations of the workload, -1 for infinite.");
ABSL_FLAG(size_t, N, 128,
          "Matrices of size NxN. Specify K or M for more options.");
ABSL_FLAG(size_t, M, 0,
          "Matrices of size MxK multiplied by KxN. Requires K, N.");
ABSL_FLAG(size_t, K, 0,
          "Matrices of size NxK multiplied by KxN. Requires N.");
ABSL_FLAG(double, high_time, 10.,
          "Approx. amount of time in seconds to do a burst of compute.");
ABSL_FLAG(double, low_time, .1,
          "Approx. amount of time in seconds to sleep between compute.");
ABSL_FLAG(double, trigger_period, 11.,
          "Approx. amount of time in seconds for each trigger phase.");
ABSL_FLAG(
    bool, nv_gaussian, false,
    "Fill input matrices with values generated by the following function: "
    "f(r1,r2) = sqrt(-10*log(r1))*cos(2*pi*r2), where r1, r2 are uniformly "
    "selected from [0,1). (suggested by nVidia)");

namespace pgmg = platforms_gpus::memcpy_gemm;
namespace pggt = platforms_gpus::gemm_test;

// Parses a single flow string. The simplest flow would be g0-g1, indicating
// a flow from GPU0 to GPU1.
static void ParseFlow(std::string arg, int *replicas,
                      pgmg::DeviceSpec *from_dev, int *from_index,
                      pgmg::DeviceSpec *to_dev, int *to_index,
                      int *counter_index) {
  std::string stripped = arg;
  absl::StripAsciiWhitespace(&stripped);
  std::string optional_replicas;
  std::string from_c, optional_from_index;
  std::string to_c, optional_to_index;
  std::string optional_counter_index;
  CHECK(RE2::FullMatch(
      stripped, "(\\d+:)?([cg])(\\d+)(/\\d+)?-([cg])(\\d+)(/\\d+)?(-a\\d+)?",
      &optional_replicas, &from_c, &from_dev->second, &optional_from_index,
      &to_c, &to_dev->second, &optional_to_index, &optional_counter_index))
      << "on flow '" << arg << "'";
  int64_t tmp = 1;
  if (optional_replicas.length()) {
    CHECK(absl::SimpleAtoi(
        optional_replicas.substr(0, optional_replicas.length() - 1), &tmp));
  }
  *replicas = tmp;
  from_dev->first = from_c[0] == 'c' ? pgmg::CPU : pgmg::GPU;
  to_dev->first = to_c[0] == 'c' ? pgmg::CPU : pgmg::GPU;
  tmp = -1;
  if (optional_from_index.length()) {
    CHECK(absl::SimpleAtoi(optional_from_index.substr(1), &tmp));
  }
  *from_index = tmp;
  tmp = -1;
  if (optional_to_index.length()) {
    CHECK(absl::SimpleAtoi(optional_to_index.substr(1), &tmp));
  }
  *to_index = tmp;
  tmp = 0;
  if (optional_counter_index.length()) {
    CHECK(absl::SimpleAtoi(optional_counter_index.substr(2), &tmp));
  }
  *counter_index = tmp;
}

class LogEachDeviceOnce {
 public:
  void Log(pgmg::DeviceSpec dev) {
    if (logged_.count(dev)) {
      return;
    }
    if (dev.first == pgmg::GPU) {
      cudaDeviceProp dev_prop;
      CUDA_CHECK(cudaGetDeviceProperties(&dev_prop, dev.second));
      LOG(INFO) << absl::StreamFormat(
          "GPU %d at %x:%02x:%02x model %s", dev.second, dev_prop.pciDomainID,
          dev_prop.pciBusID, dev_prop.pciDeviceID, dev_prop.name);
    }
    logged_[dev] = true;
  }

 protected:
  std::map<pgmg::DeviceSpec, bool> logged_;
};

// Peer registration of memory on to_dev to be accessible on from_dev
class PeerEachDeviceOnce {
 public:
  void Peer(pgmg::DeviceSpec from_dev, pgmg::DeviceSpec to_dev) {
    if (!absl::GetFlag(FLAGS_use_cudaDeviceEnablePeerAccess)) return;
    if (from_dev.first != pgmg::GPU) return;
    if (to_dev.first != pgmg::GPU) return;
    std::pair<int, int> peers(from_dev.second, to_dev.second);
    if (enabled_peers_.count(peers)) return;

    LOG(INFO) << "Enabling GPU " << from_dev.second << " to access "
              << to_dev.second;
    CUDA_CHECK(cudaSetDevice(from_dev.second));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(to_dev.second, 0));
    enabled_peers_[peers] = true;
  }

 protected:
  std::map<std::pair<int, int>, bool> enabled_peers_;
};
// Creates the compute thread on each of the requested GPUs.
std::vector<std::unique_ptr<pgmg::CopyThread>> CreateComputeThreads(
    pggt::HostContext *host_ctx, pgmg::PulseBarrier *pulse_barrier,
    const std::vector<std::string> &gpu_list) {

  std::vector<std::unique_ptr<pgmg::CopyThread>> threads;
  for (auto const &gpu : gpu_list) {
    int64_t gpu_num;
    CHECK(absl::SimpleAtoi(gpu, &gpu_num)) << "error parsing '" << gpu << "'";
    auto thread = CreateGemmThread(host_ctx, pulse_barrier, gpu_num);
    if (!thread) {
      LOG(ERROR) << "Failed to create GEMM Thread for GPU " << gpu;
      continue;
    }
    threads.push_back(std::move(thread));
  }
  return threads;
}

static volatile sig_atomic_t signal_received_ = false;
static void SignalHandler(int) { signal_received_ = true; }

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);

  setvbuf(stdout, nullptr, _IOLBF, 0);

  const std::string flow_model = absl::GetFlag(FLAGS_flow_model);
  CHECK(flow_model == "thread-per-flow" || flow_model == "event-poll" ||
        flow_model == "thread-per-gpu");

  const int32_t batch_size = absl::GetFlag(FLAGS_batch_size);
  CHECK_GE(batch_size, 1);

  const int buffer_size_KiB = absl::GetFlag(FLAGS_buffer_size_KiB);
  CHECK_GE(buffer_size_KiB, 4);
  const size_t buffer_size = static_cast<size_t>(buffer_size_KiB) * 1024;

  pgmg::BufferPool pool(buffer_size);
  LogEachDeviceOnce dev_logger;
  PeerEachDeviceOnce peering_pal;

  std::vector<std::unique_ptr<std::atomic<int>>> counters;
  std::vector<std::string> flow_strings =
      absl::StrSplit(absl::GetFlag(FLAGS_flows), ' ', absl::SkipEmpty());
  std::vector<pgmg::Flow *> flows;

  for (const auto &f : flow_strings) {
    pgmg::DeviceSpec from_dev, to_dev;
    int from_index, to_index, counter_index;
    int replicas;
    ParseFlow(f, &replicas, &from_dev, &from_index, &to_dev, &to_index,
              &counter_index);
    dev_logger.Log(from_dev);
    dev_logger.Log(to_dev);
    peering_pal.Peer(from_dev, to_dev);
    for (int i = 0; i < replicas; ++i) {
      char *from_buf = pool.GetBuffer(from_dev, from_index);
      char *to_buf = pool.GetBuffer(to_dev, to_index);
      LOG(INFO) << absl::StrFormat(
          "flow_%u from [%s,%d] = %p to [%s,%d] = %p "
          "counting in %d",
          flows.size(), DeviceSpecToString(from_dev), from_index, from_buf,
          DeviceSpecToString(to_dev), to_index, to_buf, counter_index);
      while (counters.size() <= counter_index) {
        counters.emplace_back(absl::make_unique<std::atomic<int>>(0));
      }
      flows.emplace_back(new pgmg::Flow(from_dev, from_buf, to_dev, to_buf,
                                        buffer_size,
                                        counters[counter_index].get()));
    }
  }
  std::signal(SIGTERM, SignalHandler);
  std::signal(SIGINT, SignalHandler);

  std::vector<std::unique_ptr<pgmg::CopyThread>> threads;
  std::vector<pgmg::BaseComputeStream *> compute_thread_copies;
  absl::BitGen rng;
  pgmg::PulseBarrier pulse_barrier(
      absl::Seconds(absl::GetFlag(FLAGS_high_time)),
      absl::Seconds(absl::GetFlag(FLAGS_low_time)),
      absl::GetFlag(FLAGS_sync_flows));
  pggt::ContextOption ctx_opt;
  std::unique_ptr<pggt::HostContext> host_ctx;
  std::vector<std::string> gpu_list = absl::GetFlag(FLAGS_gpus);
  if (absl::GetFlag(FLAGS_gemm)) {
    ctx_opt.rng = &rng;
    std::string fp_precision = absl::GetFlag(FLAGS_fp_precision);
    if (!fp_precision.empty()) {
      ctx_opt.data_type_in = fp_precision;
      ctx_opt.data_type_out = fp_precision;
      ctx_opt.compute_type = fp_precision;
    } else {
      ctx_opt.data_type_in = absl::GetFlag(FLAGS_input_precision);
      ctx_opt.data_type_out = absl::GetFlag(FLAGS_output_precision);
      ctx_opt.compute_type = absl::GetFlag(FLAGS_compute_precision);
    }
    LOG(INFO) << absl::Substitute(
        "Executing GEMM with input precision=$0, output precision=$1, compute "
        "precision=$2",
        ctx_opt.data_type_in, ctx_opt.data_type_out, ctx_opt.compute_type);
    ctx_opt.gaussian = absl::GetFlag(FLAGS_nv_gaussian);
    std::string algorithm = absl::GetFlag(FLAGS_algorithm);
    if (!algorithm.empty()) {
      ctx_opt.algorithm = algorithm;
    }

    size_t m = absl::GetFlag(FLAGS_M);
    size_t n = absl::GetFlag(FLAGS_N);
    size_t k = absl::GetFlag(FLAGS_K);
    ctx_opt.dim_size_n = n;
    ctx_opt.dim_size_m = m != 0 ? m : n;  // default to N
    ctx_opt.dim_size_k = k != 0 ? k : n;  // default to NN
    LOG(INFO) << absl::Substitute("(M,N,K) = ($0,$1,$2)", ctx_opt.dim_size_m,
                                  ctx_opt.dim_size_n, ctx_opt.dim_size_k);
    host_ctx = pggt::HostContext::Create(&ctx_opt);
    if (host_ctx == nullptr) {
      LOG(ERROR) << "GEMM Internal Error: Failed to create a host context.";
      return -1;
    }
    std::vector<std::unique_ptr<pgmg::CopyThread>> compute_threads =
        CreateComputeThreads(host_ctx.get(), &pulse_barrier, gpu_list);
    for (const auto &i : compute_threads) {
      compute_thread_copies.push_back(
          static_cast<pgmg::BaseComputeStream *>(i.get()));
    }
    std::move(compute_threads.begin(), compute_threads.end(),
              std::back_inserter(threads));
  }
  const bool use_cudaMemcpyPeerAsync =
      absl::GetFlag(FLAGS_use_cudaMemcpyPeerAsync);
  const bool use_cudaMemcpyDefault = absl::GetFlag(FLAGS_use_cudaMemcpyDefault);

  if (flow_model == "thread-per-flow") {
    for (int i = 0; i < flows.size(); ++i) {
      threads.emplace_back(new pgmg::CopySingleFlow(
          flows[i], &pulse_barrier, absl::GetFlag(FLAGS_wait_ns), batch_size,
          use_cudaMemcpyPeerAsync, use_cudaMemcpyDefault));
    }
  } else if (flow_model == "event-poll") {
    threads.emplace_back(new pgmg::EventPollThread(
        flows, use_cudaMemcpyPeerAsync, use_cudaMemcpyDefault));
  } else if (flow_model == "thread-per-gpu") {
    absl::flat_hash_set<pgmg::DeviceSpec> gpu_flows;
    for (auto f : flows) {
      pgmg::DeviceSpec d =
          absl::GetFlag(FLAGS_group_by_dest) ? f->to_dev_ : f->from_dev_;
      if (bool new_device = gpu_flows.insert(d).second; new_device) {
        threads.emplace_back(new pgmg::PerGpuThread(
            "per_gpu_thread_" + DeviceSpecToString(d),
            std::vector<pgmg::Flow *>{f}, absl::GetFlag(FLAGS_group_by_dest),
            use_cudaMemcpyPeerAsync, use_cudaMemcpyDefault));
      }
    }
  }

  absl::Time start_time = absl::Now();
  for (auto &t : threads) {
    LOG(INFO) << "Starting thread";
    t->Start();
  }

  size_t nr_counters = counters.size();
  std::vector<intptr_t> last_sample(nr_counters, 0);
  absl::Time last_time = absl::Now();
  absl::Time test_end = start_time + absl::GetFlag(FLAGS_duration);
  const int32_t sample_time = absl::GetFlag(FLAGS_sample_time);
  while (!signal_received_ && last_time < test_end) {
    sleep(sample_time);
    std::vector<intptr_t> sample(nr_counters);
    for (int i = 0; i < nr_counters; ++i) {
      sample[i] = *counters[i];
    }
    absl::Time now = absl::Now();
    std::cout << now;
    for (int i = 0; i < nr_counters; ++i) {
      std::cout << absl::StrFormat(
          "\t%.3f", (sample[i] - last_sample[i]) * buffer_size * 1e-9 /
                        absl::ToDoubleSeconds(now - last_time));
    }
    std::cout << "\n";
    last_sample = sample;
    last_time = now;
  }

  for (auto &t : threads) {
    t->StopCopying();
  }
  for (auto &t : threads) {
    t->Join();
  }
  absl::Time end_time = absl::Now();

  if (absl::GetFlag(FLAGS_gemm)) {
    int loops = 0;
    double accumulated_time = 0;
    for (const auto &copy_thread : compute_thread_copies) {
      loops += copy_thread->Loops();
      accumulated_time += copy_thread->AccumulatedTime();
    }
    // Adapted from multi_gemm.cc; this should really be MK(2N+3).
    constexpr float kTeraScalingFactor = 1.0E12;
    size_t dim_m = ctx_opt.dim_size_m;
    size_t dim_n = ctx_opt.dim_size_n;
    size_t dim_k = ctx_opt.dim_size_k;
    double ops_per_loop = 2.0 * dim_m * dim_n * dim_k;

    double total_ops = loops * ops_per_loop;
    double ops_per_gpu = total_ops / gpu_list.size();
    double total_time = absl::ToDoubleSeconds(end_time - start_time);
    LOG(INFO) << "Average flops per GPU = "
              << (ops_per_gpu / total_time) / kTeraScalingFactor << " TFLOPS.";
    LOG(INFO) << "Average flops per GPU during high pulse = "
              << (total_ops / accumulated_time) / kTeraScalingFactor
              << " TFLOPS.";
  }

  return 0;
}
