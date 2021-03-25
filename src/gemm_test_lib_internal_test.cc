#include "src/gemm_test_lib_internal.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/gemm_test_lib.h"
#include "src/multi_gemm_lib.h"

namespace platforms_gpus::gemm_test::internal {
namespace {

#if CUDA_VERSION >= 10010  // CUDA 10.1 or greater
TEST(SelectGemmInterfaceTest, Int8TensorTuringSquareMatrix) {
  const ContextOption options{.data_type_in = "int8",
                              .data_type_out = "int32",
                              .compute_type = "int32",
                              .dim_size_m = 2048,
                              .dim_size_n = 2048,
                              .dim_size_k = 2048};
  std::unique_ptr<GpuComputationInterface> result =
      SelectGemmInterface(options, ComputeCapability{.major = 7, .minor = 5});
  EXPECT_NE(nullptr, dynamic_cast<CudaCublasLtInterface*>(result.get()));
}

TEST(SelectGemmInterfaceTest, Int8TensorTuringNoSquareMatrix) {
  const ContextOption options{.data_type_in = "int8",
                              .data_type_out = "int32",
                              .compute_type = "int32",
                              .dim_size_m = 4096,
                              .dim_size_n = 2048,
                              .dim_size_k = 1024};
  std::unique_ptr<GpuComputationInterface> result =
      SelectGemmInterface(options, ComputeCapability{.major = 7, .minor = 5});
  EXPECT_NE(nullptr, dynamic_cast<CudaCublasLtInterface*>(result.get()));
}

#if CUDA_VERSION >= 11000  // CUDA 11 or greater
TEST(SelectGemmInterfaceTest, Int8TensorAmpereSquareMatrix) {
  const ContextOption options{.data_type_in = "int8",
                              .data_type_out = "int32",
                              .compute_type = "int32",
                              .dim_size_m = 2048,
                              .dim_size_n = 2048,
                              .dim_size_k = 2048};
  std::unique_ptr<GpuComputationInterface> result =
      SelectGemmInterface(options, ComputeCapability{.major = 7, .minor = 5});
  EXPECT_NE(nullptr, dynamic_cast<CudaCublasLtInterface*>(result.get()));
}

TEST(SelectGemmInterfaceTest, Int8TensorAmpereNoSquareMatrix) {
  const ContextOption options{.data_type_in = "int8",
                              .data_type_out = "int32",
                              .compute_type = "int32",
                              .dim_size_m = 4096,
                              .dim_size_n = 2048,
                              .dim_size_k = 1024};
  std::unique_ptr<GpuComputationInterface> result =
      SelectGemmInterface(options, ComputeCapability{.major = 7, .minor = 5});
  EXPECT_NE(nullptr, dynamic_cast<CudaCublasInterface*>(result.get()));
}

TEST(SelectGemmInterfaceTest, Bf16Bf16Matrices) {
  const ContextOption options{.data_type_in = "bf16",
                              .data_type_out = "bf16",
                              .compute_type = "single",
                              .dim_size_m = 4096,
                              .dim_size_n = 2048,
                              .dim_size_k = 1024};
  std::unique_ptr<GpuComputationInterface> result =
      SelectGemmInterface(options, ComputeCapability{.major = 8, .minor = 0});
  EXPECT_NE(nullptr, dynamic_cast<CudaCublasInterface*>(result.get()));
}

TEST(SelectGemmInterfaceTest, Bf16SingleMatrices) {
  const ContextOption options{.data_type_in = "bf16",
                              .data_type_out = "single",
                              .compute_type = "single",
                              .dim_size_m = 4096,
                              .dim_size_n = 2048,
                              .dim_size_k = 1024};
  std::unique_ptr<GpuComputationInterface> result =
      SelectGemmInterface(options, ComputeCapability{.major = 8, .minor = 0});
  EXPECT_NE(nullptr, dynamic_cast<CudaCublasInterface*>(result.get()));
}

#endif  // CUDA_VERSION >= 11000
#endif  // CUDA_VERSION >= 10010

TEST(SelectGemmInterfaceTest, Int8NonTensor) {
  const ContextOption options{.data_type_in = "int8",
                              .data_type_out = "int32",
                              .compute_type = "int32"};
  auto result =
      SelectGemmInterface(options, ComputeCapability{.major = 7, .minor = 0});
  EXPECT_NE(nullptr, dynamic_cast<CudaCublasInterface*>(result.get()));
}

TEST(SelectGemmInterfaceTest, ModernInterface) {
  const ContextOption options{.data_type_in = "half",
                              .data_type_out = "single",
                              .compute_type = "single"};
  auto result =
      SelectGemmInterface(options, ComputeCapability{.major = 7, .minor = 0});
  EXPECT_NE(nullptr, dynamic_cast<CudaCublasInterface*>(result.get()));
}

TEST(SelectGemmInterfaceTest, LegacySinglePrecision) {
  const ContextOption options{.data_type_in = "single",
                              .data_type_out = "single",
                              .compute_type = "single"};
  auto result =
      SelectGemmInterface(options, ComputeCapability{.major = 3, .minor = 0});
  EXPECT_NE(nullptr,
            dynamic_cast<LegacyCudaCublasInterface<float>*>(result.get()));
}

TEST(SelectGemmInterfaceTest, LegacyDoublePrecision) {
  const ContextOption options{.data_type_in = "double",
                              .data_type_out = "double",
                              .compute_type = "double"};
  auto result =
      SelectGemmInterface(options, ComputeCapability{.major = 3, .minor = 0});
  EXPECT_NE(nullptr,
            dynamic_cast<LegacyCudaCublasInterface<double>*>(result.get()));
}

TEST(SelectGemmInterfaceTest, FailuresDueToInadequateComputeCap) {
  const ContextOption options{
      .data_type_in = "half", .data_type_out = "half", .compute_type = "half"};
  EXPECT_EQ(nullptr, SelectGemmInterface(
                         options, ComputeCapability{.major = 3, .minor = 0}));
}

TEST(SelectGemmInterfaceTest, FailuresDueToUnknownTypes) {
  const ContextOption options{.data_type_in = "quarter",
                              .data_type_out = "triple",
                              .compute_type = "bf16"};
  EXPECT_EQ(nullptr, SelectGemmInterface(
                         options, ComputeCapability{.major = 3, .minor = 0}));
}

TEST(GpuDataHandlerTest, DestructsSafelyWithoutAllocation) {
  GpuDataHandler<float, float, float> data_handler;
}

struct HalfInHalfOut {
  using Input = half_float::half;
  using Output = half_float::half;
  using Compute = half_float::half;
};

struct HalfInFloatOut {
  using Input = half_float::half;
  using Output = float;
  using Compute = float;
};

struct FloatInFloatOut {
  using Input = float;
  using Output = float;
  using Compute = float;
};

struct DoubleInDoubleOut {
  using Input = double;
  using Output = double;
  using Compute = double;
};

struct IntInIntOut {
  using Input = int8_t;
  using Output = int32_t;
  using Compute = int32_t;
};

struct IntInFloatOut {
  using Input = int8_t;
  using Output = float;
  using Compute = float;
};

#if CUDA_VERSION >= BF16_CUDA_VERSION
struct Bf16InBf16Out {
  using Input = nv_bfloat16;
  using Output = nv_bfloat16;
  using Compute = nv_bfloat16;
};
#endif  // CUDA_VERSION >= BF16_CUDA_VERSION

template <typename T>
class GpuDataHandlerAllocationTest : public ::testing::Test {
 public:
  GpuDataHandlerAllocationTest() { CUDA_CHECK(cudaStreamCreate(&stream_)); }
  ~GpuDataHandlerAllocationTest() override {
    CUDA_CHECK(cudaStreamDestroy(stream_));
  }

 protected:
  cudaStream_t stream_;
};

TYPED_TEST_SUITE_P(GpuDataHandlerAllocationTest);

TYPED_TEST_P(GpuDataHandlerAllocationTest, SetupAndCleanup) {
  using InputPrecision = typename TypeParam::Input;
  using OutputPrecision = typename TypeParam::Output;
  using ComputePrecision = typename TypeParam::Compute;
  cudaSetDevice(0);

  const int array_size = 2048;

  RandomMatrix<InputPrecision> input_a(array_size, array_size);
  RandomMatrix<InputPrecision> input_b(array_size, array_size);
  absl::BitGen bitgen;
  input_a.Initialize(&bitgen, 1, false);
  input_b.Initialize(&bitgen, 1, false);

  GpuDataHandler<InputPrecision, OutputPrecision, ComputePrecision>
      data_handler;
  data_handler.SetGpuId(0);
  data_handler.Initialize(&input_a, &input_b, this->stream_);
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream_));
}

REGISTER_TYPED_TEST_SUITE_P(GpuDataHandlerAllocationTest, SetupAndCleanup);

#if CUDA_VERSION >= BF16_CUDA_VERSION
using MyTypes = ::testing::Types<HalfInHalfOut, HalfInFloatOut, FloatInFloatOut,
                                 DoubleInDoubleOut, IntInIntOut, IntInFloatOut,
                                 Bf16InBf16Out>;
#else
using MyTypes = ::testing::Types<HalfInHalfOut, HalfInFloatOut, FloatInFloatOut,
                                 DoubleInDoubleOut, IntInIntOut, IntInFloatOut>;
#endif  // CUDA_VERSION >= BF16_CUDA_VERSION

INSTANTIATE_TYPED_TEST_SUITE_P(MixedPrecisionGpuDataHandler,
                               GpuDataHandlerAllocationTest, MyTypes);

// Gets string representation for types, for context options population
// in typed tests.
template <typename T>
std::string StringRep();
template <>
std::string StringRep<float>() {
  return "single";
}
template <>
std::string StringRep<double>() {
  return "double";
}
template <>
std::string StringRep<half_float::half>() {
  return "half";
}
template <>
std::string StringRep<int8_t>() {
  return "int8";
}
template <>
std::string StringRep<int32_t>() {
  return "int32";
}

#if CUDA_VERSION >= BF16_CUDA_VERSION
template <>
std::string StringRep<nv_bfloat16>() {
  return "bf16";
}
#endif

// Initializes and copies over host data through the handler. Note that host
// data is not retained beyond this function, but that's fine since we operate
// only on the GPU data.
template <typename InputPrecision, typename OutputPrecision,
          typename ComputePrecision>
void InitializeGPUDataForGEMM(const ContextOption& options,
                              const cudaStream_t& stream,
                              GpuDataHandler<InputPrecision, OutputPrecision,
                                             ComputePrecision>* data_handler) {
  const int array_size = options.dim_size_m;
  RandomMatrix<InputPrecision> input_a(array_size, array_size);
  RandomMatrix<InputPrecision> input_b(array_size, array_size);
  absl::BitGen bitgen;
  input_a.Initialize(&bitgen, 1, false);
  input_b.Initialize(&bitgen, 1, false);

  data_handler->SetGpuId(0);
  data_handler->Initialize(&input_a, &input_b, stream);
  // Wait for data transfer to complete.
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename T>
class LegacyCublasTest : public ::testing::Test {
 public:
  LegacyCublasTest() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
    options_.dim_size_m = 2048;
    options_.dim_size_n = 2048;
    options_.dim_size_k = 2048;
    options_.transa = false;
    options_.transb = false;
    options_.algorithm = "gemm_algo_0";
  }

  ~LegacyCublasTest() override {
    CUDA_CHECK(cudaStreamDestroy(stream_));
  };

 protected:
  ContextOption options_;
  cudaStream_t stream_;
};

TYPED_TEST_SUITE_P(LegacyCublasTest);

TYPED_TEST_P(LegacyCublasTest, LegacyCublas) {
  using InputPrecision = typename TypeParam::Input;
  using OutputPrecision = typename TypeParam::Output;
  using ComputePrecision = typename TypeParam::Compute;

  this->options_.data_type_in = StringRep<InputPrecision>();
  this->options_.data_type_out = StringRep<OutputPrecision>();
  this->options_.compute_type = StringRep<OutputPrecision>();

  GpuDataHandler<InputPrecision, OutputPrecision, ComputePrecision>
      data_handler;
  // Set up some random data on the GPU for us to compute on, we dont' care
  // about the actual contents.
  InitializeGPUDataForGEMM<InputPrecision, OutputPrecision>(
      this->options_, this->stream_, &data_handler);

  GEMMData data{
      .alpha = data_handler.Alpha(),
      .beta = data_handler.Beta(),
      .matA = data_handler.InputA(),
      .matB = data_handler.InputB(),
      .matC = data_handler.Output(),
      .scalePtrType = kPtrType::kDevicePtr,
  };
  Algo algo{
      .cublas_algo_ = CUBLAS_GEMM_DFALT,
  };
  LegacyCudaCublasInterface<InputPrecision> cublas;
  cublas.Initialize(this->stream_);
  cublas.BindGemmMatrices(this->options_, data);
  CUBLAS_CHECK(cublas.MatrixMultiComputation(algo));
  CUDA_CHECK(cudaStreamSynchronize(this->stream_));
}

REGISTER_TYPED_TEST_SUITE_P(LegacyCublasTest, LegacyCublas);

using LegacyTypes = ::testing::Types<FloatInFloatOut, DoubleInDoubleOut>;
INSTANTIATE_TYPED_TEST_SUITE_P(LegacyCublasPrecisionsTest, LegacyCublasTest,
                               LegacyTypes);

// This class has the same logic as the Legacy cuBLAS test, but inheritance
// doesn't play well with typed tests.
template <typename T>
class ModernCublasTest : public ::testing::Test {
 public:
  ModernCublasTest() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
    options_.dim_size_m = 2048;
    options_.dim_size_n = 2048;
    options_.dim_size_k = 2048;
    options_.transa = false;
    options_.transb = false;
    options_.algorithm = "gemm_tensor_algo_default";
  }

  ~ModernCublasTest() override {
    CUDA_CHECK(cudaStreamDestroy(stream_));
  };

 protected:
  ContextOption options_;
  cudaStream_t stream_;
};

TYPED_TEST_SUITE_P(ModernCublasTest);

TYPED_TEST_P(ModernCublasTest, ModernCublas) {
  using InputPrecision = typename TypeParam::Input;
  using OutputPrecision = typename TypeParam::Output;
  using ComputePrecision = typename TypeParam::Compute;

  this->options_.data_type_in = StringRep<InputPrecision>();
  this->options_.data_type_out = StringRep<OutputPrecision>();
  this->options_.compute_type = StringRep<ComputePrecision>();

  GpuDataHandler<InputPrecision, OutputPrecision, ComputePrecision>
      data_handler;
  // Set up some random data on the GPU for us to compute on, we dont' care
  // about the actual contents.
  InitializeGPUDataForGEMM<InputPrecision, OutputPrecision>(
      this->options_, this->stream_, &data_handler);

  GEMMData data{
      .alpha = data_handler.Alpha(),
      .beta = data_handler.Beta(),
      .matA = data_handler.InputA(),
      .matB = data_handler.InputB(),
      .matC = data_handler.Output(),
      .scalePtrType = kPtrType::kDevicePtr,
  };
  Algo algo{
      .cublas_algo_ = CUBLAS_GEMM_DFALT,
  };
  CudaCublasInterface cublas;
  cublas.Initialize(this->stream_);
  cublas.BindGemmMatrices(this->options_, data);
  CUBLAS_CHECK(cublas.MatrixMultiComputation(algo));
  CUDA_CHECK(cudaStreamSynchronize(this->stream_));
}

REGISTER_TYPED_TEST_SUITE_P(ModernCublasTest, ModernCublas);

// Note that we don't add bf16 here, forge does not have hardware capable of
// testing device-side bf16 compute.
using ModernTypes =
    ::testing::Types<HalfInHalfOut, HalfInFloatOut, FloatInFloatOut,
                     DoubleInDoubleOut, IntInIntOut, IntInFloatOut>;

INSTANTIATE_TYPED_TEST_SUITE_P(ModernCublasPrecisionsTest, ModernCublasTest,
                               ModernTypes);
}  // namespace
}  // namespace platforms_gpus::gemm_test::internal
